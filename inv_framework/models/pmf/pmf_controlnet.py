"""ControlNet for pMF (PixArt-alpha "ControlNet-Half" recipe).

The CN is a side network mirroring the first N (default 4) blocks of
base.shared_blocks. Each cn_block is a deepcopy of the corresponding base
block plus two zero-init linear projections:

    - before_proj: only on cn_block 0; projects the cond seq into cn space
    - after_proj:  on every cn_block; produces the c_skip residual added
                   into the next base shared_block input

Injection follows PixArt-alpha skip-first convention:

    seq = base.shared_blocks[0](seq, rope)        # un-controlled
    for i in 0..N-1:
        c, c_skip = cn_blocks[i](seq_or_None, c, rope)
        seq = base.shared_blocks[i+1](seq + c_skip, rope)
    for j in N+1..len(shared)-1:
        seq = base.shared_blocks[j](seq, rope)

At init (zero before_proj + zero after_proj), the wrapper reproduces base
exactly: c_skip == 0 -> base trunk runs unchanged.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from .pmfDiT import TransformerBlock


def _zero_linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.zeros_(lin.weight)
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


class PMFControlNetBlock(nn.Module):
    """One CN block: copy of a base TransformerBlock + zero-init projections."""

    def __init__(self, base_block: TransformerBlock, block_index: int, hidden_size: int):
        super().__init__()
        self.copied_block = deepcopy(base_block)
        for p in self.copied_block.parameters():
            p.requires_grad_(True)
        self.block_index = block_index
        self.hidden_size = hidden_size
        if block_index == 0:
            self.before_proj = _zero_linear(hidden_size, hidden_size)
        self.after_proj = _zero_linear(hidden_size, hidden_size)

    def forward(self, x, c, rope_freqs):
        if self.block_index == 0:
            assert x is not None, "cn_block[0] requires x (= seq_after_base_block_0)"
            c = self.before_proj(c)                      # zero-init -> 0 at start
            c = self.copied_block(x + c, rope_freqs)
        else:
            c = self.copied_block(c, rope_freqs)
        c_skip = self.after_proj(c)                      # zero-init -> 0 at start
        return c, c_skip


class PMFControlNet(nn.Module):
    """Side network of N cn_blocks mirroring base.shared_blocks[0..N-1].

    `forward(seq_after_block0, cn_seq, rope_freqs)` returns a list of N
    c_skip tensors that the wrapper injects into base.shared_blocks[1..N].
    """

    def __init__(self, base_pmfdit, copy_blocks_num: int = 4):
        super().__init__()
        self.copy_blocks_num = copy_blocks_num
        self.hidden_size = base_pmfdit.hidden_size
        max_n = len(base_pmfdit.shared_blocks) - 1
        assert 1 <= copy_blocks_num <= max_n, (
            f"copy_blocks_num must be in [1, {max_n}] (skip-first convention "
            f"with {len(base_pmfdit.shared_blocks)} shared_blocks); got {copy_blocks_num}"
        )
        self.cn_blocks = nn.ModuleList([
            PMFControlNetBlock(
                base_pmfdit.shared_blocks[i],
                block_index=i,
                hidden_size=base_pmfdit.hidden_size,
            )
            for i in range(copy_blocks_num)
        ])

    def forward(self, seq_after_block0, cn_seq, rope_freqs):
        """
        Args:
            seq_after_block0: (B, T, D) - output of base.shared_blocks[0]
            cn_seq:           (B, T, D) - base._build_sequence(cond, ...)
            rope_freqs:                 - base.rope_freqs
        Returns:
            controls: list of length copy_blocks_num, each (B, T, D)
        """
        controls = []
        c = cn_seq
        for i, cn_block in enumerate(self.cn_blocks):
            if i == 0:
                c, c_skip = cn_block(seq_after_block0, c, rope_freqs)
            else:
                c, c_skip = cn_block(None, c, rope_freqs)
            controls.append(c_skip)
        return controls
