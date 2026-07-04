"""Tests for LoRA injection utilities (model-agnostic).

Mirrors the five invariants from indus_CT/NaSn/lora_pmfdit.py's smoke test,
but on a tiny generic nn.Module so it needs no pmfDiT:
  1. scale=0 reproduces the base output exactly
  2. scale=1 with zero-init up still reproduces the base output
  3. kicking the up weights makes the output diverge
  4. save/load roundtrip is exact
  5. count_trainable reports only the LoRA params
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.solvers.diffusion.lora import (
    inject_lora_linear, set_lora_scale, save_lora_state, load_lora_state,
    count_trainable, LoraInjectedLinear)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 8)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def test_lora_invariants():
    torch.manual_seed(0)
    m = _Net()
    x = torch.randn(4, 16)
    y_base = m(x).clone()

    groups = inject_lora_linear(m, r=4)
    n_inj = sum(isinstance(mm, LoraInjectedLinear) for mm in m.modules())
    assert n_inj == 2, n_inj

    set_lora_scale(m, 0.0)
    assert (m(x) - y_base).abs().max().item() == 0.0, "scale=0 must equal base"

    set_lora_scale(m, 1.0)
    assert (m(x) - y_base).abs().max().item() == 0.0, "up=0 must equal base"

    for n, p in m.named_parameters():
        if 'lora_up.weight' in n:
            nn.init.normal_(p, std=0.3)
    y_kick = m(x).clone()
    assert (y_kick - y_base).abs().max().item() > 1e-3, "kick must diverge"

    state = save_lora_state(m)
    for n, p in m.named_parameters():
        if 'lora_up' in n or 'lora_down' in n:
            p.data.zero_()
    load_lora_state(m, state, strict=True)
    assert (m(x) - y_kick).abs().max().item() == 0.0, "roundtrip must be exact"

    cats = count_trainable(m)
    print(f"[lora] injected={n_inj} trainable={cats}")
    assert cats['other'] == 0 and cats['lora_linear'] > 0


def test_targeted_injection():
    torch.manual_seed(0)
    m = _Net()
    inject_lora_linear(m, targets=['a'], r=4)
    kinds = {name: mm.__class__.__name__
             for name, mm in m.named_children()}
    print(f"[targeted] {kinds}")
    assert kinds['a'] == 'LoraInjectedLinear'
    assert kinds['b'] == 'Linear'  # 'b' not targeted


if __name__ == "__main__":
    test_lora_invariants()
    test_targeted_injection()
    print("\nAll LoRA tests passed.")
