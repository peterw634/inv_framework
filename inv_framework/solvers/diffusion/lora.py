"""LoRA injection utilities for in-the-loop OOD adaptation.

Ported from ``indus_CT/NaSn/lora_pmfdit.py``. Vanilla LoRA (zero-init up,
normal-init down, alpha/r = scale) with a per-module ``scale`` knob so a
sampler can toggle between the base model (``scale=0``) and the adapted model
(``scale=1``). Used by `dds_lora.LoRAAdaptiveDDSSolver` to adapt a frozen
prior to out-of-distribution measurements during reverse diffusion, in the
style of Steerable Conditional Diffusion (Barbano et al., 2308.14409) and
DDIP3D (Chung & Ye, 2407.10641).

The `LoraInjectedLinear` / `LoraInjectedConv2d` modules and the scale/save/load
helpers are model-agnostic (pure torch). `inject_lora_linear` LoRA-fies any
`nn.Module` by attribute path; `adapt_pmfdit` is the pmfDiT-specific recipe
(kept for the real pMF prior) and lives here so callers import LoRA from one
place.
"""

from typing import Iterable, List, Optional, Sequence, Set, Type

import torch
import torch.nn as nn


__all__ = [
    "LoraInjectedLinear",
    "LoraInjectedConv2d",
    "inject_lora_linear",
    "adapt_pmfdit",
    "set_lora_scale",
    "save_lora_state",
    "load_lora_state",
    "count_trainable",
]


# ---------------------------------------------------------------------------
# LoRA modules (vendored from cloneofsimo/lora; used by SCD and DDIP3D)
# ---------------------------------------------------------------------------


class LoraInjectedLinear(nn.Module):
    """nn.Linear with low-rank residual: y = W x + scale * up(down(x))."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 r: int = 4, dropout_p: float = 0.0, scale: float = 1.0):
        super().__init__()
        if r > min(in_features, out_features):
            raise ValueError(f"r={r} > min({in_features},{out_features})")
        self.r = r
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.lora_up = nn.Linear(r, out_features, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1.0 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.scale != 0.0:
            out = out + self.dropout(self.lora_up(self.lora_down(x))) * self.scale
        return out


class LoraInjectedConv2d(nn.Module):
    """nn.Conv2d with low-rank residual via two conv layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1,
                 bias: bool = True, r: int = 4, dropout_p: float = 0.0,
                 scale: float = 1.0):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(f"r={r} > min({in_channels},{out_channels})")
        self.r = r
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.lora_down = nn.Conv2d(in_channels, r, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=groups, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.lora_up = nn.Conv2d(r, out_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1.0 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.scale != 0.0:
            out = out + self.dropout(self.lora_up(self.lora_down(x))) * self.scale
        return out


# ---------------------------------------------------------------------------
# Generic injection
# ---------------------------------------------------------------------------


def _replace_linear(parent: nn.Module, name: str, child: nn.Linear,
                    r: int, dropout_p: float, scale: float) -> LoraInjectedLinear:
    new = LoraInjectedLinear(child.in_features, child.out_features,
                             bias=child.bias is not None,
                             r=r, dropout_p=dropout_p, scale=scale)
    new.linear.weight = child.weight
    if child.bias is not None:
        new.linear.bias = child.bias
    new.to(device=child.weight.device, dtype=child.weight.dtype)
    parent._modules[name] = new
    return new


def inject_lora_linear(module: nn.Module,
                       targets: Optional[Sequence[str]] = None,
                       r: int = 8,
                       dropout_p: float = 0.0,
                       scale: float = 1.0,
                       verbose: bool = False) -> List[Iterable[nn.Parameter]]:
    """Freeze ``module`` and LoRA-inject its nn.Linear layers.

    Parameters
    ----------
    targets : sequence of str, optional
        Substrings; only Linear layers whose qualified name contains one of
        them are injected. ``None`` injects every nn.Linear.

    Returns
    -------
    trainable_groups : list of parameter iterables (the LoRA up/down params).
    """
    module.requires_grad_(False)
    to_replace = []
    for full_name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if targets is not None and not any(t in full_name for t in targets):
            continue
        to_replace.append(full_name)

    groups: List[Iterable[nn.Parameter]] = []
    n = 0
    for full_name in to_replace:
        *path, name = full_name.split('.')
        parent = module
        for seg in path:
            parent = parent.get_submodule(seg)
        child = parent._modules[name]
        if isinstance(child, (LoraInjectedLinear, LoraInjectedConv2d)):
            continue
        new = _replace_linear(parent, name, child, r, dropout_p, scale)
        groups.append(new.lora_up.parameters())
        groups.append(new.lora_down.parameters())
        n += 1
    if verbose:
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"[inject_lora_linear] injected {n} Linear; "
              f"trainable {train/1e3:.1f}K / {total/1e6:.2f}M")
    return groups


# ---------------------------------------------------------------------------
# Module discovery + pmfDiT-specific recipe
# ---------------------------------------------------------------------------


def _find_modules(root: nn.Module,
                  ancestor_class: Set[str],
                  search_class: List[Type[nn.Module]],
                  exclude_children_of: tuple = (LoraInjectedLinear,
                                                LoraInjectedConv2d)):
    """Yield (parent, attr_name, child) for every ``search_class`` instance
    whose closest named ancestor's class name is in ``ancestor_class``."""
    ancestors = [m for m in root.modules()
                 if m.__class__.__name__ in ancestor_class]
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if not any(isinstance(module, c) for c in search_class):
                continue
            *path, name = fullname.split('.')
            parent = ancestor
            for seg in path:
                parent = parent.get_submodule(seg)
            if isinstance(parent, exclude_children_of):
                continue
            yield parent, name, module


def adapt_pmfdit(net: nn.Module,
                 r: int = 8,
                 scale: float = 1.0,
                 dropout_p: float = 0.0,
                 include_patch_proj2: bool = True,
                 unfreeze_gates: bool = True,
                 block_lists: Optional[Set[str]] = None,
                 verbose: bool = True) -> List[Iterable[nn.Parameter]]:
    """Freeze a pmfDiT (``pmf.net``), inject LoRA, return trainable param groups.

    pmfDiT-specific: injects the 7 Linear layers of every ``TransformerBlock``
    under ``block_lists`` (q/k/v/out_proj + SwiGLU w1/w3/w2), LoRA-fies
    ``x_embedder.proj2`` (1x1 Conv2d), and unfreezes the per-block ``attn_scale``
    / ``mlp_scale`` gates. See ``indus_CT/NaSn/lora_pmfdit.py`` for the rationale.
    """
    if block_lists is None:
        block_lists = {'shared_blocks', 'u_heads'}

    net.requires_grad_(False)

    n_gate = 0
    if unfreeze_gates:
        for name, p in net.named_parameters():
            if name.endswith('attn_scale') or name.endswith('mlp_scale'):
                p.requires_grad_(True)
                n_gate += p.numel()

    selected = nn.ModuleList(
        [child for name, child in net.named_children() if name in block_lists])

    groups: List[Iterable[nn.Parameter]] = []
    n_lin = 0
    for parent, name, child in _find_modules(
            selected, ancestor_class={'TransformerBlock'},
            search_class=[nn.Linear]):
        new = _replace_linear(parent, name, child, r, dropout_p, scale)
        groups.append(new.lora_up.parameters())
        groups.append(new.lora_down.parameters())
        n_lin += 1

    n_conv = 0
    if include_patch_proj2:
        old = net.x_embedder.proj2
        assert isinstance(old, nn.Conv2d), f"expected Conv2d, got {type(old)}"
        new = LoraInjectedConv2d(
            old.in_channels, old.out_channels, old.kernel_size,
            stride=old.stride, padding=old.padding, dilation=old.dilation,
            groups=old.groups, bias=old.bias is not None,
            r=r, dropout_p=dropout_p, scale=scale)
        new.conv.weight = old.weight
        if old.bias is not None:
            new.conv.bias = old.bias
        new.to(device=old.weight.device, dtype=old.weight.dtype)
        net.x_embedder.proj2 = new
        groups.append(new.lora_up.parameters())
        groups.append(new.lora_down.parameters())
        n_conv += 1

    if verbose:
        total = sum(p.numel() for p in net.parameters())
        train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"[adapt_pmfdit] r={r} gates={unfreeze_gates} proj2={include_patch_proj2}")
        print(f"  injected {n_lin} Linear + {n_conv} Conv2d; gates {n_gate/1e3:.1f}K")
        print(f"  trainable {train/1e6:.3f}M / {total/1e6:.2f}M = {train/total*100:.3f}%")

    return groups


# ---------------------------------------------------------------------------
# Scale toggle + state I/O
# ---------------------------------------------------------------------------


def set_lora_scale(net: nn.Module, scale: float) -> None:
    """Toggle LoRA contribution (0.0 = base model, 1.0 = LoRA-on)."""
    for m in net.modules():
        if isinstance(m, (LoraInjectedLinear, LoraInjectedConv2d)):
            m.scale = scale


def save_lora_state(net: nn.Module) -> dict:
    """Extract LoRA up/down weights + pmfDiT gate params as a CPU dict."""
    state: dict = {}
    for name, m in net.named_modules():
        if isinstance(m, (LoraInjectedLinear, LoraInjectedConv2d)):
            state[f'{name}.lora_down.weight'] = m.lora_down.weight.detach().cpu().clone()
            state[f'{name}.lora_up.weight'] = m.lora_up.weight.detach().cpu().clone()
    for name, p in net.named_parameters():
        if name.endswith('attn_scale') or name.endswith('mlp_scale'):
            state[name] = p.detach().cpu().clone()
    return state


def load_lora_state(net: nn.Module, state: dict, strict: bool = True) -> dict:
    """Load LoRA + gates from a dict produced by `save_lora_state`."""
    loaded, missed = 0, []
    for name, m in net.named_modules():
        if isinstance(m, (LoraInjectedLinear, LoraInjectedConv2d)):
            params = dict(m.named_parameters())
            for sub in ('lora_down.weight', 'lora_up.weight'):
                key = f'{name}.{sub}'
                if key in state:
                    tgt = params[sub]
                    tgt.data.copy_(state[key].to(device=tgt.device, dtype=tgt.dtype))
                    loaded += 1
                else:
                    missed.append(key)
    for name, p in net.named_parameters():
        if name.endswith('attn_scale') or name.endswith('mlp_scale'):
            if name in state:
                p.data.copy_(state[name].to(device=p.device, dtype=p.dtype))
                loaded += 1
            else:
                missed.append(name)
    if strict and missed:
        raise KeyError(f"missing {len(missed)} keys; first: {missed[:3]}")
    return {'loaded': loaded, 'missed': missed}


def count_trainable(net: nn.Module) -> dict:
    """Breakdown of trainable params by category."""
    cats = {'lora_linear': 0, 'lora_conv': 0, 'gates': 0, 'other': 0}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_down' in n or 'lora_up' in n:
            if 'x_embedder' in n or '.conv.' in n:
                cats['lora_conv'] += p.numel()
            else:
                cats['lora_linear'] += p.numel()
        elif n.endswith('attn_scale') or n.endswith('mlp_scale'):
            cats['gates'] += p.numel()
        else:
            cats['other'] += p.numel()
    cats['total'] = sum(cats.values())
    return cats
