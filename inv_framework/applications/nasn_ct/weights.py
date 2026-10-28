"""Safe release-weight export and loading."""
from __future__ import annotations

import json
from pathlib import Path

import torch


DEFAULT_MODEL_CONFIG = {
    "model": "pmfDiT_B_128", "img_size": 2048, "img_channels": 1,
    "num_classes": 1, "P_mean": -0.4, "P_std": 1.0, "cfg_max": 0.0,
    "data_proportion": 0.5, "cfg_beta": 1.0, "norm_p": 1.0,
    "norm_eps": 0.01, "tr_uniform": False,
}


def export_ema(checkpoint, output, metadata_output=None):
    """Extract only EMA tensors; optimizer state and arbitrary pickle stay private."""
    from safetensors.torch import save_file

    checkpoint, output = Path(checkpoint), Path(output)
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = value.get("ema")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no EMA state dictionary")
    tensors = {
        name: tensor.detach().contiguous() for name, tensor in state.items()
        if not name.startswith(("lpips_model.", "net.v_heads.", "net.v_final_layer."))
        and not tensor.is_complex()
    }
    args = value.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)
    config = {**DEFAULT_MODEL_CONFIG, **{
        key: args[key] for key in DEFAULT_MODEL_CONFIG if key in args
    }}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={
        "format": "pt", "source_epoch": str(value.get("epoch", "unknown")),
        "architecture": config["model"], "license": "MIT",
    })
    metadata = {
        "format": "safetensors", "state": "ema", "epoch": value.get("epoch"),
        "step": value.get("step"), "model": config,
        "excluded": [
            "lpips_model.* (training loss network)",
            "net.v_heads.* and net.v_final_layer.* (training-only instantaneous-velocity head)",
            "net.rope_freqs (deterministically regenerated model buffer)",
        ],
    }
    destination = Path(metadata_output) if metadata_output else output.with_suffix(".json")
    destination.write_text(json.dumps(metadata, indent=2))
    return output, destination


def load_pmf(weights, config_path=None, device="cuda"):
    from safetensors.torch import load_file
    from ...models.pmf import pixelMeanFlow

    weights = Path(weights)
    metadata_path = Path(config_path) if config_path else weights.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    config = {**DEFAULT_MODEL_CONFIG, **metadata.get("model", {})}
    model = pixelMeanFlow(
        config["model"], img_size=config["img_size"],
        img_channels=config["img_channels"], num_classes=config["num_classes"],
        eval=True, P_mean=config["P_mean"], P_std=config["P_std"],
        cfg_max=config["cfg_max"], data_proportion=config["data_proportion"],
        cfg_beta=config["cfg_beta"], class_dropout_prob=0,
        norm_p=config["norm_p"], norm_eps=config["norm_eps"],
        tr_uniform=config["tr_uniform"], msssim_lambda=0, lpips_lambda=0,
    ).to(device)
    missing, unexpected = model.load_state_dict(
        load_file(str(weights), device=str(device)), strict=False
    )
    if unexpected or set(missing) != {"net.rope_freqs"}:
        raise ValueError(
            f"release weight mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model
