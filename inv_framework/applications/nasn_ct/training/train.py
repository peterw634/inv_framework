"""Training entry point matching the released 2048 pMF configuration."""
from __future__ import annotations

import argparse
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ....models.pmf import pixelMeanFlow
from .dataset import CTSliceDataset


def _rank():
    return distributed.get_rank() if distributed.is_initialized() else 0


def _print(*values):
    if _rank() == 0:
        print(*values, flush=True)


def _adapt_pretrained(model, path):
    source = torch.load(path, map_location="cpu", weights_only=True)
    target = model.state_dict()
    loaded, skipped = 0, []
    for name, value in source.items():
        if name not in target:
            continue
        if value.shape == target[name].shape:
            target[name] = value; loaded += 1; continue
        if "x_embedder.proj1.weight" in name and value.shape[1] == 3 and target[name].shape[1] == 1:
            value = value.mean(dim=1, keepdim=True)
            if value.shape[-1] != target[name].shape[-1]:
                old, new = value.shape[-1], target[name].shape[-1]
                value = functional.interpolate(
                    value.reshape(-1, 1, old, old).float(), (new, new),
                    mode="bilinear", align_corners=False,
                ).reshape(target[name].shape) / ((new / old) ** 2)
            target[name] = value.to(target[name].dtype); loaded += 1
        else:
            skipped.append(name)
    model.load_state_dict(target)
    _print(f"pretrained tensors loaded={loaded}, skipped={len(skipped)}: {skipped}")


@torch.no_grad()
def _update_ema(ema, model, decay):
    values = dict(ema.named_parameters())
    for name, parameter in model.named_parameters():
        values[name].mul_(decay).add_(parameter, alpha=1 - decay)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", required=True)
    result.add_argument("--workdir", required=True)
    result.add_argument("--pretrained", required=True)
    result.add_argument("--resume")
    result.add_argument("--epochs", type=int, default=500)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--lr", type=float, default=4e-4)
    result.add_argument("--warmup-epochs", type=int, default=10)
    result.add_argument("--ema-decay", type=float, default=0.9999)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--checkpoint-every", type=int, default=50)
    result.add_argument("--augment", action="store_true")
    return result


def entrypoint(argv=None):
    args = parser().parse_args(argv)
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1:
        distributed.init_process_group("nccl")
    torch.manual_seed(args.seed + _rank())
    dataset = CTSliceDataset(args.data_root, "train", args.augment)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world > 1 else None
    if args.batch_size % world:
        raise ValueError("global batch size must be divisible by WORLD_SIZE")
    loader = DataLoader(
        dataset, batch_size=args.batch_size // world, sampler=sampler,
        shuffle=sampler is None, num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )
    model_args = dict(
        model_str="pmfDiT_B_128", img_size=2048, img_channels=1, num_classes=1,
        P_mean=-0.4, P_std=1.0, cfg_max=0.0, data_proportion=0.5,
        cfg_beta=1.0, class_dropout_prob=0.0, norm_p=1.0, norm_eps=0.01,
        msssim_lambda=0.0, lpips_lambda=0.4, lpips_net="vgg",
        lpips_crop_size=224, perceptual_max_t=0.8,
    )
    model = pixelMeanFlow(**model_args).cuda()
    _adapt_pretrained(model, args.pretrained)
    ema = deepcopy(model).requires_grad_(False)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = len(loader)
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))),
    )
    start_epoch = step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
    workdir = Path(args.workdir)
    if _rank() == 0:
        workdir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for images, labels in loader:
            loss, parts = wrapped(images.cuda(non_blocking=True), labels.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
            _update_ema(ema, model, args.ema_decay); step += 1
            if step % 10 == 0:
                _print(f"epoch={epoch} step={step} loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.3g}")
        if _rank() == 0 and (epoch + 1) % args.checkpoint_every == 0:
            torch.save({
                "epoch": epoch, "step": step, "model": model.state_dict(),
                "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(), "args": vars(args),
            }, workdir / f"ckpt_epoch_{epoch}.pt")
    if distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    entrypoint()
