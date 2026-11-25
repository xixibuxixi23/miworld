"""Finetune the original M1 flow DiT as one initialization/frontier model."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miworld.data.m1_flow_cache import M1FlowCache  # noqa: E402
from miworld.models.m1_flow import UnifiedM1Flow, UnifiedM1FlowArgs  # noqa: E402


MODE_NAMES = ("initialization_image", "frontier_no_image", "frontier_image")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4, help="per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=96)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--visible-loss-weight", type=float, default=4.0)
    parser.add_argument("--initialization-probability", type=float, default=0.5)
    parser.add_argument("--frontier-image-probability", type=float, default=1 / 3)
    parser.add_argument("--visible-frontier-probability", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if bool(args.initialize) == bool(args.resume):
        parser.error("exactly one of --initialize or --resume is required")
    if not 0 < args.initialization_probability < 1:
        parser.error("--initialization-probability must be in (0,1)")
    if not 0 <= args.frontier_image_probability <= 1:
        parser.error("--frontier-image-probability must be in [0,1]")
    return args


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return world, rank, local_rank


def atomic_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def sample_modes(batch: int, args, device):
    draw = torch.rand(batch, device=device)
    modes = torch.zeros(batch, dtype=torch.long, device=device)
    frontier = draw >= args.initialization_probability
    image_frontier = frontier & (
        torch.rand(batch, device=device) < args.frontier_image_probability
    )
    modes[frontier] = 1
    modes[image_frontier] = 2
    return modes


def random_slab_mask(batch: int, device):
    coordinate = torch.arange(48, device=device)
    axis = torch.randint(0, 2, (batch,), device=device)
    reverse = torch.rand(batch, device=device) < 0.5
    boundary = torch.randint(12, 37, (batch,), device=device)
    x = coordinate[None, :, None, None]
    y = coordinate[None, None, :, None]
    along = torch.where(axis[:, None, None, None] == 0, x, y)
    lower = along < boundary[:, None, None, None]
    return torch.where(reverse[:, None, None, None], ~lower, lower).expand(-1, -1, -1, 48)


def deterministic_slab_mask(indices: torch.Tensor, device):
    indices = indices.to(device)
    coordinate = torch.arange(48, device=device)
    axis = indices.remainder(2)
    reverse = indices.div(2, rounding_mode="floor").remainder(2).bool()
    boundary = 12 + indices.remainder(25)
    x = coordinate[None, :, None, None]
    y = coordinate[None, None, :, None]
    along = torch.where(axis[:, None, None, None] == 0, x, y)
    lower = along < boundary[:, None, None, None]
    return torch.where(reverse[:, None, None, None], ~lower, lower).expand(-1, -1, -1, 48)


def build_known_mask(batch, modes, args, deterministic=False):
    raw = batch["raw"]
    device = raw.device
    known = torch.zeros_like(raw, dtype=torch.bool)
    frontier = modes.ne(0)
    if not frontier.any():
        return known
    slab = (
        deterministic_slab_mask(batch["index"], device)
        if deterministic
        else random_slab_mask(len(raw), device)
    )
    known[frontier] = slab[frontier]

    # Some frontier examples use the true camera-visible support from the old
    # cache.  Slabs remain the majority because overlapping moving 48^3 windows
    # expose large contiguous faces rather than sparse individual voxels.
    if args.visible_frontier_probability > 0:
        choose_visible = frontier & (
            torch.rand(len(raw), device=device) < args.visible_frontier_probability
        )
        if choose_visible.any():
            visible = F.interpolate(
                batch["visible_latent_mask"][:, None].float(),
                size=(48, 48, 48),
                mode="nearest",
            )[:, 0].bool()
            fractions = visible.flatten(1).float().mean(1)
            usable = choose_visible & fractions.gt(0.05) & fractions.lt(0.95)
            known[usable] = visible[usable]
    return known


def prepare_condition(batch, modes, raw_lookup, args, deterministic=False):
    known = build_known_mask(batch, modes, args, deterministic=deterministic)
    raw = batch["raw"].long()
    if raw.min() < 0 or raw.max() >= len(raw_lookup):
        raise ValueError("raw node is outside context lookup")
    context_ids = raw_lookup[raw]
    if torch.any(context_ids < 0):
        raise ValueError("raw node is absent from legacy cache vocabulary")
    view_valid = batch["view_valid"].bool() & modes.ne(1)[:, None]
    unknown_fraction = F.avg_pool3d(
        (~known)[:, None].float(), kernel_size=4, stride=4
    )[:, 0]
    weights = unknown_fraction
    visible = batch["visible_latent_mask"].float()
    weights = weights * (1 + (args.visible_loss_weight - 1) * visible)
    return context_ids, known, view_valid, weights


def flow_loss(model, batch, modes, raw_lookup, args, deterministic=False):
    context_ids, known, view_valid, weights = prepare_condition(
        batch, modes, raw_lookup, args, deterministic=deterministic
    )
    target = batch["latent"].float()
    noise = torch.randn_like(target)
    if deterministic:
        # Evaluation has stable time; caller fixes the RNG for stable noise.
        time_value = torch.full((len(target),), 0.5, device=target.device)
    else:
        time_value = torch.randn(len(target), device=target.device).sigmoid()
    scaled_time = time_value[:, None, None, None, None]
    noisy = (1 - scaled_time) * target + (
        1e-5 + (1 - 1e-5) * scaled_time
    ) * noise
    velocity = (1 - 1e-5) * noise - target
    prediction = model(
        noisy,
        time_value,
        batch["image_rays"].float(),
        view_valid,
        context_ids,
        known,
    )
    error = (prediction.float() - velocity).square().mean(1)
    # Normalize within each scene first.  A full initialization tile otherwise
    # contributes roughly twice as many cells as a half-known frontier tile and
    # can hide a failing image-free branch in the aggregate objective.
    numerator = (error * weights).flatten(1).sum(1)
    denominator = weights.flatten(1).sum(1).clamp_min(1)
    per_sample = numerator / denominator
    loss = per_sample.mean()
    return loss, {
        "known_fraction": float(known.float().mean().detach()),
        "weighted_voxels": float(weights.sum().detach()),
        "mode_losses": {
            MODE_NAMES[mode]: (
                float(per_sample[modes.eq(mode)].mean().detach())
                if modes.eq(mode).any()
                else None
            )
            for mode in range(3)
        },
    }


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, raw_lookup, args, device, rank, world):
    model.eval()
    count = min(len(dataset), args.max_val_samples)
    indices = list(range(rank, count, world))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    totals = torch.zeros((3, 2), dtype=torch.float64, device=device)
    rng_state = torch.cuda.get_rng_state(device)
    for batch in loader:
        batch = move(batch, device)
        for mode in range(3):
            torch.cuda.manual_seed(args.seed + 10_000 * mode + int(batch["index"][0]))
            modes = torch.full((len(batch["latent"]),), mode, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = flow_loss(
                    model, batch, modes, raw_lookup, args, deterministic=True
                )
            totals[mode, 0] += loss.double() * len(modes)
            totals[mode, 1] += len(modes)
    torch.cuda.set_rng_state(rng_state, device)
    if world > 1:
        dist.all_reduce(totals)
    model.train()
    return {
        MODE_NAMES[mode]: float(totals[mode, 0] / totals[mode, 1].clamp_min(1))
        for mode in range(3)
    }


def save_checkpoint(path, model, optimizer, step, epoch, source, args):
    atomic_save(
        {
            "schema": "unified-m1-flow-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "source_checkpoint": str(source),
            "config": model.config,
            "train_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main():
    args = parse_args()
    world, rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(4)

    train_dataset = M1FlowCache(
        args.cache, "train", max_samples=args.max_train_samples
    )
    val_dataset = M1FlowCache(args.cache, "val_id")
    raw_lookup = train_dataset.raw_to_context.to(device)
    model_args = UnifiedM1FlowArgs(context_classes=len(train_dataset.context_nodes))
    model = model_args.build()

    start_step = 0
    start_epoch = 0
    source = args.resume or args.initialize
    state = load_checkpoint(source)
    if args.resume:
        model.load_state_dict(state["model"], strict=True)
        start_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
        load_report = {"resume": True, "tensors": len(state["model"])}
    else:
        load_report = model.load_legacy_state(state["model"])
        load_report["source_step"] = int(state.get("step", -1))
    del state
    model.to(device)

    adapter = list(model.adapter_parameters())
    adapter_ids = {id(parameter) for parameter in adapter}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in adapter_ids]
    initial_backbone_lr = 0.0 if start_step < args.adapter_warmup_steps else args.backbone_lr
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": initial_backbone_lr, "name": "backbone"},
            {"params": adapter, "lr": args.adapter_lr, "name": "adapter"},
        ],
        weight_decay=args.weight_decay,
    )
    if args.resume:
        resumed = load_checkpoint(args.resume)
        optimizer.load_state_dict(resumed["optimizer"])
        del resumed
        # Preserve optimizer moments while allowing an intentional LR change
        # at resume time. optimizer.load_state_dict restores the old group LRs.
        optimizer.param_groups[0]["lr"] = initial_backbone_lr
        optimizer.param_groups[1]["lr"] = args.adapter_lr

    wrapped = (
        DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        if world > 1
        else model
    )
    sampler = DistributedSampler(
        train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
    ) if world > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        manifest = {
            "schema": "unified-m1-flow-v1",
            "source": str(source),
            "load_report": load_report,
            "model": model.config,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "world_size": world,
            "effective_batch": args.batch_size * world * args.gradient_accumulation,
            "modes": {
                "initialization_image": args.initialization_probability,
                "frontier_no_image": (1 - args.initialization_probability)
                * (1 - args.frontier_image_probability),
                "frontier_image": (1 - args.initialization_probability)
                * args.frontier_image_probability,
            },
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("MANIFEST " + json.dumps(manifest), flush=True)

    step = start_step
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    last_eval = None
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for micro_index, batch in enumerate(loader):
            batch = move(batch, device)
            modes = sample_modes(len(batch["latent"]), args, device)
            update = (micro_index + 1) % args.gradient_accumulation == 0
            sync = contextlib.nullcontext()
            if world > 1 and not update:
                sync = wrapped.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, auxiliary = flow_loss(wrapped, batch, modes, raw_lookup, args)
                (loss / args.gradient_accumulation).backward()
            if not update:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == args.adapter_warmup_steps:
                optimizer.param_groups[0]["lr"] = args.backbone_lr

            if rank == 0 and (step == start_step + 1 or step % args.log_every == 0):
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "known_fraction": auxiliary["known_fraction"],
                    "backbone_lr": optimizer.param_groups[0]["lr"],
                    "adapter_lr": optimizer.param_groups[1]["lr"],
                    "mode_counts": {
                        MODE_NAMES[mode]: int(modes.eq(mode).sum()) for mode in range(3)
                    },
                    "mode_losses": auxiliary["mode_losses"],
                    "seconds": time.time() - started,
                }
                with (args.output / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print("TRAIN " + json.dumps(row), flush=True)

            should_evaluate = args.eval_every and step % args.eval_every == 0
            if should_evaluate:
                last_eval = evaluate(
                    model, val_dataset, raw_lookup, args, device, rank, world
                )
                if rank == 0:
                    row = {"step": step, **last_eval}
                    with (args.output / "eval.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print("EVAL " + json.dumps(row), flush=True)

            should_save = args.save_every and step % args.save_every == 0
            if should_save:
                if rank == 0:
                    save_checkpoint(
                        args.output / f"checkpoint_{step:08d}.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        source,
                        args,
                    )
                    save_checkpoint(
                        args.output / "checkpoint_latest.pt",
                        model,
                        optimizer,
                        step,
                        epoch,
                        source,
                        args,
                    )
                if world > 1:
                    dist.barrier()
            if step >= args.steps:
                break
        epoch += 1

    if rank == 0:
        save_checkpoint(
            args.output / "checkpoint_latest.pt",
            model,
            optimizer,
            step,
            epoch,
            source,
            args,
        )
        (args.output / "training_complete.json").write_text(
            json.dumps({"step": step, "epoch": epoch, "last_eval": last_eval}, indent=2)
            + "\n"
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
