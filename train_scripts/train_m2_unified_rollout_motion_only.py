"""Fine-tune only the M2 displacement head on 128-frame state rollouts.

Each eight-frame block retains observed world/RGB/HP boundary conditions while
pose, velocity and camera state come from the preceding prediction.  Carried
state is detached between blocks.  All shared and event parameters are frozen;
only the private pose head is optimized from XYZ trajectory and endpoint loss.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from miworld.data.unified_rollout_stream import UnifiedRolloutStream, collate_unified_rollouts
from miworld.models.local_world_writer import LocalWorldWriter, LocalWorldWriterArgs
from miworld.training.transition_trainer import transition_loss


COUNT_METRICS = {
    "event_labels", "predicted_events", "correct_events", "block_labels",
    "correct_edits", "attack_labels", "correct_attacks",
    "supervised_event_windows", "incomplete_event_windows",
    "overflow_event_windows",
}


def move(batch, device):
    return {g: {k: v.to(device, non_blocking=True) for k, v in batch[g].items()}
            for g in ("inputs", "targets")}


def masked_mean(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def rates(metrics):
    result = dict(metrics)
    result["event_precision"] = result["correct_events"] / max(1, result["predicted_events"])
    result["event_recall"] = result["correct_events"] / max(1, result["event_labels"])
    p, r = result["event_precision"], result["event_recall"]
    result["event_f1"] = 2 * p * r / max(1e-12, p + r)
    result["edit_recall"] = result["correct_edits"] / max(1, result["block_labels"])
    result["attack_recall"] = result["correct_attacks"] / max(1, result["attack_labels"])
    return result


def reduce_metrics(metrics, device, world_size):
    names = sorted(metrics)
    values = torch.tensor([float(metrics[k]) for k in names], device=device,
                          dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(values)
    result = {}
    for name, value in zip(names, values.tolist()):
        result[name] = value if name in COUNT_METRICS else value / world_size
    return rates(result)


def rollout_inputs(observed, carry):
    inputs = dict(observed)
    if carry is None:
        return inputs
    true_pose = inputs["initial_pose"]
    predicted_pose = carry["pose"]
    inputs["voxel_relative_xyz"] = (
        inputs["voxel_relative_xyz"]
        + true_pose[:, :, None, :3] - predicted_pose[:, :, None, :3]
    )
    inputs["initial_pose"] = predicted_pose
    inputs["initial_velocity"] = carry["velocity"]
    inputs["camera_relative"] = carry["camera_relative"]
    inputs["camera_direction"] = carry["camera_direction"]
    return inputs


def next_carry(output):
    return {
        "pose": output["pose"][:, -1].detach(),
        "velocity": output["velocity"][:, -1].detach(),
        "camera_relative": output["camera_relative"][:, -1].detach(),
        "camera_direction": output["camera_direction"][:, -1].detach(),
    }


def endpoint_terms(output, inputs, targets, stationary_threshold):
    valid = targets["state_valid"].bool() & inputs["active"][:, None].bool()
    end_valid = valid[:, -1]
    endpoint = masked_mean(F.smooth_l1_loss(
        output["pose"][:, -1, :, :3], targets["pose"][:, -1, :, :3],
        reduction="none"), end_valid)
    displacement = (targets["pose"][:, -1, :, :3]
                    - targets["pose"][:, 0, :, :3]).norm(dim=-1)
    stationary = end_valid & (displacement <= stationary_threshold)
    stationary_endpoint = masked_mean(F.smooth_l1_loss(
        output["pose"][:, -1, :, :3], targets["pose"][:, -1, :, :3],
        reduction="none"), stationary)
    error = (output["pose"][:, -1, :, :3]
             - targets["pose"][:, -1, :, :3]).norm(dim=-1)
    return endpoint, stationary_endpoint, float(masked_mean(error, end_valid).detach())


def accumulate(total, metrics):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


@torch.no_grad()
def evaluate(model, loader, device, max_batches, blocks, positive_weight,
             stationary_threshold):
    model.eval()
    totals = {}
    count = 0
    for sequence in loader:
        carry = None
        for block, batch in enumerate(sequence):
            values = move(batch, device)
            inputs = rollout_inputs(values["inputs"], carry)
            output = model(inputs)
            _, metrics = transition_loss(
                model, output, inputs=inputs, targets=values["targets"],
                objective="unified", event_count_positive_weight=positive_weight)
            _, _, position_error = endpoint_terms(
                output, inputs, values["targets"], stationary_threshold)
            accumulate(totals, metrics)
            if block in {0, 3, 7, blocks - 1}:
                totals[f"position_error_{(block + 1) * 8}f"] = (
                    totals.get(f"position_error_{(block + 1) * 8}f", 0.0)
                    + position_error)
            carry = next_carry(output)
        count += 1
        if count >= max_batches:
            break
    if not count:
        raise ValueError("validation stream has no complete rollout sequences")
    result = {}
    for key, value in totals.items():
        if key in COUNT_METRICS:
            result[key] = value
        elif key.startswith("position_error_"):
            result[key] = value / count
        else:
            result[key] = value / (count * blocks)
    model.train()
    return rates(result)


def distributed_environment(device_name):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return rank, world_size, local_rank, torch.device("cuda", local_rank)
    return rank, world_size, local_rank, torch.device(device_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "index", "vocabulary", "items", "cache-root",
                "source-checkpoint", "output-dir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--rollout-blocks", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences per GPU")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--event-count-positive-weight", type=float, default=4.)
    parser.add_argument("--endpoint-weight", type=float, default=1.)
    parser.add_argument("--stationary-weight", type=float, default=.5)
    parser.add_argument("--stationary-threshold", type=float, default=.05)
    parser.add_argument("--edit-oversample", type=int, default=1)
    parser.add_argument("--attack-oversample", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.rollout_blocks, args.batch_size, args.save_every,
           args.eval_batches, args.log_every) < 1:
        parser.error("steps, rollout blocks, batch size and intervals must be positive")
    if args.weight_decay != 0:
        parser.error("motion-only training requires zero weight decay to preserve angle rows")

    rank, world_size, local_rank, device = distributed_environment(args.device)
    out = Path(args.output_dir)
    nonempty = int(out.exists() and any(out.iterdir()))
    if world_size > 1:
        flag = torch.tensor(nonempty, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        nonempty = int(flag.item())
    if nonempty:
        raise ValueError("output directory must be new or empty")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    source_config = checkpoint["config"]
    cfg_data = dict(source_config["model"])
    from miworld.kinematics import KinematicsConfig
    if isinstance(cfg_data.get("kinematics"), dict):
        cfg_data["kinematics"] = KinematicsConfig(**cfg_data["kinematics"])
    cfg = LocalWorldWriterArgs(**cfg_data)
    model = LocalWorldWriter(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.camera_decoder.pose_head.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    if trainable_names != ["camera_decoder.pose_head.weight",
                           "camera_decoder.pose_head.bias"]:
        raise ValueError(f"unexpected trainable parameters: {trainable_names}")
    model.eval()
    model.camera_decoder.pose_head.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)

    item_data = json.loads(Path(args.items).read_text())
    items = item_data.get("items", item_data)
    common = dict(
        index=args.index, vocabulary=args.vocabulary, cache_root=args.cache_root,
        dataset_root=args.dataset_root, items=items,
        rollout_blocks=args.rollout_blocks, seed=args.seed)
    train = UnifiedRolloutStream(
        split="train", rank=rank, world_size=world_size,
        edit_oversample=args.edit_oversample,
        attack_oversample=args.attack_oversample, **common)
    val = (UnifiedRolloutStream(
        split="val_id", rank=0, world_size=1, fixed_order=True, **common)
        if rank == 0 else None)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_unified_rollouts, pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers else None)
    val_loader = (DataLoader(
        val, batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_unified_rollouts, pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers else None) if rank == 0 else None)

    train_model = (DDP(model, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False)
                   if world_size > 1 else model)
    config = dict(source_config)
    config["training"] = vars(args)
    config["distributed"] = dict(
        world_size=world_size, sequences_per_gpu=args.batch_size,
        frames_per_sequence=8 * args.rollout_blocks,
        global_frames_per_step=world_size * args.batch_size * 8 * args.rollout_blocks)
    config["objective"] = "motion_only_128f_displacement_rollout"
    config["source_checkpoint"] = str(Path(args.source_checkpoint).resolve())
    config["rollout"] = dict(
        blocks=args.rollout_blocks, frames=8 * args.rollout_blocks,
        carry=["pose", "velocity", "camera_relative", "camera_direction"],
        observed_boundaries=["voxels", "previous_rgb", "hp", "held_item", "event_context"],
        detach_between_blocks=True,
        event_supervision_every_block=False,
        optimization_terms=["xyz_trajectory", "xyz_endpoint", "stationary_xyz_endpoint"],
        trainable_parameters=trainable_names,
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
        frozen_shared_trunk=True,
        frozen_event_decoder=True,
        frozen_camera_and_held_heads=True)
    if rank == 0:
        (out / "config.json").write_text(json.dumps(config, indent=2))

    def log(step, split, metrics):
        if rank:
            return
        row = dict(step=step, split=split, **metrics)
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    iterator = iter(train_loader)
    best_128f = float("inf")
    for step in range(1, args.steps + 1):
        try:
            sequence = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            sequence = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        carry = None
        totals = {}
        horizon_errors = {}
        for block, batch in enumerate(sequence):
            values = move(batch, device)
            inputs = rollout_inputs(values["inputs"], carry)
            synchronize = block == args.rollout_blocks - 1 or world_size == 1
            context = nullcontext() if synchronize else train_model.no_sync()
            with context:
                output = train_model(inputs)
                state_valid = (values["targets"]["state_valid"].bool()
                               & inputs["active"][:, None].bool())
                trajectory = masked_mean(F.smooth_l1_loss(
                    output["pose"][..., :3], values["targets"]["pose"][..., :3],
                    reduction="none"), state_valid)
                endpoint, stationary, position_error = endpoint_terms(
                    output, inputs, values["targets"], args.stationary_threshold)
                progress = block / max(1, args.rollout_blocks - 1)
                loss = (trajectory + args.endpoint_weight * (.5 + progress) * endpoint
                        + args.stationary_weight * stationary) / args.rollout_blocks
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite rollout loss")
                loss.backward()
            with torch.no_grad():
                _, metrics = transition_loss(
                    model, output, inputs=inputs, targets=values["targets"],
                    objective="unified",
                    event_count_positive_weight=args.event_count_positive_weight)
            accumulate(totals, metrics)
            totals["optimized_trajectory"] = (
                totals.get("optimized_trajectory", 0.) + float(trajectory.detach()))
            totals["endpoint"] = totals.get("endpoint", 0.) + float(endpoint.detach())
            totals["stationary_endpoint"] = totals.get("stationary_endpoint", 0.) + float(stationary.detach())
            if block in {0, 3, 7, args.rollout_blocks - 1}:
                horizon_errors[f"position_error_{(block + 1) * 8}f"] = position_error
            carry = next_carry(output)
            del output, loss, trajectory
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite rollout gradient")
        optimizer.step()
        train_metrics = {k: (v if k in COUNT_METRICS else v / args.rollout_blocks)
                         for k, v in totals.items()}
        train_metrics.update(horizon_errors, gradient_norm=float(norm))
        if device.type == "cuda":
            train_metrics["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 1024 ** 3)
            train_metrics["peak_reserved_gib"] = (
                torch.cuda.max_memory_reserved(device) / 1024 ** 3)
        train_metrics = reduce_metrics(train_metrics, device, world_size)
        if step == 1 or step % args.log_every == 0:
            log(step, "train_rollout", train_metrics)

        if step % args.save_every == 0 or step == args.steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               step=step, config=config)
                temporary = out / "checkpoint.tmp"
                torch.save(payload, temporary)
                temporary.replace(out / "checkpoint.pt")
                validation = evaluate(
                    model, val_loader, device, args.eval_batches,
                    args.rollout_blocks, args.event_count_positive_weight,
                    args.stationary_threshold)
                log(step, "val_rollout", validation)
                score = validation[f"position_error_{8 * args.rollout_blocks}f"]
                if score < best_128f:
                    best_128f = score
                    temporary = out / "best-val-128f.tmp"
                    torch.save(payload, temporary)
                    temporary.replace(out / "best-val-128f.pt")
                    (out / "best-val-128f.json").write_text(json.dumps(
                        dict(step=step, position_error_128f=score,
                             event_f1=validation["event_f1"],
                             edit_recall=validation["edit_recall"],
                             attack_recall=validation["attack_recall"]), indent=2))
            if world_size > 1:
                dist.barrier()
            model.eval()
            model.camera_decoder.pose_head.train()

    if rank == 0:
        (out / "COMPLETED.json").write_text(json.dumps(
            dict(steps=args.steps, frames=8 * args.rollout_blocks,
                 world_size=world_size)))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
