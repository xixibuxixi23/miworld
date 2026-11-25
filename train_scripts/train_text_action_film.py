"""Stage-1 M3 closed-loop behavior cloning on exported M2 recurrent inputs."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel

from miworld.data.inserted_policy_dataset import canonical_text
from miworld.models.m3 import M3Policy, M3PolicyArgs
from miworld.models.renderer_codec import WanRendererCodec
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from train_scripts.train_policy import load_renderer, lookup_text, move_conditions


def load_samples(paths):
    samples = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "m3_closed_loop_bc_v1":
            raise ValueError(f"unsupported supervision file: {path}")
        if payload.get("history_frames") != 8 or payload.get("horizons") != 16:
            raise ValueError(f"incompatible supervision contract: {path}")
        samples.extend(payload["samples"])
    if not samples:
        raise ValueError("no closed-loop supervision samples")
    return samples


def collate(samples, device):
    conditions = {key: torch.stack([row["conditions"][key] for row in samples]).to(device)
                  for key in samples[0]["conditions"]}
    rgb = torch.stack([row["rgb"] for row in samples]).to(device, non_blocking=True)
    if rgb.dtype == torch.uint8:
        rgb = rgb.float().div_(255.)
    target = torch.stack([row["target_actions"] for row in samples]).to(device)
    family = torch.tensor([FAMILY_TO_ID[row["family"]] for row in samples], device=device)
    profile = torch.tensor([PROFILE_TO_ID[row["profile"]] for row in samples], device=device)
    return rgb, conditions, target, family, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision", type=Path, action="append", required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--renderer-checkpoint", type=Path, required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--text-catalog", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--place-positive-weight", type=float, default=1.)
    parser.add_argument("--place-loss-weight", type=float, default=0.)
    parser.add_argument("--execution-place-sample-fraction", type=float, default=0.,
                        help="Fraction forced to contain place in the executed first 8 steps")
    parser.add_argument("--future-place-sample-fraction", type=float, default=0.,
                        help="Fraction forced to contain place only in future steps 9-16")
    parser.add_argument("--future-place-loss-weight", type=float, default=0.,
                        help="Auxiliary place loss weight for non-executed steps 9-16")
    parser.add_argument("--text-action-film", action="store_true",
                        help="Warm-start the text-routed action FiLM adapter")
    parser.add_argument("--adapter-only", action="store_true",
                        help="Freeze the pretrained policy and train only text action FiLM")
    parser.add_argument("--task-balanced-sampling", action="store_true",
                        help="Sample line/repair/wall/floor/cuboid/fill uniformly")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    checkpoint = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source_cfg = M3PolicyArgs(**checkpoint["config"])
    cfg = (replace(source_cfg, builder_text_action_film=True)
           if args.text_action_film else source_cfg)
    if cfg.history_frames != 8 or cfg.horizons != 16 or cfg.builder_action_history != 64:
        raise ValueError("expected history8 / horizon16 / action-history64 policy")
    renderer, renderer_config = load_renderer(args.renderer_checkpoint, return_config=True)
    raw = M3Policy(renderer, cfg).to(device).train()
    incompatible = raw.families.load_state_dict(
        checkpoint["families"], strict=not args.text_action_film)
    if args.text_action_film:
        expected_missing = {name for name, _ in raw.families.named_parameters()
                            if ".task_film." in name}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"invalid text FiLM migration: missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}")
    source_step = int(checkpoint["step"])
    del checkpoint
    codec = WanRendererCodec(args.wan_vae).to(device).eval().requires_grad_(False)
    text_cache = load_file(str(args.text_cache), device="cpu")
    catalog = json.loads(args.text_catalog.read_text())
    text_ids = {canonical_text(row["text"]): int(row["text_id"]) for row in catalog["texts"]}

    # Limit adaptation to the builder action decoder, its last two visual
    # reasoning blocks, and the compact action/place-history encoder.
    trainable = []
    head = []
    for name, parameter in raw.families.named_parameters():
        enabled = (name.startswith("language_builder.head.task_film.")
                   if args.adapter_only else
                   (name.startswith("language_builder.head.") or
                    name.startswith("language_builder.blocks.10.") or
                    name.startswith("language_builder.blocks.11.") or
                    name.startswith("language_builder.interaction_history.")))
        parameter.requires_grad_(enabled)
        if enabled:
            (head if name.startswith("language_builder.head.") else trainable).append(parameter)
    if not head or (not args.adapter_only and not trainable):
        raise ValueError("closed-loop trainable parameter selection is empty")
    groups = [{"params": head, "lr": args.head_lr,
               "role": "text_action_film" if args.adapter_only else "action_head"}]
    if trainable:
        groups.insert(0, {"params": trainable, "lr": args.lr, "role": "policy_tail"})
    optimizer = torch.optim.AdamW(groups)
    model = (DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False,
                                     find_unused_parameters=True) if world > 1 else raw)
    samples = load_samples(args.supervision)
    fractions = (args.execution_place_sample_fraction, args.future_place_sample_fraction)
    if any(not 0. <= value <= 1. for value in fractions) or sum(fractions) > 1.:
        raise ValueError("place sample fractions must be in [0,1] and sum to at most one")
    execution_place_samples = [row for row in samples
                               if bool((row["target_actions"][:8, 9] > 0).any())]
    future_place_samples = [row for row in samples
                            if (not bool((row["target_actions"][:8, 9] > 0).any()) and
                                bool((row["target_actions"][8:, 9] > 0).any()))]
    if args.execution_place_sample_fraction and not execution_place_samples:
        raise ValueError("no execution-place windows available")
    if args.future_place_sample_fraction and not future_place_samples:
        raise ValueError("no future-only place windows available")
    def task_bucket(row):
        task = row["task_id"]
        for prefix, bucket in (
            ("place_line_", "line"), ("repair_wall_", "repair"),
            ("build_wall_", "wall"), ("build_floor_", "floor"),
            ("build_cuboid_", "cuboid"), ("fill_hole_", "fill"),
        ):
            if task.startswith(prefix):
                return bucket
        raise ValueError(f"unsupported task for balanced sampling: {task}")
    task_pools = {}
    if args.task_balanced_sampling:
        for row in samples:
            task_pools.setdefault(task_bucket(row), []).append(row)
        expected = {"line", "repair", "wall", "floor", "cuboid", "fill"}
        if set(task_pools) != expected:
            raise ValueError(f"task-balanced sampling needs {sorted(expected)}, got {sorted(task_pools)}")
    task_names = sorted(task_pools)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        config = vars(args).copy()
        config.update(source_checkpoint=str(args.policy_checkpoint), source_step=source_step,
                      renderer_checkpoint=str(args.renderer_checkpoint), sample_count=len(samples),
                      execution_place_sample_count=len(execution_place_samples),
                      future_place_sample_count=len(future_place_samples),
                      task_sample_counts={key: len(value) for key, value in task_pools.items()},
                      world_size=world, effective_batch=world*args.batch_size*args.grad_accum,
                      trainable_scope=("text action FiLM only" if args.adapter_only else
                                       "builder head + blocks10/11 + interaction history"))
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
        config["supervision"] = [str(x) for x in args.supervision]
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    if world > 1:
        dist.barrier()

    order = list(range(len(samples)))
    # Ranks must not replay identical balanced samples.  The natural sampler
    # already separates ranks by cursor; use an explicit rank-specific RNG too.
    rng = random.Random(args.seed + 100003 * rank)
    rng.shuffle(order)
    cursor = rank * args.batch_size
    valid = torch.ones(args.batch_size, 16, dtype=torch.bool, device=device)
    policy_indices = torch.arange(1, 9, device=device)[None].expand(args.batch_size, -1)
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, args.steps + 1):
        total = torch.zeros((), device=device)
        total_base = torch.zeros((), device=device)
        total_place = torch.zeros((), device=device)
        total_future_place = torch.zeros((), device=device)
        for micro in range(args.grad_accum):
            chosen = []
            for _ in range(args.batch_size):
                if args.task_balanced_sampling:
                    pool = task_pools[task_names[rng.randrange(len(task_names))]]
                    chosen.append(pool[rng.randrange(len(pool))])
                elif sum(fractions):
                    draw = rng.random()
                    if draw < args.execution_place_sample_fraction:
                        pool = execution_place_samples
                    elif draw < sum(fractions):
                        pool = future_place_samples
                    else:
                        pool = samples
                    chosen.append(pool[rng.randrange(len(pool))])
                else:
                    if cursor >= len(order):
                        rng.shuffle(order)
                        cursor = rank * args.batch_size
                    chosen.append(samples[order[cursor % len(order)]])
                    cursor += world * args.batch_size
            rgb, conditions, target, family, profile = collate(chosen, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = codec.encode(rgb)
            shared_ids = torch.tensor([text_ids[canonical_text(x["shared_text"])] for x in chosen])
            current_ids = torch.tensor([text_ids[canonical_text(x["current_text"])] for x in chosen])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                shared, shared_mask = lookup_text(text_cache, shared_ids, device, latent.dtype)
                current, current_mask = lookup_text(text_cache, current_ids, device, latent.dtype)
                _, logits = model(latent, conditions, family, profile,
                                  shared_text=shared, shared_text_mask=shared_mask,
                                  current_text=current, current_text_mask=current_mask,
                                  policy_indices=policy_indices)
                base_loss, _ = raw.loss(logits, target, valid, mouse_loss_weight=.1)
                if logits["keys"].shape[1] != 16:
                    raise ValueError("closed-loop place loss expects a 16-step action head")
                place = F.binary_cross_entropy_with_logits(
                    logits["keys"][:, :8, 8], target[:, :8, 9], reduction="none",
                    pos_weight=torch.as_tensor(args.place_positive_weight, device=device))
                place_loss = (place * valid[:, :8]).sum() / valid[:, :8].sum().clamp_min(1)
                future_place = F.binary_cross_entropy_with_logits(
                    logits["keys"][:, 8:, 8], target[:, 8:, 9], reduction="none",
                    pos_weight=torch.as_tensor(args.place_positive_weight, device=device))
                future_place_loss = ((future_place * valid[:, 8:]).sum() /
                                     valid[:, 8:].sum().clamp_min(1))
                loss = (base_loss + args.place_loss_weight * place_loss +
                        args.future_place_loss_weight * future_place_loss)
            (loss / args.grad_accum).backward()
            total += loss.detach() / args.grad_accum
            total_base += base_loss.detach() / args.grad_accum
            total_place += place_loss.detach() / args.grad_accum
            total_future_place += future_place_loss.detach() / args.grad_accum
        torch.nn.utils.clip_grad_norm_([*trainable, *head], 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reduced = torch.stack((total, total_base, total_place, total_future_place))
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
        step = source_step + update
        if rank == 0 and (update == 1 or update % 10 == 0):
            print(json.dumps({"update": update, "step": step,
                              "loss": float(reduced[0]), "base_loss": float(reduced[1]),
                              "execution_place_loss": float(reduced[2]),
                              "future_place_loss": float(reduced[3])}), flush=True)
        if update % args.save_every == 0 or update == args.steps:
            if rank == 0:
                payload = dict(families=raw.families.state_dict(), optimizer=optimizer.state_dict(),
                               step=step, config=asdict(cfg), m3_checkpoint=str(args.renderer_checkpoint),
                               training={"mode": "closed_loop_bc_v1", "updates": update,
                                         "source_checkpoint": str(args.policy_checkpoint)})
                tmp = args.output_dir / "latest.tmp"
                torch.save(payload, tmp)
                tmp.replace(args.output_dir / "latest.pt")
                shutil.copy2(args.output_dir / "latest.pt", args.output_dir / f"step_{step:07d}.pt")
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
