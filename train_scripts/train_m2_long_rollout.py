"""Train the opt-in three-stream M2 with autoregressive pose/camera history.

Only existing revision-4 caches are read. Initial world geometry is fixed in
world coordinates and receives shared predicted edits; pose/camera and their
four-frame histories are carried from predictions. HP and held-item boundaries
remain recorded conditions. The source initializer is never changed.

The opt-in 8-to-24-rank resume preserves model and AdamW state in a new output
directory. Existing ranks restore their saved RNG; new ranks are reseeded at
the migration boundary. The effective global batch becomes three times larger.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from miworld.data.transition_dataset import VOXEL_COUNT, collate_transition
from miworld.models.local_world_writer import build_m2_from_config
from miworld.training.m2_v2_rollout import (
    detach_carry, next_carry, rollout_batch, supervised_velocity_loss, to_device,
)
from miworld.training.unified_transition import unified_transition_loss


COUNT_METRICS = {
    "event_labels", "predicted_events", "correct_events", "block_labels",
    "correct_edits", "attack_labels", "correct_attacks",
    "supervised_event_windows", "incomplete_event_windows", "overflow_event_windows",
    "predicted_edits", "scored_edit_frames", "original_valid_edit_targets",
    "covered_edit_targets", "out_of_initial_memory_edit_targets",
    "shared_world_writes", "same_frame_edit_conflicts",
}
NEW_PARAMETER_PREFIXES = (
    "state_residual_head.", "action_gate_logits", "action_time_embedding",
    "current_time_embedding", "state_role_embedding.", "action_item_valid_embedding.",
    "velocity_encoder.",
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relocate_catalog(catalog, old_root, new_root):
    """Rewrite only cache locations; sequence membership/signature stay frozen."""
    relocated = deepcopy(catalog)
    old_root, new_root = Path(old_root), Path(new_root)
    for collection in ("entries", "episodes"):
        for row in relocated[collection]:
            cache = Path(row["cache"])
            try:
                relative = cache.relative_to(old_root)
            except ValueError as error:
                raise ValueError(f"catalog cache escapes its recorded root: {cache}") from error
            row["cache"] = str(new_root / relative)
    return relocated


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rate_metrics(values):
    values = dict(values)
    precision = values.get("correct_events", 0.) / max(1., values.get("predicted_events", 0.))
    recall = values.get("correct_events", 0.) / max(1., values.get("event_labels", 0.))
    values.update(event_precision=precision, event_recall=recall,
                  event_f1=2 * precision * recall / max(1e-12, precision + recall),
                  edit_recall=values.get("correct_edits", 0.) / max(1., values.get("block_labels", 0.)),
                  attack_recall=values.get("correct_attacks", 0.) / max(1., values.get("attack_labels", 0.)))
    edit_precision = values.get("correct_edits", 0.) / max(1., values.get("predicted_edits", 0.))
    edit_recall = values["edit_recall"]
    values.update(edit_precision=edit_precision,
                  edit_f1=2 * edit_precision * edit_recall / max(1e-12, edit_precision + edit_recall),
                  edit_memory_coverage=values.get("covered_edit_targets", 0.) / max(
                      1., values.get("original_valid_edit_targets", 0.)))
    return values


class CachedRollouts:
    """Stable descriptors and a small lazy LRU; never materialize/write caches.

    Sampling depends only on (seed, optimizer step, rank), so restarting does
    not replay DataLoader workers or silently skip/repeat a saved optimizer step.
    """

    def __init__(self, index, cache_root, split, blocks, *, max_episodes=0,
                 edit_oversample=1, attack_oversample=1, timesteps=8, catalog=None):
        self.blocks, self.timesteps = int(blocks), int(timesteps)
        self.split = split
        self.entries, self.weights, self.episodes, self._cache = [], [], [], OrderedDict()
        self._fixed_world_coverage = {}
        if catalog is not None:
            for name in ("entries", "weights", "episodes", "discovery", "signature"):
                setattr(self, name, catalog[name])
            if not self.entries:
                raise ValueError("empty broadcast training catalog")
            return
        manifest = json.loads(Path(index).read_text())
        records = manifest["splits"][split]
        indexed_count = len(records)
        # Enumerate only existing caches once. The collection index may list
        # tens of thousands of episodes that were never cached; no stat/read
        # or automatic RGB/voxel materialization is attempted for those rows.
        existing = {path.name: path for path in (Path(cache_root) / split).glob("*.pt")}
        available = [record for record in records if Path(record["path"]).name + ".pt" in existing]
        self.discovery = {"indexed_episodes": indexed_count, "existing_cache_files": len(existing),
                          "matched_index_caches": len(available),
                          "missing_index_caches": indexed_count - len(available),
                          "selection": "existing caches authorized by the requested index split"}
        records = available
        if max_episodes:
            records = records[:max_episodes]
        self.discovery["selected_episodes"] = len(records)
        print(json.dumps({"stage": "cache_catalog", "split": split, **self.discovery}), flush=True)
        for record in records:
            episode = record["path"]
            cache = Path(cache_root) / split / (Path(episode).name + ".pt")
            data = self._read(cache)
            samples = data["samples"]
            ordered = sorted(range(len(samples)), key=lambda i: int(samples[i]["metadata"]["start"]))
            run = []
            sequences = 0
            for index_in_cache in ordered:
                start = int(samples[index_in_cache]["metadata"]["start"])
                if run and start != int(samples[run[-1]]["metadata"]["start"]) + self.timesteps:
                    run = []
                run.append(index_in_cache)
                if len(run) != self.blocks:
                    continue
                has_edit = any(bool(samples[i]["targets"]["block_valid"].any()) for i in run)
                has_attack = False
                for i in run:
                    address = samples[i]["targets"]["address"]
                    valid = samples[i]["targets"]["address_valid"].bool()
                    agents = samples[i]["inputs"]["active"].numel()
                    has_attack |= bool((valid & (address >= VOXEL_COUNT)
                                        & (address < VOXEL_COUNT + agents)).any())
                self.entries.append({
                    "episode": episode, "cache": str(cache.resolve()), "indices": list(run),
                    "start": int(samples[run[0]]["metadata"]["start"]),
                    "has_edit": has_edit, "has_attack": has_attack,
                })
                self.weights.append(1 + (edit_oversample - 1 if has_edit else 0)
                                    + (attack_oversample - 1 if has_attack else 0))
                sequences += 1
                run = []
            self.episodes.append({"episode": episode, "cache": str(cache.resolve()),
                                  "windows": len(samples), "sequences": sequences})
            del data, samples
        if not self.entries:
            raise ValueError(f"{split} has no complete {blocks}-block sequences in existing revision-4 caches")
        self.signature = hashlib.sha256(json.dumps(self.entries, sort_keys=True).encode()).hexdigest()

    def catalog(self):
        return {name: getattr(self, name) for name in ("entries", "weights", "episodes", "discovery", "signature")}

    def _read(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"cache missing (automatic generation disabled): {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("transition_cache_revision") != 4 or data.get("timesteps", 8) != self.timesteps:
            raise ValueError(f"cache revision/horizon mismatch: {path}")
        return data

    def _samples(self, cache):
        if cache not in self._cache:
            self._cache[cache] = self._read(cache)["samples"]
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(cache)
        return self._cache[cache]

    def batch(self, indices, *, block_offset=0, blocks=None):
        blocks = self.blocks if blocks is None else blocks
        if block_offset < 0 or block_offset + blocks > self.blocks:
            raise ValueError("requested batch exceeds its recorded sequence")
        sequences = []
        for index in indices:
            entry = self.entries[index]
            samples = self._samples(entry["cache"])
            sequences.append([samples[i] for i in entry["indices"][block_offset:block_offset + blocks]])
        return [collate_transition([sequence[k] for sequence in sequences]) for k in range(blocks)]

    def training_batch(self, step, rank, batch_size, seed, blocks):
        rng = random.Random(seed + 1000003 * step + 104729 * rank)
        indices = []
        attempts = 0
        max_attempts = max(1000, batch_size * 100)
        while len(indices) < batch_size:
            index = rng.choices(range(len(self.entries)), weights=self.weights, k=1)[0]
            attempts += 1
            if self._entry_has_fixed_world_coverage(index):
                indices.append(index)
            if attempts >= max_attempts and len(indices) < batch_size:
                raise ValueError("unable to sample a full-known fixed-world training sequence")
        offset = rng.randrange(self.blocks) if blocks == 1 else 0
        return self.batch(indices, block_offset=offset, blocks=blocks), indices

    def validation_indices(self, batches, batch_size):
        eligible = [index for index in range(len(self.entries))
                    if self._entry_has_fixed_world_coverage(index)]
        count = min(len(eligible), batches * batch_size)
        if not count:
            raise ValueError("validation has no full-known fixed-world sequences")
        offsets = np.linspace(0, len(eligible) - 1, count, dtype=np.int64)
        return [eligible[int(offset)] for offset in offsets]

    def _entry_has_fixed_world_coverage(self, index):
        """Require initial crops for every actor active anywhere in the rollout."""
        if index in self._fixed_world_coverage:
            return self._fixed_world_coverage[index]
        entry = self.entries[index]
        samples = self._samples(entry["cache"])
        selected = [samples[i] for i in entry["indices"]]
        initial = selected[0]["inputs"]
        known = initial.get("voxel_known")
        if known is None:
            valid = True
        else:
            known_by_actor = known.bool().reshape(known.shape[0], -1).all(-1)
            active_any = torch.stack(
                [sample["inputs"]["active"].bool() for sample in selected]
            ).any(0)
            valid = bool((~active_any | known_by_actor).all())
        self._fixed_world_coverage[index] = valid
        return valid


def masked_mean(value, valid):
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(value)
    return value.masked_fill(~valid, 0.).sum() / valid.sum().clamp_min(1)


def endpoint_loss(output, inputs, targets, stationary_threshold):
    valid = targets["state_valid"].bool() & inputs["active"][:, None].bool()
    endpoint = masked_mean(F.smooth_l1_loss(output["pose"][:, -1, :, :3],
                                           targets["pose"][:, -1, :, :3], reduction="none"), valid[:, -1])
    # Classify stationarity from recorded start/end positions, not the drifted
    # carried start. This term only applies to the final GT-aligned endpoint.
    observed_start = inputs.get("_recorded_initial_pose", inputs["initial_pose"])
    stationary = (targets["pose"][:, -1, :, :3] - observed_start[..., :3]).norm(dim=-1) <= stationary_threshold
    stationary = stationary & valid[:, -1]
    stationary_loss = masked_mean(F.smooth_l1_loss(output["pose"][:, -1, :, :3],
                                                  targets["pose"][:, -1, :, :3], reduction="none"), stationary)
    return endpoint, stationary_loss


def residual_penalty(model, output, active):
    valid = active[:, None].expand_as(output["pose"][..., 0]).bool()
    angle = masked_mean(output["state_residual"][..., 3:5].tanh().square(), valid)
    eye = output["camera_eye_height_residual"] / max(model.cfg.camera_eye_height_residual_limit, 1e-8)
    penalty = angle + masked_mean(eye.square(), valid)
    if model.cfg.velocity_residual_penalty:
        scale = max(model.cfg.velocity_residual_limit, 1e-8)
        penalty = penalty + masked_mean((output["velocity_residual"] / scale).square(), valid)
    return penalty


def endpoint_metrics(output, inputs, targets):
    valid = targets["state_valid"][:, -1].bool() & inputs["active"].bool()
    error = output["pose"][:, -1] - targets["pose"][:, -1]
    angles = torch.atan2(error[..., 3:5].sin(), error[..., 3:5].cos()).abs() * (180. / torch.pi)
    ray = F.cosine_similarity(output["camera_direction"][:, -1].double(),
                              targets["camera_direction"][:, -1].double(), dim=-1).clamp(-1, 1).acos() * (180. / torch.pi)
    return {"position": float(masked_mean(error[..., :3].norm(dim=-1), valid)),
            "yaw": float(masked_mean(angles[..., 0], valid)),
            "pitch": float(masked_mean(angles[..., 1], valid)),
            "ray": float(masked_mean(ray, valid))}


def edit_metrics(inputs, targets, decoded, carry):
    """Exact time/address/material edits, with a shared scoreable-frame mask.

    Null/attack GT frames still penalize a predicted edit. An unknown block
    payload or a target outside initial memory is unscorable, never a negative.
    """
    valid = targets["address_valid"].bool() & targets["state_valid"].bool() & inputs["active"][:, None].bool()
    target_is_edit = (targets["address"] >= 0) & (targets["address"] < VOXEL_COUNT)
    scoreable = valid & (~target_is_edit | targets["block_valid"].bool())
    predicted = (decoded["address"] >= 0) & (decoded["address"] < VOXEL_COUNT) & scoreable
    return {
        "predicted_edits": int(predicted.sum()), "scored_edit_frames": int(scoreable.sum()),
        "original_valid_edit_targets": int(targets["_original_edit_valid"].sum()),
        "covered_edit_targets": int(targets["_fixed_edit_covered"].sum()),
        "out_of_initial_memory_edit_targets": int(targets["_fixed_edit_uncovered"].sum()),
        "shared_world_writes": int(carry["_fixed_shared_write_count"]),
        "same_frame_edit_conflicts": int(carry["_fixed_same_frame_conflicts"]),
    }


@torch.no_grad()
def evaluate(model, dataset, device, max_batches, batch_size, positive_weight, velocity_weight):
    was_training = model.training
    model.eval()
    indices = dataset.validation_indices(max_batches, batch_size)
    totals, endpoints, count, block_count = {}, {}, 0, 0
    for offset in range(0, len(indices), batch_size):
        chosen = indices[offset:offset + batch_size]
        carry = None
        for block, batch in enumerate(dataset.batch(chosen)):
            values = to_device(batch, device)
            inputs, targets = rollout_batch(values["inputs"], values["targets"], carry)
            output = model(inputs)
            _, metrics = unified_transition_loss(model, output, inputs, targets, positive_weight)
            # The recorded block boundary is loss-only metadata.  The model
            # still receives its own carried pose, while the velocity target
            # remains aligned to the dataset rather than accumulated drift.
            loss_inputs = dict(inputs, _recorded_initial_pose=values["inputs"]["initial_pose"])
            velocity, velocity_metrics = supervised_velocity_loss(output, loss_inputs, targets)
            metrics.update(velocity_metrics, velocity_objective=float(velocity.detach()) * velocity_weight)
            decoded = model.decode_interactions(output)
            carry = next_carry(output, inputs, decoded=decoded)
            metrics.update(edit_metrics(inputs, targets, decoded, carry))
            size = len(chosen)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value) * (1 if key in COUNT_METRICS else size)
            if block in {0, 3, 7, dataset.blocks - 1}:
                horizon = (block + 1) * dataset.timesteps
                for key, value in endpoint_metrics(output, inputs, targets).items():
                    name = f"{key}_error_{horizon}f"
                    endpoints[name] = endpoints.get(name, 0.) + value * size
            block_count += size
        count += len(chosen)
    if not count:
        raise ValueError("empty validation selection")
    result = {key: value if key in COUNT_METRICS else value / block_count for key, value in totals.items()}
    result.update({key: value / count for key, value in endpoints.items()})
    result["validation_sequences"] = count
    model.train(was_training)
    return rate_metrics(result)


def parameter_groups(model, backbone_lr, new_lr, weight_decay):
    old, new, assignments = [], [], {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_new = name.startswith(NEW_PARAMETER_PREFIXES) or ".action_attention." in name or ".action_norm." in name
        (new if is_new else old).append(parameter)
        assignments[name] = "new" if is_new else "reused"
    groups = []
    if old:
        groups.append(dict(params=old, lr=backbone_lr, base_lr=backbone_lr,
                           weight_decay=weight_decay, name="reused"))
    if new:
        groups.append(dict(params=new, lr=new_lr, base_lr=new_lr,
                           weight_decay=weight_decay, name="new"))
    return groups, assignments


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-checkpoint", "index", "cache-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-scope", choices=("pose_head", "full"), default="pose_head",
                        help="Train only the displacement residual head or all M2 parameters")
    parser.add_argument("--resume-world-size-change", action="store_true",
                        help="Explicitly migrate a saved run to a new world size in a new output directory")
    parser.add_argument("--resume-batch-size-change", action="store_true",
                        help="Explicitly resume with a different per-rank batch size in a new output directory")
    parser.add_argument("--resume-dataset-change", action="store_true",
                        help="Explicitly resume with a changed training catalog in a new output directory")
    parser.add_argument("--allow-train-val-overlap", action="store_true",
                        help="Allow validation episodes to be present in training; reported validation is train replay")
    parser.add_argument("--allow-cache-root-relocation", action="store_true",
                        help="Use a relocated cache only when the catalog's recorded root no longer exists")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=100, help="Single-block optimizer steps before multi-block carry")
    parser.add_argument("--lr-warmup-steps", type=int, default=50)
    parser.add_argument("--rollout-blocks", type=int, default=4)
    parser.add_argument("--bptt-blocks", type=int, default=1,
                        help="Backpropagate through this many consecutive rollout blocks before detaching")
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="Validation microbatch; 0 reuses --batch-size")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--endpoint-weight", type=float, default=1.)
    parser.add_argument("--stationary-weight", type=float, default=.25)
    parser.add_argument("--stationary-threshold", type=float, default=.05)
    parser.add_argument("--residual-penalty", type=float, default=.01)
    parser.add_argument("--velocity-weight", type=float, default=0.,
                        help="Weight for recorded per-step velocity supervision; use 0.5 for the integrated variant")
    parser.add_argument("--event-count-positive-weight", type=float, default=4.)
    parser.add_argument("--edit-oversample", type=int, default=3)
    parser.add_argument("--attack-oversample", type=int, default=2)
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-val-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.eval_batch_size == 0:
        args.eval_batch_size = args.batch_size
    if args.bptt_blocks < 1 or args.rollout_blocks % args.bptt_blocks:
        raise ValueError("bptt-blocks must be a positive divisor of rollout-blocks")
    if args.rollout_blocks not in (8, 16) or args.eval_blocks not in (8, 16) or args.warmup_steps != 0:
        parser.error("pose rollout experiment requires 8 or 16 train/eval blocks and no single-block warmup")
    if args.resume_world_size_change and args.resume is None:
        parser.error("--resume-world-size-change requires --resume")
    if args.resume_dataset_change and args.resume is None:
        parser.error("--resume-dataset-change requires --resume")
    if min(args.steps, args.rollout_blocks, args.eval_blocks, args.batch_size, args.eval_batches,
           args.eval_batch_size,
           args.save_every, args.log_every, args.threads, args.edit_oversample, args.attack_oversample) < 1:
        parser.error("steps, rollout lengths, sizes and intervals must be positive")
    if min(args.warmup_steps, args.lr_warmup_steps, args.max_train_episodes, args.max_val_episodes) < 0:
        parser.error("warmup lengths and episode caps must be nonnegative")
    if args.velocity_weight < 0:
        parser.error("velocity-weight must be nonnegative")
    rank, world_size, local_rank = (int(os.environ.get(k, default)) for k, default in
                                     (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    requested_device = torch.device(args.device)
    device = (torch.device("cuda", local_rank if world_size > 1 else (requested_device.index or 0))
              if requested_device.type == "cuda" else requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            device_id=device if device.type == "cuda" else None,
        )
    torch.set_num_threads(args.threads)
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32))
    torch.manual_seed(args.seed + rank)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (out / "run.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume is None and any(p.name != "run.lock" for p in out.iterdir()):
            raise ValueError("a fresh run requires a new output directory; use --resume explicitly")
    if world_size > 1:
        dist.barrier()
    began = time.monotonic()
    step = 0

    def status(stage, **extra):
        if rank == 0:
            atomic_json(out / "status.json", dict(stage=stage, step=step, total_steps=args.steps,
                        pid=os.getpid(), hostname=socket.gethostname(), world_size=world_size,
                        updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - began, **extra))

    def log(split, metrics):
        if rank == 0:
            row = dict(step=step, split=split, elapsed_seconds=time.monotonic() - began, **metrics)
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)

    try:
        status("preparing")
        loaded_path = args.resume or args.source_checkpoint
        checkpoint = torch.load(loaded_path, map_location="cpu", weights_only=False)
        source_config = checkpoint["config"]
        supported_schemas = {
            "m2-local-world-writer-v2",
            "m2-local-world-writer-v2-engine-prior",
            "m2-local-world-writer-v2-engine-prior-xyz",
        }
        if source_config.get("schema") not in supported_schemas:
            raise ValueError("training requires an explicit v2 or v2-engine-prior initializer/checkpoint")
        model = build_m2_from_config(source_config)
        model.load_state_dict(checkpoint["model"], strict=True)
        integrated_velocity = model.cfg.velocity_mode == "conditioned_integrated"
        if integrated_velocity != (args.velocity_weight > 0):
            raise ValueError("conditioned-integrated models require positive velocity-weight; compatibility models require zero")
        # Both modes reuse the same long-rollout objective and frozen data.
        # The explicit scope switch prevents an accidental partial/full update.
        trainable = []
        for name, parameter in model.named_parameters():
            enabled = (args.train_scope == "full" or name.startswith("state_residual_head.")
                       or (integrated_velocity and name.startswith("velocity_encoder.")))
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(name)
        expected_pose = [name for name, _ in model.named_parameters()
                         if name.startswith(("state_residual_head.", "velocity_encoder."))]
        if args.train_scope == "pose_head" and trainable != expected_pose:
            raise ValueError(f"unexpected displacement/velocity parameters: {trainable}")
        if args.train_scope == "full" and len(trainable) != len(list(model.named_parameters())):
            raise ValueError("full training did not enable every M2 parameter")
        model.to(device).train()
        groups, assignments = parameter_groups(model, args.backbone_lr, args.new_lr, args.weight_decay)
        optimizer = torch.optim.AdamW(groups)
        catalog_path = Path(os.environ["M2_CATALOG_PATH"])
        catalog_artifact = torch.load(catalog_path, map_location="cpu", weights_only=True)
        if catalog_artifact.get("schema") != "m2-formal-catalog-v1":
            raise ValueError("precomputed catalog schema mismatch")
        if catalog_artifact["index_sha256"] != sha256(args.index):
            raise ValueError("precomputed catalog index mismatch")
        catalog_cache_root = Path(catalog_artifact["cache_root"])
        requested_cache_root = args.cache_root.resolve()
        cache_relocated = str(catalog_cache_root) != str(requested_cache_root)
        if cache_relocated:
            if not args.allow_cache_root_relocation:
                raise ValueError("precomputed catalog cache root mismatch")
            if catalog_cache_root.exists():
                raise ValueError("refusing cache relocation while the catalog's original root still exists")
            if not requested_cache_root.is_dir():
                raise ValueError("relocated cache root is not a directory")
        if catalog_artifact["rollout_blocks"] != args.rollout_blocks or catalog_artifact["eval_blocks"] != args.eval_blocks:
            raise ValueError("precomputed catalog rollout length mismatch")
        train_catalog = catalog_artifact["train"]
        val_catalog = catalog_artifact["val_id"]
        if cache_relocated:
            train_catalog = relocate_catalog(train_catalog, catalog_cache_root, requested_cache_root)
            val_catalog = relocate_catalog(val_catalog, catalog_cache_root, requested_cache_root)
        train_data = CachedRollouts(args.index, args.cache_root, "train", args.rollout_blocks,
                                   timesteps=model.cfg.timesteps, catalog=train_catalog)
        val_data = CachedRollouts(args.index, args.cache_root, "val_id", args.eval_blocks,
                                 timesteps=model.cfg.timesteps, catalog=val_catalog) if rank == 0 else None
        overlap_count = 0
        if rank == 0:
            train_names = {Path(row["episode"]).name for row in train_data.episodes}
            val_names = {Path(row["episode"]).name for row in val_data.episodes}
            overlap = train_names & val_names
            overlap_count = len(overlap)
            if overlap and not args.allow_train_val_overlap:
                raise ValueError("train/validation episode overlap")
        best_position, best_edit = float("inf"), -1.
        config = deepcopy(source_config)
        config.update(training={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      long_rollout_finetune={"scope": args.train_scope,
                                            "trainable": trainable,
                                            "frozen": ("none" if args.train_scope == "full"
                                                       else "all other M2 parameters"),
                                            "objective": "unified state/event plus endpoint, stationary and supervised velocity terms",
                                            "source": "best-edit checkpoint"},
                      training_data_source={"index": str(args.index.resolve()),
                                            "cache_root": str(args.cache_root.resolve()),
                                            "cache_revision": 4, "cache_access": "read_only"},
                      source_checkpoint=str(args.source_checkpoint.resolve()),
                      source_checkpoint_sha256=sha256(args.source_checkpoint),
                      distributed={"world_size": world_size, "batch_size_per_rank": args.batch_size},
                      optimizer_parameter_groups=assignments,
                      checkpoint_selection={"position": "last_horizon_position_error", "edit": "edit_f1"},
                      sampling={
                          "train_signature": train_data.signature,
                          "deterministic_step_rank_sampling": True,
                          "fixed_world_coverage_filter": (
                              "deterministically resample sequences when any actor active during "
                              "the rollout lacks a fully known initial crop"
                          ),
                      },
                      rollout={"training_blocks": args.rollout_blocks, "evaluation_blocks": args.eval_blocks,
                               "backpropagation_blocks": args.bptt_blocks,
                               "backpropagation_frames": args.bptt_blocks * model.cfg.timesteps,
                               "single_block_warmup_steps": args.warmup_steps,
                               "predicted_carry": ["pose", "camera_relative", "camera_direction",
                                                   "pose_camera_history", "engine_prior_velocity",
                                                   "engine_prior_jump_step", "engine_prior_jump_origin_z"],
                               "world_protocol": "fixed_anchor_predicted_world",
                               "initial_world": "union of both players' initial 13^3 crops at fixed world coordinates",
                               "predicted_edits": "shared writeback to every initial crop covering the edited world coordinate",
                               "event_supervision": "GT addresses remapped to initial crop; out-of-crop targets invalid",
                               "ground_truth_crop_anchor_leakage": False,
                               "recorded_boundaries": ["hp", "held_item", "hotbar"],
                               "recorded_controls": True, "full_world_closed_loop": False,
                               "detach_between_blocks": True})
        config["training_data_source"]["precomputed_catalog"] = str(catalog_path.resolve())
        if args.resume:
            step = int(checkpoint["step"])
            if not 0 <= step < args.steps:
                raise ValueError("resume step must be below requested total steps")
            previous = checkpoint["config"]
            previous_train_signature = previous.get("sampling", {}).get("train_signature")
            if previous_train_signature != train_data.signature:
                if not args.resume_dataset_change:
                    raise ValueError("resume training dataset descriptors changed")
                if out == args.resume.resolve().parent:
                    raise ValueError("dataset migration requires a new output directory")
                config["dataset_migration"] = {
                    "checkpoint": str(args.resume.resolve()),
                    "checkpoint_step": step,
                    "old_train_signature": previous_train_signature,
                    "new_train_signature": train_data.signature,
                    "optimizer": "restored",
                    "sampler": "deterministic step/rank sampling remapped to the expanded catalog",
                    "validation_semantics": "train_replay" if args.allow_train_val_overlap else "heldout",
                    "train_val_overlap_episodes": overlap_count if rank == 0 else "computed_on_rank0",
                }
            elif args.resume_dataset_change:
                raise ValueError("explicit dataset migration was requested without a dataset change")
            previous_world_size = int(previous.get("distributed", {}).get("world_size", -1))
            previous_batch_size = int(previous["training"]["batch_size"])
            if previous_world_size != world_size:
                if not args.resume_world_size_change:
                    raise ValueError("world-size migration requires explicit authorization")
                if min(previous_world_size, world_size) < 1:
                    raise ValueError("world sizes must be positive")
                if out == args.resume.resolve().parent:
                    raise ValueError("world-size migration requires a new output directory")
                if len(checkpoint["rng_by_rank"]) != previous_world_size:
                    raise ValueError("checkpoint RNG rank count does not match its world size")
                config["world_size_migration"] = {
                    "from_world_size": previous_world_size,
                    "to_world_size": world_size,
                    "from_global_batch": previous_world_size * previous_batch_size,
                    "to_global_batch": world_size * args.batch_size,
                    "checkpoint": str(args.resume.resolve()),
                    "checkpoint_step": step,
                    "optimizer": "restored",
                    "surviving_rank_rng": "restored",
                    "new_rank_rng": "seeded with seed plus global rank when growing",
                    "dropped_rank_rng": "not used when shrinking",
                    "sampling": "deterministic step and global rank under the new world size",
                }
            elif args.resume_world_size_change:
                raise ValueError("explicit world-size migration was requested without a world-size change")
            if previous.get("rollout", {}).get("world_protocol") != "fixed_anchor_predicted_world":
                raise ValueError("resume must retain the fixed initial predicted-world protocol")
            if previous.get("checkpoint_selection", {}).get("edit") != "edit_f1":
                raise ValueError("resume must retain exact edit-F1 checkpoint selection")
            if int(previous["training"].get("bptt_blocks", 1)) != args.bptt_blocks:
                raise ValueError("resume must retain the same temporal backpropagation length")
            if previous_batch_size != args.batch_size:
                if not args.resume_batch_size_change:
                    raise ValueError("resume batch-size change requires explicit authorization")
                if out == args.resume.resolve().parent:
                    raise ValueError("batch-size migration requires a new output directory")
                config["batch_size_migration"] = {
                    "from_per_rank": previous_batch_size,
                    "to_per_rank": args.batch_size,
                    "world_size": world_size,
                    "from_global_batch": previous_batch_size * previous_world_size,
                    "to_global_batch": args.batch_size * world_size,
                    "checkpoint": str(args.resume.resolve()),
                    "checkpoint_step": step,
                    "optimizer": "restored",
                    "rank_rng": "restored",
                    "learning_rates": "unchanged",
                }
            elif args.resume_batch_size_change:
                raise ValueError("explicit batch-size migration was requested without a batch-size change")
            for key in ("seed", "train_scope", "warmup_steps", "lr_warmup_steps", "rollout_blocks",
                        "backbone_lr", "new_lr", "weight_decay", "endpoint_weight", "stationary_weight",
                        "stationary_threshold", "residual_penalty", "event_count_positive_weight",
                        "velocity_weight", "edit_oversample", "attack_oversample"):
                previous_value = previous["training"].get(key, 0.) if key == "velocity_weight" else previous["training"][key]
                if previous_value != vars(args)[key]:
                    raise ValueError(f"resume changes training contract {key}")
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_position, best_edit = checkpoint["best_position"], checkpoint["best_edit"]
            if args.resume_dataset_change:
                # The criterion is no longer comparable once held-out episodes
                # are added to training, so start a fresh best-of-run record.
                best_position, best_edit = float("inf"), -1.
            if rank < previous_world_size:
                restore_rng(checkpoint["rng_by_rank"][rank], device)
            else:
                random.seed(args.seed + rank)
                np.random.seed((args.seed + rank) % (2**32))
                torch.manual_seed(args.seed + rank)
                if device.type == "cuda":
                    torch.cuda.manual_seed(args.seed + rank)
            if rank == 0 and (out / "metrics.jsonl").exists():
                recorded_steps = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
                if recorded_steps and max(recorded_steps) > step:
                    raise ValueError("output has metrics after resume checkpoint; resume in a fresh directory")
        train_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                          broadcast_buffers=False) if world_size > 1 else model
        if rank == 0:
            atomic_json(out / "config.json", config)
            chosen = val_data.validation_indices(args.eval_batches, args.eval_batch_size)
            atomic_json(out / "evaluation_manifest.json", {
                "signature": val_data.signature, "blocks": args.eval_blocks,
                "sequences": [val_data.entries[i] for i in chosen],
                "semantics": "train_replay" if args.allow_train_val_overlap else "heldout",
                "protocol": "fixed initial geometry + shared predicted edits; GT controls/HP/held boundaries; predicted pose/camera/history",
            })
            atomic_json(out / "data_inventory.json", {"train": train_data.episodes, "val_id": val_data.episodes,
                                                       "train_discovery": train_data.discovery,
                                                       "val_discovery": val_data.discovery,
                                                       "train_sequences": len(train_data.entries),
                                                       "val_sequences": len(val_data.entries)})
        # Rank zero records an initializer baseline on the exact recurring set.
        if step == 0:
            status("initial_validation")
            if rank == 0:
                initializer_validation = evaluate(model, val_data, device, args.eval_batches,
                                                   args.eval_batch_size, args.event_count_positive_weight,
                                                   args.velocity_weight)
                log("val_state_rollout", initializer_validation)
                atomic_json(out / "initializer_validation.json", dict(step=0,
                            metrics=initializer_validation,
                            protocol=config["rollout"], source_checkpoint=config["source_checkpoint"]))
            if world_size > 1:
                dist.barrier()
        timed_stop = False
        for step in range(step + 1, args.steps + 1):
            blocks = 1 if step <= args.warmup_steps else args.rollout_blocks
            bptt_blocks = 1 if blocks == 1 else args.bptt_blocks
            sequence, sampled = train_data.training_batch(step, rank, args.batch_size, args.seed, blocks)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * min(1., step / max(1, args.lr_warmup_steps))
            optimizer.zero_grad(set_to_none=True)
            totals, carry = {}, None
            for chunk_start in range(0, blocks, bptt_blocks):
                chunk_end = chunk_start + bptt_blocks
                synchronized = chunk_end == blocks or world_size == 1
                context = nullcontext() if synchronized else train_model.no_sync()
                chunk_loss = None
                with context:
                    for block in range(chunk_start, chunk_end):
                        batch = sequence[block]
                        values = to_device(batch, device)
                        inputs, targets = rollout_batch(values["inputs"], values["targets"], carry)
                        # Metadata used only by the loss, never inserted into model input.
                        output = train_model(inputs)
                        base_loss, metrics = unified_transition_loss(model, output, inputs, targets,
                                                                     args.event_count_positive_weight)
                        loss_inputs = dict(inputs, _recorded_initial_pose=values["inputs"]["initial_pose"])
                        endpoint, stationary = endpoint_loss(output, loss_inputs, targets, args.stationary_threshold)
                        velocity, velocity_metrics = supervised_velocity_loss(output, loss_inputs, targets)
                        residual = residual_penalty(model, output, inputs["active"])
                        loss = (base_loss + args.endpoint_weight * endpoint
                                + args.stationary_weight * stationary + args.residual_penalty * residual
                                + args.velocity_weight * velocity) / blocks
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite training loss")
                        chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                        metrics.update(endpoint_loss=float(endpoint.detach()), stationary_loss=float(stationary.detach()),
                                       residual_penalty=float(residual.detach()), objective=float(loss.detach()) * blocks,
                                       **velocity_metrics)
                        decoded = model.decode_interactions(output)
                        carry = next_carry(output, inputs, decoded=decoded, detach=False)
                        metrics.update(edit_metrics(inputs, targets, decoded, carry))
                        for key, value in metrics.items():
                            totals[key] = totals.get(key, 0.) + float(value)
                        del output, loss, base_loss
                    chunk_loss.backward()
                carry = detach_carry(carry)
                del chunk_loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite training gradient")
            optimizer.step()
            metrics = {key: value if key in COUNT_METRICS else value / blocks for key, value in totals.items()}
            group_lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
            metrics.update(gradient_norm=float(norm), rollout_blocks=blocks,
                           backbone_lr=group_lrs.get("reused", 0.),
                           new_lr=group_lrs.get("new", 0.))
            names = sorted(metrics)
            scalars = torch.tensor([metrics[key] for key in names], device=device, dtype=torch.float64)
            if world_size > 1:
                dist.all_reduce(scalars)
            metrics = rate_metrics({key: value if key in COUNT_METRICS else value / world_size
                                    for key, value in zip(names, scalars.tolist())})
            if step == 1 or step % args.log_every == 0:
                log("train_state_rollout", metrics)
                status("training", rollout_blocks=blocks, train_metrics=metrics)
            if step % args.save_every != 0 and step != args.steps:
                continue
            if world_size > 1:
                dist.barrier()
            status("validating", rollout_blocks=blocks)
            validation = None
            if rank == 0:
                validation = evaluate(model, val_data, device, args.eval_batches,
                                      args.eval_batch_size, args.event_count_positive_weight, args.velocity_weight)
                log("val_state_rollout", validation)
            # Capture RNG after validation, at the exact point resumed training
            # continues. Sampling is independent of the process-global RNG.
            local_rng = capture_rng(device)
            rng_by_rank = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rng_by_rank, local_rng)
            else:
                rng_by_rank[0] = local_rng
            if rank == 0:
                pos_key = f"position_error_{args.eval_blocks * model.cfg.timesteps}f"
                better_position = validation[pos_key] < best_position
                better_edit = validation["edit_f1"] > best_edit
                best_position = min(best_position, validation[pos_key])
                best_edit = max(best_edit, validation["edit_f1"])
                payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                               config=config, rng_by_rank=rng_by_rank, best_position=best_position,
                               best_edit=best_edit, validation=validation, init_only=False)
                numbered = out / f"step-{step:06d}.pt"
                atomic_checkpoint(numbered, payload)
                atomic_checkpoint(out / "latest.pt", payload)
                if better_position:
                    atomic_checkpoint(out / "best-position.pt", payload)
                    atomic_json(out / "best-position.json", dict(step=step, criterion=pos_key, validation=validation))
                if better_edit:
                    atomic_checkpoint(out / "best-edit.pt", payload)
                    atomic_json(out / "best-edit.json", dict(step=step, criterion="edit_f1", validation=validation))
                status("training" if step < args.steps else "complete", latest_checkpoint=str(numbered),
                       validation=validation, best_position=best_position, best_edit=best_edit)
            if world_size > 1:
                dist.barrier()
            stop_now = bool(args.stop_file and args.stop_file.exists()) if rank == 0 else False
            if world_size > 1:
                stop_flag = [stop_now]
                dist.broadcast_object_list(stop_flag, src=0)
                stop_now = bool(stop_flag[0])
            if stop_now:
                timed_stop = True
                if rank == 0:
                    atomic_json(out / "TIMED_STOPPED.json", dict(step=step,
                        checkpoint=str(out / "latest.pt"), requested_by=str(args.stop_file),
                        updated_utc=datetime.now(timezone.utc).isoformat()))
                    status("timed_stop", latest_checkpoint=str(out / "latest.pt"))
                break
            train_model.train()
        if rank == 0:
            latest = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
            check = build_m2_from_config(latest["config"])
            check.load_state_dict(latest["model"], strict=True)
            if latest["step"] != step or not latest.get("optimizer") or len(latest["rng_by_rank"]) != world_size:
                raise RuntimeError("final checkpoint acceptance failed")
            if not timed_stop:
                if step != args.steps:
                    raise RuntimeError("unexpected early completion")
                atomic_json(out / "COMPLETED.json", dict(steps=args.steps, world_size=world_size,
                            checkpoint=str(out / "latest.pt"), strict_reload=True,
                            optimizer_and_rank_rng_saved=True,
                            trained_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
