"""Consecutive cached unified-M2 windows for long-horizon fine-tuning."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .fill_dataset import BlockVocabulary
from .transition_dataset import VOXEL_COUNT, collate_transition, episode_windows
from .transition_stream import TRANSITION_CACHE_REVISION


def _contains_edit(sequence):
    return any(bool(sample["targets"]["block_valid"].any()) for sample in sequence)


def _contains_attack(sequence):
    for sample in sequence:
        address = sample["targets"]["address"]
        valid = sample["targets"]["address_valid"].bool()
        residents = sample["inputs"]["active"].numel()
        if bool((valid & (address >= VOXEL_COUNT)
                 & (address < VOXEL_COUNT + residents)).any()):
            return True
    return False


def consecutive_sequences(samples, rollout_blocks, timesteps=8):
    """Yield non-overlapping runs separated by one complete M2 block."""
    run = []
    for sample in sorted(samples, key=lambda row: int(row["metadata"]["start"])):
        start = int(sample["metadata"]["start"])
        if run and start != int(run[-1]["metadata"]["start"]) + timesteps:
            run = []
        run.append(sample)
        if len(run) == rollout_blocks:
            yield run
            run = []


class UnifiedRolloutStream(IterableDataset):
    """Load the production M2 cache and group it into long contiguous runs."""

    def __init__(self, index, vocabulary, cache_root, split, *, dataset_root,
                 items, rollout_blocks=16, seed=7, rank=0, world_size=1,
                 fixed_order=False, edit_oversample=1, attack_oversample=1,
                 timesteps=8):
        manifest = json.loads(Path(index).read_text())
        self.root = Path(dataset_root)
        self.records = manifest["splits"][split]
        self.vocabulary = BlockVocabulary.load(vocabulary)
        self.cache_root = Path(cache_root) / split
        self.split = split
        self.items = dict(items)
        self.rollout_blocks = int(rollout_blocks)
        self.timesteps = int(timesteps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.fixed_order = bool(fixed_order)
        self.edit_oversample = int(edit_oversample)
        self.attack_oversample = int(attack_oversample)
        self.epoch = 0
        if min(self.rollout_blocks, self.timesteps) < 1:
            raise ValueError("rollout_blocks and timesteps must be positive")
        if min(self.edit_oversample, self.attack_oversample) < 1:
            raise ValueError("event oversampling must be at least one")

    def _load(self, path, worker_id):
        cache_name = (path.name + ".pt" if self.timesteps == 8
                      else f"{path.name}.t{self.timesteps}.pt")
        cache = self.cache_root / cache_name
        result = None
        if cache.exists():
            candidate = torch.load(cache, map_location="cpu", weights_only=True)
            cached_steps = candidate.get("timesteps", 8)
            if (candidate.get("transition_cache_revision") == TRANSITION_CACHE_REVISION
                    and cached_steps == self.timesteps):
                result = candidate
        if result is None:
            samples, audit = episode_windows(
                path, self.vocabulary, self.items, windows_per_episode=None,
                split=self.split, stride=self.timesteps,
                strict_block_payload=True, timesteps=self.timesteps)
            result = {
                "samples": samples, "audit": dict(audit), "episode": str(path),
                "transition_cache_revision": TRANSITION_CACHE_REVISION,
                "timesteps": self.timesteps,
            }
            self.cache_root.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(f".{self.rank}.{worker_id}.tmp")
            torch.save(result, temporary)
            temporary.replace(cache)
        return result["samples"]

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + (0 if self.fixed_order else self.epoch))
        order = list(range(len(self.records)))
        if not self.fixed_order:
            rng.shuffle(order)
        self.epoch += 1
        shard = self.rank * workers + worker_id
        shards = self.world_size * workers
        for index in order[shard::shards]:
            path = self.root / self.records[index]["path"]
            base = list(consecutive_sequences(
                self._load(path, worker_id), self.rollout_blocks,
                timesteps=self.timesteps))
            sequences = list(base)
            if self.edit_oversample > 1:
                sequences.extend([s for s in base if _contains_edit(s)]
                                 * (self.edit_oversample - 1))
            if self.attack_oversample > 1:
                sequences.extend([s for s in base if _contains_attack(s)]
                                 * (self.attack_oversample - 1))
            if not self.fixed_order:
                rng.shuffle(sequences)
            yield from sequences


def collate_unified_rollouts(sequences):
    blocks = len(sequences[0])
    if any(len(sequence) != blocks for sequence in sequences):
        raise ValueError("rollout sequences must have equal length")
    return [collate_transition([sequence[block] for sequence in sequences])
            for block in range(blocks)]
