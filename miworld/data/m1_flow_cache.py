"""Memory-mapped legacy M1 cache reader for unified flow finetuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FIELDS = (
    "latent",
    "image_rays",
    "raw",
    "agent_mask",
    "camera_valid",
)


class M1FlowCache(Dataset):
    """Read one split without copying the 36GB cache into host memory."""

    def __init__(self, root: str | Path, split: str = "train", max_samples: int = 0):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        splits = self.metadata["splits"]
        if split not in splits:
            raise ValueError(f"unknown M1 cache split {split!r}; available={sorted(splits)}")
        order = list(splits)
        self.start = sum(int(splits[name]) for name in order[: order.index(split)])
        self.count = int(splits[split])
        if max_samples:
            self.count = min(self.count, int(max_samples))
        self.split = split
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r") for name in FIELDS
        }
        visible_path = self.root / "visible_latent_mask.npy"
        self.visible = np.load(visible_path, mmap_mode="r") if visible_path.exists() else None
        nodes = sorted(set(int(value) for value in self.metadata["raw_lut"]))
        self.context_nodes = tuple(nodes)
        lookup = np.full(max(nodes) + 1, -1, dtype=np.int64)
        lookup[np.asarray(nodes)] = np.arange(len(nodes), dtype=np.int64)
        self.raw_to_context = torch.from_numpy(lookup)
        self.air_context_class = nodes.index(126)

    def __len__(self):
        return self.count

    def __getitem__(self, index: int):
        if not 0 <= index < self.count:
            raise IndexError(index)
        source = self.start + int(index)
        row = {
            "index": torch.tensor(source, dtype=torch.int64),
            "latent": torch.from_numpy(np.array(self.arrays["latent"][source], copy=True)),
            "image_rays": torch.from_numpy(
                np.array(self.arrays["image_rays"][source], copy=True)
            ),
            "raw": torch.from_numpy(np.array(self.arrays["raw"][source], copy=True)).long(),
            "view_valid": torch.from_numpy(
                np.array(
                    self.arrays["agent_mask"][source]
                    & self.arrays["camera_valid"][source],
                    copy=True,
                )
            ),
        }
        row["visible_latent_mask"] = (
            torch.from_numpy(np.array(self.visible[source], copy=True))
            if self.visible is not None
            else torch.zeros((12, 12, 12), dtype=torch.bool)
        )
        return row

    def encode_raw(self, raw: torch.Tensor) -> torch.Tensor:
        lookup = self.raw_to_context.to(raw.device)
        if raw.min() < 0 or raw.max() >= len(lookup):
            raise ValueError("raw node lies outside the cache vocabulary lookup")
        encoded = lookup[raw.long()]
        if torch.any(encoded < 0):
            raise ValueError("raw node is absent from the legacy M1 vocabulary")
        return encoded
