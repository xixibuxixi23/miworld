"""Build exact-anchor M2 caches and a fixed closed-loop case index."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from miworld.data.fill_dataset import BlockVocabulary
from miworld.data.transition_dataset import episode_windows


DEFAULT_ITEMS = {
    name: index
    for index, name in enumerate((
        "", "mcl_core:brick_block", "mcl_core:cobble", "mcl_core:dirt",
        "mcl_core:glass", "mcl_core:stonebrick", "mcl_core:tree",
        "mcl_core:wood", "mcl_ocean:sea_lantern", "mcl_tools:pick_iron",
        "textagent_task:npc_axe", "textagent_task:npc_sword",
    ))
}


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--anchor", type=int, help="Use the same observation frame for each selected case.")
    args = parser.parse_args()
    if args.output_index.exists():
        raise FileExistsError(args.output_index)

    rows = [json.loads(line) for line in args.source_index.open() if line.strip()]
    if len(set(args.indices)) != len(args.indices):
        parser.error("--indices must be unique")
    selected = []
    for index in args.indices:
        if not 0 <= index < len(rows):
            parser.error(f"index {index} is outside [0, {len(rows)})")
        row = dict(rows[index])
        row["source_row_index"] = index
        task = row.get("corrective_fault") or row.get("showcase_task")
        if not task:
            raise ValueError(f"missing evaluation task for row {index}")
        row["showcase_task"] = str(task)
        if args.anchor is not None:
            if args.anchor < 8:
                parser.error("--anchor must retain at least eight observed context frames")
            row["anchor"] = args.anchor
        selected.append(row)

    by_episode: dict[Path, list[dict]] = defaultdict(list)
    for row in selected:
        by_episode[Path(row["episode_path"])].append(row)
    vocabulary = BlockVocabulary.load(args.vocabulary)
    receipts = []
    for episode, episode_rows in by_episode.items():
        manifest = json.loads((episode / "manifest.json").read_text())
        split = str(manifest.get("split", episode_rows[0].get("split", "train")))
        anchors = {int(row["anchor"]) for row in episode_rows}
        samples, audit = episode_windows(
            episode, vocabulary, DEFAULT_ITEMS, windows_per_episode=None,
            split=split, stride=1, strict_block_payload=True, timesteps=8)
        kept = [sample for sample in samples if int(sample["metadata"]["start"]) in anchors]
        found = {int(sample["metadata"]["start"]) for sample in kept}
        if found != anchors:
            raise ValueError(f"missing exact M2 anchors for {episode.name}: {sorted(anchors - found)}")
        cache_path = args.cache_root / split / f"{episode.name}.pt"
        if cache_path.exists():
            raise FileExistsError(cache_path)
        atomic_torch_save({
            "samples": kept,
            "audit": dict(audit),
            "episode": str(episode),
            "transition_cache_revision": 4,
            "timesteps": 8,
            "evaluation_exact_anchors": sorted(anchors),
        }, cache_path)
        receipts.append({
            "episode": episode.name,
            "split": split,
            "anchors": sorted(anchors),
            "cache": str(cache_path),
        })

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_index.with_name(f".{args.output_index.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    temporary.replace(args.output_index)
    receipt_path = args.output_index.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps({
        "source_index": str(args.source_index),
        "source_indices": args.indices,
        "cases": receipts,
    }, indent=2) + "\n")
    print(json.dumps({"cases": len(selected), "episodes": len(by_episode)}, sort_keys=True))


if __name__ == "__main__":
    main()
