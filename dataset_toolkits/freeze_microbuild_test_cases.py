"""Freeze the already-held-out microbuild episodes for final closed-loop scoring.

This only creates a case index and metadata. It neither edits source episodes nor
changes train/dev/test membership or the previously selected checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source_bundle / "READY.json").read_text())
    test_path = Path(source["test_index"])
    test_rows = rows(test_path)
    train = rows(Path(source["train_index"]))
    dev = rows(Path(source["dev_index"]))
    if not args.checkpoint.is_file() or not test_rows:
        raise ValueError("selected checkpoint or frozen test index missing")
    ids = {r["episode_id"] for r in train + dev}
    groups = {r["split_group"] for r in train + dev}
    if any(r["episode_id"] in ids or r["split_group"] in groups or r["split"] != "test"
           for r in test_rows):
        raise ValueError("test episode/group overlaps training or development")
    first = {}
    for row in test_rows:
        if (int(row["anchor"]), int(row["context_start"])) == (41, 9):
            if row["episode_id"] in first:
                raise ValueError(f"duplicate first clip: {row['episode_id']}")
            first[row["episode_id"]] = row
    if len(first) != len({r["episode_id"] for r in test_rows}):
        raise ValueError("every test episode needs a frame-41 source clip")
    cases = []
    for episode_id, template in sorted(first.items(), key=lambda item: (item[1]["showcase_task"], item[0])):
        with np.load(Path(template["episode_path"]) / "data.npz", allow_pickle=False) as data:
            action = data["action_continuous"]
            if (len(action) < 232 or int(template["agent_slot"]) != 0
                    or not (action[40, 0, 17] > .5)):
                raise ValueError(f"invalid frame-40 builder setup: {episode_id}")
        row = dict(template)
        row.update(anchor=40, context_start=8,
                   evaluation_label="final_test_world_seed_episode_holdout")
        cases.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    case_path = args.output / "closed_loop_cases.jsonl"
    case_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases))
    ready = {
        "source_bundle": str(args.source_bundle.resolve()),
        "source_test_index": str(test_path.resolve()),
        "source_test_index_sha256": digest(test_path),
        "cases": str(case_path.resolve()),
        "cases_sha256": digest(case_path),
        "case_count": len(cases),
        "by_task": dict(Counter(r["showcase_task"] for r in cases)),
        "rollout_frames": 192,
        "selected_checkpoint": str(args.checkpoint.resolve()),
        "selected_checkpoint_bytes": args.checkpoint.stat().st_size,
        "selection_basis": "step500 chosen using frozen development cases only",
        "text_catalog": source["text_catalog"],
        "text_cache": source["text_cache"],
        "m2_cache_root": source["m2_cache_root"],
        "train_episode_count": len({r["episode_id"] for r in train}),
        "dev_episode_count": len({r["episode_id"] for r in dev}),
        "test_episode_count": len(cases),
        "script_sha256": digest(Path(__file__)),
    }
    (args.output / "READY.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_count": len(cases), "by_task": ready["by_task"],
                      "cases_sha256": ready["cases_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
