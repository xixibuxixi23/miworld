"""Add the omitted frame-40 material selection to a frozen microbuild split.

The original split and episode membership remain unchanged. This creates new
train/dev indices; the held-out test index and fixed closed-loop cases are only
referenced, never read or altered.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def augment(source: Path) -> tuple[list[dict], dict]:
    original = rows(source)
    first = {}
    for row in original:
        if (int(row["anchor"]), int(row["context_start"])) == (41, 9):
            if row["episode_id"] in first:
                raise ValueError(f"duplicate first clip: {row['episode_id']}")
            first[row["episode_id"]] = row
    if len(first) != len({r["episode_id"] for r in original}):
        raise ValueError("every episode must have a frame-41 first clip")
    extra = []
    for episode_id, template in first.items():
        row = dict(template)
        row.update(anchor=40, context_start=8,
                   absolute_supervision_ranges=[[40, 41]],
                   dense_valid_targets=1, dense_valid_anchors=1,
                   first8_valid_targets=1,
                   first8_action_counts={"n": 1, "move": 0, "camera": 0,
                                         "dig": 0, "place": 0, "select_material": 1},
                   sampling_stratum="select_material",
                   sampling_schema="microbuild-active-exec8-v2-select-slot")
        extra.append(row)
    output = original + extra
    groups = defaultdict(lambda: defaultdict(list))
    for row in output:
        groups[row["microbuild_task"]][row["episode_id"]].append(row)
    for episodes in groups.values():
        for clips in episodes.values():
            for row in clips:
                row["sample_weight"] = 1 / len(groups) / len(episodes) / len(clips)
    output.sort(key=lambda r: (r["microbuild_task"], r["episode_id"], r["anchor"]))
    return output, {"original_rows": len(original), "selection_rows": len(extra),
                    "tasks": dict(Counter(r["microbuild_task"] for r in extra))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ready = json.loads((args.bundle / "READY.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"source_bundle": str(args.bundle.resolve()),
              "fixed_dev_cases": ready["cases"], "test_index_untouched": ready["test_index"]}
    for split in ("train", "dev"):
        source = Path(ready[f"{split}_index"])
        prepared, counts = augment(source)
        target = args.output / f"{split}.jsonl"
        target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared))
        result[split] = {**counts, "index": str(target.resolve()),
                         "source_sha256": digest(source), "index_sha256": digest(target)}
    (args.output / "READY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
