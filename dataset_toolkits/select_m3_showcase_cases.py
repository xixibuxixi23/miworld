"""Select early, movement-bearing closed-loop cases from the showcase dev index."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def select_cases(rows: list[dict], per_task: int, seed: int) -> list[dict]:
    if per_task < 1:
        raise ValueError("per_task must be positive")
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[(row.get("showcase_task"), row["episode_id"])].append(row)
    candidates = defaultdict(list)
    for (task, _), values in by_episode.items():
        # Start as early as possible, while preferring a clip whose first
        # decision contains locomotion/select preparation.
        values = sorted(values, key=lambda row: (
            0 if row.get("sampling_stratum") == "prepare" else 1,
            int(row.get("context_start", row["anchor"])),
            int(row.get("agent_slot", 0)),
        ))
        candidates[task].append(values[0])
    rng = random.Random(seed)
    selected = []
    for task, values in sorted(candidates.items()):
        values = sorted(values, key=lambda row: (int(row.get("context_start", row["anchor"])),
                                                  row["episode_id"]))
        # Keep the earliest half deterministic; shuffle only ties/tail so the
        # case list stays reproducible when source ordering changes.
        head = values[:per_task]
        if len(head) < per_task:
            tail = values[len(head):]
            rng.shuffle(tail)
            head.extend(tail[:per_task-len(head)])
        selected.extend({**row, "evaluation_label": "showcase_closed_loop_dev"} for row in head)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.source.open() if line.strip()]
    selected = select_cases(rows, args.per_task, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "source": str(args.source.resolve()),
        "cases": len(selected),
        "case_indices": [{"index": i, "showcase_task": row.get("showcase_task"),
                          "episode_id": row["episode_id"], "anchor": row["anchor"]}
                         for i, row in enumerate(selected)],
    }
    (args.output.with_suffix(".manifest.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
