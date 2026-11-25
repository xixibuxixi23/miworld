"""Repair the first autonomous builder decision without changing split membership.

The original index begins at frame 41. At evaluation the first policy call is
frame 40 and executes eight actions, so this additional row must supervise the
material selection *and* its immediately following movement and camera actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_decision(template: dict) -> dict:
    row = dict(template)
    ranges = template["absolute_supervision_ranges"]
    if len(ranges) != 1 or ranges[0][0] != 41:
        raise ValueError("expected the original activity segment beginning at frame 41")
    episode = Path(template["episode_path"])
    with np.load(episode / "data.npz", allow_pickle=False) as data:
        action = data["action_continuous"][:, int(template["agent_slot"])]
        active = data["language_policy_train_mask"][:, int(template["agent_slot"])]
    end = int(ranges[0][1])
    if not (end > 48 and active[40:end].all() and action[40, 17] > 0.5):
        raise ValueError(f"first decision or supervision mask invalid: {episode}")
    initial = action[40:48]
    counts = {
        "n": 8,
        "select_material": int((initial[:, 12:21] > 0.5).any(axis=1).sum()),
        "move": int((initial[:, :8] > 0.5).any(axis=1).sum()),
        "place": int((initial[:, 9] > 0.5).sum()),
        "dig": int((initial[:, 8] > 0.5).sum()),
        "camera": int((np.abs(initial[:, 21:]) > 1e-4).any(axis=1).sum()),
    }
    row.update(
        anchor=40,
        context_start=8,
        absolute_supervision_ranges=[[40, end]],
        dense_valid_targets=end - 40,
        dense_valid_anchors=(end - 40 + 7) // 8,
        first8_valid_targets=8,
        first8_action_counts=counts,
        sampling_stratum="first_decision",
        sampling_schema="microbuild-active-exec8-v3-first-decision",
    )
    return row


def augment(source: Path) -> tuple[list[dict], dict]:
    original = read_rows(source)
    first = {}
    for row in original:
        if (int(row["anchor"]), int(row["context_start"])) == (41, 9):
            if row["episode_id"] in first:
                raise ValueError(f"duplicate first activity clip: {row['episode_id']}")
            first[row["episode_id"]] = row
    if len(first) != len({r["episode_id"] for r in original}):
        raise ValueError("every episode needs a frame-41 activity clip")
    extra = [first_decision(template) for template in first.values()]
    output = original + extra
    groups = defaultdict(lambda: defaultdict(list))
    for row in output:
        groups[row["microbuild_task"]][row["episode_id"]].append(row)
    for episodes in groups.values():
        for clips in episodes.values():
            for row in clips:
                row["sample_weight"] = 1 / len(groups) / len(episodes) / len(clips)
    output.sort(key=lambda r: (r["microbuild_task"], r["episode_id"], r["anchor"]))
    return output, {
        "original_rows": len(original), "first_decision_rows": len(extra),
        "tasks": dict(Counter(r["microbuild_task"] for r in extra)),
        "first8_move": sum(r["first8_action_counts"]["move"] for r in extra),
        "first8_select": sum(r["first8_action_counts"]["select_material"] for r in extra),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ready = json.loads((args.bundle / "READY.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "source_bundle": str(args.bundle.resolve()),
        "fixed_dev_cases": ready["cases"],
        "test_index_untouched": ready["test_index"],
        "script_sha256": sha(Path(__file__)),
    }
    for split in ("train", "dev"):
        source = Path(ready[f"{split}_index"])
        prepared, counts = augment(source)
        target = args.output / f"{split}.jsonl"
        target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared))
        result[split] = {
            **counts, "index": str(target.resolve()),
            "source_sha256": sha(source), "index_sha256": sha(target),
        }
    (args.output / "READY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
