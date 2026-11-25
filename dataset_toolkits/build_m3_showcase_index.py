"""Build a compact, trajectory-biased builder policy index.

The showcase run deliberately studies a small in-distribution task suite.  It
does not claim task generalization.  Rows are filtered from an existing
component-aligned dense-policy index, split by whole episode, and reweighted
without changing actions, text, masks, or the policy loss.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


DEFAULT_TASK_PREFIXES = {
    "floor_3x1x2": "build_floor_3x1x2_",
    "wall_4x2x1": "build_wall_4x2x1_",
    "cuboid_3x1x2": "build_cuboid_3x1x2_",
}
DEFAULT_TASK_MASS = {
    "floor_3x1x2": 0.30,
    "wall_4x2x1": 0.50,
    "cuboid_3x1x2": 0.20,
}
DEFAULT_STRATUM_MASS = {"prepare": 0.45, "event": 0.30, "background": 0.25}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_group(row: dict, prefixes: dict[str, str]) -> str | None:
    if row.get("profile") != "language_builder":
        return None
    task_id = str(row.get("task_id") or "")
    return next((name for name, prefix in prefixes.items() if task_id.startswith(prefix)), None)


def filter_rows(rows: list[dict], prefixes: dict[str, str], materials: set[str]) -> list[dict]:
    selected = []
    for source in rows:
        group = task_group(source, prefixes)
        if group is None:
            continue
        task_id = str(source["task_id"])
        if materials and not any(task_id.endswith("_" + material) for material in materials):
            continue
        row = dict(source)
        row["showcase_task"] = group
        row["sampling_schema"] = "showcase-trajectory-v1"
        selected.append(row)
    return selected


def split_episodes(rows: list[dict], dev_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    rng = random.Random(seed)
    episodes = defaultdict(set)
    for row in rows:
        episodes[row["showcase_task"]].add(row["episode_id"])
    dev_ids = set()
    for task, values in sorted(episodes.items()):
        values = sorted(values)
        if len(values) < 2:
            raise ValueError(f"showcase task {task!r} needs at least two episodes")
        rng.shuffle(values)
        count = min(len(values) - 1, max(1, round(len(values) * dev_fraction)))
        dev_ids.update(values[:count])
    train = [row for row in rows if row["episode_id"] not in dev_ids]
    dev = [row for row in rows if row["episode_id"] in dev_ids]
    if {row["episode_id"] for row in train} & {row["episode_id"] for row in dev}:
        raise AssertionError("episode leakage between train and dev")
    return train, dev


def normalize(values: dict[str, float], present: set[str], label: str) -> dict[str, float]:
    unknown = present - set(values)
    if unknown:
        raise ValueError(f"missing {label} masses for {sorted(unknown)}")
    total = sum(values[key] for key in present)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"invalid {label} mass total")
    return {key: values[key] / total for key in present}


def set_weights(
    rows: list[dict],
    task_mass: dict[str, float] = DEFAULT_TASK_MASS,
    stratum_mass: dict[str, float] = DEFAULT_STRATUM_MASS,
) -> None:
    """Give tasks/strata fixed mass and episodes equal mass inside each cell."""
    tasks = {row["showcase_task"] for row in rows}
    task_mass = normalize(task_mass, tasks, "task")
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stratum = str(row.get("sampling_stratum") or "")
        groups[(row["showcase_task"], stratum)][row["episode_id"]].append(row)
    for task in sorted(tasks):
        strata = {stratum for group, stratum in groups if group == task and groups[(group, stratum)]}
        local_stratum_mass = normalize(stratum_mass, strata, f"stratum for {task}")
        for stratum in sorted(strata):
            episodes = groups[(task, stratum)]
            cell_mass = task_mass[task] * local_stratum_mass[stratum]
            for episode_rows in episodes.values():
                weight = cell_mass / len(episodes) / len(episode_rows)
                for row in episode_rows:
                    row["sample_weight"] = weight
    total = sum(float(row["sample_weight"]) for row in rows)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"weights do not sum to one: {total}")


def balanced_dev(rows: list[dict], per_task_stratum: int, seed: int) -> list[dict]:
    """Select a deterministic, compact dev set with episode diversity first."""
    rng = random.Random(seed + 1)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["showcase_task"], row.get("sampling_stratum", ""))].append(row)
    output = []
    for key, values in sorted(groups.items()):
        rng.shuffle(values)
        chosen, used = [], set()
        for row in values:
            if row["episode_id"] in used:
                continue
            chosen.append(row)
            used.add(row["episode_id"])
            if len(chosen) == per_task_stratum:
                break
        if len(chosen) < per_task_stratum:
            for row in values:
                if row in chosen:
                    continue
                chosen.append(row)
                if len(chosen) == per_task_stratum:
                    break
        output.extend({**row, "evaluation_label": "showcase_dev_episode_holdout"} for row in chosen)
    return output


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "tasks": dict(Counter(row["showcase_task"] for row in rows)),
        "task_ids": dict(Counter(row["task_id"] for row in rows)),
        "strata": dict(Counter(row.get("sampling_stratum", "") for row in rows)),
        "move_labels": sum(int(row.get("first8_action_counts", {}).get("move", 0)) for row in rows),
        "place_labels": sum(int(row.get("first8_action_counts", {}).get("place", 0)) for row in rows),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Component-aligned train JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materials", default="", help="Optional comma-separated material suffixes")
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--dev-per-task-stratum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.dev_per_task_stratum < 1:
        parser.error("--dev-per-task-stratum must be positive")
    materials = {value.strip() for value in args.materials.split(",") if value.strip()}
    rows = [json.loads(line) for line in args.source.open() if line.strip()]
    selected = filter_rows(rows, DEFAULT_TASK_PREFIXES, materials)
    missing = set(DEFAULT_TASK_PREFIXES) - {row["showcase_task"] for row in selected}
    if missing:
        raise ValueError(f"source is missing showcase tasks: {sorted(missing)}")
    train, dev_pool = split_episodes(selected, args.dev_fraction, args.seed)
    set_weights(train)
    dev = balanced_dev(dev_pool, args.dev_per_task_stratum, args.seed)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "dev.jsonl", dev)
    report = {
        "status": "showcase_trajectory_index_verified",
        "scope": "in-distribution architecture demonstration; no task-generalization claim",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "task_prefixes": DEFAULT_TASK_PREFIXES,
        "task_sampling_mass": DEFAULT_TASK_MASS,
        "stratum_sampling_mass": DEFAULT_STRATUM_MASS,
        "materials": sorted(materials),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "train": summarize(train),
        "dev": summarize(dev),
        "episode_overlap": 0,
        "loss": "unchanged structured action imitation",
        "interaction_pair_sampling": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
