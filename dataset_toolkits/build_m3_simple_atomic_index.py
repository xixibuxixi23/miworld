#!/usr/bin/env python3
"""Filter a component-aligned M3 policy index to small, atomic text tasks.

The selected builder tasks use one operation, one structure, and one material.
Composite plans such as houses, tower bridges, build-then-dig, and dig-then-fill
are excluded even when their English prompt is short.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re


TASK_PATTERNS = {
    "wall": re.compile(r"^build_wall_(\d+)x(\d+)x1_(.+)$"),
    "floor": re.compile(r"^build_floor_(\d+)x1x(\d+)_(.+)$"),
    "cuboid": re.compile(r"^build_cuboid_(\d+)x(\d+)x(\d+)_(.+)$"),
    "fill_hole": re.compile(r"^fill_hole_(\d+)x1x(\d+)_(.+)$"),
}
FAMILY_MASS = {
    "wall": 0.25,
    "floor": 0.20,
    "cuboid": 0.20,
    "fill_hole": 0.35,
}
STRATUM_MASS = {"event": 0.45, "prepare": 0.35, "background": 0.20}
FORBIDDEN_TEXT = re.compile(r"\b(and then|then|after|before|together|doorway|roof|bridge)\b", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def classify_task(task_id: str) -> tuple[str, tuple[int, ...], str] | None:
    for family, pattern in TASK_PATTERNS.items():
        match = pattern.fullmatch(task_id)
        if not match:
            continue
        groups = match.groups()
        if family == "wall":
            width, height, material = int(groups[0]), int(groups[1]), groups[2]
            if width <= 4 and height <= 2:
                return family, (width, height, 1), material
        elif family == "floor":
            width, depth, material = int(groups[0]), int(groups[1]), groups[2]
            if width * depth <= 9 and max(width, depth) <= 4:
                return family, (width, 1, depth), material
        elif family == "cuboid":
            width, height, depth, material = (
                int(groups[0]), int(groups[1]), int(groups[2]), groups[3]
            )
            if width * height * depth <= 12 and height <= 2 and max(width, depth) <= 4:
                return family, (width, height, depth), material
        elif family == "fill_hole":
            width, depth, material = int(groups[0]), int(groups[1]), groups[2]
            if width * depth <= 9 and max(width, depth) <= 4:
                return family, (width, 1, depth), material
    return None


def row_texts(row: dict) -> list[str]:
    values = [canonical(row.get("shared_task_text") or row.get("task_text"))]
    if row.get("current_task_text"):
        values.append(canonical(row["current_task_text"]))
    values.extend(
        canonical(component.get("text"))
        for component in row.get("current_text_timeline") or ()
        if component.get("text")
    )
    return [value for value in values if value]


def texts_are_simple(row: dict) -> bool:
    values = row_texts(row)
    return bool(values) and all(len(value.split()) <= 14 and not FORBIDDEN_TEXT.search(value)
                                for value in values)


def select_rows(source: Path) -> tuple[list[dict], Counter]:
    selected = []
    excluded = Counter()
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("profile") != "language_builder":
                excluded["non_builder"] += 1
                continue
            task_id = str(row.get("task_id") or "")
            task = classify_task(task_id)
            if task is None:
                excluded["non_atomic_or_large_task"] += 1
                continue
            text = canonical(row.get("shared_task_text") or row.get("task_text"))
            if not text:
                excluded["missing_text"] += 1
                continue
            if not texts_are_simple(row):
                excluded["complex_text"] += 1
                continue
            if int(row.get("first8_action_counts", {}).get("dig", 0)):
                excluded["mixed_dig_action"] += 1
                continue
            family, dimensions, material = task
            output = dict(row)
            output.update(
                simple_task_family=family,
                simple_task_dimensions=list(dimensions),
                simple_task_material=material,
                sampling_schema="m3-simple-atomic-v1",
            )
            selected.append(output)
    return selected, excluded


def split_episodes(rows: list[dict], dev_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev fraction must be in (0, 1)")
    by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[row["simple_task_family"]].add(row["episode_id"])
    rng = random.Random(seed)
    dev_ids = set()
    for family, values in sorted(by_family.items()):
        episodes = sorted(values)
        if len(episodes) < 2:
            raise ValueError(f"task family {family!r} has fewer than two episodes")
        rng.shuffle(episodes)
        count = min(len(episodes) - 1, max(1, round(len(episodes) * dev_fraction)))
        dev_ids.update(episodes[:count])
    train = [row for row in rows if row["episode_id"] not in dev_ids]
    dev = [row for row in rows if row["episode_id"] in dev_ids]
    if {row["episode_id"] for row in train} & {row["episode_id"] for row in dev}:
        raise AssertionError("episode leakage between train and dev")
    return train, dev


def normalize(values: dict[str, float], present: set[str]) -> dict[str, float]:
    total = sum(values[key] for key in present)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid sampling mass")
    return {key: values[key] / total for key in present}


def set_weights(rows: list[dict]) -> None:
    families = {row["simple_task_family"] for row in rows}
    family_mass = normalize(FAMILY_MASS, families)
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["simple_task_family"], str(row.get("sampling_stratum") or ""))][
            row["episode_id"]
        ].append(row)
    for family in sorted(families):
        strata = {stratum for group, stratum in groups if group == family}
        stratum_mass = normalize(STRATUM_MASS, strata)
        for stratum in sorted(strata):
            episodes = groups[(family, stratum)]
            cell_mass = family_mass[family] * stratum_mass[stratum]
            for episode_rows in episodes.values():
                weight = cell_mass / len(episodes) / len(episode_rows)
                for row in episode_rows:
                    row["sample_weight"] = weight
    if abs(sum(float(row["sample_weight"]) for row in rows) - 1.0) > 1e-9:
        raise AssertionError("sample weights do not sum to one")


def summarize(rows: list[dict]) -> dict:
    texts = Counter(canonical(row.get("shared_task_text") or row.get("task_text")) for row in rows)
    components = Counter()
    for row in rows:
        if row.get("current_task_text"):
            components[canonical(row["current_task_text"])] += 1
        for component in row.get("current_text_timeline") or ():
            if component.get("text"):
                components[canonical(component["text"])] += 1
    return {
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "task_ids": len({row["task_id"] for row in rows}),
        "families": dict(sorted(Counter(row["simple_task_family"] for row in rows).items())),
        "strata": dict(sorted(Counter(str(row.get("sampling_stratum") or "") for row in rows).items())),
        "materials": dict(sorted(Counter(row["simple_task_material"] for row in rows).items())),
        "place_labels": sum(int(row.get("first8_action_counts", {}).get("place", 0)) for row in rows),
        "dig_labels": sum(int(row.get("first8_action_counts", {}).get("dig", 0)) for row in rows),
        "move_labels": sum(int(row.get("first8_action_counts", {}).get("move", 0)) for row in rows),
        "unique_shared_texts": len(texts),
        "unique_current_texts": len(components),
        "longest_shared_texts": sorted(texts, key=lambda value: (len(value.split()), value), reverse=True)[:10],
        "longest_current_texts": sorted(components, key=lambda value: (len(value.split()), value), reverse=True)[:10],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)

    selected, excluded = select_rows(args.source)
    if not selected:
        raise ValueError("no simple atomic tasks selected")
    train, dev = split_episodes(selected, args.dev_fraction, args.seed)
    set_weights(train)
    report = {
        "schema": "miworld-m3-simple-atomic-index-v1",
        "status": "dry_run" if args.dry_run else "verified",
        "scope": "small atomic builder tasks; not broad task generalization",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "task_rules": {
            "wall": "width<=4 and height<=2",
            "floor": "area<=9 and max side<=4",
            "cuboid": "volume<=12, height<=2, max horizontal side<=4",
            "fill_hole": "area<=9 and max side<=4",
            "text": "all shared/component instructions <=14 words; no sequencing or compound-plan terms",
            "excluded": "houses, towers, bridges, compound build/dig/fill, collection, mining collaboration, mixed dig windows",
        },
        "family_sampling_mass": FAMILY_MASS,
        "stratum_sampling_mass": STRATUM_MASS,
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "train": summarize(train),
        "dev": summarize(dev),
        "excluded": dict(sorted(excluded.items())),
        "episode_overlap": 0,
        "labels_and_text": "copied unchanged from source rows",
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    assert args.output is not None
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "dev.jsonl", dev)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
