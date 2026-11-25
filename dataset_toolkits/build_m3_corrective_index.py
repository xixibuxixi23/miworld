#!/usr/bin/env python3
"""Build an exact-range M3 policy index from validated corrective episodes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np


def canonical(value) -> str:
    return " ".join(str(value or "").strip().split())


def action_counts(actions: np.ndarray) -> dict[str, int]:
    return {
        "n": int(len(actions)),
        "move": int((actions[:, :4] > .5).any(-1).sum()),
        "dig": int((actions[:, 8] > .5).sum()),
        "place": int((actions[:, 9] > .5).sum()),
        "camera": int((np.abs(actions[:, 21:23]) > 1e-8).any(-1).sum()),
    }


def appearance_aliases(manifest: dict) -> dict[str, str]:
    """Explicitly map new human skins onto the frozen wide/slim ID contract."""
    aliases = {}
    for value in manifest.get("skin_catalog_ids", {}).values():
        if value in {
            "human-default-v1-steve-wide",
            "human-default-v1-alex-slim",
        }:
            continue
        if value.endswith("-wide"):
            aliases[value] = "human-default-v1-steve-wide"
        elif value.endswith("-slim"):
            aliases[value] = "human-default-v1-alex-slim"
        else:
            raise ValueError(f"cannot alias unknown human body type: {value}")
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--context-frames", type=int, default=129)
    parser.add_argument("--dense-warmup", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = []
    excluded = Counter()
    validated = [json.loads(line) for line in args.manifest.open() if line.strip()]
    for validated_row in validated:
        episode = Path(validated_row["episode_dir"])
        manifest = json.loads((episode / "manifest.json").read_text())
        frames = int(manifest["frames"])
        task_text = canonical(manifest.get("task_text"))
        aliases = appearance_aliases(manifest)
        if not task_text:
            raise ValueError(f"missing task text: {episode}")
        with np.load(
            episode / manifest.get("training_data_file", "data.npz"),
            allow_pickle=False,
        ) as data:
            actions = np.asarray(data["action_continuous"])
        for line in Path(validated_row["window_file"]).open():
            window = json.loads(line)
            start = int(window["action_start"])
            end = int(window["action_end_exclusive"])
            if end - start != 8 or int(window["horizon"]) != 8:
                raise ValueError(f"corrective window is not exactly eight actions: {episode}")
            context_start = start - args.dense_warmup
            if context_start < 0:
                excluded["insufficient_causal_warmup"] += 1
                continue
            if context_start + args.context_frames > frames:
                excluded["insufficient_context_tail"] += 1
                continue
            agent = int(window["agent_index"])
            target = actions[start:end, agent]
            counts = action_counts(target)
            stratum = "event" if counts["place"] or counts["dig"] else (
                "prepare" if counts["move"] else "background"
            )
            rows.append({
                "episode_id": episode.name,
                "episode_path": str(episode),
                "split": "train",
                "scenario_id": manifest["scenario_id"],
                "task_id": manifest["task_id"],
                "task_text": task_text,
                "shared_task_text": task_text,
                "current_task_text": task_text,
                "current_text_schema": "component_observation_v1",
                "current_text_timeline": [{
                    "component_id": manifest.get("recipe_id", "corrective_task"),
                    "observation_frame": 0,
                    "text": task_text,
                }],
                "family": "language_builder",
                "profile": "language_builder",
                "agent_slot": agent,
                "context_start": context_start,
                "anchor": start,
                "dense_valid_targets": 8,
                "dense_valid_anchors": 1,
                "absolute_supervision_ranges": [[start, end]],
                "appearance_aliases": aliases,
                "sampling_stratum": stratum,
                "sampling_schema": "corrective-recovery-exact-range-v1",
                "first8_action_counts": counts,
                "corrective_fault": window["fault"],
                "corrective_intervention_id": int(window["intervention_id"]),
                "teacher_movement_steps": int(window["teacher_movement_steps"]),
                "teacher_camera_steps": int(window["teacher_camera_steps"]),
                "engine_recovery_success": bool(window["engine_recovery_success"]),
                "fault_action_included": bool(window["fault_action_included"]),
                "evaluation_label": "train_replay_not_holdout",
            })

    if not rows:
        raise ValueError("no corrective windows satisfy the M3 causal context contract")
    if any(row["fault_action_included"] or not row["engine_recovery_success"] for row in rows):
        raise ValueError("unsafe corrective supervision row survived validation")

    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode_rows in by_episode.values():
        weight = 1.0 / len(by_episode) / len(episode_rows)
        for row in episode_rows:
            row["sample_weight"] = weight
    if abs(sum(row["sample_weight"] for row in rows) - 1.0) > 1e-9:
        raise AssertionError("corrective sample weights do not sum to one")

    random.Random(args.seed).shuffle(rows)
    args.output.mkdir(parents=True)
    for name in ("train.jsonl", "train_replay.jsonl"):
        with (args.output / name).open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema": "miworld-m3-corrective-index-v1",
        "status": "verified",
        "source_manifest": str(args.manifest.resolve()),
        "episodes": len(by_episode),
        "windows": len(rows),
        "supervised_actions": 8 * len(rows),
        "faults": dict(sorted(Counter(row["corrective_fault"] for row in rows).items())),
        "tasks": dict(sorted(Counter(row["task_id"] for row in rows).items())),
        "strata": dict(sorted(Counter(row["sampling_stratum"] for row in rows).items())),
        "windows_with_movement": sum(row["teacher_movement_steps"] > 0 for row in rows),
        "windows_with_camera": sum(row["first8_action_counts"]["camera"] > 0 for row in rows),
        "windows_with_place": sum(row["first8_action_counts"]["place"] > 0 for row in rows),
        "excluded": dict(sorted(excluded.items())),
        "sampling": "equal episode mass; uniform windows within episode",
        "appearance_contract": (
            "row-local Solaris human skin aliases preserve wide/slim body type; "
            "raw manifests and reference images are unchanged"
        ),
        "loss": "unchanged M3 policy loss; exact absolute action-range mask",
    }
    (args.output / "data_verified.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
