#!/usr/bin/env python3
"""Freeze and index accepted microbuild episodes for the existing M3 loader.

No data are copied or changed. Snapshot membership is immutable, metadata are
hashed, and large payloads are fingerprinted by size/mtime (optionally SHA256).
Run with the project's training Python; inspection/window helpers use stdlib.
The dense loader has no left padding: its first anchor is model_start + 32,
not 128. Early teacher actions are rejected instead of silently discarded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

TASKS = ("wall3", "platform2x2", "corner_l5")
CONTEXT, WARMUP, HORIZON, STRIDE = 129, 32, 32, 8
EXECUTION_COVERAGE = 72  # Nine anchors, each supervising its first eight actions.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, *, hash_payload: bool = False) -> dict:
    stat = path.stat()
    row = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_payload or path.suffix in {".json", ".jsonl"}:
        row["sha256"] = sha256(path)
    return row


def window_specs(*, frames: int, model_start: int, active_start: int,
                 active_end_inclusive: int, post_wait: int = 8) -> tuple[list[dict], dict]:
    """Plan disjoint supervision ranges on legal, possibly overlapping clips.

    Future context pixels are loaded by the existing causal renderer; the policy
    uses only the observed prefix at each dense anchor. The caller must retain
    the trainer's causal preflight and tail-horizon-weight=0.
    """
    if not 0 <= post_wait <= 8:
        raise ValueError("post_wait must be between zero and eight actions")
    if not 0 <= model_start < frames:
        raise ValueError("invalid model_start")
    if not 0 <= active_start <= active_end_inclusive < frames - 1:
        raise ValueError("active interval is outside the action array")
    if frames - model_start < CONTEXT:
        raise ValueError("episode has fewer than 129 observations after model_start")
    first = max(active_start, model_start + WARMUP)
    end = min(active_end_inclusive + 1 + post_wait, frames - 1)
    audit = {"active_start": active_start, "active_end_exclusive": active_end_inclusive + 1,
             "earliest_loader_anchor": model_start + WARMUP,
             "early_active_steps_unavailable": max(0, min(first, active_end_inclusive + 1) - active_start),
             "requested_post_wait": post_wait, "supervision_end_exclusive": end}
    if first > active_end_inclusive:
        raise ValueError("all active actions precede the loader's first legal dense anchor")
    specs = []
    cursor = first
    while cursor < end:
        context_start = min(cursor - WARMUP, frames - CONTEXT)
        # Tail clamping may change the anchor grid. Keep only actions that fall
        # in executed first-eight prefixes; no action appears in two rows.
        cover_end = min(context_start + WARMUP + EXECUTION_COVERAGE, end)
        if cover_end <= cursor:
            raise ValueError("active tail cannot fit the loader's predict32 context contract")
        specs.append({"context_start": context_start, "anchor": cursor,
                      "absolute_supervision_ranges": [[cursor, cover_end]]})
        cursor = cover_end
    audit["windows"] = len(specs)
    return specs, audit


def identity_keys(manifest: dict) -> list[str]:
    """Conservatively join episodes sharing any world/seed/group identity."""
    keys = []
    for field in ("seed", "world_seed"):
        if manifest.get(field) is not None:
            keys.append("seed:" + str(manifest[field]))
    for field in ("world_id", "root_group_id"):
        if manifest.get(field) not in (None, ""):
            keys.append(field + ":" + str(manifest[field]))
    if not any(key.startswith("seed:") for key in keys):
        raise ValueError("manifest needs an explicit seed or world_seed for disjoint splitting")
    return sorted(set(keys))


def assign_splits(episodes: list[dict], *, seed: int, dev_fraction: float,
                  test_fraction: float) -> None:
    """Union shared identities across tasks, then hash whole groups once."""
    if not (0 < dev_fraction < 1 and 0 < test_fraction < 1
            and dev_fraction + test_fraction < 1):
        raise ValueError("dev/test fractions must be positive and sum to less than one")
    parent = list(range(len(episodes)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner = {}
    for index, episode in enumerate(episodes):
        for key in episode["identity_keys"]:
            if key in owner:
                parent[root(index)] = root(owner[key])
            owner[key] = index
    groups = defaultdict(list)
    for index, episode in enumerate(episodes):
        groups[root(index)].append(episode)
    for group in groups.values():
        keys = sorted({key for episode in group for key in episode["identity_keys"]})
        group_id = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
        rank = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(), 16) / 2**256
        split = "test" if rank < test_fraction else (
            "dev" if rank < test_fraction + dev_fraction else "train")
        for episode in group:
            episode.update(split=split, split_group=group_id)


def discover(collection_root: Path, *, hash_data: bool = False,
             recorded_pre_roll: bool = False,
             catalogs: tuple[Path, ...] = ()) -> tuple[list[dict], dict]:
    accepted, rejected = [], Counter()
    # A sorted initial listing freezes membership even while CPU collection
    # continues elsewhere. Partial publishing directories are never selected.
    if catalogs:
        paths = []
        for catalog in catalogs:
            for line in catalog.read_text().splitlines():
                row = json.loads(line)
                if row.get("accepted") is True:
                    episode = Path(row["destination"]).resolve()
                    if not episode.is_relative_to(collection_root):
                        raise ValueError(f"catalog episode is outside collection root: {episode}")
                    paths.append(episode / "simple_build_pilot.json")
        paths.sort()
    else:
        paths = sorted(collection_root.rglob("simple_build_pilot.json"))
    seen = set()
    for summary_path in paths:
        if any(part.startswith(".") or part == "_rejected" for part in summary_path.relative_to(collection_root).parts):
            rejected["hidden_or_rejected"] += 1
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("main_set_usable") is not True:
            rejected["main_set_unusable"] += 1
            continue
        if summary.get("structure") not in TASKS:
            rejected["different_structure"] += 1
            continue
        if recorded_pre_roll and summary.get("pre_roll_steps_requested") != 40:
            rejected["no_recorded_pre_roll"] += 1
            continue
        episode = summary_path.parent.resolve()
        if episode in seen:
            raise ValueError(f"duplicate episode path: {episode}")
        seen.add(episode)
        manifest_path = episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if summary.get("task_id") != manifest.get("task_id"):
            raise ValueError(f"summary/manifest task mismatch: {episode}")
        payload = episode / manifest.get("training_data_file", "data.npz")
        if not payload.is_file():
            raise FileNotFoundError(payload)
        files = {}
        for path in sorted(episode.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(episode))] = fingerprint(path, hash_payload=hash_data)
        accepted.append({"episode_id": episode.name, "episode_path": str(episode),
                         "structure": summary["structure"], "identity_keys": identity_keys(manifest),
                         "manifest": manifest, "summary": summary, "files": files})
    return accepted, dict(rejected)


def verify_fingerprints(episodes: list[dict]) -> None:
    for episode in episodes:
        root = Path(episode["episode_path"])
        for name, expected in episode["files"].items():
            actual = fingerprint(root / name, hash_payload="sha256" in expected)
            if actual != expected:
                raise ValueError(f"source changed during indexing: {root / name}")


def episode_rows(episode: dict, *, post_wait: int) -> tuple[list[dict], dict]:
    # Import production code at execution time, so pure split/window checks can
    # run on a laptop without Torch. This avoids duplicating loader semantics.
    import numpy as np
    from miworld.data.fill_dataset import TextAgentFillDataset
    from miworld.data.inserted_policy_dataset import dense_action_targets, mask_targets_to_absolute_ranges
    from dataset_toolkits.build_m3_corrective_index import action_counts, appearance_aliases, canonical

    manifest, summary = episode["manifest"], episode["summary"]
    task_text = canonical(manifest.get("task_text"))
    if not task_text:
        raise ValueError(f"missing task_text: {episode['episode_path']}")
    path = Path(episode["episode_path"])
    rows = []
    with np.load(path / manifest.get("training_data_file", "data.npz"), allow_pickle=False) as data:
        frames = len(data["cam_pos"])
        actions = np.asarray(data["action_continuous"])
        if actions.ndim != 3 or actions.shape[1:] != (2, 23) or len(actions) + 1 != frames:
            raise ValueError(f"unexpected action/observation contract: {path}: {actions.shape}, {frames}")
        model_start = TextAgentFillDataset._model_start(data, manifest)
        specs, audit = window_specs(frames=frames, model_start=model_start,
                                   active_start=int(summary["active_start"]),
                                   active_end_inclusive=int(summary["active_end_inclusive"]),
                                   post_wait=post_wait)
        # Calibration movement in the summary may precede the trainable region.
        # Audit actual policy labels rather than treating calibration as BC data.
        mask_key = "language_policy_train_mask" if "language_policy_train_mask" in data else "policy_train_mask"
        trainable = np.asarray(data[mask_key][:, 0], dtype=bool)
        substantive = (np.abs(actions[:, 0, :10]) > 1e-8).any(-1) | (np.abs(actions[:, 0, 21:23]) > 1e-8).any(-1)
        early_end = model_start + WARMUP
        early = trainable[:early_end] & substantive[:early_end]
        audit["early_trainable_active_actions"] = int(early.sum())
        if early.any():
            raise ValueError(f"{path}: {int(early.sum())} active teacher actions precede anchor {early_end}; "
                             "the existing loader has no left padding; fix collection/preprocessing before training")
        seen_first8 = set()
        for spec in specs:
            targets, valid, indices = dense_action_targets(data, "language_builder", 0, spec["context_start"])
            valid = mask_targets_to_absolute_ranges(valid, indices, spec["absolute_supervision_ranges"], spec["context_start"])
            executed = valid[:, :8]
            if not executed.any():
                continue
            absolute = indices[:, -1].numpy()[:, None] + spec["context_start"] + np.arange(8)[None]
            supervised = set(map(int, absolute[executed]))
            if supervised & seen_first8:
                raise AssertionError("executed action labels repeat across microbuild windows")
            seen_first8.update(supervised)
            counts = action_counts(targets[:, :8][executed])
            stratum = "event" if counts["place"] or counts["dig"] else (
                "prepare" if counts["move"] or counts["camera"] else "background")
            rows.append({"episode_id": episode["episode_id"], "episode_path": str(path),
                         "split": episode["split"], "split_group": episode["split_group"],
                         "microbuild_task": episode["structure"], "showcase_task": episode["structure"],
                         "scenario_id": manifest["scenario_id"], "task_id": manifest["task_id"],
                         "task_text": task_text, "shared_task_text": task_text, "current_task_text": task_text,
                         "current_text_schema": "component_observation_v1",
                         "current_text_timeline": [{"component_id": "whole_microbuild_task", "observation_frame": 0, "text": task_text}],
                         "family": "language_builder", "profile": "language_builder", "agent_slot": 0,
                         **spec, "dense_valid_targets": int(valid.sum()),
                         "dense_valid_anchors": int(valid.any(-1).sum()),
                         "first8_valid_targets": int(executed.sum()), "first8_action_counts": counts,
                         "appearance_aliases": appearance_aliases(manifest), "sampling_stratum": stratum,
                         "sampling_schema": "microbuild-active-exec8-v1",
                         "evaluation_label": "training" if episode["split"] == "train" else "world_seed_episode_holdout"})
        expected = set(np.flatnonzero(trainable & substantive).tolist())
        expected = {index for index in expected if summary["active_start"] <= index <= summary["active_end_inclusive"]}
        uncovered = sorted(expected - seen_first8)
        audit.update(first8_unique_labels=len(seen_first8), uncovered_trainable_active_actions=uncovered)
        if uncovered:
            raise ValueError(f"{path}: active actions not covered by valid first-eight supervision: {uncovered[:16]}")
    if not rows:
        raise ValueError(f"no supervised execution prefixes: {path}")
    return rows, audit


def set_weights(rows: list[dict]) -> None:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["microbuild_task"]][row["episode_path"]].append(row)
    for episodes in groups.values():
        for values in episodes.values():
            for row in values:
                row["sample_weight"] = 1 / len(groups) / len(episodes) / len(values)
    if not math.isclose(sum(row["sample_weight"] for row in rows), 1):
        raise AssertionError("weights must sum to one")


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, action="append", default=[],
                        help="Freeze committed episodes from these catalogs without scanning other phases.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--dev-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--post-wait", type=int, default=8)
    parser.add_argument("--hash-data", action="store_true", help="Hash all payload bytes; otherwise large files use size/mtime")
    parser.add_argument("--require-recorded-pre-roll", action="store_true",
                        help="Select the cohort recorded with 40 idle steps before building; still audit real actions.")
    parser.add_argument("--min-train-per-task", type=int, default=0)
    parser.add_argument("--min-dev-per-task", type=int, default=0)
    parser.add_argument("--snapshot-only", action="store_true", help="Inspect immutable membership/splits without loading Torch or creating training indices")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.collection_root.is_dir():
        raise NotADirectoryError(args.collection_root)
    episodes, rejected = discover(args.collection_root.resolve(), hash_data=args.hash_data,
                                  recorded_pre_roll=args.require_recorded_pre_roll, catalogs=tuple(args.catalog))
    if not episodes:
        raise ValueError("no accepted microbuild episodes")
    assign_splits(episodes, seed=args.seed, dev_fraction=args.dev_fraction, test_fraction=args.test_fraction)
    counts = {split: dict(Counter(e["structure"] for e in episodes if e["split"] == split))
              for split in ("train", "dev", "test")}
    if min(args.min_train_per_task, args.min_dev_per_task) < 0:
        parser.error("minimum split counts must be nonnegative")
    deficits = {f"{split}/{task}": minimum - counts[split].get(task, 0)
                for split, minimum in (("train", args.min_train_per_task), ("dev", args.min_dev_per_task))
                for task in TASKS if counts[split].get(task, 0) < minimum}
    if deficits and not args.snapshot_only:
        raise ValueError(f"cannot freeze first training batch; insufficient fixed split counts: {deficits}")
    missing = {split: sorted(set(TASKS) - set(tasks)) for split, tasks in counts.items() if set(tasks) != set(TASKS)}
    snapshot = {"schema": "miworld-microbuild-snapshot-v1", "collection_root": str(args.collection_root.resolve()),
                "seed": args.seed, "dev_fraction": args.dev_fraction, "test_fraction": args.test_fraction,
                "payload_fingerprint": "sha256" if args.hash_data else "size_and_mtime_ns",
                "require_recorded_pre_roll": args.require_recorded_pre_roll,
                "catalogs": [str(p.resolve()) for p in args.catalog],
                "scope": "fixed membership and source fingerprints; episode files are not copied or locked",
                "episodes": episodes, "rejected": rejected, "counts": counts, "missing_split_tasks": missing}
    args.output.mkdir(parents=True)
    write_json(args.output / "source_snapshot.json", snapshot)
    if args.snapshot_only:
        print(json.dumps({"status": "snapshot_only_not_training_ready", "counts": counts, "missing": missing}, indent=2))
        return
    if missing:
        raise ValueError(f"split lacks tasks; collect more independent episodes before freezing training: {missing}")
    rows, audits = [], []
    for episode in episodes:
        values, audit = episode_rows(episode, post_wait=args.post_wait)
        rows.extend(values)
        audits.append({"episode_path": episode["episode_path"], **audit})
    verify_fingerprints(episodes)
    outputs = {}
    for split in ("train", "dev", "test"):
        selected = [row for row in rows if row["split"] == split]
        set_weights(selected)
        index = args.output / f"{split}.jsonl"
        with index.open("x") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        outputs[split] = {"rows": len(selected), "sha256": sha256(index),
                          "episodes": len({row["episode_path"] for row in selected}),
                          "first8_action_counts": dict(sum((Counter(row["first8_action_counts"]) for row in selected), Counter()))}
    write_json(args.output / "coverage_audit.json", audits)
    report = {"schema": "miworld-microbuild-index-v1", "status": "index_verified_loader_smoke_pending",
              "snapshot_sha256": sha256(args.output / "source_snapshot.json"), "splits": outputs,
              "contract": {"context_frames": CONTEXT, "horizons": HORIZON, "dense_stride": STRIDE,
                           "dense_warmup": WARMUP, "tail_horizon_weight": 0, "builder_interaction_pairs": False,
                           "agent_slot": 0, "post_wait": args.post_wait, "causal_preflight_required": True},
              "text": "whole-task instruction at frame zero; build a fresh text catalog/cache",
              "sampling": "equal task mass, equal episode mass within task, uniform windows within episode",
              "evaluation": "test split is for final reporting; use dev for checkpoint selection; this index does not verify video decoding or full-loop compatibility"}
    write_json(args.output / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
