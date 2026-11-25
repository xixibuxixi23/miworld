"""Measure M3 action sensitivity to text with pixels and state held fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from miworld.data.inserted_policy_dataset import canonical_text
from miworld.models.m3 import M3Policy, M3PolicyArgs
from miworld.models.renderer_codec import WanRendererCodec
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from train_scripts.train_policy import load_renderer, lookup_text


def load_rows(path: Path, count: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "m3_closed_loop_bc_v1":
        raise ValueError(f"unsupported supervision file: {path}")
    return payload["samples"][:count]


def stack_conditions(rows, device):
    return {
        key: torch.stack([row["conditions"][key] for row in rows]).to(device)
        for key in rows[0]["conditions"]
    }


def repeat_conditions(conditions, repeats: int):
    return {key: value.repeat_interleave(repeats, dim=0) for key, value in conditions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH; repeat for every checkpoint")
    parser.add_argument("--supervision", action="append", type=Path, required=True)
    parser.add_argument("--renderer-checkpoint", type=Path, required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--text-catalog", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    checkpoints = []
    for spec in args.checkpoint:
        label, separator, path = spec.partition("=")
        if not separator:
            raise ValueError("checkpoint must be LABEL=PATH")
        checkpoints.append((label, Path(path)))

    groups = []
    goals = []
    for path in args.supervision:
        rows = load_rows(path, args.samples_per_task)
        groups.append((path.parent.name, rows))
        goal = rows[0]["shared_text"]
        if goal not in goals:
            goals.append(goal)
    if len(goals) != len(args.supervision):
        raise ValueError("each supervision input must provide a distinct goal text")

    catalog = json.loads(args.text_catalog.read_text())
    text_ids = {canonical_text(row["text"]): int(row["text_id"])
                for row in catalog["texts"]}
    cache = load_file(str(args.text_cache), device="cpu")
    goal_ids = torch.tensor([text_ids[canonical_text(goal)] for goal in goals])
    empty_id = text_ids[""]

    renderer, _ = load_renderer(args.renderer_checkpoint, return_config=True)
    codec = WanRendererCodec(args.wan_vae).to(device).eval().requires_grad_(False)
    first = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False, mmap=True)
    config = M3PolicyArgs(**first["config"])
    model = M3Policy(renderer, config).to(device).eval().requires_grad_(False)
    del first

    encoded = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for source, rows in groups:
            rgb = torch.stack([row["rgb"] for row in rows]).to(device).float().div_(255.)
            encoded.append((source, rows, codec.encode(rgb), stack_conditions(rows, device)))

    report = {
        "schema": "m3-text-counterfactual-v1",
        "contract": "same RGB/state/history; only shared structured goal text changes",
        "goals": goals,
        "samples_per_source": args.samples_per_task,
        "checkpoints": {},
    }
    for label, path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if M3PolicyArgs(**payload["config"]) != config:
            raise ValueError(f"policy config mismatch: {path}")
        model.families.load_state_dict(payload["families"], strict=True)
        step = int(payload["step"])
        per_source = {}
        all_ranges = []
        all_action_changes = []
        for source, rows, latent, conditions in encoded:
            repeats = len(goals)
            batch = len(rows)
            expanded_latent = latent.repeat_interleave(repeats, dim=0)
            expanded_conditions = repeat_conditions(conditions, repeats)
            families = torch.tensor(
                [FAMILY_TO_ID[row["family"]] for row in rows for _ in goals], device=device)
            profiles = torch.tensor(
                [PROFILE_TO_ID[row["profile"]] for row in rows for _ in goals], device=device)
            shared_ids = goal_ids.repeat(batch)
            current_ids = torch.full((batch * repeats,), empty_id, dtype=torch.long)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                shared, shared_mask = lookup_text(cache, shared_ids, device, expanded_latent.dtype)
                current, current_mask = lookup_text(cache, current_ids, device, expanded_latent.dtype)
                _, logits = model(
                    expanded_latent, expanded_conditions, families, profiles,
                    shared_text=shared, shared_text_mask=shared_mask,
                    current_text=current, current_text_mask=current_mask,
                    policy_indices=torch.arange(1, 9, device=device)[None].expand(batch * repeats, -1),
                )
                decoded = model.decode_execution(logits).float()
                place = logits["keys"][:, :8, 8].float().sigmoid()
            place = place.reshape(batch, repeats, 8).cpu()
            decoded = decoded.reshape(batch, repeats, 8, 23).cpu()
            rows_out = []
            for sample_index in range(batch):
                sample_place = place[sample_index]
                sample_actions = decoded[sample_index]
                place_mean = sample_place.mean(-1)
                place_max = sample_place.max(-1).values
                place_range = float(place_mean.max() - place_mean.min())
                baseline = sample_actions[0]
                action_changes = [int((sample_actions[i] != baseline).sum())
                                  for i in range(repeats)]
                mouse_delta = [float((sample_actions[i, :, 21:23] -
                                      baseline[:, 21:23]).abs().mean())
                               for i in range(repeats)]
                all_ranges.append(place_range)
                all_action_changes.extend(action_changes[1:])
                rows_out.append({
                    "sample": sample_index,
                    "block": int(rows[sample_index]["block"]),
                    "agent_slot": int(rows[sample_index]["agent_slot"]),
                    "place_mean_by_goal": place_mean.tolist(),
                    "place_max_by_goal": place_max.tolist(),
                    "decoded_place_count_by_goal": sample_actions[:, :, 9].sum(-1).int().tolist(),
                    "decoded_action_changes_vs_first_goal": action_changes,
                    "mouse_l1_vs_first_goal": mouse_delta,
                    "place_mean_range": place_range,
                })
            per_source[source] = rows_out
        report["checkpoints"][label] = {
            "path": str(path),
            "step": step,
            "mean_place_probability_range_across_text": sum(all_ranges) / len(all_ranges),
            "mean_decoded_action_changes_vs_first_goal": (
                sum(all_action_changes) / len(all_action_changes)),
            "sources": per_source,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        label: {
            "step": row["step"],
            "mean_place_probability_range_across_text":
                row["mean_place_probability_range_across_text"],
            "mean_decoded_action_changes_vs_first_goal":
                row["mean_decoded_action_changes_vs_first_goal"],
        }
        for label, row in report["checkpoints"].items()
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
