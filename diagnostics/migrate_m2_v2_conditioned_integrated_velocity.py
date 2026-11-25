"""Create an M2 initializer with past-speed conditioning and integrated velocity output."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from miworld.models.local_world_writer import build_m2_from_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate(source):
    config = deepcopy(source["config"])
    if config.get("schema") != "m2-local-world-writer-v2":
        raise ValueError("source must be an M2 v2 checkpoint")
    previous_prior = config["model"].get(
        "position_prior", "action" if "distance_per_step" in config["model"] else "direct"
    )
    config["model"].update(
        position_prior="action",
        velocity_mode="conditioned_integrated",
        velocity_residual_limit=0.2,
    )
    model = build_m2_from_config(config)
    state = model.state_dict()
    source_state = source["model"]
    unexpected = sorted(set(source_state) - set(state))
    if unexpected:
        raise ValueError(f"source contains unknown tensors: {unexpected}")
    for name, value in source_state.items():
        if state[name].shape != value.shape:
            raise ValueError(f"shape changed for inherited tensor {name}")
        state[name] = value.detach().clone()
    state["state_residual_head.weight"][:3].zero_()
    state["state_residual_head.bias"][:3].zero_()
    state["velocity_encoder.weight"].zero_()
    state["velocity_encoder.bias"].zero_()
    model.load_state_dict(state, strict=True)
    metadata = {
        "schema": "m2-v2-conditioned-integrated-velocity-initializer-v1",
        "source_step": int(source.get("step", -1)),
        "source_position_prior": previous_prior,
        "target_position_prior": "action",
        "velocity_input": "initial_pose.xyz - final_history_pose.xyz; normalized by distance_per_step and clipped to [-4,4]",
        "velocity_output": "action-prior per-step velocity + bounded learned residual; position is its cumulative sum",
        "velocity_residual_limit": 0.2,
        "preserved": "all inherited tensors except XYZ rows of state_residual_head",
        "reset": ["state_residual_head.weight[:3]", "state_residual_head.bias[:3]"],
        "added_zero_initialized": ["velocity_encoder.weight", "velocity_encoder.bias"],
        "optimizer": "new AdamW; source optimizer and RNG intentionally omitted",
        "recommended_velocity_weight": 0.5,
    }
    return state, config, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    state, config, metadata = migrate(source)
    metadata.update(source=str(args.source.resolve()), source_sha256=sha256(args.source))
    payload = {"model": state, "config": config, "step": 0,
               "init_only": True, "migration": metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    metadata.update(output=str(args.output.resolve()), output_sha256=sha256(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
