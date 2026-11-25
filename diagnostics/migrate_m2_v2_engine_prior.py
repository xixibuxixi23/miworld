"""Warm-start the minimal engine-prior M2 while resetting obsolete XYZ residuals."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from miworld.models.local_world_writer import build_m2_from_config


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def migrate(source):
    config = deepcopy(source["config"])
    if config.get("schema") != "m2-local-world-writer-v2":
        raise ValueError("source must be the conditioned-integrated M2 v2")
    model_cfg = config["model"]
    if model_cfg.get("position_prior") != "action" or model_cfg.get("velocity_mode") != "conditioned_integrated":
        raise ValueError("source must use the integrated action prior")
    config["schema"] = "m2-local-world-writer-v2-engine-prior"
    model = build_m2_from_config(config)
    state = {name: value.detach().clone() for name, value in source["model"].items()}
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("engine-prior model changed the original parameter contract")
    for name in state:
        if state[name].shape != expected[name].shape:
            raise ValueError(f"shape mismatch: {name}")
    state["state_residual_head.weight"][:3].zero_()
    state["state_residual_head.bias"][:3].zero_()
    model.load_state_dict(state, strict=True)
    metadata = {
        "schema": "m2-v2-minimal-engine-prior-initializer-v1",
        "source_step": int(source.get("step", -1)),
        "target_schema": config["schema"],
        "preserved": "all model tensors except XYZ rows of state_residual_head",
        "reset": ["state_residual_head.weight[:3]", "state_residual_head.bias[:3]"],
        "engine_prior": "5x 1/120s substeps; target speed 4; speed delta cap .25; one-frame action lag",
        "optimizer": "new AdamW; source optimizer and RNG intentionally omitted",
    }
    return state, config, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("choose a new initializer path")
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    state, config, metadata = migrate(source)
    metadata.update(source=str(args.source.resolve()), source_sha256=sha256(args.source))
    payload = dict(model=state, config=config, step=0, init_only=True, migration=metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    metadata.update(output=str(args.output.resolve()), output_sha256=sha256(args.output))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
