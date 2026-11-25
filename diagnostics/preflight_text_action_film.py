"""Audit legacy -> text-action-FiLM migration and exact zero-init parity."""
import argparse
import json
from pathlib import Path

import torch

from miworld.models.inserted_policy import InsertedPolicyArgs
from miworld.models.joint_action_decoder import JointActionDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = InsertedPolicyArgs(**payload["config"])
    prefix = "language_builder.head."
    state = {key[len(prefix):]: value for key, value in payload["families"].items()
             if key.startswith(prefix)}
    hidden = int(state["policy_projection.weight"].shape[1])
    common = dict(
        hidden_size=hidden,
        horizons=cfg.horizons,
        text_hidden=cfg.text_hidden_size,
        decoder_hidden=cfg.builder_action_decoder_hidden,
        layers=cfg.builder_action_decoder_layers,
        heads=cfg.builder_action_decoder_heads,
        current_text_scale=cfg.current_text_scale,
        continuous_mouse=cfg.continuous_mouse,
        visual_grid=cfg.builder_action_visual_grid,
        history_frames=cfg.history_frames,
        checkpoint_layers=False,
    )
    legacy = JointActionDecoder(**common)
    routed = JointActionDecoder(
        **common, text_action_film=True,
        text_action_film_hidden=cfg.builder_text_action_film_hidden)
    legacy.load_state_dict(state, strict=True)
    incompatible = routed.load_state_dict(state, strict=False)
    expected = {name for name, _ in routed.named_parameters() if name.startswith("task_film.")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError((incompatible.missing_keys, incompatible.unexpected_keys, expected))
    device = torch.device("cuda:0")
    legacy.to(device).eval(); routed.to(device).eval()
    torch.manual_seed(20261124)
    b, t, slots = 1, cfg.history_frames, max(2, cfg.builder_spatial_tokens + 1)
    states = torch.randn(b, t, slots, hidden, device=device)
    valid = torch.ones(b, t, dtype=torch.bool, device=device)
    shared = torch.randn(b, 5, cfg.text_hidden_size, device=device)
    shared_mask = torch.ones(b, 5, dtype=torch.bool, device=device)
    current = torch.randn(b, 3, cfg.text_hidden_size, device=device)
    current_mask = torch.ones(b, 3, dtype=torch.bool, device=device)
    video = torch.randn(b, t, 8, 8, hidden, device=device)
    with torch.inference_mode():
        before = legacy(states, valid, shared, shared_mask, current, current_mask, video)
        after = routed(states, valid, shared, shared_mask, current, current_mask, video)
    differences = {key: float((before[key] - after[key]).abs().max()) for key in before}
    if any(value != 0. for value in differences.values()):
        raise ValueError(f"zero-init parity failed: {differences}")
    print(json.dumps({
        "source_step": int(payload["step"]),
        "missing_adapter_keys": sorted(expected),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "max_abs_logit_difference": differences,
        "exact_parity": True,
    }, indent=2))


if __name__ == "__main__":
    main()
