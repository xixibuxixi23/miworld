"""M2 v2 contracts. These tests perform no optimizer steps or training runs."""
from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import traceback
from unittest import TestCase

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miworld.kinematics import direction_from_angles
from miworld.models.local_world_writer import (
    LocalWorldWriter, LocalWorldWriterArgs, build_m2_from_config,
    recorded_body_angle_trajectory,
)
from miworld.models.local_world_writer_v2 import LocalWorldWriterV2, LocalWorldWriterV2Args


torch.set_num_threads(2)
VOXELS = 13 ** 3


def synthetic_inputs(players=2, steps=8):
    """Fully known, genuinely different local crops and observed histories."""
    generator = torch.Generator().manual_seed(46)
    pose = torch.zeros(1, players, 5)
    pose[0, :, 0] = torch.arange(players) * 7.25
    pose[0, :, 1] = torch.arange(players) * 1.5
    pose[0, :, 2] = 3.5
    pose[0, :, 3] = torch.arange(players) * 0.4
    grid = torch.stack(torch.meshgrid(*(torch.arange(-6, 7) for _ in range(3)), indexing="ij"), -1).reshape(1, 1, VOXELS, 3).float()
    relative = grid.expand(1, players, -1, -1).clone()
    relative -= (pose[..., :3] - torch.floor(pose[..., :3] + 0.5))[:, :, None]
    history = pose[:, None].expand(-1, 4, -1, -1).clone()
    history[..., 0] -= torch.tensor([0.4, 0.3, 0.2, 0.1])[None, :, None]
    eye = torch.tensor([0.0, 0.0, 1.6]).expand(1, players, 3).clone()
    actions = torch.zeros(1, steps, players, 23)
    actions[:, :, :, 0] = 1
    actions[:, 1:5, :, 8] = 1
    actions[..., 21] = 0.01
    history_actions = torch.zeros(1, 4, players, 23)
    history_actions[..., 0] = 1
    history_actions[:, -1, :, 21] = -0.02
    return {
        "active": torch.ones(1, players, dtype=torch.bool),
        "voxels": torch.randint(0, 7, (1, players, 13, 13, 13), generator=generator),
        "voxel_known": torch.ones(1, players, 13, 13, 13, dtype=torch.bool),
        "voxel_relative_xyz": relative,
        "initial_pose": pose,
        "initial_hp": torch.full((1, players), 20.0),
        "held_item": torch.arange(players)[None] % 10,
        "resident_type": torch.arange(players)[None] % 4,
        "camera_relative": eye,
        "camera_direction": 0.1 * direction_from_angles(pose[..., 3:5]),
        "actions": actions,
        "hotbar": torch.arange(9)[None, None].expand(1, players, -1).clone(),
        "selected_slot": torch.zeros(1, players, dtype=torch.long),
        "event_context_pose": history,
        "event_context_actions": history_actions,
        "event_context_hp": torch.full((1, 4, players), 20.0),
        "event_context_held_item": (torch.arange(players)[None, None] % 10).expand(1, 4, -1).clone(),
        "event_context_camera_relative": eye[:, None].expand(-1, 4, -1, -1).clone(),
        "event_context_camera_direction": 0.1 * direction_from_angles(history[..., 3:5]),
        "event_context_valid": torch.ones(1, 4, players, dtype=torch.bool),
    }


def model():
    torch.manual_seed(7)
    return LocalWorldWriterV2(LocalWorldWriterV2Args(
        num_block_classes=7, num_items=10, air_class=0,
        width=32, heads=4, depth=2,
    )).eval()


def velocity_model():
    torch.manual_seed(7)
    return LocalWorldWriterV2(LocalWorldWriterV2Args(
        num_block_classes=7, num_items=10, air_class=0,
        width=32, heads=4, depth=2, position_prior="action",
        velocity_mode="conditioned_integrated", distance_per_step=0.1,
    )).eval()


def test_saved_config_infers_historical_position_prior_semantics():
    values = asdict(LocalWorldWriterV2Args(
        num_block_classes=7, num_items=10, air_class=0, width=32, heads=4, depth=2,
    ))
    direct = dict(values)
    direct.pop("position_prior")
    direct.pop("distance_per_step")
    action = dict(values)
    action.pop("position_prior")
    velocity = dict(values)
    assert build_m2_from_config({"schema": "m2-local-world-writer-v2", "model": direct}).cfg.position_prior == "direct"
    assert build_m2_from_config({"schema": "m2-local-world-writer-v2", "model": action}).cfg.position_prior == "action"
    assert build_m2_from_config({"schema": "m2-local-world-writer-v2", "model": velocity}).cfg.position_prior == "observed_velocity"


def clone_inputs(inputs):
    return {key: value.clone() for key, value in inputs.items()}


def permute_players(inputs, permutation):
    return {key: value.index_select(2 if key == "actions" or key.startswith("event_context_") else 1, permutation)
            for key, value in inputs.items()}


def assert_valid_outputs(output):
    for key, value in output.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        assert not torch.isnan(value).any(), key
        assert not torch.isposinf(value).any(), key
        if key != "event_address_logits":
            assert torch.isfinite(value).all(), key


def test_three_branch_shapes_and_sparse_decode_interface():
    net = model()
    inputs = synthetic_inputs()
    with torch.no_grad():
        out = net(inputs)
        decoded = net.decode_interactions(out)
    assert out["state_tokens"].shape == (1, 13, 2, 32)
    assert out["action_tokens"].shape == (1, 12, 2, 32)
    assert out["world_memory"].shape == (2, 2199, 32)
    assert out["sequence_hidden"].shape == (1, 13, 2, 32)
    assert out["hidden"].shape == (1, 8, 2, 32)
    assert out["pose"].shape == (1, 8, 2, 5)
    assert out["candidates"].shape == (1, 2, 2200, 32)
    assert out["event_address_logits"].shape == (1, 2, 8, 2200)
    assert out["event_time_logits"].shape == (1, 2, 8, 8)
    assert out["held_logits"].shape == (1, 8, 2, 10)
    assert out["state_residual"].shape == (1, 8, 2, 6)
    for key in ("address", "block_payload", "hp_payload", "held_item"):
        assert decoded[key].shape == (1, 8, 2)
    assert_valid_outputs(out)
    assert torch.isneginf(out["event_address_logits"][0, 0, :, VOXELS]).all()
    assert torch.isneginf(out["event_address_logits"][0, 1, :, VOXELS + 1]).all()


def test_rgb_and_external_velocity_are_not_inputs_but_observed_velocity_prior_is_used():
    net = model()
    inputs = synthetic_inputs()
    other = clone_inputs(inputs)
    other["previous_rgb"] = torch.full((1, 2, 3, 32, 32), float("nan"))
    other["initial_velocity"] = torch.full((1, 2, 3), float("nan"))
    with torch.no_grad():
        first, second = net(inputs), net(other)
    torch.testing.assert_close(first["sequence_hidden"], second["sequence_hidden"], rtol=0, atol=0)
    torch.testing.assert_close(first["pose"], second["pose"], rtol=0, atol=0)
    assert not any(isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)) for layer in net.modules())
    assert not any("velocity_encoder" in name or "proposal_encoder" in name for name, _ in net.named_parameters())
    velocity = inputs["initial_pose"][..., :3] - inputs["event_context_pose"][:, -1, :, :3]
    steps = torch.arange(1, 9, dtype=velocity.dtype).view(1, 8, 1, 1)
    torch.testing.assert_close(first["proposal"][..., :3], steps * velocity[:, None])
    torch.testing.assert_close(first["proposal"][..., 3:5], first["body_angle_prior"])
    torch.testing.assert_close(
        first["pose"][..., :3],
        inputs["initial_pose"][:, None, :, :3] + first["proposal"][..., :3],
    )


def test_actions_condition_residual_stream_but_not_velocity_proposal_xyz():
    net = model()
    inputs = synthetic_inputs()
    other = clone_inputs(inputs)
    other["actions"][..., :21].zero_()
    with torch.no_grad():
        first, second = net(inputs), net(other)
    torch.testing.assert_close(first["proposal"][..., :3], second["proposal"][..., :3])
    assert not torch.allclose(first["action_tokens"], second["action_tokens"])


def test_conditioned_velocity_is_integrated_exactly_from_action_prior():
    net = velocity_model()
    inputs = synthetic_inputs()
    with torch.no_grad():
        out = net(inputs)
    torch.testing.assert_close(out["velocity"], out["action_velocity_prior"])
    torch.testing.assert_close(out["pose"][..., :3],
                               inputs["initial_pose"][:, None, :, :3] + out["velocity"].cumsum(1))
    torch.testing.assert_close(out["pose"][..., :3],
                               inputs["initial_pose"][:, None, :, :3] + out["proposal"][..., :3])
    expected_input = inputs["initial_pose"][..., :3] - inputs["event_context_pose"][:, -1, :, :3]
    torch.testing.assert_close(out["observed_velocity_input"], expected_input)


def test_velocity_residual_accumulates_and_velocity_adapter_receives_gradient():
    net = velocity_model()
    inputs = synthetic_inputs()
    with torch.no_grad():
        net.state_residual_head.bias[:3].copy_(torch.tensor([0.25, -0.5, 0.75]))
    out = net(inputs)
    expected_residual = 0.2 * torch.tensor([0.25, -0.5, 0.75]).tanh()
    torch.testing.assert_close(out["velocity_residual"], expected_residual.expand_as(out["velocity_residual"]))
    steps = torch.arange(1, 9, dtype=out["pose"].dtype).view(1, 8, 1, 1)
    expected_position = (inputs["initial_pose"][:, None, :, :3] + out["proposal"][..., :3]
                         + steps * expected_residual)
    torch.testing.assert_close(out["pose"][..., :3], expected_position)
    out["hidden"].square().mean().backward()
    assert net.velocity_encoder.weight.grad is not None
    assert net.velocity_encoder.weight.grad.abs().sum() > 0


def test_formula_mouse_alignment_pitch_clamp_and_camera_lag():
    net = model()
    inputs = synthetic_inputs()
    inputs["initial_pose"][..., 3] = 4 * math.pi
    inputs["initial_pose"][..., 4] = math.pi / 2 - 0.02
    inputs["actions"].zero_()
    inputs["event_context_actions"].zero_()
    inputs["event_context_actions"][:, -1, :, 21:23] = torch.tensor([0.25, 0.5])
    inputs["actions"][:, 0, :, 21:23] = torch.tensor([-0.5, -0.5])
    inputs["actions"][:, -1, :, 21:23] = torch.tensor([0.9, 0.9])
    with torch.no_grad():
        out = net(inputs)
    prior = out["body_angle_prior"]
    torch.testing.assert_close(prior[:, 0, :, 0], torch.full((1, 2), 4 * math.pi + math.radians(32)))
    torch.testing.assert_close(prior[:, 1:, :, 0], torch.full((1, 7, 2), 4 * math.pi - math.radians(32)))
    torch.testing.assert_close(prior[:, 0, :, 1], torch.full((1, 2), math.pi / 2))
    torch.testing.assert_close(prior[:, 1:, :, 1], torch.full((1, 7, 2), math.pi / 2 - math.radians(36)))
    torch.testing.assert_close(out["pose"][..., 3:5], prior)
    previous_pose = torch.cat((inputs["initial_pose"][:, None], out["pose"][:, :-1]), 1)
    expected_camera = previous_pose[..., :3] + torch.tensor([0.0, 0.0, 1.6])
    torch.testing.assert_close(out["pose"][..., :3] + out["camera_relative"], expected_camera)
    torch.testing.assert_close(out["camera_direction"], 0.1 * direction_from_angles(previous_pose[..., 3:5]))
    torch.testing.assert_close(out["camera_direction"].norm(dim=-1), torch.full((1, 8, 2), 0.1))


def test_formula_two_blocks_match_a_single_sixteen_step_sequence():
    inputs = synthetic_inputs(steps=16)
    inputs["actions"][..., 21:23] = torch.linspace(-0.15, 0.2, 64).reshape(1, 16, 2, 2)
    complete = recorded_body_angle_trajectory(inputs)
    first_inputs = clone_inputs(inputs)
    first_inputs["actions"] = inputs["actions"][:, :8]
    first = recorded_body_angle_trajectory(first_inputs)
    next_inputs = clone_inputs(inputs)
    next_inputs["initial_pose"][..., 3:5] = first[:, -1]
    next_inputs["event_context_actions"] = inputs["actions"][:, 4:8].clone()
    next_inputs["actions"] = inputs["actions"][:, 8:]
    second = recorded_body_angle_trajectory(next_inputs)
    torch.testing.assert_close(torch.cat((first, second), 1), complete, rtol=1e-5, atol=1e-6)


def test_learned_residuals_preserve_previous_body_camera_alignment():
    net = model()
    inputs = synthetic_inputs()
    # Set an interpretable test fixture, not an optimizer update.
    with torch.no_grad():
        net.state_residual_head.bias.copy_(torch.tensor([0.2, -0.3, 0.4, 0.5, -0.5, 0.7]))
        out = net(inputs)
    expected_position = (inputs["initial_pose"][:, None, :, :3]
                         + out["proposal"][..., :3]
                         + torch.tensor([0.2, -0.3, 0.4]))
    torch.testing.assert_close(out["pose"][..., :3], expected_position)
    angle_delta = math.radians(2) * torch.tensor([0.5, -0.5]).tanh()
    torch.testing.assert_close(out["pose"][..., 3:5], out["body_angle_prior"] + angle_delta)
    previous = torch.cat((inputs["initial_pose"][:, None], out["pose"][:, :-1]), 1)
    expected_eye = 1.6 + 0.15 * math.tanh(0.7)
    torch.testing.assert_close(out["pose"][..., :3] + out["camera_relative"],
                               previous[..., :3] + torch.tensor([0., 0., expected_eye]))
    torch.testing.assert_close(out["camera_direction"], 0.1 * direction_from_angles(previous[..., 3:5]))


def test_known_hotbar_changes_action_conditioning_without_future_item_labels():
    net = model()
    inputs = synthetic_inputs()
    other = clone_inputs(inputs)
    other["hotbar"][:, :, 0] = 6
    with torch.no_grad():
        first, second = net(inputs), net(other)
    assert not torch.allclose(first["action_tokens"][:, 4:], second["action_tokens"][:, 4:])
    assert not torch.allclose(first["hidden"], second["hidden"])
    torch.testing.assert_close(first["state_tokens"], second["state_tokens"], rtol=0, atol=0)


def test_both_players_share_weights_and_world_coordinates_are_translation_equivariant():
    net = model()
    inputs = synthetic_inputs()
    permutation = torch.tensor([1, 0])
    other = permute_players(inputs, permutation)
    shifted = clone_inputs(inputs)
    shift = torch.tensor([73.0, -41.0, 12.0])
    shifted["initial_pose"][..., :3] += shift
    shifted["event_context_pose"][..., :3] += shift
    with torch.no_grad():
        original, swapped, translated = net(inputs), net(other), net(shifted)
    torch.testing.assert_close(swapped["sequence_hidden"], original["sequence_hidden"][:, :, permutation], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(swapped["pose"], original["pose"][:, :, permutation], atol=1e-5, rtol=1e-5)
    columns = torch.cat((torch.arange(VOXELS), VOXELS + permutation, torch.tensor([VOXELS + 2])))
    torch.testing.assert_close(swapped["event_address_logits"], original["event_address_logits"][:, permutation][:, :, :, columns], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(translated["hidden"], original["hidden"], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(translated["pose"][..., :3], original["pose"][..., :3] + shift, atol=1e-5, rtol=1e-5)


def test_padding_is_finite_and_has_no_effect_on_real_players():
    net = model()
    inputs = synthetic_inputs()
    inputs["active"][:, 1] = False
    inputs["event_context_valid"][:, :, 1] = False
    inputs["event_context_valid"][:, :2, 0] = False
    other = clone_inputs(inputs)
    for key in ("event_context_pose", "event_context_actions", "event_context_hp", "event_context_camera_relative", "event_context_camera_direction"):
        other[key][:, :2, 0] = 1000
        other[key][:, :, 1] = -1000
    other["initial_pose"][:, 1] = 500
    other["camera_relative"][:, 1] = -400
    other["camera_direction"][:, 1] = 200
    other["actions"][:, :, 1] = 300
    with torch.no_grad():
        first, second = net(inputs), net(other)
    assert_valid_outputs(first)
    assert_valid_outputs(second)
    torch.testing.assert_close(first["hidden"][:, :, 0], second["hidden"][:, :, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first["event_address_logits"][:, 0], second["event_address_logits"][:, 0], atol=1e-5, rtol=1e-5)


def test_missing_final_history_action_or_unknown_active_crop_is_rejected():
    net = model()
    inputs = synthetic_inputs()
    inputs["event_context_valid"][:, -1, 0] = False
    with TestCase().assertRaises(ValueError):
        net(inputs)
    inputs = synthetic_inputs()
    inputs["voxel_known"][0, 0, 1, 2, 3] = False
    with TestCase().assertRaises(ValueError):
        net(inputs)


def test_bidirectional_action_path_and_gradients_without_parameter_updates():
    net = model()
    inputs = synthetic_inputs()
    changed = clone_inputs(inputs)
    changed["actions"][:, -1, :, 8] = 1
    with torch.no_grad():
        first = net(inputs)
        second = net(changed)
    torch.testing.assert_close(first["state_tokens"], second["state_tokens"], rtol=0, atol=0)
    assert not torch.allclose(first["hidden"][:, 0], second["hidden"][:, 0], rtol=1e-6, atol=1e-7)
    before = {name: parameter.detach().clone() for name, parameter in net.named_parameters()}
    output = net(inputs)
    objective = output["held_logits"].square().mean() + output["event_operation_logits"].square().mean()
    objective.backward()
    assert net.action_encoder.weight.grad is not None
    assert net.action_encoder.weight.grad.abs().sum() > 0
    assert net.action_gate_logits.grad is not None
    assert net.action_gate_logits.grad.abs().sum() > 0
    for block in net.blocks:
        assert block.action_attention.in_proj_weight.grad is not None
        assert block.action_attention.in_proj_weight.grad.abs().sum() > 0
    for name, parameter in net.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name], rtol=0, atol=0)


def test_v1_warm_start_is_semantic_and_round_trips_strictly():
    from miworld.models.m2_v2_migration import migrate_v1_to_v2
    from miworld.models.local_world_writer import build_m2_from_config

    torch.manual_seed(12)
    old_cfg = LocalWorldWriterArgs(num_block_classes=7, num_items=10, air_class=0,
                                   width=32, heads=4, depth=2, event_context_frames=4)
    old = LocalWorldWriter(old_cfg)
    with torch.no_grad():
        old.camera_decoder.pose_head.weight.fill_(0.7)
        old.camera_decoder.camera_head.weight.fill_(0.4)
    net = model()
    migrated, report = migrate_v1_to_v2(old.state_dict(), {"schema": "m2-local-world-writer-v1", "model": asdict(old_cfg)}, net)
    net.load_state_dict(migrated, strict=True)
    assert isinstance(report, dict)
    torch.testing.assert_close(net.state_encoder[0].weight[:, :59], old.state_encoder[0].weight)
    assert torch.count_nonzero(net.state_encoder[0].weight[:, 59:]) == 0
    torch.testing.assert_close(net.memory_encoder.block_embedding.weight, old.memory_encoder.block_embedding.weight[:7])
    torch.testing.assert_close(net.blocks[0].temporal_attention.in_proj_weight, old.blocks[0].temporal_attention.in_proj_weight)
    assert torch.count_nonzero(net.state_residual_head.weight) == 0
    config = {"schema": "m2-local-world-writer-v2", "model": asdict(net.cfg)}
    with tempfile.TemporaryDirectory(prefix="m2-v2-contract-") as temporary:
        path = Path(temporary) / "init-only.pt"
        torch.save({"model": net.state_dict(), "config": config, "step": 0, "init_only": True}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
    reloaded = build_m2_from_config(saved["config"]).eval()
    reloaded.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        expected, actual = net(synthetic_inputs()), reloaded(synthetic_inputs())
    torch.testing.assert_close(expected["hidden"], actual["hidden"], rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args()
    results = []
    for name, check in sorted(list(globals().items())):
        if name.startswith("test_") and callable(check):
            try:
                check()
                results.append({"test": name, "status": "PASS"})
                print("PASS", name, flush=True)
            except Exception:
                error = traceback.format_exc()
                results.append({"test": name, "status": "FAIL", "error": error})
                print("FAIL", name, error, flush=True)
    report = {"status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
              "optimizer_steps": 0, "tests": results}
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "PASS" else 1)
