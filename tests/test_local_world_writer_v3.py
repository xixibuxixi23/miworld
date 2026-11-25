"""Engine numerics, chunk boundaries, geometry gradients and full training smoke."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import socket
import sys

import pytest
import torch

from miworld.models.local_world_writer import build_m2_from_config
from miworld.models.local_world_writer_v3 import (
    LocalWorldWriterV3, LocalWorldWriterV3Args, engine_horizontal_step,
)
from miworld.training.m2_v2_rollout import next_carry, rollout_inputs, supervised_velocity_loss
from test_local_world_writer_v2 import synthetic_inputs


def make_model():
    torch.manual_seed(91)
    return LocalWorldWriterV3(LocalWorldWriterV3Args(
        num_block_classes=7, num_items=10, air_class=0,
        width=32, heads=4, depth=2, motion_width=32, motion_heads=4,
        local_voxels=16, event_decoder_layers=1,
    )).eval()


def stationary_inputs():
    inputs = synthetic_inputs(players=1)
    inputs["event_context_pose"][:] = inputs["initial_pose"][:, None]
    inputs["event_context_actions"].zero_()
    inputs["actions"].zero_()
    inputs["voxels"].zero_()
    return inputs


def test_engine_start_stop():
    speed = torch.zeros(1, 2)
    target = torch.tensor([[0., 4.]])
    distances, speeds = [], []
    for _ in range(4):
        delta, speed = engine_horizontal_step(speed, target)
        distances.append(float(delta[0, 1]))
        speeds.append(float(speed[0, 1]))
    assert distances == pytest.approx([.03125, .083333333, .135416667, .166666667])
    assert speeds == pytest.approx([1.25, 2.5, 3.75, 4.])
    for _ in range(4):
        delta, speed = engine_horizontal_step(speed, torch.zeros_like(target))
    assert torch.count_nonzero(speed) == 0
    delta, speed = engine_horizontal_step(speed, torch.zeros_like(target))
    assert torch.count_nonzero(delta) == 0


def test_wall_cancels_carried_speed_and_allows_tangential_sliding():
    velocity = torch.tensor([[3., 2.]])
    delta, final = engine_horizontal_step(velocity, velocity, transmission=torch.tensor([[0., 1.]]))
    assert delta[0, 0] == final[0, 0] == 0
    assert delta[0, 1] > 0


def test_initial_model_exactly_preserves_formula_and_action_lag():
    model, inputs = make_model(), stationary_inputs()
    inputs["actions"][..., 0] = 1
    with torch.no_grad():
        out = model(inputs)
    assert out["velocity"][0, 0].abs().max() == 0  # previous action was no-op
    assert out["velocity"][0, 1, 0, 1] == pytest.approx(.03125)
    assert out["velocity"][0, 4, 0, 1] == pytest.approx(1 / 6)
    torch.testing.assert_close(out["velocity_residual"], torch.zeros_like(out["velocity"]))
    modified = deepcopy(inputs)
    modified["actions"][:, -1, :, :4] = torch.tensor([0., 1., 1., 0.])
    with torch.no_grad():
        torch.testing.assert_close(model(modified)["pose"], out["pose"])


def test_formula_128_frames_equals_16_chunks_without_velocity_reset():
    model, first = make_model(), stationary_inputs()
    first["actions"][..., 0] = 1
    first["event_context_actions"][:, -1, :, 0] = 1
    carry, positions = None, []
    with torch.no_grad():
        for _ in range(16):
            inputs = rollout_inputs(first, carry)
            out = model(inputs)
            positions.append(out["pose"][..., :2])
            carry = next_carry(out, inputs)
    speed = torch.zeros(1, 1, 2)
    position = first["initial_pose"][..., :2].clone()
    expected = []
    for _ in range(128):
        delta, speed = engine_horizontal_step(speed, torch.tensor([[[0., 4.]]]))
        position = position + delta
        expected.append(position)
    torch.testing.assert_close(torch.cat(positions, 1), torch.stack(expected, 1), atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(carry["engine_velocity"][..., :2], speed)


def test_contact_head_can_stop_motion_instead_of_leaving_hidden_inertia():
    model, inputs = make_model(), stationary_inputs()
    inputs["actions"][..., 0] = 1
    inputs["event_context_actions"][..., 0] = 1
    inputs["engine_velocity"] = torch.tensor([[[0., 4., 0.]]])
    with torch.no_grad():
        model.motion_decoder.head[-1].bias[:2].fill_(-1.)
        out = model(inputs)
    assert torch.count_nonzero(out["velocity"]) == 0
    assert torch.count_nonzero(out["engine_velocity"]) == 0


def test_local_geometry_changes_motion_and_inactive_padding_cannot_poison_it():
    model = make_model()
    inputs = synthetic_inputs(players=2)
    with torch.no_grad():
        model.motion_decoder.head[-1].weight.normal_(std=.003)
        first = model(inputs)
        other = deepcopy(inputs)
        other["voxels"].zero_()
        second = model(other)
    assert not torch.allclose(first["velocity"], second["velocity"])
    inputs["active"][:, 1] = False
    with torch.no_grad():
        clean = model(inputs)
        inputs["initial_pose"][:, 1] = float("nan")
        inputs["event_context_pose"][:, :, 1] = float("nan")
        inputs["voxels"][:, 1] = -99
        inputs["voxel_relative_xyz"][:, 1] = float("nan")
        poisoned = model(inputs)
    torch.testing.assert_close(poisoned["pose"][:, :, 0], clean["pose"][:, :, 0])
    assert torch.isfinite(poisoned["pose"]).all()


def test_geometry_recurrence_and_state_receive_finite_gradients():
    model, inputs = make_model(), stationary_inputs()
    with torch.no_grad():
        model.motion_decoder.head[-1].weight.normal_(std=.002)
    out = model(inputs)
    out["engine_velocity"].retain_grad()
    carry = next_carry(out, inputs, detach=False)
    second = model(rollout_inputs(inputs, carry))
    second["pose"][..., :3].square().mean().backward()
    assert out["engine_velocity"].grad is not None
    for parameter in (model.motion_decoder.blocks.weight, model.motion_decoder.offset[0].weight,
                      model.motion_decoder.cell.weight_hh, model.blocks[0].action_attention.in_proj_weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_checkpoint_reload_and_translation_equivariance():
    model, inputs = make_model(), stationary_inputs()
    copy = build_m2_from_config({"schema": "m2-local-world-writer-v3", "model": asdict(model.cfg)})
    copy.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        model.motion_decoder.head[-1].weight.normal_(std=.002)
        reference = model(inputs)
        shifted = deepcopy(inputs)
        offset = torch.tensor([20., -30., 10.])
        shifted["initial_pose"][..., :3] += offset
        shifted["event_context_pose"][..., :3] += offset
        shifted_out = model(shifted)
    torch.testing.assert_close(shifted_out["pose"][..., :3], reference["pose"][..., :3] + offset)
    torch.testing.assert_close(shifted_out["engine_velocity"], reference["engine_velocity"], atol=2e-5, rtol=2e-4)


def test_velocity_labels_exclude_accumulated_boundary_error_and_invalid_predecessor():
    initial = torch.zeros(1, 1, 5)
    target = torch.zeros(1, 3, 1, 5)
    target[0, :, 0, 0] = torch.tensor([1., 2., 3.])
    out = {"velocity": torch.tensor([[[[1., 0., 0.]], [[999., 0., 0.]], [[999., 0., 0.]]]])}
    inputs = {"initial_pose": initial + 42, "_recorded_initial_pose": initial,
              "active": torch.ones(1, 1, dtype=torch.bool)}
    targets = {"pose": target, "state_valid": torch.tensor([[[True], [False], [True]]])}
    loss, _ = supervised_velocity_loss(out, inputs, targets)
    assert loss == 0


def test_cpu_training_save_reload_and_full_state_resume(tmp_path):
    """Tiny synthetic execution check, not an accuracy experiment or dataset run."""
    model = make_model()
    config = {"schema": "m2-local-world-writer-v3", "model": asdict(model.cfg)}
    init = tmp_path / "initializer.pt"
    torch.save(dict(model=model.state_dict(), config=config, step=0, init_only=True), init)
    inputs = stationary_inputs()
    with torch.no_grad():
        out = model(inputs)
    targets = {
        "pose": out["pose"], "camera_relative": out["camera_relative"],
        "camera_direction": out["camera_direction"],
        "state_valid": torch.ones(1, 8, 1, dtype=torch.bool),
        "held_item": torch.zeros(1, 8, 1, dtype=torch.long),
        "address": torch.full((1, 8, 1), 13 ** 3 + 1),
        "address_valid": torch.ones(1, 8, 1, dtype=torch.bool),
        "block": torch.zeros(1, 8, 1, dtype=torch.long),
        "block_valid": torch.zeros(1, 8, 1, dtype=torch.bool),
        "damage": torch.zeros(1, 8, 1), "damage_valid": torch.zeros(1, 8, 1, dtype=torch.bool),
    }
    samples = [dict(metadata={"start": start},
                    inputs={k: v[0] for k, v in inputs.items()},
                    targets={k: v[0] for k, v in targets.items()}) for start in (0, 8)]
    for split in ("train", "val_id"):
        (tmp_path / "cache" / split).mkdir(parents=True)
        torch.save(dict(transition_cache_revision=4, timesteps=8, samples=samples),
                   tmp_path / "cache" / split / (split + ".pt"))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"splits": {s: [{"path": s}] for s in ("train", "val_id")}}))
    command = [sys.executable, "train_scripts/train_m2_tristream_v2.py", "--source-checkpoint", str(init),
               "--index", str(index), "--cache-root", str(tmp_path / "cache"), "--output-dir", str(tmp_path / "run"),
               "--device", "cpu", "--batch-size", "1", "--rollout-blocks", "2", "--bptt-blocks", "2",
               "--eval-blocks", "2", "--eval-batches", "1", "--warmup-steps", "0", "--save-every", "1",
               "--velocity-weight", "10", "--threads", "1"]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(command + ["--steps", "1"], cwd=root, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    latest = tmp_path / "run/latest.pt"
    result = subprocess.run(command + ["--steps", "2", "--resume", str(latest)], cwd=root,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(latest, weights_only=False)
    assert checkpoint["step"] == 2 and checkpoint["optimizer"]
    reloaded = build_m2_from_config(checkpoint["config"])
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    # Two-rank CPU DDP exercises repeated forward calls in one BPTT segment.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    distributed = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
                   "--master-addr", "127.0.0.1", "--master-port", str(port), "--nproc-per-node", "2"]
    ddp_command = command[1:].copy()
    ddp_command[ddp_command.index("--output-dir") + 1] = str(tmp_path / "ddp_run")
    result = subprocess.run(distributed + ddp_command + ["--steps", "2"], cwd=root,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(tmp_path / "ddp_run/latest.pt", weights_only=False)
    assert len(checkpoint["rng_by_rank"]) == 2
