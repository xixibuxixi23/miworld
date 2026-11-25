from dataclasses import asdict
import json

import torch

from miworld.models.local_world_writer import build_m2_from_config
from miworld.models.local_world_writer_v2 import LocalWorldWriterV2, LocalWorldWriterV2Args
from miworld.models.local_world_writer_engine_prior import (
    LocalWorldWriterEnginePrior,
    LocalWorldWriterEnginePriorXYZ,
)
from miworld.training.m2_v2_rollout import next_carry, rollout_inputs
from diagnostics.migrate_m2_v2_engine_prior import migrate
from test_local_world_writer_v2 import synthetic_inputs


def setup():
    cfg = LocalWorldWriterV2Args(num_block_classes=7, num_items=10, air_class=0,
                               width=32, depth=2, heads=4, position_prior="action",
                               velocity_mode="conditioned_integrated")
    inputs = synthetic_inputs(players=1)
    inputs["event_context_pose"][:] = inputs["initial_pose"][:, None]
    inputs["actions"].zero_()
    inputs["event_context_actions"].zero_()
    inputs["actions"][..., 0] = 1
    return cfg, inputs


def test_exact_parameter_contract_and_original_residual_semantics():
    cfg, inputs = setup()
    original = LocalWorldWriterV2(cfg)
    net = LocalWorldWriterEnginePrior(cfg)
    net.load_state_dict(original.state_dict(), strict=True)
    assert {k: v.shape for k, v in net.state_dict().items()} == {
        k: v.shape for k, v in original.state_dict().items()}
    with torch.no_grad():
        net.state_residual_head.bias[:3].fill_(.2)
        out = net(inputs)
    residual = .2 * torch.tensor(.2).tanh()
    torch.testing.assert_close(out["velocity_residual"], torch.full_like(out["velocity"], residual))
    torch.testing.assert_close(out["pose"][..., :3], inputs["initial_pose"][:, None, :, :3]
                               + out["proposal"][..., :3] + out["velocity_residual"].cumsum(1))
    assert "_engine_prior_speeds" not in inputs
    copy = build_m2_from_config({"schema": "m2-local-world-writer-v2-engine-prior", "model": asdict(cfg)})
    copy.load_state_dict(net.state_dict(), strict=True)


def test_action_lag_startup_and_128_frame_cross_chunk_continuity():
    cfg, observed = setup()
    net = LocalWorldWriterEnginePrior(cfg).eval()
    carry, all_delta = None, []
    with torch.no_grad():
        for _ in range(16):
            inputs = rollout_inputs(observed, carry)
            output = net(inputs)
            all_delta.append(output["velocity"])
            carry = next_carry(output, inputs)
    delta = torch.cat(all_delta, 1)[0, :, 0]
    torch.testing.assert_close(delta[:5, 1], torch.tensor([0., .03125, 1/12, .1354166667, 1/6]))
    torch.testing.assert_close(delta[5:, 1], torch.full_like(delta[5:, 1], 1/6), atol=1e-6, rtol=1e-5)
    assert delta[:, (0, 2)].abs().max() == 0
    torch.testing.assert_close(carry["engine_prior_velocity"], torch.tensor([[[0., 4.]]]))
    assert "motion_hidden" not in carry and "engine_velocity" not in carry


def test_xyz_prior_reproduces_voxelibre_standard_jump_and_flat_landing():
    cfg, inputs = setup()
    cfg = LocalWorldWriterV2Args(**{
        **asdict(cfg), "velocity_residual_limit": .03,
        "velocity_residual_penalty": True,
    })
    inputs["event_context_actions"][..., 4] = 1
    inputs["actions"][..., 4] = 1
    net = LocalWorldWriterEnginePriorXYZ(cfg).eval()

    with torch.no_grad():
        first = net(inputs)
    frames = torch.arange(1, 9, dtype=first["proposal"].dtype) / 24.
    expected = 6.6 * frames - .5 * 20.8 * frames.square()
    torch.testing.assert_close(first["proposal"][0, :, 0, 2], expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(first["proposal"][0, -1, 0, 2], torch.tensor(1.0444444))
    assert first["engine_prior_jump_step"][0, -1, 0] == 8

    carry = next_carry(first, inputs)
    second_inputs = rollout_inputs(inputs, carry)
    with torch.no_grad():
        second = net(second_inputs)
    torch.testing.assert_close(second["pose"][0, -1, 0, 2], inputs["initial_pose"][0, 0, 2],
                               atol=2e-6, rtol=1e-5)
    assert second["engine_prior_jump_step"][0, -1, 0] == 0
    assert set(("engine_prior_velocity", "engine_prior_jump_step",
                "engine_prior_jump_origin_z")) <= carry.keys()


def test_xyz_config_keeps_parameter_contract_and_uses_small_residual():
    with open("configs/models/m2_engine_prior_xyz_v3.json") as handle:
        config = json.load(handle)
    net = build_m2_from_config(config)
    legacy_cfg = dict(config)
    legacy_cfg["schema"] = "m2-local-world-writer-v2-engine-prior"
    legacy = build_m2_from_config(legacy_cfg)
    assert {name: value.shape for name, value in net.state_dict().items()} == {
        name: value.shape for name, value in legacy.state_dict().items()
    }
    assert net.cfg.velocity_residual_limit == .03
    assert net.cfg.velocity_residual_penalty


def test_full_config_has_exactly_original_parameter_count():
    with open("configs/models/m2_engine_prior_v2.json") as handle:
        config = json.load(handle)
    net = build_m2_from_config(config)
    assert sum(p.numel() for p in net.parameters()) == 12379814
    assert net.cfg.distance_per_step == 0.08555694974465658


def test_migration_preserves_everything_except_xyz_head_and_drops_optimizer():
    cfg, _ = setup()
    old = LocalWorldWriterV2(cfg)
    with torch.no_grad():
        old.state_residual_head.weight.fill_(.7)
        old.state_residual_head.bias.fill_(.4)
        old.velocity_encoder.weight.fill_(.3)
    source = {"config": {"schema": "m2-local-world-writer-v2", "model": asdict(cfg)},
              "model": old.state_dict(), "optimizer": {"old": True}, "step": 2000}
    state, config, metadata = migrate(source)
    assert config["schema"] == "m2-local-world-writer-v2-engine-prior"
    assert metadata["source_step"] == 2000
    assert torch.count_nonzero(state["state_residual_head.weight"][:3]) == 0
    assert torch.count_nonzero(state["state_residual_head.bias"][:3]) == 0
    torch.testing.assert_close(state["state_residual_head.weight"][3:], source["model"]["state_residual_head.weight"][3:])
    torch.testing.assert_close(state["velocity_encoder.weight"], source["model"]["velocity_encoder.weight"])
    for name in state:
        if name not in {"state_residual_head.weight", "state_residual_head.bias"}:
            torch.testing.assert_close(state[name], source["model"][name])
