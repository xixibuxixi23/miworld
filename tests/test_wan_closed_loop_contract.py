import torch

from miworld.data.transition_dataset import VOXEL_COUNT, collate_transition
from miworld.models.local_world_writer import (
    LocalWorldWriter,
    LocalWorldWriterArgs,
    adapt_m2_timestep_state,
)
from miworld.pipelines.wan_closed_loop_contract import WanClosedLoopContract


class FakeWanCodec:
    def decode(self, latent):
        frames = 1 + (latent.shape[1] - 1) * 4
        return torch.zeros(latent.shape[0], frames, 3, 2, 2)


def test_13_pixel_frame_contract_is_12_transitions_and_3_generated_latents():
    contract = WanClosedLoopContract()
    assert contract.latent_pixel_indices == (0, 4, 8, 12)
    assert contract.generated_pixel_indices == (4, 8, 12)
    actions = torch.zeros(2, 12, 3, 23)
    assert contract.validate_actions(actions) is actions
    boundary = torch.zeros(2, 1, 16, 2, 2)
    generated = torch.zeros(2, 3, 16, 2, 2)
    assert contract.decode_block(FakeWanCodec(), boundary, generated).shape == (
        2, 12, 3, 2, 2
    )


def _m2(timesteps):
    return LocalWorldWriter(LocalWorldWriterArgs(
        num_block_classes=3,
        num_items=10,
        air_class=0,
        width=8,
        depth=1,
        heads=2,
        timesteps=timesteps,
        event_queries=4,
        use_rgb=False,
    ))


def test_m2_horizon_warm_start_preserves_old_rows_and_initializes_12_steps():
    torch.manual_seed(3)
    old = _m2(8)
    new = _m2(12)
    state, resized = adapt_m2_timestep_state(old.state_dict(), new)
    new.load_state_dict(state, strict=True)
    assert set(resized) == {
        "time_embedding",
        "event_decoder.time_head.weight",
        "event_decoder.time_head.bias",
    }
    torch.testing.assert_close(new.time_embedding[:8], old.time_embedding)
    torch.testing.assert_close(
        new.event_decoder.time_head.weight[:8], old.event_decoder.time_head.weight
    )
    assert new.time_embedding.shape == (12, 8)
    assert new.event_decoder.time_head.out_features == 12


def test_m2_12_step_forward_keeps_player_and_npc_action_slots():
    model = _m2(12).eval()
    batch, residents, steps = 1, 2, 12
    actions = torch.zeros(batch, steps, residents, 23)
    actions[:, :, 0, 0] = 1  # player control input
    actions[:, :, 1, 1] = 1  # NPC GT action input in v1
    inputs = {
        "active": torch.ones(batch, residents, dtype=torch.bool),
        "voxels": torch.zeros(batch, residents, 13, 13, 13, dtype=torch.long),
        "voxel_known": torch.ones(
            batch, residents, 13, 13, 13, dtype=torch.bool
        ),
        "voxel_relative_xyz": torch.zeros(batch, residents, 13**3, 3),
        "initial_pose": torch.zeros(batch, residents, 5),
        "initial_hp": torch.full((batch, residents), 20.0),
        "held_item": torch.zeros(batch, residents, dtype=torch.long),
        "resident_type": torch.tensor([[0, 2]]),
        "camera_relative": torch.zeros(batch, residents, 3),
        "camera_direction": torch.tensor([0.0, 1.0, 0.0]).expand(
            batch, residents, 3
        ).clone(),
        "actions": actions,
        "hotbar": torch.zeros(batch, residents, 9, dtype=torch.long),
        "selected_slot": torch.zeros(batch, residents, dtype=torch.long),
    }
    with torch.no_grad():
        output = model(inputs)
    assert output["pose"].shape == (batch, steps, residents, 5)
    assert output["event_time_logits"].shape == (batch, residents, 4, steps)
    assert not torch.equal(actions[:, :, 0], actions[:, :, 1])


def test_event_history_padding_uses_resident_axis():
    def sample(residents):
        return {
            "inputs": {
                "active": torch.ones(residents, dtype=torch.bool),
                "actions": torch.zeros(12, residents, 23),
                "event_context_pose": torch.zeros(4, residents, 5),
                "event_context_actions": torch.zeros(4, residents, 23),
                "event_context_valid": torch.ones(4, residents, dtype=torch.bool),
            },
            "targets": {
                "address": torch.full((12, residents), VOXEL_COUNT + residents),
            },
            "metadata": {},
        }

    batch = collate_transition([sample(2), sample(4)])
    assert batch["inputs"]["event_context_pose"].shape == (2, 4, 4, 5)
    assert batch["inputs"]["event_context_actions"].shape == (2, 4, 4, 23)
    assert batch["inputs"]["event_context_valid"].shape == (2, 4, 4)
    assert batch["inputs"]["actions"].shape == (2, 12, 4, 23)
