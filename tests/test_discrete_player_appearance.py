import numpy as np
import pytest
import torch

from miworld.data.renderer_dataset import resident_appearance_ids
from miworld.models.renderer import RendererArgs, SimpleResidentConditionEncoder


def test_manifest_maps_exactly_two_humans_and_three_npcs():
    manifest = {
        "agent_kinds": {
            "agent0": "human_like",
            "agent1": "human_like",
            "agent2": "npc_villager",
            "agent3": "npc_zombie",
            "agent4": "npc_skeleton",
        },
        "skin_catalog_ids": {
            "agent0": "human-default-v1-steve-wide",
            "agent1": "human-default-v1-alex-slim",
            "agent2": "npc_villager-v1",
            "agent3": "npc_zombie-v1",
            "agent4": "npc_skeleton-v1",
        },
    }
    np.testing.assert_array_equal(
        resident_appearance_ids(manifest, 5), np.arange(5, dtype=np.int64)
    )


def test_unknown_human_skin_fails_closed():
    manifest = {
        "agent_kinds": {"agent0": "human_like"},
        "skin_catalog_ids": {"agent0": "unseen-skin"},
    }
    with pytest.raises(ValueError, match="unsupported resident appearance"):
        resident_appearance_ids(manifest, 1)


def test_explicit_row_local_skin_alias_preserves_frozen_body_type_ids():
    manifest = {
        "agent_kinds": {"agent0": "human_like", "agent1": "human_like"},
        "skin_catalog_ids": {
            "agent0": "human-default-v1-zuri-wide",
            "agent1": "human-default-v1-zuri-slim",
        },
    }
    aliases = {
        "human-default-v1-zuri-wide": "human-default-v1-steve-wide",
        "human-default-v1-zuri-slim": "human-default-v1-alex-slim",
    }
    np.testing.assert_array_equal(
        resident_appearance_ids(manifest, 2, aliases),
        np.array([0, 1], dtype=np.int64),
    )


def _conditions(appearance_ids):
    batch, frames, agents = 1, 1, len(appearance_ids)
    return {
        "player_position": torch.zeros(batch, frames, agents, 3),
        "camera_relative": torch.zeros(batch, frames, agents, 3),
        "hp": torch.full((batch, frames, agents), 20.0),
        "yaw_pitch": torch.zeros(batch, frames, agents, 2),
        "event_cues": torch.zeros(batch, frames, agents, 4),
        "held_item": torch.zeros(batch, frames, agents, dtype=torch.long),
        "resident_type": torch.zeros(batch, frames, agents, dtype=torch.long),
        "action": torch.zeros(batch, frames, agents, 23),
        "player_valid": torch.ones(batch, frames, agents, dtype=torch.bool),
        "player_appearance_id": torch.tensor(appearance_ids)[None, None],
    }


def test_discrete_appearance_separates_identical_human_state_and_gets_gradient():
    torch.manual_seed(7)
    cfg = RendererArgs(
        num_block_classes=4,
        item_vocab_size=2,
        actor_channels=32,
        simple_conditioning=True,
        unified_player_reference=True,
        discrete_player_appearance=True,
    )
    encoder = SimpleResidentConditionEncoder(cfg)
    _, different = encoder(_conditions([0, 1]))
    assert not torch.equal(different[:, :, 0], different[:, :, 1])

    _, same = encoder(_conditions([0, 0]))
    torch.testing.assert_close(same[:, :, 0], same[:, :, 1])

    different.sum().backward()
    gradient = encoder.appearance_embedder.weight.grad
    assert gradient[0].abs().sum() > 0
    assert gradient[1].abs().sum() > 0
    assert gradient[2:].abs().sum() == 0


def test_split_conditioning_keeps_identity_out_of_action_state():
    torch.manual_seed(11)
    cfg = RendererArgs(
        num_block_classes=4,
        item_vocab_size=2,
        actor_channels=32,
        simple_conditioning=True,
        unified_player_reference=True,
        fixed_spatial_reference=True,
        player_reference_grid_size=(16, 8),
        player_reference_position_encoding=True,
        discrete_player_appearance=True,
        disable_actor_spatial_condition=True,
        split_player_conditioning=True,
        other_player_condition_blocks=(0,),
    )
    encoder = SimpleResidentConditionEncoder(cfg)
    _, actor = encoder(_conditions([0, 1]))
    torch.testing.assert_close(actor[:, :, 0], actor[:, :, 1])

    appearance = encoder.appearance_features(_conditions([0, 1]))
    assert not torch.equal(appearance[:, :, 0], appearance[:, :, 1])
    appearance.sum().backward()
    assert encoder.appearance_embedder.weight.grad[:2].abs().sum() > 0
