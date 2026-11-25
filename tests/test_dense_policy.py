import numpy as np
import torch

from miworld.data.inserted_policy_dataset import (
    dense_action_targets,
    mask_targets_to_absolute_ranges,
)
from miworld.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from tests.test_renderer import tiny_model, conditions


def test_dense_targets_use_future_actions_and_stop_at_terminal():
    actions = np.arange(128 * 23, dtype=np.float32).reshape(128, 1, 23)
    data = {'action_continuous': actions, 'language_policy_train_mask': np.ones((128, 1), bool),
            'player_health': np.ones((129, 1)), 'player_health_valid': np.ones((129, 1), bool),
            'termination_flag': np.zeros(128, bool)}
    target, valid, indices = dense_action_targets(data, 'language_builder', 0, 0)
    assert target.shape == (9, 32, 23) and valid.sum() == 288
    assert indices[:, -1].tolist() == list(range(32, 97, 8))
    np.testing.assert_equal(target[0], actions[32:64, 0])
    np.testing.assert_equal(target[-1], actions[96:128, 0])
    data['termination_flag'][40] = True
    _, valid, _ = dense_action_targets(data, 'language_builder', 0, 0)
    assert valid[0, :8].all() and not valid[0, 8:].any()
    assert not valid[1].any()


def test_dense_sixteen_targets_train_every_step():
    actions = np.arange(128 * 23, dtype=np.float32).reshape(128, 1, 23)
    data = {'action_continuous': actions, 'language_policy_train_mask': np.ones((128, 1), bool),
            'player_health': np.ones((129, 1)), 'player_health_valid': np.ones((129, 1), bool),
            'termination_flag': np.zeros(128, bool)}
    target, valid, indices = dense_action_targets(
        data, 'language_builder', 0, 0, horizons=16)
    assert target.shape == (11, 16, 23)
    assert valid.shape == (11, 16) and valid.all()
    assert indices[:, -1].tolist() == list(range(32, 113, 8))
    np.testing.assert_equal(target[-1], actions[112:128, 0])


def test_absolute_supervision_ranges_keep_only_validated_recovery_actions():
    valid = np.ones((9, 32), dtype=bool)
    indices = torch.from_numpy(
        np.arange(32, 97, 8)[:, None] + np.arange(-7, 1)[None]
    )
    masked = mask_targets_to_absolute_ranges(valid, indices, [[40, 48]], 0)
    # The same validated action range appears in two overlapping dense
    # windows, so both copies remain supervised.
    assert masked.sum() == 16
    assert masked[0, 8:16].all()
    assert masked[1, :8].all()
    assert not masked[0, :8].any()
    assert not masked[2:].any()


def test_dense_forward_matches_separate_anchors_and_is_causal():
    torch.manual_seed(4)
    cfg = InsertedPolicyArgs(num_policy_blocks=2, text_hidden_size=16,
                             horizons=32, causal_observations=True, checkpoint_policy=True)
    model = InsertedInhabitantPolicy(tiny_model(), cfg).eval()
    cond = conditions(25); cond['condition_mask'][:] = True
    latent = torch.randn(1, 25, 16, 4, 4)
    indices = torch.stack((torch.arange(5, 13), torch.arange(13, 21)))[None]
    text = torch.randn(1, 3, 16); mask = torch.ones(1, 3, dtype=torch.bool)
    args = (torch.tensor([FAMILY_TO_ID['language_builder']]),
            torch.tensor([PROFILE_TO_ID['language_builder']]))
    kwargs = dict(shared_text=text, shared_text_mask=mask, current_text=text, current_text_mask=mask)
    with torch.no_grad():
        _, dense = model(latent, cond, *args, **kwargs, policy_indices=indices)
        assert dense['keys'].shape == (1, 2, 32, 10)
        for a in range(2):
            _, single = model(latent, cond, *args, **kwargs, policy_indices=indices[:, a])
            for key in dense: torch.testing.assert_close(dense[key][:, a], single[key], atol=2e-5, rtol=2e-5)
        latent[:, 13:] += 100
        cond['hp'][:, 13:] = 1
        _, changed = model(latent, cond, *args, **kwargs, policy_indices=indices)
        for key in dense: torch.testing.assert_close(dense[key][:, 0], changed[key][:, 0])
        assert model.decode_execution(dense).shape == (1, 2, 8, 23)
    model.train()
    _, logits = model(latent, cond, *args, **kwargs, policy_indices=indices)
    loss, _ = model.loss(logits, torch.zeros(1, 2, 32, 23), torch.ones(1, 2, 32, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.renderer.parameters())
    assert any(p.grad is not None for p in model.families.parameters())


def test_warm_start_copies_trunk_and_tiles_head():
    old = InsertedInhabitantPolicy(tiny_model(), InsertedPolicyArgs(num_policy_blocks=1, text_hidden_size=16))
    new = InsertedInhabitantPolicy(tiny_model(), InsertedPolicyArgs(num_policy_blocks=1, text_hidden_size=16, horizons=32))
    migrated = new.warm_start_horizons(old.families.state_dict())
    assert len(migrated) == 24
    old_state, new_state = old.families.state_dict(), new.families.state_dict()
    for key, value in old_state.items():
        if key in migrated:
            for chunk in new_state[key].chunk(4, 0): torch.testing.assert_close(chunk, value)
        else: torch.testing.assert_close(new_state[key], value)


def test_warm_start_slices_decoder_32_to_16_and_adds_mouse_residuals():
    from dataclasses import replace
    cfg = InsertedPolicyArgs(num_policy_blocks=1, text_hidden_size=16, horizons=32,
                             builder_spatial_tokens=2, builder_action_decoder_layers=1,
                             builder_action_decoder_hidden=16, builder_action_decoder_heads=4)
    old = InsertedInhabitantPolicy(tiny_model(), cfg)
    new = InsertedInhabitantPolicy(
        tiny_model(), replace(cfg, horizons=16, continuous_mouse=True))
    changed = new.warm_start_horizons(old.families.state_dict(), old_horizons=32)
    assert any(key.endswith('action_queries') for key in changed)
    old_state, new_state = old.families.state_dict(), new.families.state_dict()
    torch.testing.assert_close(
        new_state['language_builder.head.action_queries'],
        old_state['language_builder.head.action_queries'][:16])
    assert torch.count_nonzero(
        new_state['language_builder.head.mouse_x_residual.weight']) == 0


def test_builder_visual_warm_start_preserves_combat_and_adds_trainable_readers():
    torch.manual_seed(8)
    old_cfg = InsertedPolicyArgs(num_policy_blocks=1, text_hidden_size=16)
    new_cfg = InsertedPolicyArgs(num_policy_blocks=1, text_hidden_size=16,
                                 builder_visual_queries=4)
    old = InsertedInhabitantPolicy(tiny_model(), old_cfg).eval()
    new = InsertedInhabitantPolicy(tiny_model(), new_cfg).eval()
    added = new.warm_start_builder_vision(old.families.state_dict())
    assert len(added) == 4
    assert all(key.startswith('language_builder.blocks.0.') for key in added)
    assert new.families['language_builder'].blocks[0].visual_query_offsets.shape == (4, 32)
    for key, value in old.families['combat'].state_dict().items():
        torch.testing.assert_close(new.families['combat'].state_dict()[key], value)

    new.train()
    cond = conditions(8); cond['condition_mask'][:] = True
    latent = torch.randn(1, 8, 16, 4, 4)
    text = torch.randn(1, 3, 16); mask = torch.ones(1, 3, dtype=torch.bool)
    args = (torch.tensor([FAMILY_TO_ID['language_builder']]),
            torch.tensor([PROFILE_TO_ID['language_builder']]))
    _, logits = new(latent, cond, *args, shared_text=text, shared_text_mask=mask,
                    current_text=text, current_text_mask=mask)
    logits['keys'].sum().backward()
    block = new.families['language_builder'].blocks[0]
    assert block.visual_text_gate.grad is not None
    assert block.visual_query_offsets.grad is not None


def test_dense_mask_excludes_invalid_anchor_and_weight_normalizes():
    model = InsertedInhabitantPolicy(tiny_model(), InsertedPolicyArgs(
        num_policy_blocks=1, text_hidden_size=16, horizons=32))
    head = model.families['language_builder'].head
    logits = head(torch.randn(1, 2, 32))
    actions = torch.zeros(1, 2, 32, 23)
    valid = torch.ones(1, 2, 32, dtype=torch.bool); valid[:, 1] = False
    weights = torch.ones(32); weights[:8] = 2
    loss, _ = model.loss(logits, actions, valid, weights)
    reference, _ = model.loss({k: v[:, 0] for k, v in logits.items()}, actions[:, 0],
                              valid[:, 0], weights)
    torch.testing.assert_close(loss, reference)
    doubled, _ = model.loss(logits, actions, valid, weights * 2)
    torch.testing.assert_close(loss, doubled)


def test_weighted_sampler_resume_epoch_reproduces_remaining_order():
    from train_scripts.train_policy import DistributedWeightedSampler
    first = DistributedWeightedSampler([1, 2, 3, 4, 5], 2, 1, seed=13)
    first.set_epoch(4)
    order = list(first)
    resumed = DistributedWeightedSampler([1, 2, 3, 4, 5], 2, 1, seed=13)
    resumed.set_epoch(4)
    iterator = iter(resumed); next(iterator)
    assert list(iterator) == order[1:]


def test_validation_subset_is_fixed_profile_balanced_and_disjoint():
    from train_scripts.train_policy import ProfileValidationSampler
    rows = [{'profile': str(p)} for p in range(5) for _ in range(50)]
    shards = [list(ProfileValidationSampler(rows, 8, 4, rank)) for rank in range(4)]
    selected = sum(shards, [])
    assert len(selected) == len(set(selected)) == 32
    counts = [sum(rows[i]['profile'] == str(p) for i in selected) for p in range(5)]
    assert max(counts) - min(counts) == 1
    assert shards[2] == list(ProfileValidationSampler(rows, 8, 4, 2))
