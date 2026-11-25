"""Causal NPC-bank contract and migration from the running spatial decoder."""
from dataclasses import asdict, replace

import pytest
import torch

import tests.test_action_chunk_decoder as decoder_tests
from miworld.models.inserted_policy import InsertedInhabitantPolicy, NPCPolicyBlock
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from tests.test_renderer import tiny_model


def npc_configs():
    _, old = decoder_tests.configs()
    return old, replace(old, builder_npc_attention=True, builder_visual_queries=1)


def test_npc_migration_preserves_tensors_combat_and_roundtrip():
    torch.manual_seed(67)
    old_cfg, cfg = npc_configs()
    old = InsertedInhabitantPolicy(tiny_model(), old_cfg).eval()
    decoder_tests.activate_decoder(old)  # Match a trained, nonzero output projection.
    model = InsertedInhabitantPolicy(tiny_model(), cfg).eval()
    model.renderer.load_state_dict(old.renderer.state_dict())
    report = model.warm_start_builder_npc(old.families.state_dict(), asdict(old_cfg))
    assert len(report['added']) == cfg.num_policy_blocks
    assert all(key.endswith('.main_text_gate') for key in report['added'])
    assert report['removed'] and all(key.startswith('language_builder.blocks.') for key in report['removed'])
    for key, value in old.families.state_dict().items():
        if key not in report['removed']:
            torch.testing.assert_close(value, model.families.state_dict()[key], rtol=0, atol=0)
    latent, cond, family, profile, kwargs = decoder_tests.inputs()
    with torch.no_grad():
        for profile_name in ('zombie_melee', 'skeleton_swordsman'):
            family.fill_(FAMILY_TO_ID['combat']); profile.fill_(PROFILE_TO_ID[profile_name])
            _, before = old(latent, cond, family, profile, **kwargs)
            _, after = model(latent, cond, family, profile, **kwargs)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        family.fill_(FAMILY_TO_ID['language_builder']); profile.fill_(PROFILE_TO_ID['language_builder'])
        _, before = old(latent, cond, family, profile, **kwargs)
        _, after = model(latent, cond, family, profile, **kwargs)
        assert not torch.allclose(before['keys'], after['keys'])
        assert model.decode_execution(after).shape == (1, 2, 8, 23)
        clone = InsertedInhabitantPolicy(tiny_model(), cfg).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        _, roundtrip = clone(latent, cond, family, profile, **kwargs)
        for key in after:
            torch.testing.assert_close(after[key], roundtrip[key], rtol=0, atol=0)


def test_npc_migration_rejects_config_missing_extra_and_shape_errors():
    old_cfg, cfg = npc_configs()
    old = InsertedInhabitantPolicy(tiny_model(), old_cfg)
    model = InsertedInhabitantPolicy(tiny_model(), cfg)
    state = old.families.state_dict()
    for bad in (replace(old_cfg, current_text_scale=3.), cfg,
                replace(old_cfg, builder_action_decoder_layers=0)):
        with pytest.raises(ValueError): model.warm_start_builder_npc(state, asdict(bad))
    for key in ('language_builder.blocks.0.policy_kv.weight', 'combat.base'):
        incomplete = dict(state); incomplete.pop(key)
        with pytest.raises(ValueError): model.warm_start_builder_npc(incomplete, asdict(old_cfg))
    with pytest.raises(ValueError):
        model.warm_start_builder_npc({**state, 'unknown': torch.zeros(1)}, asdict(old_cfg))
    with pytest.raises(ValueError):
        model.warm_start_builder_npc({**state, 'combat.base': torch.zeros(1)}, asdict(old_cfg))
    for bad in (replace(cfg, builder_visual_queries=4), replace(cfg, builder_spatial_tokens=0)):
        with pytest.raises(ValueError): InsertedInhabitantPolicy(tiny_model(), bad)


@pytest.mark.parametrize('check', [
    decoder_tests.test_spatial_decoder_dense_causality_and_cached_features,
    decoder_tests.test_checkpointed_backward_reaches_spatial_text_and_decoder_but_not_renderer,
])
def test_npc_end_to_end_contracts(monkeypatch, check):
    old, cfg = npc_configs()
    monkeypatch.setattr(decoder_tests, 'configs', lambda: (old, cfg))
    check()


def block_inputs():
    states = torch.randn(2, 8, 9, 32)
    video = torch.randn(2, 8, 2, 2, 32, requires_grad=True)
    text = torch.randn(2, 8, 4, 16)
    text_mask = torch.ones(2, 8, 4, dtype=torch.bool); text_mask[:, :, -1] = False
    valid = torch.ones(2, 8, dtype=torch.bool); valid[:, 0] = False
    main = torch.randn(2, 5, 16)
    main_mask = torch.ones(2, 5, dtype=torch.bool); main_mask[:, -1] = False
    return states, video, text, text_mask, valid, main, main_mask


def test_all_npc_slots_read_history_and_same_frame_but_never_future():
    torch.manual_seed(71)
    block = NPCPolicyBlock(32, 4, 16, current_text_scale=2.).eval()
    args = block_inputs()
    with torch.no_grad():
        base = block(*args)
        future = [x.clone() for x in args]
        for i in (0, 1, 2): future[i][:, 5:] += torch.randn_like(future[i][:, 5:]) * 10
        changed = block(*future)
        torch.testing.assert_close(base[:, :5], changed[:, :5], rtol=0, atol=0)
        past = [x.clone() for x in args]
        past[0][:, 1, 4] += torch.randn(2, 32) * 10
        delta = (block(*past)[:, -1] - base[:, -1]).abs().sum(-1)
        assert (delta > 1e-5).all()  # Every slot, including the readout token.
        same = [x.clone() for x in args]
        same[0][:, 4, 7] += torch.randn(2, 32) * 10
        assert ((block(*same)[:, 4, 0] - base[:, 4, 0]).abs().sum(-1) > 1e-5).all()


def test_npc_masks_and_all_empty_backward_are_finite():
    torch.manual_seed(73)
    block = NPCPolicyBlock(32, 4, 16, current_text_scale=2.)
    args = block_inputs()
    base = block(*args)
    padded = [x.detach().clone() for x in args]
    padded[0][:, 0] += 10000; padded[1][:, 0] += 10000
    padded[2][~padded[3]] += 10000; padded[5][~padded[6]] += 10000
    torch.testing.assert_close(base, block(*padded), rtol=0, atol=0)
    for i in (2, 5): args[i].requires_grad_(True)
    block(*args).square().mean().backward()
    assert args[1].grad is None
    for i in (2, 5): assert args[i].grad.abs().sum() > 0
    for p in (block.main_text_gate, block.current_text_gate, block.temporal_qkv.weight):
        assert p.grad.abs().sum() > 0
    block.zero_grad(set_to_none=True)
    empty = list(args)
    for i in (3, 4, 6): empty[i] = torch.zeros_like(empty[i])
    out = block(*empty)
    assert torch.isfinite(out).all() and out.abs().sum() == 0
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)


def test_goal_conditioned_visual_query_zero_start_migration_and_gradient():
    torch.manual_seed(79)
    _, source_cfg = npc_configs()
    cfg = replace(source_cfg, builder_goal_conditioned_vision=True)
    source = InsertedInhabitantPolicy(tiny_model(), source_cfg).eval()
    decoder_tests.activate_decoder(source)
    model = InsertedInhabitantPolicy(tiny_model(), cfg).eval()
    model.renderer.load_state_dict(source.renderer.state_dict())
    added = model.warm_start_goal_conditioned_vision(
        source.families.state_dict(), asdict(source_cfg))
    assert len(added) == cfg.num_policy_blocks
    assert all(key.endswith('.goal_visual_gate') for key in added)
    latent, cond, family, profile, kwargs = decoder_tests.inputs()
    family.fill_(FAMILY_TO_ID['language_builder'])
    profile.fill_(PROFILE_TO_ID['language_builder'])
    with torch.no_grad():
        _, before = source(latent, cond, family, profile, **kwargs)
        _, after = model(latent, cond, family, profile, **kwargs)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    model.train()
    _, logits = model(latent, cond, family, profile, **kwargs)
    sum(value.square().mean() for value in logits.values()).backward()
    for block in model.families['language_builder'].blocks:
        assert block.goal_visual_gate.grad is not None
        assert torch.isfinite(block.goal_visual_gate.grad)


def test_pure_text_visual_queries_ignore_state_and_respond_to_goal():
    torch.manual_seed(83)
    block = NPCPolicyBlock(
        32, 4, 16, current_text_scale=2., text_only_visual_queries=True,
        visual_query_slots=9).eval()
    states, _, _, _, _, shared, shared_mask = block_inputs()
    with torch.no_grad():
        query = block.visual_query(states, shared, shared_mask)
        changed_state = block.visual_query(states + torch.randn_like(states) * 100,
                                           shared, shared_mask)
        torch.testing.assert_close(query, changed_state, rtol=0, atol=0)
        changed_goal = block.visual_query(states, shared + 3, shared_mask)
        assert not torch.allclose(query, changed_goal)
    assert query.shape == states.shape


def test_pure_text_visual_query_migration_replaces_only_gates():
    torch.manual_seed(89)
    _, base_cfg = npc_configs()
    source_cfg = replace(base_cfg, builder_goal_conditioned_vision=True)
    target_cfg = replace(source_cfg, builder_goal_conditioned_vision=False,
                         builder_text_only_visual_queries=True)
    source = InsertedInhabitantPolicy(tiny_model(), source_cfg).eval()
    model = InsertedInhabitantPolicy(tiny_model(), target_cfg).eval()
    report = model.warm_start_text_only_visual_queries(
        source.families.state_dict(), asdict(source_cfg))
    assert len(report['added']) == target_cfg.num_policy_blocks
    assert len(report['removed']) == target_cfg.num_policy_blocks
    assert all(key.endswith('.goal_visual_query_offsets') for key in report['added'])
    assert all(key.endswith('.goal_visual_gate') for key in report['removed'])
    target = model.families.state_dict()
    for key, value in source.families.state_dict().items():
        if key not in report['removed']:
            torch.testing.assert_close(value, target[key], rtol=0, atol=0)
    latent, cond, family, profile, kwargs = decoder_tests.inputs()
    family.fill_(FAMILY_TO_ID['language_builder'])
    profile.fill_(PROFILE_TO_ID['language_builder'])
    _, logits = model(latent, cond, family, profile, **kwargs)
    sum(value.square().mean() for value in logits.values()).backward()
    for block in model.families['language_builder'].blocks:
        assert block.goal_visual_query_offsets.grad is not None
        assert block.goal_visual_query_offsets.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BF16 smoke test')
def test_npc_cuda_bf16_checkpointed_training(monkeypatch):
    old, cfg = npc_configs()
    monkeypatch.setattr(decoder_tests, 'configs', lambda: (old, cfg))
    decoder_tests.test_cuda_bf16_decoder_forward_backward_and_eval()
