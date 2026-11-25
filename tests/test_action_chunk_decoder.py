from dataclasses import asdict, replace

import pytest
import torch

from miworld.models.action_chunk_decoder import ActionChunkDecoder
from miworld.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from tests.test_renderer import conditions, tiny_model


def configs():
    old = InsertedPolicyArgs(num_policy_blocks=2, text_hidden_size=16, horizons=32,
                             causal_observations=True, checkpoint_policy=True,
                             builder_visual_queries=4, current_text_scale=2., structured_inputs=True)
    return old, replace(old, builder_spatial_tokens=8, builder_action_decoder_layers=2,
                        builder_action_decoder_hidden=32, builder_action_decoder_heads=4)


def inputs(frames=25):
    cond = conditions(frames)
    cond['condition_mask'][:] = True
    cond['player_appearance_id'] = torch.zeros(1, frames, 2, dtype=torch.long)
    latent = torch.randn(1, frames, 16, 4, 4)
    indices = torch.stack((torch.arange(5, 13), torch.arange(13, 21)))[None]
    kwargs = dict(shared_text=torch.randn(1, 3, 16),
                  shared_text_mask=torch.ones(1, 3, dtype=torch.bool),
                  current_text=torch.randn(1, frames, 3, 16),
                  current_text_mask=torch.ones(1, frames, 3, dtype=torch.bool),
                  policy_indices=indices)
    family = torch.tensor([FAMILY_TO_ID['language_builder']])
    profile = torch.tensor([PROFILE_TO_ID['language_builder']])
    return latent, cond, family, profile, kwargs


def activate_decoder(model):
    # Zero-init preserves old output; exercise real dependency/gradient paths
    # after making the projection nonzero, as happens on the first update.
    torch.nn.init.normal_(model.families['language_builder'].head.output_projection.weight, std=.02)


def test_migration_preserves_old_weights_logits_loss_and_combat():
    torch.manual_seed(27)
    old_cfg, new_cfg = configs()
    old = InsertedInhabitantPolicy(tiny_model(), old_cfg).eval()
    new = InsertedInhabitantPolicy(tiny_model(), new_cfg).eval()
    new.renderer.load_state_dict(old.renderer.state_dict())
    added = new.warm_start_builder_decoder(old.families.state_dict(), asdict(old_cfg))
    assert added and all(k.startswith('language_builder.') for k in added)
    assert 'language_builder.spatial_slots' in added
    for key, value in old.families.state_dict().items():
        torch.testing.assert_close(value, new.families.state_dict()[key], rtol=0, atol=0)
    latent, cond, family, profile, kwargs = inputs()
    with torch.no_grad():
        for family_name, profile_name in [('language_builder', 'language_builder'),
                                           ('combat', 'zombie_melee'), ('combat', 'skeleton_swordsman')]:
            family.fill_(FAMILY_TO_ID[family_name]); profile.fill_(PROFILE_TO_ID[profile_name])
            _, before = old(latent, cond, family, profile, **kwargs)
            _, after = new(latent, cond, family, profile, **kwargs)
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=2e-5, rtol=2e-5)
            actions = torch.zeros(1, 2, 32, 23)
            torch.testing.assert_close(old.loss(before, actions)[0], new.loss(after, actions)[0])
            assert new.decode_execution(after).shape == (1, 2, 8, 23)


def test_spatial_decoder_dense_causality_and_cached_features():
    torch.manual_seed(31)
    _, cfg = configs()
    model = InsertedInhabitantPolicy(tiny_model(), cfg).eval()
    activate_decoder(model)
    latent, cond, family, profile, kwargs = inputs()
    features = []
    handle = model.families['language_builder'].blocks[-1].register_forward_hook(
        lambda module, args, output: features.append(output.detach()))
    with torch.no_grad():
        _, dense = model(latent, cond, family, profile, **kwargs)
        assert features[-1].shape == (2, 8, 9, 32)
        for a in range(2):
            single_kw = {**kwargs, 'policy_indices': kwargs['policy_indices'][:, a]}
            _, single = model(latent, cond, family, profile, **single_kw)
            for key in dense:
                torch.testing.assert_close(dense[key][:, a], single[key], atol=3e-5, rtol=3e-5)
        latent[:, 13:] += 100
        cond['hp'][:, 13:] = 1
        kwargs['current_text'][:, 13:] += 100
        _, changed = model(latent, cond, family, profile, **kwargs)
        for key in dense:
            torch.testing.assert_close(dense[key][:, 0], changed[key][:, 0], atol=2e-5, rtol=2e-5)
        short = conditions(8)
        short['player_appearance_id'] = torch.zeros(1, 8, 2, dtype=torch.long)
        short_text = dict(shared_text=kwargs['shared_text'], shared_text_mask=kwargs['shared_text_mask'],
                          current_text=kwargs['current_text'][:, :8],
                          current_text_mask=kwargs['current_text_mask'][:, :8])
        _, cached = model(None, short, family, profile, **short_text,
                           video_layers=[torch.randn(1, 8, 2, 2, 32) for _ in range(2)])
        assert cached['keys'].shape == (1, 32, 10)
        assert torch.isfinite(cached['keys']).all()
    handle.remove()


def test_checkpointed_backward_reaches_spatial_text_and_decoder_but_not_renderer():
    torch.manual_seed(37)
    _, cfg = configs()
    model = InsertedInhabitantPolicy(tiny_model(), cfg).train()
    activate_decoder(model)
    latent, cond, family, profile, kwargs = inputs()
    kwargs['shared_text'].requires_grad_(True)
    kwargs['current_text'].requires_grad_(True)
    _, logits = model(latent, cond, family, profile, **kwargs)
    loss, _ = model.loss(logits, torch.zeros(1, 2, 32, 23),
                         torch.ones(1, 2, 32, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    branch = model.families['language_builder']
    for parameter in (branch.spatial_slots, branch.head.action_queries,
                       branch.head.decoder[0].self_attn.in_proj_weight,
                       branch.head.decoder[0].multihead_attn.in_proj_weight,
                       branch.head.memory_types):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for text in (kwargs['shared_text'], kwargs['current_text']):
        assert text.grad is not None and text.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.renderer.parameters())


def test_decoder_masks_padding_and_reads_both_texts_and_spatial_slots():
    torch.manual_seed(41)
    head = ActionChunkDecoder(32, 32, 16, decoder_hidden=32, layers=2, heads=4).eval()
    torch.nn.init.normal_(head.output_projection.weight, std=.02)
    states = torch.randn(2, 8, 9, 32)
    valid = torch.ones(2, 8, dtype=torch.bool); valid[:, 0] = False
    main, current = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        base = head(states, valid, main, mask, current, mask)
        changed_states = states.clone(); changed_states[:, 0] += 100
        changed_main, changed_current = main.clone(), current.clone()
        changed_main[~mask] += 100; changed_current[~mask] += 100
        padded = head(changed_states, valid, changed_main, mask, changed_current, mask)
        for key in base: torch.testing.assert_close(base[key], padded[key])
        for mode in ('spatial', 'main', 'current'):
            s, m, c = states.clone(), main.clone(), current.clone()
            if mode == 'spatial': s[:, :, 1:] += torch.randn_like(s[:, :, 1:]) * 2
            if mode == 'main': m[:, :2] += torch.randn_like(m[:, :2]) * 2
            if mode == 'current': c[:, :2] += torch.randn_like(c[:, :2]) * 2
            other = head(s, valid, m, mask, c, mask)
            assert not torch.allclose(base['keys'][0], other['keys'][0])
        empty = head(states, torch.zeros_like(valid), main, torch.zeros_like(mask),
                     current, torch.zeros_like(mask))
        assert all(torch.isfinite(v).all() for v in empty.values())


def test_roundtrip_and_reject_incompatible_migration():
    old_cfg, cfg = configs()
    old = InsertedInhabitantPolicy(tiny_model(), old_cfg)
    model = InsertedInhabitantPolicy(tiny_model(), cfg)
    model.warm_start_builder_decoder(old.families.state_dict(), asdict(old_cfg))
    reloaded = InsertedInhabitantPolicy(tiny_model(), InsertedPolicyArgs(**asdict(cfg)))
    reloaded.load_state_dict(model.state_dict(), strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key])
    with pytest.raises(ValueError, match='other policy settings'):
        model.warm_start_builder_decoder(old.families.state_dict(), asdict(replace(old_cfg, current_text_scale=1.)))
    broken = dict(old.families.state_dict()); broken.pop('combat.base')
    with pytest.raises(ValueError, match='unexpected keys'):
        model.warm_start_builder_decoder(broken, asdict(old_cfg))
    with pytest.raises(ValueError, match='require an action decoder'):
        InsertedInhabitantPolicy(tiny_model(), replace(old_cfg, builder_spatial_tokens=8))


def test_zero_residual_initialization_opens_decoder_gradient_after_first_update():
    torch.manual_seed(53)
    head = ActionChunkDecoder(32, 32, 16, decoder_hidden=32, layers=2, heads=4)
    optimizer = torch.optim.SGD(head.parameters(), lr=.1)
    states = torch.randn(1, 8, 9, 32)
    text = torch.randn(1, 3, 16)
    valid, mask = torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)
    for update in range(2):
        optimizer.zero_grad()
        loss, _ = head.loss(head(states, valid, text, mask, text, mask), torch.zeros(1, 32, 23))
        loss.backward()
        assert torch.isfinite(loss)
        assert head.output_projection.weight.grad.abs().sum() > 0
        if update == 1:
            assert head.action_queries.grad.abs().sum() > 0
            assert head.memory_projection.weight.grad.abs().sum() > 0
        optimizer.step()


def test_continuous_mouse_decoder_residual_matches_horizon_shape():
    head = ActionChunkDecoder(32, 16, 16, decoder_hidden=32, layers=1, heads=4,
                              continuous_mouse=True)
    states = torch.randn(2, 8, 3, 32)
    text = torch.randn(2, 3, 16)
    valid = torch.ones(2, 8, dtype=torch.bool)
    mask = torch.ones(2, 3, dtype=torch.bool)
    logits = head(states, valid, text, mask, text, mask)
    assert logits['mouse_x'].shape == (2, 16, 17)
    assert logits['mouse_x_residual'].shape == (2, 16)
    actions = torch.zeros(2, 16, 23)
    loss, _ = head.loss(logits, actions)
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BF16 smoke test')
def test_cuda_bf16_decoder_forward_backward_and_eval():
    torch.manual_seed(59)
    _, cfg = configs()
    cfg = replace(cfg, builder_action_decoder_layers=4,
                  builder_action_decoder_hidden=512, builder_action_decoder_heads=8)
    model = InsertedInhabitantPolicy(tiny_model(), cfg).cuda().train()
    activate_decoder(model)
    latent, cond, family, profile, kwargs = inputs()
    cond = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cond.items()}
    kwargs = {key: value.cuda() for key, value in kwargs.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, logits = model(latent.cuda(), cond, family.cuda(), profile.cuda(), **kwargs)
        loss, _ = model.loss(logits, torch.zeros(1, 2, 32, 23, device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.families.parameters() if p.grad is not None)
    assert model.families['language_builder'].spatial_slots.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.renderer.parameters())
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        _, logits = model(latent.cuda(), cond, family.cuda(), profile.cuda(), **kwargs)
    assert all(torch.isfinite(value).all() for value in logits.values())
