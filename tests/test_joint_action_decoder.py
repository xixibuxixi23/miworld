from dataclasses import asdict, replace

import pytest
import torch

from miworld.models.joint_action_decoder import JointActionDecoder
from miworld.models.inserted_policy import InsertedInhabitantPolicy
from tests.test_action_chunk_decoder import configs, inputs
from tests.test_renderer import tiny_model


def head_inputs():
    torch.manual_seed(101)
    states = torch.randn(2, 8, 3, 32)
    video = torch.randn(2, 8, 4, 4, 32)
    text = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[:, 0] = False
    return states, valid, text, mask, text.clone(), mask.clone(), video


def new_head(checkpoint_layers=False):
    return JointActionDecoder(32, 16, 16, decoder_hidden=32, layers=2, heads=4,
                              visual_grid=2, continuous_mouse=True,
                              checkpoint_layers=checkpoint_layers)


@pytest.mark.parametrize('checkpoint_layers', [False, True])
def test_joint_v2_gradients_reach_direct_vision_both_texts_and_group_queries(checkpoint_layers):
    head = new_head(checkpoint_layers).train()
    data = list(head_inputs())
    for index in (0, 2, 4, 6):
        data[index].requires_grad_(True)
    logits = head(*data)
    assert logits['keys'].shape == (2, 16, 10)
    assert logits['mouse_x_residual'].shape == (2, 16)
    assert head.keys.weight.shape == (10, 32)  # shared, not horizon-specific
    actions = torch.zeros(2, 16, 23)
    actions[..., 9] = 1
    actions[..., 21] = .01
    loss, _ = head.loss(logits, actions)
    loss.backward()
    for index in (0, 2, 4, 6):
        assert data[index].grad is not None
        assert torch.isfinite(data[index].grad).all()
        assert data[index].grad.abs().sum() > 0
    for parameter in (head.video_projection.weight, head.group_queries,
                      head.horizon_queries, head.decoder[0].self_attn.in_proj_weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_joint_v2_padding_invariance_all_invalid_fallback_and_input_ablation():
    head = new_head().eval()
    data = head_inputs()
    with torch.no_grad():
        base = head(*data)
        padded = [value.clone() for value in data]
        padded[0][:, 0] = 1e5
        padded[6][:, 0] = 1e5
        padded[2][~padded[3]] = -1e5
        padded[4][~padded[5]] = 1e5
        for key, value in head(*padded).items():
            torch.testing.assert_close(value, base[key], rtol=0, atol=0)
        # Verify separate observed memory paths alter predictions at initialization.
        for index in (0, 2, 4, 6):
            changed = [value.clone() for value in data]
            changed[index] += torch.randn_like(changed[index]) * 2
            assert not torch.allclose(head(*changed)['keys'], base['keys'])
        empty = [value.clone() for value in data]
        for index in (1, 3, 5):
            empty[index].zero_()
        assert all(torch.isfinite(value).all() for value in head(*empty).values())


def test_joint_v2_migration_retains_nonhead_and_rejects_corrupt_or_changed_source():
    _, old_cfg = configs()
    new_cfg = replace(old_cfg, builder_action_architecture='joint_v2',
                      builder_action_decoder_hidden=64, builder_action_decoder_layers=3,
                      builder_action_visual_grid=2)
    old = InsertedInhabitantPolicy(tiny_model(), old_cfg)
    model = InsertedInhabitantPolicy(tiny_model(), new_cfg)
    migration = model.warm_start_action_v2(old.families.state_dict(), asdict(old_cfg))
    assert migration['retained_tensors'] > 0 and migration['initialized_head_tensors']
    assert all(key.startswith('language_builder.head.')
               for key in migration['initialized_head_tensors'])
    for key, value in old.families.state_dict().items():
        if not key.startswith('language_builder.head.'):
            torch.testing.assert_close(value, model.families.state_dict()[key], rtol=0, atol=0)
    broken = dict(old.families.state_dict())
    broken.pop('language_builder.head.decoder.0.norm1.weight')
    with pytest.raises(ValueError, match='source schema'):
        model.warm_start_action_v2(broken, asdict(old_cfg))
    with pytest.raises(ValueError, match='other policy settings'):
        model.warm_start_action_v2(old.families.state_dict(),
                                   asdict(replace(old_cfg, current_text_scale=3.)))
    reloaded = InsertedInhabitantPolicy(tiny_model(), new_cfg)
    reloaded.load_state_dict(model.state_dict(), strict=True)


def test_joint_v2_dense_future_invariance_and_cached_forward():
    _, cfg = configs()
    cfg = replace(cfg, builder_action_architecture='joint_v2', builder_action_visual_grid=2)
    model = InsertedInhabitantPolicy(tiny_model(), cfg).eval()
    latent, cond, family, profile, kwargs = inputs()
    with torch.no_grad():
        _, baseline = model(latent, cond, family, profile, **kwargs)
        latent[:, 13:] += 100
        cond['hp'][:, 13:] = 1
        kwargs['current_text'][:, 13:] += 100
        _, changed = model(latent, cond, family, profile, **kwargs)
        for key in baseline:
            torch.testing.assert_close(baseline[key][:, 0], changed[key][:, 0],
                                       atol=3e-5, rtol=3e-5)
        # Cache path must provide the same direct vision modality as full forward.
        short = {key: (value[:, :8] if torch.is_tensor(value) and value.ndim >= 2
                       and value.shape[1] == 25 else value) for key, value in cond.items()}
        _, cached = model(None, short, family, profile,
                           shared_text=kwargs['shared_text'], shared_text_mask=kwargs['shared_text_mask'],
                           current_text=kwargs['current_text'][:, :8],
                           current_text_mask=kwargs['current_text_mask'][:, :8],
                           video_layers=[torch.randn(1, 8, 2, 2, 32) for _ in range(2)])
        assert cached['keys'].shape == (1, 32, 10)
        assert all(torch.isfinite(value).all() for value in cached.values())


def test_joint_v2_full_policy_backward_keeps_renderer_frozen():
    _, cfg = configs()
    cfg = replace(cfg, builder_action_architecture='joint_v2', builder_action_visual_grid=2)
    model = InsertedInhabitantPolicy(tiny_model(), cfg).train()
    latent, cond, family, profile, kwargs = inputs()
    _, logits = model(latent, cond, family, profile, **kwargs)
    loss, _ = model.loss(logits, torch.zeros(1, 2, 32, 23))
    loss.backward()
    branch = model.families['language_builder']
    for parameter in (branch.head.video_projection.weight, branch.blocks[-1].spatial_out.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.renderer.parameters())
