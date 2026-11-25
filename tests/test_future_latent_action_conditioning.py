import torch

from miworld.models.joint_action_decoder import JointActionDecoder
from tests.test_joint_action_decoder import head_inputs


def new_future_head():
    return JointActionDecoder(
        32, 16, 16, decoder_hidden=32, layers=2, heads=4,
        visual_grid=2, continuous_mouse=True,
        future_latent_channels=4, future_latent_grid=2,
    )


def test_future_plan_migration_preserves_logits_and_has_exact_new_keys():
    torch.manual_seed(20261125)
    parent = new_future_head().eval()
    state = parent.state_dict()
    parent_state = {
        key: value for key, value in state.items()
        if 'future_latent_action_' not in key
    }
    migrated = new_future_head().eval()
    incompatible = migrated.load_state_dict(parent_state, strict=False)
    expected = {
        key for key, _ in migrated.named_parameters()
        if 'future_latent_action_' in key
    }
    assert set(incompatible.missing_keys) == expected
    assert not incompatible.unexpected_keys
    assert torch.count_nonzero(
        migrated.future_latent_action_attention.out_proj.weight) == 0
    with torch.no_grad():
        parent_logits = parent(*head_inputs())
        migrated_logits = migrated(*head_inputs())
    for key in ('keys', 'hotbar', 'mouse_x', 'mouse_y',
                'mouse_x_residual', 'mouse_y_residual', 'future_latent'):
        torch.testing.assert_close(
            migrated_logits[key], parent_logits[key], rtol=0, atol=0)


def test_action_loss_trains_future_plan_attention_and_prediction():
    torch.manual_seed(20261125)
    head = new_future_head().train()
    logits = head(*head_inputs())
    assert logits['future_latent'].shape == (2, 2, 4, 2, 2)
    loss = logits['keys'].square().mean() + logits['mouse_x'].square().mean()
    loss.backward()
    output = head.future_latent_action_attention.out_proj.weight
    assert output.grad is not None and output.grad.abs().sum() > 0

    # After the zero-start output learns, action loss must reach the predicted
    # future map and its spatial projection rather than treating it as an
    # unrelated auxiliary head.
    with torch.no_grad():
        output.normal_(std=.01)
    head.zero_grad(set_to_none=True)
    logits = head(*head_inputs())
    loss = logits['keys'].square().mean() + logits['mouse_x'].square().mean()
    loss.backward()
    for parameter in (
        head.future_latent_head[-1].weight,
        head.future_latent_action_projection.weight,
        head.future_latent_action_attention.in_proj_weight,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
