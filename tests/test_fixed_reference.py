from dataclasses import asdict

import torch

from miworld.models.renderer import (
    Renderer,
    RendererArgs,
    player_orientation_features,
)
from miworld.models.renderer_backbone.fixed_reference import (
    FixedSpatialReferenceProjector,
    relative_camera_pose,
)
from miworld.models.renderer_backbone.dit_pixel import FrameDepthStackPixelDiT
from train_scripts.train_renderer import migrate_roi_appearance_fusion


def test_zero_roi_keeps_fixed_reference_in_the_gradient_graph():
    model = FixedSpatialReferenceProjector(gradient_checkpointing=False)
    reference = torch.randn(1, 2, 4, 128, 256, requires_grad=True)
    actor = torch.randn(1, 2, 2, 30, requires_grad=True)
    output = model(
        reference,
        torch.zeros(1, 2, 2, 36, 64),
        torch.zeros(1, 2, 2, 36, 64, 2),
        actor,
        torch.ones(1, 2, 4, dtype=torch.bool),
        torch.randn(1, 2, 2, 8),
    )

    assert output.count_nonzero() == 0
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert reference.grad is not None
    assert actor.grad is not None


def test_relative_camera_pose_is_finite_and_has_expected_shape():
    torch.manual_seed(23)
    conditions = {
        "player_position": torch.randn(2, 3, 4, 3),
        "camera_relative": torch.randn(2, 3, 4, 3),
        "camera_direction": torch.randn(2, 3, 4, 3),
        "yaw_pitch": torch.randn(2, 3, 4, 2),
        "fov_x": torch.rand(2, 3, 4),
    }
    pose = relative_camera_pose(conditions, torch.tensor([1, 3]))
    assert pose.shape == (2, 3, 4, 8)
    assert torch.isfinite(pose).all()


def test_depth_routing_prefers_nearer_player_without_increasing_union_mass():
    model = FixedSpatialReferenceProjector(
        height=1, width=1, gradient_checkpointing=False, depth_sharpness=8
    )
    roi = torch.ones(1, 1, 2, 1, 1)
    depth = torch.tensor([[[[[4.]], [[2.]]]]])
    routing = model._depth_route(roi, depth)

    torch.testing.assert_close(routing.sum(2), torch.ones(1, 1, 1, 1))
    assert float(routing[0, 0, 1, 0, 0]) > .999


def test_fixed_reference_supports_256_output_channels():
    model = FixedSpatialReferenceProjector(
        height=2,
        width=2,
        gradient_checkpointing=False,
        output_channels=256,
    )
    output = model(
        torch.zeros(1, 1, 4, 128, 256),
        torch.ones(1, 1, 1, 2, 2),
        torch.zeros(1, 1, 1, 2, 2, 2),
        torch.randn(1, 1, 1, 30),
        torch.ones(1, 1, 4, dtype=torch.bool),
        torch.zeros(1, 1, 1, 8),
        appearance_only=True,
    )
    assert output.shape == (1, 1, 256, 2, 2)


def test_appearance_only_projector_ignores_pose_and_coordinate_query_channels():
    torch.manual_seed(29)
    model = FixedSpatialReferenceProjector(
        height=2, width=2, gradient_checkpointing=False
    )
    reference = torch.zeros(1, 1, 4, 128, 256)
    roi = torch.ones(1, 1, 1, 2, 2)
    appearance = torch.randn(1, 1, 1, 30)
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    first = model(
        reference, roi, torch.randn(1, 1, 1, 2, 2, 2), appearance,
        valid, torch.randn(1, 1, 1, 8), appearance_only=True,
    )
    second = model(
        reference, roi, torch.randn(1, 1, 1, 2, 2, 2), appearance,
        valid, torch.randn(1, 1, 1, 8), appearance_only=True,
    )
    torch.testing.assert_close(first, second)


def test_other_player_tokens_share_action_state_embedders_and_stay_spatial():
    model = FrameDepthStackPixelDiT(
        input_h=4, input_w=4, hidden_size=8, depth=1, num_heads=2,
        raster_cond_shape=(1, 4, 4), action_cond_dim=2,
        extra_condition_dim=3, actor_condition_dim=2, roi_appearance_dim=2,
        simple_conditioning=True, split_player_conditioning=True,
        other_player_condition_blocks=(0,), depth_patch_size=1,
        depth_stride_size=1, depth_output_dim=1,
    )
    with torch.no_grad():
        model.action_embedder.weight.zero_()
        model.action_embedder.bias.zero_()
        model.action_embedder.weight[:2, :2] = torch.eye(2)
        model.extra_condition_embedder.weight.zero_()
        model.extra_condition_embedder.bias.zero_()
    routing = torch.zeros(1, 1, 2, 4, 4)
    routing[:, :, 0, :, :2] = 1
    routing[:, :, 1, :, 2:] = 1
    tokens = model._spatial_other_player_tokens(
        {
            "other_player_action": torch.tensor([[[[1., 0.], [0., 1.]]]]),
            "other_player_state": torch.zeros(1, 1, 2, 3),
            "other_player_routing": routing,
        },
        (2, 2),
        torch.float32,
    )
    assert tokens.shape == (1, 1, 2, 2, 8)
    assert torch.all(tokens[..., 0, 0] > 0)
    assert torch.all(tokens[..., 1, 0] == 0)
    assert torch.all(tokens[..., 0, 1] == 0)
    assert torch.all(tokens[..., 1, 1] > 0)


def test_other_player_action_only_does_not_read_other_state():
    model = FrameDepthStackPixelDiT(
        input_h=4, input_w=4, hidden_size=8, depth=1, num_heads=2,
        raster_cond_shape=(1, 4, 4), action_cond_dim=2,
        extra_condition_dim=3, actor_condition_dim=2, roi_appearance_dim=2,
        simple_conditioning=True, split_player_conditioning=True,
        other_player_condition_blocks=(0,), other_player_action_only=True,
        depth_patch_size=1, depth_stride_size=1, depth_output_dim=1,
    )
    routing = torch.ones(1, 1, 1, 4, 4)
    conditions = {
        "other_player_action": torch.tensor([[[[1., 0.]]]]),
        "other_player_routing": routing,
    }
    without_state = model._spatial_other_player_tokens(
        conditions, (2, 2), torch.float32
    )
    with_state = model._spatial_other_player_tokens(
        dict(conditions, other_player_state=torch.randn(1, 1, 1, 3)),
        (2, 2), torch.float32,
    )
    torch.testing.assert_close(without_state, with_state)


def test_player_orientation_features_are_absolute_and_periodic():
    angles = torch.tensor([[[[0., 0.], [torch.pi / 2, -torch.pi / 2]]]])
    encoded = player_orientation_features(angles)
    expected = torch.tensor([[[[0., 1., 0., 1.], [1., 0., -1., 0.]]]])
    torch.testing.assert_close(encoded, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        encoded,
        player_orientation_features(angles + 2 * torch.pi),
        atol=1e-6,
        rtol=0,
    )


def test_other_player_action_orientation_only_uses_heading_not_state():
    model = FrameDepthStackPixelDiT(
        input_h=4, input_w=4, hidden_size=8, depth=1, num_heads=2,
        raster_cond_shape=(1, 4, 4), action_cond_dim=2,
        extra_condition_dim=3, actor_condition_dim=2, roi_appearance_dim=2,
        simple_conditioning=True, split_player_conditioning=True,
        other_player_condition_blocks=(0,),
        other_player_action_orientation_only=True,
        depth_patch_size=1, depth_stride_size=1, depth_output_dim=1,
    )
    with torch.no_grad():
        model.action_embedder.weight.zero_()
        model.action_embedder.bias.zero_()
        model.other_player_orientation_embedder.weight.zero_()
    conditions = {
        "other_player_action": torch.zeros(1, 1, 1, 2),
        "other_player_orientation": torch.tensor([[[[1., 0., 0., 1.]]]]),
        "other_player_routing": torch.ones(1, 1, 1, 4, 4),
    }
    zero_initialized = model._spatial_other_player_tokens(
        conditions, (2, 2), torch.float32
    )
    zero_initialized.sum().backward()
    assert model.other_player_orientation_embedder.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.other_player_orientation_embedder.weight[0, 0] = 1
    first = model._spatial_other_player_tokens(
        conditions, (2, 2), torch.float32
    )
    second = model._spatial_other_player_tokens(
        dict(
            conditions,
            other_player_orientation=torch.tensor([[[[0., 1., 0., 1.]]]]),
            other_player_state=torch.randn(1, 1, 1, 3),
        ),
        (2, 2),
        torch.float32,
    )
    assert torch.all(first[..., 0] == 1)
    assert torch.all(second[..., 0] == 0)


def test_independent_256_token_migration_preserves_joint_patch_function():
    torch.manual_seed(37)
    old_patch = torch.randn(8, 36, 2, 2)
    warm_state = {"core.x_embedder.proj.weight": old_patch.clone()}
    current_state = {
        "core.x_embedder.proj.weight": torch.randn_like(old_patch),
        "core.roi_appearance_token_embedder.0.weight": torch.randn(256, 32, 2, 2),
        "core.roi_appearance_token_embedder.1.weight": torch.randn(8, 256, 1, 1),
        "core.roi_appearance_token_gate": torch.tensor(1.),
    }
    migrated = migrate_roi_appearance_fusion(
        warm_state,
        current_state,
        roi_appearance_channels=32,
        independent_tokens=True,
    )
    inputs = torch.randn(2, 36, 6, 6)
    old_output = torch.nn.functional.conv2d(inputs, old_patch, stride=2)
    main_inputs = inputs.clone()
    main_inputs[:, -32:].zero_()
    main_output = torch.nn.functional.conv2d(
        main_inputs, warm_state["core.x_embedder.proj.weight"], stride=2
    )
    appearance_tokens = torch.nn.functional.conv2d(
        inputs[:, -32:],
        warm_state["core.roi_appearance_token_embedder.0.weight"],
        stride=2,
    )
    appearance_output = torch.nn.functional.conv2d(
        appearance_tokens,
        warm_state["core.roi_appearance_token_embedder.1.weight"],
    )
    torch.testing.assert_close(main_output + appearance_output, old_output)
    assert migrated == ["roi_appearance:joint32->independent256"]


def test_256_concat_migration_preserves_old_channels_and_zeros_new_outputs():
    torch.manual_seed(41)
    old_patch = torch.randn(8, 36, 2, 2)
    old_projector_weight = torch.randn(32, 64, 1, 1)
    old_projector_bias = torch.randn(32)
    warm_state = {
        "core.x_embedder.proj.weight": old_patch.clone(),
        "roi_appearance_projector.fuse.2.weight": old_projector_weight.clone(),
        "roi_appearance_projector.fuse.2.bias": old_projector_bias.clone(),
    }
    current_state = {
        "core.x_embedder.proj.weight": torch.randn(8, 260, 2, 2),
        "roi_appearance_projector.fuse.2.weight": torch.randn(256, 64, 1, 1),
        "roi_appearance_projector.fuse.2.bias": torch.randn(256),
    }
    migrate_roi_appearance_fusion(
        warm_state,
        current_state,
        roi_appearance_channels=256,
        independent_tokens=False,
    )
    old_inputs = torch.randn(2, 36, 6, 6)
    new_inputs = torch.cat(
        (old_inputs[:, :4], old_inputs[:, 4:], torch.zeros(2, 224, 6, 6)), dim=1
    )
    old_output = torch.nn.functional.conv2d(old_inputs, old_patch, stride=2)
    new_output = torch.nn.functional.conv2d(
        new_inputs, warm_state["core.x_embedder.proj.weight"], stride=2
    )
    torch.testing.assert_close(new_output, old_output)
    torch.testing.assert_close(
        warm_state["roi_appearance_projector.fuse.2.weight"][:32],
        old_projector_weight,
    )
    assert warm_state["roi_appearance_projector.fuse.2.weight"][32:].count_nonzero() == 0
    assert warm_state["roi_appearance_projector.fuse.2.bias"][32:].count_nonzero() == 0


def test_renderer_fixed_reference_config_round_trip():
    config = RendererArgs(
        10,
        3,
        hidden_size=64,
        depth=1,
        num_heads=4,
        actor_channels=32,
        context_frames=3,
        cache_frames=2,
        block_frames=1,
        gpu_rasterizer=False,
        simple_conditioning=True,
        unified_player_reference=True,
        player_reference_grid_size=(16, 8),
        player_reference_position_encoding=True,
        fixed_spatial_reference=True,
    )
    model = Renderer(config)
    restored = Renderer(RendererArgs(**asdict(config)))
    restored.load_state_dict(model.state_dict(), strict=True)

    assert isinstance(
        model.roi_appearance_projector, FixedSpatialReferenceProjector
    )
    assert not any("rgb_head" in key for key in model.state_dict())
