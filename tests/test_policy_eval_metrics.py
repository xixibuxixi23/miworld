from types import SimpleNamespace

import torch

from miworld.models.structured_action import StructuredActionHead
from miworld.policy_schema import PROFILE_TO_ID
from train_scripts.train_policy import (
    action_counts,
    format_action_counts,
    format_mouse_axis_counts,
    mouse_axis_counts,
)


def _h16_fixture(*, continuous_mouse=True):
    head = StructuredActionHead(
        hidden_size=4, horizons=16, continuous_mouse=continuous_mouse)
    logits = head(torch.zeros(1, 1, 4))
    for key in logits:
        logits[key] = torch.zeros_like(logits[key])
    logits["hotbar"][..., 0] = 20
    logits["mouse_x"][..., 8] = 20
    logits["mouse_y"][..., 8] = 20
    actions = torch.zeros(1, 1, 16, 23)
    valid = torch.ones(1, 1, 16, dtype=torch.bool)
    family = SimpleNamespace(head=head)
    model = SimpleNamespace(
        families={"language_builder": family},
        decode_execution=lambda value: head.decode(value)[..., :8, :],
    )
    profiles = torch.tensor([PROFILE_TO_ID["language_builder"]])
    return model, logits, actions, valid, profiles


def test_h16_first8_action_metrics_are_populated():
    model, logits, actions, valid, profiles = _h16_fixture()
    summary = format_action_counts(
        action_counts(model, logits, actions, valid, profiles))
    assert summary["all"]["n"] == 8
    assert summary["all"]["noop"]["target"] == 8


def test_continuous_mouse_residual_below_deadzone_is_not_camera_motion():
    model, logits, actions, valid, profiles = _h16_fixture()
    logits["mouse_x_residual"].fill_(0.001)
    action_summary = format_action_counts(
        action_counts(model, logits, actions, valid, profiles))["all"]
    mouse_summary = format_mouse_axis_counts(
        mouse_axis_counts(model, logits, actions, valid, profiles))["all"]
    assert action_summary["camera"]["predicted"] == 0
    assert mouse_summary["predicted_nonzero"] == 0
