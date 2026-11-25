"""Structured TextAgent action head, adapted from 2DAction Stage 2."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class StructuredActionHead(nn.Module):
    """Predict grouped keys, one hotbar choice and coarse-to-fine mouse deltas."""

    def __init__(self, hidden_size=1024, horizons=8, mouse_bins_x=None, mouse_bins_y=None,
                 continuous_mouse=False):
        super().__init__()
        self.horizons = horizons
        self.continuous_mouse = bool(continuous_mouse)
        self.mouse_temperature = 4.0
        # Exclude zoom=7 and inventory=11, matching the original 2DAction head.
        self.register_buffer("key_indices", torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 9, 10]))
        self.keys = nn.Linear(hidden_size, horizons * 10)
        self.hotbar = nn.Linear(hidden_size, horizons * 10)  # none + slots 1..9
        default = [-1., -.16, -.08, -.05, -.025, -.016, -.008, -.004,
                   0., .004, .008, .016, .025, .05, .08, .16, 1.]
        self.register_buffer("mouse_bins_x", torch.tensor(mouse_bins_x or default))
        self.register_buffer("mouse_bins_y", torch.tensor(mouse_bins_y or default))
        self.mouse_x = nn.Linear(hidden_size, horizons * len(self.mouse_bins_x))
        self.mouse_y = nn.Linear(hidden_size, horizons * len(self.mouse_bins_y))
        if self.continuous_mouse:
            self.mouse_x_residual = nn.Linear(hidden_size, horizons)
            self.mouse_y_residual = nn.Linear(hidden_size, horizons)
            nn.init.zeros_(self.mouse_x_residual.weight)
            nn.init.zeros_(self.mouse_x_residual.bias)
            nn.init.zeros_(self.mouse_y_residual.weight)
            nn.init.zeros_(self.mouse_y_residual.bias)

    def forward(self, hidden):
        shape = (*hidden.shape[:-1], self.horizons)
        result = {
            "keys": self.keys(hidden).view(*shape, 10),
            "hotbar": self.hotbar(hidden).view(*shape, 10),
            "mouse_x": self.mouse_x(hidden).view(*shape, len(self.mouse_bins_x)),
            "mouse_y": self.mouse_y(hidden).view(*shape, len(self.mouse_bins_y)),
        }
        if self.continuous_mouse:
            result["mouse_x_residual"] = self.mouse_x_residual(hidden).view(*shape)
            result["mouse_y_residual"] = self.mouse_y_residual(hidden).view(*shape)
        return result

    @staticmethod
    def _nearest(values, bins):
        return (values.unsqueeze(-1) - bins.to(values)).abs().argmin(-1)

    def targets(self, actions):
        hotbar_values = actions[..., 12:21]
        hotbar = hotbar_values.argmax(-1) + 1
        hotbar = torch.where(hotbar_values.sum(-1) > 0, hotbar, 0)
        return {
            "keys": actions.index_select(-1, self.key_indices),
            "hotbar": hotbar,
            "mouse_x": self._nearest(actions[..., 21], self.mouse_bins_x),
            "mouse_y": self._nearest(actions[..., 22], self.mouse_bins_y),
        }

    @staticmethod
    def _exclusive_group_loss(logits, active):
        """Categorical none/member loss, ignoring contradictory source labels."""
        count = active.sum(-1)
        target = torch.where(count == 1, active.float().argmax(-1) + 1, 0)
        grouped = torch.cat((torch.zeros_like(logits[..., :1]), logits), -1)
        loss = F.cross_entropy(grouped.flatten(0, -2), target.flatten(),
                               reduction="none").view_as(target)
        return loss * (count <= 1)

    def _structured_key_loss(self, logits, target):
        # TextAgent order is forward/back/left/right/.../dig/place/drop.
        groups = ((0, 1), (2, 3), (7, 8, 9))
        losses = [self._exclusive_group_loss(
            logits[..., list(group)], target[..., list(group)] > .5)
            for group in groups]
        return torch.stack(losses).mean(0)

    def loss(self, logits, actions, valid_mask=None, horizon_weights=None,
             sample_weight=None, mouse_nonzero_weight=1.0, mouse_loss_weight=None):
        if mouse_nonzero_weight < 1:
            raise ValueError("mouse_nonzero_weight must be at least one")
        if mouse_loss_weight is not None and (
                not math.isfinite(mouse_loss_weight) or mouse_loss_weight < 0):
            raise ValueError("mouse_loss_weight must be finite and nonnegative")
        target = self.targets(actions)
        valid = (torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
                 if valid_mask is None else valid_mask.bool())
        keys = F.binary_cross_entropy_with_logits(
            logits["keys"], target["keys"], reduction="none").mean(-1)
        key_groups = self._structured_key_loss(logits["keys"], target["keys"])
        hotbar = F.cross_entropy(logits["hotbar"].flatten(0, -2), target["hotbar"].flatten(),
                                 reduction="none").view_as(valid)
        if self.continuous_mouse:
            mouse_x_ce = F.cross_entropy(
                logits["mouse_x"].flatten(0, -2), target["mouse_x"].flatten(),
                reduction="none").view_as(valid)
            mouse_y_ce = F.cross_entropy(
                logits["mouse_y"].flatten(0, -2), target["mouse_y"].flatten(),
                reduction="none").view_as(valid)
            x = self._continuous_mouse(logits["mouse_x"], logits["mouse_x_residual"],
                                       self.mouse_bins_x)
            y = self._continuous_mouse(logits["mouse_y"], logits["mouse_y_residual"],
                                       self.mouse_bins_y)
            # Classification retains the proven coarse mode; the zero-init
            # residual learns sub-bin corrections without an action-flow model.
            mouse_x = mouse_x_ce + F.smooth_l1_loss(
                x * 50., actions[..., 21].float() * 50., reduction="none")
            mouse_y = mouse_y_ce + F.smooth_l1_loss(
                y * 50., actions[..., 22].float() * 50., reduction="none")
        else:
            mouse_x = F.cross_entropy(logits["mouse_x"].flatten(0, -2), target["mouse_x"].flatten(),
                                      reduction="none").view_as(valid)
            mouse_y = F.cross_entropy(logits["mouse_y"].flatten(0, -2), target["mouse_y"].flatten(),
                                      reduction="none").view_as(valid)
        if mouse_nonzero_weight != 1:
            if self.continuous_mouse:
                nonzero_x = actions[..., 21].abs() > 1e-6
                nonzero_y = actions[..., 22].abs() > 1e-6
            else:
                zero_x = int((self.mouse_bins_x == 0).nonzero(as_tuple=False).item())
                zero_y = int((self.mouse_bins_y == 0).nonzero(as_tuple=False).item())
                nonzero_x = target["mouse_x"] != zero_x
                nonzero_y = target["mouse_y"] != zero_y
            mouse_x = mouse_x * torch.where(nonzero_x, mouse_nonzero_weight, 1.)
            mouse_y = mouse_y * torch.where(nonzero_y, mouse_nonzero_weight, 1.)
        discrete_group = keys + key_groups + hotbar
        mouse_group = (mouse_x + mouse_y) * .5
        # Preserve the legacy objective unless the caller explicitly opts in.
        # The grouped form prevents two mouse axes from each receiving the same
        # top-level weight as the complete discrete-action objective.
        combined = (discrete_group + mouse_x + mouse_y if mouse_loss_weight is None
                    else discrete_group + mouse_loss_weight * mouse_group)
        if horizon_weights is None:
            horizon_weights = torch.ones(self.horizons, device=actions.device)
        weights = valid.to(combined.dtype) * horizon_weights.to(combined)[None]
        denominator = weights.sum(-1)
        per_sample = (combined * weights).sum(-1) / denominator.clamp_min(1e-8)
        sample_valid = (denominator > 0).to(combined.dtype)
        if sample_weight is not None:
            sample_valid = sample_valid * sample_weight.to(combined)
        total = (per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1)
        return total, {"keys": keys, "key_groups": key_groups, "hotbar": hotbar,
                       "mouse_x": mouse_x, "mouse_y": mouse_y}

    @torch.no_grad()
    def decode(self, logits, key_threshold=.5):
        keys = logits["keys"].sigmoid() >= key_threshold
        # Resolve mutually exclusive groups with an explicit none class.
        for group in ((0, 1), (2, 3), (7, 8, 9)):
            grouped = torch.cat((torch.zeros_like(logits["keys"][..., :1]),
                                 logits["keys"][..., list(group)]), -1)
            choice = grouped.argmax(-1)
            keys[..., list(group)] = torch.stack(
                [choice == index for index in range(1, len(group) + 1)], -1)
        hotbar = logits["hotbar"].argmax(-1)
        if self.continuous_mouse:
            mouse_x = self._continuous_mouse(logits["mouse_x"], logits["mouse_x_residual"],
                                               self.mouse_bins_x)
            mouse_y = self._continuous_mouse(logits["mouse_y"], logits["mouse_y_residual"],
                                               self.mouse_bins_y)
        else:
            mouse_x = self.mouse_bins_x[logits["mouse_x"].argmax(-1)]
            mouse_y = self.mouse_bins_y[logits["mouse_y"].argmax(-1)]
        dtype = torch.float32 if self.continuous_mouse else logits["keys"].dtype
        output = torch.zeros((*keys.shape[:-1], 23), device=keys.device, dtype=dtype)
        output[..., self.key_indices] = keys.to(output.dtype)
        for slot in range(1, 10):
            output[..., 11 + slot] = (hotbar == slot).to(output.dtype)
        output[..., 21], output[..., 22] = mouse_x, mouse_y
        return output

    @staticmethod
    def _continuous_mouse(logits, residual, bins):
        index = logits.argmax(-1)
        bins = bins.float()
        center = bins[index]
        left = bins[(index - 1).clamp_min(0)]
        right = bins[(index + 1).clamp_max(len(bins) - 1)]
        negative_scale = (center - left).clamp_min(1e-6) * .5
        positive_scale = (right - center).clamp_min(1e-6) * .5
        normalized = residual.float().tanh()
        scale = torch.where(normalized < 0, negative_scale, positive_scale)
        return (center + normalized * scale).clamp(float(bins[0]), float(bins[-1]))


def constrain_peaceful_logits(logits, peaceful):
    """Peaceful residents may move/turn but cannot edit, drop or select items."""
    if not peaceful.any():
        return logits
    result = {key: value.clone() for key, value in logits.items()}
    # Structured key positions 7,8,9 are dig/place/drop.
    result["keys"][peaceful, :, 7:] = -20.
    result["hotbar"][peaceful] = -20.
    result["hotbar"][peaceful, :, 0] = 20.
    return result
