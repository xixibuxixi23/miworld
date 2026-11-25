"""Temporal contract for the 12-transition M2 -> 3-latent Wan M3 loop."""
from __future__ import annotations

from dataclasses import dataclass

import torch


PIXEL_FRAMES = 13
TRANSITIONS = 12
GENERATED_LATENTS = 3
WAN_TEMPORAL_STRIDE = 4
ACTION_DIM = 23


@dataclass(frozen=True)
class WanClosedLoopContract:
    """One observed boundary followed by one generated macro block.

    Every resident action is supplied as a condition. During offline training,
    player and NPC actions both come from the recorded trajectory. At inference,
    player actions come from the external controller while NPC actions remain a
    temporary GT oracle in v1. M4 is absent; every future state and RGB value
    remains a prediction target rather than an inference input.
    """

    pixel_frames: int = PIXEL_FRAMES
    transitions: int = TRANSITIONS
    generated_latents: int = GENERATED_LATENTS
    temporal_stride: int = WAN_TEMPORAL_STRIDE
    player_action_source: str = "external_or_recorded"
    npc_action_source: str = "ground_truth"
    use_m4: bool = False

    def __post_init__(self):
        if self.pixel_frames != self.transitions + 1:
            raise ValueError("pixel_frames must contain one boundary plus all transitions")
        if self.transitions != self.generated_latents * self.temporal_stride:
            raise ValueError("M2 transitions must align exactly with generated Wan latents")
        if self.player_action_source != "external_or_recorded":
            raise ValueError("player actions must remain explicit control inputs")
        if self.npc_action_source != "ground_truth" or self.use_m4:
            raise ValueError("the v1 bridge requires GT NPC actions and no M4")

    @property
    def latent_pixel_indices(self):
        """Pixel indices represented by boundary + three causal Wan latents."""
        return tuple(range(0, self.pixel_frames, self.temporal_stride))

    @property
    def generated_pixel_indices(self):
        return self.latent_pixel_indices[1:]

    def validate_actions(self, actions):
        """Validate all resident actions in M2 layout [B,12,A,23]."""
        if not torch.is_tensor(actions) or actions.ndim != 4:
            raise ValueError("resident actions must be a tensor [B,12,A,23]")
        if actions.shape[1] != self.transitions or actions.shape[-1] != ACTION_DIM:
            raise ValueError("resident actions must be [B,12,A,23]")
        return actions

    def validate_npc_actions(self, actions):
        """Compatibility alias; the tensor also includes player action slots."""
        return self.validate_actions(actions)

    def validate_generated_latents(self, latents):
        if not torch.is_tensor(latents) or latents.ndim != 5:
            raise ValueError("generated M3 latents must be [B,3,C,H,W]")
        if latents.shape[1] != self.generated_latents:
            raise ValueError("M3 must generate exactly three latent frames per block")
        return latents

    def decode_block(self, codec, boundary_latent, generated_latents):
        """Decode boundary + generated latents and return only 12 future RGBs."""
        if boundary_latent.ndim != 5 or boundary_latent.shape[1] != 1:
            raise ValueError("boundary latent must be [B,1,C,H,W]")
        self.validate_generated_latents(generated_latents)
        if boundary_latent.shape[0] != generated_latents.shape[0] or (
            boundary_latent.shape[2:] != generated_latents.shape[2:]
        ):
            raise ValueError("boundary and generated latents must share batch/spatial shape")
        all_latents = torch.cat((boundary_latent, generated_latents), dim=1)
        rgb = codec.decode(all_latents)
        if rgb.shape[1] != self.pixel_frames:
            raise RuntimeError(
                f"Wan decode must produce {self.pixel_frames} frames, got {rgb.shape[1]}"
            )
        return rgb[:, 1:]


DEFAULT_WAN_CLOSED_LOOP = WanClosedLoopContract()
