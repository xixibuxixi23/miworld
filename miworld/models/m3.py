"""Unified MIWORLD M3 interface.

M3 contains two cooperating parts:

* ``renderer`` is the former video M3.
* ``policy`` is the former M4 policy inserted into frozen renderer layers.

The historical checkpoints stay separate.  This preserves exact checkpoint
compatibility while presenting one deployment model and one public name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from .renderer import Renderer, RendererArgs


M3Policy = InsertedInhabitantPolicy
M3PolicyArgs = InsertedPolicyArgs


@dataclass(frozen=True)
class M3CheckpointPaths:
    """The two checkpoint artifacts that make up a policy-enabled M3."""

    renderer: Path
    policy: Path | None = None


class M3(nn.Module):
    """One public model for rendering and optional resident policy inference.

    The renderer is frozen by ``M3Policy`` exactly as it was during the
    historical policy training.  ``forward_renderer`` and ``forward_policy``
    remain explicit because their inputs and outputs are intentionally
    different contracts.
    """

    def __init__(
        self,
        renderer: Renderer,
        policy: M3Policy | None = None,
    ) -> None:
        super().__init__()
        if policy is not None and policy.renderer is not renderer:
            raise ValueError("M3 policy must wrap the same renderer instance")
        # Register only one root module.  Registering renderer and a policy that
        # owns the same renderer would duplicate state_dict paths.
        self.model = policy if policy is not None else renderer

    @property
    def renderer(self) -> Renderer:
        if isinstance(self.model, M3Policy):
            return self.model.renderer
        return self.model

    @property
    def policy(self) -> M3Policy | None:
        return self.model if isinstance(self.model, M3Policy) else None

    @property
    def policy_enabled(self) -> bool:
        return self.policy is not None

    @classmethod
    def from_checkpoints(
        cls,
        renderer_checkpoint: str | Path,
        policy_checkpoint: str | Path | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "M3":
        """Load the historical renderer and optional policy checkpoints.

        Renderer checkpoints contain ``model`` plus a renderer config.  Policy
        checkpoints contain only ``families`` and reference the renderer path;
        loading them separately is therefore required and intentional.
        """

        renderer_payload = torch.load(
            renderer_checkpoint,
            map_location=map_location,
            weights_only=False,
        )
        renderer_config: dict[str, Any] = renderer_payload["config"].get(
            "renderer", renderer_payload["config"]
        )
        renderer = Renderer(RendererArgs(**renderer_config))
        renderer.load_state_dict(renderer_payload["model"], strict=True)

        if policy_checkpoint is None:
            return cls(renderer)

        policy_payload = torch.load(
            policy_checkpoint,
            map_location=map_location,
            weights_only=False,
        )
        policy_config = policy_payload.get("config", {})
        policy = M3Policy(renderer, M3PolicyArgs(**policy_config))
        policy.families.load_state_dict(policy_payload["families"], strict=True)
        return cls(renderer, policy)

    def forward_renderer(self, *args: Any, **kwargs: Any) -> Any:
        return self.renderer(*args, **kwargs)

    def forward_policy(self, *args: Any, **kwargs: Any) -> Any:
        if self.policy is None:
            raise RuntimeError("M3 was loaded without a policy checkpoint")
        return self.policy(*args, **kwargs)

