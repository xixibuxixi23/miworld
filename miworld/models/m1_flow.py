"""Unified M1: the original multi-view flow DiT plus boundary conditioning.

The legacy image-conditioned path is intentionally kept parameter-compatible
with ``MultiViewVoxelDiT``.  New context parameters are zero-initialized, so a
legacy checkpoint produces the same initialization behavior before finetuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp

from miworld.models.renderer_backbone.attention import MultiHeadRMSNorm
from miworld.models.renderer_backbone.embeddings import (
    AbsolutePositionEmbedder,
    PixelPatchEmbedder,
    TimestepEmbedder,
    VoxelPatchEmbedder,
)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, cross: bool = False):
        super().__init__()
        self.heads = heads
        self.cross = cross
        if cross:
            self.to_q = nn.Linear(width, width, bias=False)
            self.to_kv = nn.Linear(width, width * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(width, width * 3, bias=False)
        self.to_out = nn.Linear(width, width)
        self.q_rms_norm = MultiHeadRMSNorm(width // heads, heads)
        self.k_rms_norm = MultiHeadRMSNorm(width // heads, heads)

    def forward(self, x, context=None, mask=None):
        q, k, v = (
            (self.to_q(x), *self.to_kv(context).chunk(2, -1))
            if self.cross
            else self.to_qkv(x).chunk(3, -1)
        )

        def split(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)

        q, k, v = map(split, (q, k, v))
        attended = F.scaled_dot_product_attention(
            self.q_rms_norm(q),
            self.k_rms_norm(k),
            v,
            attn_mask=None if mask is None else mask[:, None, None],
        )
        return self.to_out(attended.transpose(1, 2).flatten(2))


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        for prefix in ("s", "cross"):
            setattr(
                self,
                prefix + "_norm1",
                nn.LayerNorm(width, elementwise_affine=False, eps=1e-6),
            )
            setattr(
                self,
                prefix + "_norm2",
                nn.LayerNorm(width, elementwise_affine=False, eps=1e-6),
            )
            setattr(self, prefix + "_attn", Attention(width, heads, cross=prefix == "cross"))
            setattr(
                self,
                prefix + "_mlp",
                Mlp(width, width * 4, act_layer=lambda: nn.GELU(approximate="tanh")),
            )
            setattr(
                self,
                prefix + "_adaLN_modulation",
                nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6)),
            )

    def forward(self, x, c, context, mask):
        for prefix in ("s", "cross"):
            shift, scale, gate, shift2, scale2, gate2 = getattr(
                self, prefix + "_adaLN_modulation"
            )(c).chunk(6, -1)
            hidden = getattr(self, prefix + "_norm1")(x) * (1 + scale[:, None]) + shift[:, None]
            hidden = (
                getattr(self, prefix + "_attn")(hidden, context, mask)
                if prefix == "cross"
                else self.s_attn(hidden)
            )
            x = x + gate[:, None] * hidden
            hidden = (
                getattr(self, prefix + "_norm2")(x) * (1 + scale2[:, None]) + shift2[:, None]
            )
            x = x + gate2[:, None] * getattr(self, prefix + "_mlp")(hidden)
        return x


class FinalLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(width, 8 * 48)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, -1)
        return self.linear(self.norm_final(x) * (1 + scale[:, None]) + shift[:, None])


@dataclass(frozen=True)
class UnifiedM1FlowArgs:
    width: int = 1024
    depth: int = 12
    heads: int = 16
    context_classes: int = 256
    context_dim: int = 32

    def build(self) -> "UnifiedM1Flow":
        return UnifiedM1Flow(**vars(self))


class UnifiedM1Flow(nn.Module):
    """One generative M1 for image initialization and boundary continuation.

    ``context_ids`` and ``known_mask`` are optional 48-cubed tensors.  When no
    voxel is known, the new adapter is bypassed exactly.  ``view_valid`` may be
    all false for geometry-only continuation; a learned null token then supplies
    a valid cross-attention context.
    """

    LEGACY_PREFIXES = (
        "x_embedder.",
        "cross_cond_embedder.",
        "t_embedder.",
        "blocks.",
        "final_layer.",
    )
    ADAPTER_PREFIXES = ("context_embedding.", "context_patch.", "null_context")

    def __init__(
        self,
        width: int = 1024,
        depth: int = 12,
        heads: int = 16,
        context_classes: int = 256,
        context_dim: int = 32,
    ):
        super().__init__()
        self.config = dict(
            width=width,
            depth=depth,
            heads=heads,
            context_classes=context_classes,
            context_dim=context_dim,
        )
        self.context_classes = int(context_classes)
        self.x_embedder = VoxelPatchEmbedder(12, 12, 12, 2, 48, width)
        self.cross_cond_embedder = PixelPatchEmbedder(36, 64, 2, 22, width)
        self.t_embedder = TimestepEmbedder(width)
        coordinates = torch.stack(
            torch.meshgrid(*[torch.arange(6)] * 3, indexing="ij"), -1
        ).reshape(-1, 3)
        self.register_buffer(
            "position",
            AbsolutePositionEmbedder(width, in_channels=3, pos_range=2)(coordinates),
            persistent=False,
        )
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.final_layer = FinalLayer(width)

        # One 8^3 raw-voxel patch aligns with one legacy DiT token.  The final
        # projection is zero initialized so old initialization predictions are
        # unchanged at the beginning of unified finetuning.
        self.context_embedding = nn.Embedding(context_classes + 1, context_dim)
        self.context_patch = nn.Conv3d(
            context_dim + 1, width, kernel_size=8, stride=8, bias=True
        )
        nn.init.zeros_(self.context_patch.weight)
        nn.init.zeros_(self.context_patch.bias)
        self.null_context = nn.Parameter(torch.zeros(1, 1, width))

    def adapter_parameters(self):
        yield from self.context_embedding.parameters()
        yield from self.context_patch.parameters()
        yield self.null_context

    def load_legacy_state(self, state: dict[str, torch.Tensor]) -> dict:
        """Strictly load every historical tensor and allow only new adapters."""
        cleaned = {
            key.removeprefix("module.").removeprefix("_orig_mod."): value
            for key, value in state.items()
        }
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        invalid_missing = [
            key for key in missing if not key.startswith(self.ADAPTER_PREFIXES)
        ]
        if invalid_missing or unexpected:
            raise ValueError(
                f"Legacy M1 mismatch: missing={invalid_missing}, unexpected={unexpected}"
            )
        legacy = [key for key in cleaned if key.startswith(self.LEGACY_PREFIXES)]
        return {
            "legacy_tensors": len(legacy),
            "legacy_parameters": sum(cleaned[key].numel() for key in legacy),
            "new_tensors": sorted(missing),
        }

    def _context_tokens(self, context_ids, known_mask):
        if context_ids is None or known_mask is None:
            return None
        if context_ids.shape != known_mask.shape or context_ids.ndim != 4:
            raise ValueError("context_ids and known_mask must have shape [B,48,48,48]")
        if context_ids.shape[-3:] != (48, 48, 48):
            raise ValueError("M1 context must be a 48-cubed tile")
        known = known_mask.bool()
        indices = torch.where(
            known,
            context_ids.clamp(0, self.context_classes - 1),
            torch.full_like(context_ids, self.context_classes),
        )
        embedded = self.context_embedding(indices).movedim(-1, 1)
        values = torch.cat((embedded, known[:, None].to(embedded.dtype)), dim=1)
        tokens = self.context_patch(values).flatten(2).transpose(1, 2)
        # Biases learned for frontier mode must never perturb image-only init.
        active = known.flatten(1).any(1).to(tokens.dtype)[:, None, None]
        return tokens * active

    def forward(
        self,
        x,
        t,
        image_rays,
        view_valid,
        context_ids=None,
        known_mask=None,
    ):
        # image_rays [B,V,22,36,64].  A null token supports image-free frontier
        # completion while remaining masked for ordinary image initialization.
        batch, views = image_rays.shape[:2]
        clean = torch.where(view_valid[:, :, None, None, None], image_rays, 0)
        context = self.cross_cond_embedder(clean.flatten(0, 1)).reshape(
            batch, views * 576, -1
        )
        mask = view_valid[:, :, None].expand(-1, -1, 576).reshape(batch, -1)
        null_valid = ~view_valid.any(-1, keepdim=True)
        context = torch.cat((context, self.null_context.expand(batch, -1, -1)), dim=1)
        mask = torch.cat((mask, null_valid), dim=1)

        hidden = self.x_embedder(x) + self.position[None]
        boundary = self._context_tokens(context_ids, known_mask)
        if boundary is not None:
            hidden = hidden + boundary
        condition = self.t_embedder(t * 1000)
        for block in self.blocks:
            hidden = block(hidden, condition, context, mask)
        hidden = self.final_layer(hidden, condition).reshape(
            batch, 6, 6, 6, 2, 2, 2, 48
        )
        return hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch, 48, 12, 12, 12
        )
