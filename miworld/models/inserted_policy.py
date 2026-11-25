"""Chunked read-only M3 policy tokens inserted into the last frozen M3 DiT blocks."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from miworld.policy_schema import FAMILIES, FAMILY_PROFILES, FAMILY_TO_ID, PROFILES, PROFILE_TO_ID
from .structured_action import StructuredActionHead, constrain_peaceful_logits
from .action_chunk_decoder import ActionChunkDecoder
from .joint_action_decoder import JointActionDecoder
from .npc_input_encoder import NPCInputEncoder
from .interaction_history import InteractionHistoryEncoder


@dataclass(frozen=True)
class InsertedPolicyArgs:
    num_policy_blocks: int = 4
    horizons: int = 8
    history_frames: int = 8
    text_hidden_size: int = 768
    mlp_ratio: float = 4.
    structured_inputs: bool = False
    causal_observations: bool = False
    checkpoint_policy: bool = False
    execution_horizon: int = 8
    current_text_scale: float = 1.
    builder_visual_queries: int = 1
    builder_spatial_tokens: int = 0
    builder_action_decoder_layers: int = 0
    builder_action_decoder_hidden: int = 512
    builder_action_decoder_heads: int = 8
    builder_action_architecture: str = 'legacy'
    builder_action_visual_grid: int = 8
    builder_npc_attention: bool = False
    builder_goal_conditioned_vision: bool = False
    builder_text_only_visual_queries: bool = False
    builder_action_history: int = 0
    continuous_mouse: bool = False
    builder_text_action_film: bool = False
    builder_text_action_film_hidden: int = 256
    builder_future_latent_channels: int = 0
    builder_future_latent_grid: int = 8


class ReadOnlyPolicyBlock(nn.Module):
    """One policy-token block; detached video patches are K/V and never queries."""

    def __init__(self, hidden, heads, text_hidden, mlp_ratio=4., current_text_scale=1.,
                 visual_queries=1):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.visual_queries = int(visual_queries)
        if self.visual_queries < 1:
            raise ValueError('visual queries must be positive')
        self.policy_norm, self.video_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.policy_q = nn.Linear(hidden, hidden)
        self.policy_kv = nn.Linear(hidden, 2 * hidden)
        self.video_kv = nn.Linear(hidden, 2 * hidden)
        self.spatial_out = nn.Linear(hidden, hidden)
        if self.visual_queries > 1:
            offsets = torch.randn(self.visual_queries, hidden) * .02
            offsets[0].zero_()
            self.visual_query_offsets = nn.Parameter(offsets)
            # Start close to the proven single-reader policy while leaving
            # nonzero gradients for the new spatial readers.
            logits = torch.full((self.visual_queries,), -4.)
            logits[0] = 0
            self.visual_query_logits = nn.Parameter(logits)
            self.visual_text_projection = nn.Linear(hidden, hidden, bias=False)
            self.visual_text_gate = nn.Parameter(torch.tensor(0.))
        else:
            self.register_parameter('visual_query_offsets', None)
            self.register_parameter('visual_query_logits', None)
            self.visual_text_projection = None
            self.register_parameter('visual_text_gate', None)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.temporal_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.temporal_out = nn.Linear(hidden, hidden)
        self.text_projection = nn.Linear(text_hidden, hidden)
        self.text_norm = nn.LayerNorm(hidden)
        self.text_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.current_text_gate = nn.Parameter(torch.tensor(1.))
        self.current_text_scale = current_text_scale
        self.mlp_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden * mlp_ratio)),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(int(hidden * mlp_ratio), hidden))

    def _heads(self, value):
        return value.view(*value.shape[:-1], self.heads, self.head_dim).transpose(-3, -2)

    def forward(self, token, video, text, text_mask, valid):
        b, t, d = token.shape
        patches = video.detach().flatten(2, 3)
        projected = self.text_projection(text)
        if projected.ndim == 3:
            projected = projected[:, None].expand(-1, t, -1, -1)
            text_mask = text_mask[:, None].expand(-1, t, -1)
        if projected.ndim != 4 or projected.shape[:2] != (b, t):
            raise ValueError("current text must be [B,L,D] or [B,8,L,D]")
        if self.visual_queries > 1:
            text_weight = text_mask.to(projected.dtype)[..., None]
            pooled_text = ((projected * text_weight).sum(2) /
                           text_weight.sum(2).clamp_min(1))
            query = self.policy_q(self.policy_norm(token))[:, :, None]
            query = query + self.visual_query_offsets[None, None]
            query = query + self.visual_text_gate * self.visual_text_projection(
                pooled_text)[:, :, None]
        else:
            query = self.policy_q(self.policy_norm(token)).unsqueeze(2)
        q = self._heads(query)
        video_k, video_v = self.video_kv(self.video_norm(patches)).chunk(2, -1)
        own_k, own_v = self.policy_kv(self.policy_norm(token)).chunk(2, -1)
        k = self._heads(torch.cat((video_k, own_k.unsqueeze(2)), 2))
        v = self._heads(torch.cat((video_v, own_v.unsqueeze(2)), 2))
        spatial = ((q * self.head_dim ** -.5) @ k.transpose(-1, -2)).softmax(-1) @ v
        spatial = spatial.transpose(-3, -2).reshape(b, t, self.visual_queries, d)
        spatial = self.spatial_out(spatial)
        if self.visual_queries > 1:
            spatial = (spatial * self.visual_query_logits.softmax(0)[None, None, :, None]).sum(2)
        else:
            spatial = spatial[:, :, 0]
        token = token + spatial

        q, k, v = self.temporal_qkv(self.temporal_norm(token)).chunk(3, -1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        causal = torch.ones(t, t, dtype=torch.bool, device=token.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        temporal = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        token = token + self.temporal_out(temporal.transpose(1, 2).reshape(b, t, d))

        text_q = self.text_norm(token).reshape(b * t, 1, d)
        text_kv = projected.flatten(0, 1)
        padding = ~text_mask.flatten(0, 1).bool()
        all_padding = padding.all(-1)
        if all_padding.any():
            padding, text_kv = padding.clone(), text_kv.clone()
            padding[all_padding, 0] = False; text_kv[all_padding, 0] = 0
        attended, _ = self.text_attention(text_q, text_kv, text_kv,
                                           key_padding_mask=padding, need_weights=False)
        attended = attended.masked_fill(all_padding[:, None, None], 0)
        token = token + self.current_text_scale * self.current_text_gate * attended.reshape(b, t, d)
        token = token + self.mlp(self.mlp_norm(token))
        return token * valid[..., None].to(token.dtype)


class PersistentSpatialPolicyBlock(ReadOnlyPolicyBlock):
    """Keep spatial slots across insertion layers, alongside the legacy policy.

    The legacy summary path is retained for weight/output compatibility. The
    separate slots are never pooled together and reach the action decoder.
    Each slot reads only its own completed frame; the decoder reads history.
    """

    @staticmethod
    def position_encoding(height, width, hidden, device, dtype):
        if hidden % 4:
            raise ValueError('spatial positional encoding requires width divisible by four')
        frequency = torch.exp(-math.log(10000.) *
                              torch.arange(hidden // 4, device=device).float() / (hidden // 4))
        y, x = torch.meshgrid(torch.arange(height, device=device),
                              torch.arange(width, device=device), indexing='ij')
        x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1).to(dtype)

    def forward(self, states, video, text, text_mask, valid):
        token = super().forward(states[:, :, 0], video, text, text_mask, valid)
        slots = states[:, :, 1:]
        b, t, count, d = slots.shape
        patches = video.detach().flatten(2, 3)
        positions = self.position_encoding(video.shape[2], video.shape[3], d,
                                           video.device, patches.dtype)
        video_k, video_v = self.video_kv(self.video_norm(patches + positions)).chunk(2, -1)
        query = self.policy_q(self.policy_norm(slots + token[:, :, None]))
        spatial = F.scaled_dot_product_attention(
            self._heads(query), self._heads(video_k), self._heads(video_v))
        slots = slots + self.spatial_out(spatial.transpose(-3, -2).reshape(b, t, count, d))
        projected = self.text_projection(text)
        if projected.ndim == 3:
            projected = projected[:, None].expand(-1, t, -1, -1)
            text_mask = text_mask[:, None].expand(-1, t, -1)
        padding = ~text_mask.flatten(0, 1).bool()
        kv = projected.flatten(0, 1)
        empty = padding.all(-1)
        if empty.any():
            padding, kv = padding.clone(), kv.clone()
            padding[empty, 0] = False
            kv[empty, 0] = 0
        attended, _ = self.text_attention(
            self.text_norm(slots).reshape(b * t, count, d), kv, kv,
            key_padding_mask=padding, need_weights=False)
        attended = attended.masked_fill(empty[:, None, None], 0).reshape(b, t, count, d)
        slots = slots + self.current_text_scale * self.current_text_gate * attended
        slots = slots + self.mlp(self.mlp_norm(slots))
        slots = slots * valid[:, :, None, None].to(slots.dtype)
        return torch.cat((token[:, :, None], slots), dim=2)


class NPCPolicyBlock(ReadOnlyPolicyBlock):
    """A single bank of NPC tokens: history -> vision -> two texts -> FFN.

    Time is causal, but slots at the same completed observation can interact.
    Visual features are read-only. No future actions are embedded here.
    """

    position_encoding = staticmethod(PersistentSpatialPolicyBlock.position_encoding)

    def __init__(self, hidden, heads, text_hidden, mlp_ratio=4., current_text_scale=1.,
                 visual_queries=1, goal_conditioned_vision=False,
                 text_only_visual_queries=False, visual_query_slots=1):
        if visual_queries != 1:
            raise ValueError('NPC tokens replace the pooled multi-query visual reader')
        super().__init__(hidden, heads, text_hidden, mlp_ratio, current_text_scale, 1)
        del self.policy_kv  # NPC history now enters through joint temporal attention.
        self.main_text_gate = nn.Parameter(torch.tensor(1.))
        self.goal_conditioned_vision = bool(goal_conditioned_vision)
        self.text_only_visual_queries = bool(text_only_visual_queries)
        if self.goal_conditioned_vision and self.text_only_visual_queries:
            raise ValueError('choose gated state queries or pure text visual queries, not both')
        if self.goal_conditioned_vision:
            # Exact parent behavior at initialization. The scalar receives
            # gradients through the already-trained text projection immediately.
            self.goal_visual_gate = nn.Parameter(torch.tensor(0.))
        else:
            self.register_parameter('goal_visual_gate', None)
        if self.text_only_visual_queries:
            if visual_query_slots < 1:
                raise ValueError('pure text visual queries require at least one slot')
            self.goal_visual_query_offsets = nn.Parameter(
                torch.randn(visual_query_slots, hidden) * .02)
        else:
            self.register_parameter('goal_visual_query_offsets', None)

    def visual_query(self, states, shared_text, shared_mask):
        """Build visual queries; the pure-text path is deliberately state-independent."""
        b, t, count, _ = states.shape
        if self.text_only_visual_queries:
            if count != self.goal_visual_query_offsets.shape[0]:
                raise ValueError('visual-query slot count does not match policy slots')
            if shared_text.ndim != 3 or shared_mask.shape != shared_text.shape[:-1]:
                raise ValueError('global goal text must be [B,L,D] with a matching mask')
            weight = shared_mask.bool().to(shared_text.dtype)[..., None]
            pooled_goal = ((shared_text * weight).sum(1) /
                           weight.sum(1).clamp_min(1))
            goal = self.text_projection(pooled_goal)
            goal = goal.masked_fill(~shared_mask.bool().any(-1, keepdim=True), 0)
            query = goal[:, None, None] + self.goal_visual_query_offsets[None, None]
            query = self.policy_q(self.policy_norm(query))
            return query.expand(b, t, count, -1)
        query = self.policy_q(self.policy_norm(states))
        if self.goal_conditioned_vision:
            if shared_text.ndim != 3 or shared_mask.shape != shared_text.shape[:-1]:
                raise ValueError('global goal text must be [B,L,D] with a matching mask')
            weight = shared_mask.bool().to(shared_text.dtype)[..., None]
            pooled_goal = ((shared_text * weight).sum(1) /
                           weight.sum(1).clamp_min(1))
            goal = self.text_projection(pooled_goal).to(query.dtype)
            query = query + self.goal_visual_gate.tanh() * goal[:, None, None]
        return query

    def attend_text(self, states, text, mask):
        b, t, count, d = states.shape
        if text.ndim == 3:
            text = text[:, None].expand(-1, t, -1, -1)
            mask = mask[:, None].expand(-1, t, -1)
        if text.ndim != 4 or text.shape[:2] != (b, t) or mask.shape != text.shape[:-1]:
            raise ValueError('NPC text must match batch/history and its padding mask')
        if text.shape[2] == 0:
            return torch.zeros_like(states)
        # Mask before projection as well: padding never supplies a key/value.
        kv = self.text_projection(text.masked_fill(~mask.bool()[..., None], 0)).flatten(0, 1)
        padding = ~mask.bool().flatten(0, 1)
        empty = padding.all(-1)
        if empty.any():
            padding, kv = padding.clone(), kv.clone()
            padding[empty, 0] = False
            kv[empty, 0] = 0
        attended, _ = self.text_attention(
            self.text_norm(states).reshape(b * t, count, d), kv, kv,
            key_padding_mask=padding, need_weights=False)
        return attended.masked_fill(empty[:, None, None], 0).reshape(b, t, count, d)

    def forward(self, states, video, text, text_mask, valid, shared_text, shared_mask):
        b, t, count, d = states.shape
        state_valid = valid.bool()[:, :, None, None]
        states = states.masked_fill(~state_valid, 0)
        flat = states.flatten(1, 2)
        q, k, v = self.temporal_qkv(self.temporal_norm(flat)).chunk(3, -1)
        time = torch.arange(t, device=states.device).repeat_interleave(count)
        causal = time[:, None] >= time[None, :]
        keys_valid = valid.bool().repeat_interleave(count, dim=1)
        allowed = causal[None, None] & keys_valid[:, None, None, :]
        temporal = F.scaled_dot_product_attention(
            self._heads(q), self._heads(k), self._heads(v), attn_mask=allowed)
        states = states + self.temporal_out(
            temporal.transpose(1, 2).reshape(b, t, count, d))

        patches = video.detach().flatten(2, 3)
        patches = patches.masked_fill(~valid.bool()[:, :, None, None], 0)
        positions = self.position_encoding(video.shape[2], video.shape[3], d,
                                           video.device, patches.dtype)
        k, v = self.video_kv(self.video_norm(patches + positions)).chunk(2, -1)
        query = self.visual_query(states, shared_text, shared_mask)
        spatial = F.scaled_dot_product_attention(
            self._heads(query), self._heads(k), self._heads(v))
        states = states + self.spatial_out(spatial.transpose(-3, -2).reshape(b, t, count, d))
        # Gates act on residual updates, not embeddings that LayerNorm can rescale.
        states = states + self.main_text_gate * self.attend_text(states, shared_text, shared_mask)
        states = states + self.current_text_scale * self.current_text_gate * self.attend_text(
            states, text, text_mask)
        states = states + self.mlp(self.mlp_norm(states))
        return states.masked_fill(~state_valid, 0)


class PolicyFamilyBranch(nn.Module):
    def __init__(self, name, hidden, heads, observable_dim, cfg, item_vocab):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(hidden))
        self.temporal = nn.Parameter(torch.randn(cfg.history_frames, hidden) * .02)
        self.input_encoder = (NPCInputEncoder(hidden, heads, item_vocab,
                              len(PROFILES), cfg.text_hidden_size)
                              if cfg.structured_inputs else None)
        if self.input_encoder is None:
            self.observable = nn.Linear(observable_dim, hidden)
            self.action = nn.Linear(23, hidden)
            self.profile = nn.Embedding(len(PROFILES), hidden)
            self.shared_text = nn.Linear(cfg.text_hidden_size, hidden)
            self.shared_gate = nn.Parameter(torch.tensor(.2))
        self.spatial_tokens = cfg.builder_spatial_tokens if name == 'language_builder' else 0
        self.npc_attention = cfg.builder_npc_attention and name == 'language_builder'
        if self.spatial_tokens:
            self.spatial_slots = nn.Parameter(torch.randn(self.spatial_tokens, hidden) * .02)
        block_type = (NPCPolicyBlock if self.npc_attention else
                      PersistentSpatialPolicyBlock if self.spatial_tokens else ReadOnlyPolicyBlock)
        self.blocks = nn.ModuleList([
            block_type(
                hidden, heads, cfg.text_hidden_size, cfg.mlp_ratio,
                cfg.current_text_scale,
                cfg.builder_visual_queries if name == 'language_builder' else 1,
                **({'goal_conditioned_vision': cfg.builder_goal_conditioned_vision}
                   if block_type is NPCPolicyBlock else {}),
                **({'text_only_visual_queries': cfg.builder_text_only_visual_queries,
                    'visual_query_slots': 1 + self.spatial_tokens}
                   if block_type is NPCPolicyBlock else {}))
            for _ in range(cfg.num_policy_blocks)
        ])
        if name == 'language_builder' and cfg.builder_action_architecture == 'joint_v2':
            self.head = JointActionDecoder(
                hidden, cfg.horizons, cfg.text_hidden_size, cfg.builder_action_decoder_hidden,
                cfg.builder_action_decoder_layers, cfg.builder_action_decoder_heads,
                cfg.current_text_scale, cfg.continuous_mouse, cfg.builder_action_visual_grid,
                cfg.history_frames, cfg.checkpoint_policy,
                cfg.builder_text_action_film, cfg.builder_text_action_film_hidden,
                cfg.builder_future_latent_channels, cfg.builder_future_latent_grid)
        else:
            self.head = (ActionChunkDecoder(
                hidden, cfg.horizons, cfg.text_hidden_size, cfg.builder_action_decoder_hidden,
                cfg.builder_action_decoder_layers, cfg.builder_action_decoder_heads,
                cfg.current_text_scale, cfg.continuous_mouse)
                if name == 'language_builder' and cfg.builder_action_decoder_layers
                else StructuredActionHead(hidden, cfg.horizons, continuous_mouse=cfg.continuous_mouse))
        self.interaction_history = (InteractionHistoryEncoder(hidden, cfg.builder_action_history)
            if name == 'language_builder' and cfg.builder_action_history else None)

    def initialize(self, observable, incoming, profile, shared_text, shared_mask,
                   state=None, valid=None, history=None, history_valid=None):
        if self.input_encoder is not None:
            token = self.input_encoder(state, incoming, profile, shared_text,
                                       shared_mask, valid)
            token = (token + self.base + self.temporal[None]) * valid[..., None]
        else:
            weight = shared_mask.to(shared_text.dtype)[..., None]
            pooled = (shared_text * weight).sum(1) / weight.sum(1).clamp_min(1)
            token = (self.base + self.temporal[None] + self.observable(observable)
                     + self.action(incoming) + self.profile(profile)[:, None]
                     + self.shared_gate * self.shared_text(pooled)[:, None])
        if self.interaction_history is not None:
            if history is None or history_valid is None:
                raise ValueError('builder interaction history inputs are required')
            token = (token + self.interaction_history(history, history_valid)) * valid[...,None]
        if self.spatial_tokens:
            slots = token[:, :, None] + self.spatial_slots[None, None]
            slots = slots * valid[:, :, None, None].to(slots.dtype)
            return torch.cat((token[:, :, None], slots), dim=2)
        return token

    def predict(self, states, valid, shared_text, shared_mask, current_text, current_mask,
                video_memory=None):
        if isinstance(self.head, JointActionDecoder):
            if states.ndim == 3:
                states = states[:, :, None]
            return self.head(states, valid, shared_text, shared_mask, current_text,
                             current_mask, video_memory)
        if isinstance(self.head, ActionChunkDecoder):
            if states.ndim == 3:
                states = states[:, :, None]
            return self.head(states, valid, shared_text, shared_mask, current_text, current_mask)
        return self.head(states[:, -1])


class InsertedInhabitantPolicy(nn.Module):
    """Frozen M3 plus family-routed, spatially inserted 8-frame M3 policy branches."""

    def __init__(self, renderer, cfg=InsertedPolicyArgs()):
        super().__init__()
        if cfg.num_policy_blocks == 0:
            cfg = replace(cfg, num_policy_blocks=renderer.core.depth)
        if cfg.history_frames != 8 or cfg.horizons not in (8, 16, 32):
            raise ValueError("M3 policy uses eight completed frames -> 8 or 32 actions")
        if cfg.builder_action_history not in (0, 32, 64):
            raise ValueError('builder action history must be 0,32,64')
        if cfg.builder_action_history and not cfg.builder_npc_attention:
            raise ValueError('interaction history requires the NPC builder')
        if not 1 <= cfg.execution_horizon <= cfg.horizons:
            raise ValueError("execution_horizon must fit the prediction horizon")
        if not math.isfinite(cfg.current_text_scale) or cfg.current_text_scale <= 0:
            raise ValueError('current_text_scale must be finite and positive')
        if cfg.builder_visual_queries < 1:
            raise ValueError('builder_visual_queries must be positive')
        if cfg.builder_spatial_tokens < 0 or cfg.builder_action_decoder_layers < 0:
            raise ValueError('builder spatial tokens and decoder layers must be nonnegative')
        if cfg.builder_spatial_tokens and not cfg.builder_action_decoder_layers:
            raise ValueError('persistent spatial tokens require an action decoder')
        if cfg.builder_action_architecture not in ('legacy', 'joint_v2'):
            raise ValueError('unknown builder action architecture')
        if cfg.builder_action_visual_grid < 1:
            raise ValueError('builder action visual grid must be positive')
        if cfg.builder_action_architecture == 'joint_v2' and not cfg.builder_action_decoder_layers:
            raise ValueError('joint action decoder requires decoder layers')
        if cfg.builder_npc_attention and (not cfg.builder_spatial_tokens or
                                         cfg.builder_visual_queries != 1):
            raise ValueError('NPC attention requires spatial tokens and one reader per token')
        if cfg.builder_goal_conditioned_vision and not cfg.builder_npc_attention:
            raise ValueError('goal-conditioned vision requires NPC attention')
        if cfg.builder_text_only_visual_queries and not cfg.builder_npc_attention:
            raise ValueError('pure text visual queries require NPC attention')
        if cfg.builder_goal_conditioned_vision and cfg.builder_text_only_visual_queries:
            raise ValueError('gated and pure-text visual queries are mutually exclusive')
        if not 1 <= cfg.num_policy_blocks <= renderer.core.depth:
            raise ValueError("num_policy_blocks must fit inside M3 depth")
        self.renderer, self.cfg = renderer, cfg
        self.temporal_downsampling = int(renderer.cfg.temporal_downsampling)
        if self.temporal_downsampling < 1:
            raise ValueError("renderer temporal_downsampling must be positive")
        for parameter in renderer.parameters(): parameter.requires_grad_(False)
        renderer.core.gradient_checkpointing = False
        if cfg.causal_observations:
            # The pretrained renderer permits attention inside generation blocks.
            # Dense observation anchors must not read later frames in those blocks.
            from .renderer_backbone.dit_pixel import PixelTemporalAxialAttention
            for module in renderer.core.modules():
                if isinstance(module, PixelTemporalAxialAttention):
                    module.is_causal = True
                    module.causal_block_size = 1
        hidden, heads = renderer.core.hidden_size, renderer.core.num_heads
        # Legacy resident encoders expose output_dim; the ID-conditioned simple
        # encoder emits the same target width through renderer.condition_dim.
        observable_dim = getattr(
            renderer.resident_encoder, "output_dim", renderer.cfg.condition_dim
        )
        self.families = nn.ModuleDict({
            name: PolicyFamilyBranch(name, hidden, heads, observable_dim, cfg,
                                     renderer.cfg.item_vocab_size) for name in FAMILIES
        })

    def train(self, mode=True):
        super().train(mode); self.renderer.eval(); return self

    @staticmethod
    def _gather_time(value, indices):
        shape = (*indices.shape, *value.shape[2:])
        gather = indices.view(*indices.shape, *([1] * (value.ndim - 2))).expand(shape)
        return value.gather(1, gather)

    def _policy_latent_indices(self, pixel_indices):
        """Map completed pixel frames to the latest fully completed M3 latent."""
        return torch.div(
            pixel_indices, self.temporal_downsampling, rounding_mode="floor"
        )

    def forward(self, latents, conditions, family_id, profile_id, *,
                shared_text, shared_text_mask, current_text, current_text_mask,
                policy_indices=None, video_layers=None):
        b, pixel_frames = conditions["player_position"].shape[:2]
        if policy_indices is None:
            if pixel_frames != self.cfg.history_frames:
                raise ValueError("provide policy_indices when M3 context is longer than eight frames")
            policy_indices = torch.arange(
                pixel_frames, device=family_id.device
            )[None].expand(b, -1)
        policy_indices = policy_indices.long()
        source_b = b
        timeline_text = current_text.ndim == 4 and current_text.shape[1] == pixel_frames
        anchors = 1
        dense = policy_indices.ndim == 3
        if dense:
            if not self.cfg.causal_observations:
                raise ValueError("dense anchors require causal_observations")
            if video_layers is not None:
                raise ValueError("dense anchors require the full causal renderer pass")
            if policy_indices.shape[0] != b:
                raise ValueError("policy_indices batch mismatch")
            anchors = policy_indices.shape[1]
            policy_indices = policy_indices.reshape(b * anchors, -1)
            b *= anchors
            family_id = family_id.repeat_interleave(anchors, 0)
            profile_id = profile_id.repeat_interleave(anchors, 0)
            shared_text = shared_text.repeat_interleave(anchors, 0)
            shared_text_mask = shared_text_mask.repeat_interleave(anchors, 0)
            if not timeline_text:
                current_text = current_text.repeat_interleave(anchors, 0)
                current_text_mask = current_text_mask.repeat_interleave(anchors, 0)

        def gather(value, indices):
            selected = self._gather_time(value, indices.reshape(source_b, -1))
            return selected.reshape(b, self.cfg.history_frames, *value.shape[2:])

        if policy_indices.shape != (b, self.cfg.history_frames):
            raise ValueError("policy_indices must select exactly eight completed frames per sample")
        if (policy_indices < 0).any() or (policy_indices >= pixel_frames).any():
            raise ValueError("policy_indices fall outside the M3 context")
        if not (policy_indices[:, 1:] == policy_indices[:, :-1] + 1).all():
            raise ValueError("M3 policy frames must be consecutive")
        if timeline_text:
            # Select only the eight completed observations for each decision.
            # Text after an anchor cannot influence that anchor's prediction.
            current_text = gather(current_text, policy_indices)
            current_text_mask = gather(current_text_mask, policy_indices)
        for family, profiles in FAMILY_PROFILES.items():
            selected = family_id == family
            if selected.any() and not torch.tensor(
                    sorted(profiles), device=profile_id.device).eq(
                    profile_id[selected, None]).any(-1).all():
                raise ValueError("profile is incompatible with its parameter family")
        # Empty text still has an EOS token in T5. Mask non-language families
        # explicitly so text cannot become an accidental behavior condition.
        language = family_id == FAMILY_TO_ID["language_builder"]
        shared_text_mask = shared_text_mask.bool() & language[:, None]
        language_text = language.view(b, *([1] * (current_text_mask.ndim - 1)))
        current_text_mask = current_text_mask.bool() & language_text
        target = conditions["target_agent"].long()
        with torch.no_grad():
            observable = None
            if not self.cfg.structured_inputs:
                observable, _ = self.renderer.resident_encoder(conditions)
                observable = gather(
                    self.renderer.select(observable, target), policy_indices)
            if video_layers is None:
                visual_conditions = {key: value for key, value in conditions.items()
                                     if not key.startswith('interaction_history')}
                aligned = self.renderer.align_conditions(visual_conditions)
                encoded = self.renderer.encode_conditions(
                    aligned, temporal_aligned=True
                )
                latent_indices = self._policy_latent_indices(policy_indices)
                latent_frames = aligned["player_position"].shape[1]
                if (latent_indices >= latent_frames).any():
                    raise ValueError("policy frames fall outside the aligned M3 context")
            else:
                encoded = None
                latent_indices = self._policy_latent_indices(policy_indices)
        incoming = gather(
            self.renderer.select(conditions["action"], target), policy_indices)
        valid = gather(
            self.renderer.select(conditions["player_valid"], target), policy_indices).bool()
        own_state = None
        if self.cfg.structured_inputs:
            own_state = {key: gather(
                self.renderer.select(conditions[key], target), policy_indices)
                for key in ('yaw_pitch', 'camera_relative', 'hp', 'event_cues',
                            'held_item', 'resident_type', 'player_appearance_id')}
        states, selections, direct_visual_memory = {}, {}, {}
        history = history_valid = None
        if self.cfg.builder_action_history and language.any():
            length = self.cfg.builder_action_history
            values = conditions.get('interaction_history')
            mask = conditions.get('interaction_history_valid')
            if values is None or mask is None or values.shape != (source_b, pixel_frames+length-1, 25) or mask.shape != values.shape[:2]:
                raise ValueError('interaction history must include the past prefix before visual context')
            # Prefix offset L-1 cancels the -(L-1) history start.
            positions = policy_indices[...,None] + torch.arange(length, device=policy_indices.device)
            source = torch.arange(source_b, device=positions.device).repeat_interleave(anchors)
            history = values[source[:,None,None], positions]
            history_valid = mask[source[:,None,None], positions]
        for name, index in FAMILY_TO_ID.items():
            selected = (family_id == index).nonzero().flatten()
            selections[name] = selected
            if selected.numel():
                branch = self.families[name]
                states[name] = branch.initialize(
                    None if observable is None else observable[selected],
                    incoming[selected], profile_id[selected],
                    shared_text[selected], shared_text_mask[selected],
                    None if own_state is None else {k: v[selected] for k, v in own_state.items()},
                    valid[selected],
                    None if history is None or name != 'language_builder' else history[selected],
                    None if history_valid is None or name != 'language_builder' else history_valid[selected])
        first = self.renderer.core.depth - self.cfg.num_policy_blocks

        def consume(block_index, video_hidden):
            if block_index < first: return
            local = block_index - first
            selected_video = gather(video_hidden, latent_indices)
            for name, selected in selections.items():
                if selected.numel():
                    branch = self.families[name]
                    args = (states[name], selected_video[selected], current_text[selected],
                            current_text_mask[selected], valid[selected])
                    if branch.npc_attention:
                        args += (shared_text[selected], shared_text_mask[selected])
                    states[name] = (checkpoint(branch.blocks[local], *args, use_reentrant=False)
                                    if self.cfg.checkpoint_policy and self.training
                                    else branch.blocks[local](*args))
                    if local == self.cfg.num_policy_blocks - 1 and isinstance(branch.head, JointActionDecoder):
                        direct_visual_memory[name] = selected_video[selected].detach()

        # The frozen video forward creates no parameter gradients. Policy blocks
        # receive detached patches through the callback and remain trainable.
        if video_layers is None:
            if latents is None or latents.shape[:2] != (source_b, latent_frames):
                raise ValueError(
                    "latents must cover the complete M3 context after temporal alignment"
                )
            video = self.renderer.core(
                latents,
                torch.zeros((source_b, latent_frames), device=latents.device),
                encoded,
                cache_write=False, block_callback=consume)
        else:
            if len(video_layers) != self.cfg.num_policy_blocks:
                raise ValueError("video_layers must contain the final M3 policy insertion layers")
            video = None
            for local, hidden in enumerate(video_layers):
                # Cached rollout may expose one feature per pixel frame, or the
                # temporally compressed latent block produced by Wan-VAE.
                if hidden.shape[1] == self.cfg.history_frames:
                    selected_hidden = hidden
                elif hidden.shape[1] * self.temporal_downsampling == self.cfg.history_frames:
                    local_indices = torch.div(
                        torch.arange(
                            self.cfg.history_frames, device=hidden.device
                        ),
                        self.temporal_downsampling,
                        rounding_mode="floor",
                    )[None].expand(b, -1)
                    selected_hidden = self._gather_time(hidden, local_indices)
                else:
                    selected_hidden = self._gather_time(hidden, latent_indices)
                for name, selected in selections.items():
                    if selected.numel():
                        branch = self.families[name]
                        args = (
                            states[name], selected_hidden[selected], current_text[selected],
                            current_text_mask[selected], valid[selected])
                        if branch.npc_attention:
                            args += (shared_text[selected], shared_text_mask[selected])
                        states[name] = branch.blocks[local](*args)
                        if local == self.cfg.num_policy_blocks - 1 and isinstance(branch.head, JointActionDecoder):
                            direct_visual_memory[name] = selected_hidden[selected].detach()
        family_logits = {
            name: self.families[name].predict(
                states[name], valid[selected], shared_text[selected], shared_text_mask[selected],
                current_text[selected], current_text_mask[selected], direct_visual_memory.get(name))
            for name, selected in selections.items() if selected.numel()
        }
        # Auxiliary builder-only outputs are allowed.  Other families receive
        # zeros and are masked out by the auxiliary loss in the trainer.
        keys = set().union(*(value.keys() for value in family_logits.values()))
        templates = {key: next(value[key] for value in family_logits.values()
                               if key in value) for key in keys}
        outputs = {key: value.new_zeros((b, *value.shape[1:]))
                   for key, value in templates.items()}
        for name, selected in selections.items():
            if selected.numel():
                logits = family_logits[name]
                for key, value in logits.items(): outputs[key][selected] = value
        peaceful = profile_id == PROFILE_TO_ID["villager_peaceful"]
        outputs = constrain_peaceful_logits(outputs, peaceful)
        if dense:
            outputs = {k: v.reshape(source_b, anchors, *v.shape[1:]) for k, v in outputs.items()}
        return video, outputs

    def decode(self, logits):
        # Every family uses the same output schema/bins.
        return next(iter(self.families.values())).head.decode(logits)

    def decode_execution(self, logits):
        """Execute only the prefix; obtain a fresh observation before replanning."""
        return self.decode(logits)[..., :self.cfg.execution_horizon, :]

    def loss(self, logits, actions, valid_mask=None, horizon_weights=None,
             sample_weight=None, mouse_nonzero_weight=1.0, mouse_loss_weight=None):
        if actions.ndim == 4:
            anchors = actions.shape[1]
            actions = actions.flatten(0, 1)
            logits = {k: v.flatten(0, 1) for k, v in logits.items()}
            if valid_mask is not None: valid_mask = valid_mask.flatten(0, 1)
            if sample_weight is not None: sample_weight = sample_weight.repeat_interleave(anchors)
        return next(iter(self.families.values())).head.loss(
            logits, actions, valid_mask, horizon_weights, sample_weight,
            mouse_nonzero_weight=mouse_nonzero_weight,
            mouse_loss_weight=mouse_loss_weight)

    def warm_start_action_v2(self, state, source_cfg):
        """Strict weights-only initialization: keep perception, replace builder head."""
        source_cfg = InsertedPolicyArgs(**source_cfg)
        if source_cfg.builder_action_architecture != 'legacy':
            raise ValueError('action-v2 warm start source must use legacy architecture')
        if self.cfg.builder_action_architecture != 'joint_v2':
            raise ValueError('action-v2 warm start requires joint_v2 architecture')
        architecture_fields = ('builder_action_architecture', 'builder_action_visual_grid',
                               'builder_action_decoder_layers', 'builder_action_decoder_hidden',
                               'builder_action_decoder_heads')
        if replace(self.cfg, **{key: getattr(source_cfg, key)
                               for key in architecture_fields}) != source_cfg:
            raise ValueError('action-v2 migration must preserve all other policy settings')
        with torch.device('meta'):
            source = nn.ModuleDict({name: PolicyFamilyBranch(
                name, self.renderer.core.hidden_size, self.renderer.core.num_heads,
                getattr(self.renderer.resident_encoder, 'output_dim', self.renderer.cfg.condition_dim),
                source_cfg, self.renderer.cfg.item_vocab_size) for name in self.families})
        expected = source.state_dict()
        if set(state) != set(expected) or any(state[key].shape != expected[key].shape for key in expected):
            raise ValueError('action-v2 source schema mismatch')
        target = self.families.state_dict()
        head_prefix = 'language_builder.head.'
        retained = {key: value for key, value in state.items() if not key.startswith(head_prefix)}
        for key, value in retained.items():
            if value.shape != target[key].shape:
                raise ValueError(f'action-v2 non-head shape mismatch: {key}')
        # Preserve action vocabulary/bin definitions even when replacing readouts.
        for name in ('key_indices', 'mouse_bins_x', 'mouse_bins_y'):
            key = head_prefix + name
            if state[key].shape != target[key].shape:
                raise ValueError(f'action-v2 action schema mismatch: {key}')
            retained[key] = state[key]
        self.families.load_state_dict({**target, **retained}, strict=True)
        return {'retained_tensors': len(retained),
                'initialized_head_tensors': sorted(set(target) - set(retained)),
                'discarded_head_tensors': sorted(set(state) - set(retained)),
                'initialization': 'random joint action decoder; source non-head weights retained',
                'optimizer': 'new; source optimizer incompatible with redesigned head'}

    def warm_start_horizons(self, state, old_horizons=8):
        """Resize horizon tensors and add zero mouse residuals; never migrate optimizer."""
        target = self.families.state_dict()
        unexpected = set(state) - set(target)
        missing = set(target) - set(state)
        residual_suffixes = ('.head.mouse_x_residual.weight', '.head.mouse_x_residual.bias',
                             '.head.mouse_y_residual.weight', '.head.mouse_y_residual.bias')
        if unexpected or any(not key.endswith(residual_suffixes) for key in missing):
            raise ValueError("warm-start state keys do not match policy architecture")
        migrated = {}
        resized = []
        readout_widths = {'keys': 10, 'hotbar': 10, 'mouse_x': 17, 'mouse_y': 17}
        for key, value in state.items():
            if value.shape == target[key].shape:
                migrated[key] = value
            else:
                width = next((size for name, size in readout_widths.items()
                              if f'.head.{name}.' in '.' + key), None)
                if key.endswith('.head.action_queries'):
                    width = 1
                if width is None or value.shape[0] != old_horizons * width:
                    raise ValueError(f"unexpected warm-start shape mismatch: {key}")
                shaped = value.reshape(old_horizons, width, *value.shape[1:])
                repeats = (self.cfg.horizons + old_horizons - 1) // old_horizons
                shaped = shaped.repeat(repeats, 1, *([1] * (value.ndim - 1)))
                migrated[key] = shaped[:self.cfg.horizons].reshape_as(target[key])
                resized.append(key)
        for key in missing:
            migrated[key] = target[key]
        self.families.load_state_dict(migrated, strict=True)
        return resized + sorted(missing)

    def warm_start_builder_history(self, state, source_cfg):
        """Add only a zero-residual history encoder; preserve all old tensors."""
        source_cfg = InsertedPolicyArgs(**source_cfg)
        if not source_cfg.builder_npc_attention or source_cfg.builder_action_history:
            raise ValueError('history migration requires an NPC source without long history')
        if not self.cfg.builder_action_history or replace(self.cfg, builder_action_history=0) != source_cfg:
            raise ValueError('history migration must preserve all other policy settings')
        with torch.device('meta'):
            source = nn.ModuleDict({name: PolicyFamilyBranch(name, self.renderer.core.hidden_size,
                self.renderer.core.num_heads, getattr(self.renderer.resident_encoder,
                    'output_dim', self.renderer.cfg.condition_dim), source_cfg,
                self.renderer.cfg.item_vocab_size) for name in self.families})
        expected = source.state_dict()
        target = self.families.state_dict()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError('history migration source schema mismatch')
        added = set(target) - set(state)
        if not added or any(not k.startswith('language_builder.interaction_history.') for k in added):
            raise ValueError('unexpected history migration parameters')
        self.families.load_state_dict({**target, **state}, strict=True)
        return dict(added=sorted(added), source_parameter_names=[name for name, _ in source.named_parameters()])

    def warm_start_goal_conditioned_vision(self, state, source_cfg):
        """Add zero-start global-goal gates without changing parent logits."""
        source_cfg = InsertedPolicyArgs(**source_cfg)
        if source_cfg.builder_goal_conditioned_vision:
            raise ValueError('goal-conditioned vision source must be the ungated parent')
        if (not self.cfg.builder_goal_conditioned_vision or
                replace(self.cfg, builder_goal_conditioned_vision=False) != source_cfg):
            raise ValueError('goal-conditioned vision migration must change only its gate flag')
        target = self.families.state_dict()
        added = set(target) - set(state)
        expected = {f'language_builder.blocks.{index}.goal_visual_gate'
                    for index in range(self.cfg.num_policy_blocks)}
        if added != expected or set(state) - set(target):
            raise ValueError('goal-conditioned vision source schema mismatch')
        for key, value in state.items():
            if value.shape != target[key].shape:
                raise ValueError(f'goal-conditioned vision shape mismatch: {key}')
        incompatible = self.families.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise ValueError('goal-conditioned vision migration did not match expected gates')
        if any(target[key].detach().item() != 0 for key in expected):
            raise ValueError('goal-conditioned vision gates must start at zero')
        return sorted(expected)

    def warm_start_text_only_visual_queries(self, state, source_cfg):
        """Replace weak gated state queries with explicit task-text visual queries."""
        source_cfg = InsertedPolicyArgs(**source_cfg)
        expected_source = replace(
            self.cfg, builder_goal_conditioned_vision=True,
            builder_text_only_visual_queries=False)
        if (not source_cfg.builder_goal_conditioned_vision
                or source_cfg.builder_text_only_visual_queries
                or not self.cfg.builder_text_only_visual_queries
                or self.cfg.builder_goal_conditioned_vision
                or source_cfg != expected_source):
            raise ValueError('pure-text visual-query migration must replace only gated queries')
        target = self.families.state_dict()
        added = set(target) - set(state)
        removed = set(state) - set(target)
        expected_added = {
            f'language_builder.blocks.{index}.goal_visual_query_offsets'
            for index in range(self.cfg.num_policy_blocks)
        }
        expected_removed = {
            f'language_builder.blocks.{index}.goal_visual_gate'
            for index in range(self.cfg.num_policy_blocks)
        }
        if added != expected_added or removed != expected_removed:
            raise ValueError('unexpected pure-text visual-query parameter migration')
        retained = {key: value for key, value in state.items() if key in target}
        for key, value in retained.items():
            if value.shape != target[key].shape:
                raise ValueError(f'pure-text visual-query shape mismatch: {key}')
        self.families.load_state_dict({**target, **retained}, strict=True)
        return {
            'added': sorted(added),
            'removed': sorted(removed),
            'query_source': 'structured shared text plus learned slot offsets; no state token',
        }

    def warm_start_builder_vision(self, state):
        """Load an old one-query policy and initialize new builder visual readers."""
        target = self.families.state_dict()
        extra = set(target) - set(state)
        expected_suffixes = (
            'visual_query_offsets', 'visual_query_logits',
            'visual_text_projection.weight', 'visual_text_gate')
        if set(state) - set(target):
            raise ValueError('builder-vision warm start has unexpected source parameters')
        if any(not (key.startswith('language_builder.blocks.') and
                    key.endswith(expected_suffixes)) for key in extra):
            raise ValueError(f'unexpected builder-vision parameters: {sorted(extra)}')
        for key, value in state.items():
            if value.shape != target[key].shape:
                raise ValueError(f'builder-vision warm-start shape mismatch: {key}')
        incompatible = self.families.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != extra or incompatible.unexpected_keys:
            raise ValueError('builder-vision warm start did not match expected new parameters')
        return sorted(extra)

    def warm_start_builder_decoder(self, state, source_cfg):
        """Strict weights-only migration; all old tensors must still match exactly."""
        source_cfg = InsertedPolicyArgs(**source_cfg)
        if source_cfg.builder_spatial_tokens or source_cfg.builder_action_decoder_layers:
            raise ValueError('decoder warm start source must use the legacy action head')
        if not self.cfg.builder_action_decoder_layers:
            raise ValueError('decoder warm start requires the new action decoder')
        architecture_fields = ('builder_spatial_tokens', 'builder_action_decoder_layers',
                               'builder_action_decoder_hidden', 'builder_action_decoder_heads')
        if replace(self.cfg, **{key: getattr(source_cfg, key) for key in architecture_fields}) != source_cfg:
            raise ValueError('decoder migration must preserve all other policy settings')
        target = self.families.state_dict()
        unexpected = set(state) - set(target)
        extra = set(target) - set(state)
        head_prefixes = ('memory_projection.', 'text_projection.', 'memory_types',
                         'action_queries', 'memory_norm.', 'decoder.', 'output_norm.',
                         'output_projection.')
        allowed = {key for key in extra if key == 'language_builder.spatial_slots' or
                   (key.startswith('language_builder.head.') and
                    key.removeprefix('language_builder.head.').startswith(head_prefixes))}
        if unexpected or extra != allowed:
            raise ValueError(f'decoder migration has unexpected keys: {sorted(unexpected | (extra - allowed))}')
        for key, value in state.items():
            if value.shape != target[key].shape:
                raise ValueError(f'decoder migration shape mismatch: {key}')
        self.families.load_state_dict({**target, **state}, strict=True)
        return sorted(extra)

    def warm_start_builder_npc(self, state, source_cfg):
        """Explicit weights-only migration from persistent slots; outputs change."""
        source_cfg = InsertedPolicyArgs(**source_cfg)
        if source_cfg.builder_npc_attention or not source_cfg.builder_spatial_tokens or not source_cfg.builder_action_decoder_layers:
            raise ValueError('NPC migration requires a persistent-spatial decoder source')
        if not self.cfg.builder_npc_attention:
            raise ValueError('NPC migration requires builder_npc_attention')
        if replace(self.cfg, builder_npc_attention=False,
                   builder_visual_queries=source_cfg.builder_visual_queries) != source_cfg:
            raise ValueError('NPC migration must preserve all other policy settings')
        # Build the exact source schema too: a missing obsolete tensor must not
        # silently make a corrupt source look like a supported migration.
        with torch.device('meta'):
            source = PolicyFamilyBranch('language_builder', self.renderer.core.hidden_size,
                self.renderer.core.num_heads, getattr(self.renderer.resident_encoder,
                    'output_dim', self.renderer.cfg.condition_dim), source_cfg,
                self.renderer.cfg.item_vocab_size)
        target = self.families.state_dict()
        expected = {key: value for key, value in target.items()
                    if not key.startswith('language_builder.')}
        expected.update({'language_builder.' + key: value for key, value in source.state_dict().items()})
        if set(state) != set(expected):
            raise ValueError('NPC migration source keys do not match its declared architecture')
        for key, value in state.items():
            if value.shape != expected[key].shape:
                raise ValueError(f'NPC migration shape mismatch: {key}')
        added, removed = set(target) - set(state), set(state) - set(target)
        if any(not key.endswith('.main_text_gate') for key in added):
            raise ValueError('unexpected new NPC parameters')
        migrated = {**target, **{key: value for key, value in state.items() if key in target}}
        self.families.load_state_dict(migrated, strict=True)
        return {'added': sorted(added), 'removed': sorted(removed)}
