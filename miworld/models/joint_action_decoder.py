"""Joint action queries with direct observed video memory and shared readouts.

There are three queries for every future step: buttons, hotbar and camera.
Queries interact across steps and groups; none contain target actions. The
renderer remains frozen, but its spatial features bypass the policy-slot
compression and enter a trainable visual projection directly.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .structured_action import StructuredActionHead


class JointActionDecoder(StructuredActionHead):
    def __init__(self, hidden_size, horizons, text_hidden, decoder_hidden=1024,
                 layers=12, heads=16, current_text_scale=1., continuous_mouse=False,
                 visual_grid=8, history_frames=8, checkpoint_layers=False,
                 text_action_film=False, text_action_film_hidden=256,
                 future_latent_channels=0, future_latent_grid=8):
        if (layers < 1 or heads < 1 or decoder_hidden < 1
                or decoder_hidden % heads or visual_grid < 1 or history_frames < 1):
            raise ValueError('joint action decoder needs positive sizes and head-divisible width')
        # A single shared readout for each group, applied at every horizon.
        # Retain the common target/loss/decode schema and bin buffers.
        super().__init__(decoder_hidden, horizons=1, continuous_mouse=continuous_mouse)
        self.horizons = horizons
        self.visual_grid = int(visual_grid)
        self.history_frames = int(history_frames)
        self.checkpoint_layers = bool(checkpoint_layers)
        self.current_text_scale = current_text_scale
        self.policy_projection = nn.Linear(hidden_size, decoder_hidden)
        self.video_projection = nn.Linear(hidden_size, decoder_hidden)
        self.text_projection = nn.Linear(text_hidden, decoder_hidden)
        self.memory_types = nn.Parameter(torch.randn(5, decoder_hidden) * .02)
        self.observation_times = nn.Parameter(torch.randn(history_frames, decoder_hidden) * .02)
        self.horizon_queries = nn.Parameter(torch.randn(horizons, decoder_hidden) * .02)
        self.group_queries = nn.Parameter(torch.randn(3, decoder_hidden) * .02)
        self.memory_norm = nn.LayerNorm(decoder_hidden)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                decoder_hidden, heads, 4 * decoder_hidden, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(decoder_hidden)
        self.text_action_film = bool(text_action_film)
        self.future_latent_channels = int(future_latent_channels)
        self.future_latent_grid = int(future_latent_grid)
        if self.future_latent_channels < 0 or self.future_latent_grid < 1:
            raise ValueError('future latent channels must be nonnegative and grid positive')
        if self.future_latent_channels and horizons < 16:
            raise ValueError('future latent supervision requires at least 16 action queries')
        self.future_latent_head = None
        self.future_latent_action_query_norm = None
        self.future_latent_action_projection = None
        self.future_latent_action_stages = None
        self.future_latent_action_attention = None
        if self.future_latent_channels:
            self.future_latent_head = nn.Sequential(
                nn.LayerNorm(decoder_hidden),
                nn.Linear(decoder_hidden, decoder_hidden),
                nn.GELU(),
                nn.Linear(
                    decoder_hidden,
                    self.future_latent_channels * self.future_latent_grid ** 2,
                ),
            )
            # Predict the visual plan first, then make every action query read
            # that spatial plan.  The attention output projection starts at
            # zero so a migrated checkpoint preserves its parent logits
            # exactly before the first optimizer update.
            self.future_latent_action_query_norm = nn.LayerNorm(decoder_hidden)
            self.future_latent_action_projection = nn.Linear(
                self.future_latent_channels, decoder_hidden)
            self.future_latent_action_stages = nn.Parameter(
                torch.randn(2, decoder_hidden) * .02)
            self.future_latent_action_attention = nn.MultiheadAttention(
                decoder_hidden, heads, dropout=0., batch_first=True)
            nn.init.zeros_(self.future_latent_action_attention.out_proj.weight)
            nn.init.zeros_(self.future_latent_action_attention.out_proj.bias)
        self.task_film = None
        if self.text_action_film:
            if text_action_film_hidden < 1:
                raise ValueError('text action FiLM hidden width must be positive')
            self.task_film = nn.Sequential(
                nn.LayerNorm(text_hidden),
                nn.Linear(text_hidden, text_action_film_hidden),
                nn.GELU(),
                nn.Linear(text_action_film_hidden, 2 * decoder_hidden),
            )
            nn.init.zeros_(self.task_film[-1].weight)
            nn.init.zeros_(self.task_film[-1].bias)

    @staticmethod
    def spatial_positions(height, width, channels, device, dtype):
        """Deterministic normalized XY sinusoidal codes, supporting tiny tests."""
        bands = (channels + 3) // 4
        frequency = torch.exp(torch.arange(bands, device=device).float()
                              * (-math.log(10000.) / max(1, bands - 1)))
        y, x = torch.meshgrid(torch.linspace(-1., 1., height, device=device),
                              torch.linspace(-1., 1., width, device=device), indexing='ij')
        x, y = x[..., None] * frequency * math.pi, y[..., None] * frequency * math.pi
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1)[..., :channels].to(dtype)

    def forward(self, states, valid, shared_text, shared_mask, current_text,
                current_mask, video_memory):
        if states.ndim != 4 or video_memory is None or video_memory.ndim != 5:
            raise ValueError('joint action decoder requires policy slots and observed video [B,T,H,W,D]')
        b, t, slots, _ = states.shape
        if t != self.history_frames or video_memory.shape[:2] != (b, t):
            raise ValueError('joint action decoder observation history mismatch')
        if current_text.ndim == 4:
            current_text, current_mask = current_text[:, -1], current_mask[:, -1]
        valid = valid.bool()
        # Mask before pooling/projection so arbitrary padded values cannot leak.
        visual = video_memory.masked_fill(~valid[:, :, None, None, None], 0)
        vh, vw = min(self.visual_grid, visual.shape[2]), min(self.visual_grid, visual.shape[3])
        visual = F.adaptive_avg_pool2d(
            visual.permute(0, 1, 4, 2, 3).flatten(0, 1), (vh, vw))
        visual = visual.view(b, t, -1, vh, vw).permute(0, 1, 3, 4, 2)
        visual = self.video_projection(visual)
        position = self.spatial_positions(vh, vw, visual.shape[-1], visual.device, visual.dtype)
        visual = (visual + position + self.observation_times[None, :, None, None]
                  + self.memory_types[2])
        projected = self.policy_projection(states.masked_fill(~valid[:, :, None, None], 0))
        policy = projected[:, :, 0] + self.observation_times[None] + self.memory_types[0]
        spatial = (projected[:, :, 1:] + self.observation_times[None, :, None]
                   + self.memory_types[1])
        memories = [policy, spatial.flatten(1, 2), visual.flatten(1, 3),
                    self.text_projection(shared_text) + self.memory_types[3],
                    self.current_text_scale * self.text_projection(current_text) + self.memory_types[4]]
        masks = [valid, valid[:, :, None].expand(b, t, slots - 1).flatten(1),
                 valid[:, :, None].expand(b, t, vh * vw).flatten(1),
                 shared_mask.bool(), current_mask.bool()]
        memory = self.memory_norm(torch.cat([
            value.masked_fill(~mask[..., None], 0) for value, mask in zip(memories, masks)], 1))
        padding = ~torch.cat(masks, 1)
        empty = padding.all(-1)
        if empty.any():
            memory, padding = memory.clone(), padding.clone()
            memory[empty, 0] = 0
            padding[empty, 0] = False
        last = projected[:, -1, 0].masked_fill(~valid[:, -1, None], 0)
        query = (last[:, None, None] + self.horizon_queries[None, :, None]
                 + self.group_queries[None, None]).reshape(b, self.horizons * 3, -1)
        for layer in self.decoder:
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                query = checkpoint(layer, query, memory,
                                   memory_key_padding_mask=padding, use_reentrant=False)
            else:
                query = layer(query, memory, memory_key_padding_mask=padding)
        decoded = self.output_norm(query).reshape(b, self.horizons, 3, -1)
        if self.task_film is not None:
            weight = shared_mask.to(shared_text.dtype)[..., None]
            pooled_goal = ((shared_text * weight).sum(1) /
                           weight.sum(1).clamp_min(1))
            scale, shift = self.task_film(pooled_goal).chunk(2, -1)
            scale = .25 * scale.tanh()
            shift = .25 * shift.tanh()
            decoded = (decoded * (1 + scale[:, None, None]) +
                       shift[:, None, None])
        future_latent = None
        if self.future_latent_head is not None:
            # The two summaries share the same decoded action-query states as
            # the action readouts.  They predict visual change after the
            # executed 8-step prefix and after the full 16-step plan.
            future_queries = torch.stack((
                decoded[:, :8].mean((1, 2)),
                decoded[:, :16].mean((1, 2)),
            ), 1)
            future_latent = self.future_latent_head(future_queries).view(
                b, 2, self.future_latent_channels,
                self.future_latent_grid, self.future_latent_grid)
            latent_tokens = future_latent.permute(0, 1, 3, 4, 2)
            latent_tokens = self.future_latent_action_projection(latent_tokens)
            latent_position = self.spatial_positions(
                self.future_latent_grid, self.future_latent_grid,
                latent_tokens.shape[-1], latent_tokens.device, latent_tokens.dtype)
            latent_tokens = (latent_tokens + latent_position[None, None]
                             + self.future_latent_action_stages[None, :, None, None])
            latent_tokens = latent_tokens.flatten(1, 3)
            action_queries = self.future_latent_action_query_norm(
                decoded.reshape(b, self.horizons * 3, -1))
            # Prefix actions may read only the +8 plan.  Tail actions can use
            # both +8 progress and the final +16 plan.
            plan_size = self.future_latent_grid ** 2
            plan_mask = torch.zeros(
                self.horizons * 3, 2 * plan_size,
                dtype=torch.bool, device=decoded.device)
            plan_mask[:min(8, self.horizons) * 3, plan_size:] = True
            planned, _ = self.future_latent_action_attention(
                action_queries, latent_tokens, latent_tokens,
                attn_mask=plan_mask, need_weights=False)
            decoded = decoded + planned.reshape_as(decoded)
        buttons, hotbar, camera = decoded.unbind(2)
        result = dict(keys=self.keys(buttons), hotbar=self.hotbar(hotbar),
                      mouse_x=self.mouse_x(camera), mouse_y=self.mouse_y(camera))
        if future_latent is not None:
            result['future_latent'] = future_latent
        if self.continuous_mouse:
            result.update(mouse_x_residual=self.mouse_x_residual(camera).squeeze(-1),
                          mouse_y_residual=self.mouse_y_residual(camera).squeeze(-1))
        return result
