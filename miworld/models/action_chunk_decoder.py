"""Parallel, observation-only action-chunk decoding with legacy readout migration."""
from __future__ import annotations

import torch
from torch import nn

from .structured_action import StructuredActionHead


class ActionChunkDecoder(StructuredActionHead):
    """One interacting query per future action; targets never enter the decoder.

    Keep the old horizon-specific readout tensors, but apply each slice to its
    own decoded state, not all slices to a single state. A zero residual output
    projection preserves the old logits at migration. Other new parameters
    receive gradients once that projection has taken its first update.
    """

    def __init__(self, hidden_size, horizons, text_hidden, decoder_hidden=512,
                 layers=4, heads=8, current_text_scale=1., continuous_mouse=False):
        super().__init__(hidden_size, horizons, continuous_mouse=continuous_mouse)
        if layers < 1 or heads < 1 or decoder_hidden < 1 or decoder_hidden % heads:
            raise ValueError('action decoder needs positive layers and head-divisible width')
        self.memory_projection = nn.Linear(hidden_size, decoder_hidden)
        self.text_projection = nn.Linear(text_hidden, decoder_hidden)
        self.memory_types = nn.Parameter(torch.randn(4, decoder_hidden) * .02)
        self.action_queries = nn.Parameter(torch.randn(horizons, decoder_hidden) * .02)
        self.memory_norm = nn.LayerNorm(decoder_hidden)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                decoder_hidden, heads, 4 * decoder_hidden, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(decoder_hidden)
        self.output_projection = nn.Linear(decoder_hidden, hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.current_text_scale = current_text_scale

    def forward(self, states, valid, shared_text, shared_mask, current_text, current_mask):
        # states: [B, history, 1 + spatial_slots, renderer_hidden].
        policy, spatial = states[:, :, 0], states[:, :, 1:]
        b, t, slots, _ = spatial.shape
        if current_text.ndim == 4:
            current_text, current_mask = current_text[:, -1], current_mask[:, -1]
        memories = [
            self.memory_projection(policy) + self.memory_types[0],
            self.memory_projection(spatial.flatten(1, 2)) + self.memory_types[1],
            self.text_projection(shared_text) + self.memory_types[2],
            self.current_text_scale * self.text_projection(current_text) + self.memory_types[3],
        ]
        masks = [valid.bool(), valid[:, :, None].expand(b, t, slots).reshape(b, t * slots).bool(),
                 shared_mask.bool(), current_mask.bool()]
        # Zero masked values as well as blocking attention to them. This avoids
        # padding values affecting outputs even if they are very large.
        memories = [value.masked_fill(~mask[..., None], 0)
                    for value, mask in zip(memories, masks)]
        memory = self.memory_norm(torch.cat(memories, dim=1))
        padding = ~torch.cat(masks, dim=1)
        # A zero fallback for fully invalid samples avoids all-masked softmax.
        empty = padding.all(-1)
        if empty.any():
            memory, padding = memory.clone(), padding.clone()
            memory[empty, 0] = 0
            padding[empty, 0] = False
        last = policy[:, -1].masked_fill(~valid[:, -1, None].bool(), 0)
        query = self.action_queries[None] + self.memory_projection(last)[:, None]
        for layer in self.decoder:
            # Bidirectional future-query attention is safe: no ground-truth
            # future action, observation, or instruction is supplied.
            query = layer(query, memory, memory_key_padding_mask=padding)
        decoded = last[:, None] + self.output_projection(self.output_norm(query))
        result = {}
        names = ['keys', 'hotbar', 'mouse_x', 'mouse_y']
        if self.continuous_mouse:
            names += ['mouse_x_residual', 'mouse_y_residual']
        for name in names:
            readout = getattr(self, name)
            weight = readout.weight.reshape(self.horizons, -1, decoded.shape[-1])
            bias = readout.bias.reshape(self.horizons, -1)
            value = torch.einsum('bhd,hcd->bhc', decoded, weight) + bias
            result[name] = value.squeeze(-1) if name.endswith('_residual') else value
        return result
