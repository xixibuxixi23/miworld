"""Small, past-only interaction memory with an exactly zero initial residual."""
import torch
from torch import nn


class InteractionHistoryEncoder(nn.Module):
    def __init__(self, output_hidden, length=64, hidden=256, slots=4):
        super().__init__()
        self.length = length
        self.input = nn.Sequential(nn.Linear(25, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.positions = nn.Parameter(torch.randn(length, hidden) * .02)
        self.queries = nn.Parameter(torch.randn(slots, hidden) * .02)
        self.stats = nn.Linear(7, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(slots*hidden, output_hidden)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values, valid):
        # values [decisions,8,history,25], one causal prefix per NPC time token.
        shape = values.shape[:-2]
        x = values.reshape(-1, self.length, 25).float()
        mask = valid.reshape(-1, self.length).bool()
        # A death/reset/gap invalidates every older slot, not just that slot.
        mask = mask.flip(-1).long().cumprod(-1).bool().flip(-1)
        x = x.masked_fill(~mask[..., None], 0)
        clicks = (x[..., 9] > .5) & mask
        placed = (x[..., 23] > 0) & mask
        pos = torch.arange(self.length, device=x.device)[None]
        last_click = pos.masked_fill(~clicks, -1).amax(-1)
        last_place = pos.masked_fill(~placed, -1).amax(-1)
        stats = torch.stack(((self.length-last_click).clamp_max(self.length)/self.length,
            (self.length-1-last_place).clamp_max(self.length)/self.length,
            (last_click>=0).float(), (last_place>=0).float(),
            clicks.sum(-1)/self.length, x[...,23].sum(-1)/self.length,
            x[...,24].sum(-1)/self.length), -1)
        memory = self.input(x) + self.positions[None]
        memory = memory.masked_fill(~mask[...,None], 0)
        empty = ~mask.any(-1)
        padding = ~mask
        padding = padding.clone(); padding[empty, 0] = False
        query = self.queries[None] + self.stats(stats)[:,None]
        result, _ = self.attention(query, memory, memory, key_padding_mask=padding, need_weights=False)
        result = self.output(self.norm(result).flatten(1))
        return result.masked_fill(empty[:,None], 0).reshape(*shape, -1)
