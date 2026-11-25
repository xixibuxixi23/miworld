"""Trainable, field-specific NPC inputs. Does not alter the video encoder."""
import torch
from torch import nn


def mlp(size, hidden):
    return nn.Sequential(nn.Linear(size, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.LayerNorm(hidden))


class NPCInputEncoder(nn.Module):
    def __init__(self, hidden, heads, item_vocab, profiles, text_hidden):
        super().__init__()
        self.pose = mlp(7, hidden)
        self.health = mlp(1, hidden)
        self.events = mlp(4, hidden)
        self.item = nn.Embedding(item_vocab, hidden)
        self.kind = nn.Embedding(4, hidden)
        self.appearance = nn.Embedding(5, hidden)
        self.profile = nn.Embedding(profiles, hidden)
        self.keys = mlp(12, hidden)
        self.hotbar = mlp(9, hidden)
        self.mouse = mlp(2, hidden)
        self.task = mlp(text_hidden, hidden)
        self.type_embedding = nn.Parameter(torch.randn(11, hidden) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * .02)
        self.input_norm = nn.LayerNorm(hidden)
        self.fusion = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, state, incoming, profile, text, text_mask, valid):
        b, t = incoming.shape[:2]
        angles = state['yaw_pitch'].float()
        # Same position/angle convention as the existing renderer; avoid
        # changing world origin or introducing sequence-dependent statistics.
        relative = state['camera_relative'].float()
        relative = relative.sign() * relative.abs().log1p()
        pose = torch.cat((relative, angles.sin(), angles.cos()), -1)
        weights = text_mask.to(text.dtype)[..., None]
        task = (text * weights).sum(1) / weights.sum(1).clamp_min(1)
        tokens = torch.stack((
            self.pose(pose), self.health(state['hp'].float()[..., None] / 20),
            self.events(state['event_cues'].float()),
            self.item(state['held_item'].long()),
            self.kind(state['resident_type'].long()),
            self.appearance(state['player_appearance_id'].long()),
            self.profile(profile)[:, None].expand(-1, t, -1),
            self.keys(incoming[..., :12].float()),
            self.hotbar(incoming[..., 12:21].float()),
            self.mouse(incoming[..., 21:23].float()),
            self.task(task)[:, None].expand(-1, t, -1)), dim=2)
        padding = torch.zeros(b, t, 11, dtype=torch.bool, device=incoming.device)
        padding[..., -1] = ~text_mask.any(-1)[:, None]
        tokens = self.input_norm(tokens + self.type_embedding).flatten(0, 1)
        fused, _ = self.fusion(self.query.expand(b * t, -1, -1), tokens,
                               tokens, key_padding_mask=padding.flatten(0, 1),
                               need_weights=False)
        return self.output_norm(fused[:, 0].reshape(b, t, -1)) * valid[..., None]
