import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from miworld.training.renderer_trainer import renderer_flow_loss, renderer_bidirectional_sample

class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, 5, 2, 2, 2))
        self.cfg = SimpleNamespace(context_frames=5, block_frames=2, bidirectional_attention=True, clip_noise_time_shift=1.)
        self.core = SimpleNamespace(kv_caches=None, use_condition_mask=False)
    def forward(self, x, t, c):
        self.seen = (x.detach(), t.detach())
        return self.p.expand_as(x)

for n in (1, 2, 3):
    m = Probe()
    clean = torch.zeros(1,5,2,2,2)
    diag = {}
    with patch('torch.randn', side_effect=lambda shape, **kw: torch.ones(shape, device=kw['device'], dtype=kw['dtype'])):
        loss = renderer_flow_loss(m, clean, {}, reference_latents=n,
            player_region_mask=torch.ones(1,5,1,2,2), diagnostics=diag)
    assert loss.item() == 1.
    assert torch.equal(m.seen[0][:,:n], clean[:,:n])
    assert (m.seen[1][:,:n] == 0).all()
    loss.backward()
    assert (m.p.grad[:,:n] == 0).all()
    assert (m.p.grad[:,n:] != 0).all()
    assert diag['player_flow_loss'].item() == 1.
    with patch('torch.randn', side_effect=lambda shape, **kw: torch.ones(shape, device=kw['device'], dtype=kw['dtype'])):
        weighted = renderer_flow_loss(m, clean, {}, reference_latents=n, region_weight=torch.ones(1,5,1,2,2))
    assert weighted.item() == 1.

class Sampler:
    def __init__(self):
        self.cfg = SimpleNamespace(context_frames=5,bidirectional_attention=True,clip_noise_time_shift=1.)
        self.core = self
        self.kv_caches = None
        self.use_condition_mask = False
    def encode_conditions(self, c): return c
    def __call__(self, x, t, c, **kw):
        assert (x[:,:3] == 7).all() and (t[:,:3] == 0).all()
        return torch.ones_like(x)
m = Sampler()
y = renderer_bidirectional_sample(m, torch.full((1,3,2,2,2),7.), torch.ones(1,2,2,2,2), {}, denoising_steps=4)
assert torch.allclose(y, torch.zeros_like(y))
print('PASS: clean prefixes, future-only MSE/gradients/diagnostics, weighted parity, fixed-prefix Euler sampling')
