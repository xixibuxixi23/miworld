"""PERSIST b37a211 pixel sampling protocol adapted to Wan latent blocks.

Unlike append-only KV rollout, rebuild the complete sliding history at a
stable noise level before each new latent. No future RGB enters inference.
"""
import torch
from .renderer_trainer import STATIC_KEYS, slice_conditions

class PersistRendererRollout:
    def __init__(self, model, *, denoising_steps=20, time_shift=1.0,
                 stable_t=0.1, sigma_min=1e-5):
        if denoising_steps < 1 or not 0 < time_shift < float('inf'):
            raise ValueError('invalid PERSIST sampling schedule')
        if model.cfg.bidirectional_attention or model.cfg.block_frames != 1:
            raise ValueError('PERSIST sampling requires causal single-latent blocks')
        self.model=model;self.denoising_steps=denoising_steps;self.time_shift=time_shift
        self.stable_t=stable_t;self.sigma_min=sigma_min;self.block_frames=1
        self.history_limit=min(model.cfg.cache_frames,model.cfg.context_frames-1)
        if self.history_limit<1:raise ValueError('PERSIST needs history capacity')
        self.next_frame=None;self.last_policy_features=None;self.last_policy_layers=None

    def _aligned(self, cond, temporal_aligned):
        c=dict(cond) if temporal_aligned else self.model.align_conditions(cond)
        c.pop('condition_mask',None)
        return c

    @torch.no_grad()
    def start(self, first_latent, conditions, *, global_start_idx=0, temporal_aligned=False):
        if first_latent.shape[1]!=1:raise ValueError('start requires one observed latent')
        self.model.eval();self.model.clear_cache()
        self.history=first_latent.detach().clone()
        self.conditions=self._aligned(conditions,temporal_aligned)
        self.next_frame=global_start_idx+1

    @torch.no_grad()
    def prefill(self, *, history_noise=None):
        # One fresh noise draw per generated block, held fixed during its Euler steps.
        h=self.history
        eps=torch.randn_like(h) if history_noise is None else history_noise
        eps=eps.clamp(-20,20)
        noisy=(1-self.stable_t)*h+(self.sigma_min+(1-self.sigma_min)*self.stable_t)*eps
        tt=torch.full(h.shape[:2],self.stable_t,device=h.device)
        start=self.next_frame-h.shape[1]
        self.model.clear_cache()
        _,_,candidates=self.model(noisy,tt,self.conditions,temporal_aligned=True,
            global_start_idx=start,cache_write=False,return_features=True,return_kv_candidates=True)
        self.model.init_kv_cache(len(h),dtype=h.dtype);self.model.set_kv_cache_start(start)
        self.model.commit_kv_candidates(candidates,start)
        return noisy,tt

    @torch.no_grad()
    def generate(self, noise, conditions, *, temporal_aligned=False):
        if self.next_frame is None:raise RuntimeError('call start first')
        if noise.shape[1]!=1:raise ValueError('generate exactly one latent')
        self.prefill()
        cond=self._aligned(conditions,temporal_aligned)
        encoded=self.model.encode_conditions(cond,temporal_aligned=True)
        x=noise.clone().clamp(-20,20)
        grid=torch.linspace(1,0,self.denoising_steps+1,device=x.device)
        grid=self.time_shift*grid/(1+(self.time_shift-1)*grid)
        for i in range(self.denoising_steps):
            v=self.model.core(x,grid[i].expand(x.shape[:2]),encoded,
                global_start_idx=self.next_frame,cache_write=False)
            x=x+(grid[i+1]-grid[i])*v
        # Keep policy consumers functional; this read does not append intermediate KV.
        _,features=self.model.core(x,torch.zeros(x.shape[:2],device=x.device),encoded,
            global_start_idx=self.next_frame,cache_write=False,return_features=True)
        self.last_policy_features=self.model.compress_policy_features(features)
        self.history=torch.cat((self.history,x.detach()),1)[:,-self.history_limit:]
        self.conditions={k:v if k in STATIC_KEYS else torch.cat((v,cond[k]),1)[:,-self.history_limit:]
                         for k,v in self.conditions.items()}
        self.next_frame+=1
        return x
