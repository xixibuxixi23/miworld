import torch
import pytest
from dataclasses import replace
from types import SimpleNamespace
from test_renderer import tiny_model,conditions
from miworld.models.renderer import Renderer
from miworld.training.renderer_trainer import RendererRollout,renderer_flow_loss,slice_conditions
from miworld.training.persist_renderer import PersistRendererRollout

def test_persist_loss_noises_and_scores_first_frame():
 class Spy:
  cfg=SimpleNamespace(context_frames=5,block_frames=1,bidirectional_attention=False,persist_protocol=True)
  core=SimpleNamespace(kv_caches=None,use_condition_mask=False)
  def __call__(self,x,t,c,**kw):
   self.x=x;self.t=t;assert kw['noise_cond']==.015
   return torch.zeros_like(x)
 m=Spy();clean=torch.randn(2,5,2,2,2);g=torch.Generator().manual_seed(9)
 loss=renderer_flow_loss(m,clean,{},generator=g)
 g=torch.Generator().manual_seed(9);tt=torch.randn(2,5,generator=g).sigmoid();eps=torch.randn(clean.shape,generator=g).clamp(-20,20)
 torch.testing.assert_close(m.t,tt)
 torch.testing.assert_close(m.x,(1-tt[...,None,None,None])*clean+(1e-5+(1-1e-5)*tt[...,None,None,None])*eps)
 torch.testing.assert_close(loss,((1-1e-5)*eps-clean).square().mean())
 assert (m.t[:,0]>0).all() and not torch.equal(m.t[:,0],m.t[:,1])
 unit_loss=renderer_flow_loss(m,clean,{},region_weight=torch.ones(2,5,1,2,2),generator=torch.Generator().manual_seed(9))
 torch.testing.assert_close(unit_loss,loss)
 with pytest.raises(ValueError,match='unweighted'):
  renderer_flow_loss(m,clean,{},region_weight=torch.full((2,5,1,2,2),2.0))

def test_refreshed_noisy_kv_matches_full_window_after_sliding():
 torch.manual_seed(10)
 base=tiny_model(simple_conditioning=True,qk_rms_norm=True)
 cfg=replace(base.cfg,persist_protocol=True,block_frames=1,context_frames=5,cache_frames=4)
 m=Renderer(cfg).eval();m.load_state_dict(base.state_dict());cond=conditions(9)
 r=RendererRollout(m,denoising_steps=2);assert isinstance(r,PersistRendererRollout)
 r.start(torch.randn(1,1,16,4,4),slice_conditions(cond,0,1))
 with torch.no_grad():
  for i in range(1,8):
   x=torch.randn(1,1,16,4,4);cc=slice_conditions(cond,i,i+1)
   noisy,tt=r.prefill(history_noise=torch.randn_like(r.history));start=i-r.history.shape[1]
   actual=m(x,torch.full((1,1),.6),cc,global_start_idx=i,cache_write=False)
   m.clear_cache();fullc={k:v if k in {'target_agent','player_skin','player_reference','player_appearance_valid'} else torch.cat((v,cc[k]),1) for k,v in r.conditions.items()}
   expected=m(torch.cat((noisy,x),1),torch.cat((tt,torch.full((1,1),.6)),1),fullc,global_start_idx=start)[:,-1:]
   torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
   out=r.generate(x,cc);assert torch.isfinite(out).all();assert r.history.shape[1]<=4
