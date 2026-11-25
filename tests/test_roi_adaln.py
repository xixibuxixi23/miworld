import torch
from miworld.models.renderer_backbone.dit_pixel import FrameDepthStackPixelDiT

torch.set_num_threads(2)

def setup():
    torch.manual_seed(83)
    model=FrameDepthStackPixelDiT(
        input_h=4,input_w=4,hidden_size=32,depth=2,num_heads=4,
        raster_cond_shape=(4,4,4),action_cond_dim=23,extra_condition_dim=16,
        actor_condition_dim=4,roi_appearance_dim=32,simple_conditioning=True,
        split_player_conditioning=True,other_player_condition_blocks=(0,1),
        gated_other_player_conditioning=True,other_player_action_orientation_only=True,
        other_player_roi_adaln=True,independent_roi_appearance_tokens=False,
        context_window_size=17,cache_window_size=16,is_causal=True,
        qk_rms_norm=True,gradient_checkpointing=False,aggregation_config=None)
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=.03)
        for block in model.blocks:
            block.s_adaLN_modulation[-1].weight.normal_(std=.03)
            block.t_adaLN_modulation[-1].weight.normal_(std=.03)
    assert model.other_player_condition_blocks == tuple(range(len(model.blocks)))
    n=3
    channels=model.x_embedder.proj.in_channels-16-4-32
    cond=dict(raster_embedding=torch.randn(n,channels,4,4),
        actor_spatial_condition=torch.randn(1,n,4,4,4),
        roi_appearance_condition=torch.randn(1,n,32,4,4),
        action=torch.randn(1,n,23),extra_condition=torch.randn(1,n,16),
        other_player_action=torch.randn(1,n,2,23,requires_grad=True),
        other_player_orientation=torch.randn(1,n,2,4),
        other_player_routing=torch.zeros(1,n,2,4,4))
    cond['other_player_routing'][:,:,0,:2,:2]=1
    return model,cond,torch.randn(1,n,16,4,4),torch.rand(1,n)

def test_roi_adaln_local_condition_and_gradient():
    model,cond,x,t=setup()
    seen=[]
    hook=model.blocks[0].register_forward_pre_hook(lambda module,args: seen.append(args[1].detach().clone()))
    output=model(x,t,external_cond=cond)
    output.square().mean().backward()
    assert torch.isfinite(cond['other_player_action'].grad).all()
    assert cond['other_player_action'].grad[:,:,0].abs().sum()>0
    assert cond['other_player_action'].grad[:,:,1].abs().sum()==0
    with torch.no_grad():
        cond['other_player_action'][:,:,0]+=1
        changed=model(x,t,external_cond=cond)
    hook.remove()
    assert (output-changed).abs().max()>1e-7
    delta=seen[1]-seen[0]
    assert delta[:,:,0,0].abs().sum()>0
    assert delta[:,:,1:,1:].abs().sum()==0

def test_zero_roi_matches_legacy_and_checkpoint_backward():
    model,cond,x,t=setup()
    cond['other_player_routing'].zero_()
    spatial=model(x,t,external_cond=cond)
    model.other_player_roi_adaln=False
    legacy=model(x,t,external_cond=cond)
    torch.testing.assert_close(spatial,legacy,atol=2e-6,rtol=2e-5)
    model.other_player_roi_adaln=True
    model.gradient_checkpointing=True
    checked=model(x,t,external_cond=cond)
    torch.testing.assert_close(checked,spatial)
    checked.square().mean().backward()
    assert torch.isfinite(model.x_embedder.proj.weight.grad).all()
    assert model.x_embedder.proj.weight.grad[:,-32:].abs().sum()>0
