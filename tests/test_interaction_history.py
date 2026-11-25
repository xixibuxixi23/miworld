from dataclasses import asdict, replace
import copy

import numpy as np
import pytest
import torch

from miworld.data.interaction_history import (build_interaction_history,
    dropout_history_action_blocks, successful_place_frames, resample_interaction_pairs)
from miworld.models.interaction_history import InteractionHistoryEncoder
from miworld.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from train_scripts.train_policy import migrate_history_optimizer
from tests.test_action_chunk_decoder import inputs, activate_decoder
from tests.test_renderer import tiny_model
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID


def test_history_before_visual_context_and_observation_alignment():
    actions = np.zeros((149, 2, 23), np.float32); actions[50,0,9] = 1; actions[80,0,9] = 1
    health = np.ones((150,2))*20; alive = np.ones((150,2),bool)
    terminal = np.zeros(149,bool)
    events = [dict(actor='agent0', event='block_placed',observation_frame=53),
              dict(actor='agent0', event='block_dug',observation_frame=54),
              dict(actor='agent1', event='block_placed',observation_frame=53)]
    x, mask = build_interaction_history(actions,health,alive,terminal,events,0,80,40,64)
    # Index 0 corresponds to observation 17; action 50 becomes visible at 51.
    assert x.shape==(103,25) and x[51-17,9]==1 and x[50-17,9]==0
    assert x[53-17,23]==1 and x[54-17,24]==1
    assert x[80-17,9]==0 and x[81-17,9]==1
    changed=actions.copy(); changed[80:]=77
    y,_=build_interaction_history(changed,health,alive,terminal,events,0,80,40,64)
    torch.testing.assert_close(x[:64],y[:64])
    x,mask=build_interaction_history(actions,health,alive,terminal,events,0,0,20,64)
    assert not mask[:63].any() and not x[:63].any()
    terminal[70]=True
    _,mask=build_interaction_history(actions,health,alive,terminal,events,0,80,40,64)
    assert not mask[71-17] and mask[72-17]


def test_history_action_dropout_uses_execution_chunks_and_preserves_feedback():
    history=torch.arange(20*25,dtype=torch.float32).reshape(20,25)
    valid=torch.ones(20,dtype=torch.bool);valid[9]=False
    unchanged=dropout_history_action_blocks(history,valid,0.)
    assert unchanged.data_ptr()==history.data_ptr()
    dropped=dropout_history_action_blocks(history,valid,1.,block_size=8)
    assert not dropped[valid,:23].any()
    torch.testing.assert_close(dropped[:,23:],history[:,23:])
    torch.testing.assert_close(dropped[~valid,:23],history[~valid,:23])
    with pytest.raises(ValueError,match='probability'):
        dropout_history_action_blocks(history,valid,-.1)


def test_sampling_pairs_keep_labels_masks_and_failed_attempts_in_base():
    target=np.zeros((9,32,23),np.float32); valid=np.ones((9,32),bool)
    indices=torch.stack([torch.arange(a-7,a+1) for a in range(32,104,8)])
    target[3,2,9]=1; target[7,2,9]=1
    actions=np.zeros((150,1,23),np.float32); actions[58,0,9]=1;actions[90,0,9]=1
    events=[dict(actor='agent0',event='block_placed',transition_index=60,observation_frame=61)]
    successes=successful_place_frames(actions,events,0)
    assert successes=={58}
    seen_failed=False
    for seed in range(20):
        out,mask,idx=resample_interaction_pairs(target,valid,indices,0,successes,np.random.RandomState(seed))
        for i in (0,2,4):
            assert idx[i,-1]==56 and out[i,2,9]==1
            assert not out[i+1,:8,9].any()
            assert abs(int(idx[i+1,-1]-idx[i,-1]))<=24
        for j in range(9):
            source=int((idx[j,-1]-32)//8)
            np.testing.assert_array_equal(out[j],target[source]); np.testing.assert_array_equal(mask[j],valid[source])
        seen_failed |= bool((idx[6:,-1]==88).any())
    assert seen_failed
    out,mask,idx=resample_interaction_pairs(target,valid,indices,0,set())
    np.testing.assert_array_equal(out,target);torch.testing.assert_close(idx,indices)


def test_encoder_zero_start_gradient_old_history_reset_and_padding():
    torch.manual_seed(12)
    m=InteractionHistoryEncoder(32,64)
    x=torch.randn(1,8,64,25,requires_grad=True);valid=torch.ones(1,8,64,dtype=torch.bool)
    assert not m(x,valid).any()
    m(x,valid).sum().backward()
    assert m.output.weight.grad.abs().sum()>0
    torch.nn.init.normal_(m.output.weight,std=.02)
    m.zero_grad();x.grad=None
    result=m(x,valid);result.square().sum().backward()
    assert x.grad[:,:,:40].abs().sum()>0
    valid[:,:,30]=False
    old=m(x,valid)
    changed=x.detach().clone();changed[:,:,:31]=10000
    torch.testing.assert_close(old,m(changed,valid))
    assert not m(x,torch.zeros_like(valid)).any()


def cfgs():
    source=InsertedPolicyArgs(num_policy_blocks=2,text_hidden_size=16,horizons=32,
        causal_observations=True,checkpoint_policy=True,current_text_scale=2.,structured_inputs=True,
        builder_spatial_tokens=8,builder_action_decoder_layers=2,builder_action_decoder_hidden=32,
        builder_action_decoder_heads=4,builder_npc_attention=True)
    return source,replace(source,builder_action_history=64)


def history_inputs(device='cpu'):
    latent,cond,family,profile,kw=inputs()
    cond['interaction_history']=torch.randn(1,88,25)
    cond['interaction_history_valid']=torch.ones(1,88,dtype=torch.bool)
    return latent.to(device),{k:v.to(device) for k,v in cond.items()},family.to(device),profile.to(device),{k:v.to(device) for k,v in kw.items()}


def test_migration_preserves_logits_combat_optimizer_and_rejects_schema():
    source_cfg,cfg=cfgs()
    source=InsertedInhabitantPolicy(tiny_model(),source_cfg).eval();activate_decoder(source)
    model=InsertedInhabitantPolicy(tiny_model(),cfg).eval()
    oldopt=torch.optim.AdamW(source.families.parameters(),lr=2e-4)
    for p in source.families.parameters():p.grad=torch.ones_like(p)*.01
    oldopt.step();oldopt.zero_grad()
    result=model.warm_start_builder_history(source.families.state_dict(),asdict(source_cfg))
    opt=torch.optim.AdamW(model.families.parameters(),lr=2e-4)
    count=migrate_history_optimizer(opt,model.families,oldopt.state_dict(),result['source_parameter_names'])
    assert count==len(oldopt.state)
    for name,p in model.families.named_parameters():
        if name.startswith('language_builder.interaction_history.'):
            assert p not in opt.state
        else:
            oldp=dict(source.families.named_parameters())[name]
            for k in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(opt.state[p][k],oldopt.state[oldp][k])
    latent,cond,family,profile,kw=history_inputs()
    with torch.no_grad():
        for fam,pro in [('language_builder','language_builder'),('combat','zombie_melee')]:
            family.fill_(FAMILY_TO_ID[fam]);profile.fill_(PROFILE_TO_ID[pro])
            _,a=source(latent,cond,family,profile,**kw);_,b=model(latent,cond,family,profile,**kw)
            for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    bad=dict(source.families.state_dict());bad.pop('combat.base')
    with pytest.raises(ValueError,match='schema'):model.warm_start_builder_history(bad,asdict(source_cfg))
    with pytest.raises(ValueError,match='settings'):
        model.warm_start_builder_history(source.families.state_dict(),asdict(replace(source_cfg,current_text_scale=1.)))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_dense_history_causality_nonzero_gradients_and_roundtrip(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,cfg=cfgs();model=InsertedInhabitantPolicy(tiny_model(),cfg).to(device).eval()
    activate_decoder(model)
    torch.nn.init.normal_(model.families['language_builder'].interaction_history.output.weight,std=.02)
    latent,cond,family,profile,kw=history_inputs(device)
    with torch.no_grad(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
        _,a=model(latent,cond,family,profile,**kw)
        for i in range(2):
            _,single=model(latent,cond,family,profile,**{**kw,'policy_indices':kw['policy_indices'][:,i]})
            for k in a:torch.testing.assert_close(a[k][:,i],single[k],atol=.003 if device=='cuda' else 3e-5,rtol=.003 if device=='cuda' else 3e-5)
        changed=copy.deepcopy(cond);changed['interaction_history'][:,76:]+=100
        _,b=model(latent,changed,family,profile,**kw)
        for k in a:torch.testing.assert_close(a[k][:,0],b[k][:,0],atol=0,rtol=0)
        changed=copy.deepcopy(cond);changed['interaction_history'][:,:40]+=torch.randn_like(changed['interaction_history'][:,:40])*3
        _,b=model(latent,changed,family,profile,**kw)
        assert not torch.allclose(a['keys'],b['keys'])
    model.train()
    with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
        _,logits=model(latent,cond,family,profile,**kw)
        loss,_=model.loss(logits,torch.zeros(1,2,32,23,device=device))
    loss.backward()
    m=model.families['language_builder'].interaction_history
    assert m.input[0].weight.grad.abs().sum()>0 and torch.isfinite(m.input[0].weight.grad).all()
    assert all(p.grad is None for p in model.renderer.parameters())
    clone=InsertedInhabitantPolicy(tiny_model(),cfg).to(device)
    clone.load_state_dict(model.state_dict(),strict=True)
