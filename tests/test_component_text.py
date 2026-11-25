import copy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from dataset_toolkits.build_m3_component_text import event_timeline, text_at
from miworld.data.inserted_policy_dataset import (
    InsertedPolicyDataset,
    component_text_at_frames,
    mask_unseen_component_targets,
    resample_place_anchors,
    structured_goal_text,
)
from miworld.data.renderer_dataset import TextAgentRendererDataset
from miworld.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from tests.test_renderer import tiny_model, conditions
from train_scripts.train_policy import check_policy_config, check_resume_provenance, check_text_only_index, lookup_current_text


def test_boundaries_use_observation_after_transition_and_clear_completed():
    events = [dict(event='component_started',observation_frame=0,component_text='first'),
              dict(event='component_started',observation_frame=40,transition_index=39,component_text='second'),
              dict(event='task_completed',observation_frame=100,transition_index=99)]
    timeline = event_timeline(events)
    assert component_text_at_frames(timeline,[0,39,40,99,100]) == ['first','first','second','second','']
    assert text_at(timeline,39) == 'first'
    events[1]['observation_frame'] = 39
    with pytest.raises(ValueError): event_timeline(events)


def test_dense_component_boundary_mask_drops_unsafe_prefix_and_truncates_tail():
    timeline = [dict(observation_frame=0, text='first'),
                dict(observation_frame=38, text='second'),
                dict(observation_frame=60, text='third')]
    indices = torch.stack((torch.arange(25, 33), torch.arange(33, 41)))
    valid = np.ones((2, 32), bool)
    masked = mask_unseen_component_targets(valid, indices, timeline, context_start=0)
    # Anchor 32 would execute past the new instruction at frame 38.
    assert not masked[0].any()
    # Anchor 40 keeps labels through 59 and removes the unseen instruction.
    assert masked[1, :20].all() and not masked[1, 20:].any()
    assert valid.all()  # helper does not mutate its caller


def test_place_anchor_resampling_repeats_positive_anchor_without_loss_weights():
    target = np.zeros((9, 32, 23), np.float32)
    valid = np.ones((9, 32), bool)
    indices = torch.stack([torch.arange(i, i + 8) for i in range(9)])
    target[2, 3, 9] = 1
    selected_target, selected_valid, selected_indices = resample_place_anchors(
        target, valid, indices, .5)
    positive = ((selected_target[:, :8, 9] > .5) & selected_valid[:, :8]).any(-1)
    assert positive.sum() == 5
    assert selected_indices[positive, -1].unique().tolist() == [9]
    assert selected_target.shape == target.shape


def test_component_dataset_keeps_main_and_actions_but_returns_timeline(tmp_path,monkeypatch):
    episode=tmp_path/'e'; episode.mkdir()
    arrays=dict(action_continuous=np.zeros((128,1,23),np.float32),cam_pos=np.zeros((129,1,3)),
        language_policy_train_mask=np.ones((128,1),bool),player_health=np.ones((129,1)),
        player_health_valid=np.ones((129,1),bool),termination_flag=np.zeros(128,bool))
    np.savez(episode/'data.npz',**arrays)
    (episode/'manifest.json').write_text(json.dumps({'model_start_observation':0}))
    def read(cls,*args,**kw): return {'rgb':torch.zeros(129,3,2,2),'conditions':{}}
    monkeypatch.setattr(TextAgentRendererDataset,'read_window',classmethod(read))
    row=dict(episode_path='e',agent_slot=0,anchor=32,context_start=0,dense_valid_targets=288,
        family='language_builder',profile='language_builder',sample_weight=1.,shared_task_text='main',
        simple_task_family='floor',simple_task_material='sea_lantern',
        simple_task_dimensions=[3,1,3],
        current_text_schema='component_observation_v1',current_text_timeline=[
            dict(observation_frame=0,text='first'),dict(observation_frame=40,text='second'),
            dict(observation_frame=100,text='')])
    index=tmp_path/'index.jsonl';index.write_text(json.dumps(row)+'\n')
    catalog=tmp_path/'catalog.json'
    goal=structured_goal_text(row)
    catalog.write_text(json.dumps({'texts':[{'text_id':i,'text':t} for i,t in
        enumerate(['main','first','second','',goal])]}))
    kw=dict(index=index,vocabulary='unused',text_catalog=catalog,dataset_root=tmp_path,
            context_frames=129,horizons=32,dense_stride=8)
    old=InsertedPolicyDataset(**kw)[0]
    new=InsertedPolicyDataset(**kw,current_text_mode='components')[0]
    structured=InsertedPolicyDataset(**kw,current_text_mode='structured_goal')[0]
    structured_only=InsertedPolicyDataset(**kw,current_text_mode='structured_goal_only')[0]
    assert new['shared_text_id'].item()==0
    assert new['current_text_id'][[39,40,100]].tolist()==[1,2,3]
    assert new['current_text_valid'][[39,40,100]].tolist()==[True,True,False]
    assert structured['shared_text_id'].item()==0
    assert structured['current_text_id'].item()==4
    assert structured['current_text_valid'].item()
    assert structured_only['shared_text_id'].item()==4
    assert structured_only['current_text_id'].item()==3
    assert not structured_only['current_text_valid'].item()
    torch.testing.assert_close(old['target_actions'],new['target_actions'])
    torch.testing.assert_close(old['valid_mask'],new['valid_mask'])
    torch.testing.assert_close(old['target_actions'],structured['target_actions'])
    torch.testing.assert_close(old['target_actions'],structured_only['target_actions'])
    cache={'encoder_hidden':torch.zeros(4,2,3),'attention_mask':torch.ones(4,2,dtype=torch.uint8)}
    batch={k:new[k][None] for k in ('current_text_id','current_text_valid')}
    _,mask=lookup_current_text(cache,batch,'cpu',torch.float32)
    assert mask[0,99].all() and not mask[0,100].any()


def test_structured_goal_is_canonical_and_rejects_ambiguous_fields():
    row={'simple_task_family':' floor ','simple_task_material':'sea_lantern',
         'simple_task_dimensions':[3,1,3]}
    assert structured_goal_text(row)=='task=floor material=sea_lantern size=3x1x3'
    for bad in ({**row,'simple_task_family':''}, {**row,'simple_task_material':None},
                {**row,'simple_task_dimensions':[3,1]},
                {**row,'simple_task_dimensions':[3,0,3]}):
        with pytest.raises(ValueError):
            structured_goal_text(bad)


def test_dense_timeline_matches_history_and_never_reads_future_text():
    torch.manual_seed(23)
    cfg=InsertedPolicyArgs(num_policy_blocks=2,text_hidden_size=16,horizons=32,
        causal_observations=True,current_text_scale=2.,checkpoint_policy=True)
    model=InsertedInhabitantPolicy(tiny_model(),cfg).eval()
    cond=conditions(25);cond['condition_mask'][:]=True
    latent=torch.randn(1,25,16,4,4)
    indices=torch.stack((torch.arange(5,13),torch.arange(13,21)))[None]
    current=torch.randn(1,25,3,16,requires_grad=True)
    mask=torch.ones(1,25,3,dtype=torch.bool)
    shared=torch.randn(1,3,16)
    args=(torch.tensor([FAMILY_TO_ID['language_builder']]),torch.tensor([PROFILE_TO_ID['language_builder']]))
    kwargs=dict(shared_text=shared,shared_text_mask=mask[:,0],current_text=current,current_text_mask=mask)
    with torch.no_grad():
        _,dense=model(latent,cond,*args,**kwargs,policy_indices=indices)
        for a in range(2):
            manual={**kwargs,'current_text':current[:,indices[0,a]],'current_text_mask':mask[:,indices[0,a]]}
            _,single=model(latent,cond,*args,**manual,policy_indices=indices[:,a])
            for key in dense:torch.testing.assert_close(dense[key][:,a],single[key],atol=2e-5,rtol=2e-5)
        changed=current.detach().clone();changed[:,13:]=torch.randn_like(changed[:,13:])*10
        _,future=model(latent,cond,*args,**{**kwargs,'current_text':changed},policy_indices=indices)
        for key in dense:torch.testing.assert_close(dense[key][:,0],future[key][:,0])
        assert not torch.allclose(dense['keys'][:,1],future['keys'][:,1])
    model.train()
    _,logits=model(latent,cond,*args,**kwargs,policy_indices=indices[:,:1])
    logits['keys'].sum().backward()
    assert current.grad[:,:13].abs().sum()>0
    assert current.grad[:,13:].abs().sum()==0


def test_scale_changes_no_parameters_and_non_language_ignores_all_text():
    torch.manual_seed(24)
    cfg=InsertedPolicyArgs(num_policy_blocks=1,text_hidden_size=16)
    one=InsertedInhabitantPolicy(tiny_model(),cfg).eval()
    two=InsertedInhabitantPolicy(tiny_model(),replace(cfg,current_text_scale=2.)).eval()
    two.load_state_dict(one.state_dict(),strict=True)
    assert list(one.state_dict())==list(two.state_dict())
    cond=conditions(8);text=torch.randn(1,3,16);mask=torch.ones(1,3,dtype=torch.bool)
    latent=torch.randn(1,8,16,4,4)
    kw=dict(shared_text=text,shared_text_mask=mask,current_text=text,current_text_mask=mask)
    with torch.no_grad():
        for family,profile in [('language_builder','language_builder'),('combat','skeleton_swordsman')]:
            args=(torch.tensor([FAMILY_TO_ID[family]]),torch.tensor([PROFILE_TO_ID[profile]]))
            _,a=one(latent,cond,*args,**kw);_,b=two(latent,cond,*args,**kw)
            if family=='combat':
                for k in a:torch.testing.assert_close(a[k],b[k])
            else: assert not torch.allclose(a['keys'],b['keys'])


def test_text_migration_cannot_change_architecture_loss_or_sampling(tmp_path):
    cfg=InsertedPolicyArgs()
    check_policy_config(asdict(cfg),replace(cfg,current_text_scale=2.),text_change=True)
    with pytest.raises(ValueError):check_policy_config(asdict(cfg),replace(cfg,current_text_scale=2.))
    with pytest.raises(ValueError):check_policy_config(asdict(cfg),replace(cfg,horizons=32),text_change=True)
    old={'inputs':{'text_cache':{'sha256':'a'}},'source':{'miworld/models/structured_action.py':'loss'}}
    new=copy.deepcopy(old);new['inputs']['text_cache']['sha256']='b'
    check_resume_provenance(old,new,text_change=True)
    new['source']['miworld/models/structured_action.py']='changed'
    with pytest.raises(ValueError):check_resume_provenance(old,new,text_change=True)
    a,b=tmp_path/'old',tmp_path/'new'
    a.write_text(json.dumps({'sample_weight':.5})+'\n')
    b.write_text(json.dumps({'sample_weight':.5,'current_text_schema':'component_observation_v1','current_text_timeline':[]})+'\n')
    check_text_only_index(a,b)
    b.write_text(json.dumps({'sample_weight':.6})+'\n')
    with pytest.raises(ValueError):check_text_only_index(a,b)
