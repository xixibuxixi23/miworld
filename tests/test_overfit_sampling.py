import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dataset_toolkits.build_m3_overfit_index import (
    MASS, MASS_THREE_ROLES, PLACE_BOOST_MASS_THREE_ROLES, audit, classify,
    set_place_boost_weights, set_weights,
)
from miworld.data.inserted_policy_dataset import InsertedPolicyDataset
from miworld.models.structured_action import StructuredActionHead
from miworld.policy_schema import PROFILE_TO_ID
from train_scripts.train_policy import (
    DistributedWeightedSampler, action_counts, check_resume_provenance,
    format_action_counts, format_mouse_axis_counts, mouse_axis_counts,
)


def test_stratum_uses_only_valid_executable_first_anchor():
    actions = np.zeros((9, 32, 23), np.float32); valid = np.ones((9, 32), bool)
    actions[0, 8, 9] = 1; actions[1, 0, 9] = 1
    assert classify(actions, valid, 'language_builder') == 'background'
    actions[0, 1, 0] = 1
    assert classify(actions, valid, 'language_builder') == 'prepare'
    actions[0, 2, 9] = 1
    assert classify(actions, valid, 'language_builder') == 'event'
    valid[0, 2] = False
    assert classify(actions, valid, 'language_builder') == 'prepare'
    valid[0] = False
    assert classify(actions, valid, 'language_builder') is None


def test_role_stratum_mass_and_equal_episode_contribution():
    rows = []
    for p in MASS:
        for s in MASS[p]:
            for ep, count in [('short', 1), ('long', 7)]:
                rows.extend(dict(profile=p, sampling_stratum=s, episode_id=ep) for _ in range(count))
    set_weights(rows)
    for p in MASS:
        for s in MASS[p]:
            for ep in ('short', 'long'):
                assert sum(r['sample_weight'] for r in rows if r['profile']==p and
                           r['sampling_stratum']==s and r['episode_id']==ep) == pytest.approx(MASS[p][s]/2)
    weights = [r['sample_weight'] for r in rows]
    orders = [list(DistributedWeightedSampler(weights, 8, rank, seed=5)) for rank in range(8)]
    assert all(len(x) == 6 for x in orders)


def test_index_weights_preserved_only_when_explicit(tmp_path):
    index, catalog = tmp_path/'index.jsonl', tmp_path/'text.json'
    rows = [dict(episode_path='e', agent_slot=0, anchor=i, context_start=i,
                 profile='language_builder', sample_weight=w) for i,w in enumerate([.1,.9])]
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    catalog.write_text(json.dumps({'texts': [{'text':'', 'text_id':0}]}))
    kwargs = dict(index=index, vocabulary='unused', text_catalog=catalog, context_frames=129,
                  horizons=32, dense_stride=8)
    assert [r['sample_weight'] for r in InsertedPolicyDataset(**kwargs).rows] == [1,1]
    assert [r['sample_weight'] for r in InsertedPolicyDataset(**kwargs, preserve_index_weights=True).rows] == [.1,.9]
    rows[0]['sample_weight'] = 0
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): InsertedPolicyDataset(**kwargs, preserve_index_weights=True)


def test_three_role_mass_and_skeleton_executable_actions():
    rows = [dict(profile=p, sampling_stratum=s, episode_id=f'{p}-{i}')
            for p in MASS_THREE_ROLES for s in MASS_THREE_ROLES[p] for i in range(3)]
    set_weights(rows, MASS_THREE_ROLES)
    for profile, total in [('language_builder', .6), ('zombie_melee', .2), ('skeleton_swordsman', .2)]:
        assert sum(r['sample_weight'] for r in rows if r['profile'] == profile) == pytest.approx(total)
    with pytest.raises(ValueError): set_weights(rows)
    actions = np.zeros((9, 32, 23), np.float32); valid = np.ones((9, 32), bool)
    actions[0, 1, 9] = 1
    assert classify(actions, valid, 'skeleton_swordsman') == 'background'
    actions[0, 1, 0] = 1
    assert classify(actions, valid, 'skeleton_swordsman') == 'prepare'
    actions[0, 1, 8] = 1
    assert classify(actions, valid, 'skeleton_swordsman') == 'event'


def test_place_boost_preserves_roles_and_equal_episode_mass():
    rows = []
    for profile, strata in PLACE_BOOST_MASS_THREE_ROLES.items():
        for stratum in strata:
            for episode in ('short', 'long'):
                count = 2 if episode == 'short' else 5
                for i in range(count):
                    rows.append({'profile': profile, 'sampling_stratum': stratum,
                        'episode_id': episode,
                        'first8_action_counts': {'place': int(profile == 'language_builder' and
                            stratum == 'event' and i == 0)}})
    set_place_boost_weights(rows, place_multiplier=3)
    for profile, total in [('language_builder', .6), ('zombie_melee', .2),
                           ('skeleton_swordsman', .2)]:
        assert sum(r['sample_weight'] for r in rows if r['profile'] == profile) == pytest.approx(total)
    for episode in ('short', 'long'):
        event = [r for r in rows if r['profile'] == 'language_builder' and
                 r['sampling_stratum'] == 'event' and r['episode_id'] == episode]
        assert sum(r['sample_weight'] for r in event) == pytest.approx(.42 / 2)
        place = next(r for r in event if r['first8_action_counts']['place'])
        ordinary = next(r for r in event if not r['first8_action_counts']['place'])
        assert place['sample_weight'] == pytest.approx(3 * ordinary['sample_weight'])


def test_skeleton_requires_own_hits_and_pre_hit_approach(tmp_path):
    ep = tmp_path/'train'/'skeleton'; ep.mkdir(parents=True)
    manifest = {'behavior_routes': {'agent1': {'profile': 'skeleton_swordsman'}},
                'task_id': 'combat', 'model_start_observation': 0}
    (ep/'manifest.json').write_text(json.dumps(manifest))
    actions = np.zeros((160,2,23), np.float32)
    actions[:20,1,0] = 1; actions[[60,70],1,8] = 1
    positions = np.zeros((160,2,3), np.float32)
    positions[:,1,0] = np.linspace(5, 1, 160)
    np.savez(ep/'data.npz', cam_pos=positions, action_continuous=actions,
             player_health=np.full((160,2),20), player_health_valid=np.ones((160,2),bool),
             termination_flag=np.zeros(160,bool), npc_policy_train_mask=np.ones((160,2),bool))
    damage = [dict(event='damage', source='agent1', target='agent0', damage=2, transition_index=i)
              for i in (60,70)]
    (ep/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in damage))
    row = dict(episode_id='skeleton',scenario_id='S08')
    clips, proof = audit(tmp_path, row, 'skeleton_swordsman')
    assert proof['damage_events_by_actor'] == 2
    assert proof['movement_steps_before_first_hit'] == 20
    assert all(r['profile']=='skeleton_swordsman' and r['agent_slot']==1 for r in clips)
    assert any(r['first8_action_counts']['dig'] > 0 for r in clips)
    for e in damage: e.update(source='agent0',target='agent1')
    (ep/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in damage))
    assert audit(tmp_path, row, 'skeleton_swordsman') is None


def test_sampling_migration_cannot_change_model_or_loss():
    old = {'inputs': {'train_index': {'sha256':'old'}, 'val_index': {'sha256':'old'},
                      'm3_checkpoint': {'sha256':'model'}},
           'source': {'miworld/models/structured_action.py':'loss', 'train_scripts/train_policy.py':'old'}}
    new = copy.deepcopy(old)
    new['inputs']['train_index']['sha256'] = 'new'
    new['source']['train_scripts/train_policy.py'] = 'new'
    assert check_resume_provenance(old,new,sampling_change=True)['changed_inputs'] == ['train_index']
    with pytest.raises(ValueError): check_resume_provenance(old,new)
    new['source']['miworld/models/structured_action.py'] = 'changed'
    with pytest.raises(ValueError): check_resume_provenance(old,new,sampling_change=True)
    new['source']['miworld/models/structured_action.py'] = 'loss'
    new['inputs']['m3_checkpoint']['sha256'] = 'changed'
    with pytest.raises(ValueError): check_resume_provenance(old,new,sampling_change=True)


def test_action_metrics_measure_rare_action_misses_without_affecting_loss():
    head = StructuredActionHead(hidden_size=4, horizons=32)
    model = SimpleNamespace(families={'x':head}, decode_execution=lambda x: head.decode(x)[..., :8, :])
    logits = head(torch.zeros(1,9,4))
    for k in logits: logits[k] = torch.full_like(logits[k], -20.)
    logits['hotbar'][..., 0] = 20
    logits['mouse_x'][..., 8] = 20; logits['mouse_y'][..., 8] = 20
    actions = torch.zeros(1,9,32,23); actions[0,0,3,9] = 1
    valid = torch.zeros(1,9,32,dtype=torch.bool); valid[0,0,:8] = True
    fake = SimpleNamespace(head=head)
    model.families = {'x':fake}
    profile = torch.tensor([PROFILE_TO_ID['language_builder']])
    counts = action_counts(model, logits, actions, valid, profile)
    summary = format_action_counts(counts)['all']
    assert summary['n'] == 8 and summary['place']['target'] == 1
    assert summary['place']['recall'] == 0 and summary['noop']['fp'] == 1
    logits['keys'][0,0,3,8] = 20
    assert format_action_counts(action_counts(model, logits, actions, valid, profile))['all']['place']['recall'] == 1


def test_mouse_metrics_ignore_zero_axis_accuracy_in_direction_score():
    head = StructuredActionHead(hidden_size=4, horizons=32)
    logits = head(torch.zeros(1, 9, 4))
    for key in logits:
        logits[key] = torch.full_like(logits[key], -20.)
    logits['hotbar'][..., 0] = 20
    logits['mouse_x'][..., 8] = 20
    logits['mouse_y'][..., 8] = 20
    # Predict the correct positive x bin, but miss the negative y direction.
    logits['mouse_x'][0, 0, 0, 8] = -20
    logits['mouse_x'][0, 0, 0, 13] = 20
    actions = torch.zeros(1, 9, 32, 23)
    actions[0, 0, 0, 21] = .05
    actions[0, 0, 0, 22] = -.05
    valid = torch.zeros(1, 9, 32, dtype=torch.bool)
    valid[0, 0, :8] = True
    fake = SimpleNamespace(head=head)
    model = SimpleNamespace(
        families={'x': fake},
        decode_execution=lambda value: head.decode(value)[..., :8, :],
    )
    profile = torch.tensor([PROFILE_TO_ID['language_builder']])
    summary = format_mouse_axis_counts(
        mouse_axis_counts(model, logits, actions, valid, profile))['all']
    assert summary['target_nonzero'] == 2
    assert summary['nonzero_sign_accuracy'] == .5
    assert summary['nonzero_exact_accuracy'] == .5
    assert summary['nonzero_within_004_accuracy'] == .5
