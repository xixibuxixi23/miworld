"""Fixed cooperative-building / melee NPC episodes; sampling only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miworld.data.inserted_policy_dataset import dense_action_targets
from miworld.data.fill_dataset import TextAgentFillDataset

MASS = {'language_builder': {'event': .30, 'prepare': .18, 'background': .12},
        'zombie_melee': {'event': .12, 'prepare': .20, 'background': .08}}
MASS_THREE_ROLES = {
    'language_builder': {'event': .30, 'prepare': .18, 'background': .12},
    'zombie_melee': {'event': .06, 'prepare': .10, 'background': .04},
    'skeleton_swordsman': {'event': .06, 'prepare': .10, 'background': .04},
}
PLACE_BOOST_MASS_THREE_ROLES = {
    # Preserve the 60/20/20 role mixture.  Within building, move probability
    # from preparation/background to executable interactions; place-positive
    # rows receive an additional within-episode multiplier below.
    'language_builder': {'event': .42, 'prepare': .12, 'background': .06},
    'zombie_melee': {'event': .06, 'prepare': .10, 'background': .04},
    'skeleton_swordsman': {'event': .06, 'prepare': .10, 'background': .04},
}


def classify(target, valid, profile):
    """Classify first decision's executable prefix, NOT future/invalid labels."""
    a = target[0, :8][valid[0, :8]]
    if not len(a): return None
    interaction = (a[:, [8, 9]] > .5).any() if profile == 'language_builder' else (a[:, 8] > .5).any()
    if interaction: return 'event'
    move = (a[:, :4] > .5).any()
    select = (a[:, 12:21] > .5).any()
    if move or (profile == 'language_builder' and select): return 'prepare'
    # Includes look-only alignment/search and legitimate stationary negatives.
    return 'background'


def set_weights(rows, masses=None):
    """Exact role/stratum mass; equal episode mass within each stratum."""
    masses = MASS if masses is None else masses
    if set(r['profile'] for r in rows) != set(masses):
        raise ValueError('index profiles do not match sampling masses')
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows: groups[(row['profile'], row['sampling_stratum'])][row['episode_id']].append(row)
    for profile, strata in masses.items():
        for stratum, mass in strata.items():
            episodes = groups[(profile, stratum)]
            if not episodes: raise ValueError(f'empty required stratum: {profile}/{stratum}')
            for clips in episodes.values():
                for row in clips: row['sample_weight'] = mass / len(episodes) / len(clips)
    assert abs(sum(r['sample_weight'] for r in rows) - 1) < 1e-9


def is_place_positive(row):
    return int(row.get('first8_action_counts', {}).get('place', 0)) > 0


def set_place_boost_weights(rows, place_multiplier=3., masses=None):
    """Preserve role totals while emphasizing builder event/place windows.

    Episodes retain equal mass inside each role/stratum.  The multiplier only
    redistributes an episode's builder-event mass among its source clips, so a
    long episode cannot dominate merely because it produced more windows.
    """
    masses = PLACE_BOOST_MASS_THREE_ROLES if masses is None else masses
    if not np.isfinite(place_multiplier) or place_multiplier < 1:
        raise ValueError('place multiplier must be finite and at least one')
    if set(r['profile'] for r in rows) != set(masses):
        raise ValueError('index profiles do not match place-boost sampling masses')
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row['profile'], row['sampling_stratum'])][row['episode_id']].append(row)
    for profile, strata in masses.items():
        for stratum, mass in strata.items():
            episodes = groups[(profile, stratum)]
            if not episodes: raise ValueError(f'empty required stratum: {profile}/{stratum}')
            for clips in episodes.values():
                scores = [place_multiplier if (profile == 'language_builder' and
                          stratum == 'event' and is_place_positive(row)) else 1.
                          for row in clips]
                normalizer = sum(scores)
                for row, score in zip(clips, scores):
                    row['sample_weight'] = mass / len(episodes) * score / normalizer
                    row['sampling_schema'] = 'overfit-role-placeboost-v2'
    assert abs(sum(r['sample_weight'] for r in rows) - 1) < 1e-9


def _diverse_rows(pool, count, rng, used=None):
    """Choose rows from distinct episodes first, then fill deterministically."""
    used = set() if used is None else used
    candidates = list(pool); rng.shuffle(candidates)
    chosen, chosen_indices = [], set()
    for index, row in enumerate(candidates):
        if len(chosen) == count: break
        if row['episode_id'] in used: continue
        chosen.append(row); chosen_indices.add(index); used.add(row['episode_id'])
    for index, row in enumerate(candidates):
        if len(chosen) == count: break
        if index in chosen_indices: continue
        chosen.append(row); used.add(row['episode_id'])
    if len(chosen) != count: raise ValueError(f'insufficient replay rows: wanted {count}, got {len(chosen)}')
    return chosen


def _combat_replay(rows, profile, count, rng):
    proportions = {'event': .30, 'prepare': .50, 'background': .20}
    raw = {name: count * value for name, value in proportions.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    for name in sorted(quotas, key=lambda x: raw[x] - quotas[x], reverse=True)[:count-sum(quotas.values())]:
        quotas[name] += 1
    selected, used = [], set()
    for stratum in ('event', 'prepare', 'background'):
        pool = [row for row in rows if row['profile'] == profile and
                row['sampling_stratum'] == stratum]
        selected.extend(_diverse_rows(pool, quotas[stratum], rng, used))
    return selected


def build_place_boost(source, output, seed, place_multiplier, replay_place_targets):
    """Reweight an aligned component-text index without changing examples/loss."""
    if output.exists(): raise FileExistsError(output)
    rng = random.Random(seed)
    rows = [json.loads(line) for line in (source/'train.jsonl').open() if line.strip()]
    set_place_boost_weights(rows, place_multiplier)

    builder_pool = [row for row in rows if row['profile'] == 'language_builder' and
                    row['sampling_stratum'] == 'event' and is_place_positive(row)]
    rng.shuffle(builder_pool)
    builder, used, targets = [], set(), 0
    # Start with one window per episode so the larger replay measures broad
    # behavior rather than repeatedly scoring a handful of trajectories.
    for row in builder_pool:
        if row['episode_id'] in used: continue
        builder.append(row); used.add(row['episode_id'])
        targets += int(row['first8_action_counts']['place'])
        if targets >= replay_place_targets: break
    if targets < replay_place_targets:
        raise ValueError(f'insufficient place labels for replay: {targets}')
    # Equal profile counts and a multiple of eight make the distributed
    # profile sampler visit every replay row exactly once.
    target_count = ((len(builder) + 7) // 8) * 8
    remaining = [row for row in builder_pool if row not in builder]
    builder.extend(_diverse_rows(remaining, target_count-len(builder), rng, used))
    targets = sum(int(row['first8_action_counts']['place']) for row in builder)
    if not 200 <= targets <= 500:
        raise ValueError(f'place replay target count outside requested range: {targets}')

    replay = list(builder)
    for profile in ('zombie_melee', 'skeleton_swordsman'):
        replay.extend(_combat_replay(rows, profile, target_count, rng))
    replay = [{**row, 'evaluation_label': 'train_replay_not_holdout'} for row in replay]
    assert len(replay) == 3 * target_count and len(replay) % 8 == 0

    output.mkdir(parents=True)
    for name, values in (('train', rows), ('train_replay', replay)):
        with (output/f'{name}.jsonl').open('x') as handle:
            for row in values: handle.write(json.dumps(row, sort_keys=True)+'\n')
    report = {
        'status': 'place_boost_sampling_verified', 'source': str(source.resolve()),
        'seed': seed, 'clips': len(rows),
        'episodes': len({row['episode_id'] for row in rows}),
        'profiles': dict(Counter(row['profile'] for row in rows)),
        'sampling_masses': PLACE_BOOST_MASS_THREE_ROLES,
        'place_multiplier_within_builder_event_episode': place_multiplier,
        'weighted_masses': {p+'/'+s: sum(row['sample_weight'] for row in rows
            if row['profile'] == p and row['sampling_stratum'] == s)
            for p in PLACE_BOOST_MASS_THREE_ROLES for s in PLACE_BOOST_MASS_THREE_ROLES[p]},
        'replay_windows': len(replay), 'replay_windows_per_profile': target_count,
        'replay_place_targets': targets, 'val_batches_per_rank': len(replay)//8,
        'text': 'unchanged shared task and causal component timelines',
        'loss': 'unchanged; sampler weights are not loss weights',
    }
    (output/'data_verified.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def interleaved(rows, key, rng):
    groups = defaultdict(list)
    for row in rows: groups[row[key]].append(row)
    for group in groups.values(): rng.shuffle(group)
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]: yield groups[name].pop()


def audit(root, row, profile):
    ep = root / 'train' / row['episode_id']
    manifest = json.loads((ep / 'manifest.json').read_text())
    routes = manifest.get('behavior_routes') or {}
    slots = [int(k[5:]) for k, v in routes.items() if v.get('profile') == profile]
    if len(slots) != (2 if profile == 'language_builder' else 1): return None
    events = [json.loads(line) for line in (ep / manifest.get('event_file', 'events.jsonl')).open()]
    if profile == 'language_builder':
        if not manifest.get('success') or not any(e['event'] == 'task_completed' for e in events): return None
        placed = {i: [int(e['transition_index']) for e in events if e['event'] == 'block_placed'
                  and e.get('actor') == f'agent{i}' and 'transition_index' in e] for i in slots}
        if not all(placed.values()): return None
        evidence = {'success': True, 'placed_per_agent': {str(i): len(v) for i, v in placed.items()}}
    else:
        slot = slots[0]
        damage = [e for e in events if e['event'] == 'damage' and e.get('source') == f'agent{slot}'
                  and e.get('target', '').startswith('agent') and e.get('target') != f'agent{slot}'
                  and float(e.get('damage', 0)) > 0]
        if len(damage) < 2: return None
        evidence = {'damage_events_by_actor': len(damage),
                    'damage_total': sum(e['damage'] for e in damage)}
    with np.load(ep / manifest.get('training_data_file', 'data.npz'), allow_pickle=False) as data:
        start = TextAgentFillDataset._model_start(data, manifest)
        frames = len(data['cam_pos'])
        if frames - start < 129: return None
        keys = ['action_continuous', 'player_health', 'player_health_valid', 'termination_flag']
        keys += [k for k in ('language_policy_train_mask', 'policy_train_mask', 'npc_policy_train_mask',
                            'native_npc_policy_train_mask') if k in data]
        arrays = {k: data[k] for k in keys}
        if profile in ('zombie_melee', 'skeleton_swordsman'):
            slot = slots[0]; other = 1 - slot
            positions = data['cam_pos']
            health = arrays['player_health']
            live = (health[:, slot] > 0) & (health[:, other] > 0)
            distance = np.linalg.norm(positions[:, slot] - positions[:, other], axis=-1)
            before_hit = min(int(e.get('attempt_transition_index', e['transition_index'])) for e in damage)
            approach = distance[start:before_hit+1][live[start:before_hit+1]]
            move = (arrays['action_continuous'][start:before_hit+1, slot, :4] > .5).any(-1)
            if len(approach) < 2 or approach[0] - approach.min() < 1.0 or move.sum() < 8: return None
            evidence.update(initial_distance=float(approach[0]), minimum_pre_hit_distance=float(approach.min()),
                            movement_steps_before_first_hit=int(move.sum()))
        contexts = sorted(set(range(start, frames-128, 8)) | {frames-129})
        output = []
        for slot in slots:
            slot_rows = []
            family = 'language_builder' if profile == 'language_builder' else 'combat'
            for context in contexts:
                target, valid, _ = dense_action_targets(arrays, family, slot, context)
                stratum = classify(target, valid, profile)
                if stratum is None: continue
                shared = manifest.get('task_text', '') if profile == 'language_builder' else ''
                subtasks = manifest.get('agent_task_texts') or manifest.get('agent_subtasks') or {}
                near = target[:, :8][valid[:, :8]]
                slot_rows.append({'episode_id': row['episode_id'], 'episode_path': str(ep.relative_to(root)),
                    'split': 'train', 'scenario_id': row['scenario_id'], 'task_id': manifest['task_id'],
                    'task_text': shared, 'shared_task_text': shared,
                    'current_task_text': subtasks.get(f'agent{slot}', shared),
                    'family': family, 'profile': profile, 'agent_slot': slot, 'context_start': context,
                    'anchor': context + 32, 'dense_valid_targets': int(valid.sum()),
                    'dense_valid_anchors': int(valid.any(-1).sum()), 'sampling_stratum': stratum,
                    'sampling_schema': 'overfit-role-event-v1',
                    'first8_action_counts': {'n': len(near), 'move': int((near[:, :4] > .5).any(-1).sum()),
                        'dig': int((near[:, 8] > .5).sum()), 'place': int((near[:, 9] > .5).sum())}})
            # Each builder perspective must contain executable placement labels.
            if not slot_rows or not any(r['first8_action_counts']['place' if profile == 'language_builder' else 'dig'] for r in slot_rows):
                return None
            output.extend(slot_rows)
    evidence.update(episode_id=row['episode_id'], profile=profile, scenario=row['scenario_id'],
                    task_id=manifest['task_id'], task_text=manifest.get('task_text', ''),
                    component_count=len(manifest.get('components') or []),
                    text_mode='episode_task_not_temporal_components',
                    agent_slots=slots, clips=len(output),
                    manifest_sha256=hashlib.sha256((ep/'manifest.json').read_bytes()).hexdigest(),
                    event_sha256=hashlib.sha256((ep/'events.jsonl').read_bytes()).hexdigest())
    return output, evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reweight-source', type=Path,
                   help='Reweight an existing component-aligned index for builder place windows')
    p.add_argument('--place-multiplier', type=float, default=3.)
    p.add_argument('--replay-place-targets', type=int, default=256)
    p.add_argument('--episodes-per-role', type=int, default=32)
    p.add_argument('--builder-episodes', type=int)
    p.add_argument('--zombie-episodes', type=int)
    p.add_argument('--skeleton-episodes', type=int, default=0)
    p.add_argument('--all-qualified', action='store_true',
                   help='Audit every candidate for all three roles instead of stopping at fixed quotas')
    p.add_argument('--max-builder-episodes-per-task', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261028)
    a = p.parse_args(); rng = random.Random(a.seed)
    if a.reweight_source:
        build_place_boost(a.reweight_source, a.output, a.seed, a.place_multiplier,
                          a.replay_place_targets)
        return
    if a.dataset_root is None: p.error('--dataset-root is required unless --reweight-source is used')
    desired = {'language_builder': a.builder_episodes if a.builder_episodes is not None else a.episodes_per_role,
               'zombie_melee': a.zombie_episodes if a.zombie_episodes is not None else a.episodes_per_role}
    include_skeleton = bool(a.skeleton_episodes or a.all_qualified)
    if include_skeleton: desired['skeleton_swordsman'] = a.skeleton_episodes
    if a.all_qualified: desired = {profile: None for profile in desired}
    if (any(value is not None and value < 1 for value in desired.values()) or
            a.max_builder_episodes_per_task < 1):
        p.error('episode counts and per-task cap must be positive')
    masses = MASS_THREE_ROLES if include_skeleton else MASS
    if a.output.exists(): raise FileExistsError(a.output)
    release = [json.loads(x) for x in (a.dataset_root/'train.jsonl').open()]
    candidates = {
        'language_builder': [r for r in release if r['scenario_id'] == 'S01' and
                            r['activity'] in ('build_structure', 'build_house', 'build_tower', 'tower_bridge')],
        'zombie_melee': [r for r in release if 'npc_zombie' in r.get('agent_kinds', {}).values()
                        and r.get('damage_events', 0) >= 2 and r.get('attack_contacts', 0) >= 2]}
    if include_skeleton:
        candidates['skeleton_swordsman'] = [r for r in release
            if 'npc_skeleton' in r.get('agent_kinds', {}).values()
            and r.get('damage_events', 0) >= 2 and r.get('attack_contacts', 0) >= 2]
    rows, evidence, rejected = [], [], Counter()
    selected_episodes = set()
    for profile, pool in candidates.items():
        accepted = 0; seen_tasks = Counter()
        for candidate in interleaved(pool, 'task_family' if profile == 'language_builder' else 'scenario_id', rng):
            if candidate['episode_id'] in selected_episodes: continue
            if profile == 'language_builder' and seen_tasks[candidate['task_id']] >= a.max_builder_episodes_per_task: continue
            result = audit(a.dataset_root, candidate, profile)
            if result is None:
                rejected[profile] += 1; continue
            clips, proof = result; rows.extend(clips); evidence.append(proof)
            selected_episodes.add(candidate['episode_id'])
            seen_tasks[candidate['task_id']] += 1; accepted += 1
            print(json.dumps({'profile': profile, 'accepted': accepted, 'rejected': rejected[profile],
                              'episode': candidate['episode_id']}), flush=True)
            if desired[profile] is not None and accepted == desired[profile]: break
        if desired[profile] is not None and accepted != desired[profile]:
            raise ValueError(f'insufficient qualified {profile} episodes: {accepted}')
    set_weights(rows, masses)
    # Fixed training replay, never held out. Three-role mode has 48 windows
    # (16 per role), so the profile-balanced evaluator visits each exactly once.
    replay = []; used = set()
    replay_quotas = [('language_builder', {'event': 8, 'prepare': 5, 'background': 3}),
                    ('zombie_melee', {'event': 5, 'prepare': 8, 'background': 3})]
    if include_skeleton:
        replay_quotas.append(('skeleton_swordsman', {'event': 5, 'prepare': 8, 'background': 3}))
    for profile, quotas in replay_quotas:
        for stratum, n in quotas.items():
            pool = [r for r in rows if r['profile'] == profile and r['sampling_stratum'] == stratum]
            rng.shuffle(pool); chosen = 0
            while chosen < n and pool:
                index = next((i for i, r in enumerate(pool) if r['episode_id'] not in used), 0)
                row = pool.pop(index)
                replay.append({**row, 'evaluation_label': 'train_replay_not_holdout'})
                used.add(row['episode_id']); chosen += 1
                if chosen == n: break
            if chosen != n: raise ValueError('insufficient replay stratum')
    a.output.mkdir(parents=True)
    for name, values in [('train', rows), ('train_replay', replay)]:
        with (a.output/f'{name}.jsonl').open('x') as f:
            for row in values: f.write(json.dumps(row, sort_keys=True) + '\n')
    summary = {'seed': a.seed, 'all_qualified': a.all_qualified,
        'candidate_counts': {profile: len(pool) for profile, pool in candidates.items()},
        'episodes': evidence, 'rejected': dict(rejected), 'clips': len(rows),
        'episodes_by_profile': dict(Counter(e['profile'] for e in evidence)),
        'max_builder_episodes_per_task': a.max_builder_episodes_per_task,
        'sampling_masses': masses, 'stratum_definition': 'first anchor, valid first8; event > movement/hotbar > look/wait background',
        'counts': dict(Counter(r['profile']+'/'+r['sampling_stratum'] for r in rows)),
        'weighted_masses': {p+'/'+s: sum(r['sample_weight'] for r in rows if r['profile']==p and r['sampling_stratum']==s)
                            for p in masses for s in masses[p]},
        'evaluation': f'{len(replay)} fixed training windows, NOT independent validation; original heldout data retained',
        'text_mode': 'unchanged episode task; temporal component instructions NOT consumed',
        'loss': 'unchanged; sampler weights are not loss weights',
        'context': 129, 'horizons': 32, 'execution_horizon': 8, 'dense_stride': 8}
    (a.output/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'episodes'}), flush=True)


if __name__ == '__main__': main()
