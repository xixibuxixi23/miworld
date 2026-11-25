"""Audit/deduplicate legacy clips and discard clips with no valid dense targets."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miworld.data.inserted_policy_dataset import dense_action_targets
from miworld.data.fill_dataset import TextAgentFillDataset


def audit_episode(job):
    root, path, rows = job
    episode = Path(root) / path
    manifest = json.loads((episode / 'manifest.json').read_text())
    result = {}
    with np.load(episode / manifest.get('training_data_file', 'data.npz'), allow_pickle=False) as data:
        start = TextAgentFillDataset._model_start(data, manifest)
        frames = len(data['cam_pos'])
        if frames - start < 129: return []
        required = ['action_continuous', 'player_health', 'player_health_valid', 'termination_flag']
        required += [k for k in ('language_policy_train_mask', 'policy_train_mask',
                                'npc_policy_train_mask', 'native_npc_policy_train_mask') if k in data]
        arrays = {k: data[k] for k in required}
        for row in rows:
            # Put the old selected decision in the supervised middle of the new
            # clip, rather than at its right edge where no 32-step target fits.
            context = min(max(start, int(row['anchor']) - 64), frames - 129)
            key = (int(row['agent_slot']), context)
            if key in result: continue
            _, valid, _ = dense_action_targets(arrays, row['family'], key[0], context)
            if not valid.any(): continue
            result[key] = {**row, 'context_start': context,
                           'dense_valid_targets': int(valid.sum()),
                           'dense_valid_anchors': int(valid.any(-1).sum())}
    return list(result.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--one-per-profile', action='store_true', help='Small training-only smoke index')
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    groups = defaultdict(list)
    seen_profiles = set()
    for line in Path(a.source).open():
        row = json.loads(line)
        if a.one_per_profile and row['profile'] in seen_profiles: continue
        seen_profiles.add(row['profile'])
        groups[row['episode_path']].append(row)
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for n, result in enumerate(pool.map(audit_episode,
                ((a.dataset_root, path, records) for path, records in groups.items())), 1):
            rows.extend(result)
            if n % 100 == 0: print(json.dumps({'episodes': n, 'clips': len(rows)}), flush=True)
    profiles = Counter(row['profile'] for row in rows)
    for row in rows: row['sample_weight'] = min(64., max(profiles.values()) / profiles[row['profile']])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        for row in rows: handle.write(json.dumps(row) + '\n')
    summary = {'clips': len(rows), 'episodes': len(groups), 'profiles': dict(profiles),
               'valid_targets': sum(row['dense_valid_targets'] for row in rows),
               'valid_anchors': sum(row['dense_valid_anchors'] for row in rows),
               'source': a.source, 'context': 129, 'horizons': 32, 'stride': 8}
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
