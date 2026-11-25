"""Continuous episodes -> integrated read-only M3 eight-frame chunks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .renderer_dataset import TextAgentRendererDataset, collate_renderer
from .fill_dataset import TextAgentFillDataset
from miworld.policy_diagnostics import progress
from miworld.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from .interaction_history import (build_interaction_history, dropout_history_action_blocks,
                                  successful_place_frames, resample_interaction_pairs)


def canonical_text(value):
    return " ".join(str(value or "").strip().split())


def structured_goal_text(row):
    """Return the canonical, label-derived builder goal used as current_text."""
    family = row.get('simple_task_family')
    material = row.get('simple_task_material')
    dimensions = row.get('simple_task_dimensions')
    if not isinstance(family, str) or not canonical_text(family):
        raise ValueError('structured goal requires simple_task_family')
    if not isinstance(material, str) or not canonical_text(material):
        raise ValueError('structured goal requires simple_task_material')
    if (not isinstance(dimensions, list) or len(dimensions) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in dimensions)):
        raise ValueError('structured goal requires three positive integer dimensions')
    size = 'x'.join(map(str, dimensions))
    return f'task={canonical_text(family)} material={canonical_text(material)} size={size}'


def component_text_at_frames(timeline, frames):
    """Instructions become visible at observation_frame, never transition_before."""
    times = [int(e['observation_frame']) for e in timeline]
    if times != sorted(set(times)):
        raise ValueError('component timeline must have unique increasing observation frames')
    slots = np.searchsorted(times, np.asarray(frames), side='right') - 1
    return [canonical_text(timeline[i]['text']) if i >= 0 else '' for i in slots]


def mask_unseen_component_targets(valid, policy_indices, timeline, context_start,
                                  execution_horizon=8):
    """Exclude labels that require a component instruction not yet observed.

    An anchor whose executed prefix crosses a text change is unsafe as an
    open-loop chunk and is removed entirely.  For later changes, only the
    ambiguous auxiliary tail is truncated.
    """
    valid = np.asarray(valid, dtype=bool).copy()
    times = [int(entry['observation_frame']) for entry in timeline]
    texts = [canonical_text(entry.get('text')) for entry in timeline]
    for index, history in enumerate(policy_indices):
        anchor = int(context_start + int(history[-1]))
        visible = component_text_at_frames(timeline, [anchor])[0]
        change = next((time for time, text in zip(times, texts)
                       if time > anchor and text != visible), None)
        if change is None: continue
        delta = change - anchor
        if delta < execution_horizon:
            valid[index] = False
        elif delta < valid.shape[1]:
            valid[index, delta:] = False
    return valid


def mask_targets_to_absolute_ranges(valid, policy_indices, ranges, context_start):
    """Keep labels only inside explicitly validated absolute action ranges."""
    valid = np.asarray(valid, dtype=bool).copy()
    if not ranges:
        raise ValueError('absolute supervision ranges must be nonempty')
    normalized = []
    for value in ranges:
        if len(value) != 2:
            raise ValueError('absolute supervision ranges must be [start, end) pairs')
        start, end = map(int, value)
        if start < 0 or end <= start:
            raise ValueError('absolute supervision ranges must be positive nonempty intervals')
        normalized.append((start, end))
    anchors = context_start + policy_indices[:, -1].numpy().astype(np.int64)
    absolute = anchors[:, None] + np.arange(valid.shape[1], dtype=np.int64)[None]
    allowed = np.zeros_like(valid)
    for start, end in normalized:
        allowed |= (absolute >= start) & (absolute < end)
    valid &= allowed
    if not valid.any():
        raise ValueError('absolute supervision ranges do not overlap valid policy targets')
    return valid


def resample_place_anchors(target, valid, policy_indices, fraction):
    """Repeat place-positive dense anchors without changing the action loss."""
    if not 0 <= fraction <= 1:
        raise ValueError('builder place anchor fraction must be in [0,1]')
    if fraction == 0: return target, valid, policy_indices
    positive = (((target[:, :8, 9] > .5) & valid[:, :8]).any(-1))
    positive_indices = np.flatnonzero(positive).tolist()
    if not positive_indices: return target, valid, policy_indices
    count = len(target)
    desired = max(len(positive_indices), int(np.ceil(count * fraction)))
    desired = min(count, desired)
    negative_indices = np.flatnonzero(~positive & valid.any(-1)).tolist()
    if not negative_indices:
        negative_indices = np.flatnonzero(~positive).tolist()

    def repeat(values, size):
        if size == 0: return []
        if not values: values = positive_indices
        return [values[index % len(values)] for index in range(size)]

    selected = repeat(positive_indices, desired) + repeat(negative_indices, count - desired)
    selection = np.asarray(selected, dtype=np.int64)
    return target[selection], valid[selection], policy_indices[torch.from_numpy(selection)]


def dense_action_targets(data, family, agent, context_start, context_frames=129,
                         horizons=32, stride=8, warmup=32):
    """Only supervise live, trainable prefixes; never cross invalid/terminal steps."""
    actions = data['action_continuous']
    mask_key = 'language_policy_train_mask' if family == 'language_builder' else 'npc_policy_train_mask'
    if mask_key not in data:
        mask_key = 'policy_train_mask' if family == 'language_builder' else 'native_npc_policy_train_mask'
    trainable = np.asarray(data[mask_key][:, agent], bool)
    health = np.asarray(data['player_health'][:, agent])
    alive = np.asarray(data['player_health_valid'][:, agent], bool) & (health > 0)
    terminal = np.asarray(data['termination_flag'], bool)
    if terminal.ndim > 1: terminal = terminal.any(axis=tuple(range(1, terminal.ndim)))
    anchors = np.arange(warmup, context_frames - horizons, stride)
    if not len(anchors): raise ValueError('context has no dense prediction anchors')
    targets, masks = [], []
    for relative in anchors:
        anchor = context_start + relative
        target = actions[anchor:anchor + horizons, agent].astype(np.float32)
        if len(target) != horizons: raise ValueError('dense target exceeds episode')
        valid = trainable[anchor:anchor + horizons].copy()
        valid &= ~terminal[anchor:anchor + horizons]
        valid &= alive[anchor:anchor + horizons] & alive[anchor + 1:anchor + horizons + 1]
        valid &= alive[anchor - 7:anchor + 1].all()
        valid &= ~terminal[anchor - 7:anchor].any()
        # After a reset/invalid transition the subsequent states belong to a
        # different rollout, even when health becomes positive again.
        valid = np.logical_and.accumulate(valid)
        targets.append(target); masks.append(valid)
    indices = anchors[:, None] + np.arange(-7, 1)[None]
    return np.stack(targets), np.stack(masks), torch.from_numpy(indices)


class InsertedPolicyDataset(Dataset):
    def __init__(self, index, vocabulary, text_catalog, dataset_root=None,
                 context_frames=65, horizons=8, dense_stride=0, dense_warmup=32,
                 preserve_index_weights=False, current_text_mode='episode',
                 mask_component_boundaries=False, builder_place_anchor_fraction=0.,
                 builder_action_history=0, builder_interaction_pairs=False,
                 builder_history_action_dropout=0.):
        self.rows = [json.loads(line) for line in Path(index).open() if line.strip()]
        if not self.rows: raise ValueError("M3 policy index is empty")
        self.vocabulary = vocabulary
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.context_frames = int(context_frames)
        if current_text_mode not in ('episode', 'components', 'structured_goal',
                                     'structured_goal_only'):
            raise ValueError('unsupported current text mode')
        self.current_text_mode = current_text_mode
        self.mask_component_boundaries = bool(mask_component_boundaries)
        self.builder_place_anchor_fraction = float(builder_place_anchor_fraction)
        self.builder_action_history = int(builder_action_history)
        self.builder_interaction_pairs = bool(builder_interaction_pairs)
        self.builder_history_action_dropout = float(builder_history_action_dropout)
        if self.builder_action_history not in (0, 32, 64):
            raise ValueError('builder action history must be 0, 32 or 64')
        if not 0 <= self.builder_history_action_dropout <= 1:
            raise ValueError('builder history action dropout must be in [0,1]')
        if self.builder_history_action_dropout and not self.builder_action_history:
            raise ValueError('builder history action dropout requires action history')
        if self.builder_interaction_pairs and (not dense_stride or builder_place_anchor_fraction):
            raise ValueError('interaction pairs require dense anchors and replace old place resampling')
        if not 0 <= self.builder_place_anchor_fraction <= 1:
            raise ValueError('builder place anchor fraction must be in [0,1]')
        if (self.mask_component_boundaries or self.builder_place_anchor_fraction) and (
                current_text_mode not in ('components', 'structured_goal',
                                          'structured_goal_only') or not dense_stride):
            raise ValueError('component boundary/anchor sampling requires dense aligned text')
        if current_text_mode in ('components', 'structured_goal', 'structured_goal_only'):
            for row in self.rows:
                if row.get('current_text_schema') != 'component_observation_v1':
                    raise ValueError('aligned text mode requires an explicitly aligned index')
                if row['family'] == 'language_builder' and not row.get('current_text_timeline'):
                    raise ValueError('missing component instruction timeline')
                if current_text_mode in ('structured_goal', 'structured_goal_only') and (
                        row['family'] == 'language_builder'):
                    structured_goal_text(row)
        self.horizons, self.dense_stride, self.dense_warmup = horizons, dense_stride, dense_warmup
        if horizons != 8 and not dense_stride:
            raise ValueError('long-horizon training requires dense_stride')
        if dense_stride and (context_frames != 129 or horizons not in (16, 32)
                             or dense_stride != 8 or dense_warmup != 32):
            raise ValueError('dense v1 contract is context129 / predict16-or-32 / stride8 / warmup32')
        if dense_stride:
            # Existing indices may contain several decisions in the same clip.
            # Keep one record per known source clip, then rebalance profiles.
            from collections import Counter
            unique = {}
            for row in self.rows:
                key = (row['episode_path'], row['agent_slot'], row.get('context_start', row['anchor']))
                unique.setdefault(key, row)
            self.rows = list(unique.values())
            counts = Counter(row['profile'] for row in self.rows)
            for row in self.rows:
                if preserve_index_weights:
                    weight = float(row['sample_weight'])
                    if not np.isfinite(weight) or weight <= 0:
                        raise ValueError('index sample_weight must be finite and positive')
                else:
                    row['sample_weight'] = min(64., max(counts.values()) / counts[row['profile']])
        if self.context_frames < 9:
            raise ValueError("M3 policy context must contain at least eight completed frames")
        catalog = json.loads(Path(text_catalog).read_text())
        self.text_ids = {canonical_text(row["text"]): int(row["text_id"])
                         for row in catalog["texts"]}
        if "" not in self.text_ids: raise ValueError("text catalog must contain an empty entry")
        if current_text_mode in ('structured_goal', 'structured_goal_only'):
            missing = sorted({structured_goal_text(row) for row in self.rows
                              if row['family'] == 'language_builder'} - self.text_ids.keys())
            if missing:
                raise ValueError(f'text catalog is missing {len(missing)} structured goals: {missing[:3]}')

    def __len__(self): return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]; anchor, agent = int(row["anchor"]), int(row["agent_slot"])
        progress('data_read', index=int(item), episode=row['episode_path'], anchor=anchor)
        episode = Path(row["episode_path"])
        if not episode.is_absolute():
            if self.dataset_root is None:
                raise ValueError("relative M3 policy episode paths require dataset_root")
            episode = self.dataset_root / episode
        manifest = json.loads((episode / "manifest.json").read_text())
        history = {}
        events = []
        if row['family'] == 'language_builder' and (self.builder_action_history or self.builder_interaction_pairs):
            events = [json.loads(line) for line in (episode / manifest.get('event_file', 'events.jsonl')).open()]
        with np.load(episode / manifest.get("training_data_file", "data.npz"),
                     allow_pickle=False) as data:
            model_start = TextAgentFillDataset._model_start(data, manifest)
            frames = len(data["cam_pos"])
            if frames - model_start < self.context_frames:
                raise ValueError(
                    "M3 policy episode is shorter than the configured M3 context"
                )
            # Early decisions use the episode prefix; later decisions use the
            # longest 65-frame causal history ending at the current observation.
            context_start = min(
                max(model_start, anchor - (self.context_frames - 1)),
                frames - self.context_frames,
            )
            if self.dense_stride and 'dense_valid_targets' in row:
                context_start = int(row['context_start'])
            if self.dense_stride:
                target, valid_mask, policy_indices = dense_action_targets(
                    data, row['family'], agent, context_start, self.context_frames,
                    self.horizons, self.dense_stride, self.dense_warmup)
                if row['family'] == 'language_builder' and self.mask_component_boundaries:
                    valid_mask = mask_unseen_component_targets(
                        valid_mask, policy_indices, row['current_text_timeline'], context_start)
                if row.get('absolute_supervision_ranges') is not None:
                    valid_mask = mask_targets_to_absolute_ranges(
                        valid_mask, policy_indices, row['absolute_supervision_ranges'], context_start)
                if row['family'] == 'language_builder' and self.builder_place_anchor_fraction:
                    target, valid_mask, policy_indices = resample_place_anchors(
                        target, valid_mask, policy_indices, self.builder_place_anchor_fraction)
                if row['family'] == 'language_builder' and self.builder_interaction_pairs:
                    successes = successful_place_frames(data['action_continuous'], events, agent)
                    target, valid_mask, policy_indices = resample_interaction_pairs(
                        target, valid_mask, policy_indices, context_start, successes)
                    absolute_anchors = policy_indices[:, -1].numpy() + context_start
                    successful = sum(any(valid_mask[i,h] and target[i,h,9] > .5 and
                        int(a+h) in successes for h in range(8)) for i,a in enumerate(absolute_anchors))
                    negative = (~((target[:,:8,9] > .5) & valid_mask[:,:8]).any(-1) &
                                valid_mask[:,:8].all(-1)).sum()
                    progress('interaction_pair_sampling', index=int(item), anchors=absolute_anchors.tolist(),
                             successful_click_anchors=int(successful), no_click_anchors=int(negative))
            else:
                target = data["action_continuous"][anchor:anchor+8, agent].astype(np.float32)
                valid_mask = np.ones(8, dtype=bool)
                policy_indices = torch.arange(anchor - 7 - context_start, anchor + 1 - context_start)
            if self.builder_action_history:
                if row['family'] == 'language_builder':
                    features, history_valid = build_interaction_history(
                        data['action_continuous'], data['player_health'], data['player_health_valid'],
                        data['termination_flag'], events, agent, context_start, self.context_frames,
                        self.builder_action_history)
                    features = dropout_history_action_blocks(
                        features, history_valid, self.builder_history_action_dropout)
                else:
                    length = self.context_frames + self.builder_action_history - 1
                    features, history_valid = torch.zeros(length, 25), torch.zeros(length, dtype=torch.bool)
                history = dict(interaction_history=features, interaction_history_valid=history_valid)
        renderer = TextAgentRendererDataset.read_window(
            episode, self.vocabulary, start=context_start, target=agent,
            context_frames=self.context_frames,
            appearance_aliases=row.get('appearance_aliases'))
        renderer["conditions"]["condition_mask"] = torch.ones(
            self.context_frames, dtype=torch.bool
        )
        shared = canonical_text(row.get("shared_task_text", row.get("task_text", "")))
        current = canonical_text(row.get("current_task_text", shared))
        if row["family"] != "language_builder": shared = current = ""
        if self.current_text_mode == 'components':
            texts = (component_text_at_frames(row['current_text_timeline'],
                     range(context_start, context_start + self.context_frames))
                     if row['family'] == 'language_builder' else [''] * self.context_frames)
            current_ids = torch.tensor([self.text_ids[text] for text in texts])
            current_present = torch.tensor([bool(text) for text in texts])
        elif self.current_text_mode == 'structured_goal':
            goal = structured_goal_text(row) if row['family'] == 'language_builder' else ''
            current_ids = torch.tensor(self.text_ids[goal])
            current_present = torch.tensor(bool(goal))
        elif self.current_text_mode == 'structured_goal_only':
            shared = structured_goal_text(row) if row['family'] == 'language_builder' else ''
            current_ids = torch.tensor(self.text_ids[''])
            current_present = torch.tensor(False)
        else:
            current_ids = torch.tensor(self.text_ids[current])
            current_present = torch.tensor(bool(current))
        progress('data_ready', index=int(item))
        return {
            **history,
            'sample_index': int(item),
            'sample_profile': row['profile'],
            "renderer": renderer,
            "family_id": torch.tensor(FAMILY_TO_ID[row["family"]]),
            "profile_id": torch.tensor(PROFILE_TO_ID[row["profile"]]),
            "shared_text_id": torch.tensor(self.text_ids[shared]),
            "current_text_id": current_ids,
            "current_text_valid": current_present,
            "target_actions": torch.from_numpy(target),
            "policy_indices": policy_indices,
            "valid_mask": torch.from_numpy(valid_mask),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.))),
        }


def collate_inserted_policy(samples):
    renderer = collate_renderer([sample.pop("renderer") for sample in samples])
    result = default_collate(samples)
    result.update(rgb=renderer["rgb"], conditions=renderer["conditions"])
    for key in ('interaction_history', 'interaction_history_valid'):
        if key in result:
            result['conditions'][key] = result.pop(key)
    return result
