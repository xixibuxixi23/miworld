#!/usr/bin/env python3
"""Train inserted M3 policy: eight completed M3 frames -> one eight-action chunk."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time
import shutil
import random
import numpy as np
import hashlib
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors.torch import load_file
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from miworld.data.inserted_policy_dataset import InsertedPolicyDataset, collate_inserted_policy
from miworld.models.m3 import M3Policy, M3PolicyArgs
from miworld.models.renderer import Renderer, RendererArgs
from miworld.models.renderer_codec import RendererCodec, WanRendererCodec
from miworld.policy_diagnostics import progress
from miworld.policy_schema import FAMILY_TO_ID, PROFILES


SAMPLING_MIGRATION_FILES = {
    'miworld/data/inserted_policy_dataset.py', 'train_scripts/train_policy.py',
    'dataset_toolkits/build_m3_overfit_index.py',
    'train_scripts/recipes/m3/train_m3_policy_overfit.sh',
}
TEXT_MIGRATION_FILES = {
    'miworld/data/inserted_policy_dataset.py', 'miworld/models/inserted_policy.py',
    'train_scripts/train_policy.py', 'dataset_toolkits/build_m3_component_text.py',
    'train_scripts/recipes/m3/train_m3_policy_dense32.sh',
}


def check_resume_provenance(previous, current, *, sampling_change=False, layout_change=False, text_change=False):
    changed_inputs = []
    for name, entry in current['inputs'].items():
        if name == 'warm_start': continue
        if previous['inputs'].get(name, {}).get('sha256') != entry['sha256']:
            permitted = ({'train_index', 'val_index', 'text_catalog', 'text_cache'} if text_change else
                         {'train_index', 'val_index'} if sampling_change else set())
            if name not in permitted:
                raise ValueError(f'resume input changed: {name}')
            changed_inputs.append(name)
    changed_source = {key for key in previous['source'].keys() | current['source'].keys()
                      if previous['source'].get(key) != current['source'].get(key)}
    allowed = (TEXT_MIGRATION_FILES if text_change else SAMPLING_MIGRATION_FILES if sampling_change else
               {'train_scripts/train_policy.py', 'train_scripts/recipes/m3/train_m3_policy_dense32.sh'}
               if layout_change else set())
    if not changed_source <= allowed:
        raise ValueError(f'resume source changed outside approved migration: {sorted(changed_source - allowed)}')
    return {'changed_inputs': sorted(changed_inputs), 'changed_source': sorted(changed_source)}


def check_text_only_index(previous, current):
    old = [json.loads(x) for x in Path(previous).open()]
    new = [json.loads(x) for x in Path(current).open()]
    if len(old) != len(new): raise ValueError('text migration cannot change sample count')
    for left, right in zip(old, new):
        stripped = {k: v for k, v in right.items() if k not in {'current_text_schema', 'current_text_timeline'}}
        if left != stripped: raise ValueError('text migration cannot change samples, ordering, labels or weights')


def check_policy_config(previous, current, *, text_change=False):
    old, new = asdict(M3PolicyArgs(**previous)), asdict(current)
    changed = {key for key in old if old[key] != new[key]}
    if changed - ({'current_text_scale'} if text_change else set()):
        raise ValueError('resume policy config mismatch')


def migrate_history_optimizer(optimizer, families, source, source_names):
    """Keep AdamW moments for old named parameters, leave new history uninitialized."""
    current = optimizer.state_dict()
    if len(source['param_groups']) != 1 or len(current['param_groups']) != 1:
        raise ValueError('history optimizer migration expects one AdamW group')
    old_ids = source['param_groups'][0]['params']
    names = [name for name, _ in families.named_parameters()]
    new_ids = current['param_groups'][0]['params']
    if len(old_ids) != len(source_names) or len(names) != len(new_ids) or len(set(source_names)) != len(source_names):
        raise ValueError('optimizer parameter name/order mismatch')
    expected = [name for name in names if not name.startswith('language_builder.interaction_history.')]
    if expected != source_names:
        raise ValueError('old optimizer parameter ordering changed')
    target_ids = dict(zip(names, new_ids))
    params = dict(families.named_parameters())
    state = {}
    for name, old_id in zip(source_names, old_ids):
        if old_id not in source['state']:
            continue
        old = source['state'][old_id]
        for key in ('exp_avg', 'exp_avg_sq'):
            if old[key].shape != params[name].shape:
                raise ValueError(f'optimizer moment shape mismatch: {name}')
        state[target_ids[name]] = old
    group = {**source['param_groups'][0], 'params': new_ids}
    optimizer.load_state_dict(dict(state=state, param_groups=[group]))
    return len(state)


MOUSE_ACTIVITY_DEADZONE = .002


def action_counts(model, logits, actions, valid, profiles):
    """Offline first8 metrics only. Never used to construct the training loss."""
    head = next(iter(model.families.values())).head
    target = head.targets(actions[..., :8, :])
    predicted = model.decode_execution(logits)
    keys = target['keys'] >= .5
    camera = (actions[..., :8, 21:23].abs() >= MOUSE_ACTIVITY_DEADZONE).any(-1)
    gt = {'move': keys[..., :4].any(-1), 'dig': keys[..., 7], 'place': keys[..., 8],
          'camera': camera, 'noop': ~(keys.any(-1) | camera | (target['hotbar'] != 0))}
    pkeys = predicted[..., head.key_indices] >= .5
    pcamera = (predicted[..., 21:23].abs() >= MOUSE_ACTIVITY_DEADZONE).any(-1)
    pred = {'move': pkeys[..., :4].any(-1), 'dig': pkeys[..., 7], 'place': pkeys[..., 8],
            'camera': pcamera, 'noop': ~(pkeys.any(-1) | pcamera | (predicted[..., 12:21] > .5).any(-1))}
    result = torch.zeros(len(PROFILES), 1 + 4*len(gt), device=actions.device, dtype=torch.float64)
    for i in range(len(PROFILES)):
        mask = valid[..., :8].bool() & (profiles == i).view(-1, *([1]*(valid.ndim-1)))
        result[i, 0] = mask.sum()
        for j, name in enumerate(gt):
            g, p = gt[name], pred[name]
            result[i, 1+4*j:5+4*j] = torch.stack(((g & mask).sum(), (p & mask).sum(),
                                                 (g & p & mask).sum(), (~g & p & mask).sum()))
    return result


def format_action_counts(counts):
    output = {}
    for name, row in [('all', counts.sum(0)), *zip(PROFILES, counts)]:
        values = row.cpu().tolist()
        if not values[0]: continue
        result = {'n': int(values[0])}
        for j, action in enumerate(('move', 'dig', 'place', 'camera', 'noop')):
            g, p, tp, fp = values[1+4*j:5+4*j]
            result[action] = {'target': int(g), 'predicted': int(p), 'tp': int(tp), 'fp': int(fp),
                              'recall': tp/g if g else None, 'precision': tp/p if p else None}
        output[name] = result
    return output


def mouse_axis_counts(model, logits, actions, valid, profiles):
    """Direction-sensitive first8 mouse metrics, excluding zero-label inflation."""
    target_xy = actions[..., :8, 21:23].float()
    predicted_xy = model.decode_execution(logits)[..., 21:23]
    result = torch.zeros(len(PROFILES), 8, device=actions.device, dtype=torch.float64)
    for i in range(len(PROFILES)):
        step_mask = valid[..., :8].bool() & (
            profiles == i).view(-1, *([1] * (valid.ndim - 1)))
        axis_mask = step_mask[..., None].expand_as(target_xy)
        nonzero = (target_xy.abs() >= MOUSE_ACTIVITY_DEADZONE) & axis_mask
        predicted_nonzero = (predicted_xy.abs() >= MOUSE_ACTIVITY_DEADZONE) & axis_mask
        absolute_error = (predicted_xy - target_xy).abs()
        result[i] = torch.stack((
            axis_mask.sum(), nonzero.sum(), predicted_nonzero.sum(),
            ((predicted_xy.sign() == target_xy.sign()) & nonzero & predicted_nonzero).sum(),
            ((predicted_xy == target_xy) & nonzero).sum(),
            ((absolute_error <= .004) & nonzero).sum(),
            (absolute_error * axis_mask).sum(),
            (absolute_error * nonzero).sum(),
        )).to(torch.float64)
    return result


def format_mouse_axis_counts(counts):
    output = {}
    for name, row in [('all', counts.sum(0)), *zip(PROFILES, counts)]:
        (axes, target_nonzero, predicted_nonzero, sign_correct, exact_nonzero,
         within_004, error, nz_error) = row.cpu().tolist()
        if not axes:
            continue
        output[name] = {
            'axes': int(axes),
            'target_nonzero': int(target_nonzero),
            'predicted_nonzero': int(predicted_nonzero),
            'nonzero_sign_accuracy': sign_correct / target_nonzero if target_nonzero else None,
            'nonzero_exact_accuracy': exact_nonzero / target_nonzero if target_nonzero else None,
            'nonzero_within_004_accuracy': within_004 / target_nonzero if target_nonzero else None,
            'mae_all_axes': error / axes,
            'mae_nonzero_axes': nz_error / target_nonzero if target_nonzero else None,
        }
    return output


class DistributedWeightedSampler(Sampler):
    """Deterministic weighted replacement sampling, evenly sharded across ranks."""
    def __init__(self, weights, replicas, rank, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replicas, self.rank, self.seed, self.epoch = replicas, rank, seed, 0
        self.samples = (len(self.weights) + replicas - 1) // replicas

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.samples
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected = torch.multinomial(self.weights, self.samples * self.replicas,
                                     replacement=True, generator=generator)
        return iter(selected[self.rank::self.replicas].tolist())


class ProfileValidationSampler(Sampler):
    """Fixed, profile-stratified validation subset shared and sharded by rank."""
    def __init__(self, rows, samples_per_rank, replicas, rank, seed=12345):
        groups = {}
        for index, row in enumerate(rows): groups.setdefault(row['profile'], []).append(index)
        rng = np.random.default_rng(seed)
        groups = {name: rng.permutation(values).tolist() for name, values in sorted(groups.items())}
        names = list(groups)
        selected = []
        for position in range(samples_per_rank * replicas):
            name = names[position % len(names)]
            offset = position // len(names)
            selected.append(groups[name][offset % len(groups[name])])
        self.indices = selected[rank::replicas]

    def __len__(self): return len(self.indices)
    def __iter__(self): return iter(self.indices)


def move_conditions(conditions, device):
    return {key: value.to(device, non_blocking=True) for key, value in conditions.items()}


def lookup_text(cache, ids, device, dtype):
    ids = ids.long().cpu()
    return (cache["encoder_hidden"][ids].to(device=device, dtype=dtype, non_blocking=True),
            cache["attention_mask"][ids].to(device=device, dtype=torch.bool, non_blocking=True))


def lookup_current_text(cache, batch, device, dtype):
    hidden, mask = lookup_text(cache, batch['current_text_id'], device, dtype)
    if batch['current_text_id'].ndim == 2 and 'current_text_valid' in batch:
        mask = mask & batch['current_text_valid'].to(device=device, dtype=torch.bool)[..., None]
    return hidden, mask


def loader_worker_options(workers, timeout):
    """Build DataLoader options without turning transient shared-FS stalls fatal."""
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if timeout < 0:
        raise ValueError("loader timeout must be nonnegative")
    if workers == 0:
        return {}
    return {
        "timeout": timeout,
        "persistent_workers": True,
        "multiprocessing_context": "spawn",
    }


def load_renderer(path, *, return_config=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["config"].get("renderer", checkpoint["config"])
    renderer = Renderer(RendererArgs(**config))
    renderer.load_state_dict(checkpoint["model"], strict=True)
    return (renderer, checkpoint["config"]) if return_config else renderer


def policy_horizon_weights(model, device, tail_horizon_weight=1.0):
    if tail_horizon_weight < 0:
        raise ValueError('tail horizon weight must be nonnegative')
    if model.cfg.horizons == 8:
        return torch.arange(8, 0, -1, device=device)
    weights = torch.ones(model.cfg.horizons, device=device)
    weights[:model.cfg.execution_horizon] = 2
    weights[model.cfg.execution_horizon:] = tail_horizon_weight
    return weights


def future_latent_delta_loss(logits, latent, policy_indices, valid, family_id,
                             temporal_downsampling):
    """Predict pooled VAE-latent change at pixel offsets +8 and +16.

    The future latent is a detached label.  Only the eight completed policy
    frames enter the action decoder; this helper never feeds the label back as
    an input, so inference remains causal and unchanged.
    """
    prediction = logits.get('future_latent')
    if prediction is None:
        return None
    if policy_indices.ndim != 3 or prediction.ndim != 6:
        raise ValueError('future latent supervision requires dense policy anchors')
    b, anchors = policy_indices.shape[:2]
    if prediction.shape[:2] != (b, anchors) or prediction.shape[2] != 2:
        raise ValueError('future latent prediction shape does not match dense anchors')
    last = policy_indices[..., -1].long()
    pixel_offsets = torch.tensor((0, 8, 16), device=last.device)
    latent_indices = torch.div(
        last[..., None] + pixel_offsets, temporal_downsampling,
        rounding_mode='floor')
    if (latent_indices < 0).any() or (latent_indices >= latent.shape[1]).any():
        raise ValueError('future latent target falls outside encoded context')
    rows = torch.arange(b, device=latent.device)[:, None, None]
    selected = latent.detach()[rows, latent_indices]
    grid = prediction.shape[-1]
    pooled = F.adaptive_avg_pool2d(
        selected.flatten(0, 2), (grid, grid)).view(
            b, anchors, 3, latent.shape[2], grid, grid)
    target = pooled[:, :, 1:] - pooled[:, :, :1]
    if target.shape != prediction.shape:
        raise ValueError(
            f'future latent target {tuple(target.shape)} != prediction {tuple(prediction.shape)}')
    per_anchor = F.smooth_l1_loss(
        prediction.float(), target.float(), reduction='none').mean((-1, -2, -3, -4))
    builder = family_id == FAMILY_TO_ID['language_builder']
    mask = valid.bool().any(-1) & builder[:, None]
    return (per_anchor * mask).sum() / mask.sum().clamp_min(1)


def should_snapshot(step, final_step, snapshot_every):
    if snapshot_every < 1:
        raise ValueError('snapshot interval must be positive')
    return step % snapshot_every == 0 or step == final_step


def scheduled_learning_rate(step, *, peak_lr, total_steps, warmup_steps=0,
                            schedule='constant', min_lr_ratio=.01):
    """Absolute update schedule; resume never restarts the warmup."""
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    if schedule == 'constant':
        return peak_lr
    if schedule != 'cosine':
        raise ValueError('unknown learning-rate schedule')
    fraction = min(1., max(0., (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * .5 * (1 + math.cos(math.pi * fraction)))


def reduce_loss_parts(parts, valid, horizon_weights, sample_weight=None):
    """Use the exact per-anchor, then per-sample reduction of the action loss."""
    anchors = valid.shape[1] if valid.ndim == 3 else 1
    valid = valid.reshape(-1, valid.shape[-1])
    template = next(iter(parts.values()))
    weights = valid.to(template) * horizon_weights.to(template)[None]
    denominator = weights.sum(-1)
    samples = (denominator > 0).to(template)
    if sample_weight is not None:
        samples = samples * sample_weight.to(template).repeat_interleave(anchors)
    reduced = {name: (((value * weights).sum(-1) / denominator.clamp_min(1e-8))
                      * samples).sum() / samples.sum().clamp_min(1)
               for name, value in parts.items()}
    reduced['discrete_group'] = reduced['keys'] + reduced['key_groups'] + reduced['hotbar']
    reduced['mouse_group'] = (reduced['mouse_x'] + reduced['mouse_y']) * .5
    return reduced


def shared_action_gradient_diagnostics(raw, reduced, mouse_loss_weight):
    """Single-device probe of the first shared action self-attention matrix."""
    head = raw.families['language_builder'].head
    candidates = [(name, parameter) for name, parameter in head.named_parameters()
                  if name.endswith('self_attn.in_proj_weight') and parameter.requires_grad]
    if not candidates:
        raise ValueError('gradient probe requires a shared action self-attention matrix')
    name, parameter = candidates[0]
    discrete = torch.autograd.grad(reduced['discrete_group'], parameter,
                                  retain_graph=True, allow_unused=True)[0]
    mouse = torch.autograd.grad(reduced['mouse_group'], parameter,
                               retain_graph=True, allow_unused=True)[0]
    if discrete is None or mouse is None:
        return {'parameter': name, 'active': False}
    discrete, mouse = discrete.float(), mouse.float()
    coefficient = 2. if mouse_loss_weight is None else mouse_loss_weight
    dn, mn = discrete.norm(), mouse.norm()
    return {'parameter': name, 'active': True, 'discrete_norm': float(dn),
            'mouse_raw_norm': float(mn), 'mouse_weighted_norm': float(mn * coefficient),
            'cosine': float((discrete * mouse).sum() / (dn * mn).clamp_min(1e-12))}


def record_provenance(args, output):
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''): digest.update(chunk)
        return digest.hexdigest()
    files = {name: getattr(args, name, None) for name in ('m3_checkpoint', 'train_index', 'val_index',
             'vocabulary', 'text_catalog', 'text_cache', 'wan_vae', 'pixel_vae', 'warm_start',
             'warm_start_builder_vision', 'warm_start_builder_decoder', 'warm_start_builder_npc',
             'warm_start_builder_history', 'warm_start_action_v2')}
    result = {'inputs': {name: {'path': str(Path(path).resolve()), 'sha256': sha256(path)}
                         for name, path in files.items() if path}}
    if args.resume:
        result['resume_checkpoint'] = {'path': str(Path(args.resume).resolve()), 'sha256': sha256(args.resume)}
    root = Path(__file__).resolve().parents[1]
    source = [path for folder in ('miworld', 'train_scripts', 'dataset_toolkits')
              for path in (root / folder).rglob('*') if path.suffix in ('.py', '.sh')]
    result['source'] = {str(path.relative_to(root)): sha256(path) for path in sorted(source)}
    source_id = hashlib.sha256(json.dumps(result['source'], sort_keys=True).encode()).hexdigest()[:12]
    snapshot = output.parent / f'source-{source_id}'
    for path in source:
        target = snapshot / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(path, target)
    result['source_snapshot'] = str(snapshot.resolve())
    return result


@torch.no_grad()
def causal_preflight(raw, codec, text_cache, batch, device, precision):
    """Test anchor 40, inside the legacy 8-latent attention block (latent 10)."""
    raw.eval()
    cutoff = int(batch['policy_indices'][0, 1, -1])
    latent_prefix = cutoff // raw.temporal_downsampling + 1
    conditions = move_conditions(batch['conditions'], device)
    rgb = batch['rgb'].to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        latent = codec.encode(rgb)
        perturbed = rgb.clone(); perturbed[:, cutoff + 1:] = 1 - perturbed[:, cutoff + 1:]
        changed = codec.encode(perturbed)
        torch.testing.assert_close(latent[:, :latent_prefix], changed[:, :latent_prefix], atol=1e-5, rtol=1e-5)
        del rgb, perturbed
        shared, shared_mask = lookup_text(text_cache, batch['shared_text_id'], device, latent.dtype)
        current, current_mask = lookup_current_text(text_cache, batch, device, latent.dtype)
        args = (batch['family_id'].to(device), batch['profile_id'].to(device))
        kwargs = dict(shared_text=shared, shared_text_mask=shared_mask, current_text=current,
                      current_text_mask=current_mask, policy_indices=batch['policy_indices'][:, 1:2].to(device))
        _, first = raw(latent, conditions, *args, **kwargs)
        # Modify every time-indexed condition after the decision, preserving
        # shape/type validity by copying an existing earlier state.
        changed_conditions = {}
        for key, value in conditions.items():
            if key.startswith('interaction_history'):
                history_cutoff = cutoff + raw.cfg.builder_action_history - 1
                value = value.clone(); value[:, history_cutoff+1:] = 0
            elif value.ndim >= 2 and value.shape[1] == 129:
                value = value.clone(); value[:, cutoff + 1:] = value[:, cutoff:cutoff + 1]
            changed_conditions[key] = value
        if current.ndim == 4 and current.shape[1] == batch['rgb'].shape[1]:
            future_text = current.clone(); future_mask = current_mask.clone()
            future_text[:, cutoff+1:] = -13 * future_text[:, cutoff+1:] + 7
            future_mask[:, cutoff+1:] = ~future_mask[:, cutoff+1:]
            kwargs.update(current_text=future_text, current_text_mask=future_mask)
        _, second = raw(changed, changed_conditions, *args, **kwargs)
        for key in first: torch.testing.assert_close(first[key], second[key], atol=2e-3, rtol=2e-3)
    raw.train()
    progress('causal_preflight_passed', pixel_anchor=cutoff, latent_prefix=latent_prefix)


@torch.no_grad()
def evaluate(model, codec, text_cache, loader, device, batches, precision,
             mouse_nonzero_weight=1.0, tail_horizon_weight=1.0, mouse_loss_weight=None):
    model.eval(); values = []; prefixes = []; tails = []
    component_values = []
    reference_values = []
    counts = torch.zeros(len(PROFILES), 21, device=device, dtype=torch.float64)
    mouse_counts = torch.zeros(len(PROFILES), 8, device=device, dtype=torch.float64)
    iterator = iter(loader)
    for number in range(batches):
        started = time.monotonic()
        progress('validation_fetch', batch=number)
        try:
            batch = next(iterator)
        except StopIteration:
            break
        progress('validation_transfer', batch=number,
                 samples=batch['sample_index'].tolist())
        conditions = move_conditions(batch["conditions"], device)
        progress('validation_vae', batch=number)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            latent = codec.encode(batch["rgb"].to(device, non_blocking=True))
            torch.cuda.synchronize(device)
            progress('validation_forward', batch=number)
            dtype = latent.dtype
            shared, shared_mask = lookup_text(text_cache, batch["shared_text_id"], device, dtype)
            current, current_mask = lookup_current_text(text_cache, batch, device, dtype)
            family_id = batch["family_id"].to(device)
            _, logits = model(latent, conditions, family_id,
                              batch["profile_id"].to(device), shared_text=shared,
                              shared_text_mask=shared_mask, current_text=current,
                              current_text_mask=current_mask,
                              policy_indices=batch["policy_indices"].to(device))
            loss, parts = model.loss(logits, batch["target_actions"].to(device),
                                 batch["valid_mask"].to(device),
                                 horizon_weights=policy_horizon_weights(
                                     model, device, tail_horizon_weight),
                                 mouse_nonzero_weight=mouse_nonzero_weight,
                                 mouse_loss_weight=mouse_loss_weight)
            values.append(loss)
            reduced = reduce_loss_parts(parts, batch['valid_mask'].to(device),
                                        policy_horizon_weights(model, device, tail_horizon_weight))
            future_latent = future_latent_delta_loss(
                logits, latent, batch['policy_indices'].to(device),
                batch['valid_mask'].to(device), family_id,
                codec.temporal_downsampling)
            if future_latent is not None:
                reduced['future_latent'] = future_latent
            component_values.append(reduced)
            # Fixed legacy objective for comparisons with the existing v5 run.
            reference, _ = model.loss(logits, batch['target_actions'].to(device),
                batch['valid_mask'].to(device),
                horizon_weights=policy_horizon_weights(model, device, tail_horizon_weight),
                mouse_nonzero_weight=4., mouse_loss_weight=None)
            reference_values.append(reference)
            if model.cfg.horizons >= 8:
                counts += action_counts(model, logits, batch['target_actions'].to(device),
                                        batch['valid_mask'].to(device), batch['profile_id'].to(device))
                mouse_counts += mouse_axis_counts(
                    model, logits, batch['target_actions'].to(device),
                    batch['valid_mask'].to(device), batch['profile_id'].to(device))
                sections = [(slice(0, 8), prefixes)]
                if model.cfg.horizons > 8:
                    sections.append((slice(8, model.cfg.horizons), tails))
                for section, results in sections:
                    sliced = {k: (v[..., section] if k.endswith('_residual')
                                  else v[..., section, :]) for k, v in logits.items()
                              if k != 'future_latent'}
                    part, _ = model.loss(sliced,
                        batch['target_actions'][..., section, :].to(device),
                        batch['valid_mask'][..., section].to(device),
                        horizon_weights=torch.ones(section.stop - section.start, device=device),
                        mouse_nonzero_weight=mouse_nonzero_weight,
                        mouse_loss_weight=mouse_loss_weight)
                    results.append(part)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite validation loss at batch {number}')
        progress('validation_batch_done', batch=number, loss=float(loss),
                 seconds=time.monotonic()-started)
    model.train()
    progress('validation_complete', batches=len(values))
    model.last_validation_parts = {}
    if component_values:
        components = {name: torch.stack([row[name] for row in component_values]).mean()
                      for name in component_values[0]}
        components['reference_legacy_nonzero4_loss'] = torch.stack(reference_values).mean()
        for name, value in components.items():
            if dist.is_initialized():
                dist.all_reduce(value); value /= dist.get_world_size()
            model.last_validation_parts[name] = float(value)
    if dist.is_initialized():
        dist.all_reduce(counts)
        dist.all_reduce(mouse_counts)
    model.last_validation_actions = format_action_counts(counts)
    model.last_validation_mouse = format_mouse_axis_counts(mouse_counts)
    tail_name = f'tail{max(0, model.cfg.horizons - 8)}_loss'
    for name, results in (('first8_loss', prefixes), (tail_name, tails)):
        if results:
            result = torch.stack(results).mean()
            if dist.is_initialized(): dist.all_reduce(result); result /= dist.get_world_size()
            model.last_validation_parts[name] = float(result)
    return torch.stack(values).mean() if values else torch.tensor(float("nan"), device=device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--train-index", required=True); p.add_argument("--val-index", required=True)
    p.add_argument("--vocabulary", required=True); p.add_argument("--text-catalog", required=True)
    p.add_argument("--text-cache", required=True); p.add_argument("--m3-checkpoint", required=True)
    p.add_argument("--pixel-vae"); p.add_argument("--wan-vae")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume"); p.add_argument("--policy-blocks", type=int, default=4,
                                          help="0 inserts one policy block at every video layer")
    p.add_argument("--structured-inputs", action="store_true")
    p.add_argument('--resume-layout-change', action='store_true',
                   help='Explicit world-size/accumulation migration: retain optimizer, reseed sampler/RNG')
    p.add_argument('--resume-sampling-change', action='store_true',
                   help='Explicit new data/sampler run: retain weights, optimizer and step; reset sampler/RNG')
    p.add_argument('--resume-text-change', action='store_true',
                   help='Explicit component-text migration; retain model tensors, optimizer, sampler/RNG and step')
    p.add_argument('--current-text-mode',
                   choices=('episode', 'components', 'structured_goal', 'structured_goal_only'),
                   default='episode')
    p.add_argument('--current-text-scale', type=float, default=1.)
    p.add_argument('--mask-component-boundaries', action='store_true',
                   help='Drop execute8 anchors or truncate tail24 at unseen component-text changes')
    p.add_argument('--builder-place-anchor-fraction', type=float, default=0.,
                   help='Repeat place-positive dense anchors to this fraction inside builder clips')
    p.add_argument('--validation-label', choices=('heldout', 'train_replay'), default='heldout')
    p.add_argument("--warm-start", help="Weights-only initialization; new optimizer and step counter")
    p.add_argument('--warm-start-builder-vision',
                   help='Load a one-query policy, add builder visual queries, and reset optimizer')
    p.add_argument('--builder-visual-queries', type=int, default=1,
                   help='Parallel text-guided visual readers per builder frame')
    p.add_argument('--warm-start-builder-decoder',
                   help='Keep old policy/readout weights, add persistent spatial slots and action decoder; reset optimizer')
    p.add_argument('--builder-spatial-tokens', type=int, default=0)
    p.add_argument('--builder-action-decoder-layers', type=int, default=0)
    p.add_argument('--builder-action-decoder-hidden', type=int, default=512)
    p.add_argument('--builder-action-decoder-heads', type=int, default=8)
    p.add_argument('--builder-action-architecture', choices=('legacy', 'joint_v2'), default='legacy')
    p.add_argument('--builder-action-visual-grid', type=int, default=8)
    p.add_argument('--warm-start-action-v2',
                   help='Keep non-head policy weights; randomly initialize joint_v2 action head and a new optimizer')
    p.add_argument('--builder-npc-attention', action='store_true',
                   help='Unified causal NPC tokens with per-layer main/component text attention')
    p.add_argument('--warm-start-builder-npc',
                   help='Migrate persistent spatial decoder weights into NPC attention; reset optimizer')
    p.add_argument('--builder-goal-conditioned-vision', action='store_true',
                   help='Use global task text to steer every NPC spatial visual query')
    p.add_argument('--warm-start-goal-conditioned-vision',
                   help='Add zero-start text-to-visual-query gates to an NPC policy; reset optimizer')
    p.add_argument('--builder-text-only-visual-queries', action='store_true',
                   help='Build NPC visual queries only from structured task text and learned slot offsets')
    p.add_argument('--warm-start-text-only-visual-queries',
                   help='Replace gated state visual queries with pure task-text visual queries')
    p.add_argument('--warm-start-future-latent',
                   help='Load an existing policy, add future-latent prediction, and reset optimizer')
    p.add_argument('--builder-action-history', type=int, choices=(0,32,64), default=0)
    p.add_argument('--builder-text-action-film', action='store_true')
    p.add_argument('--builder-text-action-film-hidden', type=int, default=256)
    p.add_argument('--builder-future-latent-channels', type=int, default=0)
    p.add_argument('--builder-future-latent-grid', type=int, default=8)
    p.add_argument('--future-latent-loss-weight', type=float, default=.05)
    p.add_argument('--builder-interaction-pairs', action='store_true')
    p.add_argument('--builder-history-action-dropout', type=float, default=0.,
                   help='Drop complete 8-frame teacher-action history blocks during training')
    p.add_argument('--warm-start-builder-history',
                   help='Add zero-residual action history; retain existing AdamW moments, reset sampler/RNG')
    p.add_argument("--horizons", type=int, choices=(8, 16, 32), default=8)
    p.add_argument("--dense-stride", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--checkpoint-policy", action="store_true")
    p.add_argument("--gpu-memory-fraction", type=float, default=1.)
    p.add_argument("--causal-preflight", action="store_true")
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=10000); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument('--action-head-lr', type=float,
                   help='Optional separate peak LR for joint_v2 action head; remaining policy uses --lr')
    p.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    p.add_argument('--lr-warmup-steps', type=int, default=0)
    p.add_argument('--lr-min-ratio', type=float, default=.01)
    p.add_argument('--gradient-diagnostics-every', type=int, default=0,
                   help='Single-GPU only: record discrete/mouse gradients on shared action attention')
    p.add_argument('--mouse-nonzero-weight', type=float, default=1.,
                   help='Multiply mouse-axis CE when the quantized target bin is nonzero')
    p.add_argument('--mouse-loss-weight', type=float,
                   help='Opt into discrete-group + weight * mean(mouse X, mouse Y); omitted keeps legacy sum')
    p.add_argument('--continuous-mouse', action='store_true',
                   help='Train expected mouse delta with Smooth L1 instead of categorical bin CE')
    p.add_argument('--tail-horizon-weight', type=float, default=1.,
                   help='Loss weight for predicted actions after the executed prefix')
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument('--snapshot-every', type=int, default=1000,
                   help='Retain a numbered checkpoint at this interval')
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=32); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=("balanced", "uniform", "index"), default="balanced")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--context-frames", type=int, default=65)
    p.add_argument(
        "--loader-timeout", type=int, default=0,
        help=("Seconds to wait for a DataLoader batch. Zero disables the hard timeout; "
              "this is the safe default for high-latency shared storage."),
    )
    p.add_argument('--validate-on-resume', action='store_true',
                   help='Complete the normal validation on every rank before further updates.')
    p.add_argument('--eval-only', action='store_true',
                   help='Load --resume weights and run held-out evaluation without optimizer updates.')
    p.add_argument('--initial-replay-reference', help='Require initial replay to reproduce this baseline JSON')
    a = p.parse_args()
    if sum(bool(value) for value in (a.resume, a.warm_start, a.warm_start_builder_vision,
                                     a.warm_start_builder_decoder, a.warm_start_builder_npc,
                                     a.warm_start_builder_history, a.warm_start_action_v2,
                                     a.warm_start_goal_conditioned_vision,
                                     a.warm_start_text_only_visual_queries,
                                     a.warm_start_future_latent)) > 1:
        p.error('choose only one resume or warm-start mode')
    if a.resume_layout_change and not a.resume: p.error('--resume-layout-change requires --resume')
    if a.resume_sampling_change and (not a.resume or a.resume_layout_change):
        p.error('--resume-sampling-change requires resume and cannot combine with layout migration')
    if a.resume_text_change and (not a.resume or a.resume_sampling_change or a.resume_layout_change):
        p.error('--resume-text-change requires resume and cannot combine with other migrations')
    if a.grad_accum < 1 or a.steps < 1: p.error('steps and grad-accum must be positive')
    if a.mouse_nonzero_weight < 1: p.error('--mouse-nonzero-weight must be at least one')
    if a.mouse_loss_weight is not None and (not math.isfinite(a.mouse_loss_weight) or a.mouse_loss_weight < 0):
        p.error('--mouse-loss-weight must be finite and nonnegative')
    if not 0 <= a.lr_warmup_steps < a.steps:
        p.error('--lr-warmup-steps must be nonnegative and smaller than total steps')
    if not 0 <= a.lr_min_ratio <= 1:
        p.error('--lr-min-ratio must be in [0,1]')
    if a.gradient_diagnostics_every < 0:
        p.error('--gradient-diagnostics-every must be nonnegative')
    if a.builder_action_visual_grid < 1:
        p.error('--builder-action-visual-grid must be positive')
    if a.warm_start_action_v2 and a.builder_action_architecture != 'joint_v2':
        p.error('--warm-start-action-v2 requires --builder-action-architecture joint_v2')
    if a.action_head_lr is not None and (
            not math.isfinite(a.action_head_lr) or a.action_head_lr <= 0
            or a.builder_action_architecture != 'joint_v2'
            or not (a.warm_start or a.warm_start_action_v2
                    or a.warm_start_goal_conditioned_vision
                    or a.warm_start_text_only_visual_queries
                    or a.warm_start_future_latent or a.resume)):
        p.error('--action-head-lr requires positive LR and joint_v2 warm-start/resume')
    if a.tail_horizon_weight < 0: p.error('--tail-horizon-weight must be nonnegative')
    if a.builder_future_latent_channels < 0 or a.builder_future_latent_grid < 1:
        p.error('future latent channels must be nonnegative and grid positive')
    if not math.isfinite(a.future_latent_loss_weight) or a.future_latent_loss_weight < 0:
        p.error('--future-latent-loss-weight must be finite and nonnegative')
    if (not a.resume
            and bool(a.builder_future_latent_channels) != bool(a.warm_start_future_latent)):
        p.error('future latent channels and --warm-start-future-latent must be enabled together')
    if a.warm_start_future_latent and (
            a.horizons < 16 or a.builder_action_architecture != 'joint_v2'):
        p.error('future latent warm-start requires horizon16 joint_v2')
    if a.snapshot_every < 1: p.error('--snapshot-every must be positive')
    if not 0 < a.gpu_memory_fraction <= 1: p.error('gpu-memory-fraction must be in (0,1]')
    if bool(a.dense_stride) != (a.horizons > 8):
        p.error('16/32 horizons require dense-stride=8; 8 horizons require dense-stride=0')
    if a.validate_on_resume and not (a.resume or a.warm_start or a.warm_start_builder_vision or a.warm_start_builder_decoder or a.warm_start_builder_npc or a.warm_start_builder_history or a.warm_start_action_v2 or a.warm_start_goal_conditioned_vision or a.warm_start_text_only_visual_queries or a.warm_start_future_latent):
        p.error('--validate-on-resume requires resume or a builder warm-start')
    if a.workers < 0:
        p.error('--workers must be nonnegative')
    if a.loader_timeout < 0:
        p.error('--loader-timeout must be nonnegative')
    if not 0 <= a.builder_place_anchor_fraction <= 1:
        p.error('--builder-place-anchor-fraction must be in [0,1]')
    if a.builder_visual_queries < 1:
        p.error('--builder-visual-queries must be positive')
    if a.builder_spatial_tokens < 0 or a.builder_action_decoder_layers < 0:
        p.error('builder spatial tokens and decoder layers must be nonnegative')
    if a.builder_spatial_tokens and not a.builder_action_decoder_layers:
        p.error('persistent spatial tokens require an action decoder')
    if a.builder_action_decoder_layers and (
            a.builder_action_decoder_heads < 1 or a.builder_action_decoder_hidden < 1 or
            a.builder_action_decoder_hidden % a.builder_action_decoder_heads):
        p.error('action decoder width must be positive and divisible by heads')
    if a.builder_visual_queries > 1 and not (
            a.warm_start or a.warm_start_builder_vision
            or a.warm_start_builder_decoder or a.resume):
        p.error('multi-query builder training requires a builder warm-start or resume')
    if a.builder_action_decoder_layers and not (
            a.warm_start or a.warm_start_builder_decoder or a.warm_start_builder_npc
            or a.warm_start_builder_history or a.warm_start_action_v2
            or a.warm_start_goal_conditioned_vision
            or a.warm_start_text_only_visual_queries
            or a.warm_start_future_latent or a.resume):
        p.error('action decoder training requires a decoder/NPC warm-start or --resume')
    if a.warm_start_builder_decoder and not a.builder_action_decoder_layers:
        p.error('--warm-start-builder-decoder requires decoder layers')
    if a.builder_npc_attention and (not a.builder_spatial_tokens or a.builder_visual_queries != 1):
        p.error('NPC attention requires spatial tokens and --builder-visual-queries 1')
    if a.builder_npc_attention and not (
            a.warm_start or a.warm_start_builder_npc
            or a.warm_start_builder_history or a.warm_start_action_v2
            or a.warm_start_goal_conditioned_vision
            or a.warm_start_text_only_visual_queries
            or a.warm_start_future_latent or a.resume):
        p.error('NPC attention requires --warm-start-builder-npc or --resume')
    if a.warm_start_builder_npc and not a.builder_npc_attention:
        p.error('--warm-start-builder-npc requires --builder-npc-attention')
    if a.builder_goal_conditioned_vision and not a.builder_npc_attention:
        p.error('--builder-goal-conditioned-vision requires --builder-npc-attention')
    if a.warm_start_goal_conditioned_vision and not a.builder_goal_conditioned_vision:
        p.error('--warm-start-goal-conditioned-vision requires --builder-goal-conditioned-vision')
    if a.builder_text_only_visual_queries and not a.builder_npc_attention:
        p.error('--builder-text-only-visual-queries requires --builder-npc-attention')
    if a.builder_text_only_visual_queries and a.builder_goal_conditioned_vision:
        p.error('pure-text and gated visual queries are mutually exclusive')
    if a.warm_start_text_only_visual_queries and not a.builder_text_only_visual_queries:
        p.error('--warm-start-text-only-visual-queries requires pure-text visual queries')
    if a.builder_interaction_pairs and (a.builder_place_anchor_fraction or not a.dense_stride):
        p.error('interaction pair sampling replaces old place-anchor resampling and requires dense anchors')
    if not 0 <= a.builder_history_action_dropout <= 1:
        p.error('--builder-history-action-dropout must be in [0,1]')
    if a.builder_history_action_dropout and not a.builder_action_history:
        p.error('--builder-history-action-dropout requires --builder-action-history')
    if a.warm_start_builder_history and not a.builder_action_history:
        p.error('history migration requires --builder-action-history')
    if a.builder_action_history and not (
            a.warm_start or a.warm_start_builder_history or a.warm_start_action_v2
            or a.warm_start_goal_conditioned_vision
            or a.warm_start_text_only_visual_queries
            or a.warm_start_future_latent or a.resume):
        p.error('history training requires explicit history migration or resume')
    if a.initial_replay_reference and not a.validate_on_resume:
        p.error('initial replay reference requires --validate-on-resume')
    if a.eval_only:
        if not a.resume or a.validate_on_resume:
            p.error('--eval-only requires --resume and cannot combine with --validate-on-resume')
        if a.resume_layout_change or a.resume_sampling_change or a.resume_text_change:
            p.error('--eval-only cannot combine with resume migrations')
        if Path(a.output_dir).resolve() == Path(a.resume).resolve().parent:
            p.error('--eval-only requires a separate output directory')
    os.environ['POLICY_DIAGNOSTIC_DIR'] = str(Path(a.output_dir).parent / 'diagnostics')
    progress('initializing')
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    if a.gradient_diagnostics_every and world != 1:
        p.error('gradient diagnostics require a single-GPU probe; DDP autograd hooks are not reused')
    local = int(os.environ.get("LOCAL_RANK", "0")); requested = torch.device(a.device)
    if requested.type != "cuda": p.error("formal M3 policy training uses M3's CUDA voxel rasterizer")
    if world > 1: dist.init_process_group("nccl")
    device = torch.device(
        f"cuda:{local}" if world > 1
        else ("cuda:0" if requested.index is None else a.device)
    )
    torch.cuda.set_device(device); torch.manual_seed(a.seed + rank)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction, device)
    random.seed(a.seed + rank); np.random.seed(a.seed + rank)

    train = InsertedPolicyDataset(
        a.train_index, a.vocabulary, a.text_catalog, dataset_root=a.dataset_root,
        context_frames=a.context_frames, horizons=a.horizons, dense_stride=a.dense_stride,
        preserve_index_weights=a.sampling == 'index',
        current_text_mode=a.current_text_mode,
        mask_component_boundaries=a.mask_component_boundaries,
        builder_place_anchor_fraction=a.builder_place_anchor_fraction,
        builder_action_history=a.builder_action_history,
        builder_interaction_pairs=a.builder_interaction_pairs,
        builder_history_action_dropout=a.builder_history_action_dropout,
    )
    val = InsertedPolicyDataset(
        a.val_index, a.vocabulary, a.text_catalog, dataset_root=a.dataset_root,
        context_frames=a.context_frames, horizons=a.horizons, dense_stride=a.dense_stride,
        current_text_mode=a.current_text_mode,
        mask_component_boundaries=a.mask_component_boundaries,
        builder_action_history=a.builder_action_history,
    )
    weights = [float(row.get("sample_weight", 1.)) for row in train.rows]
    if a.sampling in ("balanced", "index"):
        sampler = DistributedWeightedSampler(weights, world, rank, a.seed)
    else:
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                     seed=a.seed)
    val_sampler = (ProfileValidationSampler(val.rows, a.val_batches * a.batch_size, world, rank)
                   if a.dense_stride else
                   (DistributedSampler(val, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None))
    worker_options = loader_worker_options(a.workers, a.loader_timeout)
    loader = DataLoader(train, batch_size=a.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=a.workers,
                        pin_memory=True, collate_fn=collate_inserted_policy, **worker_options)
    val_loader = DataLoader(val, batch_size=a.batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=a.workers, pin_memory=True,
                            collate_fn=collate_inserted_policy, **worker_options)

    renderer, renderer_config = load_renderer(a.m3_checkpoint, return_config=True)
    text_cache = load_file(str(a.text_cache), device="cpu")
    text_hidden = int(text_cache["encoder_hidden"].shape[-1])
    cfg = M3PolicyArgs(num_policy_blocks=a.policy_blocks, text_hidden_size=text_hidden,
                       structured_inputs=a.structured_inputs, horizons=a.horizons,
                       causal_observations=bool(a.dense_stride), checkpoint_policy=a.checkpoint_policy,
                       current_text_scale=a.current_text_scale,
                       builder_visual_queries=a.builder_visual_queries,
                       builder_spatial_tokens=a.builder_spatial_tokens,
                       builder_action_decoder_layers=a.builder_action_decoder_layers,
                       builder_action_decoder_hidden=a.builder_action_decoder_hidden,
                       builder_action_decoder_heads=a.builder_action_decoder_heads,
                       builder_action_architecture=a.builder_action_architecture,
                       builder_action_visual_grid=a.builder_action_visual_grid,
                       builder_npc_attention=a.builder_npc_attention,
                       builder_goal_conditioned_vision=a.builder_goal_conditioned_vision,
                       builder_text_only_visual_queries=a.builder_text_only_visual_queries,
                       builder_action_history=a.builder_action_history,
                       builder_text_action_film=a.builder_text_action_film,
                       builder_text_action_film_hidden=a.builder_text_action_film_hidden,
                       builder_future_latent_channels=a.builder_future_latent_channels,
                       builder_future_latent_grid=a.builder_future_latent_grid,
                       continuous_mouse=a.continuous_mouse)
    raw = M3Policy(renderer, cfg).to(device).train()
    cfg = raw.cfg
    codec_config = renderer_config.get("codec", {})
    if codec_config.get("type") == "wan2.1-vae":
        weights = a.wan_vae or codec_config.get("weights_path")
        if not weights:
            p.error("Wan M3 requires --wan-vae or codec.weights_path in its checkpoint")
        codec = WanRendererCodec(weights).to(device).eval()
    else:
        if not a.pixel_vae:
            p.error("Pixel M3 requires --pixel-vae")
        codec = RendererCodec.from_run_config(
            load_file(str(a.pixel_vae)), renderer_config
        ).to(device).eval()
    if codec.temporal_downsampling != raw.temporal_downsampling:
        raise ValueError("M3 and codec temporal downsampling disagree")
    if (a.builder_future_latent_channels
            and codec.latent_dim != a.builder_future_latent_channels):
        raise ValueError(
            f'configured future latent channels {a.builder_future_latent_channels} '
            f'do not match codec latent dim {codec.latent_dim}')
    if codec.pixel_frames_to_latent_frames(a.context_frames) != renderer.cfg.context_frames:
        raise ValueError("pixel context does not match the M3 latent context")
    if a.action_head_lr is None:
        optimizer = torch.optim.AdamW(raw.families.parameters(), lr=a.lr)
    else:
        head_params, policy_params = [], []
        for name, parameter in raw.families.named_parameters():
            (head_params if name.startswith('language_builder.head.') else policy_params).append(parameter)
        optimizer = torch.optim.AdamW([
            {'params': policy_params, 'lr': a.lr, 'role': 'policy'},
            {'params': head_params, 'lr': a.action_head_lr, 'role': 'action_head'},
        ], lr=a.lr)
    start = 0
    resume_state = None
    previous_provenance = None
    layout_migration = None
    sampling_migration = None
    text_migration = None
    initial_epoch = 0
    migration = None
    if a.warm_start:
        checkpoint = torch.load(a.warm_start, map_location='cpu', weights_only=False, mmap=True)
        expanded = raw.warm_start_horizons(checkpoint['families'], checkpoint['config'].get('horizons', 8))
        migration = {'source': str(Path(a.warm_start).resolve()), 'source_step': checkpoint['step'],
                     'resized_or_added_heads': expanded,
                     'method': f'resize horizon {checkpoint["config"].get("horizons", 8)}->{a.horizons}; '
                               'zero-init continuous mouse residuals',
                     'optimizer': 'new AdamW; NOT exact resume', 'new_updates': a.steps}
        progress('warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_builder_vision:
        checkpoint = torch.load(a.warm_start_builder_vision, map_location='cpu',
                                weights_only=False, mmap=True)
        if checkpoint['config'].get('builder_visual_queries', 1) != 1:
            raise ValueError('builder-vision warm start source must use one visual query')
        new_parameters = raw.warm_start_builder_vision(checkpoint['families'])
        start = int(checkpoint['step'])
        migration = {
            'source': str(Path(a.warm_start_builder_vision).resolve()),
            'source_step': start,
            'method': 'exact old tensors plus new text-guided builder visual queries',
            'new_parameters': new_parameters,
            'optimizer': 'new AdamW; NOT exact resume',
            'step_counter': 'continued from source for experiment chronology',
        }
        progress('builder_vision_warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_builder_decoder:
        checkpoint = torch.load(a.warm_start_builder_decoder, map_location='cpu',
                                weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_builder_decoder).resolve().parent:
            raise ValueError('builder decoder migration requires a new output directory')
        new_parameters = raw.warm_start_builder_decoder(checkpoint['families'], checkpoint['config'])
        start = int(checkpoint['step'])
        migration = {
            'source': str(Path(a.warm_start_builder_decoder).resolve()), 'source_step': start,
            'method': 'persistent builder spatial slots and parallel action-query decoder; legacy tensors preserved',
            'new_parameters': new_parameters, 'optimizer': 'new AdamW; NOT exact resume',
            'initialization': 'zero decoder output projection; old horizon readouts retained per query',
            'step_counter': 'continued from source for experiment chronology',
        }
        progress('builder_decoder_warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_builder_npc:
        checkpoint = torch.load(a.warm_start_builder_npc, map_location='cpu',
                                weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_builder_npc).resolve().parent:
            raise ValueError('NPC migration requires a new output directory')
        changes = raw.warm_start_builder_npc(checkpoint['families'], checkpoint['config'])
        start = int(checkpoint['step'])
        migration = {
            'source': str(Path(a.warm_start_builder_npc).resolve()), 'source_step': start,
            'method': 'unified causal NPC tokens; per-layer main and component text attention',
            'parameters': changes, 'optimizer': 'new AdamW; NOT exact resume',
            'initialization': 'compatible weights retained; changed graph does NOT preserve builder logits',
            'step_counter': 'continued from source for experiment chronology',
        }
        progress('builder_npc_warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_builder_history:
        checkpoint = torch.load(a.warm_start_builder_history, map_location='cpu', weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_builder_history).resolve().parent:
            raise ValueError('history migration requires a new output directory')
        changes = raw.warm_start_builder_history(checkpoint['families'], checkpoint['config'])
        if checkpoint['training']['lr'] != a.lr:
            raise ValueError('history migration must preserve learning rate')
        preserved = migrate_history_optimizer(optimizer, raw.families, checkpoint['optimizer'],
                                              changes.pop('source_parameter_names'))
        start = int(checkpoint['step'])
        migration = dict(source=str(Path(a.warm_start_builder_history).resolve()), source_step=start,
            method='zero-residual 64/32-frame action and event history with interaction pair sampling',
            parameters=changes, optimizer='old parameter moments retained; new history states start empty',
            optimizer_states_preserved=preserved, rng_sampler='reset; NOT exact resume',
            initialization='zero history output preserves baseline logits', loss='unchanged')
        progress('builder_history_warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_goal_conditioned_vision:
        checkpoint = torch.load(a.warm_start_goal_conditioned_vision, map_location='cpu',
                                weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_goal_conditioned_vision).resolve().parent:
            raise ValueError('goal-conditioned vision migration requires a new output directory')
        changes = raw.warm_start_goal_conditioned_vision(
            checkpoint['families'], checkpoint['config'])
        start = int(checkpoint['step'])
        source_text_mode = checkpoint.get('training', {}).get('current_text_mode', 'episode')
        text_input_changed = source_text_mode != a.current_text_mode
        migration = dict(
            source=str(Path(a.warm_start_goal_conditioned_vision).resolve()),
            source_step=start,
            method='zero-start global-task gates steer every NPC spatial visual query',
            parameters={'added': changes}, optimizer='new AdamW; parent optimizer NOT restored',
            initialization=('all added gates are zero; parameters preserve the parent function, '
                'but full logits may change because current_text input changed'
                if text_input_changed else
                'all added gates are zero; parent logits preserved exactly'),
            text_input={'source_mode': source_text_mode, 'target_mode': a.current_text_mode,
                        'changed': text_input_changed},
            step_counter='continued from source for experiment chronology', loss='unchanged')
        progress('goal_conditioned_vision_warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_text_only_visual_queries:
        checkpoint = torch.load(a.warm_start_text_only_visual_queries, map_location='cpu',
                                weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_text_only_visual_queries).resolve().parent:
            raise ValueError('pure-text visual-query migration requires a new output directory')
        changes = raw.warm_start_text_only_visual_queries(
            checkpoint['families'], checkpoint['config'])
        start = int(checkpoint['step'])
        migration = dict(
            source=str(Path(a.warm_start_text_only_visual_queries).resolve()),
            source_step=start,
            method=('structured task text directly creates every spatial visual query; '
                    'state tokens no longer participate in visual-query construction'),
            parameters=changes,
            optimizer='new AdamW; parent optimizer NOT restored',
            initialization=('shared text projection and all compatible policy/action weights retained; '
                            'per-slot query offsets newly initialized'),
            text_input={'source_mode': checkpoint.get('training', {}).get(
                            'current_text_mode', 'episode'),
                        'target_mode': a.current_text_mode,
                        'changed': checkpoint.get('training', {}).get(
                            'current_text_mode', 'episode') != a.current_text_mode},
            step_counter='continued from source for experiment chronology', loss='unchanged')
        progress('text_only_visual_queries_warm_start_loaded', **migration)
        del checkpoint
    if a.warm_start_future_latent:
        checkpoint = torch.load(a.warm_start_future_latent, map_location='cpu',
                                weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_future_latent).resolve().parent:
            raise ValueError('future latent migration requires a new output directory')
        source_cfg = M3PolicyArgs(**checkpoint['config'])
        expected_cfg = replace(
            source_cfg,
            builder_future_latent_channels=a.builder_future_latent_channels,
            builder_future_latent_grid=a.builder_future_latent_grid,
        )
        if cfg != expected_cfg:
            changed = {name: (getattr(source_cfg, name), getattr(cfg, name))
                       for name in asdict(cfg)
                       if getattr(expected_cfg, name) != getattr(cfg, name)}
            raise ValueError(f'future latent migration changed unrelated policy config: {changed}')
        incompatible = raw.families.load_state_dict(checkpoint['families'], strict=False)
        source_has_future_latent = bool(source_cfg.builder_future_latent_channels)
        expected_missing = {
            name for name, _ in raw.families.named_parameters()
            if '.future_latent_action_' in name
            or (not source_has_future_latent and '.future_latent_head.' in name)
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError(
                f'invalid future latent migration: missing={incompatible.missing_keys} '
                f'unexpected={incompatible.unexpected_keys}')
        start = int(checkpoint['step'])
        migration = dict(
            source=str(Path(a.warm_start_future_latent).resolve()),
            source_step=start,
            method=('predict detached VAE latent deltas at +8/+16, then spatially '
                    'cross-attend every action query to the predicted visual plan'),
            parameters={'added': sorted(expected_missing)},
            optimizer='new AdamW; parent optimizer NOT restored',
            initialization=('future-latent head retained; new attention output projection is zero, '
                            'so all parent action logits are preserved exactly before the first update'),
            step_counter='continued from source for experiment chronology',
            loss_weight=a.future_latent_loss_weight,
        )
        progress('future_latent_warm_start_loaded', **migration)
        print(f'FUTURE_LATENT_LOADED rank={rank} source={migration["source"]} '
              f'source_step={start} optimizer_states={len(optimizer.state)}', flush=True)
        del checkpoint
    if a.warm_start_action_v2:
        checkpoint = torch.load(a.warm_start_action_v2, map_location='cpu', weights_only=False, mmap=True)
        if Path(a.output_dir).resolve() == Path(a.warm_start_action_v2).resolve().parent:
            raise ValueError('action v2 migration requires a new output directory')
        declared_renderer = checkpoint.get('m3_checkpoint')
        if not declared_renderer or Path(declared_renderer).resolve() != Path(a.m3_checkpoint).resolve():
            expected_hash = checkpoint.get('provenance', {}).get('inputs', {}).get('m3_checkpoint', {}).get('sha256')
            actual_hash = hashlib.sha256()
            with Path(a.m3_checkpoint).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    actual_hash.update(chunk)
            if not expected_hash or actual_hash.hexdigest() != expected_hash:
                raise ValueError('action v2 parent renderer lineage does not match --m3-checkpoint')
        changes = raw.warm_start_action_v2(checkpoint['families'], checkpoint['config'])
        migration = dict(source=str(Path(a.warm_start_action_v2).resolve()),
            source_step=int(checkpoint['step']), parameters=changes,
            method='retain non-head policy tensors; randomly initialize joint action decoder with direct visual memory',
            optimizer='new AdamW; optimizer and sampler states NOT restored',
            step_counter='new action experiment starts at zero',
            renderer_checkpoint=str(Path(a.m3_checkpoint).resolve()))
        progress('action_v2_warm_start_loaded', **migration)
        print(f'ACTION_V2_LOADED rank={rank} source={migration["source"]} '
              f'source_step={migration["source_step"]} new_step=0 optimizer_states={len(optimizer.state)}', flush=True)
        del checkpoint
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
        raw.families.load_state_dict(checkpoint["families"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); start = int(checkpoint["step"])
        check_policy_config(checkpoint['config'], cfg, text_change=a.resume_text_change)
        if a.resume_text_change:
            if Path(a.output_dir).resolve() == Path(checkpoint['training']['output_dir']).resolve():
                raise ValueError('text migration requires a new output directory')
            for key in ('train_index', 'val_index'):
                check_text_only_index(checkpoint['training'][key], getattr(a, key))
            text_migration = {'source_step': start, 'checkpoint': str(Path(a.resume).resolve()),
                'optimizer': 'preserved', 'rng_sampler': 'preserved; inputs changed, NOT bitwise continuation',
                'old_mode': checkpoint['training'].get('current_text_mode', 'episode'),
                'new_mode': a.current_text_mode,
                'old_current_text_scale': checkpoint['config'].get('current_text_scale', 1.),
                'new_current_text_scale': a.current_text_scale, 'shared_text': 'unchanged', 'loss': 'unchanged'}
            progress('text_migration', **text_migration)
        elif checkpoint['training'].get('current_text_mode', 'episode') != a.current_text_mode:
            raise ValueError('text mode change requires explicit text migration')
        if 'runtime' in checkpoint and not a.eval_only:
            if a.resume_sampling_change:
                if Path(a.output_dir).resolve() == Path(a.resume).resolve().parent:
                    raise ValueError('sampling migration requires a separate output directory')
                if len(checkpoint['runtime']) != world or checkpoint['training']['grad_accum'] != a.grad_accum:
                    raise ValueError('sampling-only migration must preserve world size and accumulation')
                initial_epoch = 0
                sampling_migration = {'checkpoint': str(Path(a.resume).resolve()), 'source_step': start,
                    'optimizer': 'preserved', 'rng_sampler': 'reseeded; NOT exact continuation',
                    'old_sampling': checkpoint['training']['sampling'], 'new_sampling': a.sampling,
                    'loss': 'unchanged', 'world_size': world, 'grad_accum': a.grad_accum}
                seed = a.seed + 1000003 * start + rank
                torch.manual_seed(seed); np.random.seed(seed % 2**32); random.seed(seed)
                progress('sampling_migration', **sampling_migration)
            elif a.resume_layout_change:
                if Path(a.output_dir).resolve() == Path(a.resume).resolve().parent:
                    raise ValueError('layout migration requires a separate output directory')
                initial_epoch = max(s['epoch'] for s in checkpoint['runtime']) + 1
                layout_migration = {
                    'checkpoint': str(Path(a.resume).resolve()), 'source_step': start,
                    'old_world_size': len(checkpoint['runtime']), 'new_world_size': world,
                    'old_grad_accum': checkpoint['training']['grad_accum'], 'new_grad_accum': a.grad_accum,
                    'optimizer': 'preserved', 'rng_sampler': 'reseeded; not bitwise/exact continuation',
                    'initial_epoch': initial_epoch,
                }
                torch.manual_seed(a.seed + 1000003 * start + rank)
                np.random.seed((a.seed + 1000003 * start + rank) % 2**32)
                random.seed(a.seed + 1000003 * start + rank)
                progress('layout_migration', **layout_migration)
            else:
                if len(checkpoint['runtime']) != world: raise ValueError('exact resume requires same world size')
                resume_state = checkpoint['runtime'][rank]
                if checkpoint['training']['grad_accum'] != a.grad_accum: raise ValueError('resume accumulation mismatch')
            for field in ('batch_size', 'sampling', 'seed', 'precision', 'lr', 'context_frames',
                          'builder_action_history', 'builder_interaction_pairs',
                          'builder_history_action_dropout',
                          'mouse_nonzero_weight', 'mouse_loss_weight', 'tail_horizon_weight',
                          'lr_schedule', 'lr_warmup_steps', 'lr_min_ratio', 'action_head_lr'):
                if a.resume_sampling_change and field == 'sampling': continue
                default = {'mouse_nonzero_weight': 1., 'mouse_loss_weight': None,
                           'tail_horizon_weight': 1., 'lr_schedule': 'constant',
                           'lr_warmup_steps': 0, 'lr_min_ratio': .01, 'action_head_lr': None,
                           'builder_history_action_dropout': 0.}.get(field, 0)
                if checkpoint['training'].get(field, default) != getattr(a, field):
                    raise ValueError(f'exact resume training mismatch: {field}')
            if a.lr_schedule == 'cosine' and checkpoint['training']['steps'] != a.steps:
                raise ValueError('cosine resume requires the same total steps to preserve the LR trajectory')
        elif a.dense_stride and not a.eval_only:
            raise ValueError('dense resume requires sampler and RNG state; use warm-start for old checkpoints')
        migration = checkpoint.get('migration')
        if not a.resume_layout_change: layout_migration = checkpoint.get('layout_migration')
        if not a.resume_sampling_change: sampling_migration = checkpoint.get('sampling_migration')
        if not a.resume_text_change: text_migration = checkpoint.get('text_migration')
        previous_provenance = checkpoint.get('provenance')
        progress('resume_loaded', step=start, optimizer_states=len(optimizer.state))
        print(f'RESUME_LOADED rank={rank} step={start} optimizer_states={len(optimizer.state)}', flush=True)
        del checkpoint
    model = (DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False,
                                     find_unused_parameters=True) if world > 1 else raw)
    output = Path(a.output_dir)
    provenance = None
    if rank == 0:
        if (output / 'config.json').exists() and (not a.resume or a.resume_layout_change or a.resume_sampling_change or a.resume_text_change):
            raise FileExistsError(f'output already initialized: {output}; choose a new experiment directory')
        output.mkdir(parents=True, exist_ok=True)
        if a.dense_stride:
            progress('hashing_provenance')
            provenance = record_provenance(a, output)
            if previous_provenance and not a.eval_only:
                changes = check_resume_provenance(previous_provenance, provenance,
                    sampling_change=a.resume_sampling_change, layout_change=a.resume_layout_change,
                    text_change=a.resume_text_change)
                if a.resume_sampling_change: sampling_migration.update(changes)
                elif a.resume_layout_change: layout_migration.update(changes)
                elif a.resume_text_change: text_migration.update(changes)
            (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        (output / "config.json").write_text(json.dumps(
            {"policy": asdict(cfg), "training": vars(a), "codec": codec.get_config(),
             "contract": f"predict {a.horizons}; execute {cfg.execution_horizon}; dense stride {a.dense_stride}",
             "effective_clip_batch": a.batch_size * world * a.grad_accum,
             "migration": migration, "layout_migration": layout_migration, 'sampling_migration': sampling_migration,
             'text_migration': text_migration,
             "train_clips": len(train), "val_clips": len(val)}, indent=2))
        if a.dense_stride:
            (output / 'validation_subset.json').write_text(json.dumps({
                'seed': 12345, 'label': a.validation_label, 'profiles': sorted({row['profile'] for row in val.rows}),
                'indices_by_rank': [ProfileValidationSampler(val.rows, a.val_batches * a.batch_size,
                                    world, r).indices for r in range(world)]}, indent=2))
    if world > 1: dist.barrier()

    if a.eval_only:
        progress('evaluation_only', step=start)
        metric = evaluate(raw, codec, text_cache, val_loader, device, a.val_batches, a.precision,
                          a.mouse_nonzero_weight, a.tail_horizon_weight, a.mouse_loss_weight)
        if world > 1:
            dist.all_reduce(metric)
            metric /= world
        result = {'step': start, 'checkpoint': str(Path(a.resume).resolve()),
                  'structured_loss': float(metric), 'label': a.validation_label,
                  **raw.last_validation_parts,
                  'first8_actions': raw.last_validation_actions,
                  'first8_mouse': raw.last_validation_mouse,
                  'val_batches_per_rank': a.val_batches, 'world_size': world}
        if rank == 0:
            (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    if a.causal_preflight:
        if not a.dense_stride: raise ValueError('causal-preflight requires dense training')
        causal_preflight(raw, codec, text_cache, next(iter(loader)), device, a.precision)

    if a.validate_on_resume:
        progress('resume_validation', step=start)
        metric = evaluate(raw, codec, text_cache, val_loader, device, a.val_batches, a.precision,
                          a.mouse_nonzero_weight, a.tail_horizon_weight, a.mouse_loss_weight)
        if world > 1: dist.all_reduce(metric); metric /= world
        torch.cuda.synchronize(device)
        if not torch.isfinite(metric):
            raise FloatingPointError('non-finite resume validation metric')
        if a.initial_replay_reference:
            reference = json.loads(Path(a.initial_replay_reference).read_text())
            if reference['step'] != start or abs(float(metric)-reference['structured_loss']) > 2e-4:
                raise ValueError('initial replay loss does not reproduce the source checkpoint')
            if raw.last_validation_actions != reference['first8_actions']:
                raise ValueError('initial action metrics do not reproduce the source checkpoint')
            if ('first8_mouse' in reference
                    and raw.last_validation_mouse != reference['first8_mouse']):
                raise ValueError('initial mouse metrics do not reproduce the source checkpoint')
        if rank == 0:
            with (output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps({'step': start, 'structured_loss': float(metric),
                    'phase': 'resume_gate', 'label': a.validation_label,
                    **raw.last_validation_parts, 'first8_actions': raw.last_validation_actions,
                    'first8_mouse': raw.last_validation_mouse}) + '\n')
        progress('resume_validation_passed', step=start, loss=float(metric))
        print(f'RESUME_VALIDATION_PASSED rank={rank} step={start} loss={float(metric)}', flush=True)
        if world > 1: dist.barrier()

    progress('loader_start')
    epoch = initial_epoch if resume_state is None else resume_state['epoch']
    consumed = 0 if resume_state is None else resume_state['consumed']
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(consumed): next(iterator)
    if resume_state is not None:
        torch.set_rng_state(resume_state['torch'])
        torch.cuda.set_rng_state(resume_state['cuda'], device)
        np.random.set_state(resume_state['numpy']); random.setstate(resume_state['python'])
    horizon_weights = policy_horizon_weights(raw, device, a.tail_horizon_weight)
    for step in range(start + 1, a.steps + 1):
        started = time.monotonic()
        learning_rate = scheduled_learning_rate(step, peak_lr=a.lr, total_steps=a.steps,
            warmup_steps=a.lr_warmup_steps, schedule=a.lr_schedule, min_lr_ratio=a.lr_min_ratio)
        for group in optimizer.param_groups:
            peak_lr = a.action_head_lr if group.get('role') == 'action_head' else a.lr
            group['lr'] = scheduled_learning_rate(step, peak_lr=peak_lr, total_steps=a.steps,
                warmup_steps=a.lr_warmup_steps, schedule=a.lr_schedule, min_lr_ratio=a.lr_min_ratio)
        optimizer.zero_grad(set_to_none=True)
        reported = torch.zeros((), device=device)
        reported_parts = {}
        gradient_diagnostics = None
        valid_targets = 0
        for micro in range(a.grad_accum):
            progress('fetch', step=step, micro=micro)
            try: batch = next(iterator)
            except StopIteration:
                epoch += 1; consumed = 0; sampler.set_epoch(epoch)
                iterator = iter(loader); batch = next(iterator)
            consumed += 1
            if not batch['valid_mask'].any():
                raise ValueError(f'clip has no valid dense targets: {batch["sample_index"].tolist()}')
            valid_targets += int(batch['valid_mask'].sum())
            progress('transfer', step=step, micro=micro, samples=batch['sample_index'].tolist(),
                     profiles=batch['sample_profile'],
                     strata=[train.rows[i].get('sampling_stratum', 'unspecified') for i in batch['sample_index'].tolist()])
            conditions = move_conditions(batch["conditions"], device)
            progress('vae', step=step, micro=micro)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                latent = codec.encode(batch["rgb"].to(device, non_blocking=True))
            progress('forward', step=step, micro=micro)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                shared, shared_mask = lookup_text(text_cache, batch["shared_text_id"], device, latent.dtype)
                current, current_mask = lookup_current_text(text_cache, batch, device, latent.dtype)
                family_id = batch["family_id"].to(device)
                _, logits = model(latent, conditions, family_id,
                                  batch["profile_id"].to(device), shared_text=shared,
                                  shared_text_mask=shared_mask, current_text=current,
                                  current_text_mask=current_mask,
                                  policy_indices=batch["policy_indices"].to(device))
                loss, parts = raw.loss(
                    logits, batch["target_actions"].to(device), batch["valid_mask"].to(device),
                    horizon_weights=horizon_weights,
                    sample_weight=batch["sample_weight"].to(device) if a.sampling == "uniform" else None,
                    mouse_nonzero_weight=a.mouse_nonzero_weight,
                    mouse_loss_weight=a.mouse_loss_weight)
                reduced = reduce_loss_parts(parts, batch['valid_mask'].to(device), horizon_weights,
                    batch['sample_weight'].to(device) if a.sampling == 'uniform' else None)
                future_latent = future_latent_delta_loss(
                    logits, latent, batch['policy_indices'].to(device),
                    batch['valid_mask'].to(device), family_id,
                    codec.temporal_downsampling)
                if future_latent is not None:
                    loss = loss + a.future_latent_loss_weight * future_latent
                    reduced['future_latent'] = future_latent
            for name, value in reduced.items():
                reported_parts[name] = reported_parts.get(name, torch.zeros_like(value)) + value.detach() / a.grad_accum
            if (a.gradient_diagnostics_every and micro == 0
                    and step % a.gradient_diagnostics_every == 0):
                gradient_diagnostics = shared_action_gradient_diagnostics(raw, reduced, a.mouse_loss_weight)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite M3 policy loss at step {step}")
            progress('backward', step=step, micro=micro)
            # Synchronize each microbatch: family routes differ across samples
            # and ranks, so no_sync would require special unused-gradient handling.
            (loss / a.grad_accum).backward()
            reported += loss.detach() / a.grad_accum
            del latent, logits, loss, parts, reduced, conditions, batch
        torch.cuda.synchronize(device)
        progress('optimizer', step=step)
        torch.nn.utils.clip_grad_norm_(raw.families.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        progress('loss_reduce', step=step)
        if world > 1: dist.all_reduce(reported); reported /= world
        for value in reported_parts.values():
            if world > 1: dist.all_reduce(value); value /= world
        if rank == 0:
            record = {'step': step, 'loss': reported.item(), 'seconds': time.monotonic()-started,
                      'lr': learning_rate, 'loss_parts': {name: float(value) for name, value in reported_parts.items()},
                      'lr_groups': {group.get('role', 'policy'): group['lr'] for group in optimizer.param_groups},
                      'valid_targets_rank0': valid_targets,
                      'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 2**30}
            if gradient_diagnostics is not None:
                record['shared_action_gradients'] = gradient_diagnostics
            print(json.dumps(record), flush=True)
            with (output / 'train.jsonl').open('a') as handle: handle.write(json.dumps(record) + '\n')
        progress('step_done', step=step, seconds=time.monotonic()-started)
        if step % a.validate_every == 0 or step == a.steps:
            progress('validation', step=step)
            metric = evaluate(raw, codec, text_cache, val_loader, device, a.val_batches, a.precision,
                              a.mouse_nonzero_weight, a.tail_horizon_weight, a.mouse_loss_weight)
            if world > 1: dist.all_reduce(metric); metric /= world
            if not torch.isfinite(metric): raise FloatingPointError('non-finite validation loss')
            if rank == 0:
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, "structured_loss": float(metric),
                        'label': a.validation_label, **raw.last_validation_parts,
                        'first8_actions': raw.last_validation_actions,
                        'first8_mouse': raw.last_validation_mouse}) + "\n")
        if step == start + 1 or step % a.save_every == 0 or step == a.steps:
            progress('save', step=step)
            runtime = {'epoch': epoch, 'consumed': consumed, 'torch': torch.get_rng_state(),
                       'cuda': torch.cuda.get_rng_state(device), 'numpy': np.random.get_state(),
                       'python': random.getstate()}
            states = [None] * world
            if world > 1: dist.all_gather_object(states, runtime)
            else: states[0] = runtime
            if rank == 0:
                destination = output / 'latest.pt'; temporary = destination.with_suffix('.tmp')
                torch.save({'families': raw.families.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step, 'config': asdict(cfg), 'runtime': states,
                            'training': vars(a), 'migration': migration,
                            'layout_migration': layout_migration,
                            'sampling_migration': sampling_migration,
                            'text_migration': text_migration,
                            'provenance': provenance,
                            'm3_checkpoint': str(Path(a.m3_checkpoint).resolve())}, temporary)
                temporary.replace(destination)
                if should_snapshot(step, a.steps, a.snapshot_every):
                    shutil.copy2(destination, output / f'step_{step:07d}.pt')
            if world > 1: dist.barrier()
    # Stop persistent CPU workers before the interpreter/CUDA runtime exits.
    for active_loader in (loader, val_loader):
        active_iterator = getattr(active_loader, '_iterator', None)
        if active_iterator is not None:
            active_iterator._shutdown_workers()
            active_loader._iterator = None
    progress('finished')
    if world > 1: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__": main()
