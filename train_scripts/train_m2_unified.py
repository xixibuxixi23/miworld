"""Train the production M2 Local World Writer from scratch or a strict resume.

The model has one joint-action trunk and two decoders: typed sparse events and
resident camera/state. Launch with torchrun for single- or multi-node DDP.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from miworld.data.transition_dataset import collate_transition
from miworld.data.transition_stream import TransitionStream
from miworld.models.local_world_writer import LocalWorldWriter, LocalWorldWriterArgs
from miworld.training.transition_trainer import transition_loss


COUNT_METRICS = {
    'event_labels', 'predicted_events', 'correct_events', 'block_labels',
    'correct_edits', 'attack_labels', 'correct_attacks',
    'supervised_event_windows', 'incomplete_event_windows',
    'overflow_event_windows',
}


def move(batch, device):
    return {g: {k: v.to(device) for k, v in batch[g].items()} for g in ('inputs', 'targets')}


def event_rates(metrics):
    result = dict(metrics)
    result['event_precision'] = result['correct_events'] / max(1, result['predicted_events'])
    result['event_recall'] = result['correct_events'] / max(1, result['event_labels'])
    p, r = result['event_precision'], result['event_recall']
    result['event_f1'] = 2 * p * r / max(1e-12, p + r)
    result['edit_recall'] = result['correct_edits'] / max(1, result['block_labels'])
    result['attack_recall'] = result['correct_attacks'] / max(1, result['attack_labels'])
    return result


def reduce_metrics(metrics, device, world_size):
    if world_size == 1:
        return event_rates(metrics)
    names = sorted(metrics)
    values = torch.tensor([float(metrics[name]) for name in names], device=device,
                          dtype=torch.float64)
    dist.all_reduce(values)
    reduced = {}
    for name, value in zip(names, values.tolist()):
        reduced[name] = value if name in COUNT_METRICS else value / world_size
    return event_rates(reduced)


@torch.no_grad()
def evaluate(model, loader, device, max_batches, event_count_positive_weight):
    model.eval()
    rows = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        values = move(batch, device)
        _, metrics = transition_loss(
            model, model(values['inputs']), **values, objective='unified',
            event_count_positive_weight=event_count_positive_weight)
        rows.append(metrics)
    if not rows:
        raise ValueError('validation stream contains no windows')
    result = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    for key in COUNT_METRICS:
        result[key] = sum(r[key] for r in rows)
    model.train()
    return event_rates(result)


def distributed_environment(requested_device):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        if not requested_device.startswith('cuda'):
            raise ValueError('distributed training requires CUDA')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return rank, world_size, local_rank, torch.device('cuda', local_rank)
    return rank, world_size, local_rank, torch.device(requested_device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-root', 'index', 'vocabulary', 'items', 'cache-root', 'output-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=4, help='Per-GPU batch size')
    p.add_argument('--workers', type=int, default=2, help='DataLoader workers per rank')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--event-queries', type=int, default=8)
    p.add_argument('--event-context-frames', type=int, default=4)
    p.add_argument('--edit-oversample', type=int, default=8)
    p.add_argument('--attack-oversample', type=int, default=32)
    p.add_argument('--event-count-positive-weight', type=float, default=4.)
    rgb = p.add_mutually_exclusive_group()
    rgb.add_argument('--use-rgb', dest='use_rgb', action='store_true')
    rgb.add_argument('--no-rgb', dest='use_rgb', action='store_false',
                     help='Voxel/state-only ablation; formal M2 uses RGB')
    p.set_defaults(use_rgb=True)
    p.add_argument('--memory-gate-init', type=float, default=.45)
    p.add_argument('--air-raw-id', type=int, default=126)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=128)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--resume', help='Local World Writer checkpoint only; use a fresh output directory')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.eval_batches,
           args.log_every, args.edit_oversample, args.attack_oversample) < 1:
        p.error('steps, batch size, save/eval/log intervals and oversampling must be positive')
    if args.workers < 0 or args.width % args.heads:
        p.error('workers must be nonnegative and width divisible by heads')
    if args.event_context_frames < 0 or args.event_count_positive_weight <= 0:
        p.error('event context must be nonnegative and positive weight must be positive')

    rank, world_size, local_rank, device = distributed_environment(args.device)
    out = Path(args.output_dir)
    nonempty = int(out.exists() and any(out.iterdir()))
    if world_size > 1:
        flag = torch.tensor(nonempty, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        nonempty = int(flag.item())
    if nonempty:
        raise ValueError('output directory must be new or empty; existing runs are never overwritten')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    item_data = json.loads(Path(args.items).read_text())
    items = item_data.get('items', item_data)
    cache = Path(args.cache_root)
    stamp = cache / 'unified_cache_manifest.json'
    cache_identity = dict(
        schema='m2-unified-cache-v1', items=items,
        dataset_root=str(Path(args.dataset_root).resolve()),
        index_sha256=hashlib.sha256(Path(args.index).read_bytes()).hexdigest(),
        vocabulary_sha256=hashlib.sha256(Path(args.vocabulary).read_bytes()).hexdigest())
    if rank == 0:
        if stamp.exists():
            if json.loads(stamp.read_text()) != cache_identity:
                raise ValueError('cache identity differs; use a fresh cache root')
        elif cache.exists() and any(cache.iterdir()):
            raise ValueError('legacy/nonempty cache has no unified identity; use a fresh cache root')
        else:
            cache.mkdir(parents=True, exist_ok=True)
            stamp.write_text(json.dumps(cache_identity, indent=2))
    if world_size > 1:
        dist.barrier()
    if not stamp.exists() or json.loads(stamp.read_text()) != cache_identity:
        raise ValueError('cache identity is missing or differs')

    train = TransitionStream(
        args.index, args.vocabulary, args.cache_root, 'train', args.seed,
        rank=rank, world_size=world_size, dataset_root=args.dataset_root, items=items,
        edit_oversample=args.edit_oversample, attack_oversample=args.attack_oversample)
    val = None
    if rank == 0:
        val = TransitionStream(
            args.index, args.vocabulary, args.cache_root, 'val_id', args.seed,
            fixed_order=True, dataset_root=args.dataset_root, items=items,
            edit_oversample=args.edit_oversample, attack_oversample=args.attack_oversample)
        train_seeds = {Path(r['path']).name.split('_seed')[-1] for r in train.records}
        val_seeds = {Path(r['path']).name.split('_seed')[-1] for r in val.records}
        if train_seeds & val_seeds:
            raise ValueError('train/validation scenario seeds overlap')

    classes = tuple(train.vocabulary.class_to_raw)
    if args.air_raw_id not in classes:
        raise ValueError('air raw ID is absent from block vocabulary')
    cfg = LocalWorldWriterArgs(
        len(classes), len(items), air_class=classes.index(args.air_raw_id),
        width=args.width, depth=args.depth, heads=args.heads,
        event_queries=args.event_queries, event_context_frames=args.event_context_frames,
        use_rgb=args.use_rgb, memory_gate_init=args.memory_gate_init)
    model = LocalWorldWriter(cfg).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    first_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        old = checkpoint['config']
        if old['model'] != asdict(cfg) or old['items'] != items or tuple(old['class_to_raw']) != classes:
            raise ValueError('resume requires identical architecture and vocabularies')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        first_step = int(checkpoint['step'])
        if first_step >= args.steps:
            raise ValueError('--steps must exceed the resumed checkpoint step')

    train_model = (DDP(model, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False) if world_size > 1 else model)
    torch.manual_seed(args.seed + rank)
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.workers,
                              collate_fn=collate_transition,
                              persistent_workers=args.workers > 0)
    val_loader = (DataLoader(val, batch_size=args.batch_size, num_workers=args.workers,
                             collate_fn=collate_transition,
                             persistent_workers=args.workers > 0) if rank == 0 else None)
    config = dict(
        schema='m2-local-world-writer-v1', training=vars(args), model=asdict(cfg),
        distributed=dict(world_size=world_size, per_gpu_batch=args.batch_size,
                         global_batch=world_size * args.batch_size),
        objective='unified', items=items, class_to_raw=classes,
        operations=['remove', 'place', 'attack'],
        resident_outputs=['pose_residual', 'velocity', 'camera', 'held_item'],
        architecture=['action+state tokens', '4-frame event history',
                      '13^3 voxel+RGB memory',
                      f'{cfg.depth}x temporal/resident/memory/ffn',
                      'event decoder', 'camera decoder'],
        parameters=sum(p.numel() for p in model.parameters()))
    if rank == 0:
        (out / 'config.json').write_text(json.dumps(config, indent=2))

    def log(step, split, metrics):
        if rank != 0:
            return
        row = dict(step=step, split=split, **metrics)
        with (out / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    iterator = iter(train_loader)
    best_event_f1 = -1.
    for step in range(first_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError('training stream contains no windows on this rank') from error
        values = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = train_model(values['inputs'])
        loss, metrics = transition_loss(
            model, output, **values, objective='unified',
            event_count_positive_weight=args.event_count_positive_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite unified loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('non-finite unified gradient')
        optimizer.step()
        metrics = reduce_metrics(dict(metrics, gradient_norm=float(norm)), device, world_size)
        if step % args.log_every == 0 or step == first_step + 1:
            log(step, 'train', metrics)

        if step % args.save_every == 0 or step == args.steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                tmp = out / 'checkpoint.tmp'
                payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               step=step, config=config)
                torch.save(payload, tmp)
                tmp.replace(out / 'checkpoint.pt')
                validation = evaluate(model, val_loader, device, args.eval_batches,
                                      args.event_count_positive_weight)
                log(step, 'val_event_balanced', validation)
                if validation['event_f1'] > best_event_f1:
                    best_event_f1 = validation['event_f1']
                    best_tmp = out / 'best-event-f1.tmp'
                    torch.save(payload, best_tmp)
                    best_tmp.replace(out / 'best-event-f1.pt')
                    (out / 'best-event-f1.json').write_text(json.dumps(
                        dict(step=step, event_f1=best_event_f1,
                             edit_recall=validation['edit_recall'],
                             attack_recall=validation['attack_recall']), indent=2))
            if world_size > 1:
                dist.barrier()
            train_model.train()

    if rank == 0:
        (out / 'COMPLETED.json').write_text(json.dumps(
            dict(steps=args.steps, schema='m2-local-world-writer-v1',
                 world_size=world_size)))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
