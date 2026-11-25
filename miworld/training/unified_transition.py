"""Shared-trunk M2 supervision and typed interaction decoding.

Address space: 13^3 voxels, A residents, null. Operation: remove/place/attack.
The dataset/committer support at most one effective write per actor per step;
multi-write labels are invalid, not null negatives. No hard raycast is assumed.
"""
import torch
from torch.nn import functional as F

from miworld.models.m2_decode import (
    ATTACK,
    PLACE,
    REMOVE,
    VOXELS,
    decode_unified_interactions,
)


def _mean(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def unified_transition_loss(model, output, inputs, targets, event_count_positive_weight=1.):
    """Supervise dense state and a SINGLE sequence containing edits and attacks.

    Unknown frames make ordered slot positions/count ambiguous: exclude that
    actor-window from event supervision, but retain valid state supervision.
    Overflow is reported/excluded rather than silently clipping event counts.
    """
    from miworld.training.transition_trainer import ordered_event_targets

    state = targets['state_valid'].bool() & inputs['active'][:, None].bool()
    address = targets['address'].long()
    null = output['event_address_logits'].shape[-1] - 1
    queries = output['event_time_logits'].shape[2]
    valid = targets['address_valid'].bool() & state
    if torch.any(valid & ((address < 0) | (address > null))):
        raise ValueError('unified target address is out of range')
    positive = valid & (address != null)
    actual_count = positive.sum(1)
    complete = valid.all(1) & inputs['active'].bool()
    overflow = complete & (actual_count > queries)
    supervised = complete & ~overflow
    event_valid = valid & supervised[:, None]
    count, slot_address, slot_time, slot_valid, matches = ordered_event_targets(
        address, event_valid, null, queries)
    # Unused slots select null, avoiding accidental gather of an unknown voxel.
    slot_address = torch.where(slot_valid, slot_address, null)
    teacher_block, teacher_damage = model.event_payloads(output, slot_address)
    zero = sum(x.float().clamp(-1e4, 1e4).sum() * 0. for x in (
        output['event_time_logits'], output['event_address_logits'],
        output['event_operation_logits'], teacher_block, teacher_damage))
    terms = dict(
        position=_mean(F.smooth_l1_loss(output['pose'][..., :3], targets['pose'][..., :3],
                                       reduction='none'), state),
        angle=_mean(1 - torch.cos(output['pose'][..., 3:] - targets['pose'][..., 3:]), state),
        camera_position=_mean(F.smooth_l1_loss(output['camera_relative'], targets['camera_relative'],
                                              reduction='none'), state),
        camera_direction=_mean(1 - F.cosine_similarity(output['camera_direction'],
                                                       targets['camera_direction'], dim=-1), state),
    )
    held_valid = state & targets.get('held_item_valid', torch.ones_like(state)).bool()
    held_target = torch.where(held_valid, targets['held_item'], 0)
    terms['held_item'] = _mean(F.cross_entropy(output['held_logits'].flatten(0, 2),
                                              held_target.flatten(), reduction='none').reshape_as(state), held_valid)
    count_ce = F.cross_entropy(output['event_count_logits'].flatten(0, 1), count.flatten(),
                              reduction='none').reshape_as(count)
    count_weights = torch.where(count > 0, event_count_positive_weight, 1.)
    terms['event_count'] = _mean(count_ce * count_weights, supervised)
    for name in ('event_time', 'event_address', 'event_operation', 'event_block', 'event_damage'):
        terms[name] = zero
    if matches:
        bi, ai, qi, ti = (torch.tensor([m[k] for m in matches], device=address.device)
                          for k in range(4))
        selected = slot_address[bi, ai, qi]
        selected_logits = output['event_address_logits'][bi, ai, qi]
        if not torch.isfinite(selected_logits.gather(-1, selected[:, None])).all():
            raise ValueError('supervised event points to unknown/self/padded target')
        terms['event_time'] = F.cross_entropy(output['event_time_logits'][bi, ai, qi].float(), ti)
        terms['event_address'] = F.cross_entropy(selected_logits.float(), selected)
        voxel_event = selected < VOXELS
        block_valid = targets['block_valid'][bi, ti, ai].bool() & voxel_event
        damage_valid = targets['damage_valid'][bi, ti, ai].bool() & ~voxel_event
        operation = torch.full_like(selected, ATTACK)
        operation_valid = ~voxel_event | block_valid
        # Confirmed post-edit payload determines remove versus place. The air
        # class is resolved from the dataset vocabulary, never assumed to be 0.
        operation = torch.where(voxel_event,
                                torch.where(targets['block'][bi, ti, ai] == model.cfg.air_class,
                                            REMOVE, PLACE), operation)
        if 'edit_kind' in targets:
            kind = targets['edit_kind'][bi, ti, ai]
            kind_valid = voxel_event & ((kind == 1) | (kind == 2))
            operation = torch.where(kind_valid, kind - 1, operation)
            operation_valid |= kind_valid
        if operation_valid.any():
            terms['event_operation'] = F.cross_entropy(
                output['event_operation_logits'][bi, ai, qi][operation_valid].float(),
                operation[operation_valid])
        if block_valid.any():
            terms['event_block'] = F.cross_entropy(teacher_block[bi, ai, qi][block_valid].float(),
                                                   targets['block'][bi, ti, ai][block_valid])
        if damage_valid.any():
            target_damage = targets['damage'][bi, ti, ai][damage_valid]
            if torch.any(target_damage >= 0):
                raise ValueError('attack targets must be negative observed HP deltas')
            terms['event_damage'] = F.smooth_l1_loss(teacher_damage[bi, ai, qi][damage_valid].float(),
                                                    target_damage.float())
    loss = sum(terms.values())
    with torch.no_grad():
        decoded = model.decode_interactions(output)
        pred = decoded['address']
        predicted = valid & (pred != null)
        correct = positive & (pred == address)
        block_mask = valid & (address < VOXELS) & targets['block_valid'].bool()
        attack_mask = valid & (address >= VOXELS) & (address < null)
        block_correct = block_mask & (pred == address) & (decoded['block_payload'] == targets['block'])
        metrics = {name: float(value.detach()) for name, value in terms.items()}
        metrics.update(loss=float(loss.detach()), event_labels=int(positive.sum()),
                       predicted_events=int(predicted.sum()), correct_events=int(correct.sum()),
                       event_recall=float(correct.sum()/positive.sum().clamp_min(1)),
                       event_precision=float(correct.sum()/predicted.sum().clamp_min(1)),
                       block_labels=int(block_mask.sum()), correct_edits=int(block_correct.sum()),
                       attack_labels=int(attack_mask.sum()), correct_attacks=int((correct & attack_mask).sum()),
                       supervised_event_windows=int(supervised.sum()),
                       incomplete_event_windows=int((inputs['active'].bool() & ~complete).sum()),
                       overflow_event_windows=int(overflow.sum()),
                       position_error=float(_mean((output['pose'][..., :3]-targets['pose'][..., :3]).norm(dim=-1),state)))
        if 'memory_gates' in output:
            gates = output['memory_gates'].float()
            metrics.update(memory_gate_mean=float(gates.mean()),
                           memory_gate_min=float(gates.min()),
                           memory_gate_max=float(gates.max()))
    return loss, metrics
