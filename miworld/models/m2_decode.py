"""Inference-only decoding for M2's unified interaction address space."""

import torch


VOXELS = 13**3
REMOVE, PLACE, ATTACK = 0, 1, 2


@torch.no_grad()
def decode_unified_interactions(model, output):
    """Resolve interaction slots into dense writes with type checks."""
    raw = output["event_address_logits"].float()
    batch, actors, queries, candidates = raw.shape
    null = candidates - 1
    if null != VOXELS + actors:
        raise ValueError("unexpected unified address space")

    count = output["event_count_logits"].argmax(-1)
    operation = output["event_operation_logits"].argmax(-1)
    time = output["event_time_logits"].argmax(-1)
    ids = torch.arange(candidates, device=raw.device)
    voxel = ids < VOXELS
    resident = (ids >= VOXELS) & (ids < null)
    allowed = torch.where((operation == ATTACK)[..., None], resident, voxel)
    allowed = allowed | (ids == null)
    typed = raw.masked_fill(~allowed, -torch.inf)
    address = typed.argmax(-1)

    block_logits, damage = model.event_payloads(output, address)
    block_logits = block_logits.clone()
    block_logits[..., model.cfg.air_class] = -torch.inf
    block = block_logits.argmax(-1)
    block = torch.where(operation == REMOVE, model.cfg.air_class, block)
    active = torch.arange(queries, device=raw.device) < count[..., None]
    active = active & output["active"][..., None] & (address != null)
    confidence = (
        typed.softmax(-1).amax(-1)
        * output["event_time_logits"].float().softmax(-1).amax(-1)
        * output["event_operation_logits"].float().softmax(-1).amax(-1)
    )

    frames = output["pose"].shape[1]
    dense_address = address.new_full((batch, frames, actors), null)
    dense_block = block.new_zeros((batch, frames, actors))
    dense_damage = damage.new_zeros((batch, frames, actors))
    best = confidence.new_full((batch, frames, actors), -1.0)
    for batch_index, actor_index, query_index in active.nonzero(
        as_tuple=False
    ).tolist():
        frame_index = int(time[batch_index, actor_index, query_index])
        score = confidence[batch_index, actor_index, query_index]
        if score <= best[batch_index, frame_index, actor_index]:
            continue
        best[batch_index, frame_index, actor_index] = score
        dense_address[batch_index, frame_index, actor_index] = address[
            batch_index, actor_index, query_index
        ]
        is_attack = operation[batch_index, actor_index, query_index] == ATTACK
        dense_block[batch_index, frame_index, actor_index] = torch.where(
            is_attack, 0, block[batch_index, actor_index, query_index]
        )
        dense_damage[batch_index, frame_index, actor_index] = torch.where(
            is_attack, damage[batch_index, actor_index, query_index], 0.0
        )

    return {
        "pose": output["pose"],
        "velocity": output["velocity"],
        "address": dense_address,
        "block_payload": dense_block,
        "hp_payload": dense_damage,
        "held_item": output["held_logits"].argmax(-1),
        "camera_relative": output["camera_relative"],
        "camera_direction": output["camera_direction"],
    }
