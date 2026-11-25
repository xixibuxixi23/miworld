"""Observed-world rollouts with predicted pose/camera history.

The observed-world diagnostic refreshes recorded crops and is explicitly
subject to GT crop-anchor localization cues. The primary rollout_batch API
uses fixed initial geometry with predicted edits shared across player crops.
HP, inventory and executed controls remain recorded boundary conditions.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_velocity_loss(output, inputs, targets):
    """Past-safe per-step velocity supervision derived from recorded poses."""
    valid = targets["state_valid"].bool() & inputs["active"][:, None].bool()
    recorded_initial = inputs.get("_recorded_initial_pose", inputs["initial_pose"])
    previous = torch.cat((recorded_initial[:, None, :, :3],
                          targets["pose"][:, :-1, :, :3]), dim=1)
    valid = valid & torch.cat((inputs["active"][:, None].bool(),
                              targets["state_valid"][:, :-1].bool()), dim=1)
    target = targets["pose"][..., :3] - previous
    element = F.smooth_l1_loss(output["velocity"], target, reduction="none")
    mask = valid[..., None].expand_as(element)
    loss = element.masked_fill(~mask, 0.).sum() / mask.sum().clamp_min(1)
    error = (output["velocity"] - target).norm(dim=-1)
    mae = error.masked_fill(~valid, 0.).sum() / valid.sum().clamp_min(1)
    return loss, {"velocity_loss": float(loss.detach()), "velocity_error": float(mae.detach())}


def rollout_inputs(observed, carry):
    """Rebase an observed crop on the carried pose without changing addresses."""
    inputs = dict(observed)
    if carry is None:
        return inputs
    initial = observed["initial_pose"]
    inputs["voxel_relative_xyz"] = (
        observed["voxel_relative_xyz"]
        + initial[:, :, None, :3] - carry["pose"][:, :, None, :3]
    )
    inputs["initial_pose"] = carry["pose"]
    inputs["camera_relative"] = carry["camera_relative"]
    inputs["camera_direction"] = carry["camera_direction"]
    # Kept for v1 comparison; v2 never reads the derived velocity input.
    if "velocity" in carry:
        inputs["initial_velocity"] = carry["velocity"]
    for key in ("engine_velocity", "motion_hidden", "engine_prior_velocity",
                "engine_prior_jump_step", "engine_prior_jump_origin_z"):
        if key in carry:
            inputs[key] = carry[key]
    for key in ("event_context_pose", "event_context_camera_relative",
                "event_context_camera_direction", "event_context_actions",
                "event_context_valid"):
        inputs[key] = carry[key]
    return inputs


def detach_carry(carry):
    """Cut temporal gradients at an explicitly chosen rollout boundary."""
    return {key: value.detach() for key, value in carry.items()}


def next_carry(output, inputs, decoded=None, *, detach=True):
    """Boundary t+T, with states t+T-H..t+T-1 and actions t+T-H..t+T-1.

For T=8 and H=4: predicted observations 4,5,6,7 are the next
history; observation 8 is current. The history controls are actions 4..7.
The endpoint is deliberately excluded from the state-history slice.
"""
    history = inputs["event_context_pose"].shape[1]
    if output["pose"].shape[1] < history + 1:
        raise ValueError("rollout block must predict at least history_frames+1 states")
    take = slice(-history - 1, -1)
    result = {
        "pose": output["pose"][:, -1],
        "camera_relative": output["camera_relative"][:, -1],
        "camera_direction": output["camera_direction"][:, -1],
        "event_context_pose": output["pose"][:, take],
        "event_context_camera_relative": output["camera_relative"][:, take],
        "event_context_camera_direction": output["camera_direction"][:, take],
        "event_context_actions": inputs["actions"][:, -history:],
        "event_context_valid": inputs["active"][:, None].bool().expand(-1, history, -1),
    }
    if "velocity" in output:
        result["velocity"] = output["velocity"][:, -1]
    for key in ("engine_velocity", "motion_hidden", "engine_prior_velocity",
                "engine_prior_jump_step", "engine_prior_jump_origin_z"):
        if key in output:
            result[key] = output[key][:, -1]
    if "_fixed_world_xyz" in inputs:
        if decoded is None:
            raise ValueError("fixed-world carry requires decoded predicted edits")
        voxels, world_stats = apply_shared_edits(inputs, decoded)
        result.update(_fixed_world_xyz=inputs["_fixed_world_xyz"],
                      _fixed_world_voxels=voxels,
                      _fixed_world_known=inputs["voxel_known"])
        if "previous_rgb" in inputs:
            result["_fixed_rgb"] = inputs["previous_rgb"]
        result.update(world_stats)
    return detach_carry(result) if detach else result


def fixed_initial_geometry_inputs(observed, carry, initial):
    """Freeze the first crop and its world coordinates for a drift diagnostic.

No later GT crop or crop-anchor position is passed to the model. The initial
geometry is not updated by predicted edits, so this remains a separate static
world diagnostic rather than a complete interactive world rollout.
"""
    fixed = dict(observed)
    for key in ("voxels", "voxel_relative_xyz", "initial_pose"):
        fixed[key] = initial[key]
    if "voxel_known" in initial:
        fixed["voxel_known"] = initial["voxel_known"]
    else:
        fixed.pop("voxel_known", None)
    if "previous_rgb" in initial:
        fixed["previous_rgb"] = initial["previous_rgb"]
    return rollout_inputs(fixed, carry)


def _local_index(world_xyz, origin):
    local = torch.round(world_xyz - origin).long()
    inside = ((local >= 0) & (local < 13)).all(-1)
    address = (local[..., 0] * 13 + local[..., 1]) * 13 + local[..., 2]
    return address, inside


def _validate_shared_initial(inputs, xyz):
    """Fail on conflicting known values in overlapping initial crops."""
    voxels = inputs["voxels"].flatten(2)
    known = inputs["voxel_known"].flatten(2).bool()
    actors = voxels.shape[1]
    for source in range(actors):
        for dest in range(source + 1, actors):
            index, inside = _local_index(xyz[:, source], xyz[:, dest, :1])
            safe = index.clamp(0, 13**3-1)
            scope = (inside & known[:, source] & known[:, dest].gather(1, safe)
                     & inputs["active"][:, source, None].bool()
                     & inputs["active"][:, dest, None].bool())
            if (scope & (voxels[:, source] != voxels[:, dest].gather(1, safe))).any():
                raise ValueError("initial player crops disagree at shared world coordinates")


@torch.no_grad()
def apply_shared_edits(inputs, decoded):
    """Apply predicted edits in (time, actor) order to every covering crop.

Same-frame conflicting edits resolve deterministically: the higher actor slot
writes last. This diagnostic ordering is recorded, not inferred from GT.
"""
    xyz = inputs["_fixed_world_xyz"]
    voxels = inputs["voxels"].clone()
    flat = voxels.flatten(2)
    addresses = decoded["address"].detach().cpu()
    batch, frames, actors = addresses.shape
    active = inputs["active"].detach().cpu().bool()
    conflicts, writes = 0, 0
    for ti in range(frames):
        frame_positions = set()
        for actor in range(actors):
            for bi in range(batch):
                address = int(addresses[bi, ti, actor])
                if not active[bi, actor] or not 0 <= address < 13**3:
                    continue
                position = xyz[bi, actor, address]
                key = (bi, *position.tolist())
                conflicts += int(key in frame_positions)
                frame_positions.add(key)
                writes += 1
                payload = decoded["block_payload"][bi, ti, actor]
                for dest in range(actors):
                    index, inside = _local_index(position, xyz[bi, dest, 0])
                    if bool(inside) and active[bi, dest]:
                        flat[bi, dest, int(index)] = payload
    return voxels, {
        "_fixed_shared_write_count": voxels.new_tensor(writes),
        "_fixed_same_frame_conflicts": voxels.new_tensor(conflicts),
    }


def rollout_batch(observed, targets, carry):
    """Primary adapter: fixed-anchor shared predicted world, remapped labels.

Later GT poses/crops are read ONLY while remapping supervision addresses.
They do not enter the returned model inputs. RGB is frozen at the first
boundary for v1; recorded HP/held/hotbar/actions remain external conditions.
Unrepresentable edit targets are marked invalid with address -1, never null.
"""
    inputs = dict(observed)
    if carry is None:
        xyz = torch.round(observed["voxel_relative_xyz"]
                          + observed["initial_pose"][:, :, None, :3])
        inputs["voxel_known"] = observed.get("voxel_known", torch.ones_like(observed["voxels"], dtype=torch.bool))
        _validate_shared_initial(inputs, xyz)
    else:
        xyz = carry["_fixed_world_xyz"]
        # Both terms used by rollout_inputs are now predicted; later recorded
        # position is intentionally inaccessible to input-coordinate creation.
        inputs["initial_pose"] = carry["pose"]
        inputs["voxels"] = carry["_fixed_world_voxels"]
        inputs["voxel_known"] = carry["_fixed_world_known"]
        inputs["voxel_relative_xyz"] = xyz - carry["pose"][:, :, None, :3]
        if "_fixed_rgb" in carry:
            inputs["previous_rgb"] = carry["_fixed_rgb"]
        else:
            inputs.pop("previous_rgb", None)
        inputs = rollout_inputs(inputs, carry)
    inputs["_fixed_world_xyz"] = xyz
    result = dict(targets)
    address = targets["address"]
    edit = (address >= 0) & (address < 13**3)
    original_world = (observed["voxel_relative_xyz"]
                      + observed["initial_pose"][:, :, None, :3])
    # [B,T,A] -> gather each actor's original indexed voxel in world space.
    selected = original_world.gather(2, address.transpose(1, 2).clamp(0, 13**3-1)[..., None].expand(-1, -1, -1, 3))
    remapped, inside = _local_index(selected, xyz[:, :, :1])
    known = inputs["voxel_known"].flatten(2).gather(2, remapped.clamp(0, 13**3-1))
    representable = (inside & known.bool()).transpose(1, 2)
    remapped = remapped.transpose(1, 2)
    original_valid = targets["address_valid"].bool()
    result["address"] = torch.where(edit, torch.where(representable, remapped, -1), address)
    result["address_valid"] = original_valid & (~edit | representable)
    result["_original_edit_valid"] = original_valid & edit
    result["_fixed_edit_covered"] = original_valid & edit & representable
    result["_fixed_edit_uncovered"] = original_valid & edit & ~representable
    return inputs, result


def to_device(batch, device):
    return {group: {key: value.to(device) for key, value in batch[group].items()}
            for group in ("inputs", "targets")}
