"""Action-only coarse trajectories; no collision or combat rules."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KinematicsConfig:
    # Pilot calibration knobs, in world units and radians per control step.
    distance_per_step: float = 0.4
    radians_per_mouse_unit: float = 1.0
    yaw_gain: float | None = None
    pitch_gain: float | None = None


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def direction_from_angles(angles, *, yaw_offset=0.0, pitch_sign=1.0):
    """Unit ENU viewing ray from yaw and pitch, with calibrated conventions."""
    yaw = angles[..., 0] + yaw_offset
    pitch = angles[..., 1]
    return torch.stack((yaw.sin() * pitch.cos(),
                        yaw.cos() * pitch.cos(),
                        pitch_sign * pitch.sin()), dim=-1)


def propose_trajectory(initial_pose, actions, cfg=KinematicsConfig(), *,
                       absolute_angles=None, movement_angles=None):
    """[B,A,5], [B,T,A,23] -> [B,T,A,5], relative xyz and absolute angles.

    TextAgent ENU yaw: 0 faces north (+Y); positive yaw rotates toward +X. Data mouse
    signs and gain are explicit configuration, not estimated from future poses.
    Vertical movement is left to the learned residual in this first pilot.
    """
    if actions.ndim != 4 or actions.shape[1] < 1 or actions.shape[-1] != 23:
        raise ValueError("M2 expects a positive-length block of continuous 23-D actions")
    if initial_pose.ndim != 3 or initial_pose.shape[-1] != 5:
        raise ValueError("initial_pose must be [B,A,5]")
    if actions.shape[0] != initial_pose.shape[0] or actions.shape[2] != initial_pose.shape[1]:
        raise ValueError("actions and initial_pose must share batch and resident axes")
    gains = actions.new_tensor((
        cfg.radians_per_mouse_unit if cfg.yaw_gain is None else cfg.yaw_gain,
        cfg.radians_per_mouse_unit if cfg.pitch_gain is None else cfg.pitch_gain,
    ))
    supplied_angles = absolute_angles is not None
    if not supplied_angles:
        angle_delta = actions[..., 21:23] * gains
        angles = initial_pose[:, None, :, 3:5] + angle_delta.cumsum(1)
    else:
        if absolute_angles.shape != (*actions.shape[:3], 2):
            raise ValueError("absolute_angles must be [B,T,A,2]")
        angles = absolute_angles
    if movement_angles is None:
        movement_angles = angles
    elif movement_angles.shape != angles.shape:
        raise ValueError("movement_angles must match absolute_angles")
    yaw = movement_angles[..., 0]
    forward = actions[..., 0] - actions[..., 1]
    right = actions[..., 3] - actions[..., 2]
    norm = (forward.square() + right.square()).sqrt().clamp_min(1.)
    dx = (yaw.sin()*forward + yaw.cos()*right) / norm
    dy = (yaw.cos()*forward - yaw.sin()*right) / norm
    delta = torch.stack((dx,dy,torch.zeros_like(dx)), -1) * cfg.distance_per_step
    output_angles = angles if supplied_angles else wrap_angle(angles)
    return torch.cat((delta.cumsum(1), output_angles), -1)
