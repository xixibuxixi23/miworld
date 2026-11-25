"""M2 v3: engine horizontal dynamics and a recurrent geometry-aware motion decoder.

Units are explicit: engine_velocity is blocks/second at the last substep;
the legacy output named velocity remains displacement in blocks/frame.
Only the first observed boundary estimates speed from pose history. Thereafter
the corrected terminal speed and decoder memory are carried across chunks.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .local_world_writer_v2 import LocalWorldWriterV2, LocalWorldWriterV2Args


@dataclass(frozen=True)
class LocalWorldWriterV3Args(LocalWorldWriterV2Args):
    width: int = 384
    depth: int = 8
    heads: int = 8
    position_prior: str = "action"
    velocity_mode: str = "conditioned_integrated"
    distance_per_step: float = 4. / 24.  # normalization only, not a motion rule
    motion_width: int = 256
    motion_heads: int = 8
    local_voxels: int = 64
    engine_substeps: int = 5
    engine_hz: float = 120.
    engine_walk_speed: float = 4.
    engine_speed_change: float = .25
    horizontal_impulse_limit: float = .48  # blocks/s per observation
    vertical_speed_limit: float = 20.

    def __post_init__(self):
        super().__post_init__()
        if self.position_prior != "action" or self.velocity_mode != "conditioned_integrated":
            raise ValueError("v3 requires engine-integrated motion")
        if self.motion_width < 1 or self.motion_heads < 1 or self.motion_width % self.motion_heads:
            raise ValueError("motion_width must be divisible by motion_heads")
        if not 1 <= self.local_voxels <= 13 ** 3:
            raise ValueError("local_voxels must fit the observed crop")
        if self.engine_substeps != 5 or self.engine_hz != 120.:
            raise ValueError("this data contract uses five 1/120-second substeps")
        if min(self.engine_walk_speed, self.engine_speed_change,
               self.horizontal_impulse_limit, self.vertical_speed_limit) <= 0:
            raise ValueError("motion scales must be positive")


def target_horizontal_velocity(action, yaw, speed=4.):
    forward, right = action[..., 0] - action[..., 1], action[..., 3] - action[..., 2]
    norm = torch.linalg.vector_norm(torch.stack((forward, right), -1), dim=-1).clamp_min(1.)
    return speed * torch.stack(((yaw.sin() * forward + yaw.cos() * right) / norm,
                                (yaw.cos() * forward - yaw.sin() * right) / norm), -1)


def engine_horizontal_step(velocity, target, *, substeps=5, hz=120., speed_change=.25,
                           transmission=None, impulse=None):
    """Integrate one observation. Contact also changes the carried terminal speed.

Transmission has independent world-X/Y components, allowing wall sliding.
The optional small velocity impulse is distributed over the five substeps.
No collision labels, force, mass or acceleration state is required.
"""
    displacement = torch.zeros_like(velocity)
    for _ in range(substeps):
        wanted = target - velocity
        length = torch.linalg.vector_norm(wanted, dim=-1, keepdim=True)
        velocity = velocity + wanted * (speed_change / length.clamp_min(speed_change))
        if impulse is not None:
            velocity = velocity + impulse / substeps
        if transmission is not None:
            velocity = velocity * transmission
        displacement = displacement + velocity / hz
    return displacement, velocity


class EngineMotionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = cfg.motion_width
        self.trunk = nn.Linear(cfg.width, w)
        # relative position, speed, target XY, free displacement XY, applied action
        self.state = nn.Sequential(nn.Linear(33, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.Embedding(cfg.num_block_classes, w)
        self.offset = nn.Sequential(nn.Linear(3, w), nn.SiLU(), nn.Linear(w, w))
        self.resident = nn.Sequential(nn.Linear(6, w), nn.SiLU(), nn.Linear(w, w))
        self.geometry = nn.MultiheadAttention(w, cfg.motion_heads, batch_first=True)
        self.norm = nn.LayerNorm(w)
        self.cell = nn.GRUCell(2 * w, w)
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 5))
        # Exact free horizontal formula and fixed height before any training.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs, active, hidden, angles):
        cfg = self.cfg
        b, t, a, _ = hidden.shape
        origin = inputs["initial_pose"][..., :3].float()
        origin = origin.masked_fill(~active[..., None], 0.)
        position = origin
        # Accumulate locally: subtracting two large world positions can change
        # KNN tie ordering and amplify rounding into different motion decisions.
        relative = torch.zeros_like(origin)
        dt = cfg.engine_substeps / cfg.engine_hz
        speed = inputs.get("engine_velocity")
        if speed is None:
            speed = (origin - inputs["event_context_pose"][:, -1, :, :3].float()) / dt
        elif speed.shape != (b, a, 3):
            raise ValueError("engine_velocity must be [B,A,3] in blocks/second")
        speed = speed.float().masked_fill(~active[..., None], 0.)
        state = inputs.get("motion_hidden", hidden.new_zeros(b, a, cfg.motion_width))
        if state.shape != (b, a, cfg.motion_width):
            raise ValueError("motion_hidden shape mismatch")
        state = state.masked_fill(~active[..., None], 0.).reshape(b * a, -1)
        applied = torch.cat((inputs["event_context_actions"][:, -1:], inputs["actions"][:, :-1]), 1)
        applied = applied.float().masked_fill(~active[:, None, :, None], 0.)
        xyz = inputs["voxel_relative_xyz"].float().masked_fill(~active[..., None, None], 0.)
        ids = inputs["voxels"].flatten(2).long()
        ids = torch.where(active[..., None], ids, cfg.air_class)
        rows = {key: [] for key in ("position", "velocity", "engine_velocity", "motion_hidden",
                                    "action_velocity_prior", "velocity_residual", "motion_transmission")}
        for frame in range(t):
            target = target_horizontal_velocity(applied[:, frame], angles[:, frame, :, 0].float(),
                                                cfg.engine_walk_speed)
            free, _ = engine_horizontal_step(speed[..., :2], target, substeps=cfg.engine_substeps,
                                             hz=cfg.engine_hz, speed_change=cfg.engine_speed_change)
            features = torch.cat((relative / cfg.coordinate_scale, speed / cfg.engine_walk_speed,
                                  target / cfg.engine_walk_speed, free / cfg.distance_per_step,
                                  applied[:, frame]), -1)
            query = self.trunk(hidden[:, frame]) + self.state(features.to(hidden.dtype))
            # Select around the swept body midpoint, then encode offsets from
            # the *predicted* position, not a future recorded crop/position.
            center = relative + torch.cat((free * .5, torch.full_like(free[..., :1], .8)), -1)
            distance = (xyz - center[..., None, :]).square().sum(-1)
            nearest = distance.topk(cfg.local_voxels, largest=False, sorted=False).indices
            offsets = xyz.gather(2, nearest[..., None].expand(-1, -1, -1, 3)) - relative[..., None, :]
            local = self.blocks(ids.gather(2, nearest)) + self.offset(offsets.to(hidden.dtype))
            other_relative = (origin[:, None] - origin[:, :, None]
                              + relative[:, None] - relative[:, :, None])
            other_speed = speed[:, None].expand(-1, a, -1, -1)
            pair_active = active[:, :, None] & active[:, None, :]
            resident_features = torch.cat((other_relative / cfg.coordinate_scale,
                                           other_speed / cfg.engine_walk_speed), -1)
            resident_features = resident_features.masked_fill(~pair_active[..., None], 0.)
            other = self.resident(resident_features.to(hidden.dtype))
            local = torch.cat((local, other), 2)
            local = local.reshape(b * a, cfg.local_voxels + a, cfg.motion_width)
            padding = torch.cat((torch.zeros(b, a, cfg.local_voxels, dtype=torch.bool, device=active.device),
                                 ~pair_active | torch.eye(a, dtype=torch.bool, device=active.device)[None]), -1)
            context = self.geometry(query.reshape(b * a, 1, -1), local, local,
                                    key_padding_mask=padding.reshape(b * a, -1), need_weights=False)[0][:, 0]
            state = self.cell(torch.cat((self.norm(query.reshape(b * a, -1)), context), -1), state)
            state = state.masked_fill(~active.reshape(b * a, 1), 0.)
            raw = self.head(state).reshape(b, a, 5).float()
            # At initialization gate=1 exactly, with a nonzero one-sided gradient.
            transmission = (1. + raw[..., :2]).clamp(0., 1.)
            impulse = cfg.horizontal_impulse_limit * raw[..., 2:4].tanh()
            xy, final_xy_speed = engine_horizontal_step(
                speed[..., :2], target, substeps=cfg.engine_substeps, hz=cfg.engine_hz,
                speed_change=cfg.engine_speed_change, transmission=transmission, impulse=impulse)
            z_speed = cfg.vertical_speed_limit * raw[..., 4:5].tanh()
            delta = torch.cat((xy, z_speed * dt), -1).masked_fill(~active[..., None], 0.)
            speed = torch.cat((final_xy_speed, z_speed), -1).masked_fill(~active[..., None], 0.)
            relative = relative + delta
            position = origin + relative
            prior = torch.cat((free, torch.zeros_like(z_speed)), -1).masked_fill(~active[..., None], 0.)
            values = dict(position=position, velocity=delta, engine_velocity=speed,
                          motion_hidden=state.reshape(b, a, -1), action_velocity_prior=prior,
                          velocity_residual=delta - prior, motion_transmission=transmission)
            for key in rows:
                rows[key].append(values[key])
        out = {key: torch.stack(value, 1) for key, value in rows.items()}
        out["position_residual"] = out["velocity_residual"].cumsum(1)
        # Diagnostic free step proposals are conditional on corrected speed.
        out["proposal"] = torch.cat((out["action_velocity_prior"].cumsum(1), angles), -1)
        return out


class LocalWorldWriterV3(LocalWorldWriterV2):
    def __init__(self, cfg: LocalWorldWriterV3Args):
        super().__init__(cfg)
        # Separate angular/eye head: no obsolete XYZ rows remain in the model.
        self.state_residual_head = nn.Linear(cfg.width, 3)
        nn.init.zeros_(self.state_residual_head.weight)
        nn.init.zeros_(self.state_residual_head.bias)
        self.motion_decoder = EngineMotionDecoder(cfg)

    def _position_proposal(self, inputs, active, body_angles):
        return torch.cat((torch.zeros_like(body_angles[..., :1]).expand(-1, -1, -1, 3), body_angles), -1)

    def _state_residual(self, future):
        angles_eye = self.state_residual_head(future)
        return torch.cat((torch.zeros_like(angles_eye), angles_eye), -1)

    def _decode_position(self, inputs, active, future, residual, angles, proposal):
        return self.motion_decoder(inputs, active, future, angles)
