"""M2 v2: bidirectional state, action and exact-voxel streams.

This is an opt-in architecture.  The v1 implementation and checkpoints remain
loadable.  No RGB or convolutional voxel encoder is used.  The optional
conditioned-integrated mode observes the last past displacement, predicts a
bounded per-step velocity residual around the known-action prior, and obtains
position only by integrating those velocities.  Future states are query slots,
never future observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from miworld.kinematics import KinematicsConfig, direction_from_angles, propose_trajectory
from miworld.transition_state import held_item_trajectory
from miworld.models.local_world_writer import (
    ACTION_DIM, VOXELS, VOXEL_SIDE, EventDecoder, recorded_body_angle_trajectory,
)


@dataclass(frozen=True)
class LocalWorldWriterV2Args:
    num_block_classes: int
    num_items: int
    air_class: int
    width: int = 256
    depth: int = 6
    heads: int = 8
    timesteps: int = 8
    history_frames: int = 4
    event_context_frames: int = 4
    event_queries: int = 8
    event_decoder_layers: int = 2
    memory_gate_init: float = 0.45
    action_gate_init: float = 0.1
    camera_angle_residual_limit: float = math.radians(2.)
    camera_eye_height_residual_limit: float = 0.15
    camera_eye_height: float = 1.6
    camera_direction_scale: float = 0.1
    coordinate_scale: float = 6.
    distance_per_step: float = 0.4
    position_prior: str = "observed_velocity"
    velocity_mode: str = "compatibility"
    velocity_residual_limit: float = 0.2
    velocity_residual_penalty: bool = False
    engine_jump_speed: float = 6.6
    engine_gravity: float = 20.8
    engine_substeps: int = 5
    engine_fps: float = 24.0
    use_rgb: bool = False
    recorded_camera_timing: bool = True

    def __post_init__(self):
        if self.num_block_classes < 2 or self.num_items < 1:
            raise ValueError("invalid block or item vocabulary size")
        if not 0 <= self.air_class < self.num_block_classes:
            raise ValueError("air_class is outside block vocabulary")
        if self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if self.depth < 1 or self.timesteps < 1 or self.history_frames < 1:
            raise ValueError("depth, horizon and history must be positive")
        if self.event_context_frames != self.history_frames:
            raise ValueError("v2 preserves the same history for trunk and event context")
        if not 1 <= self.event_queries <= self.timesteps or self.event_decoder_layers < 1:
            raise ValueError("invalid event decoder size")
        if not 0 < self.memory_gate_init < 1 or not 0 < self.action_gate_init < 1:
            raise ValueError("attention gate initial values must be between zero and one")
        if self.camera_angle_residual_limit < 0 or self.camera_eye_height_residual_limit < 0:
            raise ValueError("residual limits must be nonnegative")
        if self.coordinate_scale <= 0 or self.camera_direction_scale <= 0:
            raise ValueError("coordinate and direction scales must be positive")
        if self.distance_per_step <= 0:
            raise ValueError("distance_per_step must be positive")
        if self.position_prior not in {"direct", "action", "observed_velocity"}:
            raise ValueError("position_prior must be direct, action or observed_velocity")
        if self.velocity_mode not in {"compatibility", "conditioned_integrated"}:
            raise ValueError("velocity_mode must be compatibility or conditioned_integrated")
        if self.velocity_residual_limit <= 0:
            raise ValueError("velocity_residual_limit must be positive")
        if self.engine_jump_speed <= 0 or self.engine_gravity <= 0:
            raise ValueError("engine jump speed and gravity must be positive")
        if self.engine_substeps < 1 or self.engine_fps <= 0:
            raise ValueError("engine substeps and frame rate must be positive")
        if self.velocity_mode == "conditioned_integrated" and self.position_prior != "action":
            raise ValueError("conditioned_integrated velocity requires the action position prior")
        if self.use_rgb or not self.recorded_camera_timing:
            raise ValueError("v2 has no RGB branch and requires recorded camera timing")


def _safe_padding(valid):
    """Give inactive rows one zero key; their query outputs are masked away.

    PyTorch attention can return NaNs for rows whose keys are all masked.  The
    temporary key is only a numerical safeguard, not a real observation.
    """
    padding = ~valid
    empty = ~valid.any(-1)
    padding = padding.clone()
    padding[empty, 0] = False
    return padding


class JointStateActionBlock(nn.Module):
    """Bidir time -> action cross -> resident -> world cross -> FFN."""

    def __init__(self, width, heads):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.action_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.resident_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.memory_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        # Keep v1's semantic norm indices to make migration unambiguous.
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
        self.action_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )

    def forward(self, tokens, actions, memory, memory_padding, state_valid,
                action_valid, memory_scale, action_scale):
        batch, steps, residents, width = tokens.shape

        def mask(value):
            return value.masked_fill(~state_valid[..., None], 0.)

        temporal = self.norms[0](tokens).transpose(1, 2).reshape(
            batch * residents, steps, width)
        temporal_valid = state_valid.transpose(1, 2).reshape(batch * residents, steps)
        temporal = temporal.masked_fill(~temporal_valid[..., None], 0.)
        update = self.temporal_attention(
            temporal, temporal, temporal,
            key_padding_mask=_safe_padding(temporal_valid), need_weights=False,
        )[0]
        tokens = mask(tokens + update.reshape(batch, residents, steps, width).transpose(1, 2))

        query = self.action_norm(tokens).transpose(1, 2).reshape(batch * residents, steps, width)
        action_memory = actions.transpose(1, 2).reshape(batch * residents, -1, width)
        action_valid_flat = action_valid.transpose(1, 2).reshape(batch * residents, -1)
        action_memory = action_memory.masked_fill(~action_valid_flat[..., None], 0.)
        update = self.action_attention(
            query, action_memory, action_memory,
            key_padding_mask=_safe_padding(action_valid_flat), need_weights=False,
        )[0]
        tokens = mask(tokens + action_scale * update.reshape(
            batch, residents, steps, width).transpose(1, 2))

        resident = self.norms[1](tokens).reshape(batch * steps, residents, width)
        resident_valid = state_valid.reshape(batch * steps, residents)
        resident = resident.masked_fill(~resident_valid[..., None], 0.)
        update = self.resident_attention(
            resident, resident, resident,
            key_padding_mask=_safe_padding(resident_valid), need_weights=False,
        )[0]
        tokens = mask(tokens + update.reshape(batch, steps, residents, width))

        query = self.norms[2](tokens).transpose(1, 2).reshape(batch * residents, steps, width)
        update = self.memory_attention(
            query, memory, memory, key_padding_mask=memory_padding, need_weights=False,
        )[0]
        tokens = mask(tokens + memory_scale * update.reshape(
            batch, residents, steps, width).transpose(1, 2))
        return mask(tokens + self.ffn(self.norms[3](tokens)))


class ExactVoxelMemory(nn.Module):
    """One token per voxel, with no pooling, CNN or image features."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_embedding = nn.Embedding(cfg.num_block_classes, cfg.width)
        self.coordinate = nn.Sequential(
            nn.Linear(3, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width),
        )
        self.target_type = nn.Embedding(3, cfg.width)
        self.null_token = nn.Parameter(torch.randn(cfg.width) * 0.02)

    def forward(self, inputs, current_state, active):
        batch, residents = active.shape
        width = self.cfg.width
        ids = inputs["voxels"].long()
        known = inputs.get("voxel_known", torch.ones_like(ids, dtype=torch.bool)).bool()
        active_voxel = active[..., None, None, None].expand_as(ids)
        if (active_voxel & ~known).any():
            raise ValueError("v2 requires a fully known active 13^3 crop; check coverage/vocabulary")
        if (active_voxel & ((ids < 0) | (ids >= self.cfg.num_block_classes))).any():
            raise ValueError("active voxel ID is outside block vocabulary")
        ids = torch.where(active_voxel, ids, self.cfg.air_class)
        relative_xyz = inputs["voxel_relative_xyz"].masked_fill(~active[..., None, None], 0.)
        voxels = self.block_embedding(ids).reshape(batch, residents, VOXELS, width)
        voxels = voxels + self.coordinate(relative_xyz / self.cfg.coordinate_scale)
        pose = inputs["initial_pose"]
        relative = (pose[:, None, :, :3] - pose[:, :, None, :3]) / self.cfg.coordinate_scale
        pair_active = active[:, :, None] & active[:, None, :]
        relative = relative.masked_fill(~pair_active[..., None], 0.)
        resident_memory = current_state[:, None].expand(-1, residents, -1, -1) + self.coordinate(relative)
        resident_memory = resident_memory.masked_fill(~active[:, None, :, None], 0.)
        memory = torch.cat((voxels, resident_memory), dim=2).reshape(
            batch * residents, VOXELS + residents, width)
        # Inactive source rows still have harmless voxel keys and are masked
        # by the trunk; resident keys always respect active-agent membership.
        padding = torch.cat((
            torch.zeros(batch, residents, VOXELS, dtype=torch.bool, device=active.device),
            ~active[:, None].expand(batch, residents, residents),
        ), dim=2).reshape(batch * residents, -1)
        candidates = torch.cat((
            voxels + self.target_type.weight[0],
            resident_memory + self.target_type.weight[1],
            (self.null_token + self.target_type.weight[2]).expand(batch, residents, 1, width),
        ), dim=2)
        candidate_valid = torch.cat((
            active[..., None].expand(batch, residents, VOXELS),
            active[:, None].expand(batch, residents, residents),
            torch.ones(batch, residents, 1, dtype=torch.bool, device=active.device),
        ), dim=2)
        diagonal = torch.eye(residents, dtype=torch.bool, device=active.device)[None]
        candidate_valid[:, :, VOXELS:VOXELS + residents] &= ~diagonal
        return memory, padding, candidates, candidate_valid


class LocalWorldWriterV2(nn.Module):
    """Shared all-player M2 with separate state/action/voxel streams."""

    unified_interactions = True

    def __init__(self, cfg: LocalWorldWriterV2Args):
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        self.item_embedding = nn.Embedding(cfg.num_items, 32)
        self.type_embedding = nn.Embedding(4, 16)
        # The first 59 fields exactly match v1; relative XYZ is appended.
        self.state_encoder = nn.Sequential(nn.Linear(62, width), nn.SiLU(), nn.Linear(width, width))
        self.action_encoder = nn.Linear(ACTION_DIM, width)
        if cfg.velocity_mode == "conditioned_integrated":
            # Separate zero-initialized adapter makes checkpoint migration
            # explicit and initially preserves the inherited trunk behavior.
            self.velocity_encoder = nn.Linear(3, width)
            nn.init.zeros_(self.velocity_encoder.weight)
            nn.init.zeros_(self.velocity_encoder.bias)
        self.time_embedding = nn.Parameter(torch.randn(cfg.timesteps, width) * 0.02)
        self.event_context_time_embedding = nn.Parameter(torch.randn(cfg.history_frames, width) * 0.02)
        self.current_time_embedding = nn.Parameter(torch.randn(1, width) * 0.02)
        self.state_role_embedding = nn.Embedding(2, width)
        nn.init.normal_(self.state_role_embedding.weight, std=0.02)
        self.action_time_embedding = nn.Parameter(torch.randn(cfg.history_frames + cfg.timesteps, width) * 0.02)
        # The hotbar makes an otherwise identical select-slot action identify
        # its actual item.  This is an action-side known control condition.
        self.held_encoder = nn.Linear(32, width)
        self.action_item_valid_embedding = nn.Embedding(2, width)
        nn.init.normal_(self.action_item_valid_embedding.weight, std=0.02)
        self.event_context_norm = nn.LayerNorm(width)
        self.memory_encoder = ExactVoxelMemory(cfg)
        self.blocks = nn.ModuleList(JointStateActionBlock(width, cfg.heads) for _ in range(cfg.depth))
        self.memory_gate_logits = nn.Parameter(torch.full((cfg.depth,), math.log(
            cfg.memory_gate_init / (1 - cfg.memory_gate_init))))
        self.action_gate_logits = nn.Parameter(torch.full((cfg.depth,), math.log(
            cfg.action_gate_init / (1 - cfg.action_gate_init))))
        self.state_residual_head = nn.Linear(width, 6)
        nn.init.zeros_(self.state_residual_head.weight)
        nn.init.zeros_(self.state_residual_head.bias)
        self.held_head = nn.Linear(width, cfg.num_items)
        self.event_decoder = EventDecoder(cfg)

    def _validate(self, inputs):
        active = inputs["active"].bool()
        if active.ndim != 2 or not active.any(dim=1).all():
            raise ValueError("each sample must contain at least one active player")
        batch, residents = active.shape
        history, steps = self.cfg.history_frames, self.cfg.timesteps
        shapes = {
            "initial_pose": (batch, residents, 5), "initial_hp": (batch, residents),
            "held_item": (batch, residents), "resident_type": (batch, residents),
            "hotbar": (batch, residents, 9), "selected_slot": (batch, residents),
            "camera_relative": (batch, residents, 3), "camera_direction": (batch, residents, 3),
            "actions": (batch, steps, residents, ACTION_DIM),
            "voxels": (batch, residents, VOXEL_SIDE, VOXEL_SIDE, VOXEL_SIDE),
            "voxel_relative_xyz": (batch, residents, VOXELS, 3),
            "event_context_pose": (batch, history, residents, 5),
            "event_context_hp": (batch, history, residents),
            "event_context_held_item": (batch, history, residents),
            "event_context_camera_relative": (batch, history, residents, 3),
            "event_context_camera_direction": (batch, history, residents, 3),
            "event_context_actions": (batch, history, residents, ACTION_DIM),
            "event_context_valid": (batch, history, residents),
        }
        for key, expected in shapes.items():
            if key not in inputs or tuple(inputs[key].shape) != expected:
                raise ValueError(f"{key} must have shape {expected}")
        if "voxel_known" in inputs and inputs["voxel_known"].shape != inputs["voxels"].shape:
            raise ValueError("voxel_known must have the same shape as voxels")
        if (active & ~inputs["event_context_valid"][:, -1].bool()).any():
            raise ValueError("recorded angle timing requires the final history action for every active player")
        if (active[..., None] & ((inputs["hotbar"] < 0) | (inputs["hotbar"] >= self.cfg.num_items))).any():
            raise ValueError("active hotbar item is outside item vocabulary")
        if (active & ((inputs["selected_slot"] < 0) | (inputs["selected_slot"] >= 9))).any():
            raise ValueError("active selected_slot must be in [0, 8]")
        return active

    def _state_features(self, pose, hp, held, resident_type, camera_relative,
                        camera_direction, relative_position, valid):
        # Invalid history/padding may contain sentinels.  Mask before embedding
        # and MLP, so neither NaN floats nor invalid integer IDs can leak in.
        pose = pose.masked_fill(~valid[..., None], 0.)
        hp = hp.masked_fill(~valid, 0.)
        held = held.masked_fill(~valid, 0).long()
        resident_type = resident_type.masked_fill(~valid, 0).long()
        if ((held < 0) | (held >= self.cfg.num_items)).any():
            raise ValueError("known held item is outside item vocabulary")
        if ((resident_type < 0) | (resident_type >= 4)).any():
            raise ValueError("known resident type is outside type vocabulary")
        values = torch.cat((
            hp[..., None] / 20., pose[..., 3:5].sin(), pose[..., 3:5].cos(),
            self.item_embedding(held), self.type_embedding(resident_type),
            camera_relative.masked_fill(~valid[..., None], 0.),
            camera_direction.masked_fill(~valid[..., None], 0.),
            relative_position.masked_fill(~valid[..., None], 0.) / self.cfg.coordinate_scale,
        ), dim=-1)
        return self.state_encoder(values).masked_fill(~valid[..., None], 0.)

    def _encode_inputs(self, inputs, active, body_angles):
        batch, residents = active.shape
        history, steps = self.cfg.history_frames, self.cfg.timesteps
        pose = inputs["initial_pose"]
        history_valid = inputs["event_context_valid"].bool() & active[:, None]
        history_pose = inputs["event_context_pose"]
        hist = self._state_features(
            history_pose, inputs["event_context_hp"], inputs["event_context_held_item"],
            inputs["resident_type"][:, None].expand(-1, history, -1),
            inputs["event_context_camera_relative"], inputs["event_context_camera_direction"],
            history_pose[..., :3] - pose[:, None, :, :3], history_valid,
        )
        current = self._state_features(
            pose, inputs["initial_hp"], inputs["held_item"], inputs["resident_type"],
            inputs["camera_relative"], inputs["camera_direction"], torch.zeros_like(pose[..., :3]), active,
        )
        future_valid = active[:, None].expand(batch, steps, residents)
        future_pose = torch.cat((pose[:, None, :, :3].expand(-1, steps, -1, -1), body_angles), dim=-1)
        # This is an anchored camera prior, not a claimed future GT position.
        prior_camera_angles = torch.cat((pose[:, None, :, 3:5], body_angles[:, :-1]), dim=1)
        prior_camera_direction = self.cfg.camera_direction_scale * direction_from_angles(prior_camera_angles)
        future = self._state_features(
            future_pose, inputs["initial_hp"][:, None].expand(-1, steps, -1),
            inputs["held_item"][:, None].expand(-1, steps, -1),
            inputs["resident_type"][:, None].expand(-1, steps, -1),
            inputs["camera_relative"][:, None].expand(-1, steps, -1, -1),
            prior_camera_direction, torch.zeros_like(future_pose[..., :3]), future_valid,
        )
        if self.cfg.velocity_mode == "conditioned_integrated":
            observed_velocity = (pose[..., :3] - inputs["event_context_pose"][:, -1, :, :3])
            observed_velocity = observed_velocity.masked_fill(~active[..., None], 0.)
            normalized_velocity = (observed_velocity / self.cfg.distance_per_step).clamp(-4., 4.)
            velocity_token = self.velocity_encoder(normalized_velocity)
            current = current + velocity_token
            future = future + velocity_token[:, None]
        observed_role, future_role = self.state_role_embedding.weight
        tokens = torch.cat((
            hist + self.event_context_time_embedding[None, :, None] + observed_role,
            current[:, None] + self.current_time_embedding[None, :, None] + observed_role,
            future + self.time_embedding[None, :, None] + future_role,
        ), dim=1)
        state_valid = torch.cat((history_valid, active[:, None], future_valid), dim=1)
        tokens = tokens.masked_fill(~state_valid[..., None], 0.)
        action_valid = torch.cat((history_valid, future_valid), dim=1)
        raw_actions = torch.cat((inputs["event_context_actions"], inputs["actions"]), dim=1)
        raw_actions = raw_actions.masked_fill(~action_valid[..., None], 0.)
        control_inputs = {
            "actions": raw_actions[:, history:],
            "hotbar": inputs["hotbar"].masked_fill(~active[..., None], 0).long(),
            "selected_slot": inputs["selected_slot"].masked_fill(~active, 0).long(),
        }
        future_held, future_item_valid = held_item_trajectory(control_inputs)
        future_item_valid = future_item_valid & active[:, None]
        initial_held = inputs["held_item"].masked_fill(~active, 0).long()
        future_held = torch.where(future_item_valid, future_held, initial_held[:, None])
        held = inputs["event_context_held_item"].masked_fill(~history_valid, 0).long()
        action_held = torch.cat((held, future_held), dim=1)
        action_item_valid = torch.cat((history_valid, future_item_valid), dim=1)
        actions = (
            self.action_encoder(raw_actions) + self.action_time_embedding[None, :, None]
            + self.held_encoder(self.item_embedding(action_held))
            + self.action_item_valid_embedding(action_item_valid.long())
        )
        actions = actions.masked_fill(~action_valid[..., None], 0.)
        # Preserve the v1 event decoder's four-frame context connection.  The
        # trunk state stream remains action-free; fusion here is decoder-only.
        context = self.event_context_norm(
            hist + self.action_encoder(raw_actions[:, :history])
            + self.held_encoder(self.item_embedding(held))
            + self.event_context_time_embedding[None, :, None]
        ).masked_fill(~history_valid[..., None], 0.)
        context = context.transpose(1, 2).reshape(batch * residents, history, self.cfg.width)
        context_valid = history_valid.transpose(1, 2).reshape(batch * residents, history)
        return (tokens, actions, current, state_valid, action_valid, context,
                context_valid, action_held, action_item_valid)

    def _position_proposal(self, inputs, active, body_angles):
        angle_inputs = inputs
        if self.cfg.position_prior == "action":
            # Compatibility path for the calibrated action-prior experiment.
            movement_angles = torch.cat((
                angle_inputs["initial_pose"][:, None, :, 3:5], body_angles[:, :-1],
            ), dim=1)
            proposal = propose_trajectory(
                angle_inputs["initial_pose"], angle_inputs["actions"],
                KinematicsConfig(distance_per_step=self.cfg.distance_per_step),
                absolute_angles=body_angles, movement_angles=movement_angles,
            )
        elif self.cfg.position_prior == "observed_velocity":
            # The final history observation is t-1 and initial_pose is t.  Their
            # difference is an observed one-step displacement, not a supervised
            # velocity head.  Actions and geometry condition the learned residual.
            velocity = (angle_inputs["initial_pose"][..., :3]
                        - inputs["event_context_pose"][:, -1, :, :3])
            velocity = velocity.masked_fill(~active[..., None], 0.)
            steps = torch.arange(
                1, self.cfg.timesteps + 1, device=velocity.device, dtype=velocity.dtype,
            ).view(1, -1, 1, 1)
            proposal = torch.cat((steps * velocity[:, None], body_angles), dim=-1)
        else:
            proposal = torch.cat((
                torch.zeros((*body_angles.shape[:-1], 3),
                            device=body_angles.device, dtype=body_angles.dtype),
                body_angles,
            ), dim=-1)
        return proposal

    def _state_residual(self, future):
        return self.state_residual_head(future)

    def _decode_position(self, inputs, active, future, residual, angles, proposal):
        initial = inputs["initial_pose"]
        if self.cfg.velocity_mode == "conditioned_integrated":
            zero = torch.zeros_like(proposal[:, :1, :, :3])
            action_velocity = torch.diff(torch.cat((zero, proposal[..., :3]), dim=1), dim=1)
            velocity_residual = self.cfg.velocity_residual_limit * residual[..., :3].tanh()
            velocity = action_velocity + velocity_residual
            position = initial[:, None, :, :3] + velocity.cumsum(dim=1)
            position_residual = velocity_residual.cumsum(dim=1)
        else:
            position = initial[:, None, :, :3] + proposal[..., :3] + residual[..., :3]
            prior = torch.cat((initial[:, None, :, :3], position[:, :-1]), dim=1)
            velocity = position - prior
            action_velocity = torch.zeros_like(velocity)
            velocity_residual = velocity
            position_residual = residual[..., :3]
        return dict(position=position, velocity=velocity, action_velocity_prior=action_velocity,
                    velocity_residual=velocity_residual, position_residual=position_residual)

    def forward(self, inputs):
        active = self._validate(inputs)
        # Only preceding mouse action and known future controls, never GT future yaw.
        angle_inputs = dict(inputs)
        angle_inputs["initial_pose"] = inputs["initial_pose"].masked_fill(~active[..., None], 0.)
        angle_inputs["event_context_actions"] = inputs["event_context_actions"].masked_fill(
            ~(inputs["event_context_valid"].bool() & active[:, None])[..., None], 0.)
        angle_inputs["actions"] = inputs["actions"].masked_fill(~active[:, None, :, None], 0.)
        body_angles = recorded_body_angle_trajectory(angle_inputs)
        proposal = self._position_proposal(angle_inputs, active, body_angles)
        (tokens, actions, current, state_valid, action_valid, context, context_valid,
         action_held, action_item_valid) = self._encode_inputs(inputs, active, body_angles)
        memory, padding, candidates, candidate_valid = self.memory_encoder(inputs, current, active)
        hidden = tokens
        memory_gates, action_gates = self.memory_gate_logits.sigmoid(), self.action_gate_logits.sigmoid()
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, actions, memory, padding, state_valid, action_valid,
                           memory_gates[index], action_gates[index])
        future = hidden[:, self.cfg.history_frames + 1:]
        residual = self._state_residual(future)
        initial = angle_inputs["initial_pose"]
        angle_residual = self.cfg.camera_angle_residual_limit * residual[..., 3:5].tanh()
        angles = body_angles + angle_residual
        angles = torch.stack((angles[..., 0], angles[..., 1].clamp(-math.pi / 2, math.pi / 2)), dim=-1)
        motion = self._decode_position(angle_inputs, active, future, residual, angles, proposal)
        position = motion["position"]
        pose = torch.cat((position, angles), dim=-1)
        prior_position = torch.cat((initial[:, None, :, :3], pose[:, :-1, :, :3]), dim=1)
        prior_angles = torch.cat((initial[:, None, :, 3:5], pose[:, :-1, :, 3:5]), dim=1)
        eye_residual = self.cfg.camera_eye_height_residual_limit * residual[..., 5:6].tanh()
        eye = pose.new_tensor((0., 0., self.cfg.camera_eye_height))
        eye_delta = torch.cat((torch.zeros_like(eye_residual), torch.zeros_like(eye_residual), eye_residual), dim=-1)
        camera_position = prior_position + eye + eye_delta
        output = {
            "pose": pose,
            "velocity": motion["velocity"],
            "observed_velocity_input": (
                initial[..., :3] - inputs["event_context_pose"][:, -1, :, :3]
            ).masked_fill(~active[..., None], 0.),
            "action_velocity_prior": motion["action_velocity_prior"],
            "velocity_residual": motion["velocity_residual"],
            "proposal": proposal,
            "body_angle_prior": body_angles,
            "pose_residual": torch.cat((motion["position_residual"], residual[..., 3:5]), dim=-1),
            "state_residual": residual,
            "camera_relative": camera_position - pose[..., :3],
            "camera_position": camera_position,
            "camera_direction": self.cfg.camera_direction_scale * direction_from_angles(prior_angles),
            "camera_eye_height_residual": eye_residual,
            "held_logits": self.held_head(future),
            "hidden": future, "player_hidden": future,
            "sequence_hidden": hidden, "state_tokens": tokens, "action_tokens": actions,
            "current_state_tokens": current, "world_memory": memory,
            "state_valid_mask": state_valid, "action_valid_mask": action_valid,
            "action_held_items": action_held, "action_item_valid": action_item_valid,
            "candidates": candidates, "candidate_valid": candidate_valid,
            "active": active, "memory_gates": memory_gates, "action_gates": action_gates,
        }
        output.update(motion)
        output.update(self.event_decoder(future, candidates, candidate_valid, context, context_valid))
        return output

    def forward_player(self, inputs):
        return self.forward(inputs)

    def event_payloads(self, output, addresses):
        return self.event_decoder.payloads(output["event_hidden"], output["candidates"], addresses)

    def decode_interactions(self, output):
        from miworld.models.m2_decode import decode_unified_interactions
        return decode_unified_interactions(self, output)

    @property
    def event_time_head(self):
        return self.event_decoder.time_head

    @property
    def event_operation_head(self):
        return self.event_decoder.operation_head

    @property
    def event_pointer_query(self):
        return self.event_decoder.address_query

    @property
    def block_head(self):
        return self.event_decoder.block_head

    @property
    def damage_head(self):
        return self.event_decoder.damage_head
