"""Production M2 Local World Writer.

The model has one shared six-layer joint-action trunk.  Every layer applies
bidirectional temporal attention, resident attention, cross-attention to the
13^3 local world/RGB memory, and an FFN.  Typed sparse writes and resident
camera/state updates are decoded from the same hidden representation.

The legacy M2 implementations remain in :mod:`miworld.models.transition` solely
for loading old checkpoints.  New training and closed-loop runs use this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from miworld.kinematics import (KinematicsConfig, direction_from_angles,
                               propose_trajectory, wrap_angle)
from miworld.transition_state import held_item_trajectory


DEFAULT_TIMESTEPS = 8
VOXEL_SIDE = 13
VOXELS = VOXEL_SIDE ** 3
ACTION_DIM = 23


@dataclass(frozen=True)
class LocalWorldWriterArgs:
    num_block_classes: int
    num_items: int
    air_class: int
    width: int = 256
    depth: int = 6
    heads: int = 8
    timesteps: int = DEFAULT_TIMESTEPS
    event_queries: int = 8
    event_decoder_layers: int = 2
    event_context_frames: int = 0
    use_rgb: bool = True
    memory_gate_init: float = 0.45
    geometric_camera: bool = False
    camera_yaw_offset: float = 0.0
    camera_pitch_sign: float = 1.0
    camera_angle_yaw_gain: float | None = None
    camera_angle_pitch_gain: float | None = None
    recorded_camera_timing: bool = False
    camera_angle_residual_limit: float = math.radians(2.)
    camera_position_residual_limit: float = 0.15
    camera_eye_height_residual_limit: float = 0.15
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)

    def __post_init__(self):
        if self.num_block_classes < 2:
            raise ValueError("Local World Writer needs at least two block classes")
        if self.num_items < 1:
            raise ValueError("Local World Writer needs at least one item class")
        if not 0 <= self.air_class < self.num_block_classes:
            raise ValueError("air_class is outside the block vocabulary")
        if self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if not 1 <= self.event_queries <= self.timesteps:
            raise ValueError("event_queries must be in [1, timesteps]")
        if self.event_decoder_layers < 1:
            raise ValueError("event_decoder_layers must be positive")
        if self.event_context_frames < 0:
            raise ValueError("event_context_frames must be nonnegative")
        if not 0 < self.memory_gate_init < 1:
            raise ValueError("memory_gate_init must be in (0, 1)")
        if (self.camera_angle_residual_limit < 0
                or self.camera_position_residual_limit < 0
                or self.camera_eye_height_residual_limit < 0):
            raise ValueError("camera residual limits must be nonnegative")


def recorded_body_angle_trajectory(inputs):
    """Dataset body angles for observations t+1..t+T.

    Recorded action a_t changes the body angle at t+2, so the first update in
    a block uses the final already-executed context action. Yaw deliberately
    stays on its incoming numerical branch; pitch clamps after every step.
    """
    pose = inputs["initial_pose"]
    context = inputs["event_context_actions"]
    actions = inputs["actions"]
    if context.shape[1] < 1:
        raise ValueError("recorded camera timing needs a preceding action")
    mouse = torch.cat((context[:, -1:, :, 21:23],
                       actions[:, :-1, :, 21:23]), dim=1)
    gain = pose.new_tensor((math.radians(128.), math.radians(72.)))
    delta = mouse * gain
    yaw = pose[:, None, :, 3] + delta[..., 0].cumsum(1)
    pitch = pose[..., 4]
    pitch_steps = []
    for step in range(delta.shape[1]):
        pitch = (pitch + delta[:, step, :, 1]).clamp(-math.pi / 2, math.pi / 2)
        pitch_steps.append(pitch)
    return torch.stack((yaw, torch.stack(pitch_steps, dim=1)), dim=-1)


class JointActionBlock(nn.Module):
    """One temporal -> resident -> world-memory -> FFN block."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.resident_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.memory_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, tokens, memory, memory_padding, active, memory_scale):
        batch, steps, residents, width = tokens.shape

        temporal = self.norms[0](tokens).transpose(1, 2).reshape(
            batch * residents, steps, width
        )
        temporal = self.temporal_attention(
            temporal, temporal, temporal, need_weights=False
        )[0]
        tokens = tokens + temporal.reshape(
            batch, residents, steps, width
        ).transpose(1, 2)

        resident = self.norms[1](tokens).reshape(batch * steps, residents, width)
        resident_padding = ~active[:, None].expand(batch, steps, residents).reshape(
            batch * steps, residents
        )
        resident = self.resident_attention(
            resident,
            resident,
            resident,
            key_padding_mask=resident_padding,
            need_weights=False,
        )[0]
        tokens = tokens + resident.reshape(batch, steps, residents, width)

        query = self.norms[2](tokens).transpose(1, 2).reshape(
            batch * residents, steps, width
        )
        update = self.memory_attention(
            query,
            memory,
            memory,
            key_padding_mask=memory_padding,
            need_weights=False,
        )[0]
        tokens = tokens + memory_scale * update.reshape(
            batch, residents, steps, width
        ).transpose(1, 2)
        tokens = tokens + self.ffn(self.norms[3](tokens))
        return tokens * active[:, None, :, None]


class LocalWorldMemory(nn.Module):
    """Encode 13^3 voxels, visible residents, and the previous RGB frame."""

    def __init__(self, cfg: LocalWorldWriterArgs):
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        self.block_embedding = nn.Embedding(cfg.num_block_classes + 1, width)
        self.voxel_cnn = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv3d(width, width, 1),
        )
        self.coordinate = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.visual = (
            nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(32, width, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            if cfg.use_rgb
            else None
        )
        self.target_type = nn.Embedding(3, width)
        self.null_token = nn.Parameter(torch.randn(width) * 0.02)

    def forward(self, inputs, state_tokens, active):
        batch, residents = active.shape
        width = self.cfg.width
        known = inputs["voxel_known"].bool()
        ids = torch.where(
            known,
            inputs["voxels"].long(),
            self.cfg.num_block_classes,
        )
        voxels = self.block_embedding(ids).reshape(
            batch * residents, VOXEL_SIDE, VOXEL_SIDE, VOXEL_SIDE, width
        ).permute(0, 4, 1, 2, 3)
        voxels = (voxels + self.voxel_cnn(voxels)).flatten(2).transpose(1, 2)
        voxels = voxels.reshape(batch, residents, VOXELS, width)
        voxels = voxels + self.coordinate(inputs["voxel_relative_xyz"] / 6.0)

        pose = inputs["initial_pose"]
        relative = (pose[:, None, :, :3] - pose[:, :, None, :3]) / 6.0
        resident_memory = (
            state_tokens[:, None].expand(-1, residents, -1, -1)
            + self.coordinate(relative)
        )

        if self.cfg.use_rgb:
            if "previous_rgb" not in inputs:
                raise ValueError("Local World Writer with use_rgb=True needs previous_rgb")
            rgb = inputs["previous_rgb"]
            pixels = self.visual(
                rgb.reshape(batch * residents, 3, *rgb.shape[-2:]).float()
            )
            pixels = pixels.flatten(2).transpose(1, 2).reshape(
                batch, residents, 16, width
            )
        else:
            pixels = voxels.new_zeros(batch, residents, 0, width)

        memory = torch.cat((voxels, resident_memory, pixels), dim=2).reshape(
            batch * residents, VOXELS + residents + pixels.shape[2], width
        )
        padding = torch.cat(
            (
                torch.zeros(
                    batch, residents, VOXELS, device=active.device, dtype=torch.bool
                ),
                ~active[:, None].expand(batch, residents, residents),
                torch.zeros(
                    batch,
                    residents,
                    pixels.shape[2],
                    device=active.device,
                    dtype=torch.bool,
                ),
            ),
            dim=2,
        ).reshape(batch * residents, -1)

        candidates = torch.cat(
            (
                voxels + self.target_type.weight[0],
                resident_memory + self.target_type.weight[1],
                (self.null_token + self.target_type.weight[2]).expand(
                    batch, residents, 1, width
                ),
            ),
            dim=2,
        )
        candidate_valid = torch.cat(
            (
                known.flatten(2),
                active[:, None].expand(batch, residents, residents),
                torch.ones(
                    batch, residents, 1, dtype=torch.bool, device=active.device
                ),
            ),
            dim=2,
        )
        diagonal = torch.eye(residents, dtype=torch.bool, device=active.device)[None]
        candidate_valid[:, :, VOXELS : VOXELS + residents] &= ~diagonal
        return memory, padding, candidates, candidate_valid


class EventDecoder(nn.Module):
    """Decode one ordered typed sparse-event sequence per resident."""

    def __init__(self, cfg: LocalWorldWriterArgs):
        super().__init__()
        width = cfg.width
        layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=cfg.heads,
            dim_feedforward=4 * width,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, cfg.event_decoder_layers, norm=nn.LayerNorm(width)
        )
        self.queries = nn.Embedding(cfg.event_queries, width)
        self.count_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, cfg.event_queries + 1),
        )
        self.time_head = nn.Linear(width, cfg.timesteps)
        self.operation_head = nn.Linear(width, 3)
        self.address_query = nn.Linear(width, width)
        self.address_key = nn.Linear(width, width)
        self.payload = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.block_head = nn.Linear(width, cfg.num_block_classes)
        self.damage_head = nn.Linear(width, 1)
        self.width = width
        self.event_queries = cfg.event_queries

    def forward(self, hidden, candidates, candidate_valid, context=None, context_valid=None):
        batch, steps, residents, width = hidden.shape
        memory = hidden.transpose(1, 2).reshape(batch * residents, steps, width)
        memory_valid = torch.ones(
            batch * residents, steps, dtype=torch.bool, device=hidden.device
        )
        if context is not None:
            if context_valid is None:
                raise ValueError("event context needs a validity mask")
            memory = torch.cat((context, memory), dim=1)
            memory_valid = torch.cat((context_valid, memory_valid), dim=1)
        queries = self.queries.weight[None].expand(batch * residents, -1, -1)
        events = self.decoder(
            queries, memory, memory_key_padding_mask=~memory_valid
        ).reshape(
            batch, residents, self.event_queries, width
        )
        summary = (memory * memory_valid[..., None]).sum(dim=1)
        summary = summary / memory_valid.sum(dim=1, keepdim=True).clamp_min(1)
        summary = summary.reshape(batch, residents, width)
        count = self.count_head(torch.cat((summary, events.mean(dim=2)), dim=-1))
        addresses = torch.einsum(
            "baqd,band->baqn",
            self.address_query(events),
            self.address_key(candidates),
        ) / width**0.5
        addresses = addresses.masked_fill(~candidate_valid[:, :, None], -torch.inf)
        return {
            "event_hidden": events,
            "event_count_logits": count,
            "event_time_logits": self.time_head(events),
            "event_operation_logits": self.operation_head(events),
            "event_address_logits": addresses,
        }

    def payloads(self, events, candidates, addresses):
        batch, residents, queries, width = events.shape
        selected = candidates[:, :, None].expand(-1, -1, queries, -1, -1).gather(
            3, addresses[..., None, None].expand(batch, residents, queries, 1, width)
        ).squeeze(3)
        hidden = self.payload(torch.cat((events, selected), dim=-1))
        block = self.block_head(hidden)
        damage = -F.softplus(self.damage_head(hidden).squeeze(-1))
        return block, damage


class CameraDecoder(nn.Module):
    """Decode resident trajectory, camera pose, and held item from shared tokens."""

    def __init__(self, cfg: LocalWorldWriterArgs):
        super().__init__()
        self.cfg = cfg
        self.pose_head = nn.Linear(cfg.width, 5)
        self.camera_head = nn.Linear(cfg.width, 6)
        self.held_head = nn.Linear(cfg.width, cfg.num_items)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.camera_head.weight)
        nn.init.zeros_(self.camera_head.bias)

    def forward(self, hidden, inputs, proposal):
        pose = inputs["initial_pose"]
        residual = self.pose_head(hidden)
        if self.cfg.recorded_camera_timing:
            base_angles = proposal[..., 3:5]
            angle_residual = self.cfg.camera_angle_residual_limit * residual[..., 3:5].tanh()
            angles = torch.stack((
                base_angles[..., 0] + angle_residual[..., 0],
                (base_angles[..., 1] + angle_residual[..., 1]).clamp(-math.pi / 2, math.pi / 2),
            ), dim=-1)
        elif self.cfg.camera_angle_yaw_gain is None:
            angles = wrap_angle(proposal[..., 3:] + residual[..., 3:])
        else:
            gains = pose.new_tensor((self.cfg.camera_angle_yaw_gain,
                                     self.cfg.camera_angle_pitch_gain))
            mouse = inputs["actions"][..., 21:23]
            angles = wrap_angle(pose[:, None, :, 3:5]
                                + (mouse * gains).cumsum(1)
                                + residual[..., 3:5])
        predicted = torch.cat(
            (
                pose[:, None, :, :3] + proposal[..., :3] + residual[..., :3],
                angles,
            ),
            dim=-1,
        )
        velocity = torch.cat(
            (
                predicted[:, :1, :, :3] - pose[:, None, :, :3],
                predicted[:, 1:, :, :3] - predicted[:, :-1, :, :3],
            ),
            dim=1,
        )
        camera = self.camera_head(hidden)
        if self.cfg.recorded_camera_timing:
            # Observed camera at t+1 belongs to the body pose at t.
            prior_angles = torch.cat((pose[:, None, :, 3:5],
                                      predicted[:, :-1, :, 3:5]), dim=1)
            direction = 0.1 * direction_from_angles(prior_angles)
            prior_position = torch.cat((pose[:, None, :, :3],
                                        predicted[:, :-1, :, :3]), dim=1)
            eye = pose.new_tensor((0., 0., 1.6))
            eye_residual = (self.cfg.camera_eye_height_residual_limit
                            * camera[..., 2:3].tanh())
            camera_relative = prior_position + eye - predicted[..., :3]
            camera_relative = camera_relative + torch.cat(
                (torch.zeros_like(eye_residual), torch.zeros_like(eye_residual),
                 eye_residual), dim=-1)
        elif self.cfg.geometric_camera:
            base_direction = direction_from_angles(
                predicted[..., 3:5], yaw_offset=self.cfg.camera_yaw_offset,
                pitch_sign=self.cfg.camera_pitch_sign)
            direction = F.normalize(base_direction + camera[..., 3:], dim=-1)
            camera_relative = inputs["camera_relative"][:, None] + camera[..., :3]
        else:
            base_direction = inputs["camera_direction"][:, None]
            direction = F.normalize(base_direction + camera[..., 3:], dim=-1)
            camera_relative = inputs["camera_relative"][:, None] + camera[..., :3]
        return {
            "pose": predicted,
            "velocity": velocity,
            "proposal": proposal,
            "pose_residual": residual,
            "camera_relative": camera_relative,
            "camera_direction": direction,
            "camera_eye_height_residual": (eye_residual if self.cfg.recorded_camera_timing
                                             else camera[..., 2:3]),
            "held_logits": self.held_head(hidden),
            "player_hidden": hidden,
        }


class LocalWorldWriter(nn.Module):
    """Unified production M2 network shown in the Local World Writer diagram."""

    unified_interactions = True

    def __init__(self, cfg: LocalWorldWriterArgs):
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        self.item_embedding = nn.Embedding(cfg.num_items, 32)
        self.type_embedding = nn.Embedding(4, 16)
        self.state_encoder = nn.Sequential(
            nn.Linear(1 + 4 + 32 + 16 + 3 + 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_encoder = nn.Linear(ACTION_DIM, width)
        self.proposal_encoder = nn.Linear(7, width)
        self.held_encoder = nn.Linear(32, width)
        self.velocity_encoder = nn.Linear(3, width)
        self.time_embedding = nn.Parameter(torch.randn(cfg.timesteps, width) * 0.02)
        if cfg.event_context_frames:
            self.event_context_time_embedding = nn.Parameter(
                torch.randn(cfg.event_context_frames, width) * 0.02
            )
            self.event_context_norm = nn.LayerNorm(width)
        self.memory_encoder = LocalWorldMemory(cfg)
        self.blocks = nn.ModuleList(
            JointActionBlock(width, cfg.heads) for _ in range(cfg.depth)
        )
        gate_logit = torch.tensor(cfg.memory_gate_init).logit().item()
        self.memory_gate_logits = nn.Parameter(torch.full((cfg.depth,), gate_logit))
        self.event_decoder = EventDecoder(cfg)
        self.camera_decoder = CameraDecoder(cfg)

    def _state_tokens(self, inputs, active):
        pose = inputs["initial_pose"]
        state = torch.cat(
            (
                inputs["initial_hp"][..., None] / 20.0,
                pose[..., 3:].sin(),
                pose[..., 3:].cos(),
                self.item_embedding(inputs["held_item"]),
                self.type_embedding(inputs["resident_type"]),
                inputs["camera_relative"],
                inputs["camera_direction"],
            ),
            dim=-1,
        )
        velocity = inputs.get("initial_velocity", torch.zeros_like(pose[..., :3]))
        velocity = velocity / self.cfg.kinematics.distance_per_step
        return (
            self.state_encoder(state) + self.velocity_encoder(velocity)
        ) * active[..., None]

    def _action_tokens(self, inputs, proposal):
        features = torch.cat(
            (
                proposal[..., :3] / 6.0,
                proposal[..., 3:].sin(),
                proposal[..., 3:].cos(),
            ),
            dim=-1,
        )
        held, held_valid = held_item_trajectory(inputs)
        held = torch.where(held_valid, held, inputs["held_item"][:, None])
        return (
            self.action_encoder(inputs["actions"])
            + self.proposal_encoder(features)
            + self.held_encoder(self.item_embedding(held))
            + self.time_embedding[None, :, None]
        )

    def _event_context(self, inputs, active):
        if not self.cfg.event_context_frames:
            return None, None
        frames = self.cfg.event_context_frames
        pose = inputs["event_context_pose"]
        if pose.shape[1] != frames:
            raise ValueError("event context length does not match model configuration")
        held = inputs["event_context_held_item"].long()
        state = torch.cat(
            (
                inputs["event_context_hp"][..., None] / 20.0,
                pose[..., 3:].sin(),
                pose[..., 3:].cos(),
                self.item_embedding(held),
                self.type_embedding(inputs["resident_type"])[:, None].expand(
                    -1, frames, -1, -1
                ),
                inputs["event_context_camera_relative"],
                inputs["event_context_camera_direction"],
            ),
            dim=-1,
        )
        valid = inputs["event_context_valid"].bool() & active[:, None]
        context = (
            self.action_encoder(inputs["event_context_actions"])
            + self.state_encoder(state)
            + self.held_encoder(self.item_embedding(held))
            + self.event_context_time_embedding[None, :, None]
        )
        context = self.event_context_norm(context) * valid[..., None]
        batch, _, residents, width = context.shape
        return (
            context.transpose(1, 2).reshape(batch * residents, frames, width),
            valid.transpose(1, 2).reshape(batch * residents, frames),
        )

    def forward(self, inputs):
        active = inputs["active"].bool()
        if not active.any(dim=1).all():
            raise ValueError("each sample needs at least one real resident")
        if inputs["actions"].shape[1] != self.cfg.timesteps:
            raise ValueError(
                "Local World Writer requires exactly "
                f"{self.cfg.timesteps} action steps"
            )

        state_tokens = self._state_tokens(inputs, active)
        if self.cfg.recorded_camera_timing:
            body_angles = recorded_body_angle_trajectory(inputs)
            movement_angles = torch.cat(
                (inputs["initial_pose"][:, None, :, 3:5], body_angles[:, :-1]), dim=1)
            proposal = propose_trajectory(
                inputs["initial_pose"], inputs["actions"], self.cfg.kinematics,
                absolute_angles=body_angles, movement_angles=movement_angles)
        else:
            proposal = propose_trajectory(
                inputs["initial_pose"], inputs["actions"], self.cfg.kinematics)
        action_tokens = self._action_tokens(inputs, proposal)
        hidden = (action_tokens + state_tokens[:, None]) * active[:, None, :, None]
        memory, padding, candidates, candidate_valid = self.memory_encoder(
            inputs, state_tokens, active
        )
        gates = self.memory_gate_logits.sigmoid()
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden, memory, padding, active, memory_scale=gates[index]
            )

        output = self.camera_decoder(hidden, inputs, proposal)
        context, context_valid = self._event_context(inputs, active)
        output.update(self.event_decoder(
            hidden, candidates, candidate_valid, context, context_valid
        ))
        output.update(
            hidden=hidden,
            candidates=candidates,
            active=active,
            memory_gates=gates,
            action_tokens=action_tokens,
            state_tokens=state_tokens,
        )
        return output

    def forward_player(self, inputs):
        """The resident path is the exact same shared Local World Writer path."""
        return self.forward(inputs)

    def event_payloads(self, output, addresses):
        return self.event_decoder.payloads(
            output["event_hidden"], output["candidates"], addresses
        )

    def decode_interactions(self, output):
        from miworld.models.m2_decode import decode_unified_interactions

        return decode_unified_interactions(self, output)

    @property
    def pose_head(self):
        return self.camera_decoder.pose_head

    @property
    def camera_head(self):
        return self.camera_decoder.camera_head

    @property
    def held_head(self):
        return self.camera_decoder.held_head

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


def build_m2_from_config(config):
    """Construct current or legacy M2 from a saved checkpoint config."""
    model_config = dict(config["model"])
    if config.get("schema") == "m2-local-world-writer-v2-engine-prior-xyz":
        from .local_world_writer_engine_prior import LocalWorldWriterEnginePriorXYZ
        from .local_world_writer_v2 import LocalWorldWriterV2Args
        return LocalWorldWriterEnginePriorXYZ(LocalWorldWriterV2Args(**model_config))
    if config.get("schema") == "m2-local-world-writer-v2-engine-prior":
        from .local_world_writer_engine_prior import LocalWorldWriterEnginePrior
        from .local_world_writer_v2 import LocalWorldWriterV2Args
        return LocalWorldWriterEnginePrior(LocalWorldWriterV2Args(**model_config))
    if config.get("schema") == "m2-local-world-writer-v3":
        from .local_world_writer_v3 import LocalWorldWriterV3, LocalWorldWriterV3Args
        return LocalWorldWriterV3(LocalWorldWriterV3Args(**model_config))
    if config.get("schema") == "m2-local-world-writer-v2":
        from .local_world_writer_v2 import LocalWorldWriterV2, LocalWorldWriterV2Args
        # Preserve historical checkpoint semantics.  Original v2 checkpoints
        # predicted full XYZ displacement; the intermediate formula experiment
        # added distance_per_step before position_prior became explicit.
        if "position_prior" not in model_config:
            model_config["position_prior"] = (
                "action" if "distance_per_step" in model_config else "direct"
            )
        return LocalWorldWriterV2(LocalWorldWriterV2Args(**model_config))
    if isinstance(model_config.get("kinematics"), dict):
        model_config["kinematics"] = KinematicsConfig(**model_config["kinematics"])
    if config.get("schema") == "m2-local-world-writer-v1":
        return LocalWorldWriter(LocalWorldWriterArgs(**model_config))

    from miworld.models.transition import TransitionArgs, TransitionNetwork

    return TransitionNetwork(TransitionArgs(**model_config))


def adapt_m2_timestep_state(state_dict, model):
    """Warm-start a different M2 horizon without silently reshaping weights.

    Existing time positions are copied exactly.  Extra time embeddings repeat
    the final learned embedding, while new event-time classifier rows retain
    the destination model's random initialization.  Optimizer state is not
    compatible across this change and must be created afresh.
    """
    target = model.state_dict()
    result = dict(state_dict)
    resized = {}
    for key in (
        "time_embedding",
        "event_decoder.time_head.weight",
        "event_decoder.time_head.bias",
    ):
        if key not in result or key not in target:
            continue
        source = result[key]
        destination = target[key].clone()
        if source.shape == destination.shape:
            continue
        if source.ndim != destination.ndim or source.shape[1:] != destination.shape[1:]:
            raise ValueError(
                f"cannot adapt {key} from {tuple(source.shape)} "
                f"to {tuple(destination.shape)}"
            )
        copied = min(source.shape[0], destination.shape[0])
        destination[:copied].copy_(source[:copied])
        if key == "time_embedding" and destination.shape[0] > source.shape[0]:
            destination[source.shape[0]:].copy_(source[-1:].expand(
                destination.shape[0] - source.shape[0], *source.shape[1:]
            ))
        result[key] = destination
        resized[key] = (tuple(source.shape), tuple(destination.shape))
    return result, resized
