"""Deterministic engine priors for the parameter-compatible M2 v2 trunk.

The legacy class preserves the horizontal-only experiment exactly.  The XYZ
variant adds VoxeLibre's normal jump parabola and flat-ground landing state,
while the original XYZ head predicts only a bounded velocity residual.
"""
import torch

from .local_world_writer_v2 import LocalWorldWriterV2


def target_horizontal_velocity(action, yaw):
    forward, right = action[..., 0] - action[..., 1], action[..., 3] - action[..., 2]
    norm = torch.linalg.vector_norm(torch.stack((forward, right), -1), dim=-1).clamp_min(1.)
    return 4. * torch.stack(((yaw.sin() * forward + yaw.cos() * right) / norm,
                             (yaw.cos() * forward - yaw.sin() * right) / norm), -1)


def engine_horizontal_step(velocity, target):
    displacement = torch.zeros_like(velocity)
    for _ in range(5):
        wanted = target - velocity
        length = torch.linalg.vector_norm(wanted, dim=-1, keepdim=True)
        velocity = velocity + wanted * (.25 / length.clamp_min(.25))
        displacement = displacement + velocity / 120.
    return displacement, velocity


class LocalWorldWriterEnginePrior(LocalWorldWriterV2):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.position_prior != "action" or cfg.velocity_mode != "conditioned_integrated":
            raise ValueError("engine prior requires the original integrated residual architecture")

    def _position_proposal(self, inputs, active, body_angles):
        speed = inputs.get("engine_prior_velocity")
        if speed is None:
            speed = (inputs["initial_pose"][..., :2].float()
                     - inputs["event_context_pose"][:, -1, :, :2].float()) * 24.
        if speed.shape != (*active.shape, 2):
            raise ValueError("engine_prior_velocity must be [B,A,2] in blocks/second")
        speed = speed.float().masked_fill(~active[..., None], 0.)
        applied = torch.cat((inputs["event_context_actions"][:, -1:], inputs["actions"][:, :-1]), 1)
        relative = torch.zeros_like(speed)
        positions, speeds = [], []
        for frame in range(applied.shape[1]):
            target = target_horizontal_velocity(applied[:, frame].float(), body_angles[:, frame, :, 0].float())
            delta, speed = engine_horizontal_step(speed, target)
            speed = speed.masked_fill(~active[..., None], 0.)
            relative = relative + delta.masked_fill(~active[..., None], 0.)
            positions.append(torch.cat((relative, torch.zeros_like(relative[..., :1])), -1))
            speeds.append(speed)
        # `inputs` here is forward's private dictionary, never the caller's.
        inputs["_engine_prior_speeds"] = torch.stack(speeds, 1)
        return torch.cat((torch.stack(positions, 1), body_angles), -1)

    def _decode_position(self, inputs, active, future, residual, angles, proposal):
        output = super()._decode_position(inputs, active, future, residual, angles, proposal)
        output["engine_prior_velocity"] = inputs["_engine_prior_speeds"]
        return output


class LocalWorldWriterEnginePriorXYZ(LocalWorldWriterV2):
    """Horizontal engine prior plus a sampled VoxeLibre jump table.

    This deliberately does not attempt voxel collision detection.  Standard
    flat-ground jumps use a fixed 16-frame table; the bounded residual remains
    responsible for stairs, ledges, ceilings and special surfaces.  No
    trainable parameters are added relative to M2 v2.
    """

    engine_prior_xyz = True

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.position_prior != "action" or cfg.velocity_mode != "conditioned_integrated":
            raise ValueError("XYZ engine prior requires the integrated action-residual architecture")

    def _jump_offsets(self, reference):
        frames = torch.arange(
            1, 17, device=reference.device, dtype=reference.dtype,
        ) / self.cfg.engine_fps
        offsets = (
            self.cfg.engine_jump_speed * frames
            - .5 * self.cfg.engine_gravity * frames.square()
        ).clamp_min(0.)
        # The real collision solver lands during frame 16 on flat ground.
        offsets[-1] = 0.
        return offsets

    def _initial_jump_state(self, inputs, active, offsets):
        """Match recent recorded Z deltas to the deterministic jump phase."""
        history_z = inputs["event_context_pose"][..., 2].float()
        current_z = inputs["initial_pose"][..., 2].float()
        trajectory = torch.cat((history_z, current_z[:, None]), dim=1)
        displacement = torch.diff(trajectory, dim=1)
        valid = torch.cat((inputs["event_context_valid"].bool(), active[:, None]), dim=1)
        displacement_valid = valid[:, 1:] & valid[:, :-1]

        zero = offsets.new_zeros(1)
        phase_offsets = torch.cat((zero, offsets))
        phase_delta = torch.diff(phase_offsets)
        history = displacement.shape[1]
        candidates = []
        for current_step in range(17):
            indices = torch.arange(current_step - history + 1, current_step + 1,
                                   device=offsets.device)
            pattern = torch.where(
                indices > 0,
                phase_delta[(indices - 1).clamp(0, 15)],
                torch.zeros_like(indices, dtype=offsets.dtype),
            )
            candidates.append(pattern)
        candidates = torch.stack(candidates)  # [phase 0..16, history]
        error = (displacement[:, None] - candidates[None, :, :, None]).abs()
        mask = displacement_valid[:, None].expand_as(error)
        error = error.masked_fill(~mask, 0.).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        best_error, best_phase = error.min(dim=1)
        matched = (best_error < .02) & active
        jumping = matched & (best_phase > 0) & (best_phase < 16)
        jump_step = torch.where(jumping, best_phase, torch.zeros_like(best_phase))
        safe_index = (jump_step - 1).clamp(0, 15)
        origin = torch.where(jumping, current_z - offsets[safe_index], current_z)
        return jump_step, origin

    def _position_proposal(self, inputs, active, body_angles):
        carried_speed = inputs.get("engine_prior_velocity")
        if carried_speed is None:
            speed = (
                inputs["initial_pose"][..., :2].float()
                - inputs["event_context_pose"][:, -1, :, :2].float()
            ) * self.cfg.engine_fps
            offsets = self._jump_offsets(speed)
            jump_step, jump_origin = self._initial_jump_state(inputs, active, offsets)
        else:
            if carried_speed.shape != (*active.shape, 2):
                raise ValueError("engine_prior_velocity must be [B,A,2] in blocks/second")
            speed = carried_speed.float()
            offsets = self._jump_offsets(speed)
            if "engine_prior_jump_step" not in inputs or "engine_prior_jump_origin_z" not in inputs:
                raise ValueError("jump-table carry requires jump step and origin Z")
            jump_step = inputs["engine_prior_jump_step"].long()
            jump_origin = inputs["engine_prior_jump_origin_z"].float()
        if speed.shape != (*active.shape, 2):
            raise ValueError("engine_prior_velocity must be [B,A,2] in blocks/second")

        speed = speed.masked_fill(~active[..., None], 0.)
        absolute_z = inputs["initial_pose"][..., 2].float()
        jump_step = jump_step.masked_fill(~active, 0)
        jump_origin = torch.where((jump_step > 0) & active, jump_origin, absolute_z)
        applied = torch.cat((inputs["event_context_actions"][:, -1:], inputs["actions"][:, :-1]), 1)

        relative_xy = torch.zeros_like(speed[..., :2])
        initial_z = absolute_z
        positions, speeds, jump_steps, jump_origins = [], [], [], []
        for frame in range(applied.shape[1]):
            action = applied[:, frame].float()
            target = target_horizontal_velocity(action, body_angles[:, frame, :, 0].float())
            delta_xy, speed = engine_horizontal_step(speed, target)
            relative_xy = relative_xy + delta_xy.masked_fill(~active[..., None], 0.)

            jump = (action[..., 4] > .5) & active
            takeoff = jump & (jump_step == 0)
            jump_origin = torch.where(takeoff, absolute_z, jump_origin)
            next_step = torch.where(
                takeoff, torch.ones_like(jump_step),
                torch.where(jump_step > 0, jump_step + 1, jump_step),
            ).clamp(0, 16)
            safe_index = (next_step - 1).clamp(0, 15)
            table_z = jump_origin + offsets[safe_index]
            absolute_z = torch.where(next_step > 0, table_z, absolute_z)
            # Frame 16 is the hard flat-ground landing endpoint.
            jump_step = torch.where(next_step >= 16, torch.zeros_like(next_step), next_step)
            jump_origin = torch.where(jump_step > 0, jump_origin, absolute_z)

            speed = speed.masked_fill(~active[..., None], 0.)
            relative_z = (absolute_z - initial_z).masked_fill(~active, 0.)
            positions.append(torch.cat((relative_xy, relative_z[..., None]), dim=-1))
            speeds.append(speed)
            jump_steps.append(jump_step)
            jump_origins.append(jump_origin)

        inputs["_engine_prior_speeds"] = torch.stack(speeds, 1)
        inputs["_engine_prior_jump_steps"] = torch.stack(jump_steps, 1)
        inputs["_engine_prior_jump_origins"] = torch.stack(jump_origins, 1)
        return torch.cat((torch.stack(positions, 1), body_angles), dim=-1)

    def _decode_position(self, inputs, active, future, residual, angles, proposal):
        output = super()._decode_position(inputs, active, future, residual, angles, proposal)
        output.update(
            engine_prior_velocity=inputs["_engine_prior_speeds"],
            engine_prior_jump_step=inputs["_engine_prior_jump_steps"],
            engine_prior_jump_origin_z=inputs["_engine_prior_jump_origins"],
        )
        return output
