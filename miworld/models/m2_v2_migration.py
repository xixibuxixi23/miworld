"""Audited, semantic warm-start of the three-stream M2 from a v1 checkpoint.

This is initialization, not resumption.  In particular, the old XYZ head
corrected a hand-written movement proposal; its weights cannot initialize a
head that predicts the entire displacement from the current position.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from typing import Mapping

import torch


V1_SCHEMA = "m2-local-world-writer-v1"
V2_SCHEMA = "m2-local-world-writer-v2"


def _normalise_state(state: Mapping[str, torch.Tensor]):
    result, original = {}, {}
    for raw_name, value in state.items():
        name = raw_name
        # DDP and torch.compile sometimes wrap the same canonical model.
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name in result:
            raise ValueError(f"checkpoint key collision after prefix removal: {name}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state {raw_name!r} is not a tensor")
        result[name], original[name] = value, raw_name
    return result, original


def v2_args_from_v1_config(config):
    """Keep compatible model sizes while fixing the agreed 4/8-step contract."""
    from .local_world_writer_v2 import LocalWorldWriterV2Args

    if config.get("schema") != V1_SCHEMA:
        raise ValueError(f"expected {V1_SCHEMA}, got {config.get('schema')!r}")
    source = config["model"]
    required = ("num_block_classes", "num_items", "air_class")
    values = {key: source[key] for key in required}
    for key in ("width", "depth", "heads", "event_queries",
                "event_decoder_layers", "memory_gate_init"):
        if key in source:
            values[key] = source[key]
    # A 4-frame observed prefix and 8-step prediction horizon are explicit
    # target requirements, not inferred from a possibly longer v1 checkpoint.
    values.update(timesteps=8, history_frames=4, event_context_frames=4)
    if values.get("event_queries", 8) > 8:
        raise ValueError("source has more than 8 event queries; choose an 8-step source")
    if "class_to_raw" in config and len(config["class_to_raw"]) != values["num_block_classes"]:
        raise ValueError("source voxel vocabulary length disagrees with its model")
    return LocalWorldWriterV2Args(**values)


def _camera_compatibility(config, cfg):
    source = config["model"]
    reasons = []
    if not source.get("recorded_camera_timing", False):
        reasons.append("source does not use recorded action/body/camera timing")
    rule = config.get("camera_rule") or {}
    # Legacy "lag=1" described a shift within a future-state tensor, not the
    # absolute observation index: action[t] actually affects body[t+2].
    # Never guess this convention from the word "lag" when reusing a head.
    for legacy, explicit in (
        ("action_to_body_lag", "action_to_body_observation_offset"),
        ("body_to_camera_lag", "body_to_camera_observation_offset"),
    ):
        if legacy in rule and explicit not in rule:
            reasons.append(
                f"ambiguous legacy lag metadata camera_rule.{legacy}; "
                f"explicit {explicit} is required for residual-row reuse")
    for key, expected in {
        "action_to_body_observation_offset": 2,
        "body_to_camera_observation_offset": 1,
        "yaw_degrees_per_action": 128.0,
        "pitch_degrees_per_action": 72.0,
    }.items():
        if key in rule and not math.isclose(float(rule[key]), expected, abs_tol=1e-9):
            reasons.append(f"source camera_rule.{key} differs from {expected}")
    angle_reasons = list(reasons)
    source_limit = source.get("camera_angle_residual_limit", math.radians(2.0))
    if not math.isclose(source_limit, cfg.camera_angle_residual_limit, abs_tol=1e-12):
        angle_reasons.append("bounded angle residual scale differs")
    if "angle_residual_limit_degrees" in rule and not math.isclose(
            math.radians(rule["angle_residual_limit_degrees"]),
            cfg.camera_angle_residual_limit, abs_tol=1e-12):
        angle_reasons.append("camera_rule angle residual scale differs")
    if "angle_residual_limit_radians" in rule and not math.isclose(
            rule["angle_residual_limit_radians"],
            cfg.camera_angle_residual_limit, abs_tol=1e-12):
        angle_reasons.append("camera_rule angle residual scale differs")
    eye_reasons = list(reasons)
    source_eye_limit = source.get("camera_eye_height_residual_limit", 0.15)
    if not math.isclose(source_eye_limit, cfg.camera_eye_height_residual_limit, abs_tol=1e-12):
        eye_reasons.append("bounded eye-height residual scale differs")
    for key, expected in {
        "eye_height": cfg.camera_eye_height,
        "eye_height_residual_limit": cfg.camera_eye_height_residual_limit,
        "camera_direction_magnitude": cfg.camera_direction_scale,
    }.items():
        if key in rule and not math.isclose(float(rule[key]), expected, abs_tol=1e-9):
            eye_reasons.append(f"source camera_rule.{key} differs")
    for key in ("direction_residual", "horizontal_camera_residual"):
        if rule.get(key, False):
            eye_reasons.append(f"source camera_rule.{key} is enabled")
    return {
        "angle": {"compatible": not angle_reasons, "reasons": angle_reasons},
        "eye_height": {"compatible": not eye_reasons, "reasons": eye_reasons},
    }


def migrate_v1_to_v2(source_state_dict, source_config, model):
    """Return a complete strict-loadable state dict and per-tensor audit report.

    Only explicitly listed semantic matches are reused.  Shape mismatch in a
    supposedly compatible weight fails instead of silently randomizing it.
    The destination model supplies initialization for genuinely new modules.
    """
    if source_config.get("schema") != V1_SCHEMA:
        raise ValueError(f"source must be {V1_SCHEMA}")
    source, original = _normalise_state(source_state_dict)
    cfg = model.cfg
    old_cfg = source_config["model"]
    for field in ("num_block_classes", "num_items", "air_class", "width", "depth", "heads"):
        if field in old_cfg and old_cfg[field] != getattr(cfg, field):
            raise ValueError(f"incompatible {field}: source={old_cfg[field]}, target={getattr(cfg, field)}")
    target = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    audit, consumed = {}, Counter()

    def record(name, reason="new v2 parameter; keep destination initialization"):
        audit[name] = {
            "target": name, "target_shape": list(target[name].shape),
            "target_numel": target[name].numel(), "copied_numel": 0,
            "kind": "new", "reason": reason, "sources": [],
        }

    for name in target:
        record(name)

    def contribution(dst, src, count, reason, source_slice=None, target_slice=None):
        item = audit[dst]
        item["copied_numel"] += count
        item["sources"].append({
            "source": original[src], "source_shape": list(source[src].shape),
            "source_slice": source_slice, "target_slice": target_slice,
            "copied_numel": count,
        })
        item["kind"] = "reused" if item["copied_numel"] == item["target_numel"] else "partial"
        item["reason"] = reason
        consumed[src] += count

    def exact(dst, src=None, reason="same parameter shape and module semantics"):
        src = src or dst
        if src not in source:
            raise KeyError(f"required compatible source parameter missing: {src}")
        if source[src].shape != target[dst].shape:
            raise ValueError(f"shape mismatch for {src} -> {dst}: {tuple(source[src].shape)} vs {tuple(target[dst].shape)}")
        target[dst].copy_(source[src].to(target[dst]))
        contribution(dst, src, target[dst].numel(), reason)

    compatible_prefixes = (
        "item_embedding.", "type_embedding.", "action_encoder.", "held_encoder.",
        "event_context_norm.", "memory_encoder.coordinate.",
        "memory_encoder.target_type.", "event_decoder.",
    )
    for name in target:
        if name in ("state_encoder.0.weight", "memory_encoder.block_embedding.weight",
                    "time_embedding", "event_context_time_embedding"):
            continue
        if name.startswith("event_decoder.time_head."):
            continue
        trunk = name.startswith("blocks.") and any(
            f".{component}." in name for component in (
                "temporal_attention", "resident_attention", "memory_attention", "norms", "ffn"))
        if (name.startswith(compatible_prefixes) or trunk
                or name in ("memory_gate_logits", "memory_encoder.null_token",
                            "state_encoder.0.bias", "state_encoder.2.weight", "state_encoder.2.bias")):
            # V1 with event_context_frames=0 legitimately has no context norm.
            if name.startswith("event_context_norm.") and not old_cfg.get("event_context_frames", 0):
                audit[name]["reason"] = "source has no event-history normalization"
            else:
                exact(name)

    name = "state_encoder.0.weight"
    if source[name].ndim != 2 or source[name].shape[1] != 59 or target[name].shape != (source[name].shape[0], 62):
        raise ValueError("expected v1 59-column and v2 62-column state encoders")
    target[name].zero_()
    target[name][:, :59].copy_(source[name].to(target[name]))
    contribution(name, name, source[name].numel(),
                 "copy unchanged 59 state columns; initialize appended relative XYZ columns to zero",
                 "[:, :59]", "[:, :59]")

    name = "memory_encoder.block_embedding.weight"
    expected = (cfg.num_block_classes, cfg.width)
    if tuple(target[name].shape) != expected or tuple(source[name].shape) not in (
            expected, (cfg.num_block_classes + 1, cfg.width)):
        raise ValueError("voxel embedding rows/width do not match the unchanged block vocabulary")
    target[name].copy_(source[name][:cfg.num_block_classes].to(target[name]))
    contribution(name, name, target[name].numel(),
                 "reuse all known block classes; discard the old reserved unknown row",
                 f"[:{cfg.num_block_classes}]", ":")

    for name in ("time_embedding", "event_context_time_embedding",
                 "event_decoder.time_head.weight", "event_decoder.time_head.bias"):
        if name not in target:
            continue
        if name not in source:
            if name == "event_context_time_embedding" and not old_cfg.get("event_context_frames", 0):
                audit[name]["reason"] = "source has no history time embedding"
                continue
            raise KeyError(f"required source time parameter missing: {name}")
        if source[name].ndim != target[name].ndim or source[name].shape[1:] != target[name].shape[1:]:
            raise ValueError(f"incompatible time parameter {name}")
        count = min(source[name].shape[0], target[name].shape[0])
        # History positions are relative to the current boundary: align the
        # most recent histories, whereas future indices always start at +1.
        if name == "event_context_time_embedding":
            target[name][-count:].copy_(source[name][-count:].to(target[name]))
            sl = f"[-{count}:]"
        else:
            target[name][:count].copy_(source[name][:count].to(target[name]))
            sl = f"[:{count}]"
        contribution(name, name, target[name][:count].numel(),
                     "reuse overlapping time indices; retain initialization for any extra target indices", sl, sl)

    for suffix in ("weight", "bias"):
        exact(f"held_head.{suffix}", f"camera_decoder.held_head.{suffix}",
              "unchanged hand-held item classification head")

    camera = _camera_compatibility(source_config, cfg)
    for suffix in ("weight", "bias"):
        dst = f"state_residual_head.{suffix}"
        target[dst].zero_()
        audit[dst]["reason"] = "zero-initialize cumulative displacement and incompatible angle/eye residual rows"
        for src, source_rows, target_rows, compatible, label in (
            (f"camera_decoder.pose_head.{suffix}", slice(3, 5), slice(3, 5), camera["angle"]["compatible"], "bounded body-angle residual"),
            (f"camera_decoder.camera_head.{suffix}", slice(2, 3), slice(5, 6), camera["eye_height"]["compatible"], "bounded vertical eye-height residual"),
        ):
            if compatible:
                if src not in source or source[src][source_rows].shape != target[dst][target_rows].shape:
                    raise ValueError(f"compatible {label} source tensor missing or wrong shape: {src}")
                target[dst][target_rows].copy_(source[src][source_rows].to(target[dst]))
                contribution(dst, src, target[dst][target_rows].numel(),
                             "XYZ rows reset; only semantically compatible bounded camera/body residual rows copied",
                             f"[{source_rows.start}:{source_rows.stop}]", f"[{target_rows.start}:{target_rows.stop}]")

    discarded = []
    for name, value in source.items():
        remainder = value.numel() - consumed[name]
        if remainder < 0:
            raise AssertionError(f"source parameter copied more than once unexpectedly: {name}")
        if remainder == 0:
            continue
        reason = "no compatible v2 parameter, or unused source rows"
        for prefix, why in (
            ("memory_encoder.visual.", "RGB branch removed"),
            ("memory_encoder.voxel_cnn.", "voxel CNN removed"),
            ("velocity_encoder.", "explicit input velocity removed"),
            ("proposal_encoder.", "old mixed XYZ/angle action-proposal encoder removed; angles now enter state encoder"),
            ("camera_decoder.pose_head.", "old XYZ corrects hand-written motion; incompatible angle rows are reset"),
            ("camera_decoder.camera_head.", "free camera XYZ/direction removed; only compatible vertical eye row may be reused"),
            ("memory_encoder.block_embedding.", "reserved unknown-voxel row removed"),
        ):
            if name.startswith(prefix):
                reason = why
                break
        discarded.append({"source": original[name], "source_shape": list(value.shape),
                          "source_numel": value.numel(), "copied_numel": consumed[name],
                          "discarded_numel": remainder, "reason": reason})

    target_total = sum(value.numel() for value in target.values())
    copied_total = sum(row["copied_numel"] for row in audit.values())
    report = {
        "schema": "m2-v1-to-v2-initialization-report-v1",
        "source_schema": V1_SCHEMA, "target_schema": V2_SCHEMA,
        "target_model_config": asdict(cfg),
        "camera_row_compatibility": camera,
        "summary": {
            "source_numel": sum(value.numel() for value in source.values()),
            "target_numel": target_total, "copied_numel": copied_total,
            "new_or_reset_numel": target_total - copied_total,
            "discarded_source_numel": sum(row["discarded_numel"] for row in discarded),
            "copied_target_fraction": copied_total / max(target_total, 1),
            "target_tensor_kinds": dict(Counter(row["kind"] for row in audit.values())),
        },
        "target_parameters": list(audit.values()), "discarded_source_parameters": discarded,
        "optimizer_state_reused": False, "scheduler_state_reused": False,
        "training_started": False,
        "limitations": [
            "A semantic warm-start is not function-equivalent to v1: input organization and attention connectivity changed.",
            "Position prediction and new action attention require subsequent training; initialization is not an accuracy result.",
            "Camera row reuse is conditional on source checkpoint timing and residual metadata.",
        ],
    }
    # A complete state dict, rather than strict=False, is the public contract.
    model.load_state_dict(target, strict=True)
    return target, report
