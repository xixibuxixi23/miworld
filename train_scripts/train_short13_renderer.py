"""Train M3 with a frozen Pixel-VAE or Wan2.1 causal video VAE."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import sys
import os
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from miworld.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from miworld.data.appearance_counterfactual_dataset import AppearanceCounterfactualRendererDataset
from miworld.models.renderer import Renderer, RendererArgs
from miworld.models.renderer_codec import RendererCodec, WanRendererCodec
from miworld.models.latent_normalization import resolve_training_normalization
from miworld.models.player_identity import PlayerIdentityEncoder
from miworld.checkpoint_io import record_checkpoint_failure, staged_torch_save
from miworld.training.renderer_monitoring import RendererProbe, render_probe, save_probe_manifest, select_renderer_probes
from miworld.training.renderer_optimizer import build_full_player_optimizer
from miworld.training.renderer_diagnostics import (
    PLAYER_DIAGNOSTIC_KEYS, pack_player_diagnostics, summarize_player_diagnostics,
)
from miworld.training.renderer_trainer import (
    renderer_flow_loss,
    renderer_history_self_condition_loss,
    renderer_training_losses,
)


LOSS_NAMES = (
    "total_loss",
    "flow_loss",
    "unweighted_flow_loss",
    "injection_roi_flow_loss",
    "player_flow_loss",
    "auxiliary_loss",
    "entity_pixel_l1",
    "entity_pixel_edge",
    "player_pixel_l1",
    "player_pixel_edge",
    "player_roi_l1",
    "player_roi_edge",
    "health_pixel_l1",
    "player_identity_loss",
    "player_identity_similarity",
    "player_identity_ranking_accuracy",
    "counterfactual_player_difference_loss",
)


class EvaluationEMA:
    """Detached inference weights; no change to optimized loss or gradients."""
    def __init__(self, model, decay, saved=None):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if saved is not None:
            if set(saved['model']) != set(self.shadow):
                raise ValueError('EMA checkpoint keys differ')
            for k, v in self.shadow.items():
                v.copy_(saved['model'][k])
            self.updates = saved['updates']

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].lerp_(v.detach(), 1 - self.decay)
            else:
                self.shadow[k].copy_(v)
        self.updates += 1

    @contextlib.contextmanager
    def apply(self, model):
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow, strict=True)
            yield
        finally:
            model.load_state_dict(raw, strict=True)
            model.clear_cache()

    def state_dict(self):
        return {'model': self.shadow, 'updates': self.updates, 'decay': self.decay}

AUXILIARY_LOSS_WEIGHT_NAMES = (
    "entity_pixel_l1_weight",
    "entity_pixel_edge_weight",
    "player_pixel_l1_weight",
    "player_pixel_edge_weight",
    "player_roi_l1_weight",
    "player_roi_edge_weight",
    "health_pixel_l1_weight",
    "player_identity_loss_weight",
    "counterfactual_player_difference_weight",
)


def load_weights(path):
    if Path(path).suffix == ".safetensors":
        return load_file(str(path))
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value.get("model", value)


def build_flow_region_weight(
    base_weight,
    player_mask,
    *,
    player_upweight,
    probability,
    randomize,
):
    """Apply player-focused flow supervision on a random subset of samples."""
    if player_upweight <= 0 or probability <= 0:
        return base_weight
    batch, frames = player_mask.shape[:2]
    latent_mask = torch.nn.functional.interpolate(
        player_mask.float().flatten(0, 1),
        size=base_weight.shape[-2:],
        mode="area",
    ).unflatten(0, (batch, frames))
    if randomize:
        gate = (torch.rand(batch, device=base_weight.device) < probability).float()
    else:
        gate = base_weight.new_full((batch,), probability)
    return base_weight + player_upweight * gate[:, None, None, None, None] * latent_mask


def use_counterfactual_step(step: int, probability: float, seed: int) -> bool:
    """Choose mixed S11 steps reproducibly, including after checkpoint resume."""
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    # SplitMix64 gives every absolute training step a stable pseudo-random draw.
    # Depending on the absolute step (rather than iterator RNG state) makes a
    # resumed run use exactly the same data-source schedule.
    mask = (1 << 64) - 1
    value = (int(step) + int(seed) * 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value < int(float(probability) * (1 << 64))


def effective_auxiliary_loss_weights(args):
    """Resolve configured auxiliary weights for the selected training mode."""
    if args.loss_mode == "flow":
        return {name: 0.0 for name in AUXILIARY_LOSS_WEIGHT_NAMES}
    return {name: float(getattr(args, name)) for name in AUXILIARY_LOSS_WEIGHT_NAMES}


def restrict_gradient_to_last_input_channels(parameter, channel_count):
    """Train only the final input-channel slice of a convolution weight.

    M3-Simple appends ROI appearance after every established joint-patch input,
    so this preserves latent, voxel, and coarse-player columns exactly.
    """
    if parameter.ndim < 2 or not 1 <= channel_count <= parameter.shape[1]:
        raise ValueError("invalid trainable input-channel count")
    mask = torch.zeros_like(parameter)
    mask[:, -channel_count:] = 1
    return parameter.register_hook(lambda gradient: gradient * mask)


def migrate_roi_appearance_fusion(
    warm_state,
    current_state,
    *,
    roi_appearance_channels,
    independent_tokens,
):
    """Build an exact-function warm start for the two appearance A/B branches."""
    migrated = []
    patch_key = "core.x_embedder.proj.weight"
    old_patch = warm_state.get(patch_key)
    current_patch = current_state[patch_key]
    if old_patch is None:
        raise RuntimeError("warm start is missing the joint patch embedding")
    if not independent_tokens and old_patch.shape == current_patch.shape:
        # The source checkpoint already uses the requested concatenated ROI
        # width.  No 32->256 migration is needed; preserve it exactly.
        return migrated
    old_appearance_channels = 32
    common_channels = old_patch.shape[1] - old_appearance_channels
    source_token_key = "core.roi_appearance_token_embedder.0.weight"
    if independent_tokens and source_token_key in warm_state:
        if roi_appearance_channels != old_appearance_channels:
            raise RuntimeError(
                "independent ROI token warm start requires 32 appearance channels"
            )
        if old_patch.shape != current_patch.shape:
            raise RuntimeError(
                "independent ROI token warm start changed joint patch shape"
            )
        return migrated

    if roi_appearance_channels != old_appearance_channels:
        expected = common_channels + int(roi_appearance_channels)
        if current_patch.shape[1] != expected:
            raise RuntimeError(
                "ROI appearance expansion has unexpected patch channels: "
                f"old={tuple(old_patch.shape)}, current={tuple(current_patch.shape)}"
            )
        expanded = current_patch.clone()
        expanded[:, :common_channels].copy_(old_patch[:, :common_channels])
        expanded[
            :, common_channels:common_channels + old_appearance_channels
        ].copy_(old_patch[:, common_channels:])
        # Keep the newly initialized patch columns random. The corresponding
        # projector outputs start at zero, so the old function is exact while
        # the new projector rows receive a first-step gradient.
        warm_state[patch_key] = expanded

        for suffix in ("weight", "bias"):
            key = f"roi_appearance_projector.fuse.2.{suffix}"
            old_value = warm_state.get(key)
            current_value = current_state[key]
            if old_value is None or old_value.shape[0] != old_appearance_channels:
                raise RuntimeError(f"cannot expand ROI projector tensor {key}")
            expanded_value = torch.zeros_like(current_value)
            expanded_value[:old_appearance_channels].copy_(old_value)
            warm_state[key] = expanded_value
        migrated.append("roi_appearance:32->256_concat")

    if independent_tokens:
        if roi_appearance_channels != old_appearance_channels:
            raise RuntimeError("independent ROI tokens require 32 appearance channels")
        if current_patch.shape != old_patch.shape:
            raise RuntimeError("independent ROI tokens require unchanged patch shape")
        kernel_h, kernel_w = old_patch.shape[-2:]
        flattened = old_appearance_channels * kernel_h * kernel_w
        first_key = "core.roi_appearance_token_embedder.0.weight"
        second_key = "core.roi_appearance_token_embedder.1.weight"
        gate_key = "core.roi_appearance_token_gate"
        first = torch.zeros_like(current_state[first_key])
        second = torch.zeros_like(current_state[second_key])
        if first.shape[0] < flattened or second.shape[1] < flattened:
            raise RuntimeError("256-token appearance factorization is too narrow")
        output = 0
        for channel in range(old_appearance_channels):
            for row in range(kernel_h):
                for column in range(kernel_w):
                    first[output, channel, row, column] = 1
                    output += 1
        second[:, :flattened, 0, 0].copy_(
            old_patch[:, common_channels:].reshape(old_patch.shape[0], flattened)
        )
        main_patch = old_patch.clone()
        main_patch[:, common_channels:].zero_()
        warm_state[patch_key] = main_patch
        warm_state[first_key] = first
        warm_state[second_key] = second
        warm_state[gate_key] = torch.ones_like(current_state[gate_key])
        migrated.append("roi_appearance:joint32->independent256")

    return migrated


def migrate_player_appearance_id_dimension(warm_state, current_state):
    """Expand the five-class ID embedding and fixed projector exactly.

    Existing appearance coordinates retain their old function.  New embedding
    coordinates start random while their projector columns start at zero, so
    those columns receive gradients without perturbing the parent checkpoint.
    """
    embedding_key = "resident_encoder.appearance_embedder.weight"
    old_embedding = warm_state.get(embedding_key)
    current_embedding = current_state.get(embedding_key)
    if old_embedding is None or current_embedding is None:
        return []
    if old_embedding.shape == current_embedding.shape:
        return []
    if (
        old_embedding.ndim != 2
        or current_embedding.ndim != 2
        or old_embedding.shape[0] != current_embedding.shape[0]
        or old_embedding.shape[1] >= current_embedding.shape[1]
    ):
        raise RuntimeError(
            "unsupported player appearance ID migration: "
            f"old={tuple(old_embedding.shape)} current={tuple(current_embedding.shape)}"
        )
    old_dim = old_embedding.shape[1]
    new_dim = current_embedding.shape[1]
    expanded_embedding = current_embedding.clone()
    expanded_embedding[:, :old_dim].copy_(old_embedding)
    warm_state[embedding_key] = expanded_embedding

    projector_key = "roi_appearance_projector.fuse.0.weight"
    old_projector = warm_state.get(projector_key)
    current_projector = current_state.get(projector_key)
    if old_projector is None or current_projector is None:
        raise RuntimeError("appearance ID expansion requires fixed ROI projector")
    pose_channels = 8
    prefix_channels = old_projector.shape[1] - old_dim - pose_channels
    if (
        prefix_channels <= 0
        or current_projector.shape[1]
        != prefix_channels + new_dim + pose_channels
        or old_projector.shape[0] != current_projector.shape[0]
        or old_projector.shape[2:] != current_projector.shape[2:]
    ):
        raise RuntimeError(
            "unexpected fixed ROI projector shape during ID expansion: "
            f"old={tuple(old_projector.shape)} current={tuple(current_projector.shape)}"
        )
    expanded_projector = torch.zeros_like(current_projector)
    expanded_projector[:, :prefix_channels].copy_(
        old_projector[:, :prefix_channels]
    )
    expanded_projector[
        :, prefix_channels:prefix_channels + old_dim
    ].copy_(old_projector[:, prefix_channels:prefix_channels + old_dim])
    expanded_projector[:, prefix_channels + new_dim:].copy_(
        old_projector[:, prefix_channels + old_dim:]
    )
    warm_state[projector_key] = expanded_projector
    return [f"player_appearance_id:{old_dim}->{new_dim}"]


def migrate_player_fusion_blocks(
    warm_state, current_state, *, player_fusion_depth, residual_gate=0.01,
):
    """Initialize pre-block player fusion from the first trained scene blocks.

    Attention and MLP weights inherit useful scene features. Their AdaLN gate
    rows start at a small nonzero value, so the added blocks perturb the parent
    only slightly while every copied parameter receives gradient immediately.
    The dedicated other-player action embedder starts as an exact copy of the
    established shared action embedder.
    """
    depth = int(player_fusion_depth)
    if depth <= 0:
        return []
    if any(key.startswith("core.player_fusion_blocks.") for key in warm_state):
        return []
    migrated = []
    for index in range(depth):
        source_prefix = f"core.blocks.{index}."
        target_prefix = f"core.player_fusion_blocks.{index}."
        target_keys = [key for key in current_state if key.startswith(target_prefix)]
        if not target_keys:
            raise RuntimeError(f"missing target player fusion block {index}")
        for target_key in target_keys:
            suffix = target_key[len(target_prefix):]
            source_key = source_prefix + suffix
            if source_key not in warm_state:
                raise RuntimeError(f"cannot initialize {target_key} from {source_key}")
            value = warm_state[source_key].clone()
            if suffix in (
                "s_adaLN_modulation.1.weight",
                "t_adaLN_modulation.1.weight",
            ):
                width = value.shape[0] // 6
                value[2 * width:3 * width].zero_()
                value[5 * width:6 * width].zero_()
            elif suffix in (
                "s_adaLN_modulation.1.bias",
                "t_adaLN_modulation.1.bias",
            ):
                width = value.shape[0] // 6
                value[2 * width:3 * width].fill_(residual_gate)
                value[5 * width:6 * width].fill_(residual_gate)
            warm_state[target_key] = value
        migrated.append(f"core.blocks.{index}->core.player_fusion_blocks.{index}")
    for suffix in ("weight", "bias"):
        source_key = f"core.action_embedder.{suffix}"
        target_key = f"core.other_player_action_embedder.{suffix}"
        if target_key in current_state and target_key not in warm_state:
            warm_state[target_key] = warm_state[source_key].clone()
    migrated.append("core.action_embedder->core.other_player_action_embedder")
    return migrated


def load_renderer_resume(
    model,
    optimizer,
    checkpoint,
    item_vocabulary,
    *,
    allow_item_vocabulary_extension=False,
):
    """Strictly resume M3, optionally appending new item embedding rows.

    Item IDs already present in the checkpoint may never be renamed or moved.
    New IDs retain the model's normal random initialization, while the Adam
    moments for those rows start at zero. Every other model tensor and all
    scalar optimizer state remain an exact resume.
    """
    embedding_key = "resident_encoder.item_embedder.weight"
    saved_model = dict(checkpoint["model"])
    saved_embedding = saved_model[embedding_key]
    current_embedding = model.state_dict()[embedding_key]
    checkpoint_vocabulary = checkpoint.get("config", {}).get("item_vocabulary")

    if checkpoint_vocabulary is not None:
        changed = {
            name: (index, item_vocabulary.get(name))
            for name, index in checkpoint_vocabulary.items()
            if item_vocabulary.get(name) != index
        }
        if changed:
            raise RuntimeError(
                "item vocabulary changed existing checkpoint IDs: "
                f"{changed}"
            )

    if saved_embedding.shape == current_embedding.shape:
        model.load_state_dict(saved_model, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        return {"item_vocabulary_extended": False, "new_items": {}}

    compatible_shape = (
        saved_embedding.ndim == current_embedding.ndim
        and saved_embedding.shape[1:] == current_embedding.shape[1:]
        and saved_embedding.shape[0] < current_embedding.shape[0]
    )
    if not allow_item_vocabulary_extension or not compatible_shape:
        raise RuntimeError(
            "resume item embedding shape differs: "
            f"checkpoint={tuple(saved_embedding.shape)} "
            f"current={tuple(current_embedding.shape)}; pass "
            "--allow-item-vocabulary-extension only when new items were appended"
        )
    if checkpoint_vocabulary is None:
        raise RuntimeError(
            "cannot verify an item-vocabulary extension because the checkpoint "
            "does not record config.item_vocabulary"
        )
    new_items = {
        name: index
        for name, index in item_vocabulary.items()
        if name not in checkpoint_vocabulary
    }
    if not new_items or any(
        index < saved_embedding.shape[0] for index in new_items.values()
    ):
        raise RuntimeError(
            "new item vocabulary entries must use IDs appended after all "
            "checkpoint embedding rows"
        )

    expanded_embedding = current_embedding.clone()
    expanded_embedding[: saved_embedding.shape[0]].copy_(saved_embedding)
    saved_model[embedding_key] = expanded_embedding
    model.load_state_dict(saved_model, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])

    embedding_parameter = dict(model.named_parameters())[embedding_key]
    optimizer_state = optimizer.state.get(embedding_parameter, {})
    for name, value in list(optimizer_state.items()):
        if torch.is_tensor(value) and value.shape == saved_embedding.shape:
            expanded = value.new_zeros(current_embedding.shape)
            expanded[: saved_embedding.shape[0]].copy_(value)
            optimizer_state[name] = expanded
    return {
        "item_vocabulary_extended": True,
        "old_embedding_rows": saved_embedding.shape[0],
        "new_embedding_rows": current_embedding.shape[0],
        "new_items": dict(sorted(new_items.items(), key=lambda pair: pair[1])),
    }


def _start_profile_range(enabled, device):
    if not enabled:
        return None
    torch.cuda.synchronize(device)
    return perf_counter()


def _finish_profile_range(timings, name, started, device):
    if started is None:
        return
    torch.cuda.synchronize(device)
    timings[name] = timings.get(name, 0.0) + perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--train-split", choices=("train", "pilot"), default="train",
        help="pilot is intended only for small controlled diagnostics such as S11",
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--codec", choices=("pixel", "wan"), default="pixel")
    parser.add_argument("--pixel-vae")
    parser.add_argument(
        "--wan-vae",
        help="Wan2.1_VAE.pth used by --codec wan",
    )
    parser.add_argument(
        "--latent-normalization", choices=("none", "pixel-vae"),
        help="Fresh M3-Simple defaults to fixed Pixel VAE mean/std; resume/warm-start inherits checkpoint scale",
    )
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument(
        "--warm-start",
        help="Load model weights without optimizer state; intended for compatible architecture changes",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-index")
    parser.add_argument(
        "--chunk-cache-root",
        help="Optional M3 chunk-cache root; requires a portable chunk-cache window index",
    )
    parser.add_argument(
        "--appearance-counterfactual",
        action="store_true",
        help="group complete S11 skin variants by identical trajectory/start/target",
    )
    parser.add_argument("--counterfactual-variants", type=int, default=4)
    parser.add_argument(
        "--counterfactual-dataset-root",
        help=(
            "Optional S11 root mixed with the ordinary --dataset-root while "
            "preserving complete appearance groups"
        ),
    )
    parser.add_argument("--counterfactual-window-index")
    parser.add_argument(
        "--counterfactual-chunk-cache-root",
        help="Optional independent chunk-cache root for the grouped S11 dataset",
    )
    parser.add_argument(
        "--counterfactual-step-probability",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps drawn from grouped S11 data",
    )
    parser.add_argument(
        "--counterfactual-player-mask-probability",
        type=float,
        default=1.0,
        help="Player-region flow-focus probability on grouped S11 steps",
    )
    parser.add_argument(
        "--counterfactual-random-timesteps",
        action="store_true",
        help=(
            "Train grouped S11 variants at shared ordinary random diffusion times "
            "instead of forcing every future block to the pure-noise endpoint"
        ),
    )
    parser.add_argument(
        "--ordinary-pure-noise-probability",
        type=float,
        default=0.0,
        help=(
            "Fraction of ordinary training samples whose complete future is "
            "forced to the inference endpoint t=1"
        ),
    )
    parser.add_argument("--persist-protocol", action="store_true", help="PERSIST all-frame DF training and refreshed noisy-history KV inference")
    parser.add_argument(
        "--flow-time-distribution",
        choices=("uniform", "logit-normal"),
        default="uniform",
        help="Training noise-time distribution; PERSIST uses logit-normal.",
    )
    parser.add_argument(
        "--flow-sigma-min",
        type=float,
        default=0.0,
        help="Minimum noise scale at t=0; PERSIST uses 1e-5.",
    )
    parser.add_argument(
        "--flow-low-noise-mix-probability",
        type=float,
        default=0.0,
        help="Probability of replacing the base flow time with U(0, flow-low-noise-max).",
    )
    parser.add_argument(
        "--flow-low-noise-max",
        type=float,
        default=0.3,
        help="Upper endpoint for the low-noise uniform mixture component.",
    )
    parser.add_argument(
        "--inference-time-shift",
        type=float,
        default=1.0,
        help="Rational Euler time-grid shift; PERSIST pixel inference uses 3.",
    )
    parser.add_argument(
        "--history-self-condition-step-probability",
        type=float,
        default=0.0,
        help=(
            "Fraction of ordinary optimizer steps that train the second output "
            "block after a detached model-generated first block"
        ),
    )
    parser.add_argument(
        "--history-self-condition-denoising-steps",
        type=int,
        default=4,
        help="Euler steps used to generate history on self-conditioning steps",
    )
    parser.add_argument("--val-window-index")
    parser.add_argument("--health-focus-index")
    parser.add_argument(
        "--health-focus-oversample",
        type=int,
        default=1,
        help="Total sampling multiplicity for windows containing a non-full-health target",
    )
    parser.add_argument("--resume")
    parser.add_argument('--fixed-spatial-reference',action='store_true')
    parser.add_argument(
        '--parallel-player-feature', action='store_true',
        help=(
            'Route the 2D player appearance map around the DiT and add it to '
            'the final latent features; no mask is used at inference'
        ),
    )
    parser.add_argument(
        '--parallel-player-feature-block', type=int, default=-1,
        help='Zero-indexed DiT block receiving the player branch; -1 means before the output head',
    )
    parser.add_argument(
        '--parallel-player-feature-blocks', type=int, nargs='*', default=(),
        help=(
            'Zero-indexed DiT blocks receiving equal-strength player branch '
            'reinjections; mutually exclusive with --parallel-player-feature-block'
        ),
    )
    parser.add_argument(
        '--parallel-player-fusion', action='store_true',
        help='Fuse player and environment tokens with a trainable gated FiLM residual',
    )
    parser.add_argument(
        '--parallel-player-skip-gate-init', type=float, default=1.0,
        help=(
            'Initial direct player-residual scale used with --parallel-player-fusion'
        ),
    )
    parser.add_argument(
        '--parallel-player-masked-fusion', action='store_true',
        help=(
            'Restrict the parallel player residual and FiLM correction to the '
            'inference-time routed player ROI at every frame'
        ),
    )
    parser.add_argument(
        '--mask-player-temporal-attention', action='store_true',
        help=(
            'Suppress temporal-attention updates along the routed player ROI '
            'tube while preserving temporal modelling for the rest of the scene'
        ),
    )
    parser.add_argument(
        '--disable-actor-spatial-condition',
        action='store_true',
        help=(
            'Zero the standalone actor spatial map during training and rollout; '
            'actor state remains fused into the ROI appearance projector'
        ),
    )
    parser.add_argument(
        '--discrete-player-appearance',
        action='store_true',
        help=(
            'Embed the fixed Steve/Alex/villager/zombie/skeleton appearance id '
            'in each resident spatial feature'
        ),
    )
    parser.add_argument(
        '--disable-player-reference-condition',
        action='store_true',
        help=(
            'Zero image-reference pixels before encoding while preserving the '
            'checkpoint-compatible spatial projector'
        ),
    )
    parser.add_argument(
        '--split-player-conditioning',
        action='store_true',
        help=(
            'Keep ROI-routed appearance/ID at the image input while routing '
            'other-player action/state through the same learned embedders as '
            'the target player'
        ),
    )
    parser.add_argument(
        '--other-player-condition-blocks',
        type=int,
        nargs='*',
        default=(),
        help='DiT blocks receiving the spatial other-player action/state tokens',
    )
    parser.add_argument(
        '--gated-other-player-conditioning',
        action='store_true',
        help=(
            'Use one learned residual gate per selected other-player block; '
            'intended for sparse ROI-routed action/orientation injection'
        ),
    )
    parser.add_argument(
        '--other-player-action-only',
        action='store_true',
        help=(
            'Route only raw other-player actions through the shared action '
            'embedder; do not add the other-player state embedding'
        ),
    )
    parser.add_argument(
        '--other-player-roi-adaln', action='store_true',
        help='Inject routed resident actions through local spatial/temporal AdaLN',
    )
    parser.add_argument(
        '--other-player-action-orientation-only',
        action='store_true',
        help=(
            'Route other-player action plus sin/cos absolute yaw and pitch; '
            'do not add the remaining other-player state'
        ),
    )
    parser.add_argument(
        '--roi-appearance-channels',
        type=int,
        default=32,
        choices=(32, 256),
        help='High-resolution ROI appearance channels concatenated at the input',
    )
    parser.add_argument(
        '--independent-roi-appearance-tokens',
        action='store_true',
        help=(
            'Keep the ROI appearance map at 32 channels, factor it through '
            'independent 256-wide patch tokens, and add it with a learned gate'
        ),
    )
    parser.add_argument(
        '--player-appearance-id-dim',
        type=int,
        default=0,
        help=(
            'Discrete appearance ID embedding width; 0 preserves actor_channels-2'
        ),
    )
    parser.add_argument(
        '--roi-appearance-condition-blocks',
        type=int,
        nargs='*',
        default=(),
        help=(
            'Inject independent ROI appearance tokens only before these early '
            'DiT blocks instead of adding them directly at the patch input'
        ),
    )
    parser.add_argument(
        '--roi-appearance-reinject-blocks',
        type=int,
        nargs='*',
        default=(),
        help='DiT blocks receiving gated copies of the independent ROI appearance tokens',
    )
    parser.add_argument(
        '--oracle-player-geometry',
        action='store_true',
        help=(
            'Diagnostic only: inject target-frame GT player silhouette plus '
            'camera-distance depth proxy'
        ),
    )
    parser.add_argument(
        '--player-fusion-depth',
        type=int,
        default=0,
        help='Number of trainable player-fusion DiT blocks inserted before scene block 0',
    )
    parser.add_argument(
        "--allow-item-vocabulary-extension",
        action="store_true",
        help=(
            "Allow --resume when the dataset only appends new item IDs; preserves "
            "old embedding rows and optimizer moments and zero-initializes moments "
            "for new rows"
        ),
    )
    parser.add_argument(
        "--deep-condition-reinjection",
        action="store_true",
        help="Reinject aligned raster, actor, action, resident state, and HUD conditions at every DiT block",
    )
    parser.add_argument(
        "--simple-m3",
        action="store_true",
        help=(
            "Use one fused scene encoder, target-state AdaLN, and one compact "
            "player-reference attention"
        ),
    )
    parser.add_argument(
        "--world-only-conditioning",
        action="store_true",
        help=(
            "Keep checkpoint-compatible player modules active in the graph but "
            "zero every player-derived condition; only world voxels, camera, "
            "noise and diffusion time can affect the denoiser"
        ),
    )
    parser.add_argument(
        "--spatial-appearance-only-conditioning",
        action="store_true",
        help=(
            "Keep world/camera plus player ROI, appearance ID and orientation, "
            "while zeroing action, motion and other dynamic resident state"
        ),
    )
    parser.add_argument(
        "--view-aware-appearance",
        action="store_true",
        help="Warp dense four-view resident references into masked per-block appearance adapters",
    )
    parser.add_argument(
        "--detail-preserving-appearance",
        action="store_true",
        help="Preserve 2x2 sub-patch resident appearance and use nonattenuating occupancy gates",
    )
    parser.add_argument(
        "--entity-reference-attention",
        action="store_true",
        help="Cross-attend native-resolution per-resident RGBA reference tokens inside coarse ROIs",
    )
    parser.add_argument(
        "--unified-player-reference",
        action="store_true",
        help=(
            "Use one input-level all-view player reference adapter; replaces pooled, "
            "view-warped and per-block reference appearance paths"
        ),
    )
    parser.add_argument(
        "--unified-reference-reinject-blocks",
        nargs="*",
        type=int,
        default=(),
        metavar="BLOCK",
        help=(
            "Reuse the single unified reference adapter before selected zero-indexed "
            "DiT blocks; additional residual gates start at zero"
        ),
    )
    parser.add_argument(
        "--player-reference-token-grid",
        nargs=2,
        type=int,
        default=(8, 4),
        metavar=("HEIGHT", "WIDTH"),
        help="Spatial token grid retained from each native 256x128 RGBA player view",
    )
    parser.add_argument(
        "--player-reference-position-encoding",
        action="store_true",
        help="Add fixed 2D source coordinates to each per-view appearance token",
    )
    parser.add_argument(
        "--geometry-aware-player-reference",
        action="store_true",
        help=(
            "Bias unified reference attention by camera-facing view and matching "
            "source/target local player coordinates"
        ),
    )
    parser.add_argument(
        "--freeze-base-for-appearance",
        action="store_true",
        help="Stage-one training: update only the new dense appearance modules",
    )
    parser.add_argument(
        "--freeze-base-for-reference",
        action="store_true",
        help="Stage-one training: update only native-resolution reference encoder and adapters",
    )
    parser.add_argument(
        "--freeze-base-for-player-appearance",
        action="store_true",
        help=(
            "M3-Simple appearance stage: freeze the DiT and all scene/state modules; "
            "update only the player reference encoder and ROI appearance projector"
        ),
    )
    parser.add_argument(
        "--freeze-base-for-player-fusion",
        action="store_true",
        help=(
            "Freeze the scene renderer and train only pre-block fusion plus "
            "appearance/action/orientation player injection parameters"
        ),
    )
    parser.add_argument(
        "--freeze-base-for-parallel-player",
        action="store_true",
        help=(
            "Freeze the scene DiT and train only the reference encoder, 2D ROI "
            "projector and final parallel player residual"
        ),
    )
    parser.add_argument(
        "--parallel-player-unfreeze-last-blocks",
        type=int,
        default=0,
        help=(
            "With --freeze-base-for-parallel-player, also train the final N DiT "
            "blocks and latent output layer"
        ),
    )
    parser.add_argument(
        "--appearance-unfreeze-last-spatial-blocks",
        type=int,
        default=0,
        help=(
            "With --freeze-base-for-appearance, also train the spatial attention, spatial "
            "MLP/AdaLN in the last N DiT blocks and the final output layer"
        ),
    )
    parser.add_argument(
        "--reference-unfreeze-last-spatial-blocks",
        type=int,
        default=0,
        help=(
            "With --freeze-base-for-reference, also train the final N spatial DiT "
            "blocks and output layer"
        ),
    )
    parser.add_argument("--context-frames", type=int, default=65)
    parser.add_argument("--bidirectional-attention", action="store_true",
                        help="Full-clip temporal attention with consistent future noise time and joint sampling")
    parser.add_argument("--clip-noise-time-shift", type=float, default=1.0,
                        help="Bidirectional timestamp shift: 1=uniform, 5=Gamma-style rational shift")
    parser.add_argument("--cache-frames", type=int, default=64)
    parser.add_argument("--block-frames", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute transformer activations during backward to reduce memory; "
            "pass --no-gradient-checkpointing to retain them for faster training"
        ),
    )
    parser.add_argument(
        "--qk-rms-norm", action="store_true",
        help="Apply the QK RMSNorm used by the pretrained 2DAction denoiser",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--voxel-channels", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--actor-channels", type=int, default=16)
    parser.add_argument("--target-views-per-window", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers per DDP rank (4 means 32 total workers on 8 GPUs)",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Batches prefetched by each DataLoader worker (used only when --workers > 0)",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--visualize-every", type=int, default=1000)
    parser.add_argument("--visualization-denoising-steps", type=int, default=20)
    parser.add_argument("--visualization-probe-manifest", help="Explicit qualitative probes, including labelled train/held-out short clips")
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--loss-mode",
        choices=("combined", "flow"),
        default="combined",
        help=(
            "combined uses the configured auxiliary losses; flow forces all "
            "decoded-pixel, identity, and counterfactual auxiliary weights to zero"
        ),
    )
    parser.add_argument(
        "--flow-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier on latent flow loss; use 0 for decoded-pixel-only adaptation",
    )
    parser.add_argument(
        "--player-flow-loss-weight", type=float, default=0.0,
        help="Independent mask-normalized player latent flow loss, added to full-frame flow",
    )
    parser.add_argument(
        "--profile-local-step",
        type=int,
        default=0,
        help=(
            "time the Nth optimizer step after resume, print a synchronized stage "
            "breakdown, then exit without writing a checkpoint; 0 disables profiling"
        ),
    )
    parser.add_argument(
        "--latent-entity-region-upweight",
        type=float,
        default=0.0,
        help="Extra latent flow weight inside entity masks; 0 keeps flow loss uniform",
    )
    parser.add_argument(
        "--latent-player-region-upweight",
        type=float,
        default=0.0,
        help="Extra flow weight inside visible human-player masks",
    )
    parser.add_argument(
        "--player-mask-probability",
        type=float,
        default=0.5,
        help="Probability of applying the player-region flow upweight per sample",
    )
    parser.add_argument(
        "--pixel-loss-frames",
        type=int,
        default=2,
        help="Entity-rich and HP-informative future frames decoded per target view",
    )
    parser.add_argument(
        "--pixel-frame-selection",
        choices=("mixed", "player", "player_unique", "player_stratified"),
        default="mixed",
        help="Choose decoded supervision frames from mixed signals or visible-player masks",
    )
    parser.add_argument(
        "--log-player-noise-bins", action="store_true",
        help="Log additive low/mid/high-noise player flow and pixel diagnostics across ranks",
    )
    parser.add_argument("--entity-pixel-l1-weight", type=float, default=0.5)
    parser.add_argument("--entity-pixel-edge-weight", type=float, default=0.2)
    parser.add_argument("--player-pixel-l1-weight", type=float, default=0.0)
    parser.add_argument("--player-pixel-edge-weight", type=float, default=0.0)
    parser.add_argument("--player-roi-l1-weight", type=float, default=0.0)
    parser.add_argument("--player-roi-edge-weight", type=float, default=0.0)
    parser.add_argument("--player-roi-size", type=int, nargs=2, default=(128, 64))
    parser.add_argument("--player-roi-padding", type=float, default=0.25)
    parser.add_argument("--player-ring-radius", type=int, default=0)
    parser.add_argument("--player-ring-weight", type=float, default=0.25)
    parser.add_argument("--health-pixel-l1-weight", type=float, default=1.0)
    parser.add_argument("--damaged-health-upweight", type=float, default=4.0)
    parser.add_argument("--player-identity-checkpoint")
    parser.add_argument("--player-identity-loss-weight", type=float, default=0.0)
    parser.add_argument("--player-identity-margin", type=float, default=0.2)
    parser.add_argument("--player-identity-negative-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-player-difference-weight", type=float, default=0.0)
    parser.add_argument(
        "--mask-prefix-player-probability", type=float, default=0.0,
        help="replace visible other-player pixels in the observed prefix with mid-gray",
    )
    parser.add_argument(
        "--target-state-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "zero the selected target/observer extra-state condition for a "
            "random fraction of training clips while preserving actions, "
            "other-player routing, world geometry and camera"
        ),
    )
    parser.add_argument(
        "--counterfactual-mask-prefix-player-probability",
        type=float,
        default=1.0,
        help="prefix-player masking probability on grouped S11 steps",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--resume-lr-override", action="store_true",
        help="On --resume, preserve optimizer moments but scale checkpoint learning rates to --lr",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-warmup-relative-to-start", action="store_true")
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--preserve-intragroup-motion", action="store_true")
    parser.add_argument("--diffusion-time-scale", type=float, default=1.0)
    parser.add_argument(
        "--appearance-lr", type=float,
        help="Full M3-Simple fine-tuning: appearance encoder, ROI projector and appended input columns use this LR; all other weights use --lr",
    )
    parser.add_argument(
        "--unfrozen-base-lr-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for selectively unfrozen pretrained DiT parameters",
    )
    parser.add_argument("--appearance-uv-view-condition", action="store_true")
    parser.add_argument("--roi-flow-multiplier", type=float, default=1.0,
                        help="Explicit experiment: injection ROI flow weighting, training only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-pixel-frames", type=int, default=1)
    parser.add_argument("--comparison-reference-pixel-frames", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project", default="plot-m3")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--checkpoint-staging-dir",
        help="Local/PFS directory used to serialize before copying to output (also PLOT_CHECKPOINT_STAGING_DIR)",
    )
    parser.add_argument(
        "--checkpoint-errors",
        choices=("warn", "raise"),
        default="warn",
        help="Keep training and record failures, or stop if a checkpoint cannot be copied",
    )
    args = parser.parse_args()
    if args.codec == "pixel" and not args.pixel_vae:
        parser.error("--pixel-vae is required by --codec pixel")
    if args.codec == "wan" and not args.wan_vae:
        parser.error("--wan-vae is required by --codec wan")
    if args.codec == "wan" and args.latent_normalization is not None:
        parser.error("Wan-VAE owns its channel normalization; omit --latent-normalization")
    if args.codec == "wan" and not args.simple_m3:
        parser.error("the initial Wan-VAE integration requires --simple-m3")
    if args.codec == "wan" and args.loss_mode != "flow":
        parser.error("the Wan-VAE experiment is flow-only; use --loss-mode flow")
    if args.codec == "wan" and (
        (args.context_frames - 1) % 4
        or args.cache_frames % 4
        or args.block_frames % 4
    ):
        parser.error(
            "Wan-VAE requires (context-frames-1), cache-frames and "
            "block-frames to be divisible by 4"
        )
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if not 0.0 <= args.flow_low_noise_mix_probability <= 1.0:
        parser.error("--flow-low-noise-mix-probability must be between 0 and 1")
    if not 0.0 < args.flow_low_noise_max <= 1.0:
        parser.error("--flow-low-noise-max must be in (0,1]")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive")
    if sum(bool(path) for path in (args.resume, args.warm_start, args.backbone_checkpoint)) > 1:
        parser.error("--resume, --warm-start, and --backbone-checkpoint are mutually exclusive")
    if args.resume_lr_override and not args.resume:
        parser.error("--resume-lr-override requires --resume")
    if args.resume_lr_override and not 0 < args.lr < float("inf"):
        parser.error("--resume-lr-override requires a finite positive --lr")
    if args.freeze_base_for_appearance and not args.view_aware_appearance:
        parser.error("--freeze-base-for-appearance requires --view-aware-appearance")
    if args.detail_preserving_appearance and not args.view_aware_appearance:
        parser.error("--detail-preserving-appearance requires --view-aware-appearance")
    if args.freeze_base_for_appearance and args.resume:
        parser.error("use --warm-start for staged appearance training")
    if args.freeze_base_for_reference and not (
        args.entity_reference_attention or args.unified_player_reference
    ):
        parser.error(
            "--freeze-base-for-reference requires --entity-reference-attention "
            "or --unified-player-reference"
        )
    if (
        args.freeze_base_for_reference
        and args.resume
        and not args.unified_player_reference
    ):
        parser.error("use --warm-start for staged legacy reference training")
    if sum((
        args.freeze_base_for_reference,
        args.freeze_base_for_appearance,
        args.freeze_base_for_player_appearance,
        args.freeze_base_for_parallel_player,
        args.freeze_base_for_player_fusion,
    )) > 1:
        parser.error("choose only one staged-freezing mode")
    if args.freeze_base_for_player_appearance and not args.simple_m3:
        parser.error("--freeze-base-for-player-appearance requires --simple-m3")
    if args.player_fusion_depth < 0:
        parser.error("--player-fusion-depth must be nonnegative")
    if args.player_fusion_depth and not (
        args.bidirectional_attention
        and args.split_player_conditioning
        and args.independent_roi_appearance_tokens
    ):
        parser.error(
            "--player-fusion-depth requires bidirectional split-player "
            "conditioning with independent ROI appearance tokens"
        )
    if args.freeze_base_for_player_fusion and args.player_fusion_depth < 1:
        parser.error(
            "--freeze-base-for-player-fusion requires --player-fusion-depth"
        )
    if args.parallel_player_feature and not (
        args.simple_m3 and args.fixed_spatial_reference
    ):
        parser.error(
            "--parallel-player-feature requires --simple-m3 and "
            "--fixed-spatial-reference"
        )
    if args.freeze_base_for_parallel_player and not args.parallel_player_feature:
        parser.error(
            "--freeze-base-for-parallel-player requires --parallel-player-feature"
        )
    if args.parallel_player_feature_block != -1 and not args.parallel_player_feature:
        parser.error(
            "--parallel-player-feature-block requires --parallel-player-feature"
        )
    if args.parallel_player_feature_blocks and not args.parallel_player_feature:
        parser.error(
            "--parallel-player-feature-blocks requires --parallel-player-feature"
        )
    if (
        args.parallel_player_feature_blocks
        and args.parallel_player_feature_block != -1
    ):
        parser.error(
            "--parallel-player-feature-block and --parallel-player-feature-blocks "
            "are mutually exclusive"
        )
    if len(set(args.parallel_player_feature_blocks)) != len(
        args.parallel_player_feature_blocks
    ):
        parser.error("--parallel-player-feature-blocks must be unique")
    if any(
        index < 0 or index >= args.depth
        for index in args.parallel_player_feature_blocks
    ):
        parser.error(
            "--parallel-player-feature-blocks must contain valid DiT block indices"
        )
    if args.parallel_player_fusion and not args.parallel_player_feature:
        parser.error("--parallel-player-fusion requires --parallel-player-feature")
    if args.parallel_player_masked_fusion and not args.parallel_player_feature:
        parser.error(
            "--parallel-player-masked-fusion requires --parallel-player-feature"
        )
    if args.mask_player_temporal_attention and not args.split_player_conditioning:
        parser.error(
            "--mask-player-temporal-attention requires --split-player-conditioning"
        )
    if args.parallel_player_skip_gate_init < 0:
        parser.error("--parallel-player-skip-gate-init must be nonnegative")
    if (
        args.parallel_player_unfreeze_last_blocks
        and not args.freeze_base_for_parallel_player
    ):
        parser.error(
            "--parallel-player-unfreeze-last-blocks requires "
            "--freeze-base-for-parallel-player"
        )
    if args.unified_player_reference and (
        args.entity_reference_attention
        or args.view_aware_appearance
        or args.detail_preserving_appearance
    ):
        parser.error(
            "--unified-player-reference replaces all legacy appearance/reference flags"
        )
    if args.simple_m3 and (
        args.deep_condition_reinjection
        or args.view_aware_appearance
        or args.detail_preserving_appearance
        or args.entity_reference_attention
        or args.geometry_aware_player_reference
        or args.unified_reference_reinject_blocks
    ):
        parser.error(
            "--simple-m3 is incompatible with legacy, geometry-aware, or repeated adapters"
        )
    if args.simple_m3 and args.actor_channels < 3:
        parser.error("--simple-m3 requires --actor-channels >= 3")
    if args.discrete_player_appearance and not args.simple_m3:
        parser.error("--discrete-player-appearance requires --simple-m3")
    if args.disable_player_reference_condition and not args.simple_m3:
        parser.error("--disable-player-reference-condition requires --simple-m3")
    if args.split_player_conditioning and not (
        args.simple_m3
        and args.fixed_spatial_reference
        and args.discrete_player_appearance
    ):
        parser.error(
            "--split-player-conditioning requires --simple-m3, "
            "--fixed-spatial-reference, and --discrete-player-appearance"
        )
    if args.other_player_roi_adaln:
        args.other_player_condition_blocks = list(range(args.depth))
    if args.split_player_conditioning != bool(args.other_player_condition_blocks):
        parser.error(
            "--split-player-conditioning and --other-player-condition-blocks "
            "must be enabled together"
        )
    if args.gated_other_player_conditioning and not args.split_player_conditioning:
        parser.error(
            "--gated-other-player-conditioning requires "
            "--split-player-conditioning"
        )
    if args.gated_other_player_conditioning and args.player_fusion_depth:
        parser.error(
            "--gated-other-player-conditioning cannot be combined with "
            "--player-fusion-depth"
        )
    if args.other_player_action_only and not args.split_player_conditioning:
        parser.error(
            "--other-player-action-only requires --split-player-conditioning"
        )
    if (
        args.other_player_action_orientation_only
        and not args.split_player_conditioning
    ):
        parser.error(
            "--other-player-action-orientation-only requires "
            "--split-player-conditioning"
        )
    if (
        args.other_player_action_only
        and args.other_player_action_orientation_only
    ):
        parser.error(
            "--other-player-action-only and "
            "--other-player-action-orientation-only are mutually exclusive"
        )
    if (
        args.independent_roi_appearance_tokens
        and args.roi_appearance_channels != 32
    ):
        parser.error(
            "--independent-roi-appearance-tokens requires "
            "--roi-appearance-channels 32"
        )
    if args.independent_roi_appearance_tokens and not args.simple_m3:
        parser.error("--independent-roi-appearance-tokens requires --simple-m3")
    if args.player_appearance_id_dim < 0:
        parser.error("--player-appearance-id-dim must be nonnegative")
    if args.player_appearance_id_dim and not (
        args.discrete_player_appearance and args.split_player_conditioning
    ):
        parser.error(
            "--player-appearance-id-dim requires split discrete player conditioning"
        )
    if args.roi_appearance_reinject_blocks and not (
        args.independent_roi_appearance_tokens and args.split_player_conditioning
    ):
        parser.error(
            "--roi-appearance-reinject-blocks requires independent split-player tokens"
        )
    if args.roi_appearance_condition_blocks and not (
        args.independent_roi_appearance_tokens and args.split_player_conditioning
    ):
        parser.error(
            "--roi-appearance-condition-blocks requires independent "
            "split-player tokens"
        )
    if (
        args.roi_appearance_condition_blocks
        and args.roi_appearance_reinject_blocks
    ):
        parser.error(
            "--roi-appearance-condition-blocks and "
            "--roi-appearance-reinject-blocks are mutually exclusive"
        )
    if len(set(args.roi_appearance_condition_blocks)) != len(
        args.roi_appearance_condition_blocks
    ) or any(
        index < 0 or index >= args.depth
        for index in args.roi_appearance_condition_blocks
    ):
        parser.error(
            "--roi-appearance-condition-blocks must contain unique valid "
            "block indices"
        )
    if len(set(args.roi_appearance_reinject_blocks)) != len(
        args.roi_appearance_reinject_blocks
    ) or any(
        index < 0 or index >= args.depth
        for index in args.roi_appearance_reinject_blocks
    ):
        parser.error(
            "--roi-appearance-reinject-blocks must contain unique valid block indices"
        )
    if args.oracle_player_geometry and not args.simple_m3:
        parser.error("--oracle-player-geometry requires --simple-m3")
    if len(set(args.other_player_condition_blocks)) != len(
        args.other_player_condition_blocks
    ) or any(
        index < 0 or index >= args.depth
        for index in args.other_player_condition_blocks
    ):
        parser.error(
            "--other-player-condition-blocks must contain unique valid block indices"
        )
    if args.unified_reference_reinject_blocks and not args.unified_player_reference:
        parser.error(
            "--unified-reference-reinject-blocks requires --unified-player-reference"
        )
    if args.geometry_aware_player_reference and not args.unified_player_reference:
        parser.error(
            "--geometry-aware-player-reference requires --unified-player-reference"
        )
    if any(size < 1 for size in args.player_reference_token_grid):
        parser.error("--player-reference-token-grid dimensions must be positive")
    if len(set(args.unified_reference_reinject_blocks)) != len(
        args.unified_reference_reinject_blocks
    ) or any(
        index < 0 or index >= args.depth
        for index in args.unified_reference_reinject_blocks
    ):
        parser.error(
            "--unified-reference-reinject-blocks must contain unique valid block indices"
        )
    if args.appearance_unfreeze_last_spatial_blocks and not args.freeze_base_for_appearance:
        parser.error(
            "--appearance-unfreeze-last-spatial-blocks requires "
            "--freeze-base-for-appearance"
        )
    if args.reference_unfreeze_last_spatial_blocks and not args.freeze_base_for_reference:
        parser.error(
            "--reference-unfreeze-last-spatial-blocks requires --freeze-base-for-reference"
        )
    if not 0 <= args.appearance_unfreeze_last_spatial_blocks <= args.depth:
        parser.error("--appearance-unfreeze-last-spatial-blocks must be between 0 and depth")
    if not 0 <= args.reference_unfreeze_last_spatial_blocks <= args.depth:
        parser.error("--reference-unfreeze-last-spatial-blocks must be between 0 and depth")
    if not 0 <= args.parallel_player_unfreeze_last_blocks <= args.depth:
        parser.error("--parallel-player-unfreeze-last-blocks must be between 0 and depth")
    if not 0 <= args.player_mask_probability <= 1:
        parser.error("--player-mask-probability must be between 0 and 1")
    if not 0 <= args.mask_prefix_player_probability <= 1:
        parser.error("--mask-prefix-player-probability must be between 0 and 1")
    if not 0 <= args.target_state_dropout_probability <= 1:
        parser.error("--target-state-dropout-probability must be between 0 and 1")
    if not 0 <= args.counterfactual_player_mask_probability <= 1:
        parser.error("--counterfactual-player-mask-probability must be between 0 and 1")
    if not 0 <= args.counterfactual_mask_prefix_player_probability <= 1:
        parser.error(
            "--counterfactual-mask-prefix-player-probability must be between 0 and 1"
        )
    if args.appearance_counterfactual and args.target_views_per_window != 1:
        parser.error("--appearance-counterfactual requires --target-views-per-window 1")
    if not 0 <= args.counterfactual_step_probability <= 1:
        parser.error("--counterfactual-step-probability must be between 0 and 1")
    if not 0 <= args.ordinary_pure_noise_probability <= 1:
        parser.error("--ordinary-pure-noise-probability must be between 0 and 1")
    if args.appearance_uv_view_condition and not (args.fixed_spatial_reference and args.split_player_conditioning and args.disable_player_reference_condition):
        parser.error("UV/view experiment requires fixed split ID-only appearance conditioning")
    if not 1 <= args.roi_flow_multiplier <= 10:
        parser.error("ROI flow multiplier must be between 1 and 10")
    if args.roi_flow_multiplier > 1 and (not args.persist_protocol or not args.log_player_noise_bins):
        parser.error("ROI flow experiment requires PERSIST mode and diagnostic logging")
    if args.persist_protocol:
        if args.bidirectional_attention or args.history_self_condition_step_probability:
            parser.error("PERSIST mode requires causal attention and no self-conditioning")
        if args.loss_mode != "flow" or args.flow_loss_weight != 1:
            parser.error("PERSIST mode requires pure flow loss")
        aux = (args.player_flow_loss_weight, args.latent_entity_region_upweight,
               args.latent_player_region_upweight, args.entity_pixel_l1_weight,
               args.entity_pixel_edge_weight, args.player_pixel_l1_weight,
               args.player_pixel_edge_weight, args.player_roi_l1_weight,
               args.player_roi_edge_weight, args.health_pixel_l1_weight,
               args.player_identity_loss_weight, args.counterfactual_player_difference_weight)
        if any(aux) or args.ordinary_pure_noise_probability or args.appearance_counterfactual:
            parser.error("PERSIST mode requires unweighted ordinary flow MSE only")
        args.flow_time_distribution = "logit-normal"
        args.flow_low_noise_mix_probability = 0.0
        args.flow_sigma_min = 1e-5
    if not 0 <= args.flow_sigma_min < 1:
        parser.error("--flow-sigma-min must be in [0,1)")
    if not 0 < args.inference_time_shift < float("inf"):
        parser.error("--inference-time-shift must be finite and positive")
    if not 0 <= args.history_self_condition_step_probability <= 1:
        parser.error(
            "--history-self-condition-step-probability must be between 0 and 1"
        )
    if args.history_self_condition_denoising_steps < 1:
        parser.error("--history-self-condition-denoising-steps must be positive")
    if not 0 < args.clip_noise_time_shift < float("inf"):
        parser.error("--clip-noise-time-shift must be finite and positive")
    if not args.bidirectional_attention and args.clip_noise_time_shift != 1.0:
        parser.error("--clip-noise-time-shift requires --bidirectional-attention")
    if args.bidirectional_attention and args.history_self_condition_step_probability > 0:
        parser.error("bidirectional training cannot use causal block history self-conditioning")
    if bool(args.counterfactual_dataset_root) != bool(args.counterfactual_step_probability):
        parser.error(
            "--counterfactual-dataset-root and a positive "
            "--counterfactual-step-probability must be used together"
        )
    if args.appearance_counterfactual and args.counterfactual_dataset_root:
        parser.error(
            "use either dedicated --appearance-counterfactual training or mixed "
            "--counterfactual-dataset-root training"
        )
    if (
        args.loss_mode == "combined"
        and args.counterfactual_player_difference_weight > 0
        and not (args.appearance_counterfactual or args.counterfactual_dataset_root)
    ):
        parser.error("counterfactual difference loss requires grouped S11 data")
    if args.unfrozen_base_lr_scale <= 0:
        parser.error("--unfrozen-base-lr-scale must be positive")
    if args.lr_warmup_steps < 0:
        parser.error("--lr-warmup-steps must be nonnegative")
    if not 0 < args.adam_beta2 < 1 or not 0 <= args.weight_decay < float("inf"):
        parser.error("invalid AdamW beta2 or weight decay")
    if not 0 < args.grad_clip_norm < float("inf"):
        parser.error("--grad-clip-norm must be finite and positive")
    if args.preserve_intragroup_motion and not (args.codec == "wan" and args.simple_m3):
        parser.error("intragroup motion requires Wan and simple M3")
    if not 0 < args.diffusion_time_scale < float("inf"):
        parser.error("--diffusion-time-scale must be finite and positive")
    if args.appearance_lr is not None:
        if not args.simple_m3 or any((args.freeze_base_for_appearance,
                                     args.freeze_base_for_reference,
                                     args.freeze_base_for_player_appearance)):
            parser.error("--appearance-lr requires --simple-m3 with the entire M3 trainable")
        if not 0 < args.appearance_lr < float("inf") or not 0 < args.lr < float("inf"):
            parser.error("learning rates must be finite and positive")
    if args.visualization_probe_manifest and not args.val_window_index:
        parser.error("--visualization-probe-manifest requires --val-window-index")
    if args.profile_local_step < 0:
        parser.error("--profile-local-step must be nonnegative")
    if args.profile_local_step and args.gradient_accumulation != 1:
        parser.error("--profile-local-step currently requires --gradient-accumulation 1")
    if (
        args.loss_mode == "combined"
        and args.player_identity_loss_weight > 0
        and not args.player_identity_checkpoint
    ):
        parser.error("--player-identity-loss-weight requires --player-identity-checkpoint")
    if (
        min(
            args.steps,
            args.save_every,
            args.validate_every,
            args.target_views_per_window,
            args.batch_size,
            args.gradient_accumulation,
        )
        < 1
    ):
        parser.error("steps, save interval, anchors and batch size must be positive")
    if args.context_frames == 1:
        if args.pixel_loss_frames != 0:
            parser.error("single-frame zero-reference training requires --pixel-loss-frames 0")
    elif not 1 <= args.pixel_loss_frames < args.context_frames:
        parser.error("pixel-loss-frames must be within the future-frame count")
    if (
        args.history_self_condition_step_probability > 0
        and args.context_frames < 1 + 2 * args.block_frames
    ):
        parser.error(
            "history self-conditioning requires context-frames >= 1 + 2 * block-frames"
        )
    if args.health_focus_oversample < 1:
        parser.error("health-focus-oversample must be positive")
    if args.player_ring_radius < 0:
        parser.error("player-ring-radius must be nonnegative")
    if not 0 <= args.player_ring_weight <= 1:
        parser.error("player-ring-weight must be between zero and one")
    if len(args.player_roi_size) != 2 or min(args.player_roi_size) < 1:
        parser.error("player-roi-size must contain two positive integers")
    if args.player_roi_padding < 0:
        parser.error("player-roi-padding must be nonnegative")
    if args.health_focus_oversample > 1 and not args.health_focus_index:
        parser.error("health-focus-oversample above 1 requires --health-focus-index")
    if min(
        args.latent_entity_region_upweight,
        args.latent_player_region_upweight,
        args.entity_pixel_l1_weight,
        args.entity_pixel_edge_weight,
        args.player_pixel_l1_weight,
        args.player_pixel_edge_weight,
        args.player_roi_l1_weight,
        args.player_roi_edge_weight,
        args.health_pixel_l1_weight,
        args.damaged_health_upweight,
        args.player_identity_loss_weight,
        args.player_identity_margin,
        args.player_identity_negative_weight,
        args.counterfactual_player_difference_weight,
        args.flow_loss_weight,
        args.player_flow_loss_weight,
    ) < 0:
        parser.error("region and pixel loss weights must be nonnegative")
    if args.codec == "wan":
        model_context_frames = 1 + (args.context_frames - 1) // 4
        model_cache_frames = args.cache_frames // 4
        model_block_frames = args.block_frames // 4
        model_input_h, model_input_w = 45, 80
        model_patch_size = (3, 2)
        temporal_downsampling = 4
    else:
        model_context_frames = args.context_frames
        model_cache_frames = args.cache_frames
        model_block_frames = args.block_frames
        model_input_h, model_input_w = 36, 64
        model_patch_size = (2, 2)
        temporal_downsampling = 1

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if world > 1 else args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type != "cuda":
        parser.error(
            "raw voxel rasterization needs CUDA; use CPU contract tests for smoke validation"
        )
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    compiled_arch = f"sm_{capability[0]}{capability[1]}"
    if capability >= (10, 0) and compiled_arch not in torch.cuda.get_arch_list():
        parser.error(
            f"PyTorch {torch.__version__} was not compiled for {compiled_arch}; "
            "run scripts/check_m3_environment.py on the B200 node"
        )
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    if (args.reference_pixel_frames < 0 or args.reference_pixel_frames >= args.context_frames
            or (args.reference_pixel_frames > 0
                and (args.reference_pixel_frames - 1) % temporal_downsampling)):
        parser.error("reference prefix must align to VAE temporal stride and leave a future")
    if args.reference_pixel_frames > 1 and (not args.bidirectional_attention or args.loss_mode != "flow"):
        parser.error("multi-frame reference requires bidirectional pure flow")
    if (args.comparison_reference_pixel_frames < 0
            or args.comparison_reference_pixel_frames >= args.context_frames
            or (args.comparison_reference_pixel_frames > 0
                and (args.comparison_reference_pixel_frames - 1) % temporal_downsampling)):
        parser.error("comparison reference must align to VAE stride and leave a future")
    torch.manual_seed(args.seed + rank)
    dataset_class = (
        AppearanceCounterfactualRendererDataset
        if args.appearance_counterfactual
        else TextAgentRendererDataset
    )
    dataset_kwargs = dict(
        split=args.train_split,
        context_frames=args.context_frames,
        window_index=args.window_index,
        targets_per_window=args.target_views_per_window,
        entity_region_upweight=args.latent_entity_region_upweight,
        health_focus_index=args.health_focus_index,
        health_focus_oversample=args.health_focus_oversample,
        chunk_cache_root=args.chunk_cache_root,
    )
    if args.appearance_counterfactual:
        dataset_kwargs["variants_per_group"] = args.counterfactual_variants
    dataset = dataset_class(
        args.dataset_root,
        args.vocabulary,
        **dataset_kwargs,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        if world > 1
        else None
    )
    worker_kwargs = {"num_workers": args.workers}
    if args.workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_renderer,
        **worker_kwargs,
    )
    counterfactual_dataset = None
    counterfactual_sampler = None
    counterfactual_loader = None
    if args.counterfactual_dataset_root:
        counterfactual_dataset = AppearanceCounterfactualRendererDataset(
            args.counterfactual_dataset_root,
            args.vocabulary,
            split="train",
            context_frames=args.context_frames,
            window_index=args.counterfactual_window_index,
            chunk_cache_root=args.counterfactual_chunk_cache_root,
            targets_per_window=1,
            variants_per_group=args.counterfactual_variants,
            entity_region_upweight=args.latent_entity_region_upweight,
        )
        if counterfactual_dataset.item_vocabulary != dataset.item_vocabulary:
            raise ValueError("ordinary and S11 item vocabularies differ")
        counterfactual_sampler = (
            DistributedSampler(
                counterfactual_dataset,
                num_replicas=world,
                rank=rank,
                shuffle=True,
                seed=args.seed + 1,
            )
            if world > 1
            else None
        )
        counterfactual_loader = DataLoader(
            counterfactual_dataset,
            batch_size=args.batch_size,
            shuffle=counterfactual_sampler is None,
            sampler=counterfactual_sampler,
            collate_fn=collate_renderer,
            **worker_kwargs,
        )
    cfg = RendererArgs(
        dataset.vocabulary.size,
        max(dataset.item_vocabulary.values()) + 1,
        persist_protocol=args.persist_protocol,
        input_h=model_input_h,
        input_w=model_input_w,
        patch_size=model_patch_size,
        context_frames=model_context_frames,
        cache_frames=model_cache_frames,
        block_frames=model_block_frames,
        temporal_downsampling=temporal_downsampling,
        preserve_intragroup_motion=args.preserve_intragroup_motion,
        bidirectional_attention=args.bidirectional_attention,
        clip_noise_time_shift=args.clip_noise_time_shift,
        diffusion_time_scale=args.diffusion_time_scale,
        gradient_checkpointing=args.gradient_checkpointing,
        qk_rms_norm=args.qk_rms_norm,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        voxel_channels=args.voxel_channels,
        condition_dim=args.condition_dim,
        actor_channels=args.actor_channels,
        deep_condition_reinjection=args.deep_condition_reinjection,
        view_aware_appearance=args.view_aware_appearance,
        detail_preserving_appearance=args.detail_preserving_appearance,
        entity_reference_attention=args.entity_reference_attention,
        unified_player_reference=args.unified_player_reference or args.simple_m3,
        player_reference_grid_size=tuple(args.player_reference_token_grid),
        player_reference_position_encoding=args.player_reference_position_encoding,
        geometry_aware_player_reference=args.geometry_aware_player_reference,
        unified_reference_reinject_blocks=tuple(
            args.unified_reference_reinject_blocks
        ),
        simple_conditioning=args.simple_m3,
        world_only_conditioning=args.world_only_conditioning,
        spatial_appearance_only_conditioning=(
            args.spatial_appearance_only_conditioning
        ),
        fixed_spatial_reference=args.fixed_spatial_reference,
        disable_actor_spatial_condition=args.disable_actor_spatial_condition,
        parallel_player_feature=args.parallel_player_feature,
        parallel_player_feature_block=args.parallel_player_feature_block,
        parallel_player_feature_blocks=tuple(args.parallel_player_feature_blocks),
        parallel_player_fusion=args.parallel_player_fusion,
        parallel_player_skip_gate_init=args.parallel_player_skip_gate_init,
        parallel_player_masked_fusion=args.parallel_player_masked_fusion,
        mask_player_temporal_attention=args.mask_player_temporal_attention,
        discrete_player_appearance=args.discrete_player_appearance,
        disable_player_reference_condition=args.disable_player_reference_condition,
        appearance_uv_view_condition=args.appearance_uv_view_condition,
        split_player_conditioning=args.split_player_conditioning,
        other_player_condition_blocks=tuple(args.other_player_condition_blocks),
        gated_other_player_conditioning=(
            args.gated_other_player_conditioning
        ),
        other_player_action_only=args.other_player_action_only,
        other_player_roi_adaln=args.other_player_roi_adaln,
        other_player_action_orientation_only=(
            args.other_player_action_orientation_only
        ),
        roi_appearance_channels=args.roi_appearance_channels,
        independent_roi_appearance_tokens=(
            args.independent_roi_appearance_tokens
        ),
        player_appearance_id_dim=args.player_appearance_id_dim,
        roi_appearance_condition_blocks=tuple(
            args.roi_appearance_condition_blocks
        ),
        roi_appearance_reinject_blocks=tuple(
            args.roi_appearance_reinject_blocks
        ),
        oracle_player_geometry=args.oracle_player_geometry,
        player_fusion_depth=args.player_fusion_depth,
    )
    with torch.cuda.device(device):
        raw_model = Renderer(cfg).to(device).train()
    raw_model.roi_flow_multiplier = args.roi_flow_multiplier
    if args.roi_flow_multiplier > 1:
        if raw_model.reference_layout is None:
            raise ValueError("ROI weighting requires appearance reference layout")
        def capture_injection_roi(module, inputs, result):
            # Same soft boxes as injection; overlapping residents never multiply weight.
            raw_model._flow_injection_roi = result[0].detach().float().amax(dim=2, keepdim=True)
        raw_model.reference_layout.register_forward_hook(capture_injection_roi)
    if args.backbone_checkpoint:
        report = raw_model.load_2daction_backbone(load_weights(args.backbone_checkpoint))
        if rank == 0:
            print(json.dumps(report))
    checkpoint_path = args.resume or args.warm_start
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False,
                   mmap=bool(args.fixed_spatial_reference))
        if checkpoint_path else None
    )
    if args.appearance_uv_view_condition and args.resume and not checkpoint['config']['renderer'].get('appearance_uv_view_condition',False):
        raise RuntimeError("UV/view resume needs a checkpoint with zero-initialized UV/view columns")
    if args.codec == "wan":
        codec = WanRendererCodec(args.wan_vae).to(device).eval()
    else:
        latent_normalization = resolve_training_normalization(
            args.latent_normalization, simple_m3=args.simple_m3,
            checkpoint_config=(
                checkpoint.get("config", {}) if checkpoint is not None else None
            ),
        )
        codec = RendererCodec(
            load_weights(args.pixel_vae), latent_normalization=latent_normalization
        ).to(device).eval()
    if rank == 0:
        print(json.dumps({"codec": codec.get_config()}))
    identity_encoder = None
    if args.loss_mode == "combined" and args.player_identity_checkpoint:
        identity_checkpoint = torch.load(
            args.player_identity_checkpoint, map_location="cpu", weights_only=False
        )
        identity_encoder = PlayerIdentityEncoder(
            int(identity_checkpoint.get("embedding_dim", 128))
        ).to(device).eval()
        identity_encoder.crop_size = tuple(identity_checkpoint.get("crop_size", (128, 64)))
        identity_encoder.load_state_dict(identity_checkpoint["model"], strict=True)
        identity_encoder.requires_grad_(False)
    if args.freeze_base_for_player_fusion:
        for name, parameter in raw_model.named_parameters():
            parameter.requires_grad_(name.startswith((
                "core.player_fusion_blocks.",
                "core.other_player_action_embedder.",
                "core.other_player_orientation_embedder.",
                "core.roi_appearance_token_embedder.",
                "core.roi_appearance_token_gate",
                "core.roi_appearance_condition_gates",
                "core.roi_appearance_reinject_gates",
                "core.other_player_condition_gates",
                "core.oracle_player_geometry_embedder.",
                "reference_encoder.",
                "roi_appearance_projector.",
                "resident_encoder.appearance_embedder.",
            )))
    elif args.freeze_base_for_parallel_player:
        first_unfrozen_block = args.depth - args.parallel_player_unfreeze_last_blocks
        for name, parameter in raw_model.named_parameters():
            trainable = name.startswith((
                "reference_encoder.",
                "roi_appearance_projector.",
                "core.parallel_player_adapter.",
                "core.parallel_player_fusion_module.",
                "core.parallel_player_fusion_gate",
                "core.parallel_player_skip_gate",
                "resident_encoder.appearance_embedder.",
            ))
            if args.parallel_player_unfreeze_last_blocks:
                trainable = trainable or name.startswith("core.final_layer.")
                for block_index in range(first_unfrozen_block, args.depth):
                    trainable = trainable or name.startswith(
                        f"core.blocks.{block_index}."
                    )
            parameter.requires_grad_(trainable)
    elif args.freeze_base_for_player_appearance:
        joint_patch_weight_name = "core.x_embedder.proj.weight"
        for name, parameter in raw_model.named_parameters():
            parameter.requires_grad_(
                name.startswith((
                    "reference_encoder.",
                    "roi_appearance_projector.",
                ))
                or name == joint_patch_weight_name
            )
        joint_patch_weight = dict(raw_model.named_parameters())[joint_patch_weight_name]
        restrict_gradient_to_last_input_channels(
            joint_patch_weight, raw_model.core.roi_appearance_dim
        )
    elif args.freeze_base_for_reference:
        first_unfrozen_spatial_block = (
            args.depth - args.reference_unfreeze_last_spatial_blocks
        )
        for name, parameter in raw_model.named_parameters():
            trainable = name.startswith((
                "reference_encoder.",
                "core.entity_reference_adapters.",
                "core.unified_reference_adapter.",
            ))
            if args.reference_unfreeze_last_spatial_blocks:
                trainable = trainable or name.startswith("core.final_layer.")
                for block_index in range(first_unfrozen_spatial_block, args.depth):
                    trainable = trainable or name.startswith((
                        f"core.blocks.{block_index}.s_attn.",
                        f"core.blocks.{block_index}.s_mlp.",
                        f"core.blocks.{block_index}.s_adaLN_modulation.",
                    ))
            parameter.requires_grad_(trainable)
    elif args.freeze_base_for_appearance:
        first_unfrozen_spatial_block = (
            args.depth - args.appearance_unfreeze_last_spatial_blocks
        )
        for name, parameter in raw_model.named_parameters():
            trainable = name.startswith(
                (
                    "core.appearance_condition_embedder.",
                    "core.appearance_detail_embedder.",
                    "core.appearance_detail_reinjectors.",
                    "reference_encoder.",
                    "core.entity_reference_adapters.",
                    "core.appearance_reinjectors.",
                )
            )
            if args.appearance_unfreeze_last_spatial_blocks:
                trainable = trainable or name.startswith("core.final_layer.")
                for block_index in range(first_unfrozen_spatial_block, args.depth):
                    trainable = trainable or name.startswith(
                        (
                            f"core.blocks.{block_index}.s_attn.",
                            f"core.blocks.{block_index}.s_mlp.",
                            f"core.blocks.{block_index}.s_adaLN_modulation.",
                        )
                    )
            parameter.requires_grad_(trainable)
    trainable_parameters = [p for p in raw_model.parameters() if p.requires_grad]
    if args.appearance_lr is not None:
        optimizer = build_full_player_optimizer(
            raw_model, base_lr=args.lr, appearance_lr=args.appearance_lr,
        )
    elif (
        args.freeze_base_for_parallel_player
        and args.parallel_player_unfreeze_last_blocks
        and args.unfrozen_base_lr_scale != 1
    ):
        branch_parameters, base_parameters = [], []
        branch_prefixes = (
            "reference_encoder.",
            "roi_appearance_projector.",
            "core.parallel_player_adapter.",
            "core.parallel_player_fusion_module.",
            "core.parallel_player_fusion_gate",
            "core.parallel_player_skip_gate",
            "resident_encoder.appearance_embedder.",
        )
        for name, parameter in raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            (branch_parameters if name.startswith(branch_prefixes) else base_parameters).append(
                parameter
            )
        optimizer = torch.optim.AdamW([
            {"name": "player_branch", "params": branch_parameters, "lr": args.lr},
            {
                "name": "unfrozen_tail",
                "params": base_parameters,
                "lr": args.lr * args.unfrozen_base_lr_scale,
            },
        ])
    elif (
        args.freeze_base_for_reference
        and args.reference_unfreeze_last_spatial_blocks
        and args.unfrozen_base_lr_scale != 1
    ):
        reference_parameters, base_parameters = [], []
        for name, parameter in raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith((
                "reference_encoder.",
                "core.entity_reference_adapters.",
                "core.unified_reference_adapter.",
            )):
                reference_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": reference_parameters, "lr": args.lr},
            {"params": base_parameters, "lr": args.lr * args.unfrozen_base_lr_scale},
        ])
    elif args.freeze_base_for_player_appearance:
        joint_patch_weight = raw_model.core.x_embedder.proj.weight
        appearance_parameters = [
            parameter for parameter in trainable_parameters
            if parameter is not joint_patch_weight
        ]
        # The hook freezes the established input columns. Disable decoupled
        # weight decay for this tensor so AdamW cannot move those frozen values.
        optimizer = torch.optim.AdamW([
            {"params": appearance_parameters},
            {"params": [joint_patch_weight], "weight_decay": 0.0},
        ], lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    optimizer.defaults.update(betas=(0.9, args.adam_beta2), weight_decay=args.weight_decay)
    for group in optimizer.param_groups:
        group["betas"] = (0.9, args.adam_beta2)
        # Respect explicit no-decay groups used by partial-freezing experiments.
        if group["weight_decay"] != 0:
            group["weight_decay"] = args.weight_decay
    if rank == 0:
        effective_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        if args.freeze_base_for_player_appearance:
            joint_patch_weight = raw_model.core.x_embedder.proj.weight
            effective_trainable_parameters -= joint_patch_weight.numel()
            effective_trainable_parameters += (
                joint_patch_weight[:, -raw_model.core.roi_appearance_dim:].numel()
            )
        print(json.dumps({
            "trainable_parameters": sum(p.numel() for p in trainable_parameters),
            "effective_trainable_parameters": effective_trainable_parameters,
            "total_parameters": sum(p.numel() for p in raw_model.parameters()),
        }))
    start_step = 0
    if args.resume:
        resume_report = load_renderer_resume(
            raw_model,
            optimizer,
            checkpoint,
            dataset.item_vocabulary,
            allow_item_vocabulary_extension=args.allow_item_vocabulary_extension,
        )
        start_step = int(checkpoint["step"])
        if args.resume_lr_override:
            checkpoint_lr = float(checkpoint["config"]["training"]["lr"])
            if not 0 < checkpoint_lr < float("inf"):
                raise ValueError("checkpoint learning rate must be finite and positive")
            lr_scale = args.lr / checkpoint_lr
            for group in optimizer.param_groups:
                group["lr"] *= lr_scale
                if "warmup_target_lr" in group:
                    group["warmup_target_lr"] *= lr_scale
            resume_report["lr_override"] = {
                "checkpoint_lr": checkpoint_lr,
                "target_lr": args.lr,
                "optimizer_group_lrs": [group["lr"] for group in optimizer.param_groups],
            }
        if rank == 0:
            print(json.dumps({"resume": str(args.resume), **resume_report}))
    elif args.warm_start:
        warm_state = dict(checkpoint["model"])
        migrated_parameters = []
        if args.other_player_roi_adaln:
            gate_key = "core.other_player_condition_gates"
            old_gates = warm_state.get(gate_key)
            new_gates = raw_model.state_dict().get(gate_key)
            if old_gates is not None and new_gates is not None and old_gates.shape != new_gates.shape:
                # Start all layers with the mean learned gate amplitude.
                # This changes conditioning and is explicitly a warm start.
                amplitude = old_gates.float().tanh().mean().clamp(-.99, .99)
                warm_state[gate_key] = torch.full_like(new_gates, amplitude.atanh().item())
                migrated_parameters.append("other_player_condition_gates:expand-to-all-layers")
        if args.other_player_roi_adaln and not args.independent_roi_appearance_tokens:
            # This is an architecture warm start, NOT an exact-function resume.
            # Fold the learned appearance projection and its FIRST injection gate
            # into the joint input. Later appearance reinjection is removed.
            first_key = "core.roi_appearance_token_embedder.0.weight"
            if first_key in warm_state:
                first = warm_state[first_key].float()
                second = warm_state["core.roi_appearance_token_embedder.1.weight"].float()
                patch_key = "core.x_embedder.proj.weight"
                patch = warm_state[patch_key].clone()
                if first.shape[-2:] != patch.shape[-2:]:
                    raise RuntimeError("joint appearance migration requires unchanged patches")
                factor = warm_state["core.roi_appearance_token_gate"].float()
                gates = warm_state.get("core.roi_appearance_condition_gates")
                if gates is not None and gates.numel():
                    factor = factor * gates[0].float().tanh()
                folded = torch.einsum("om,mihw->oihw", second[:, :, 0, 0], first)
                patch[:, -first.shape[1]:] = (factor * folded).to(patch.dtype)
                warm_state[patch_key] = patch
                for key in list(warm_state):
                    if key.startswith(("core.roi_appearance_token_embedder.",
                                       "core.roi_appearance_token_gate",
                                       "core.roi_appearance_condition_gates")):
                        del warm_state[key]
                migrated_parameters.append("appearance:first-injection-folded-into-joint-input;later-injections-removed")
        if args.unified_player_reference:
            # Reuse the trained first legacy reference adapter for the single
            # input-level adapter. Shapes are identical; only routing retains
            # the four-view axis instead of pre-averaging it.
            old_prefix = "core.entity_reference_adapters.0."
            new_prefix = "core.unified_reference_adapter."
            for key, value in list(warm_state.items()):
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    warm_state[new_key] = value
                    migrated_parameters.append(f"{key}->{new_key}")
            legacy_prefixes = (
                "resident_encoder.skin_view_encoder.",
                "resident_encoder.appearance_direction_embedding",
                "resident_encoder.skin_view_fusion.",
                "appearance_spatial_encoder.",
                "core.appearance_condition_embedder.",
                "core.appearance_detail_embedder.",
                "core.appearance_detail_reinjectors.",
                "core.appearance_reinjectors.",
                "core.entity_reference_adapters.",
            )
            warm_state = {
                key: value for key, value in warm_state.items()
                if not key.startswith(legacy_prefixes)
            }
        if args.split_player_conditioning and not args.parallel_player_feature:
            # The split model deliberately replaces the old mixed player
            # residual with input-level appearance plus shared spatial
            # action/state embeddings. Do not silently retain obsolete branch
            # parameters from a parent checkpoint that enabled it.
            legacy_parallel_prefixes=(
                "core.parallel_player_adapter.",
                "core.parallel_player_fusion_module.",
                "core.parallel_player_fusion_gate",
                "core.parallel_player_skip_gate",
            )
            removed=[
                key for key in warm_state
                if key.startswith(legacy_parallel_prefixes)
            ]
            warm_state={
                key:value for key,value in warm_state.items()
                if not key.startswith(legacy_parallel_prefixes)
            }
            migrated_parameters.extend(
                f"drop:{key}" for key in removed
            )
        migrated_parameters.extend(
            migrate_player_appearance_id_dimension(
                warm_state, raw_model.state_dict()
            )
        )
        if args.codec != "wan" and (
            args.roi_appearance_channels != 32
            or args.independent_roi_appearance_tokens
        ):
            migrated_parameters.extend(
                migrate_roi_appearance_fusion(
                    warm_state,
                    raw_model.state_dict(),
                    roi_appearance_channels=args.roi_appearance_channels,
                    independent_tokens=args.independent_roi_appearance_tokens,
                )
            )
        migrated_parameters.extend(
            migrate_player_fusion_blocks(
                warm_state,
                raw_model.state_dict(),
                player_fusion_depth=args.player_fusion_depth,
            )
        )
        if not args.player_fusion_depth:
            current_state = raw_model.state_dict()
            obsolete_player_fusion_prefixes = (
                "core.player_fusion_blocks.",
                "core.other_player_action_embedder.",
            )
            obsolete = [
                key for key in warm_state
                if key.startswith(obsolete_player_fusion_prefixes)
                and key not in current_state
            ]
            for key in obsolete:
                warm_state.pop(key)
            migrated_parameters.extend(
                f"drop:{key}" for key in obsolete
            )
        # Early long-run checkpoints may contain the zero-length placeholder
        # that preceded conditional registration of ROI reinjection gates. It
        # has no value or optimizer state to restore, so omit only that exact
        # legacy tensor when the current configuration does not own the key.
        legacy_empty_gate = "core.roi_appearance_reinject_gates"
        current_state = raw_model.state_dict()
        if (
            legacy_empty_gate in warm_state
            and legacy_empty_gate not in current_state
            and warm_state[legacy_empty_gate].numel() == 0
        ):
            warm_state.pop(legacy_empty_gate)
            migrated_parameters.append(legacy_empty_gate + ":drop-empty-legacy")
        if args.codec == "wan":
            new_patch = tuple(raw_model.cfg.patch_size)
            conv_keys = (
                "core.x_embedder.proj.weight",
                "core.roi_appearance_token_embedder.0.weight",
            )
            for key in conv_keys:
                if key not in warm_state or key not in current_state:
                    continue
                source, target = warm_state[key], current_state[key]
                if source.shape == target.shape:
                    continue
                if source.shape[:2] != target.shape[:2]:
                    raise RuntimeError(f"cannot migrate Wan patch tensor {key}")
                old_area = source.shape[-2] * source.shape[-1]
                resized = F.interpolate(
                    source.flatten(0, 1).unsqueeze(1).float(),
                    size=new_patch,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).unflatten(0, source.shape[:2])
                resized *= old_area / (new_patch[0] * new_patch[1])
                warm_state[key] = resized.to(source.dtype)
                migrated_parameters.append(key + ":resize-2x2-to-3x2")

            final_weight = "core.final_layer.linear.weight"
            final_bias = "core.final_layer.linear.bias"
            channels = raw_model.cfg.in_channels
            if final_weight in warm_state:
                source = warm_state[final_weight]
                target = current_state[final_weight]
                if source.shape != target.shape:
                    old_area = source.shape[0] // channels
                    old_h, old_w = (2, old_area // 2)
                    values = source.reshape(
                        old_h, old_w, channels, source.shape[1]
                    ).permute(2, 3, 0, 1).flatten(0, 1).unsqueeze(1)
                    values = F.interpolate(
                        values.float(), size=new_patch, mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).unflatten(0, (channels, source.shape[1]))
                    warm_state[final_weight] = values.permute(
                        2, 3, 0, 1
                    ).reshape_as(target).to(source.dtype)
                    migrated_parameters.append(
                        final_weight + ":resize-2x2-to-3x2"
                    )
            if final_bias in warm_state:
                source = warm_state[final_bias]
                target = current_state[final_bias]
                if source.shape != target.shape:
                    old_area = source.shape[0] // channels
                    old_h, old_w = (2, old_area // 2)
                    values = source.reshape(
                        old_h, old_w, channels
                    ).permute(2, 0, 1).unsqueeze(1)
                    values = F.interpolate(
                        values.float(), size=new_patch, mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    warm_state[final_bias] = values.permute(
                        1, 2, 0
                    ).reshape_as(target).to(source.dtype)
                    migrated_parameters.append(
                        final_bias + ":resize-2x2-to-3x2"
                    )
        incompatible = raw_model.load_state_dict(warm_state, strict=False)
        allowed_missing = (
            "core.hud_condition_embedder.",
            "core.condition_reinjectors.",
            "core.appearance_condition_embedder.",
            "core.appearance_detail_embedder.",
            "core.appearance_detail_reinjectors.",
            "reference_encoder.",
            "core.entity_reference_adapters.",
            "core.unified_reference_adapter.",
            "core.appearance_reinjectors.",
            "core.parallel_player_adapter.",
            "core.parallel_player_fusion_module.",
            "core.parallel_player_fusion_gate",
            "core.parallel_player_skip_gate",
            "resident_encoder.appearance_embedder.",
            "core.other_player_orientation_embedder.",
            "core.roi_appearance_token_embedder.",
            "core.roi_appearance_token_gate",
            "core.roi_appearance_condition_gates",
            "core.roi_appearance_reinject_gates",
            "core.other_player_condition_gates",
            "core.oracle_player_geometry_embedder.",
            "core.player_fusion_blocks.",
            "core.other_player_action_embedder.",
            "temporal_condition_adapter.",
            "core.x_embedder.proj.",
            "core.final_layer.linear.",
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible warm start: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        start_step = int(checkpoint.get("step", 0))
        if rank == 0:
            print(json.dumps({
                "warm_start": str(args.warm_start),
                "step": start_step,
                "initialized_parameters": incompatible.missing_keys,
                "migrated_parameters": migrated_parameters,
            }))
    if not 0 <= args.ema_decay < 1:
        raise ValueError('ema-decay must be in [0,1)')
    ema = (EvaluationEMA(raw_model, args.ema_decay,
                         checkpoint.get('ema') if args.resume else None)
           if args.ema_decay and (rank == 0 or args.visualize_every) else None)
    model = (
        DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=(
                args.history_self_condition_step_probability > 0
            ),
        )
        if world > 1
        else raw_model
    )
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    run_config = {
        "renderer": asdict(cfg),
        "codec": codec.get_config(),
        "training": vars(args),
        "item_vocabulary": dataset.item_vocabulary,
        "class_to_raw": dataset.vocabulary.class_to_raw,
        "optimizer_groups": [
            {"name": group.get("name", f"group_{index}"),
             "lr": group["lr"], "parameters": sum(p.numel() for p in group["params"]),
             **{key: group[key] for key in ("appearance_lr", "appearance_input_channels") if key in group}}
            for index, group in enumerate(optimizer.param_groups)
        ],
    }
    if rank == 0:
        (output / "config.json").write_text(json.dumps(run_config, indent=2))
    run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            dir=str(output),
            config={
                **run_config,
                "world_size": world,
                "effective_source_window_batch_size": (
                    args.batch_size * world * args.gradient_accumulation
                ),
                "effective_view_batch_size": (
                    args.batch_size
                    * args.target_views_per_window
                    * getattr(dataset, "group_size", 1)
                    * world
                    * args.gradient_accumulation
                ),
            },
        )
        (output / "wandb_run.json").write_text(
            json.dumps(
                {
                    "id": run.id,
                    "url": run.url,
                    "entity": args.wandb_entity,
                    "project": args.wandb_project,
                },
                indent=2,
            )
        )
    val_loader = None
    probes = []
    if args.val_window_index:
        val_dataset = TextAgentRendererDataset(
            args.dataset_root,
            args.vocabulary,
            split="val_id",
            context_frames=args.context_frames,
            window_index=args.val_window_index,
            chunk_cache_root=args.chunk_cache_root,
            entity_region_upweight=args.latent_entity_region_upweight,
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world, rank=rank, shuffle=False)
            if world > 1
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_renderer,
            **worker_kwargs,
        )
        if rank == 0 and args.visualize_every:
            probes = (
                [RendererProbe(**row) for row in json.loads(Path(args.visualization_probe_manifest).read_text())]
                if args.visualization_probe_manifest else select_renderer_probes(val_dataset)
            )
            save_probe_manifest(probes, output / "visualizations" / "probes.json")
    elif rank == 0 and args.visualize_every:
        print(
            "warning: --visualize-every requires --val-window-index; visualization disabled",
            flush=True,
        )
    iterator = iter(loader)
    counterfactual_iterator = (
        iter(counterfactual_loader) if counterfactual_loader is not None else None
    )
    epoch = 0
    counterfactual_epoch = 0
    auxiliary_weights = effective_auxiliary_loss_weights(args)
    loss_kwargs = {
        "flow_loss_weight": args.flow_loss_weight,
        "player_flow_loss_weight": args.player_flow_loss_weight,
        "frames_per_sample": args.pixel_loss_frames,
        "pixel_frame_selection": args.pixel_frame_selection,
        "entity_pixel_l1_weight": auxiliary_weights["entity_pixel_l1_weight"],
        "entity_pixel_edge_weight": auxiliary_weights["entity_pixel_edge_weight"],
        "player_pixel_l1_weight": auxiliary_weights["player_pixel_l1_weight"],
        "player_pixel_edge_weight": auxiliary_weights["player_pixel_edge_weight"],
        "player_roi_l1_weight": auxiliary_weights["player_roi_l1_weight"],
        "player_roi_edge_weight": auxiliary_weights["player_roi_edge_weight"],
        "player_roi_size": tuple(args.player_roi_size),
        "player_roi_padding": args.player_roi_padding,
        "health_pixel_l1_weight": auxiliary_weights["health_pixel_l1_weight"],
        "damaged_health_upweight": args.damaged_health_upweight,
        "player_ring_radius": args.player_ring_radius,
        "player_ring_weight": args.player_ring_weight,
        "identity_encoder": identity_encoder,
        "player_identity_loss_weight": auxiliary_weights["player_identity_loss_weight"],
        "player_identity_margin": args.player_identity_margin,
        "player_identity_negative_weight": args.player_identity_negative_weight,
        "counterfactual_group_size": getattr(dataset, "group_size", 1),
        "counterfactual_player_difference_weight": (
            auxiliary_weights["counterfactual_player_difference_weight"]
        ),
        "counterfactual_pure_noise": not args.counterfactual_random_timesteps,
        "ordinary_pure_noise_probability": args.ordinary_pure_noise_probability,
        "time_distribution": args.flow_time_distribution,
        "sigma_min": args.flow_sigma_min,
        "low_noise_mix_probability": args.flow_low_noise_mix_probability,
        "low_noise_max": args.flow_low_noise_max,
        "reference_latents": (
            0 if args.persist_protocol or args.reference_pixel_frames == 0
            else 1 + (args.reference_pixel_frames - 1) // temporal_downsampling
        ),
    }
    train_diagnostic_totals = torch.zeros(len(PLAYER_DIAGNOSTIC_KEYS), device=device)
    if args.lr_warmup_steps:
        for group in optimizer.param_groups:
            # Persist both origin and target so resume cannot restart warmup.
            group.setdefault("warmup_target_lr", group["lr"])
            group.setdefault("warmup_start_step", start_step if (
                args.lr_warmup_relative_to_start and not args.resume
            ) else 0)
    stage_origin = optimizer.param_groups[0].get("warmup_start_step", start_step)
    if rank == 0:
        print(json.dumps({"training_stage": {"parent_step": start_step,
            "warmup_start_step": stage_origin, "reference_pixel_frames": args.reference_pixel_frames,
            "optimizer": [{k: g[k] for k in ("lr", "betas", "weight_decay")} for g in optimizer.param_groups],
            "grad_clip_norm": args.grad_clip_norm}}), flush=True)
    for step in range(start_step + 1, args.steps + 1):
        if args.lr_warmup_steps:
            for group in optimizer.param_groups:
                group["lr"] = group["warmup_target_lr"] * min(
                    max(0, step - group["warmup_start_step"]) / args.lr_warmup_steps, 1.0
                )
        profile_this_step = args.profile_local_step == step - start_step
        profile_timings = {}
        profile_step_started = _start_profile_range(profile_this_step, device)
        optimizer.zero_grad(set_to_none=True)
        accumulated = {name: 0.0 for name in LOSS_NAMES}
        counterfactual_step = bool(
            counterfactual_loader is not None
            and use_counterfactual_step(
                step, args.counterfactual_step_probability, args.seed
            )
        )
        history_self_condition_step = bool(
            not counterfactual_step
            and use_counterfactual_step(
                step,
                args.history_self_condition_step_probability,
                args.seed + 0x5E1F,
            )
        )
        active_dataset = counterfactual_dataset if counterfactual_step else dataset
        for micro in range(args.gradient_accumulation):
            data_started = perf_counter() if profile_this_step else None
            if counterfactual_step:
                try:
                    batch = next(counterfactual_iterator)
                except StopIteration:
                    counterfactual_epoch += 1
                    if counterfactual_sampler is not None:
                        counterfactual_sampler.set_epoch(counterfactual_epoch)
                    counterfactual_iterator = iter(counterfactual_loader)
                    batch = next(counterfactual_iterator)
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
            if data_started is not None:
                profile_timings["data_wait_cpu"] = perf_counter() - data_started
            transfer_started = _start_profile_range(profile_this_step, device)
            rgb = batch["rgb"].to(device)
            prefix_mask_probability = (
                args.counterfactual_mask_prefix_player_probability
                if counterfactual_step or args.appearance_counterfactual
                else args.mask_prefix_player_probability
            )
            if prefix_mask_probability > 0:
                group_size = getattr(active_dataset, "group_size", 1)
                group_count = len(rgb) // group_size
                gate = (torch.rand(group_count, device=device)
                        < prefix_mask_probability)
                gate = gate.repeat_interleave(group_size)
                prefix_player = batch["player_region_mask"][:, :1].to(device).bool()
                rgb = rgb.clone()
                rgb[:, :1] = torch.where(
                    gate[:, None, None, None, None] & prefix_player,
                    rgb.new_tensor(0.5), rgb[:, :1],
                )
            _finish_profile_range(
                profile_timings, "rgb_to_gpu_and_prefix", transfer_started, device
            )
            encode_started = _start_profile_range(profile_this_step, device)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"),
            ):
                latent = codec.encode(rgb)
            _finish_profile_range(profile_timings, "vae_encode", encode_started, device)
            loss_forward_started = _start_profile_range(profile_this_step, device)
            conditions = {k: v.to(device) for k, v in batch["conditions"].items()}
            if args.target_state_dropout_probability > 0:
                group_size = getattr(active_dataset, "group_size", 1)
                group_count = len(rgb) // group_size
                target_state_dropout_mask = (
                    torch.rand(group_count, device=device)
                    < args.target_state_dropout_probability
                )
                conditions["target_state_dropout_mask"] = (
                    target_state_dropout_mask.repeat_interleave(group_size)
                )
            flow_diagnostics = {} if args.log_player_noise_bins else None
            player_region_mask = batch["player_region_mask"].to(device)
            flow_region_weight = build_flow_region_weight(
                batch["region_weight"].to(device),
                player_region_mask,
                player_upweight=args.latent_player_region_upweight,
                probability=(
                    args.counterfactual_player_mask_probability
                    if counterfactual_step or args.appearance_counterfactual
                    else args.player_mask_probability
                ),
                randomize=True,
            )
            sync = (
                model.no_sync()
                if world > 1 and micro + 1 < args.gradient_accumulation
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    if history_self_condition_step:
                        self_condition_loss = renderer_history_self_condition_loss(
                            model,
                            latent,
                            conditions,
                            region_weight=flow_region_weight,
                            denoising_steps=args.history_self_condition_denoising_steps,
                        )
                        zero = self_condition_loss.detach().new_zeros(())
                        losses = {name: zero for name in LOSS_NAMES}
                        losses["total_loss"] = self_condition_loss
                        losses["flow_loss"] = self_condition_loss
                    else:
                        losses = renderer_training_losses(
                            model,
                            codec,
                            latent,
                            conditions,
                            rgb,
                            batch["pixel_region_mask"].to(device),
                            player_region_mask=player_region_mask,
                            region_weight=flow_region_weight,
                            player_identity_mask=batch["player_identity_mask"].to(device),
                            player_identity_frame=batch["player_identity_frame"].to(device),
                            player_identity_slot=batch["player_identity_slot"].to(device),
                            player_identity_valid=batch["player_identity_valid"].to(device),
                            profile_timings=profile_timings if profile_this_step else None,
                            flow_diagnostics=flow_diagnostics,
                            **{
                                **loss_kwargs,
                                "counterfactual_group_size": getattr(
                                    active_dataset, "group_size", 1
                                ),
                                "counterfactual_player_difference_weight": (
                                    auxiliary_weights[
                                        "counterfactual_player_difference_weight"
                                    ]
                                    if counterfactual_step or args.appearance_counterfactual
                                    else 0.0
                                ),
                            },
                        )
                    loss = losses["total_loss"] / args.gradient_accumulation
                _finish_profile_range(
                    profile_timings, "loss_forward_total", loss_forward_started, device
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite M3 loss at step {step}")
                backward_started = _start_profile_range(profile_this_step, device)
                loss.backward()
                _finish_profile_range(
                    profile_timings, "backward", backward_started, device
                )
            for name in LOSS_NAMES:
                accumulated[name] += float(losses[name].detach()) / args.gradient_accumulation
            if flow_diagnostics is not None:
                train_diagnostic_totals.add_(pack_player_diagnostics(flow_diagnostics, device=device))
        if (
            args.fixed_spatial_reference
            and (step == start_step + 1 or (
                step == 10 and not (args.resume or args.warm_start or args.backbone_checkpoint)
            ))
            and not history_self_condition_step
        ):
            # A fresh DiT has a zero output projection: upstream gradients are
            # legitimately zero on step 1. Check again at step 10, strictly.
            allow_initial_zero_gradient = (
                step == 1
                and not (args.resume or args.warm_start or args.backbone_checkpoint)
                and torch.count_nonzero(raw_model.core.final_layer.linear.weight).item() == 0
            )
            branch_gradients={}
            if args.parallel_player_feature:
                modules=[
                    ('parallel_player_adapter',raw_model.core.parallel_player_adapter)
                ]
            elif args.freeze_base_for_player_fusion:
                modules=[
                    ('player_fusion_blocks',raw_model.core.player_fusion_blocks),
                    ('roi_appearance_projector',raw_model.roi_appearance_projector),
                    (
                        'independent_roi_appearance_tokens',
                        raw_model.core.roi_appearance_token_embedder,
                    ),
                    (
                        'other_player_action_embedder',
                        raw_model.core.other_player_action_embedder,
                    ),
                    (
                        'other_player_orientation_embedder',
                        raw_model.core.other_player_orientation_embedder,
                    ),
                ]
            elif args.split_player_conditioning:
                modules=[
                    ('roi_appearance_projector',raw_model.roi_appearance_projector),
                    ('joint_patch',raw_model.core.x_embedder),
                    ('shared_action_embedder',raw_model.core.action_embedder),
                ]
                if args.independent_roi_appearance_tokens:
                    modules.append((
                        'independent_roi_appearance_tokens',
                        raw_model.core.roi_appearance_token_embedder,
                    ))
                if args.other_player_action_orientation_only:
                    modules.append((
                        'other_player_orientation_embedder',
                        raw_model.core.other_player_orientation_embedder,
                    ))
                elif not args.other_player_action_only:
                    modules.append((
                        'shared_state_embedder',raw_model.core.extra_condition_embedder,
                    ))
            else:
                modules=[
                    ('reference_encoder',raw_model.reference_encoder),
                    ('roi_appearance_projector',raw_model.roi_appearance_projector),
                    ('joint_patch',raw_model.core.x_embedder),
                ]
            if args.discrete_player_appearance:
                modules.append((
                    'appearance_embedder', raw_model.resident_encoder.appearance_embedder,
                ))
            if args.oracle_player_geometry:
                modules.append((
                    'oracle_player_geometry',
                    raw_model.core.oracle_player_geometry_embedder,
                ))
            for name,module in modules:
                gradient_terms = [
                    p.grad.detach().float().square().sum()
                    for p in module.parameters()
                    if p.grad is not None
                ]
                if not gradient_terms:
                    raise FloatingPointError(
                        f'fixed branch has no gradient tensors: {name}'
                    )
                norm=torch.stack(gradient_terms).sum().sqrt()
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'fixed branch gradient invalid: {name}')
                if norm <= 0 and not allow_initial_zero_gradient:
                    raise FloatingPointError(f'fixed branch gradient invalid: {name}')
                branch_gradients[name]=norm.item()
            if args.spatial_appearance_only_conditioning:
                # The linear action embedder bias is an unconditional learned
                # constant and legitimately receives a gradient even when its
                # input is zero.  The weight gradient is the direct test that
                # no action value reached the denoiser.
                action_weight_grad = (
                    raw_model.core.action_embedder.weight.grad.detach().float().norm()
                )
                if not torch.isfinite(action_weight_grad) or action_weight_grad != 0:
                    raise FloatingPointError(
                        'dynamic action leaked into spatial appearance stage'
                    )
                branch_gradients['shared_action_weight'] = action_weight_grad.item()
            if args.roi_appearance_channels > 32:
                new_channel_grad = (
                    raw_model.roi_appearance_projector.fuse[2].weight.grad[32:]
                    .detach().float().norm()
                )
                if not torch.isfinite(new_channel_grad) or (new_channel_grad <= 0 and not allow_initial_zero_gradient):
                    raise FloatingPointError(
                        'fixed branch gradient invalid: roi_projector_new_channels'
                    )
                branch_gradients['roi_projector_new_channels'] = (
                    new_channel_grad.item()
                )
            if args.player_appearance_id_dim > 30:
                prefix = (
                    raw_model.roi_appearance_projector.fuse[0].weight.shape[1]
                    - args.player_appearance_id_dim
                    - 8
                )
                new_id_grad = (
                    raw_model.roi_appearance_projector.fuse[0].weight.grad[
                        :, prefix + 30:prefix + args.player_appearance_id_dim
                    ].detach().float().norm()
                )
                if not torch.isfinite(new_id_grad) or (new_id_grad <= 0 and not allow_initial_zero_gradient):
                    raise FloatingPointError(
                        'fixed branch gradient invalid: appearance_id_new_columns'
                    )
                branch_gradients['appearance_id_new_columns'] = new_id_grad.item()
            if args.roi_appearance_reinject_blocks:
                reinject_grad = raw_model.core.roi_appearance_reinject_gates.grad
                if (
                    reinject_grad is None
                    or not torch.isfinite(reinject_grad).all()
                    or (
                        reinject_grad.detach().float().norm() <= 0
                        and not allow_initial_zero_gradient
                    )
                ):
                    raise FloatingPointError(
                        'fixed branch gradient invalid: roi_appearance_reinject_gates'
                    )
                branch_gradients['roi_appearance_reinject_gates'] = (
                    reinject_grad.detach().float().norm().item()
                )
            if args.roi_appearance_condition_blocks:
                appearance_gate_grad = (
                    raw_model.core.roi_appearance_condition_gates.grad
                )
                if (
                    appearance_gate_grad is None
                    or not torch.isfinite(appearance_gate_grad).all()
                    or (
                        appearance_gate_grad.detach().float().norm() <= 0
                        and not allow_initial_zero_gradient
                    )
                ):
                    raise FloatingPointError(
                        'fixed branch gradient invalid: '
                        'roi_appearance_condition_gates'
                    )
                branch_gradients['roi_appearance_condition_gates'] = (
                    appearance_gate_grad.detach().float().norm().item()
                )
            if args.gated_other_player_conditioning:
                action_gate_grad = raw_model.core.other_player_condition_gates.grad
                if (
                    action_gate_grad is None
                    or not torch.isfinite(action_gate_grad).all()
                    or (
                        action_gate_grad.detach().float().norm() <= 0
                        and not allow_initial_zero_gradient
                    )
                ):
                    raise FloatingPointError(
                        'fixed branch gradient invalid: '
                        'other_player_condition_gates'
                    )
                branch_gradients['other_player_condition_gates'] = (
                    action_gate_grad.detach().float().norm().item()
                )
            if args.appearance_uv_view_condition:
                projector=raw_model.roi_appearance_projector
                grad=projector.fuse[0].weight.grad.detach().float()
                pose_start=128+2+2+projector.actor_dim
                for label,columns in [('appearance_uv',slice(130,132)),('appearance_view',slice(pose_start,pose_start+6))]:
                    norm=grad[:,columns].norm()
                    if not torch.isfinite(norm) or norm<=0:
                        raise FloatingPointError('UV/view branch gradient invalid: '+label)
                    branch_gradients[label]=float(norm)
            if rank==0:print(json.dumps({'fixed_branch_first_step_gradients':branch_gradients}),flush=True)
        clip_started = _start_profile_range(profile_this_step, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), args.grad_clip_norm, error_if_nonfinite=True
        )
        _finish_profile_range(profile_timings, "grad_clip", clip_started, device)
        optimizer_started = _start_profile_range(profile_this_step, device)
        optimizer.step()
        if ema is not None:
            ema.update(raw_model)
        _finish_profile_range(profile_timings, "optimizer_step", optimizer_started, device)
        if profile_this_step:
            _finish_profile_range(
                profile_timings, "step_wall_gpu_synchronized", profile_step_started, device
            )
            loss_forward = profile_timings["loss_forward_total"]
            nested = (
                profile_timings.get("m3_forward", 0.0)
                + profile_timings.get("vae_decode_for_loss", 0.0)
            )
            profile_timings["loss_forward_other"] = max(0.0, loss_forward - nested)
            exclusive_names = (
                "rgb_to_gpu_and_prefix",
                "vae_encode",
                "m3_forward",
                "vae_decode_for_loss",
                "loss_forward_other",
                "backward",
                "grad_clip",
                "optimizer_step",
            )
            exclusive_total = sum(
                profile_timings.get(name, 0.0) for name in exclusive_names
            )
            wall_total = profile_timings["step_wall_gpu_synchronized"]
            accounted = exclusive_total + profile_timings.get("data_wait_cpu", 0.0)
            profile_timings["unattributed_wall"] = max(0.0, wall_total - accounted)
            report_names = ("data_wait_cpu", *exclusive_names, "unattributed_wall")
            report = {
                "rank": rank,
                "world_size": world,
                "profiled_global_step": step,
                "profiled_local_step": step - start_step,
                "loss_mode": args.loss_mode,
                "batch_size": args.batch_size,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "seconds": profile_timings,
                "wall_percent": {
                    name: 100.0 * profile_timings.get(name, 0.0) / max(wall_total, 1e-12)
                    for name in report_names
                },
            }
            print("PROFILE_STEP_TIMING " + json.dumps(report, sort_keys=True), flush=True)
            if rank == 0 and run:
                run.finish()
            if world > 1:
                dist.destroy_process_group()
            return
        reduced = torch.tensor([accumulated[name] for name in LOSS_NAMES], device=device)
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
        log_now = step == start_step + 1 or step % args.log_every == 0 or step == args.steps
        diagnostic_log = {}
        if log_now and args.log_player_noise_bins:
            diagnostic_totals = train_diagnostic_totals.clone()
            if world > 1:
                dist.all_reduce(diagnostic_totals)
            if rank == 0:
                diagnostic_log = summarize_player_diagnostics(diagnostic_totals)
            train_diagnostic_totals.zero_()
        if log_now:
            local_memory = torch.tensor(
                [
                    torch.cuda.memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30,
                ],
                device=device,
            )
            if world > 1:
                memory_by_rank = [torch.zeros_like(local_memory) for _ in range(world)]
                dist.all_gather(memory_by_rank, local_memory)
                memory_by_rank = torch.stack(memory_by_rank)
            else:
                memory_by_rank = local_memory[None]
        if rank == 0 and log_now:
            train_log = {
                "train/reference_pixel_frames": args.reference_pixel_frames,
                "train/clean_reference_latents": loss_kwargs["reference_latents"],
                "train/persist_protocol": int(args.persist_protocol),
                "train/roi_flow_multiplier": args.roi_flow_multiplier,
                "train/appearance_uv_view_condition": int(args.appearance_uv_view_condition),
                **{f"train/{name}": float(reduced[index])
                   for index, name in enumerate(LOSS_NAMES)},
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/stage_step": step - stage_origin,
                "train/grad_norm_before_clip": float(grad_norm),
                "train/grad_clip_triggered": float(grad_norm > args.grad_clip_norm),
                **{f"train/lr_{group['name']}": group["lr"]
                   for group in optimizer.param_groups if "name" in group},
                "train/cuda_memory_allocated_max_gib": float(memory_by_rank[:, 0].max()),
                "train/cuda_peak_allocated_max_gib": float(memory_by_rank[:, 1].max()),
                "train/cuda_peak_reserved_max_gib": float(memory_by_rank[:, 2].max()),
                "train/counterfactual_step": float(counterfactual_step),
                "train/history_self_condition_step": float(history_self_condition_step),
                **{f"train/{key}": value for key, value in diagnostic_log.items()},
            }
            memory_text = " ".join(
                f"rank{index}_peak_reserved_gib={float(values[2]):.2f}"
                for index, values in enumerate(memory_by_rank)
            )
            print(
                f"step={step} source={('selfcond' if history_self_condition_step else 's11' if counterfactual_step else 'ordinary')} "
                f"loss={train_log['train/total_loss']:.6f} "
                f"flow={train_log['train/flow_loss']:.6f} "
                f"stage_step={step - stage_origin} lr={optimizer.param_groups[0]['lr']:.8g} "
                f"grad_norm={float(grad_norm):.4f} "
                f"player_flow={train_log['train/player_flow_loss']:.6f} "
                f"entity_l1={train_log['train/entity_pixel_l1']:.6f} "
                f"health_l1={train_log['train/health_pixel_l1']:.6f} "
                f"identity={train_log['train/player_identity_loss']:.6f} "
                f"identity_sim={train_log['train/player_identity_similarity']:.4f} "
                f"player_l1={train_log['train/player_pixel_l1']:.6f} "
                f"counterfactual={train_log['train/counterfactual_player_difference_loss']:.6f} "
                f"peak_reserved_max_gib={train_log['train/cuda_peak_reserved_max_gib']:.2f} "
                f"{memory_text}",
                flush=True,
            )
            if run:
                run.log(train_log, step=step)
            with (output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **train_log}) + "\n")
        # A STOP marker requests a checkpoint at this completed optimizer step.
        # Unlike a process signal this also works with distributed launchers.
        stop_requested = torch.tensor(int((output / "STOP").exists()), device=device)
        if world > 1:
            dist.all_reduce(stop_requested, op=dist.ReduceOp.MAX)
        stopping = bool(stop_requested.item())
        if not stopping and val_loader is not None and (step == start_step + 1 or step % args.validate_every == 0 or step == args.steps):
            raw_model.eval()
            values = []
            reference9_values = []
            val_diagnostic_totals = torch.zeros(len(PLAYER_DIAGNOSTIC_KEYS), device=device)
            with torch.no_grad():
                for number, val in enumerate(val_loader):
                    if number >= args.val_batches:
                        break
                    rgb = val["rgb"].to(device)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"
                    ):
                        latent = codec.encode(rgb)
                        condition = {k: v.to(device) for k, v in val["conditions"].items()}
                        val_diagnostics = {} if args.log_player_noise_bins else None
                        val_loss_kwargs = dict(
                            loss_kwargs,
                            reference_latents=loss_kwargs['reference_latents'],
                            counterfactual_group_size=1,
                            counterfactual_player_difference_weight=0.0,
                        )
                        val_losses = renderer_training_losses(
                            raw_model,
                            codec,
                            latent,
                            condition,
                            rgb,
                            val["pixel_region_mask"].to(device),
                            player_region_mask=val["player_region_mask"].to(device),
                            region_weight=build_flow_region_weight(
                                val["region_weight"].to(device),
                                val["player_region_mask"].to(device),
                                player_upweight=args.latent_player_region_upweight,
                                probability=args.player_mask_probability,
                                randomize=False,
                            ),
                            player_identity_mask=val["player_identity_mask"].to(device),
                            player_identity_frame=val["player_identity_frame"].to(device),
                            player_identity_slot=val["player_identity_slot"].to(device),
                            player_identity_valid=val["player_identity_valid"].to(device),
                            generator=torch.Generator(device=device).manual_seed(
                                # Give each distributed validation shard its
                                # own stable stream. With eight fixed probes
                                # and eight ranks this preserves the former
                                # single-GPU, batch-one seeds for every probe.
                                args.seed + number * world + rank
                            ),
                            **val_loss_kwargs,
                            flow_diagnostics=val_diagnostics,
                        )
                        values.append(torch.stack([val_losses[name] for name in LOSS_NAMES]))
                        if args.reference_pixel_frames != args.comparison_reference_pixel_frames:
                            reference9_values.append(renderer_flow_loss(
                                raw_model, latent, condition, reference_latents=1 + (args.comparison_reference_pixel_frames - 1) // temporal_downsampling,
                                generator=torch.Generator(device=device).manual_seed(args.seed + number * world + rank),
                                time_distribution=args.flow_time_distribution,
                                sigma_min=args.flow_sigma_min,
                                low_noise_mix_probability=args.flow_low_noise_mix_probability,
                                low_noise_max=args.flow_low_noise_max,
                            ))
                        if val_diagnostics is not None:
                            val_diagnostic_totals.add_(pack_player_diagnostics(val_diagnostics, device=device))
            metric = (
                torch.stack(values).mean(0)
                if values
                else torch.full((len(LOSS_NAMES),), float("nan"), device=device)
            )
            if world > 1:
                dist.all_reduce(metric)
                metric /= world
                if args.log_player_noise_bins:
                    dist.all_reduce(val_diagnostic_totals)
            reference9_metric = torch.stack(reference9_values).mean() if reference9_values else None
            if reference9_metric is not None and world > 1:
                dist.all_reduce(reference9_metric)
                reference9_metric /= world
            if rank == 0:
                val_metrics = {
                    name: float(metric[index]) for index, name in enumerate(LOSS_NAMES)
                }
                if args.log_player_noise_bins:
                    val_metrics.update(summarize_player_diagnostics(val_diagnostic_totals))
                if reference9_metric is not None:
                    val_metrics[f"reference{args.comparison_reference_pixel_frames}_flow_loss"] = float(reference9_metric)
                val_metrics['reference_pixel_frames'] = args.reference_pixel_frames
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, **val_metrics}) + "\n")
                if run:
                    run.log({f"val/{name}": value for name, value in val_metrics.items()}, step=step)
            raw_model.train()
        visualize = bool(
            not stopping and args.val_window_index
            and args.visualize_every
            and (step == start_step + 1 or step % args.visualize_every == 0 or step == args.steps)
        )
        if visualize:
            # All ranks receive the same plan; each (probe, weight variant) runs once.
            started = perf_counter()
            probe_payload = [probes if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(probe_payload, src=0)
            shared_probes = probe_payload[0]
            jobs = []
            for number, probe in enumerate(shared_probes):
                jobs.append((number, probe, False, args.reference_pixel_frames))
                if args.reference_pixel_frames != args.comparison_reference_pixel_frames:
                    jobs.append((number, probe, False, args.comparison_reference_pixel_frames))
                if ema is not None:
                    jobs.append((number, probe, True, args.reference_pixel_frames))
            visual_dir = output / "visualizations" / f"step_{step:07d}"
            local_results = []
            raw_model.eval()
            for job_id in range(rank, len(jobs), world):
                number, probe, use_ema, references = jobs[job_id]
                suffix = f"{'ema_' if use_ema else ''}reference{references}"
                path = visual_dir / f"{probe.name}_{suffix}.mp4"
                result = dict(job_id=job_id, rank=rank, probe=probe.name,
                              ema=use_ema, references=references, path=str(path))
                job_started = perf_counter()
                try:
                    visual_dir.mkdir(parents=True, exist_ok=True)
                    sample = collate_renderer([TextAgentRendererDataset.read_window(
                        probe.episode, val_dataset.vocabulary, start=probe.start,
                        target=probe.target, context_frames=args.context_frames)])
                    # History noise must not depend on task assignment or disturb training RNG.
                    with torch.random.fork_rng(devices=[device.index]):
                        torch.manual_seed(args.seed + number)
                        with ema.apply(raw_model) if use_ema else contextlib.nullcontext():
                            result['metrics'] = render_probe(
                                raw_model, codec, sample, path, seed=args.seed + number,
                                denoising_steps=args.visualization_denoising_steps,
                                precision=args.precision, reference_pixel_frames=references,
                                inference_time_shift=args.inference_time_shift)
                except Exception as error:
                    result['error'] = f"{type(error).__name__}: {error}"
                finally:
                    raw_model.clear_cache()
                result['seconds'] = perf_counter() - job_started
                local_results.append(result)
                print(json.dumps({'visualization_job': result}), flush=True)
            raw_model.train()
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, local_results)
            else:
                gathered[0] = local_results
            if rank == 0:
                results = sorted((r for group in gathered for r in group), key=lambda r: r['job_id'])
                visual_log = {}
                for result in results:
                    name = result['probe']
                    prefix = (f"visual/ema_reference{result['references']}/{name}" if result['ema']
                              else f"visual/{name}" if result['references'] == args.reference_pixel_frames
                              else f"visual/reference{result['references']}/{name}")
                    if 'error' in result:
                        visual_log[prefix + '_failure'] = 1
                    else:
                        visual_log.update({prefix + '_' + k: v for k,v in result['metrics'].items()})
                try:
                    (visual_dir / 'parallel_results.json').write_text(json.dumps(results, indent=2))
                    (visual_dir / 'metrics.json').write_text(json.dumps(visual_log, indent=2))
                    if run:
                        import wandb
                        for result in results:
                            if 'error' not in result:
                                label = f"visual/video/{result['probe']}/{'ema' if result['ema'] else 'raw'}_reference{result['references']}"
                                visual_log[label] = wandb.Video(result['path'], format='mp4')
                        run.log(visual_log, step=step)
                except Exception as error:
                    print(f"warning: visualization logging failed: {error}", flush=True)
                print(json.dumps({'visualization_complete': dict(step=step, jobs=len(results),
                    failed=sum('error' in r for r in results), seconds=perf_counter()-started)}), flush=True)
            if world > 1:
                dist.barrier()
        save_now = stopping or step % args.save_every == 0 or step == args.steps
        if save_now and world > 1:
            dist.barrier()
        checkpoint_error = None
        if rank == 0 and save_now:
            destination = output / f"step_{step:07d}.pt"
            try:
                staged_torch_save(
                    {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": run_config,
                        "ema": ema.state_dict() if ema is not None else None,
                    },
                    destination,
                    staging_dir=args.checkpoint_staging_dir,
                )
                if run:
                    run.log({"checkpoint/saved_step": step}, step=step)
            except Exception as error:
                try:
                    record_checkpoint_failure(output, destination, error)
                except Exception as record_error:
                    print(
                        f"warning: could not write checkpoint failure record: {record_error}",
                        flush=True,
                    )
                print(f"warning: checkpoint save failed at step {step}: {error}", flush=True)
                if run:
                    run.log(
                        {
                            "checkpoint/save_failure": 1,
                            "checkpoint/save_failure_message": str(error),
                        },
                        step=step,
                    )
                if args.checkpoint_errors == "raise":
                    checkpoint_error = str(error)
        if save_now and world > 1:
            status = [checkpoint_error]
            dist.broadcast_object_list(status, src=0)
            checkpoint_error = status[0]
        if checkpoint_error is not None:
            raise RuntimeError(f"checkpoint save failed: {checkpoint_error}")
        if stopping:
            if rank == 0:
                print(f"Stopped by STOP marker after checkpoint at step={step}", flush=True)
            break
    if world > 1:
        dist.barrier()
    if rank == 0 and run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
