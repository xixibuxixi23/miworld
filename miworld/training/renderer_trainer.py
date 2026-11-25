"""Continuous latent flow matching over causal, bidirectional eight-frame blocks."""
from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F

from miworld.models.player_identity import crop_masked_players


STATIC_KEYS = {"target_agent", "player_skin", "player_reference", "player_appearance_valid"}


def _sample_train_time(
    shape, device, generator, distribution,
    low_noise_mix_probability=0.0, low_noise_max=0.3,
):
    """Sample flow time, optionally replacing a fraction with low-noise time."""
    if distribution == "uniform":
        time = torch.rand(shape, device=device, generator=generator)
    elif distribution == "logit-normal":
        time = torch.randn(shape, device=device, generator=generator).sigmoid()
    else:
        raise ValueError("time_distribution must be uniform or logit-normal")
    if not 0.0 <= low_noise_mix_probability <= 1.0:
        raise ValueError("low_noise_mix_probability must be between 0 and 1")
    if not 0.0 < low_noise_max <= 1.0:
        raise ValueError("low_noise_max must be in (0,1]")
    if low_noise_mix_probability:
        low_noise = torch.rand(shape, device=device, generator=generator) * low_noise_max
        use_low_noise = (
            torch.rand(shape, device=device, generator=generator)
            < low_noise_mix_probability
        )
        time = torch.where(use_low_noise, low_noise, time)
    return time


def _profile_cuda_call(timings, name, tensor, function):
    """Time one optional CUDA range without changing the normal training path."""
    if timings is None:
        return function()
    torch.cuda.synchronize(tensor.device)
    started = perf_counter()
    result = function()
    torch.cuda.synchronize(tensor.device)
    timings[name] = timings.get(name, 0.0) + perf_counter() - started
    return result


def slice_conditions(cond, start, end):
    return {key: value if key in STATIC_KEYS else value[:, start:end]
            for key, value in cond.items()}


def _sample_blockwise_train_time(
    batch_size, total_frames, block_frames, device, generator=None,
    counterfactual_group_size=1, counterfactual_pure_noise=True,
    ordinary_pure_noise_probability=0.0,
    time_distribution="uniform",
    low_noise_mix_probability=0.0,
    low_noise_max=0.3,
):
    """Keep the observed prefix clean and assign one noise time per future block."""
    future_frames = total_frames - 1
    if future_frames < block_frames or future_frames % block_frames:
        raise ValueError("future training frames must contain complete output blocks")
    if counterfactual_group_size < 1 or batch_size % counterfactual_group_size:
        raise ValueError("batch size must be divisible by counterfactual group size")
    if not 0.0 <= ordinary_pure_noise_probability <= 1.0:
        raise ValueError("ordinary pure-noise probability must be between 0 and 1")
    groups = batch_size // counterfactual_group_size
    if counterfactual_group_size > 1 and counterfactual_pure_noise:
        # Counterfactual siblings have different clean player pixels. At an
        # ordinary interpolation time, those pixels remain in x_t and let the
        # network reconstruct the appearance without reading its reference.
        # Put dedicated counterfactual batches at the pure-noise endpoint so
        # every sibling receives exactly the same future latent; appearance is
        # then identifiable only from the per-player reference condition.
        block_time = torch.ones(
            groups, future_frames // block_frames, device=device,
        )
    else:
        block_time = _sample_train_time(
            (groups, future_frames // block_frames), device, generator,
            time_distribution, low_noise_mix_probability, low_noise_max,
        )
        if ordinary_pure_noise_probability > 0:
            # Inference starts every future frame at t=1. Give ordinary
            # training examples the same endpoint often enough that the model
            # cannot recover player appearance from residual clean pixels in
            # x_t and must instead read the reference condition.
            pure_noise_group = torch.rand(
                groups, 1, device=device, generator=generator,
            ) < ordinary_pure_noise_probability
            block_time = torch.where(
                pure_noise_group, torch.ones_like(block_time), block_time,
            )
    block_time = block_time.repeat_interleave(counterfactual_group_size, dim=0)
    time = torch.zeros(batch_size, total_frames, device=device)
    time[:, 1:] = block_time.repeat_interleave(block_frames, dim=1)
    return time


def _sample_clipwise_train_time(
    batch_size, total_frames, device, generator=None,
    counterfactual_group_size=1, counterfactual_pure_noise=True,
    ordinary_pure_noise_probability=0.0, time_shift=1.0,
    time_distribution="uniform",
    low_noise_mix_probability=0.0,
    low_noise_max=0.3,
):
    """One shared noise level, NOT shared Gaussian noise, over the future clip.

    Use the existing sampler as one large block. Optional rational shifting
    matches Gamma World's timestamp transform; shift=1 preserves uniform time.
    """
    if total_frames < 2 or not 0 < time_shift < float("inf"):
        raise ValueError("clip needs future frames and a finite positive time shift")
    time = _sample_blockwise_train_time(
        batch_size, total_frames, total_frames - 1, device, generator,
        counterfactual_group_size, counterfactual_pure_noise,
        ordinary_pure_noise_probability,
        time_distribution, low_noise_mix_probability, low_noise_max,
    )
    return time_shift * time / (1 + (time_shift - 1) * time)


def _sample_unconditional_train_time(
    batch_size, total_frames, device, generator=None,
    counterfactual_group_size=1, counterfactual_pure_noise=True,
    ordinary_pure_noise_probability=0.0, time_shift=1.0,
    time_distribution="uniform",
    low_noise_mix_probability=0.0,
    low_noise_max=0.3,
):
    """Assign one shared noise time when there is no clean reference prefix."""
    if total_frames < 1 or not 0 < time_shift < float("inf"):
        raise ValueError("unconditional clips need at least one frame and a positive time shift")
    if counterfactual_group_size < 1 or batch_size % counterfactual_group_size:
        raise ValueError("batch size must be divisible by counterfactual group size")
    if not 0.0 <= ordinary_pure_noise_probability <= 1.0:
        raise ValueError("ordinary pure-noise probability must be between 0 and 1")
    groups = batch_size // counterfactual_group_size
    if counterfactual_group_size > 1 and counterfactual_pure_noise:
        clip_time = torch.ones(groups, 1, device=device)
    else:
        clip_time = _sample_train_time(
            (groups, 1), device, generator, time_distribution,
            low_noise_mix_probability, low_noise_max,
        )
        if ordinary_pure_noise_probability > 0:
            pure_noise_group = torch.rand(
                groups, 1, device=device, generator=generator,
            ) < ordinary_pure_noise_probability
            clip_time = torch.where(
                pure_noise_group, torch.ones_like(clip_time), clip_time,
            )
    time = clip_time.repeat_interleave(counterfactual_group_size, dim=0)
    time = time.expand(batch_size, total_frames)
    return time_shift * time / (1 + (time_shift - 1) * time)


def renderer_flow_loss(
    model, clean, conditions, *, region_weight=None, generator=None,
    return_clean_prediction=False, counterfactual_group_size=1,
    counterfactual_pure_noise=True, ordinary_pure_noise_probability=0.0,
    profile_timings=None,
    player_region_mask=None, diagnostics=None, reference_latents=1,
    time_distribution="uniform", sigma_min=0.0,
    low_noise_mix_probability=0.0, low_noise_max=0.3,
):
    """Flow matching with an optional clean prefix; causal or full-clip mode.

    All unknown frames are scored in one forward pass. Frames within an output
    block share one noise time and attend bidirectionally; blocks remain causal.
    In bidirectional mode ALL future frames share one time and attend globally.
    Supervision masks only weight the loss and never enter neural conditions.
    """
    base = model.module if hasattr(model, "module") else model
    if not 0 <= reference_latents < clean.shape[1]:
        raise ValueError("reference_latents must be nonnegative and leave a scored latent")
    if reference_latents > 1 and not base.cfg.bidirectional_attention:
        raise ValueError("multiple reference latents require bidirectional training")
    if clean.ndim != 5 or clean.shape[1] > base.cfg.context_frames:
        raise ValueError("clean must be [B,T,C,H,W] within the configured context")
    if clean.shape[1] != base.cfg.context_frames:
        raise ValueError("training clips must fill the configured context")
    if base.core.kv_caches is not None:
        raise RuntimeError("clear rollout caches before training")
    if time_distribution not in {"uniform", "logit-normal"}:
        raise ValueError("time_distribution must be uniform or logit-normal")
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError("sigma_min must be in [0,1)")
    c = dict(conditions)
    c.pop("condition_mask", None)
    # The window boundary has no incoming action. This action-specific flag is
    # independent of clean/noisy status, which M3-Simple gets from time alone.
    prefix = (torch.arange(clean.shape[1], device=clean.device)[None] == 0)
    c["action_prefix_mask"] = prefix.expand(len(clean), -1)
    if getattr(base.core, "use_condition_mask", False):
        c["condition_mask"] = c["action_prefix_mask"]
    block_frames = base.cfg.block_frames
    persist = getattr(base.cfg, "persist_protocol", False)
    if persist:
        if base.cfg.bidirectional_attention or block_frames != 1:
            raise ValueError("PERSIST protocol requires causal single-latent blocks")
        # The legacy entry point represents disabled region weighting as all ones.
        if region_weight is not None:
            if not torch.all(region_weight == 1).item():
                raise ValueError("PERSIST protocol requires unweighted ordinary flow MSE")
            region_weight = None
        if counterfactual_group_size != 1:
            raise ValueError("PERSIST protocol requires unweighted ordinary flow MSE")
        reference_latents = 0
        sigma_min = 1e-5
        time = torch.randn(clean.shape[:2], device=clean.device, generator=generator).sigmoid()
    elif reference_latents == 0:
        time = _sample_unconditional_train_time(
            len(clean), clean.shape[1], clean.device, generator,
            counterfactual_group_size, counterfactual_pure_noise,
            ordinary_pure_noise_probability, base.cfg.clip_noise_time_shift,
            time_distribution, low_noise_mix_probability, low_noise_max,
        )
    elif getattr(base.cfg, "bidirectional_attention", False):
        time = _sample_clipwise_train_time(
            len(clean), clean.shape[1], clean.device, generator,
            counterfactual_group_size, counterfactual_pure_noise,
            ordinary_pure_noise_probability, base.cfg.clip_noise_time_shift,
            time_distribution, low_noise_mix_probability, low_noise_max,
        )
    else:
        time = _sample_blockwise_train_time(
            len(clean), clean.shape[1], block_frames, clean.device, generator,
            counterfactual_group_size, counterfactual_pure_noise,
            ordinary_pure_noise_probability,
            time_distribution, low_noise_mix_probability, low_noise_max,
        )
    time[:, :reference_latents] = 0
    noise = torch.randn(
        (len(clean) // counterfactual_group_size, *clean.shape[1:]),
        device=clean.device, dtype=clean.dtype, generator=generator,
    ).repeat_interleave(counterfactual_group_size, dim=0)
    if persist:
        noise = noise.clamp(-20, 20)
    tau = time[..., None, None, None]
    noise_scale = sigma_min + (1 - sigma_min) * tau
    noisy = (1 - tau) * clean + noise_scale * noise
    prediction = _profile_cuda_call(
        profile_timings,
        "m3_forward",
        clean,
        lambda: model(noisy, time, c, noise_cond=0.015) if persist else model(noisy, time, c),
    )
    target = (1 - sigma_min) * noise - clean
    error = (prediction[:, reference_latents:].float() - target[:, reference_latents:].float()).square()
    if diagnostics is not None:
        diagnostics["noise_time"] = time[:, reference_latents:].detach().mean()
        diagnostics["noise_time_per_frame"] = time[:, reference_latents:].detach()
        if player_region_mask is not None:
            prefix_mask = (
                player_region_mask[:, :1]
                if reference_latents
                else torch.zeros_like(player_region_mask[:, :1])
            )
            diagnostic_mask = torch.cat(
                (prefix_mask, player_region_mask[:, reference_latents:]), dim=1,
            )
            diagnostics.update(player_flow_diagnostics(error, diagnostic_mask, time[:, reference_latents:]))
    # Explicit experiment: reuse the exact detached ROI used by appearance injection.
    roi_multiplier = float(getattr(base, "roi_flow_multiplier", 1.0))
    roi = getattr(base, "_flow_injection_roi", None)
    unweighted_flow = error.mean()
    roi_flow = error.new_zeros(())
    if roi_multiplier > 1:
        if roi is None or roi.shape != (len(clean), clean.shape[1], 1, *clean.shape[-2:]):
            raise ValueError("injection ROI missing or not aligned to latent flow")
        roi = roi[:, reference_latents:].float().detach().clamp(0, 1)
        roi_flow = (error * roi).sum() / (roi.sum().clamp_min(1) * clean.shape[2])
    if roi_multiplier > 1 and base.training:
        if region_weight is not None:
            raise ValueError("ROI flow experiment cannot combine other region weights")
        weight = 1 + (roi_multiplier - 1) * roi
        loss = ((error * weight).flatten(1).sum(1) /
                (weight.flatten(1).sum(1) * clean.shape[2]).clamp_min(1)).mean()
    elif region_weight is None:
        loss = unweighted_flow
    else:
        weight = region_weight[:, reference_latents:].float()
        expected = (len(clean), clean.shape[1] - reference_latents, 1, *clean.shape[-2:])
        if weight.shape != expected:
            raise ValueError("region_weight must be [B,T,1,H,W]")
        loss = (error * weight).sum() / (weight.sum().clamp_min(1) * clean.shape[2])
    if diagnostics is not None:
        diagnostics["unweighted_flow_loss"] = unweighted_flow.detach()
        diagnostics["injection_roi_flow_loss"] = roi_flow.detach()
    if return_clean_prediction:
        # For v = epsilon - x_0 and x_t = (1-t)x_0 + t epsilon,
        # x_0 = x_t - t*v. Pixel supervision decodes only selected frames.
        return loss, (1 - sigma_min) * noisy - noise_scale * prediction
    return loss


def renderer_history_self_condition_loss(
    model,
    clean,
    conditions,
    *,
    region_weight=None,
    denoising_steps=4,
    generator=None,
):
    """Train one block after a detached model-generated history block.

    Long rollout inference repeatedly conditions on the model's own previous
    output, while the ordinary full-window objective always exposes clean/noisy
    ground-truth history.  This dedicated objective closes that mismatch for
    the first block transition without changing inference-time conditions.
    """
    base = model.module if hasattr(model, "module") else model
    if getattr(base.cfg, "bidirectional_attention", False):
        raise ValueError("block history self-conditioning is incompatible with bidirectional mode")
    if clean.ndim != 5 or clean.shape[1] > base.cfg.context_frames:
        raise ValueError("clean must be [B,T,C,H,W] within the configured context")
    block_frames = base.cfg.block_frames
    required_frames = 1 + 2 * block_frames
    if clean.shape[1] < required_frames:
        raise ValueError(
            "history self-conditioning requires a prefix and two complete output blocks"
        )
    if denoising_steps < 1:
        raise ValueError("denoising_steps must be positive")
    if base.core.kv_caches is not None:
        raise RuntimeError("clear rollout caches before training")

    c = dict(conditions)
    c.pop("condition_mask", None)
    prefix_mask = torch.arange(required_frames, device=clean.device)[None] == 0
    c["action_prefix_mask"] = prefix_mask.expand(len(clean), -1)
    if getattr(base.core, "use_condition_mask", False):
        c["condition_mask"] = c["action_prefix_mask"]

    prefix = clean[:, :1]
    history = torch.randn(
        (len(clean), block_frames, *clean.shape[2:]),
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    history_conditions = slice_conditions(c, 0, 1 + block_frames)
    # The backbone's checkpoint branches key off training mode rather than
    # grad mode. Entering torch.checkpoint under no_grad leaves checkpoint
    # frames that conflict with the following differentiable forward, so
    # disable it only while synthesizing the detached history.
    checkpointing = base.core.gradient_checkpointing
    base.core.gradient_checkpointing = False
    try:
        with torch.no_grad():
            for index in range(denoising_steps):
                tau = 1.0 - index / denoising_steps
                history_time = torch.zeros(
                    len(clean), 1 + block_frames, device=clean.device,
                )
                history_time[:, 1:] = tau
                model_input = torch.cat((prefix, history), dim=1)
                velocity = model(model_input, history_time, history_conditions)[:, 1:]
                history = history - velocity * (1.0 / denoising_steps)
    finally:
        base.core.gradient_checkpointing = checkpointing
    history = history.detach()

    target_start = 1 + block_frames
    target_end = required_frames
    target_clean = clean[:, target_start:target_end]
    target_noise = torch.randn(
        target_clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    target_time = torch.rand(
        len(clean), 1, device=clean.device, generator=generator,
    )
    tau = target_time[..., None, None, None]
    target_noisy = (1 - tau) * target_clean + tau * target_noise
    model_input = torch.cat((prefix, history, target_noisy), dim=1)
    time = torch.zeros(len(clean), required_frames, device=clean.device)
    time[:, target_start:target_end] = target_time
    # The production DiT contains nested checkpointed adapters whose
    # recomputation metadata is only stable for the normal full-context path.
    # This branch is deliberately only 17 frames, so retaining its activations
    # is affordable and avoids a checkpoint backward mismatch.
    checkpointing = base.core.gradient_checkpointing
    base.core.gradient_checkpointing = False
    try:
        prediction = model(model_input, time, slice_conditions(c, 0, required_frames))
    finally:
        base.core.gradient_checkpointing = checkpointing
    error = (
        prediction[:, target_start:target_end].float()
        - (target_noise - target_clean).float()
    ).square()

    if region_weight is None:
        return error.mean()
    expected = (len(clean), clean.shape[1], 1, *clean.shape[-2:])
    if region_weight.shape != expected:
        raise ValueError("region_weight must be [B,T,1,H,W]")
    weight = region_weight[:, target_start:target_end].float()
    return (error * weight).sum() / (weight.sum().clamp_min(1) * clean.shape[2])


def player_flow_diagnostics(error, player_region_mask, time):
    """Independently normalize player flow per valid sample, excluding prefix.

    Mask area is preserved at latent resolution. Noise-bin sums/counts are
    additive; an absent bin is not a measured zero loss.
    """
    batch, future, channels, height, width = error.shape
    if player_region_mask.shape[:3] != (batch, future + 1, 1):
        raise ValueError("player mask must contain the clean prefix and future frames")
    mask = F.interpolate(player_region_mask[:, 1:].float().flatten(0, 1),
                         size=(height, width), mode="area").unflatten(0, (batch, future))
    area = mask.sum((2, 3, 4))
    weighted = (error.float() * mask).sum((2, 3, 4)) / channels
    valid = area.sum(1) > 0
    per_sample = weighted.sum(1) / area.sum(1).clamp_min(1e-8)
    result = {"player_flow_loss": (per_sample * valid).sum() / valid.sum().clamp_min(1)}
    per_frame = weighted.detach() / area.clamp_min(1e-8)
    for name, lo, hi in (("low", 0, 1/3), ("mid", 1/3, 2/3), ("high", 2/3, 1.01)):
        selected = (time >= lo) & (time < hi) & (area > 0)
        result[f"player_flow_{name}_sum"] = (per_frame * selected).sum()
        result[f"player_flow_{name}_count"] = selected.sum()
    return result


def renderer_counterfactual_player_loss(
    clean_prediction, clean, player_region_mask, group_size,
):
    """Match appearance-induced latent differences inside other-player ROIs."""
    if group_size < 2:
        return clean_prediction.sum() * 0
    batch, frames = clean.shape[:2]
    if batch % group_size or player_region_mask.shape[:2] != (batch, frames):
        raise ValueError("counterfactual predictions/masks must form complete groups")
    groups = batch // group_size
    predicted = clean_prediction.reshape(groups, group_size, *clean_prediction.shape[1:])
    target = clean.reshape(groups, group_size, *clean.shape[1:])
    mask = player_region_mask.float().flatten(0, 1)
    mask = F.interpolate(mask, size=clean.shape[-2:], mode="area")
    mask = mask.unflatten(0, (batch, frames)).reshape(
        groups, group_size, frames, 1, *clean.shape[-2:]
    )
    predicted_difference = predicted[:, 1:, 1:] - predicted[:, :1, 1:]
    target_difference = target[:, 1:, 1:] - target[:, :1, 1:]
    roi = torch.maximum(mask[:, 1:, 1:], mask[:, :1, 1:])
    error = (predicted_difference.float() - target_difference.float()).square()
    return (error * roi).sum() / (roi.sum().clamp_min(1) * clean.shape[2])


def _masked_mean(value, mask):
    channels = value.shape[-3]
    return (value.float() * mask.float()).sum() / (mask.sum().clamp_min(1) * channels)


def _masked_edge_l1(prediction, target, mask):
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (
        target[..., 1:] - target[..., :-1]
    )
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    mask_x = torch.maximum(mask[..., 1:], mask[..., :-1])
    mask_y = torch.maximum(mask[..., 1:, :], mask[..., :-1, :])
    return (_masked_mean(dx.abs(), mask_x) + _masked_mean(dy.abs(), mask_y)) / 2


def _crop_player_rois(images, masks, *, output_size=(128, 64), padding=0.25):
    """Crop target-mask boxes to a fixed resolution without discarding context.

    The mask supplies non-differentiable box metadata.  Image crop and resize
    operations remain differentiable, so small distant players receive a
    structure loss at the same canonical resolution as nearby players.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [N,3,H,W]")
    if masks.shape != (len(images), 1, *images.shape[-2:]):
        raise ValueError("masks must be [N,1,H,W] and align with images")
    out_h, out_w = map(int, output_size)
    if out_h < 1 or out_w < 1 or padding < 0:
        raise ValueError("invalid player ROI output size or padding")

    crops, mask_crops, valid = [], [], []
    for image, mask in zip(images, masks):
        points = (mask[0] > 0.5).nonzero(as_tuple=False)
        if not len(points):
            crops.append(image.new_zeros(3, out_h, out_w))
            mask_crops.append(mask.new_zeros(1, out_h, out_w))
            valid.append(False)
            continue
        y0, x0 = points.amin(0).tolist()
        y1, x1 = (points.amax(0) + 1).tolist()
        pad_y = max(1, round((y1 - y0) * padding))
        pad_x = max(1, round((x1 - x0) * padding))
        y0, y1 = max(0, y0 - pad_y), min(image.shape[-2], y1 + pad_y)
        x0, x1 = max(0, x0 - pad_x), min(image.shape[-1], x1 + pad_x)
        crop = image[:, y0:y1, x0:x1]
        mask_crop = mask[:, y0:y1, x0:x1].float()
        scale = min(out_h / crop.shape[-2], out_w / crop.shape[-1])
        resized_h = max(1, min(out_h, round(crop.shape[-2] * scale)))
        resized_w = max(1, min(out_w, round(crop.shape[-1] * scale)))
        crop = F.interpolate(
            crop[None], size=(resized_h, resized_w), mode="bilinear", align_corners=False
        )[0]
        mask_crop = F.interpolate(
            mask_crop[None], size=(resized_h, resized_w), mode="nearest"
        )[0]
        top = (out_h - resized_h) // 2
        bottom = out_h - resized_h - top
        left = (out_w - resized_w) // 2
        right = out_w - resized_w - left
        crops.append(F.pad(crop, (left, right, top, bottom)))
        mask_crops.append(F.pad(mask_crop, (left, right, top, bottom)))
        valid.append(True)
    return (
        torch.stack(crops),
        torch.stack(mask_crops),
        torch.tensor(valid, device=images.device),
    )


def select_renderer_pixel_frames(
    pixel_region_mask,
    *,
    frames_per_sample,
    player_region_mask=None,
    selection_mode="mixed",
    hp=None,
    target_agent=None,
    generator=None,
    full_health=20.0,
):
    """Select an entity-rich frame and, when possible, a damaged-HP frame."""
    if pixel_region_mask.ndim != 5 or pixel_region_mask.shape[2] != 1:
        raise ValueError("pixel_region_mask must be [B,T,1,H,W]")
    batch, frames = pixel_region_mask.shape[:2]
    if not 1 <= frames_per_sample < frames:
        raise ValueError("frames_per_sample must be within the future-frame count")
    if selection_mode not in {"mixed", "player", "player_unique", "player_stratified"}:
        raise ValueError(
            "selection_mode must be 'mixed', 'player', 'player_unique' or "
            "'player_stratified'"
        )
    if (
        selection_mode in {"player", "player_unique", "player_stratified"}
        and player_region_mask is None
    ):
        raise ValueError("player frame selection requires player_region_mask")

    coverage = pixel_region_mask[:, 1:].flatten(2).sum(-1).float()
    entity_frame = torch.multinomial(
        coverage + 1e-3, 1, replacement=True, generator=generator
    ) + 1
    player_frame = entity_frame
    if player_region_mask is not None:
        if player_region_mask.shape != pixel_region_mask.shape:
            raise ValueError("player_region_mask must match pixel_region_mask")
        player_coverage = player_region_mask[:, 1:].flatten(2).sum(-1).float()
        sampled_player = torch.multinomial(
            player_coverage + 1e-3, 1, replacement=True, generator=generator
        ) + 1
        player_frame = torch.where(
            player_coverage.sum(1, keepdim=True) > 0, sampled_player, entity_frame
        )
    random_frame = torch.randint(
        1, frames, (batch, 1), device=pixel_region_mask.device, generator=generator
    )
    health_frame = random_frame
    has_damage = torch.zeros(batch, dtype=torch.bool, device=pixel_region_mask.device)
    target_hp = None
    if hp is not None or target_agent is not None:
        if hp is None or target_agent is None:
            raise ValueError("hp and target_agent must be supplied together")
        if hp.ndim != 3 or hp.shape[:2] != (batch, frames):
            raise ValueError("hp must be [B,T,A] and align with the masks")
        if target_agent.shape != (batch,):
            raise ValueError("target_agent must be [B]")
        gather = target_agent.long()[:, None, None].expand(-1, frames, 1)
        target_hp = hp.gather(2, gather).squeeze(2)
        future_hp = target_hp[:, 1:]
        has_damage = future_hp.min(1).values < full_health - 1e-3
        minimum = future_hp.min(1, keepdim=True).values
        minimum_frames = (future_hp <= minimum + 1e-3).float()
        sampled_minimum = torch.multinomial(
            minimum_frames, 1, replacement=True, generator=generator
        ) + 1
        health_frame = torch.where(has_damage[:, None], sampled_minimum, random_frame)

    if selection_mode == "player_stratified":
        # First draw one visible-player frame from each temporal stratum. Empty
        # strata are filled afterwards from unused visible frames anywhere in
        # the clip, then (only if necessary) from other unused future frames.
        # This preserves temporal coverage without fabricating player masks.
        future = frames - 1
        remaining = torch.ones_like(player_coverage)
        indices = torch.full(
            (batch, frames_per_sample), -1,
            dtype=torch.long, device=pixel_region_mask.device,
        )
        for slot in range(frames_per_sample):
            start = slot * future // frames_per_sample
            end = (slot + 1) * future // frames_per_sample
            weights = player_coverage[:, start:end]
            visible = weights.sum(1) > 0
            safe_weights = torch.where(
                visible[:, None], weights, torch.ones_like(weights)
            )
            local = torch.multinomial(safe_weights, 1, generator=generator).squeeze(1)
            chosen = local + start
            indices[visible, slot] = chosen[visible] + 1
            remaining[visible, chosen[visible]] = 0
        for slot in range(frames_per_sample):
            missing = indices[:, slot] < 0
            if not missing.any():
                continue
            weights = player_coverage * remaining
            weights = torch.where(weights.sum(1, keepdim=True) > 0, weights, remaining)
            # Rows whose slot was already filled still pass through the batched
            # multinomial call; give those ignored rows a valid distribution.
            weights = torch.where(missing[:, None], weights, torch.ones_like(weights))
            chosen = torch.multinomial(weights, 1, generator=generator).squeeze(1)
            indices[missing, slot] = chosen[missing] + 1
            remaining[missing, chosen[missing]] = 0
    elif selection_mode == "player_unique":
        # Sample visible-player frames by area, without replacement. If fewer
        # than K exist, fill from other unused future frames; their zero player
        # masks provide no fabricated supervision. The prefix is never drawn.
        remaining = torch.ones_like(player_coverage)
        selected = []
        for _ in range(frames_per_sample):
            weights = player_coverage * remaining
            weights = torch.where(weights.sum(1, keepdim=True) > 0, weights, remaining)
            index = torch.multinomial(weights, 1, generator=generator)
            selected.append(index + 1)
            remaining.scatter_(1, index, 0)
        indices = torch.cat(selected, dim=1)
    elif selection_mode == "player":
        extras = torch.randint(
            1,
            frames,
            (batch, frames_per_sample - 1),
            device=pixel_region_mask.device,
            generator=generator,
        )
        indices = torch.cat((player_frame, extras), dim=1)
    elif frames_per_sample == 1:
        indices = torch.where(has_damage[:, None], health_frame, entity_frame)
    elif frames_per_sample >= 3 and player_region_mask is not None:
        extras = torch.randint(
            1,
            frames,
            (batch, frames_per_sample - 3),
            device=pixel_region_mask.device,
            generator=generator,
        )
        indices = torch.cat((entity_frame, health_frame, player_frame, extras), dim=1)
    else:
        extras = torch.randint(
            1,
            frames,
            (batch, frames_per_sample - 2),
            device=pixel_region_mask.device,
            generator=generator,
        )
        indices = torch.cat((entity_frame, health_frame, extras), dim=1)
    return indices, target_hp


def renderer_pixel_losses(
    codec,
    clean_prediction,
    target_rgb,
    pixel_region_mask,
    *,
    player_region_mask=None,
    frames_per_sample=1,
    frame_selection="mixed",
    generator=None,
    hp=None,
    target_agent=None,
    damaged_health_upweight=4.0,
    player_ring_radius=0,
    player_ring_weight=0.25,
    player_roi_size=None,
    player_roi_padding=0.25,
    health_box=(190 / 640, 300 / 360, 314 / 640, 322 / 360),
    profile_timings=None,
    diagnostics=None,
    noise_time=None,
):
    """Full-resolution entity and Minecraft heart-bar losses.

    Only sampled future frames are decoded, which keeps the frozen VAE
    backward graph small. ``health_box`` is normalized ``(x0,y0,x1,y1)``.
    """
    if clean_prediction.ndim != 5 or target_rgb.ndim != 5:
        raise ValueError("predicted latents and target RGB must be [B,T,C,H,W]")
    if clean_prediction.shape[:2] != target_rgb.shape[:2]:
        raise ValueError("predicted latents and target RGB must align in B,T")
    expected_mask = (*target_rgb.shape[:2], 1, *target_rgb.shape[-2:])
    if pixel_region_mask.shape != expected_mask:
        raise ValueError("pixel_region_mask must be [B,T,1,H,W] at RGB resolution")
    if player_region_mask is not None and player_region_mask.shape != expected_mask:
        raise ValueError("player_region_mask must match the RGB-resolution entity mask")
    future_frames = clean_prediction.shape[1] - 1
    if not 1 <= frames_per_sample <= future_frames:
        raise ValueError("frames_per_sample must be within the future-frame count")
    if len(health_box) != 4 or not (
        0 <= health_box[0] < health_box[2] <= 1
        and 0 <= health_box[1] < health_box[3] <= 1
    ):
        raise ValueError("health_box must be normalized (x0,y0,x1,y1)")

    if damaged_health_upweight < 0:
        raise ValueError("damaged_health_upweight must be nonnegative")
    if player_ring_radius < 0:
        raise ValueError("player_ring_radius must be nonnegative")
    if not 0 <= player_ring_weight <= 1:
        raise ValueError("player_ring_weight must be between zero and one")
    if player_roi_size is not None:
        if len(player_roi_size) != 2 or min(map(int, player_roi_size)) < 1:
            raise ValueError("player_roi_size must contain two positive integers")
        if player_roi_padding < 0:
            raise ValueError("player_roi_padding must be nonnegative")
    batch = clean_prediction.shape[0]
    frame_indices, target_hp = select_renderer_pixel_frames(
        pixel_region_mask,
        frames_per_sample=frames_per_sample,
        player_region_mask=player_region_mask,
        selection_mode=frame_selection,
        hp=hp,
        target_agent=target_agent,
        generator=generator,
    )
    batch_indices = torch.arange(batch, device=clean_prediction.device)[:, None]
    selected_latent = clean_prediction[batch_indices, frame_indices].reshape(
        batch * frames_per_sample, 1, *clean_prediction.shape[2:]
    )
    selected_target = target_rgb[batch_indices, frame_indices].reshape(
        batch * frames_per_sample, *target_rgb.shape[2:]
    )
    selected_mask = pixel_region_mask[batch_indices, frame_indices].reshape(
        batch * frames_per_sample, 1, *target_rgb.shape[-2:]
    )
    selected_player_mask = (
        player_region_mask[batch_indices, frame_indices].reshape(
            batch * frames_per_sample, 1, *target_rgb.shape[-2:]
        )
        if player_region_mask is not None
        else torch.zeros_like(selected_mask)
    )
    decoded = _profile_cuda_call(
        profile_timings,
        "vae_decode_for_loss",
        selected_latent,
        lambda: codec.decode_for_loss(selected_latent, chunk_size=1)[:, 0],
    )

    entity_l1 = _masked_mean((decoded - selected_target).abs(), selected_mask)
    # A five-pixel dilation lets boundary gradients cover anti-aliased edges,
    # while the RGB L1 above remains confined to the exact instance mask.
    edge_mask = F.max_pool2d(selected_mask.float(), kernel_size=5, stride=1, padding=2)
    entity_edge = _masked_edge_l1(decoded, selected_target, edge_mask)
    player_weight = selected_player_mask.float()
    if player_ring_radius:
        kernel = 2 * int(player_ring_radius) + 1
        dilated = F.max_pool2d(
            player_weight, kernel_size=kernel, stride=1, padding=int(player_ring_radius)
        )
        ring = (dilated - player_weight).clamp_min(0)
        player_weight = player_weight + float(player_ring_weight) * ring
    player_error = (decoded - selected_target).abs().float()
    player_area = player_weight.flatten(1).sum(1)
    player_per_frame = (player_error * player_weight).flatten(1).sum(1)
    player_per_frame = player_per_frame / (3 * player_area.clamp_min(1))
    player_valid = player_area > 0
    player_l1 = (
        player_per_frame[player_valid].mean()
        if player_valid.any()
        else clean_prediction.sum() * 0
    )
    player_edge_mask = F.max_pool2d(
        selected_player_mask.float(), kernel_size=5, stride=1, padding=2
    )
    player_edge = _masked_edge_l1(decoded, selected_target, player_edge_mask)

    if player_roi_size is not None:
        predicted_roi, roi_mask, roi_valid = _crop_player_rois(
            decoded, selected_player_mask,
            output_size=player_roi_size, padding=player_roi_padding,
        )
        target_roi, _, target_roi_valid = _crop_player_rois(
            selected_target, selected_player_mask,
            output_size=player_roi_size, padding=player_roi_padding,
        )
        roi_valid = roi_valid & target_roi_valid
        roi_weight = roi_mask.float()
        if player_ring_radius:
            kernel = 2 * int(player_ring_radius) + 1
            dilated = F.max_pool2d(
                roi_weight, kernel_size=kernel, stride=1,
                padding=int(player_ring_radius),
            )
            roi_weight = roi_weight + float(player_ring_weight) * (
                dilated - roi_weight
            ).clamp_min(0)
        roi_error = (predicted_roi - target_roi).abs().float()
        roi_area = roi_weight.flatten(1).sum(1)
        roi_per_frame = (roi_error * roi_weight).flatten(1).sum(1)
        roi_per_frame = roi_per_frame / (3 * roi_area.clamp_min(1))
        roi_valid = roi_valid & (roi_area > 0)
        player_roi_l1 = (
            roi_per_frame[roi_valid].mean()
            if roi_valid.any()
            else clean_prediction.sum() * 0
        )
        roi_edge_mask = F.max_pool2d(
            roi_mask.float(), kernel_size=5, stride=1, padding=2
        )
        player_roi_edge = _masked_edge_l1(
            predicted_roi[roi_valid], target_roi[roi_valid], roi_edge_mask[roi_valid]
        ) if roi_valid.any() else clean_prediction.sum() * 0
    else:
        player_roi_l1 = clean_prediction.sum() * 0
        player_roi_edge = clean_prediction.sum() * 0

    if diagnostics is not None:
        if noise_time is None or noise_time.shape != (batch, future_frames):
            raise ValueError("pixel diagnostics require future-frame noise times")
        # Diagnostics use equal weight per valid decoded frame, not pixel-area
        # weighting across the batch. Sums/counts can be merged across ranks.
        area = selected_player_mask.float().flatten(1).sum(1)
        per_frame = ((decoded.detach().float() - selected_target.float()).abs()
                     * selected_player_mask.float()).flatten(1).sum(1)
        per_frame = per_frame / (3 * area.clamp_min(1e-8))
        selected_time = noise_time[batch_indices, frame_indices - 1].flatten()
        valid = area > 0
        diagnostics["player_pixel_valid_count"] = valid.sum()
        diagnostics["player_pixel_selected_count"] = area.new_tensor(area.numel())
        for name, lo, hi in (("low", 0, 1/3), ("mid", 1/3, 2/3), ("high", 2/3, 1.01)):
            included = valid & (selected_time >= lo) & (selected_time < hi)
            diagnostics[f"player_pixel_{name}_sum"] = (per_frame * included).sum()
            diagnostics[f"player_pixel_{name}_count"] = included.sum()

    height, width = target_rgb.shape[-2:]
    x0, y0, x1, y1 = health_box
    left, right = int(round(x0 * width)), int(round(x1 * width))
    top, bottom = int(round(y0 * height)), int(round(y1 * height))
    health_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
    health_mask[..., top:bottom, left:right] = True
    if target_hp is not None:
        selected_hp = target_hp[batch_indices, frame_indices].reshape(-1, 1, 1, 1)
        health_mask = health_mask.float() * (
            1 + damaged_health_upweight * (selected_hp < 20 - 1e-3).float()
        )
    health_l1 = _masked_mean((decoded - selected_target).abs(), health_mask)
    return {
        "entity_pixel_l1": entity_l1,
        "entity_pixel_edge": entity_edge,
        "player_pixel_l1": player_l1,
        "player_pixel_edge": player_edge,
        "player_roi_l1": player_roi_l1,
        "player_roi_edge": player_roi_edge,
        "health_pixel_l1": health_l1,
    }


def renderer_player_identity_loss(
    codec,
    identity_encoder,
    clean_prediction,
    player_reference,
    player_identity_mask,
    player_identity_frame,
    player_identity_slot,
    player_identity_valid,
    *,
    player_appearance_valid=None,
    margin=0.2,
    negative_weight=0.5,
):
    """Match a generated resident crop to its canonical four-view identity.

    The identity encoder is frozen. Gradients pass through its crop tower into
    the predicted clean latent, while the canonical reference embedding is a
    fixed target. A different resident provides an explicit negative whenever
    one is available.
    """
    batch = len(clean_prediction)
    expected_mask = (batch, 1, *player_identity_mask.shape[-2:])
    if player_identity_mask.shape != expected_mask:
        raise ValueError("player_identity_mask must be [B,1,H,W]")
    for value, name in (
        (player_identity_frame, "player_identity_frame"),
        (player_identity_slot, "player_identity_slot"),
        (player_identity_valid, "player_identity_valid"),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} must be [B]")
    if (
        player_reference.ndim != 6
        or player_reference.shape[0] != batch
        or player_reference.shape[2:4] != (4, 4)
    ):
        raise ValueError("player_reference must be [B,A,4,4,H,W]")
    if margin < 0 or negative_weight < 0:
        raise ValueError("identity margin and negative weight must be nonnegative")
    indices = torch.arange(batch, device=clean_prediction.device)
    frames = player_identity_frame.long().clamp(0, clean_prediction.shape[1] - 1)
    slots = player_identity_slot.long().clamp(0, player_reference.shape[1] - 1)
    selected_latent = clean_prediction[indices, frames, None]
    decoded = codec.decode_for_loss(selected_latent, chunk_size=1)[:, 0]
    crop, crop_valid = crop_masked_players(
        decoded, player_identity_mask, output_size=identity_encoder.crop_size
    )
    valid = player_identity_valid.bool() & crop_valid
    selected_reference = player_reference[indices, slots]
    with torch.no_grad():
        reference_embedding = identity_encoder.encode_reference(selected_reference)
    crop_embedding = identity_encoder.encode_crop(crop)
    correct_similarity = (crop_embedding * reference_embedding).sum(-1)

    agent_valid = (
        player_appearance_valid.bool().any(-1)
        if player_appearance_valid is not None
        else torch.ones(player_reference.shape[:2], dtype=torch.bool, device=slots.device)
    )
    wrong_slots = slots.clone()
    has_negative = torch.zeros(batch, dtype=torch.bool, device=slots.device)
    for row in range(batch):
        choices = torch.where(
            agent_valid[row]
            & (torch.arange(player_reference.shape[1], device=slots.device) != slots[row])
        )[0]
        if len(choices):
            wrong_slots[row] = choices[0]
            has_negative[row] = True
    with torch.no_grad():
        wrong_embedding = identity_encoder.encode_reference(
            player_reference[indices, wrong_slots]
        )
    wrong_similarity = (crop_embedding * wrong_embedding).sum(-1)
    positive = 1 - correct_similarity
    ranking = F.relu(margin + wrong_similarity - correct_similarity)
    per_sample = positive + negative_weight * ranking * has_negative.to(ranking.dtype)
    if valid.any():
        loss = per_sample[valid].mean()
        similarity = correct_similarity[valid].mean()
        ranking_accuracy = (
            correct_similarity[valid] > wrong_similarity[valid]
        ).float().mean()
    else:
        loss = clean_prediction.sum() * 0
        similarity = loss.detach()
        ranking_accuracy = loss.detach()
    return {
        "player_identity_loss": loss,
        "player_identity_similarity": similarity,
        "player_identity_ranking_accuracy": ranking_accuracy,
    }


def renderer_training_losses(
    model,
    codec,
    clean,
    conditions,
    target_rgb,
    pixel_region_mask,
    *,
    player_region_mask=None,
    region_weight=None,
    frames_per_sample=2,
    pixel_frame_selection="mixed",
    entity_pixel_l1_weight=0.5,
    entity_pixel_edge_weight=0.2,
    player_pixel_l1_weight=0.0,
    player_pixel_edge_weight=0.0,
    player_roi_l1_weight=0.0,
    player_roi_edge_weight=0.0,
    player_roi_size=(128, 64),
    player_roi_padding=0.25,
    health_pixel_l1_weight=1.0,
    damaged_health_upweight=4.0,
    player_ring_radius=0,
    player_ring_weight=0.25,
    identity_encoder=None,
    player_identity_mask=None,
    player_identity_frame=None,
    player_identity_slot=None,
    player_identity_valid=None,
    player_identity_loss_weight=0.0,
    player_identity_margin=0.2,
    player_identity_negative_weight=0.5,
    counterfactual_group_size=1,
    counterfactual_player_difference_weight=0.0,
    counterfactual_pure_noise=True,
    ordinary_pure_noise_probability=0.0,
    flow_loss_weight=1.0,
    reference_latents=1,
    player_flow_loss_weight=0.0,
    flow_diagnostics=None,
    generator=None,
    profile_timings=None,
    time_distribution="uniform",
    sigma_min=0.0,
    low_noise_mix_probability=0.0,
    low_noise_max=0.3,
):
    """Combine latent flow matching with sparse full-resolution supervision."""
    if region_weight is not None and region_weight.shape[1] != clean.shape[1]:
        region_weight = codec.align_pixel_frames(region_weight)
    if region_weight is not None and region_weight.shape[-2:] != clean.shape[-2:]:
        batch, frames = region_weight.shape[:2]
        region_weight = F.interpolate(
            region_weight.float().flatten(0, 1),
            size=clean.shape[-2:],
            mode="area",
        ).unflatten(0, (batch, frames)).to(region_weight.dtype)
    if (
        player_region_mask is not None
        and player_region_mask.shape[1] != clean.shape[1]
    ):
        player_region_mask = codec.align_pixel_frames(player_region_mask)
    weights = {
        "entity_pixel_l1": float(entity_pixel_l1_weight),
        "entity_pixel_edge": float(entity_pixel_edge_weight),
        "player_pixel_l1": float(player_pixel_l1_weight),
        "player_pixel_edge": float(player_pixel_edge_weight),
        "player_roi_l1": float(player_roi_l1_weight),
        "player_roi_edge": float(player_roi_edge_weight),
        "health_pixel_l1": float(health_pixel_l1_weight),
    }
    if min(flow_loss_weight, player_flow_loss_weight, *weights.values(), player_identity_loss_weight,
           counterfactual_player_difference_weight) < 0:
        raise ValueError("flow and pixel loss weights must be nonnegative")
    use_pixels = any(weight > 0 for weight in weights.values())
    use_identity = player_identity_loss_weight > 0
    use_counterfactual = counterfactual_player_difference_weight > 0
    if reference_latents > 1 and (use_pixels or use_identity or use_counterfactual or player_flow_loss_weight):
        raise ValueError("reference curriculum supports pure flow loss only")
    if player_flow_loss_weight > 0 and player_region_mask is None:
        raise ValueError("player flow loss requires player_region_mask")
    diagnostics = flow_diagnostics if flow_diagnostics is not None else {}
    if use_counterfactual and player_region_mask is None:
        raise ValueError("counterfactual appearance loss requires player_region_mask")
    if use_identity and identity_encoder is None:
        raise ValueError("player identity loss requires a frozen identity encoder")
    result = renderer_flow_loss(
        model,
        clean,
        conditions,
        region_weight=region_weight,
        generator=generator,
        return_clean_prediction=use_pixels or use_identity or use_counterfactual,
        counterfactual_group_size=counterfactual_group_size,
        counterfactual_pure_noise=counterfactual_pure_noise,
        ordinary_pure_noise_probability=ordinary_pure_noise_probability,
        profile_timings=profile_timings,
        reference_latents=reference_latents,
        player_region_mask=player_region_mask,
        diagnostics=diagnostics if player_flow_loss_weight > 0 or flow_diagnostics is not None else None,
        time_distribution=time_distribution,
        sigma_min=sigma_min,
        low_noise_mix_probability=low_noise_mix_probability,
        low_noise_max=low_noise_max,
    )
    if use_pixels or use_identity or use_counterfactual:
        flow_loss, clean_prediction = result
    else:
        flow_loss = result
        clean_prediction = None
    if use_pixels:
        pixels = renderer_pixel_losses(
            codec,
            clean_prediction,
            target_rgb,
            pixel_region_mask,
            player_region_mask=player_region_mask,
            frames_per_sample=frames_per_sample,
            frame_selection=pixel_frame_selection,
            generator=generator,
            hp=conditions.get("hp"),
            target_agent=conditions.get("target_agent"),
            damaged_health_upweight=damaged_health_upweight,
            player_ring_radius=player_ring_radius,
            player_ring_weight=player_ring_weight,
            player_roi_size=(
                player_roi_size
                if player_roi_l1_weight > 0 or player_roi_edge_weight > 0
                else None
            ),
            player_roi_padding=player_roi_padding,
            profile_timings=profile_timings,
            diagnostics=flow_diagnostics,
            noise_time=diagnostics.get("noise_time_per_frame"),
        )
    else:
        pixels = {name: flow_loss.new_zeros(()) for name in weights}
    if use_identity:
        required = {
            "player_identity_mask": player_identity_mask,
            "player_identity_frame": player_identity_frame,
            "player_identity_slot": player_identity_slot,
            "player_identity_valid": player_identity_valid,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"player identity supervision is missing {missing}")
        identity = renderer_player_identity_loss(
            codec,
            identity_encoder,
            clean_prediction,
            conditions["player_reference"],
            player_identity_mask,
            player_identity_frame,
            player_identity_slot,
            player_identity_valid,
            player_appearance_valid=conditions.get("player_appearance_valid"),
            margin=player_identity_margin,
            negative_weight=player_identity_negative_weight,
        )
    else:
        zero = flow_loss.new_zeros(())
        identity = {
            "player_identity_loss": zero,
            "player_identity_similarity": zero,
            "player_identity_ranking_accuracy": zero,
        }
    counterfactual = (
        renderer_counterfactual_player_loss(
            clean_prediction, clean, player_region_mask, counterfactual_group_size
        )
        if use_counterfactual
        else flow_loss.new_zeros(())
    )
    auxiliary = sum(weights[name] * value for name, value in pixels.items())
    auxiliary = auxiliary + player_identity_loss_weight * identity["player_identity_loss"]
    auxiliary = auxiliary + counterfactual_player_difference_weight * counterfactual
    player_flow = diagnostics.get("player_flow_loss", flow_loss.new_zeros(()))
    return {
        "total_loss": float(flow_loss_weight) * flow_loss + player_flow_loss_weight * player_flow + auxiliary,
        "flow_loss": flow_loss,
        "unweighted_flow_loss": diagnostics.get("unweighted_flow_loss", flow_loss.detach()),
        "injection_roi_flow_loss": diagnostics.get("injection_roi_flow_loss", flow_loss.new_zeros(())),
        "player_flow_loss": player_flow,
        "auxiliary_loss": auxiliary,
        **pixels,
        **identity,
        "counterfactual_player_difference_loss": counterfactual,
    }


@torch.no_grad()
def renderer_bidirectional_sample(
    model, first_latent, noise, conditions, *, denoising_steps=20,
    time_shift=1.0,
):
    """Jointly generate the scored clip from noise with an optional GT prefix.

    No future GT argument, no history cache, no blockwise commits. Geometry,
    actions and appearance ID for the whole planned clip must be available.
    Returns future frames only, matching RendererRollout.generate's convention.
    """
    base = model.module if hasattr(model, "module") else model
    if not getattr(base.cfg, "bidirectional_attention", False):
        raise ValueError("joint sampling requires a bidirectional checkpoint")
    if denoising_steps < 1:
        raise ValueError("denoising_steps must be positive")
    if first_latent.ndim != 5 or noise.ndim != 5:
        raise ValueError("expected prefix/noise tensors shaped [B,T,C,H,W]")
    if (first_latent.shape[0] != noise.shape[0]
            or first_latent.shape[2:] != noise.shape[2:]
            or noise.shape[1] + first_latent.shape[1] != base.cfg.context_frames):
        raise ValueError("prefix/noise shapes must match the full configured context")
    if base.core.kv_caches is not None:
        raise RuntimeError("clear rollout caches before joint sampling")
    cond = dict(conditions)
    cond.pop("condition_mask", None)
    prefix = torch.arange(base.cfg.context_frames, device=noise.device)[None] == 0
    cond["action_prefix_mask"] = prefix.expand(len(noise), -1)
    if base.core.use_condition_mask:
        cond["condition_mask"] = cond["action_prefix_mask"]
    # Encode invariant side conditions once, as in the causal sampler.
    encoded = base.encode_conditions(cond)
    future = noise.clone()
    shift = float(time_shift)
    if not 0 < shift < float("inf"):
        raise ValueError("time shift must be finite and positive")
    grid = torch.linspace(1, 0, denoising_steps + 1, device=noise.device)
    grid = shift * grid / (1 + (shift - 1) * grid)
    for step in range(denoising_steps):
        time = grid[step].expand(len(noise), base.cfg.context_frames).clone()
        time[:, :first_latent.shape[1]] = 0
        x = torch.cat((first_latent, future), dim=1)
        velocity = base.core(x, time, encoded, global_start_idx=0, cache_write=False)
        future = future + (grid[step + 1] - grid[step]) * velocity[:, first_latent.shape[1]:]
    return future


class RendererRollout:
    """Persistent per-view history; each call adds exactly eight completed frames."""

    def __new__(cls, model, *, denoising_steps=20, time_shift=1.0):
        base = model.module if hasattr(model, "module") else model
        if cls is RendererRollout and getattr(base.cfg, "persist_protocol", False):
            from .persist_renderer import PersistRendererRollout
            return PersistRendererRollout(model, denoising_steps=denoising_steps, time_shift=time_shift)
        return super().__new__(cls)

    def __init__(self, model, *, denoising_steps=20, time_shift=1.0):
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")
        if not 0 < time_shift < float("inf"):
            raise ValueError("time_shift must be finite and positive")
        self.model, self.denoising_steps = model, denoising_steps
        self.time_shift = float(time_shift)
        base = model.module if hasattr(model, "module") else model
        if getattr(base.cfg, "bidirectional_attention", False):
            raise ValueError("bidirectional checkpoints require joint full-clip sampling, not KV rollout")
        self.block_frames = base.cfg.block_frames
        self.next_frame = None
        self.last_policy_features = None
        self.last_policy_layers = None

    @torch.no_grad()
    def start(
        self, first_latent, conditions, *, global_start_idx=0,
        temporal_aligned=False,
    ):
        if first_latent.shape[1] != 1:
            raise ValueError("initialization requires exactly one completed observation")
        self.model.eval()
        self.model.init_kv_cache(len(first_latent), dtype=first_latent.dtype)
        self.model.set_kv_cache_start(global_start_idx)
        self.last_policy_features = self._commit(
            first_latent, conditions, global_start_idx,
            temporal_aligned=temporal_aligned,
        )
        self.next_frame = global_start_idx + 1

    def _commit(self, frames, conditions, index, *, temporal_aligned=False):
        cond = dict(conditions)
        cond.pop("condition_mask", None)
        if self.model.core.use_condition_mask:
            cond["condition_mask"] = torch.ones(
                frames.shape[:2], device=frames.device, dtype=torch.bool
            )
        layers = []
        first = max(0, self.model.core.depth - 4)
        def capture(block_index, hidden):
            if block_index >= first:
                layers.append(hidden.detach())
        _, features, candidates = self.model(
            frames, torch.zeros(frames.shape[:2], device=frames.device), cond,
            global_start_idx=index, cache_write=False, return_features=True,
            return_kv_candidates=True, block_callback=capture,
            temporal_aligned=temporal_aligned,
        )
        self.model.commit_kv_candidates(candidates, index)
        self.last_policy_layers = tuple(layers)
        return self.model.compress_policy_features(features)

    @torch.no_grad()
    def generate(self, noise, conditions, *, temporal_aligned=False):
        if self.next_frame is None:
            raise RuntimeError("start must prefill the external first observation")
        if noise.shape[1] != self.block_frames:
            raise ValueError(f"M3 rollout must generate exactly {self.block_frames} new frames")
        x = noise.clone()
        cond = dict(conditions)
        cond.pop("condition_mask", None)
        if self.model.core.use_condition_mask:
            cond["condition_mask"] = torch.zeros(
                x.shape[:2], device=x.device, dtype=torch.bool
            )
        # Geometry/appearance is invariant across the denoising iterations.
        encoded = self.model.encode_conditions(
            cond, temporal_aligned=temporal_aligned
        )
        grid = torch.linspace(1, 0, self.denoising_steps + 1, device=x.device)
        grid = self.time_shift * grid / (1 + (self.time_shift - 1) * grid)
        for step in range(self.denoising_steps):
            tau = grid[step]
            velocity = self.model.core(x, tau.expand(x.shape[:2]),
                                       encoded, global_start_idx=self.next_frame, cache_write=False)
            x = x + (grid[step + 1] - grid[step]) * velocity
        # Recompute at clean t=0; never commit an intermediate noisy candidate.
        self.last_policy_features = self._commit(
            x, conditions, self.next_frame,
            temporal_aligned=temporal_aligned,
        )
        self.next_frame += self.block_frames
        return x

    @torch.no_grad()
    def generate_64(self, noise, conditions):
        """Generate the deployment horizon as eight cached eight-frame chunks."""
        if noise.shape[1] != 64:
            raise ValueError("M3 deployment horizon requires exactly 64 noise frames")
        chunks = []
        if 64 % self.block_frames:
            raise ValueError("deployment horizon must contain complete output blocks")
        for start in range(0, 64, self.block_frames):
            chunks.append(self.generate(
                noise[:, start:start + self.block_frames],
                slice_conditions(conditions, start, start + self.block_frames)
            ))
        return torch.cat(chunks, dim=1)
