"""Frozen Pixel-VAE and Wan2.1-VAE interfaces for M3."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

import torch
from torch import nn

from .renderer_backbone.vae_pixel import ViTVae, ViTVaeArgs
from .renderer_backbone.wan_vae_impl import WanVAE_
from .latent_normalization import normalization_from_run_config, validate_latent_normalization


WAN_LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
WAN_LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


class RendererCodec(nn.Module):
    def __init__(self, weights, cfg: ViTVaeArgs | None = None, *, latent_normalization=None):
        super().__init__()
        self.cfg = cfg or ViTVaeArgs()
        self.latent_normalization = validate_latent_normalization(
            latent_normalization, self.cfg.latent_dim
        )
        for name, default in (("mean", 0.0), ("std", 1.0)):
            values = self.latent_normalization.get(name, [default] * self.cfg.latent_dim)
            self.register_buffer(
                f"latent_{name}", torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1, 1)
            )
        self.vae = ViTVae(**asdict(self.cfg)).eval().requires_grad_(False)
        self.vae.load_state_dict({k.removeprefix("vae."): v for k,v in weights.items()}, strict=True)
        self.temporal_downsampling = 1
        self.content_height = self.cfg.input_height // self.cfg.patch_size
        self.content_width = self.cfg.input_width // self.cfg.patch_size
        self.model_height = self.content_height
        self.model_width = self.content_width

    @classmethod
    def from_run_config(cls, weights, run_config, cfg: ViTVaeArgs | None = None):
        return cls(weights, cfg, latent_normalization=normalization_from_run_config(run_config))

    def get_config(self):
        return {
            "type": "pixel-vae",
            "latent_normalization": deepcopy(self.latent_normalization),
            "temporal_downsampling": self.temporal_downsampling,
            "content_size": [self.content_height, self.content_width],
            "model_size": [self.model_height, self.model_width],
        }

    def pixel_frames_to_latent_frames(self, frames):
        return int(frames)

    def latent_frames_to_pixel_frames(self, frames):
        return int(frames)

    def mask_model_latent(self, latent):
        return latent

    def align_pixel_frames(self, value):
        return value

    def normalize_latent(self, latent):
        if self.latent_normalization["mode"] == "none":
            return latent
        # Compute the affine transform in FP32, preserving the VAE output dtype
        # so BF16 rollout does not silently allocate FP32 KV caches.
        return ((latent.float() - self.latent_mean.float()) / self.latent_std.float()).to(latent.dtype)

    def denormalize_latent(self, latent):
        if self.latent_normalization["mode"] == "none":
            return latent
        return (latent.float() * self.latent_std.float() + self.latent_mean.float()).to(latent.dtype)

    def train(self, mode=True):
        # This module is always frozen, even when a parent model enters train().
        return super().train(False)

    @torch.no_grad()
    def encode(self, rgb, chunk_size=4):
        """RGB [B,T,3,H,W] in [0,1] -> latent [B,T,C,H/10,W/10]."""
        if rgb.ndim != 5 or rgb.shape[2:] != (3,self.cfg.input_height,self.cfg.input_width):
            raise ValueError("RGB shape does not match the Pixel VAE configuration")
        b,t = rgb.shape[:2]
        flat = rgb.flatten(0,1)*2-1
        latent = torch.cat([self.vae.encode(x).mode() for x in flat.split(chunk_size)])
        h,w = self.cfg.input_height//self.cfg.patch_size, self.cfg.input_width//self.cfg.patch_size
        latent = latent.reshape(b,t,h,w,self.cfg.latent_dim).permute(0,1,4,2,3).contiguous()
        return self.normalize_latent(latent)

    def _decode(self, latent, chunk_size):
        h,w = self.cfg.input_height//self.cfg.patch_size, self.cfg.input_width//self.cfg.patch_size
        if latent.ndim != 5 or latent.shape[2:] != (self.cfg.latent_dim,h,w):
            raise ValueError("latent shape does not match the Pixel VAE configuration")
        b,t = latent.shape[:2]
        latent = self.denormalize_latent(latent)
        tokens = latent.permute(0,1,3,4,2).reshape(b*t,h*w,self.cfg.latent_dim)
        rgb = torch.cat([self.vae.decode(x) for x in tokens.split(chunk_size)])
        return ((rgb+1)/2).clamp(0,1).reshape(b,t,3,self.cfg.input_height,self.cfg.input_width)

    @torch.no_grad()
    def decode(self, latent, chunk_size=4):
        """Completed latent frames -> RGB [B,T,3,H,W] in [0,1]."""
        return self._decode(latent, chunk_size)

    def decode_for_loss(self, latent, chunk_size=1):
        """Decode selected frames while retaining gradients to their latents.

        VAE parameters remain frozen; this graph is used only to send a
        full-resolution pixel loss back into the renderer prediction.
        """
        return self._decode(latent, chunk_size)


class WanRendererCodec(nn.Module):
    """Frozen Wan2.1 causal 3D VAE with exact 360x640 external I/O."""

    def __init__(self, weights_path, *, input_height=360, input_width=640, z_dim=16):
        super().__init__()
        if input_height % 8 or input_width % 8:
            raise ValueError("Wan-VAE input height and width must be divisible by 8")
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.latent_dim = int(z_dim)
        self.temporal_downsampling = 4
        self.spatial_downsampling = 8
        self.content_height = self.input_height // self.spatial_downsampling
        self.content_width = self.input_width // self.spatial_downsampling
        self.model_height = self.content_height
        self.model_width = self.content_width
        self.weights_path = str(weights_path)
        self.vae = WanVAE_(
            dim=96,
            z_dim=self.latent_dim,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.vae.load_state_dict(state, strict=True)
        self.vae.eval().requires_grad_(False)
        mean = torch.tensor(WAN_LATENT_MEAN, dtype=torch.float32)
        inv_std = 1.0 / torch.tensor(WAN_LATENT_STD, dtype=torch.float32)
        self.register_buffer("wan_mean", mean, persistent=False)
        self.register_buffer("wan_inv_std", inv_std, persistent=False)

    def train(self, mode=True):
        return super().train(False)

    def get_config(self):
        return {
            "type": "wan2.1-vae",
            "weights_path": self.weights_path,
            "latent_normalization": {"mode": "wan-per-channel"},
            "temporal_downsampling": self.temporal_downsampling,
            "spatial_downsampling": self.spatial_downsampling,
            "pixel_size": [self.input_height, self.input_width],
            "content_size": [self.content_height, self.content_width],
            "model_size": [self.model_height, self.model_width],
        }

    def pixel_frames_to_latent_frames(self, frames):
        frames = int(frames)
        if frames < 1 or (frames - 1) % self.temporal_downsampling:
            raise ValueError("Wan-VAE frame count must satisfy (T-1) % 4 == 0")
        return 1 + (frames - 1) // self.temporal_downsampling

    def latent_frames_to_pixel_frames(self, frames):
        frames = int(frames)
        if frames < 1:
            raise ValueError("latent frame count must be positive")
        return 1 + (frames - 1) * self.temporal_downsampling

    def mask_model_latent(self, latent):
        if latent.shape[-2:] != (self.model_height, self.model_width):
            raise ValueError("latent shape does not match Wan M3 model size")
        return latent

    def align_pixel_frames(self, value):
        latent_frames = self.pixel_frames_to_latent_frames(value.shape[1])
        indices = torch.arange(
            0, latent_frames * self.temporal_downsampling,
            self.temporal_downsampling, device=value.device,
        )
        indices[0] = 0
        return value.index_select(1, indices)

    def _validate_latent(self, latent):
        expected = (self.latent_dim, self.model_height, self.model_width)
        if latent.ndim != 5 or latent.shape[2:] != expected:
            raise ValueError("latent shape does not match Wan M3 configuration")
        return latent

    @torch.no_grad()
    def encode(self, rgb, chunk_size=1):
        del chunk_size
        if rgb.ndim != 5 or rgb.shape[2:] != (
            3, self.input_height, self.input_width
        ):
            raise ValueError("RGB shape does not match the Wan-VAE configuration")
        expected_t = self.pixel_frames_to_latent_frames(rgb.shape[1])
        x = (rgb * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
        dtype = next(self.vae.parameters()).dtype
        zs = [
            self.vae.encode(sample.to(dtype), [self.wan_mean, self.wan_inv_std])
            for sample in x.split(1)
        ]
        z = torch.cat(zs).permute(0, 2, 1, 3, 4).to(rgb.dtype)
        if z.shape[1:] != (
            expected_t, self.latent_dim, self.content_height, self.content_width
        ):
            raise RuntimeError(f"unexpected Wan latent shape {tuple(z.shape)}")
        return z

    def _decode(self, latent, *, retain_grad):
        z = self._validate_latent(latent).permute(0, 2, 1, 3, 4).contiguous()
        dtype = next(self.vae.parameters()).dtype
        context = torch.enable_grad() if retain_grad else torch.no_grad()
        with context:
            xs = [
                self.vae.decode(sample.to(dtype), [self.wan_mean, self.wan_inv_std]).clamp(-1, 1)
                for sample in z.split(1)
            ]
            x = torch.cat(xs).permute(0, 2, 1, 3, 4).to(latent.dtype)
        expected_t = self.latent_frames_to_pixel_frames(latent.shape[1])
        if x.shape[1:] != (expected_t, 3, self.input_height, self.input_width):
            raise RuntimeError(f"unexpected Wan decoded shape {tuple(x.shape)}")
        return ((x + 1) / 2).clamp(0, 1)

    @torch.no_grad()
    def decode(self, latent, chunk_size=1):
        del chunk_size
        return self._decode(latent, retain_grad=False)

    def decode_for_loss(self, latent, chunk_size=1):
        del chunk_size
        return self._decode(latent, retain_grad=True)
