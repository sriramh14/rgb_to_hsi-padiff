"""
padiff_hsi_reconstruction.py

A PA-Diff-inspired model adapted for RGB -> hyperspectral image reconstruction.

Original PA-Diff idea:
    - Physics Prior Generation branch
    - Implicit Neural Reconstruction branch
    - Physics-aware Diffusion Transformer branch

This RGB->HSI adaptation replaces underwater priors with spectral sensing priors:
    - coarse HSI prior from RGB
    - RGB consistency residual through a camera spectral response matrix
    - per-band uncertainty/metamer prior
    - optional INR-like coordinate-aware spectral refinement

Expected tensors:
    rgb: [B, 3, H, W], normalized to [0, 1] or [-1, 1] consistently with hsi
    hsi: [B, out_channels, H, W], usually out_channels=31

Training:
    out = model(rgb, hsi)
    # Losses are computed outside this file, e.g. in main.py via losses/padiff_loss.py.

Inference:
    pred = model.reconstruct(rgb, steps=10)     # DDIM sampling
    coarse = model.prior_only(rgb)["coarse_hsi"]  # fast deterministic prior
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def exists(x) -> bool:
    return x is not None


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal diffusion timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device).float() * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for [B, C, H, W]."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=p),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


# -----------------------------------------------------------------------------
# Camera spectral response / RGB consistency
# -----------------------------------------------------------------------------


class SpectralProjector(nn.Module):
    """
    Projects HSI to RGB.

    If camera_matrix is provided, it should have shape [3, C] where C is the
    number of spectral bands. Each RGB channel is a weighted sum over bands.

    If no camera_matrix is provided, the model learns a constrained pseudo
    camera response using softmax-normalized response logits. This is suitable
    when the true camera spectral response is unavailable. The learned response
    should be interpreted as a data-driven pseudo response, not a measured SRF.
    """

    def __init__(self, out_channels: int = 31, camera_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.out_channels = out_channels
        if camera_matrix is not None:
            if camera_matrix.ndim != 2 or camera_matrix.shape[0] != 3 or camera_matrix.shape[1] != out_channels:
                raise ValueError(
                    f"camera_matrix must have shape [3, {out_channels}], got {tuple(camera_matrix.shape)}"
                )
            cm = camera_matrix.float().clamp_min(0)
            cm = cm / (cm.sum(dim=1, keepdim=True) + 1e-8)
            self.register_buffer("camera_matrix", cm)
            self.response_logits = None
        else:
            self.camera_matrix = None
            init = self._make_grouped_rgb_logits(out_channels)
            self.response_logits = nn.Parameter(init)

    @staticmethod
    def _make_grouped_rgb_logits(out_channels: int) -> torch.Tensor:
        # Smooth initialization: early/middle/late spectral bumps.
        c = out_channels
        centers = torch.tensor([0.18, 0.50, 0.82]) * (c - 1)
        xs = torch.arange(c).float()
        sigma = max(c / 7.0, 1.0)
        curves = []
        for center in centers:
            g = torch.exp(-0.5 * ((xs - center) / sigma) ** 2)
            g = g / (g.sum() + 1e-8)
            curves.append(torch.log(g + 1e-8))
        return torch.stack(curves, dim=0)  # [3, C]

    @property
    def response(self) -> torch.Tensor:
        if self.camera_matrix is not None:
            return self.camera_matrix
        return torch.softmax(self.response_logits, dim=1)

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        # hsi: [B, C, H, W], response: [3, C] -> rgb: [B, 3, H, W]
        return torch.einsum("bchw,rc->brhw", hsi, self.response)


# -----------------------------------------------------------------------------
# Branch 1: Spectral Prior Generation (SPG)
# -----------------------------------------------------------------------------


class SpectralPriorGeneration(nn.Module):
    """
    PA-Diff PPG analogue for RGB->HSI.

    Instead of underwater transmission/background light, this branch predicts:
      1. coarse_hsi: deterministic HSI estimate
      2. uncertainty: per-band uncertainty/metamer ambiguity map
      3. rgb_residual: RGB - camera(coarse_hsi), a sensing-consistency cue
      4. global_prior: global material/illumination token
    """

    def __init__(self, out_channels: int = 31, dim: int = 48, camera_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.out_channels = out_channels
        self.projector = SpectralProjector(out_channels, camera_matrix)

        self.encoder = nn.Sequential(
            ConvBlock(3, dim),
            ConvBlock(dim, dim),
            ConvBlock(dim, dim),
        )
        self.coarse_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 3, padding=1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(dim + out_channels + 3, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.global_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim + 3, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(rgb)
        # Keep unconstrained during training; caller may clamp for visualization.
        coarse_hsi = self.coarse_head(feat)
        rgb_from_hsi = self.projector(coarse_hsi)
        rgb_residual = rgb - rgb_from_hsi
        uncertainty = self.uncertainty_head(torch.cat([feat, coarse_hsi, rgb_residual], dim=1))
        global_prior = self.global_mlp(torch.cat([feat, rgb_residual], dim=1))
        return {
            "spg_feat": feat,
            "coarse_hsi": coarse_hsi,
            "rgb_from_coarse": rgb_from_hsi,
            "rgb_residual": rgb_residual,
            "uncertainty": uncertainty,
            "global_prior": global_prior,
        }


# -----------------------------------------------------------------------------
# Branch 2: INR-like coordinate-aware spectral reconstruction
# -----------------------------------------------------------------------------


class CoordinateEncoding(nn.Module):
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs

    @property
    def out_channels(self) -> int:
        # x,y plus sin/cos for x,y at each frequency
        return 2 + 4 * self.num_freqs

    def forward(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=0)  # [2, H, W]
        enc = [coords]
        for i in range(self.num_freqs):
            freq = (2.0 ** i) * math.pi
            enc += [torch.sin(freq * coords), torch.cos(freq * coords)]
        enc = torch.cat(enc, dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return enc


class ImplicitSpectralReconstruction(nn.Module):
    """
    Lightweight INR-inspired branch.

    A strict per-pixel MLP INR is expensive for image restoration training. This
    implementation keeps the INR idea by injecting periodic coordinates into a
    local MLP/1x1-conv spectral renderer.
    """

    def __init__(self, out_channels: int = 31, dim: int = 48, num_freqs: int = 6):
        super().__init__()
        self.coord = CoordinateEncoding(num_freqs)
        in_ch = 3 + out_channels + out_channels + self.coord.out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 1),
        )

    def forward(self, rgb: torch.Tensor, coarse_hsi: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        b, _, h, w = rgb.shape
        coords = self.coord(b, h, w, rgb.device, rgb.dtype)
        delta = self.net(torch.cat([rgb, coarse_hsi, uncertainty, coords], dim=1))
        return coarse_hsi + delta


# -----------------------------------------------------------------------------
# Branch 3: Sensing-aware Diffusion Transformer blocks
# -----------------------------------------------------------------------------


class PriorAdapter(nn.Module):
    """Turns SPG/INR outputs into multi-scale prior features."""

    def __init__(self, out_channels: int = 31, dim: int = 48):
        super().__init__()
        # rgb + coarse_hsi + inr_hsi + uncertainty + rgb_residual
        in_ch = 3 + out_channels + out_channels + out_channels + 3
        self.proj = ConvBlock(in_ch, dim)
        self.down1 = ConvBlock(dim, dim * 2, stride=2)
        self.down2 = ConvBlock(dim * 2, dim * 4, stride=2)

    def forward(
        self,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        inr_hsi: torch.Tensor,
        uncertainty: torch.Tensor,
        rgb_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = self.proj(torch.cat([rgb, coarse_hsi, inr_hsi, uncertainty, rgb_residual], dim=1))
        p1 = self.down1(p0)
        p2 = self.down2(p1)
        return p0, p1, p2


class SensingAwareAttention(nn.Module):
    """
    Restormer-style transposed self-attention with prior-aware query modulation.

    This avoids full H*W x H*W attention, making it usable on 128/256 patches.
    """

    def __init__(self, dim: int, heads: int = 4, time_dim: int = 192):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.norm = LayerNorm2d(dim)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.prior_q = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
        )
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        y = y + self.time_proj(t_emb).view(b, c, 1, 1)

        q, k, v = self.qkv_dw(self.qkv(y)).chunk(3, dim=1)
        q = q + self.prior_q(prior)

        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return x + self.out(out)


class SpectralPerceptionUnit(nn.Module):
    """
    Feature-level sensing prior module.

    PA-Diff uses a feature-level physics perception unit for underwater imaging.
    Here, the prior feature encodes camera consistency residual and metamer
    uncertainty, then modulates restoration features locally and globally.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm_x = LayerNorm2d(dim)
        self.norm_p = LayerNorm2d(dim)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2),
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        xn = self.norm_x(x)
        pn = self.norm_p(prior)
        scale, shift = self.local(pn).chunk(2, dim=1)
        scale = torch.tanh(scale)
        gate = self.global_gate(pn)
        y = xn * (1.0 + scale) + shift
        y = y * gate
        return x + self.out(y)


class GatedMultiScaleFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = LayerNorm2d(dim)
        self.in_proj = nn.Conv2d(dim, hidden * 2, 1)
        # three depth-wise scales; no color choices / external deps
        self.dw3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.dw5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.dw7 = nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        a, gate = self.in_proj(y).chunk(2, dim=1)
        a = (self.dw3(a) + self.dw5(a) + self.dw7(a)) / 3.0
        y = F.gelu(a) * gate
        return x + self.out_proj(y)


class SensingAwareTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, time_dim: int = 192, ffn_expansion: float = 2.0):
        super().__init__()
        self.attn = SensingAwareAttention(dim, heads=heads, time_dim=time_dim)
        self.spu = SpectralPerceptionUnit(dim)
        self.ffn = GatedMultiScaleFFN(dim, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, prior: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, prior, t_emb)
        x = self.spu(x, prior)
        x = self.ffn(x)
        return x


class SensingAwareDiffusionTransformer(nn.Module):
    """U-shaped diffusion Transformer denoiser for HSI noise prediction."""

    def __init__(
        self,
        out_channels: int = 31,
        dim: int = 48,
        heads: Tuple[int, int, int] = (2, 4, 8),
        blocks: Tuple[int, int, int] = (2, 2, 4),
        time_dim: int = 192,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # x_t + rgb + coarse_hsi + inr_hsi
        self.embedding = ConvBlock(out_channels + 3 + out_channels + out_channels, dim)

        self.enc0 = nn.ModuleList([
            SensingAwareTransformerBlock(dim, heads=heads[0], time_dim=time_dim) for _ in range(blocks[0])
        ])
        self.down0 = ConvBlock(dim, dim * 2, stride=2)

        self.enc1 = nn.ModuleList([
            SensingAwareTransformerBlock(dim * 2, heads=heads[1], time_dim=time_dim) for _ in range(blocks[1])
        ])
        self.down1 = ConvBlock(dim * 2, dim * 4, stride=2)

        self.mid = nn.ModuleList([
            SensingAwareTransformerBlock(dim * 4, heads=heads[2], time_dim=time_dim) for _ in range(blocks[2])
        ])

        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.fuse1 = ConvBlock(dim * 4, dim * 2)
        self.dec1 = nn.ModuleList([
            SensingAwareTransformerBlock(dim * 2, heads=heads[1], time_dim=time_dim) for _ in range(blocks[1])
        ])

        self.up0 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.fuse0 = ConvBlock(dim * 2, dim)
        self.dec0 = nn.ModuleList([
            SensingAwareTransformerBlock(dim, heads=heads[0], time_dim=time_dim) for _ in range(blocks[0])
        ])

        self.out = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 3, padding=1),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        rgb: torch.Tensor,
        coarse_hsi: torch.Tensor,
        inr_hsi: torch.Tensor,
        priors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        p0, p1, p2 = priors
        x = self.embedding(torch.cat([x_t, rgb, coarse_hsi, inr_hsi], dim=1))

        for blk in self.enc0:
            x = blk(x, p0, t_emb)
        skip0 = x
        x = self.down0(x)

        for blk in self.enc1:
            x = blk(x, p1, t_emb)
        skip1 = x
        x = self.down1(x)

        for blk in self.mid:
            x = blk(x, p2, t_emb)

        x = self.up1(x)
        x = self._match_size(x, skip1)
        x = self.fuse1(torch.cat([x, skip1], dim=1))
        for blk in self.dec1:
            x = blk(x, p1, t_emb)

        x = self.up0(x)
        x = self._match_size(x, skip0)
        x = self.fuse0(torch.cat([x, skip0], dim=1))
        for blk in self.dec0:
            x = blk(x, p0, t_emb)

        return self.out(x)


# -----------------------------------------------------------------------------
# Full model with DDPM training and DDIM reconstruction
# -----------------------------------------------------------------------------


class PADiffHSIReconstruction(nn.Module):
    """
    Full PA-Diff-inspired RGB->HSI model.

    Args:
        out_channels: number of HSI bands, e.g. 31 for NTIRE/ARAD-style RGB->HSI.
        dim: base feature dimension. Use 32 for light model, 48/64 for stronger model.
        timesteps: diffusion training timesteps.
        camera_matrix: optional spectral response matrix [3, out_channels].
    """

    def __init__(
        self,
        out_channels: int = 31,
        dim: int = 48,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        camera_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.timesteps = timesteps
        self.spg = SpectralPriorGeneration(out_channels, dim, camera_matrix)
        self.inr = ImplicitSpectralReconstruction(out_channels, dim)
        self.prior_adapter = PriorAdapter(out_channels, dim)
        self.denoiser = SensingAwareDiffusionTransformer(out_channels, dim)
        self.projector = self.spg.projector


        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        b = t.shape[0]
        out = a.gather(0, t).reshape(b, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_om * noise) / (sqrt_alpha + 1e-8)

    def prior_only(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        spg = self.spg(rgb)
        inr_hsi = self.inr(rgb, spg["coarse_hsi"], spg["uncertainty"])
        spg["inr_hsi"] = inr_hsi
        return spg

    def _build_condition(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        spg = self.spg(rgb)
        inr_hsi = self.inr(rgb, spg["coarse_hsi"], spg["uncertainty"])
        priors = self.prior_adapter(
            rgb=rgb,
            coarse_hsi=spg["coarse_hsi"],
            inr_hsi=inr_hsi,
            uncertainty=spg["uncertainty"],
            rgb_residual=spg["rgb_residual"],
        )
        spg["inr_hsi"] = inr_hsi
        spg["priors"] = priors
        return spg

    def denoise(self, x_t: torch.Tensor, rgb: torch.Tensor, cond: Dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(
            x_t=x_t,
            rgb=rgb,
            coarse_hsi=cond["coarse_hsi"],
            inr_hsi=cond["inr_hsi"],
            priors=cond["priors"],
            t=t,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        hsi: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Architecture-only forward.

        When hsi is provided, returns tensors needed by the external loss modules.
        No loss is computed inside the model file.

        When hsi is None, calls reconstruct(rgb).
        """
        if hsi is None:
            return self.reconstruct(rgb)

        b = rgb.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=rgb.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(hsi)

        cond = self._build_condition(rgb)
        x_t = self.q_sample(hsi, t, noise)
        pred_noise = self.denoise(x_t, rgb, cond, t)
        pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)

        return {
            "pred_noise": pred_noise,
            "noise": noise,
            "x_t": x_t,
            "pred_x0": pred_x0,
            "coarse_hsi": cond["coarse_hsi"],
            "inr_hsi": cond["inr_hsi"],
            "uncertainty": cond["uncertainty"],
            "rgb_residual": cond["rgb_residual"],
            "rgb_from_coarse": cond["rgb_from_coarse"],
            "learned_response": self.projector.response,
            "t": t,
        }

    @torch.no_grad()
    def reconstruct(
        self,
        rgb: torch.Tensor,
        steps: int = 10,
        eta: float = 0.0,
        clamp: bool = True,
        start_from_prior: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling for RGB->HSI reconstruction.

        Args:
            steps: number of reverse steps. 10-50 is usually practical.
            eta: 0.0 gives deterministic DDIM; >0 adds stochasticity.
            start_from_prior: if True, initialize around the INR prior plus noise,
                which is usually better for supervised RGB->HSI than pure Gaussian.
        """
        self.eval()
        b, _, h, w = rgb.shape
        cond = self._build_condition(rgb)
        device = rgb.device

        if start_from_prior:
            x = cond["inr_hsi"] + torch.randn(b, self.out_channels, h, w, device=device, dtype=rgb.dtype)
        else:
            x = torch.randn(b, self.out_channels, h, w, device=device, dtype=rgb.dtype)

        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i, time in enumerate(times):
            t = torch.full((b,), int(time.item()), device=device, dtype=torch.long)
            pred_noise = self.denoise(x, rgb, cond, t)
            x0 = self.predict_x0_from_noise(x, t, pred_noise)

            if i == len(times) - 1:
                x = x0
                break

            next_t = torch.full((b,), int(times[i + 1].item()), device=device, dtype=torch.long)
            a_t = self._extract(self.alphas_cumprod, t, x.shape)
            a_next = self._extract(self.alphas_cumprod, next_t, x.shape)

            sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t + 1e-8) * (1 - a_t / (a_next + 1e-8))).clamp_min(0)
            c = torch.sqrt((1 - a_next - sigma ** 2).clamp_min(0))
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(a_next) * x0 + c * pred_noise + sigma * noise

        return x.clamp(0.0, 1.0) if clamp else x


# -----------------------------------------------------------------------------
# Small smoke test
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = PADiffHSIReconstruction(out_channels=31, dim=16, timesteps=50)
    rgb = torch.rand(1, 3, 16, 16)
    hsi = torch.rand(1, 31, 16, 16)
    out = model(rgb, hsi)
    print("keys:", sorted(out.keys()))
    print("pred_x0:", tuple(out["pred_x0"].shape))
    pred = model.reconstruct(rgb[:1], steps=2)
    print("sample:", tuple(pred.shape), float(pred.min()), float(pred.max()))
