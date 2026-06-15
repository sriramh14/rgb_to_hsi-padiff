from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_noise_loss(pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Standard DDPM epsilon-prediction objective.

    Both tensors must have shape [B, C, H, W].
    """
    return F.mse_loss(pred_noise, noise)
