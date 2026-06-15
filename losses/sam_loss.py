from __future__ import annotations

import torch


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean Spectral Angle Mapper loss in radians.

    Input shape: [B, C, H, W], where C is the number of spectral bands.
    """
    p = pred.flatten(2)
    t = target.flatten(2)
    dot = (p * t).sum(dim=1)
    denom = torch.linalg.norm(p, dim=1) * torch.linalg.norm(t, dim=1) + eps
    cos = (dot / denom).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def sam_metric_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean SAM metric in degrees, useful for validation logs."""
    return torch.rad2deg(sam_loss(pred, target, eps=eps))
