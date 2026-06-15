from __future__ import annotations

import torch
import torch.nn.functional as F


def projector_calibration_loss(projected_rgb: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Self-calibrates the learned pseudo HSI->RGB projector using paired data."""
    return F.l1_loss(projected_rgb, rgb)


def projector_smoothness_loss(response: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness regularizer for spectral response curves."""
    if response.shape[1] < 3:
        return response.new_tensor(0.0)
    second_diff = response[:, 2:] - 2 * response[:, 1:-1] + response[:, :-2]
    return second_diff.abs().mean()


def projector_diversity_loss(response: torch.Tensor) -> torch.Tensor:
    """Discourages learned R/G/B pseudo-response curves from collapsing together."""
    r = F.normalize(response, dim=1)
    sim = torch.matmul(r, r.t())
    off_diag = sim - torch.eye(3, device=sim.device, dtype=sim.dtype)
    return off_diag.clamp_min(0).mean()
