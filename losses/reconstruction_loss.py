from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel/band-wise L1 reconstruction loss for HSI tensors."""
    return F.l1_loss(pred, target)
