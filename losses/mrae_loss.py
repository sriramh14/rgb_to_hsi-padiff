from __future__ import annotations

import torch


def mrae_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Mean Relative Absolute Error for HSI reconstruction.

    Formula:
        mean(|pred - target| / (|target| + eps))

    Scaling note:
        This returns decimal MRAE. For example, 0.16 means 16% relative error.
        It does not multiply by 100.

    eps note:
        eps=1e-3 is commonly safer for normalized HSI in [0, 1], because very
        dark spectral values near zero can otherwise dominate the loss.
    """
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()
