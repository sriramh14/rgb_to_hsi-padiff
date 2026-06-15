from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def rgb_consistency_loss(
    project_hsi_to_rgb: Callable[[torch.Tensor], torch.Tensor],
    pred_hsi: torch.Tensor,
    target_hsi: torch.Tensor,
    rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Weak RGB consistency loss when the true camera response is unavailable.

    project_hsi_to_rgb should usually use a detached pseudo-response so this
    term trains the HSI predictor, not the projector itself.
    """
    rgb_from_pred = project_hsi_to_rgb(pred_hsi)
    rgb_from_target = project_hsi_to_rgb(target_hsi)
    return 0.5 * F.l1_loss(rgb_from_pred, rgb) + 0.5 * F.l1_loss(rgb_from_pred, rgb_from_target)
