from __future__ import annotations

from typing import Callable, Dict

import torch

from .diffusion_loss import diffusion_noise_loss
from .mrae_loss import mrae_loss
from .projector_losses import (
    projector_calibration_loss,
    projector_diversity_loss,
    projector_smoothness_loss,
)
from .reconstruction_loss import l1_reconstruction_loss
from .rgb_consistency_loss import rgb_consistency_loss
from .sam_loss import sam_loss


def _w(weights: Dict[str, float], key: str) -> float:
    return float(weights.get(key, 0.0))


def _zero_like_ref(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_tensor(0.0)


def compute_padiff_training_losses(
    *,
    rgb: torch.Tensor,
    hsi: torch.Tensor,
    pred_noise: torch.Tensor,
    noise: torch.Tensor,
    pred_x0: torch.Tensor,
    coarse_hsi: torch.Tensor,
    inr_hsi: torch.Tensor,
    project_hsi_to_rgb_detached: Callable[[torch.Tensor], torch.Tensor],
    projector_rgb_from_gt: torch.Tensor,
    projector_response: torch.Tensor,
    weights: Dict[str, float],
    mrae_eps: float = 1e-3,
    sam_eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Central loss combiner for PA-Diff HSI training.

    Notes
    -----
    * Individual loss formulas remain in separate files for debugging.
    * Expensive/unstable terms whose weights are zero are not computed. This is
      intentional: pred_x0 at very high diffusion timesteps can be numerically
      large early in training, so logging its MRAE when the term is disabled can
      create misleading values such as 30+.
    """
    zero = _zero_like_ref(hsi)

    loss_diff = diffusion_noise_loss(pred_noise, noise) if _w(weights, "diff") else zero

    loss_x0_l1 = l1_reconstruction_loss(pred_x0, hsi) if _w(weights, "x0_l1") else zero
    loss_x0_mrae = mrae_loss(pred_x0, hsi, eps=mrae_eps) if _w(weights, "x0_mrae") else zero
    loss_x0_sam = sam_loss(pred_x0, hsi, eps=sam_eps) if _w(weights, "x0_sam") else zero

    loss_coarse = l1_reconstruction_loss(coarse_hsi, hsi) if _w(weights, "coarse") else zero
    loss_inr = l1_reconstruction_loss(inr_hsi, hsi) if _w(weights, "inr") else zero

    if _w(weights, "rgb"):
        loss_rgb = rgb_consistency_loss(
            project_hsi_to_rgb=project_hsi_to_rgb_detached,
            pred_hsi=pred_x0,
            target_hsi=hsi,
            rgb=rgb,
        )
    else:
        loss_rgb = zero

    loss_projector = (
        projector_calibration_loss(projector_rgb_from_gt, rgb)
        if _w(weights, "projector")
        else zero
    )
    loss_projector_smooth = (
        projector_smoothness_loss(projector_response)
        if _w(weights, "projector_smooth")
        else zero
    )
    loss_projector_diversity = (
        projector_diversity_loss(projector_response)
        if _w(weights, "projector_diversity")
        else zero
    )

    loss_aux = (
        _w(weights, "x0_l1") * loss_x0_l1
        + _w(weights, "x0_mrae") * loss_x0_mrae
        + _w(weights, "x0_sam") * loss_x0_sam
        + _w(weights, "coarse") * loss_coarse
        + _w(weights, "inr") * loss_inr
        + _w(weights, "rgb") * loss_rgb
        + _w(weights, "projector") * loss_projector
        + _w(weights, "projector_smooth") * loss_projector_smooth
        + _w(weights, "projector_diversity") * loss_projector_diversity
    )
    loss = _w(weights, "diff") * loss_diff + loss_aux

    return {
        "loss": loss,
        "loss_diff": loss_diff,
        "loss_aux": loss_aux,
        "loss_x0_l1": loss_x0_l1,
        "loss_x0_mrae": loss_x0_mrae,
        "loss_x0_sam": loss_x0_sam,
        "loss_coarse": loss_coarse,
        "loss_inr": loss_inr,
        "loss_rgb": loss_rgb,
        "loss_projector": loss_projector,
        "loss_projector_smooth": loss_projector_smooth,
        "loss_projector_diversity": loss_projector_diversity,
    }
