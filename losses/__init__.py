"""Loss package for PA-Diff-inspired RGB -> HSI reconstruction."""

from .diffusion_loss import diffusion_noise_loss
from .reconstruction_loss import l1_reconstruction_loss
from .mrae_loss import mrae_loss
from .sam_loss import sam_loss, sam_metric_deg
from .rgb_consistency_loss import rgb_consistency_loss
from .projector_losses import (
    projector_calibration_loss,
    projector_smoothness_loss,
    projector_diversity_loss,
)
from .padiff_loss import compute_padiff_training_losses

__all__ = [
    "diffusion_noise_loss",
    "l1_reconstruction_loss",
    "mrae_loss",
    "sam_loss",
    "sam_metric_deg",
    "rgb_consistency_loss",
    "projector_calibration_loss",
    "projector_smoothness_loss",
    "projector_diversity_loss",
    "compute_padiff_training_losses",
]
