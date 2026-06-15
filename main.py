"""
main.py

Training script for PA-Diff-inspired RGB -> 31-band HSI reconstruction.

No command-line parser is used. Edit the CONFIG section directly and run:

    python main.py

Design used in this fixed version:
    model file = architecture only
    losses/    = individual loss implementations
    main.py    = imports and calls the loss modules
"""

from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from losses.padiff_loss import compute_padiff_training_losses
from models.PADiffHSIReconstruction import PADiffHSIReconstruction


# =============================================================================
# CONFIG: edit values here only. No argparse is used.
# =============================================================================

SEED = 42
TORCH_NUM_THREADS = 2       # Prevent CPU thread oversubscription during small-kernel ops.

# Dataset ----------------------------------------------------------------------
DATA_ROOT = "data"
DOWNLOAD_DATA = True          # Safe: existing files are reused by the loader.
CUBE_KEY = "cube"
TOTAL_IMAGES = 230
TRAIN_IMAGES = 200

# Training ---------------------------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 2
PIN_MEMORY = True

LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
USE_AMP = True

# Model ------------------------------------------------------------------------
OUT_CHANNELS = 31
DIM = 48
DIFFUSION_TIMESTEPS = 1000

# Stable default loss setup -----------------------------------------------------
# The previous very bad metrics were mainly caused by target scale mismatch and
# unstable pred_x0 supervision at high random timesteps. This default trains the
# deterministic prior/INR path strongly and trains the denoiser with noise loss,
# while disabling pred_x0 MRAE/SAM until the prior is reasonable.
#
# MRAE is decimal-scale: 0.16 means 16%, not 16.0.
LOSS_WEIGHTS = {
    "diff": 0.10,
    "x0_l1": 0.00,
    "x0_mrae": 0.00,
    "x0_sam": 0.00,
    "coarse": 0.50,
    "inr": 1.00,
    "rgb": 0.00,
    "projector": 0.02,
    "projector_smooth": 0.001,
    "projector_diversity": 0.001,
}
MRAE_EPS = 1e-3
SAM_EPS = 1e-8

# Validation / checkpoints ------------------------------------------------------
OUT_DIR = "runs/padiff_hsi"
SAVE_EVERY_EPOCH = False
VALIDATE_EVERY = 1

# Keep False first. This evaluates the deterministic INR prior and verifies that
# RGB->HSI learning works. Turn True only after prior validation metrics become
# reasonable, otherwise DDIM can hide whether the base reconstruction is learning.
VALIDATE_WITH_DDIM = False
VAL_DDIM_STEPS = 10
VAL_MAX_BATCHES = None

# Resume -----------------------------------------------------------------------
RESUME_PATH = None


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if TORCH_NUM_THREADS is not None and TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("image", batch.get("input")))
        hsi = batch.get("hsi", batch.get("target", batch.get("label")))
        if rgb is None or hsi is None:
            raise KeyError(
                "Dictionary batch must contain rgb/input/image and hsi/target/label keys. "
                f"Available keys: {list(batch.keys())}"
            )
    else:
        rgb, hsi = batch[:2]

    return rgb.float().to(device, non_blocking=True), hsi.float().to(device, non_blocking=True)


def sanitize_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def project_hsi_to_rgb_detached(model: nn.Module, hsi: torch.Tensor) -> torch.Tensor:
    """
    Project HSI to RGB while detaching projector parameters.

    Gradients still flow to pred_hsi, but not to the pseudo-camera response from
    this RGB-consistency term. The pseudo-camera response is trained by the
    projector calibration loss.
    """
    projector = getattr(model, "projector", None)
    if projector is None:
        raise AttributeError("Model must expose model.projector.")

    if hasattr(projector, "response"):
        response = projector.response
        if callable(response):
            response = response()
        return torch.einsum("bchw,rc->brhw", hsi, response.detach())

    raise AttributeError("Unsupported projector type. Expected a .response tensor/property.")


def get_projector_response(model: nn.Module) -> torch.Tensor:
    projector = getattr(model, "projector", None)
    if projector is None:
        raise AttributeError("Model must expose model.projector.")

    response = getattr(projector, "response", None)
    if response is None:
        raise AttributeError("Projector must expose .response for regularization.")
    return response() if callable(response) else response


def compute_losses_from_modules(
    model: PADiffHSIReconstruction,
    out: Dict[str, torch.Tensor],
    rgb: torch.Tensor,
    hsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Call losses/padiff_loss.py. No loss formula is implemented here."""
    return compute_padiff_training_losses(
        rgb=rgb,
        hsi=hsi,
        pred_noise=out["pred_noise"],
        noise=out["noise"],
        pred_x0=out["pred_x0"],
        coarse_hsi=out["coarse_hsi"],
        inr_hsi=out["inr_hsi"],
        project_hsi_to_rgb_detached=lambda x: project_hsi_to_rgb_detached(model, x),
        projector_rgb_from_gt=model.projector(hsi),
        projector_response=get_projector_response(model),
        weights=LOSS_WEIGHTS,
        mrae_eps=MRAE_EPS,
        sam_eps=SAM_EPS,
    )


@torch.no_grad()
def mrae_decimal(pred: torch.Tensor, target: torch.Tensor, eps: float = MRAE_EPS) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    pred = sanitize_tensor(pred.float()).clamp(0.0, 1.0)
    target = sanitize_tensor(target.float()).clamp(0.0, 1.0)

    abs_err = torch.abs(pred - target)
    mae = abs_err.mean()
    rmse = torch.sqrt(F.mse_loss(pred, target) + eps)
    mrae = (abs_err / (torch.abs(target) + MRAE_EPS)).mean()

    mse = F.mse_loss(pred, target)
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))

    p = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    t = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    dot = (p * t).sum(dim=1)
    denom = torch.linalg.norm(p, dim=1) * torch.linalg.norm(t, dim=1) + SAM_EPS
    sam = torch.rad2deg(torch.acos((dot / denom).clamp(-1.0, 1.0))).mean()

    # Lightweight global SSIM approximation over all channels. It is not the
    # official windowed SSIM, but is useful for consistent training logs.
    mu_x = pred.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)
    var_x = ((pred - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    var_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) /
            ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + eps)).mean()

    return {
        "mae": float(mae.detach().cpu()),
        "mrae": float(mrae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "psnr": float(psnr.detach().cpu()),
        "sam": float(sam.detach().cpu()),
        "ssim": float(ssim.detach().cpu()),
    }


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class MetricAverager:
    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.meters.setdefault(key, AverageMeter()).update(value, n)

    def averages(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}


def append_csv(path: Path, row: Dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_mrae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_mrae": best_mrae,
            "config": {
                "out_channels": OUT_CHANNELS,
                "dim": DIM,
                "timesteps": DIFFUSION_TIMESTEPS,
                "loss_weights": LOSS_WEIGHTS,
                "validate_with_ddim": VALIDATE_WITH_DDIM,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: GradScaler | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_mrae", float("inf")))


# =============================================================================
# Data
# =============================================================================


def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=True,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=CUBE_KEY,
        download=DOWNLOAD_DATA,
    )
    val_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=False,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=CUBE_KEY,
        download=DOWNLOAD_DATA,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check DATA_ROOT, TOTAL_IMAGES, and file pairing.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Increase TOTAL_IMAGES or reduce TRAIN_IMAGES.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if VALIDATE_WITH_DDIM else BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader


@torch.no_grad()
def print_data_sanity(loader: DataLoader, device: torch.device) -> None:
    rgb, hsi = to_device(next(iter(loader)), device)
    print(
        "Data sanity | "
        f"RGB min/max/mean: {rgb.min().item():.4f}/{rgb.max().item():.4f}/{rgb.mean().item():.4f} | "
        f"HSI min/max/mean: {hsi.min().item():.4f}/{hsi.max().item():.4f}/{hsi.mean().item():.4f}"
    )
    if hsi.max().item() > 1.05 or hsi.min().item() < -0.05:
        raise RuntimeError("HSI is not normalized to [0, 1]. Check dataset/dataset_loader.py.")


# =============================================================================
# Train / validation
# =============================================================================


def train_one_epoch(
    model: PADiffHSIReconstruction,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    meters = MetricAverager()

    for step, batch in enumerate(loader, start=1):
        rgb, hsi = to_device(batch, device)
        rgb = sanitize_tensor(rgb).clamp(0.0, 1.0)
        hsi = sanitize_tensor(hsi).clamp(0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=USE_AMP and device.type == "cuda"):
            out = model(rgb, hsi)
            loss_dict = compute_losses_from_modules(model, out, rgb, hsi)
            loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            print(f"Warning: non-finite loss at epoch {epoch}, step {step}; skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        n = rgb.shape[0]
        loss_values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
        with torch.no_grad():
            loss_values["train_inr_mrae"] = float(mrae_decimal(out["inr_hsi"], hsi).detach().cpu())
            loss_values["train_coarse_mrae"] = float(mrae_decimal(out["coarse_hsi"], hsi).detach().cpu())
        meters.update(loss_values, n=n)

    return meters.averages()


@torch.no_grad()
def validate(
    model: PADiffHSIReconstruction,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meters = MetricAverager()
    metric_meters = MetricAverager()

    for step, batch in enumerate(loader, start=1):
        if VAL_MAX_BATCHES is not None and step > VAL_MAX_BATCHES:
            break

        rgb, hsi = to_device(batch, device)
        rgb = sanitize_tensor(rgb).clamp(0.0, 1.0)
        hsi = sanitize_tensor(hsi).clamp(0.0, 1.0)

        out = model(rgb, hsi)
        loss_dict = compute_losses_from_modules(model, out, rgb, hsi)
        loss_meters.update(
            {f"val_{key}": float(value.detach().cpu()) for key, value in loss_dict.items()},
            n=rgb.shape[0],
        )

        if VALIDATE_WITH_DDIM:
            pred = model.reconstruct(rgb, steps=VAL_DDIM_STEPS, eta=0.0, clamp=True, start_from_prior=True)
        else:
            pred = model.prior_only(rgb)["inr_hsi"].clamp(0.0, 1.0)

        metric_meters.update(compute_metrics(pred, hsi), n=rgb.shape[0])

    results = {}
    results.update(loss_meters.averages())
    results.update({f"val_{k}": v for k, v in metric_meters.averages().items()})
    return results


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    set_seed(SEED)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")
    print(f"Validation mode: {'DDIM' if VALIDATE_WITH_DDIM else 'INR prior'}")

    train_loader, val_loader = build_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print_data_sanity(train_loader, device)

    model = PADiffHSIReconstruction(
        out_channels=OUT_CHANNELS,
        dim=DIM,
        timesteps=DIFFUSION_TIMESTEPS,
        camera_matrix=None,
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.05,
    )
    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    start_epoch = 1
    best_mrae = float("inf")
    if RESUME_PATH is not None and os.path.isfile(RESUME_PATH):
        start_epoch, best_mrae = load_checkpoint(
            RESUME_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(f"Resumed from {RESUME_PATH} at epoch {start_epoch}; best MRAE {best_mrae:.6f}")

    log_path = out_dir / "log.csv"

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()

        val_stats = {}
        if epoch % VALIDATE_EVERY == 0:
            val_stats = validate(model, val_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "time_sec": elapsed,
            **train_stats,
            **val_stats,
        }
        append_csv(log_path, row)

        if val_stats:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss {train_stats.get('loss', 0):.6f} | "
                f"Train INR-MRAE {train_stats.get('train_inr_mrae', 0):.6f} | "
                f"Val Loss {val_stats.get('val_loss', 0):.6f} | "
                f"Val MRAE {val_stats.get('val_mrae', 0):.6f} | "
                f"Val RMSE {val_stats.get('val_rmse', 0):.6f} | "
                f"Val SAM {val_stats.get('val_sam', 0):.4f} | "
                f"Val PSNR {val_stats.get('val_psnr', 0):.4f} | "
                f"Val SSIM {val_stats.get('val_ssim', 0):.6f} | "
                f"LR {lr_now:.2e} | {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss {train_stats.get('loss', 0):.6f} | "
                f"Train INR-MRAE {train_stats.get('train_inr_mrae', 0):.6f} | "
                f"LR {lr_now:.2e} | {elapsed:.1f}s"
            )

        save_checkpoint(out_dir / "latest.pth", epoch, model, optimizer, scheduler, scaler, best_mrae)

        if SAVE_EVERY_EPOCH:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pth", epoch, model, optimizer, scheduler, scaler, best_mrae)

        current_mrae = val_stats.get("val_mrae", None)
        if current_mrae is not None and current_mrae < best_mrae:
            best_mrae = current_mrae
            save_checkpoint(out_dir / "best.pth", epoch, model, optimizer, scheduler, scaler, best_mrae)
            print(f"Saved new best checkpoint with Val MRAE {best_mrae:.6f}")

    print("Training complete.")
    print(f"Best Val MRAE: {best_mrae:.6f}")
    print(f"Checkpoints/logs saved in: {out_dir}")


if __name__ == "__main__":
    main()
