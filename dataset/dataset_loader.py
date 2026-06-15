import os
import numpy as np
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from huggingface_hub import (
    list_repo_files,
    hf_hub_download
)


def normalize_hsi_cube(hsi: np.ndarray) -> np.ndarray:

    """Normalize an HSI cube to [0, 1] for RGB->HSI training.

    The RGB images are loaded as float32 / 255, so the HSI target must use
    the same numeric range. ARAD/NTIRE .mat cubes are often already [0, 1],
    but some local copies can be stored with larger raw ranges.
    """

    hsi = np.nan_to_num(
        hsi.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    hsi_min = float(hsi.min())
    hsi_max = float(hsi.max())

    if hsi_max <= 0.0:
        return np.zeros_like(hsi, dtype=np.float32)

    if hsi_min < 0.0 or hsi_max > 1.0:
        hsi = hsi / hsi_max

    hsi = np.clip(hsi, 0.0, 1.0)
    return hsi.astype(np.float32, copy=False)


class ARADDataset(Dataset):

    def __init__(
        self,
        root_dir="data",
        train=True,
        train_images=200,
        total_images=230,
        cube_key="cube",
        download=True
    ):

        self.cube_key = cube_key

        spectral_dir = os.path.join(
            root_dir,
            "NTIRE2020_Train_Spectral"
        )

        rgb_dir = os.path.join(
            root_dir,
            "NTIRE2020_Train_RealWorld"
        )

        os.makedirs(
            spectral_dir,
            exist_ok=True
        )

        os.makedirs(
            rgb_dir,
            exist_ok=True
        )

        ##################################################
        # Download
        ##################################################

        if download:

            existing_hsi = [
                f for f in os.listdir(
                    spectral_dir
                )
                if f.endswith(".mat")
            ]

            existing_rgb = [
                f for f in os.listdir(
                    rgb_dir
                )
                if f.endswith(".jpg")
            ]

            if (
                len(existing_hsi) < total_images
                or
                len(existing_rgb) < total_images
            ):

                print(
                    f"Downloading "
                    f"{total_images} HSI files "
                    f"and "
                    f"{total_images} RGB files..."
                )

                repo_files = list_repo_files(
                    "mhmdjouni/arad_hsdb",
                    repo_type="dataset"
                )

                hsi_files = sorted([
                    f
                    for f in repo_files
                    if (
                        f.endswith(".mat")
                        and
                        "NTIRE2020_Train_Spectral"
                        in f
                    )
                ])[:total_images]

                rgb_files = sorted([
                    f
                    for f in repo_files
                    if (
                        f.endswith(".jpg")
                        and
                        "NTIRE2020_Train_RealWorld"
                        in f
                    )
                ])[:total_images]

                for file in hsi_files:

                    hf_hub_download(
                        repo_id="mhmdjouni/arad_hsdb",
                        repo_type="dataset",
                        filename=file,
                        local_dir=root_dir,
                        local_dir_use_symlinks=False
                    )

                for file in rgb_files:

                    hf_hub_download(
                        repo_id="mhmdjouni/arad_hsdb",
                        repo_type="dataset",
                        filename=file,
                        local_dir=root_dir,
                        local_dir_use_symlinks=False
                    )

                print(
                    "Download complete"
                )

        ##################################################
        # Build RGB-HSI pairs
        ##################################################

        hsi_files = sorted([
            f
            for f in os.listdir(
                spectral_dir
            )
            if f.endswith(".mat")
        ])[:total_images]

        rgb_lookup = {
            f.replace(
                "_RealWorld.jpg",
                ""
            ): f
            for f in os.listdir(
                rgb_dir
            )
            if f.endswith(".jpg")
        }

        pairs = []

        for hsi_name in hsi_files:

            stem = os.path.splitext(
                hsi_name
            )[0]

            if stem not in rgb_lookup:
                continue

            pairs.append(
                (
                    os.path.join(
                        spectral_dir,
                        hsi_name
                    ),
                    os.path.join(
                        rgb_dir,
                        rgb_lookup[stem]
                    )
                )
            )

        print(
            f"Found {len(pairs)} paired samples"
        )

        ##################################################
        # Train / Validation split
        ##################################################

        if train:

            self.pairs = pairs[
                :train_images
            ]

        else:

            self.pairs = pairs[
                train_images:
            ]

        print(
            f"{'Train' if train else 'Val'}: "
            f"{len(self.pairs)} samples"
        )

    def __len__(self):

        return len(
            self.pairs
        )

    def __getitem__(
        self,
        idx
    ):

        hsi_path, rgb_path = self.pairs[idx]

        ##################################################
        # Load HSI
        ##################################################

        mat = sio.loadmat(
            hsi_path
        )

        if self.cube_key not in mat:
            raise KeyError(
                f"Key '{self.cube_key}' was not found in {hsi_path}. "
                f"Available keys: {list(mat.keys())}"
            )

        hsi = normalize_hsi_cube(
            mat[self.cube_key]
        )

        hsi = np.transpose(
            hsi,
            (2, 0, 1)
        )

        hsi = torch.from_numpy(
            hsi
        ).float()

        hsi = F.interpolate(
            hsi.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        ##################################################
        # Load RGB
        ##################################################

        rgb = Image.open(
            rgb_path
        ).convert("RGB")

        rgb = np.array(
            rgb,
            dtype=np.float32
        ) / 255.0

        rgb = np.transpose(
            rgb,
            (2, 0, 1)
        )

        rgb = torch.from_numpy(
            rgb
        ).float()

        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        ##################################################
        # Return pair
        ##################################################

        return rgb, hsi
