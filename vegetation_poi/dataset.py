"""
EuroSAT Multispectral Dataset loader for greenery segmentation.
Reads 13-band Sentinel-2 .tif files, extracts RGB for input and
generates NDVI-based binary masks as segmentation targets.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from utils import (
    BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR,
    compute_ndvi, ndvi_to_mask, extract_rgb, NDVI_GREENERY_THRESHOLD,
)

# EuroSAT multispectral dataset URL (13-band GeoTIFF version)
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"
DEFAULT_DATA_DIR = Path("data")


def download_eurosat(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and extract the EuroSAT multispectral dataset.

    Returns:
        Path to the extracted dataset root directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "EuroSAT_MS.zip"
    extract_dir = data_dir / "EuroSAT_MS"

    if extract_dir.exists() and any(extract_dir.rglob("*.tif")):
        print(f"EuroSAT already downloaded at {extract_dir}")
        return extract_dir

    if not zip_path.exists():
        print(f"Downloading EuroSAT multispectral dataset...")
        print(f"URL: {EUROSAT_URL}")
        urllib.request.urlretrieve(EUROSAT_URL, zip_path, _download_progress)
        print("\nDownload complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"Extracted to {extract_dir}")

    return extract_dir


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)


class EuroSATGreeneryDataset(Dataset):
    """Dataset that loads EuroSAT multispectral .tif images and produces
    (RGB input, NDVI-based greenery mask) pairs for segmentation training.

    Each sample returns:
        rgb:  Tensor of shape (3, 64, 64) — normalized RGB channels
        mask: Tensor of shape (1, 64, 64) — binary greenery mask from NDVI
        meta: dict with 'filepath', 'class_name', 'ndvi_mean'
    """

    def __init__(self, root_dir: Path, ndvi_threshold: float = NDVI_GREENERY_THRESHOLD,
                 transform=None):
        self.root_dir = Path(root_dir)
        self.ndvi_threshold = ndvi_threshold
        self.transform = transform
        self.samples = []

        # Walk through all class directories and collect .tif file paths
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for tif_file in sorted(class_dir.glob("*.tif")):
                self.samples.append((tif_file, class_name))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No .tif files found in {self.root_dir}. "
                "Ensure the EuroSAT_MS dataset is correctly extracted."
            )

        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]

        # Read all 13 bands using rasterio
        with rasterio.open(filepath) as src:
            all_bands = src.read()  # Shape: (13, 64, 64)

        # Extract RGB and normalize to [0, 1]
        rgb = extract_rgb(all_bands)  # Shape: (3, 64, 64)

        # Compute NDVI from Red (Band 4) and NIR (Band 8)
        red = all_bands[BAND_RED].astype(np.float32)
        nir = all_bands[BAND_NIR].astype(np.float32)
        ndvi = compute_ndvi(red, nir)

        # Generate binary greenery mask
        mask = ndvi_to_mask(ndvi, self.ndvi_threshold)  # Shape: (64, 64)

        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb).float()              # (3, 64, 64)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # (1, 64, 64)

        if self.transform:
            rgb_tensor, mask_tensor = self.transform(rgb_tensor, mask_tensor)

        meta = {
            "filepath": str(filepath),
            "class_name": class_name,
            "ndvi_mean": float(ndvi.mean()),
        }

        return rgb_tensor, mask_tensor, meta


def get_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 0,
    ndvi_threshold: float = NDVI_GREENERY_THRESHOLD,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download EuroSAT and create train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset_root = download_eurosat(data_dir)
    dataset = EuroSATGreeneryDataset(dataset_root, ndvi_threshold=ndvi_threshold)

    total = len(dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split: {train_size} train / {val_size} val / {test_size} test")

    # Custom collate to handle the meta dict
    def collate_fn(batch):
        rgbs = torch.stack([b[0] for b in batch])
        masks = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return rgbs, masks, metas

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
