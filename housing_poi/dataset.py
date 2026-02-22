"""
EuroSAT Multispectral Dataset loader for low-density housing detection.

Reads 13-band Sentinel-2 .tif files and generates per-pixel built-up labels
using NDBI (Normalized Difference Built-up Index) combined with Sobel gradient
magnitude and morphological closing — providing a spectral + structural proxy
for building and pavement coverage without manual annotations.

EuroSAT classes most likely to contain low-density housing:
  Residential  — suburban and urban residential blocks
  Industrial   — warehouses, factories (included as built-up, filtered at score stage)
  Highway      — roads and pavements (linear structures)
Other classes (Forest, Pasture, SeaLake, etc.) are included as negative examples.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from utils import generate_structure_label, extract_rgb

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
        print("Downloading EuroSAT multispectral dataset...")
        print(f"URL: {EUROSAT_URL}")
        urllib.request.urlretrieve(EUROSAT_URL, zip_path, _progress)
        print("\nDownload complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"Extracted to {extract_dir}")

    return extract_dir


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)


class EuroSATHousingDataset(Dataset):
    """EuroSAT dataset producing (RGB, structure label) pairs for housing detection.

    Ground-truth labels are generated automatically from each image's spectral
    bands using NDBI and gradient magnitude — no manual pixel annotations needed.

    Each sample returns:
        rgb:   Tensor (3, 64, 64) — normalised RGB satellite image
        label: Tensor (1, 64, 64) — binary built-up / structure mask
        meta:  dict with 'filepath', 'class_name', 'housing_score',
               'is_residential', 'ndbi_mean'
    """

    # EuroSAT classes that are structurally built-up
    BUILT_UP_CLASSES = {"Residential", "Industrial", "Highway"}

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for tif_file in sorted(class_dir.glob("*.tif")):
                self.samples.append((tif_file, class_name))

        if not self.samples:
            raise FileNotFoundError(
                f"No .tif files found in {self.root_dir}. "
                "Ensure the EuroSAT_MS dataset is correctly extracted."
            )

        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]

        with rasterio.open(filepath) as src:
            all_bands = src.read()   # (13, 64, 64)

        # RGB input (3, 64, 64) normalised to [0, 1]
        rgb = extract_rgb(all_bands)

        # Per-pixel built-up label from NDBI + gradient
        label = generate_structure_label(all_bands)  # (64, 64), float32 {0, 1}

        housing_score = float(label.mean())

        rgb_tensor   = torch.from_numpy(rgb).float()                   # (3, 64, 64)
        label_tensor = torch.from_numpy(label).unsqueeze(0).float()    # (1, 64, 64)

        if self.transform:
            rgb_tensor, label_tensor = self.transform(rgb_tensor, label_tensor)

        # NDBI mean for diagnostics
        nir  = all_bands[7].astype(np.float32)
        swir = all_bands[10].astype(np.float32)
        denom = swir + nir
        ndbi_mean = float(
            np.where(denom > 0, (swir - nir) / denom, 0.0).mean()
        )

        meta = {
            "filepath":      str(filepath),
            "class_name":    class_name,
            "housing_score": housing_score,
            "is_residential": class_name in self.BUILT_UP_CLASSES,
            "ndbi_mean":     ndbi_mean,
        }

        return rgb_tensor, label_tensor, meta


def get_dataloaders(
    data_dir:   Path  = DEFAULT_DATA_DIR,
    batch_size: int   = 16,
    val_split:  float = 0.15,
    test_split: float = 0.10,
    num_workers: int  = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download EuroSAT and return train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset_root = download_eurosat(data_dir)
    dataset = EuroSATHousingDataset(dataset_root)

    total = len(dataset)
    test_size  = int(total * test_split)
    val_size   = int(total * val_split)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split: {train_size} train / {val_size} val / {test_size} test")

    def collate_fn(batch):
        rgbs   = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        metas  = [b[2] for b in batch]
        return rgbs, labels, metas

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
