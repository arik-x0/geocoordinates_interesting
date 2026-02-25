"""
Utility functions for satellite greenery analysis.
NDVI computation, greenery scoring, and visualization helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path

from constants import (
    BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR,
    NDVI_GREENERY_THRESHOLD, GREENERY_THRESHOLD,
)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    Values range from -1 to 1. Higher values indicate denser vegetation.
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denominator = nir + red
    # Avoid division by zero
    ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
    return ndvi.astype(np.float32)


def ndvi_to_mask(ndvi: np.ndarray, threshold: float = NDVI_GREENERY_THRESHOLD) -> np.ndarray:
    """Convert NDVI array to binary greenery mask.

    Returns:
        Binary mask: 1 = greenery, 0 = desert/barren
    """
    return (ndvi >= threshold).astype(np.float32)


def compute_greenery_ratio(mask: np.ndarray) -> float:
    """Compute the fraction of greenery pixels in a segmentation mask."""
    return float(mask.sum()) / mask.size


def greenery_score_from_prediction(prediction: torch.Tensor, threshold: float = GREENERY_THRESHOLD) -> float:
    """Compute greenery ratio from a model's sigmoid output tensor."""
    binary = (prediction > threshold).float()
    return float(binary.sum().item()) / binary.numel()


def normalize_rgb(bands: np.ndarray) -> np.ndarray:
    """Normalize multispectral bands to 0-1 range for display.

    Args:
        bands: Array of shape (C, H, W) or (H, W, C)
    """
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0] if bands.ndim == 3 and bands.shape[0] <= 13 else bands.shape[-1]):
        band = bands[i] if bands.shape[0] <= 13 else bands[..., i]
        bmin, bmax = band.min(), band.max()
        if bmax > bmin:
            if bands.shape[0] <= 13:
                bands[i] = (band - bmin) / (bmax - bmin)
            else:
                bands[..., i] = (band - bmin) / (bmax - bmin)
    return bands


def extract_rgb(all_bands: np.ndarray) -> np.ndarray:
    """Extract and normalize RGB channels from 13-band Sentinel-2 data.

    Args:
        all_bands: Shape (13, H, W)

    Returns:
        RGB image of shape (3, H, W) normalized to [0, 1]
    """
    rgb = np.stack([all_bands[BAND_RED], all_bands[BAND_GREEN], all_bands[BAND_BLUE]], axis=0)
    return normalize_rgb(rgb)


def visualize_prediction(rgb: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray,
                         greenery_score: float, save_path: str = None):
    """Plot RGB image, ground truth mask, predicted mask, and overlay side by side.

    Args:
        rgb: Shape (3, H, W) or (H, W, 3), range [0, 1]
        mask_true: Shape (H, W), binary ground truth
        mask_pred: Shape (H, W), predicted probability or binary mask
        greenery_score: Float, greenery ratio for this image
        save_path: If provided, save the figure to this path
    """
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("Satellite RGB")
    axes[0].axis("off")

    green_cmap = mcolors.LinearSegmentedColormap.from_list("gd", ["#c2945e", "#2d6a1e"])
    axes[1].imshow(mask_true, cmap=green_cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth (NDVI)")
    axes[1].axis("off")

    axes[2].imshow(mask_pred, cmap=green_cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (green: {greenery_score:.1%})")
    axes[2].axis("off")

    # Overlay: green tint on greenery regions
    overlay = rgb.copy()
    pred_binary = (mask_pred > 0.5) if mask_pred.max() <= 1.0 else (mask_pred > 128)
    overlay[pred_binary] = overlay[pred_binary] * 0.5 + np.array([0.0, 0.7, 0.0]) * 0.5
    axes[3].imshow(np.clip(overlay, 0, 1))
    axes[3].set_title("Greenery Overlay")
    axes[3].axis("off")

    plt.suptitle(f"Greenery Ratio: {greenery_score:.1%}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_ranking(results: list, top_n: int = 10, save_path: str = None):
    """Visualize top-N greenery-ranked satellite images.

    Args:
        results: List of dicts with keys 'rgb', 'mask_pred', 'greenery_score', 'filename'
        top_n: Number of top results to show
        save_path: If provided, save the figure
    """
    results_sorted = sorted(results, key=lambda x: x["greenery_score"], reverse=True)[:top_n]
    n = len(results_sorted)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))

    if n == 1:
        axes = axes.reshape(2, 1)

    green_cmap = mcolors.LinearSegmentedColormap.from_list("gd", ["#c2945e", "#2d6a1e"])

    for i, item in enumerate(results_sorted):
        rgb = item["rgb"]
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))

        axes[0, i].imshow(np.clip(rgb, 0, 1))
        axes[0, i].set_title(f"#{i+1} ({item['greenery_score']:.0%})", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(item["mask_pred"], cmap=green_cmap, vmin=0, vmax=1)
        axes[1, i].axis("off")

    plt.suptitle("Top Greenery Areas (Ranked)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
