"""
Utility functions for water presence & geometry.

Pseudo-label strategy:
    NDWI-based soft water probability combined with a shoreline emphasis map.
    Open water scores high at the centre; the transition zone (shoreline) also
    scores high because it encodes geometric form — the shape of the water body.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BAND_GREEN, BAND_NIR, NDWI_WATER_THRESHOLD  # noqa: E402


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). High values indicate water."""
    green = green.astype(np.float32)
    nir   = nir.astype(np.float32)
    denom = green + nir
    return np.where(denom > 0, (green - nir) / denom, 0.0).astype(np.float32)


def generate_water_label(all_bands: np.ndarray) -> np.ndarray:
    """NDWI soft water heatmap with shoreline geometry emphasis.

    - Core water areas score high (probability from NDWI).
    - Shoreline pixels also score high — they encode the geometric shape of
      the water body, which is a key aesthetic trigger from above.

    Returns:
        (H, W) float32 label in [0, 1].
    """
    green = all_bands[BAND_GREEN].astype(np.float32)
    nir   = all_bands[BAND_NIR].astype(np.float32)
    ndwi  = compute_ndwi(green, nir)

    # Soft water probability: ramp from threshold to threshold+0.3
    water_prob = np.clip((ndwi - (NDWI_WATER_THRESHOLD - 0.1)) / 0.4, 0.0, 1.0)

    # Shoreline emphasis via bidirectional distance transform
    binary_water = ndwi > NDWI_WATER_THRESHOLD
    if binary_water.sum() > 0:
        dist_from = distance_transform_edt(1 - binary_water.astype(np.uint8))
        dist_to   = distance_transform_edt(binary_water.astype(np.uint8))
        shoreline = np.exp(-0.5 * (np.minimum(dist_from, dist_to) / 4.0) ** 2)
        label = 0.6 * water_prob + 0.4 * shoreline
    else:
        label = water_prob

    return gaussian_filter(label, sigma=1.0).astype(np.float32)


def compute_water_score(pred: np.ndarray, threshold: float = 0.4) -> float:
    """Fraction of pixels above the water probability threshold."""
    return float((pred > threshold).sum()) / pred.size


def visualize_water(rgb: np.ndarray, label_true: np.ndarray,
                    label_pred: np.ndarray, score: float,
                    save_path: str = None):
    """4-panel: RGB | True label | Prediction | Overlay."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list("wat", ["#c2945e", "#1a5276", "#aed6f1"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(label_true, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("True Water Label"); axes[1].axis("off")
    axes[2].imshow(label_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (score: {score:.3f})"); axes[2].axis("off")
    overlay = rgb.copy()
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + label_pred * 0.6, 0, 1)
    axes[3].imshow(overlay); axes[3].set_title("Water Overlay"); axes[3].axis("off")
    plt.suptitle(f"Water Presence & Geometry Score: {score:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
