"""
Utility functions for fractal & pattern recognition.

Pseudo-label strategy:
    Multi-scale Laplacian energy ratio — scenes with mid-range self-similar
    detail (D ~ 1.4, river deltas, tree canopies, coastlines) score highest.
    The label is a Gaussian bell centered at the target richness level so both
    flat/featureless tiles and over-complex/chaotic tiles score low.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import laplace, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BAND_RED, BAND_GREEN, BAND_BLUE, FRACTAL_TARGET_SIGMA  # noqa: E402


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) float32 RGB to (H, W) luminance."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _extract_rgb(all_bands: np.ndarray) -> np.ndarray:
    r = all_bands[BAND_RED].astype(np.float32)
    g = all_bands[BAND_GREEN].astype(np.float32)
    b = all_bands[BAND_BLUE].astype(np.float32)
    rgb = np.stack([r, g, b], axis=0)
    for i in range(3):
        vmin, vmax = rgb[i].min(), rgb[i].max()
        if vmax > vmin:
            rgb[i] = (rgb[i] - vmin) / (vmax - vmin)
    return rgb


def _normalize(arr: np.ndarray) -> np.ndarray:
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr)


def generate_fractal_label(all_bands: np.ndarray) -> np.ndarray:
    """Multi-scale Laplacian energy ratio -> Gaussian bell at mid-range detail.

    Tiles with rich self-similar structure (river deltas, tree canopies,
    coastlines) score near 1. Flat tiles and chaotic over-textured tiles
    score near 0.

    Returns:
        (H, W) float32 label in [0, 1].
    """
    rgb  = _extract_rgb(all_bands)
    gray = _to_gray(rgb)

    # Laplacian energy at three scales
    lap_fine   = np.abs(laplace(gray))
    lap_medium = np.abs(laplace(gaussian_filter(gray, sigma=2.0)))
    lap_coarse = np.abs(laplace(gaussian_filter(gray, sigma=6.0)))

    # Per-pixel multi-scale richness: fine + medium detail relative to coarse
    richness = (lap_fine + 0.5 * lap_medium) / (lap_coarse + 1e-8)
    richness = _normalize(richness)

    # Gaussian bell centered at 0.5 — penalise extremes (D near 1.0 or 2.0)
    label = np.exp(-((richness - 0.5) ** 2) / (2 * FRACTAL_TARGET_SIGMA ** 2))
    return gaussian_filter(label, sigma=1.0).astype(np.float32)


def compute_fractal_score(pred: np.ndarray) -> float:
    """Mean of top-10% pixel values as the aggregate fractal richness score."""
    flat  = pred.flatten()
    top_k = max(1, int(len(flat) * 0.10))
    return float(np.partition(flat, -top_k)[-top_k:].mean())


def visualize_fractal(rgb: np.ndarray, label_true: np.ndarray,
                      label_pred: np.ndarray, score: float,
                      save_path: str = None):
    """4-panel: RGB | True label | Prediction | Overlay."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list("frac", ["#0d0221", "#ff6b6b", "#ffd93d"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(label_true, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("True Fractal Label"); axes[1].axis("off")
    axes[2].imshow(label_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (score: {score:.3f})"); axes[2].axis("off")
    overlay = rgb.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + label_pred * 0.5, 0, 1)
    axes[3].imshow(overlay); axes[3].set_title("Fractal Overlay"); axes[3].axis("off")
    plt.suptitle(f"Fractal & Pattern Score: {score:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
