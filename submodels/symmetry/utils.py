"""
Utility functions for symmetry & geometric order detection.

Pseudo-label strategy:
    Local gradient circular variance measures how consistent the dominant
    orientation is within each neighbourhood. Low circular variance = high
    order (agricultural fields, terraces, salt flats). The label is inverted
    so that maximally ordered regions score high.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BAND_RED, BAND_GREEN, SYMMETRY_LOCAL_SIZE  # noqa: E402


def _luminance(all_bands: np.ndarray) -> np.ndarray:
    """Approximate luminance from Red + Green channels."""
    r = all_bands[BAND_RED].astype(np.float32)
    g = all_bands[BAND_GREEN].astype(np.float32)
    lum = 0.5 * r + 0.5 * g
    mn, mx = lum.min(), lum.max()
    return (lum - mn) / (mx - mn) if mx > mn else np.zeros_like(lum)


def _gradient_orientation(lum: np.ndarray) -> np.ndarray:
    """Per-pixel gradient orientation in [0, pi) using arctan2."""
    dy, dx = np.gradient(lum)
    # Use doubled-angle trick for orientation (direction-invariant)
    theta = np.arctan2(dy, dx)           # in (-pi, pi]
    return theta.astype(np.float32)


def generate_symmetry_label(all_bands: np.ndarray) -> np.ndarray:
    """Geometric order heatmap based on local gradient coherence.

    For each pixel, the mean resultant length R of gradient orientations in a
    local window is computed via circular statistics. R near 1 = all gradients
    point the same way (ordered). R near 0 = isotropic (chaotic).

    Returns:
        (H, W) float32 label in [0, 1] where 1 = maximally ordered.
    """
    lum   = _luminance(all_bands)
    theta = _gradient_orientation(lum)

    # Circular mean resultant length via local averaging of unit vectors
    cos2  = np.cos(2 * theta)
    sin2  = np.sin(2 * theta)
    sz    = SYMMETRY_LOCAL_SIZE

    mean_cos = uniform_filter(cos2, size=sz)
    mean_sin = uniform_filter(sin2, size=sz)
    R         = np.sqrt(mean_cos**2 + mean_sin**2)   # in [0, 1]

    # Also reward low total gradient magnitude (flat, uniform areas excluded)
    mag = np.sqrt(np.gradient(lum)[0]**2 + np.gradient(lum)[1]**2)
    mag_norm = np.clip(mag / (mag.max() + 1e-8), 0.0, 1.0)

    # Ordered + non-trivial structure: high R AND non-zero gradient
    label = R * np.clip(mag_norm * 4, 0.0, 1.0)
    label = np.clip(label, 0.0, 1.0)
    return gaussian_filter(label, sigma=1.5).astype(np.float32)


def compute_symmetry_score(pred: np.ndarray) -> float:
    """Mean of top-20% pixel values — higher = more geometric order."""
    flat = pred.flatten()
    k = max(1, int(len(flat) * 0.20))
    return float(np.partition(flat, -k)[-k:].mean())


def visualize_symmetry(rgb: np.ndarray, label_true: np.ndarray,
                       label_pred: np.ndarray, score: float,
                       save_path: str = None):
    """4-panel: RGB | True label | Prediction | Overlay."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sym", ["#1a1a2e", "#8e44ad", "#f39c12"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(label_true, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("True Symmetry Label"); axes[1].axis("off")
    axes[2].imshow(label_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (score: {score:.3f})"); axes[2].axis("off")
    overlay = rgb.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + label_pred * 0.3, 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + label_pred * 0.5, 0, 1)
    axes[3].imshow(overlay); axes[3].set_title("Order Overlay"); axes[3].axis("off")
    plt.suptitle(f"Symmetry & Geometric Order Score: {score:.3f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
