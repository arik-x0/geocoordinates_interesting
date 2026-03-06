"""
Utility functions for scale & sublime detection.

Pseudo-label strategy:
    The sublime emerges from a large contrast between fine-grained texture and
    macro-scale structure. We measure the deviation of each pixel from a heavily
    Gaussian-smoothed version of itself (SUBLIME_COARSE_SIGMA), which isolates
    large-scale tonal contrast.  Geological formations, mountain ridges, and
    canyon edges produce the highest scores.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BAND_RED, BAND_GREEN, BAND_NIR, SUBLIME_COARSE_SIGMA  # noqa: E402


def _luminance(all_bands: np.ndarray) -> np.ndarray:
    r = all_bands[BAND_RED].astype(np.float32)
    g = all_bands[BAND_GREEN].astype(np.float32)
    n = all_bands[BAND_NIR].astype(np.float32)
    lum = 0.4 * r + 0.4 * g + 0.2 * n
    mn, mx = lum.min(), lum.max()
    return (lum - mn) / (mx - mn) if mx > mn else np.zeros_like(lum)


def generate_sublime_label(all_bands: np.ndarray) -> np.ndarray:
    """Large-scale tonal contrast heatmap.

    Computes deviation of luminance from its heavily smoothed counterpart.
    High deviation = strong large-scale structure (cliffs, ridges, coast).

    Returns:
        (H, W) float32 label in [0, 1].
    """
    lum    = _luminance(all_bands)
    coarse = gaussian_filter(lum, sigma=SUBLIME_COARSE_SIGMA)
    detail = gaussian_filter(lum, sigma=1.0)

    # Absolute deviation from coarse structure
    deviation = np.abs(detail - coarse)

    # Also add medium-scale edge signal (coarse - very-coarse)
    very_coarse = gaussian_filter(lum, sigma=SUBLIME_COARSE_SIGMA * 2)
    macro_edge  = np.abs(coarse - very_coarse)

    label = 0.6 * deviation + 0.4 * macro_edge
    mx = label.max()
    if mx > 0:
        label = label / mx
    return gaussian_filter(label.astype(np.float32), sigma=1.5)


def compute_sublime_score(pred: np.ndarray) -> float:
    """Mean of top-20% pixels — higher means grander large-scale contrast."""
    flat = pred.flatten()
    k = max(1, int(len(flat) * 0.20))
    return float(np.partition(flat, -k)[-k:].mean())


def visualize_sublime(rgb: np.ndarray, label_true: np.ndarray,
                      label_pred: np.ndarray, score: float,
                      save_path: str = None):
    """4-panel: RGB | True label | Prediction | Overlay."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sub", ["#0d0d0d", "#c0392b", "#f39c12", "#f9e79f"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(label_true, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("True Sublime Label"); axes[1].axis("off")
    axes[2].imshow(label_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (score: {score:.3f})"); axes[2].axis("off")
    overlay = rgb.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + label_pred * 0.5, 0, 1)
    axes[3].imshow(overlay); axes[3].set_title("Sublime Overlay"); axes[3].axis("off")
    plt.suptitle(f"Scale & Sublime Score: {score:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
