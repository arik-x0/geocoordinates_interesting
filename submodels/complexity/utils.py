"""
Utility functions for complexity balance detection.

Pseudo-label strategy:
    Optimal complexity sits at the mid-point of the order-chaos continuum
    (Berlyne 1974; fractals D≈1.3). We measure local gradient standard
    deviation (a proxy for information density), then score pixels via a
    Gaussian bell centred at COMPLEXITY_TARGET_LEVEL. Areas that are neither
    too uniform nor too chaotic score highest.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BAND_RED, BAND_GREEN, COMPLEXITY_TARGET_LEVEL  # noqa: E402


def _luminance(all_bands: np.ndarray) -> np.ndarray:
    r = all_bands[BAND_RED].astype(np.float32)
    g = all_bands[BAND_GREEN].astype(np.float32)
    lum = 0.5 * r + 0.5 * g
    mn, mx = lum.min(), lum.max()
    return (lum - mn) / (mx - mn) if mx > mn else np.zeros_like(lum)


def generate_complexity_label(all_bands: np.ndarray,
                               window: int = 8) -> np.ndarray:
    """Optimal-complexity heatmap via local gradient standard deviation.

    For each pixel we compute the local std of the image gradient magnitude
    in a small window, then pass it through a Gaussian bell centred at
    COMPLEXITY_TARGET_LEVEL. The result rewards mid-complexity scenes.

    Returns:
        (H, W) float32 label in [0, 1].
    """
    lum = _luminance(all_bands)
    dy, dx = np.gradient(lum)
    mag    = np.sqrt(dx**2 + dy**2)

    # Local std ≈ sqrt(E[x^2] - E[x]^2)  via uniform filter
    mag_sq  = uniform_filter(mag**2,  size=window)
    mag_avg = uniform_filter(mag,     size=window)
    local_std = np.sqrt(np.clip(mag_sq - mag_avg**2, 0.0, None))

    # Normalise to [0, 1]
    mx = local_std.max()
    if mx > 0:
        local_std = local_std / mx

    # Gaussian bell centred at target complexity level
    sigma_bell = 0.20
    label = np.exp(-0.5 * ((local_std - COMPLEXITY_TARGET_LEVEL) / sigma_bell) ** 2)

    return gaussian_filter(label.astype(np.float32), sigma=1.0)


def compute_complexity_score(pred: np.ndarray) -> float:
    """Mean prediction value — high means well-balanced complexity scene."""
    return float(pred.mean())


def visualize_complexity(rgb: np.ndarray, label_true: np.ndarray,
                         label_pred: np.ndarray, score: float,
                         save_path: str = None):
    """4-panel: RGB | True label | Prediction | Overlay."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cpx", ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(label_true, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("True Complexity Label"); axes[1].axis("off")
    axes[2].imshow(label_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (score: {score:.3f})"); axes[2].axis("off")
    overlay = rgb.copy()
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + label_pred * 0.5, 0, 1)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + label_pred * 0.2, 0, 1)
    axes[3].imshow(overlay); axes[3].set_title("Complexity Overlay"); axes[3].axis("off")
    plt.suptitle(f"Complexity Balance Score: {score:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
