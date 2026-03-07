"""
Utility functions for color harmony detection.

Pseudo-label strategy:
    Pure HSV saturation — rewards chromatic richness and spectral diversity
    across all visible wavelengths. This is intentionally kept free of NDVI
    so that the Vegetation model owns the plant-matter signal and ColorHarmony
    owns the chromatic/spectral signal. The two complement each other in the
    meta-aggregator without redundancy.

    High scores: vivid croplands in bloom, semi-arid ochre terrain, coastal
    turquoise water, autumn canopy colour.
    Low scores: concrete, snow, bare rock with uniform grey reflectance.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BAND_RED, BAND_GREEN  # noqa: E402


def _rgb_to_saturation(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel HSV saturation from normalised [0,1] RGB channels."""
    r, g, b = [np.clip(c.astype(np.float32), 0.0, 1.0) for c in (r, g, b)]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    return np.where(cmax > 0, delta / cmax, 0.0).astype(np.float32)


def _spectral_spread(all_bands: np.ndarray) -> np.ndarray:
    """Cross-band spectral diversity: std across all 13 Sentinel-2 channels.

    High spread = spectrally varied scene (colourful).
    Low spread  = spectrally flat scene (grey, snow, uniform rock).
    """
    stack = np.stack([all_bands[i].astype(np.float32) for i in range(all_bands.shape[0])], axis=0)
    # Normalise each channel to [0,1] before computing std so brightness
    # differences between bands don't dominate
    for i in range(stack.shape[0]):
        mn, mx = stack[i].min(), stack[i].max()
        if mx > mn:
            stack[i] = (stack[i] - mn) / (mx - mn)
    spread = stack.std(axis=0)
    mx = spread.max()
    return (spread / mx) if mx > 0 else np.zeros_like(spread)


def generate_color_harmony_label(all_bands: np.ndarray) -> np.ndarray:
    """Pure chromatic richness heatmap — no NDVI, no vegetation signal.

    Blends:
        - HSV saturation (60%) from RGB reflectance
        - Cross-band spectral spread (40%) across all 13 Sentinel-2 channels

    Returns:
        (H, W) float32 label in [0, 1].
    """
    def _norm(arr):
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    r = _norm(all_bands[BAND_RED])
    g = _norm(all_bands[BAND_GREEN])
    b = _norm(all_bands[2])           # Blue band index 2 in EuroSAT

    saturation = _rgb_to_saturation(r, g, b)
    spread     = _spectral_spread(all_bands)

    label = 0.6 * saturation + 0.4 * spread
    label = np.clip(label, 0.0, 1.0)
    return gaussian_filter(label, sigma=1.0).astype(np.float32)


def compute_color_score(pred: np.ndarray) -> float:
    """Mean of top-20% pixel values — higher means richer chromatic scene."""
    flat = pred.flatten()
    k = max(1, int(len(flat) * 0.20))
    return float(np.partition(flat, -k)[-k:].mean())


def visualize_color_harmony(rgb: np.ndarray, label_true: np.ndarray,
                            label_pred: np.ndarray, score: float,
                            save_path: str = None):
    """4-panel: RGB | True label | Prediction | Overlay."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "color", ["#1a1a2e", "#e74c3c", "#f39c12", "#f9e79f"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(label_true, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("True Color Label"); axes[1].axis("off")
    axes[2].imshow(label_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (score: {score:.3f})"); axes[2].axis("off")
    overlay = rgb.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + label_pred * 0.4, 0, 1)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + label_pred * 0.2, 0, 1)
    axes[3].imshow(overlay); axes[3].set_title("Chroma Overlay"); axes[3].axis("off")
    plt.suptitle(f"Color Harmony Score: {score:.3f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
