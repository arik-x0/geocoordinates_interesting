"""
Utility functions for housing / built-up area detection.

Label generation:
  - NDBI (Normalized Difference Built-up Index) from Sentinel-2 SWIR and NIR
    bands provides a spectral proxy for built-up surfaces (rooftops, roads,
    pavements). NDBI > 0 indicates built-up material.
  - Gradient magnitude (Sobel) reinforces structural edges.
  Combined label: NDBI-positive regions intersected with strong gradients,
  morphologically closed to fill building footprints.

Scoring:
  - housing_score: fraction of predicted pixels classified as structure.
  - Low-density residential: HOUSING_DENSITY_MIN <= score <= HOUSING_DENSITY_MAX
    i.e. 5% to 20% of the image is built-up.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path
from scipy.ndimage import sobel, binary_closing, binary_dilation

from constants import (
    BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR, BAND_SWIR,
    HOUSING_DENSITY_MIN, HOUSING_DENSITY_MAX,
    NDBI_THRESHOLD, GRADIENT_WEIGHT, CLOSING_SIZE,
)


def compute_ndbi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Built-up Index.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Positive values indicate built-up areas (rooftops, roads, pavements).
    Vegetation has negative NDBI, water strongly negative.
    """
    nir  = nir.astype(np.float32)
    swir = swir.astype(np.float32)
    denom = swir + nir
    return np.where(denom > 0, (swir - nir) / denom, 0.0).astype(np.float32)


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute normalised Sobel gradient magnitude from a grayscale image."""
    gx = sobel(gray.astype(np.float32), axis=1)
    gy = sobel(gray.astype(np.float32), axis=0)
    magnitude = np.hypot(gx, gy)
    vmax = magnitude.max()
    return (magnitude / (vmax + 1e-8)).astype(np.float32)


def generate_structure_label(all_bands: np.ndarray,
                             ndbi_threshold: float = NDBI_THRESHOLD,
                             gradient_weight: float = GRADIENT_WEIGHT,
                             closing_size:    int   = CLOSING_SIZE) -> np.ndarray:
    """Generate a per-pixel built-up / structure label from 13-band Sentinel-2 data.

    Pipeline:
        1. NDBI from SWIR (band 11) and NIR (band 8): identifies built-up spectral signature.
        2. Sobel gradient on grayscale RGB: reinforces sharp man-made edges.
        3. Combined score = NDBI_positive + gradient_weight * gradient.
        4. Threshold at 0.5 for binary label.
        5. Morphological closing fills rectangular building interiors.

    Args:
        all_bands: Shape (13, H, W) raw Sentinel-2 values.

    Returns:
        Binary structure label of shape (H, W), dtype float32, values in {0, 1}.
    """
    nir  = all_bands[BAND_NIR].astype(np.float32)
    swir = all_bands[BAND_SWIR].astype(np.float32)
    ndbi = compute_ndbi(nir, swir)

    # Grayscale from RGB (luminance weights)
    r = all_bands[BAND_RED].astype(np.float32)
    g = all_bands[BAND_GREEN].astype(np.float32)
    b = all_bands[BAND_BLUE].astype(np.float32)
    rmax = r.max(); gmax = g.max(); bmax = b.max()
    r = r / (rmax + 1e-8); g = g / (gmax + 1e-8); b = b / (bmax + 1e-8)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    grad = compute_gradient_magnitude(gray)

    # Positive NDBI component (normalised to [0, 1])
    ndbi_pos = np.clip(ndbi - ndbi_threshold, 0, None)
    ndbi_max = ndbi_pos.max()
    ndbi_norm = ndbi_pos / (ndbi_max + 1e-8)

    # Combined score
    combined = ndbi_norm + gradient_weight * grad
    combined = combined / (combined.max() + 1e-8)

    # Binary threshold
    binary = (combined > 0.5).astype(bool)

    # Morphological closing to fill building footprints
    struct = np.ones((closing_size, closing_size), dtype=bool)
    binary = binary_closing(binary, structure=struct)

    return binary.astype(np.float32)


def extract_rgb(all_bands: np.ndarray) -> np.ndarray:
    """Extract and normalise RGB channels from 13-band data.

    Returns:
        Shape (3, H, W), dtype float32, range [0, 1].
    """
    rgb = np.stack([all_bands[BAND_RED],
                    all_bands[BAND_GREEN],
                    all_bands[BAND_BLUE]], axis=0).astype(np.float32)
    for i in range(3):
        vmin, vmax = rgb[i].min(), rgb[i].max()
        if vmax > vmin:
            rgb[i] = (rgb[i] - vmin) / (vmax - vmin)
    return rgb


def compute_housing_score(pred: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of pixels predicted as built-up structure.

    Args:
        pred: Shape (H, W), values in [0, 1].

    Returns:
        Float in [0, 1] — 0 = no structure, 1 = fully built-up.
    """
    return float((pred > threshold).sum()) / pred.size


def housing_score_from_tensor(pred: torch.Tensor, threshold: float = 0.5) -> float:
    """Same as compute_housing_score but accepts a (1, H, W) or (H, W) tensor."""
    binary = (pred > threshold).float()
    return float(binary.sum().item()) / binary.numel()


def is_low_density_residential(score: float,
                                min_density: float = HOUSING_DENSITY_MIN,
                                max_density: float = HOUSING_DENSITY_MAX) -> bool:
    """True if the housing density falls in the low-density residential range (5-20%)."""
    return min_density <= score <= max_density


def density_label(score: float) -> str:
    """Human-readable density category."""
    if score < HOUSING_DENSITY_MIN:
        return "undeveloped"
    elif score <= HOUSING_DENSITY_MAX:
        return "low-density residential"
    elif score <= 0.50:
        return "medium density"
    else:
        return "high density / industrial"


# ── Visualisation ────────────────────────────────────────────────────────────

def visualize_housing_detection(rgb: np.ndarray,
                                label_true: np.ndarray,
                                label_pred: np.ndarray,
                                housing_score: float,
                                save_path: str = None):
    """4-panel plot: RGB | Ground Truth | Prediction | Overlay.

    Args:
        rgb:          (3, H, W) or (H, W, 3), range [0, 1]
        label_true:   (H, W) binary ground-truth structure mask
        label_pred:   (H, W) predicted structure probability
        housing_score: fraction of predicted built-up pixels
        save_path:    if given, save figure instead of showing
    """
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 1)

    struct_cmap = mcolors.LinearSegmentedColormap.from_list(
        "struct", ["#1a1a2e", "#e94560"]   # dark navy → red-pink
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Satellite RGB")
    axes[0].axis("off")

    axes[1].imshow(label_true, cmap=struct_cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth (NDBI)")
    axes[1].axis("off")

    axes[2].imshow(label_pred, cmap=struct_cmap, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction ({housing_score:.1%} structure)")
    axes[2].axis("off")

    # Overlay: tint predicted structures red-pink on the RGB image
    overlay = rgb.copy()
    pred_binary = label_pred > 0.5
    overlay[pred_binary] = overlay[pred_binary] * 0.4 + np.array([0.9, 0.2, 0.2]) * 0.6
    axes[3].imshow(overlay)
    axes[3].set_title(f"Structure Overlay — {density_label(housing_score)}")
    axes[3].axis("off")

    plt.suptitle(
        f"Housing Density: {housing_score:.1%}  "
        f"({'IN range' if is_low_density_residential(housing_score) else 'OUT of range'} "
        f"{HOUSING_DENSITY_MIN:.0%}–{HOUSING_DENSITY_MAX:.0%})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_housing_ranking(results: list, top_n: int = 10,
                               save_path: str = None):
    """Top-N grid: satellite RGB (top row) and predicted structure mask (bottom row).

    Args:
        results:   list of dicts with keys 'rgb', 'label_pred', 'housing_score'
        top_n:     number of best low-density residential images to show
        save_path: if given, save figure
    """
    # Sort by closeness to the target density band centre (12.5%)
    target = (HOUSING_DENSITY_MIN + HOUSING_DENSITY_MAX) / 2
    in_range = [r for r in results if is_low_density_residential(r["housing_score"])]
    in_range.sort(key=lambda r: abs(r["housing_score"] - target))
    display = in_range[:top_n] if in_range else \
              sorted(results, key=lambda r: abs(r["housing_score"] - target))[:top_n]

    n = len(display)
    if n == 0:
        return

    struct_cmap = mcolors.LinearSegmentedColormap.from_list(
        "struct", ["#1a1a2e", "#e94560"]
    )

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, item in enumerate(display):
        rgb = item["rgb"]
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))

        axes[0, i].imshow(np.clip(rgb, 0, 1))
        axes[0, i].set_title(
            f"#{i+1} {item['housing_score']:.0%}\n{item['class_name']}", fontsize=8
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(item["label_pred"], cmap=struct_cmap, vmin=0, vmax=1)
        axes[1, i].axis("off")

    plt.suptitle(
        f"Low-Density Residential Areas ({HOUSING_DENSITY_MIN:.0%}–{HOUSING_DENSITY_MAX:.0%} built-up)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
