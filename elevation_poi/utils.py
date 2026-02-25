"""
Utility functions for elevation-based POI detection.
SRTM DEM handling, slope/aspect computation, NDWI water detection,
POI heatmap generation, and visualization.
"""

import os
import zipfile
import urllib.request
import math
from pathlib import Path
from typing import Tuple, Optional

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import distance_transform_edt, gaussian_filter

from constants import (
    BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR,
    NDWI_WATER_THRESHOLD, CLIFF_SLOPE_THRESHOLD, POI_PROXIMITY_SIGMA,
    POI_TOP_PERCENTILE,
)

# ─── SRTM Configuration ─────────────────────────────────────────────────────
SRTM_BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"
SRTM_CACHE_DIR = Path("data/srtm_cache")


# ═══════════════════════════════════════════════════════════════════════════════
# SRTM Elevation Data
# ═══════════════════════════════════════════════════════════════════════════════

def get_srtm_tile_id(lat: float, lon: float) -> str:
    """Get SRTM tile ID for a given latitude/longitude.

    SRTM tiles are named by their SW corner: e.g., N45E006.
    """
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_int = int(abs(math.floor(lat)))
    lon_int = int(abs(math.floor(lon)))
    return f"{lat_prefix}{lat_int:02d}{lon_prefix}{lon_int:03d}"


def download_srtm_tile(lat: float, lon: float, cache_dir: Path = SRTM_CACHE_DIR) -> Optional[Path]:
    """Download an SRTM 90m tile for the given coordinate.

    Downloads the .hgt-format SRTM tile and caches it locally.
    Returns the path to the cached file, or None if unavailable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_id = get_srtm_tile_id(lat, lon)
    hgt_filename = f"{tile_id}.hgt"
    cached_path = cache_dir / hgt_filename

    if cached_path.exists():
        return cached_path

    # Try downloading from NASA/CGIAR SRTM mirrors
    zip_filename = f"{tile_id}.SRTMGL1.hgt.zip"
    urls_to_try = [
        f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile_id[:3]}/{hgt_filename}",
    ]

    for url in urls_to_try:
        try:
            print(f"  Downloading SRTM tile {tile_id} ...")
            urllib.request.urlretrieve(url, cached_path)
            if cached_path.exists() and cached_path.stat().st_size > 0:
                return cached_path
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    return None


def read_srtm_hgt(hgt_path: Path) -> Tuple[np.ndarray, float, float]:
    """Read an SRTM .hgt file into a numpy elevation array.

    SRTM .hgt files are raw 16-bit signed integer big-endian arrays.
    SRTM1 (30m): 3601x3601, SRTM3 (90m): 1201x1201.

    Returns:
        (elevation_array, lat_origin, lon_origin)
    """
    file_size = hgt_path.stat().st_size
    if file_size == 3601 * 3601 * 2:
        dim = 3601  # SRTM1 (1 arc-second / ~30m)
    elif file_size == 1201 * 1201 * 2:
        dim = 1201  # SRTM3 (3 arc-second / ~90m)
    else:
        # Try to determine from file size
        dim = int(math.sqrt(file_size / 2))

    data = np.fromfile(str(hgt_path), dtype=">i2")  # Big-endian 16-bit signed
    elevation = data.reshape((dim, dim)).astype(np.float32)

    # Replace void values (-32768) with NaN
    elevation[elevation == -32768] = np.nan

    # Parse lat/lon from filename (e.g., "N45E006.hgt")
    name = hgt_path.stem
    lat_sign = 1 if name[0] == "N" else -1
    lon_sign = 1 if name[3] == "E" else -1
    lat_origin = lat_sign * int(name[1:3])
    lon_origin = lon_sign * int(name[4:7])

    return elevation, float(lat_origin), float(lon_origin)


def extract_dem_for_bounds(
    full_dem: np.ndarray, dem_lat: float, dem_lon: float, dem_dim: int,
    target_bounds: Tuple[float, float, float, float],
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract and resample a DEM subset matching geographic bounds.

    Args:
        full_dem: Full SRTM tile array (dim x dim)
        dem_lat, dem_lon: SW corner of the SRTM tile (degrees)
        dem_dim: Dimension of the full DEM array
        target_bounds: (west, south, east, north) in degrees
        target_size: Output array size (H, W)

    Returns:
        Resampled DEM array of shape target_size
    """
    west, south, east, north = target_bounds
    pixel_size = 1.0 / (dem_dim - 1)  # degrees per pixel

    # Convert geo bounds to pixel indices (SRTM is stored north-to-south)
    col_start = max(0, int((west - dem_lon) / pixel_size))
    col_end = min(dem_dim, int((east - dem_lon) / pixel_size) + 1)
    row_start = max(0, int((dem_lat + 1 - north) / pixel_size))
    row_end = min(dem_dim, int((dem_lat + 1 - south) / pixel_size) + 1)

    subset = full_dem[row_start:row_end, col_start:col_end]

    if subset.size == 0:
        return np.zeros(target_size, dtype=np.float32)

    # Resample to target size using bilinear interpolation
    from PIL import Image
    img = Image.fromarray(subset)
    img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def generate_synthetic_dem(shape: Tuple[int, int] = (64, 64), seed: int = None) -> np.ndarray:
    """Generate synthetic elevation data with cliff-like features for testing.

    Creates a terrain with ridges, valleys, and sharp elevation changes.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Base terrain: combination of gradients and sinusoidal ridges
    terrain = (
        200.0                                          # Base elevation
        + 80.0 * np.sin(2 * np.pi * x / w * 1.5)     # E-W ridges
        + 60.0 * np.cos(2 * np.pi * y / h * 2.0)     # N-S ridges
        + 40.0 * np.sin(2 * np.pi * (x + y) / (w + h) * 3.0)  # Diagonal features
    )

    # Add sharp cliff-like features (step functions)
    cliff_y = h // 3
    terrain[cliff_y:cliff_y + 3, :] += 120.0  # Sharp horizontal cliff
    terrain[cliff_y + 3:, :] += 80.0

    # Add another diagonal cliff
    for i in range(h):
        j = int(w * 0.6 + i * 0.3)
        if 0 <= j < w - 2:
            terrain[i, j:j+2] += 100.0

    # Add noise for realism
    terrain += np.random.normal(0, 5.0, shape).astype(np.float32)

    return terrain.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_slope_aspect(dem: np.ndarray, cell_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (radians) from a DEM.

    Uses numpy gradient to compute partial derivatives, then:
        slope = arctan(sqrt(dz/dx^2 + dz/dy^2))
        aspect = arctan2(-dy, dx)

    Args:
        dem: 2D elevation array (H, W)
        cell_size: Pixel spacing in the same units as elevation

    Returns:
        (slope_degrees, aspect_radians) — both shape (H, W)
    """
    # Handle NaN values
    dem_clean = np.nan_to_num(dem, nan=0.0)

    # Compute gradients (dz/dy, dz/dx)
    dy, dx = np.gradient(dem_clean, cell_size)

    # Slope magnitude in degrees
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect: direction of steepest descent (0=North, pi/2=East, etc.)
    aspect = np.arctan2(-dy, dx)
    # Normalize to [0, 2*pi]
    aspect = np.mod(aspect, 2 * np.pi)

    return slope_deg.astype(np.float32), aspect.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index.

    NDWI = (Green - NIR) / (Green + NIR)
    Higher values indicate water presence.
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    denominator = green + nir
    ndwi = np.where(denominator > 0, (green - nir) / denominator, 0.0)
    return ndwi.astype(np.float32)


def detect_water_mask(green: np.ndarray, nir: np.ndarray,
                      threshold: float = NDWI_WATER_THRESHOLD) -> np.ndarray:
    """Detect water pixels from Sentinel-2 bands.

    Returns:
        Binary mask: 1 = water, 0 = land
    """
    ndwi = compute_ndwi(green, nir)
    return (ndwi >= threshold).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# POI Heatmap Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_poi_heatmap(
    slope: np.ndarray,
    water_mask: np.ndarray,
    cliff_threshold: float = CLIFF_SLOPE_THRESHOLD,
    proximity_sigma: float = POI_PROXIMITY_SIGMA,
    aspect: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate a POI heatmap highlighting cliffs near water bodies.

    The heatmap peaks where:
    1. Terrain is steep (slope > cliff_threshold) — cliff candidates
    2. Water is nearby — proximity measured via distance transform
    3. (Optional) The cliff faces toward the water — aspect alignment

    The score at each pixel:
        poi = cliff_strength * water_proximity * [aspect_alignment]

    Args:
        slope: Slope in degrees, shape (H, W)
        water_mask: Binary water mask, shape (H, W)
        cliff_threshold: Minimum slope (degrees) for cliff candidates
        proximity_sigma: Gaussian decay sigma for water proximity
        aspect: Optional aspect array (radians) for directional filtering

    Returns:
        POI heatmap, shape (H, W), values in [0, 1]
    """
    # 1. Cliff strength: normalized slope above threshold
    cliff_strength = np.clip((slope - cliff_threshold) / (90.0 - cliff_threshold), 0.0, 1.0)

    # 2. Water proximity: distance transform from water pixels, converted to score
    if water_mask.sum() > 0:
        # Distance from each non-water pixel to nearest water pixel
        distance_to_water = distance_transform_edt(1.0 - water_mask)
        # Gaussian decay: closer to water = higher score
        water_proximity = np.exp(-0.5 * (distance_to_water / proximity_sigma) ** 2)
    else:
        water_proximity = np.zeros_like(slope)

    # 3. Aspect alignment (optional): prefer cliffs that face toward water
    if aspect is not None and water_mask.sum() > 0:
        # Compute direction from each pixel to nearest water
        water_y, water_x = np.where(water_mask > 0)
        if len(water_y) > 0:
            ys, xs = np.mgrid[0:slope.shape[0], 0:slope.shape[1]]
            # For efficiency, approximate using the centroid of water pixels
            water_center_y = water_y.mean()
            water_center_x = water_x.mean()
            direction_to_water = np.arctan2(water_center_y - ys, water_center_x - xs)
            direction_to_water = np.mod(direction_to_water, 2 * np.pi)

            # Aspect alignment: cos similarity between cliff face direction and water direction
            angle_diff = np.abs(aspect - direction_to_water)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            aspect_alignment = np.cos(angle_diff)
            aspect_alignment = np.clip(aspect_alignment, 0.0, 1.0)
        else:
            aspect_alignment = np.ones_like(slope)
    else:
        aspect_alignment = np.ones_like(slope)

    # Combine all factors
    poi = cliff_strength * water_proximity * aspect_alignment

    # Smooth the heatmap slightly
    poi = gaussian_filter(poi, sigma=1.0)

    # Normalize to [0, 1]
    if poi.max() > 0:
        poi = poi / poi.max()

    return poi.astype(np.float32)


def compute_poi_score(heatmap: np.ndarray) -> float:
    """Compute an aggregate POI score for ranking images.

    Uses the mean of top-10% pixel intensities to capture peak POI strength
    rather than averaging over the whole image (which dilutes the signal).
    """
    flat = heatmap.flatten()
    if len(flat) == 0:
        return 0.0
    top_k = max(1, int(len(flat) * POI_TOP_PERCENTILE))
    top_values = np.partition(flat, -top_k)[-top_k:]
    return float(top_values.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_channel(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a single channel to [0, 1]."""
    arr = arr.astype(np.float32)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    warnings.warn(
        f"normalize_channel: zero-variance input (all values={vmin:.4f}), "
        f"returning zeros (shape={arr.shape}). "
        "This may indicate a flat DEM tile or missing data.",
        stacklevel=2,
    )
    return np.zeros_like(arr, dtype=np.float32)


def normalize_rgb(bands: np.ndarray) -> np.ndarray:
    """Normalize RGB bands (3, H, W) to [0, 1] per-channel."""
    result = np.zeros_like(bands, dtype=np.float32)
    for i in range(bands.shape[0]):
        result[i] = normalize_channel(bands[i])
    return result


def extract_rgb(all_bands: np.ndarray) -> np.ndarray:
    """Extract and normalize RGB from 13-band Sentinel-2 data. Shape: (3, H, W)."""
    rgb = np.stack([all_bands[BAND_RED], all_bands[BAND_GREEN], all_bands[BAND_BLUE]], axis=0)
    return normalize_rgb(rgb)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_poi(
    rgb: np.ndarray,
    dem: np.ndarray,
    slope: np.ndarray,
    water_mask: np.ndarray,
    heatmap_true: np.ndarray,
    heatmap_pred: np.ndarray,
    poi_score: float,
    save_path: str = None,
):
    """Visualize a 6-panel POI detection result.

    Panels: RGB | DEM Elevation | Slope | Water Mask | True Heatmap | Predicted Heatmap + Overlay
    """
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Inputs
    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title("Satellite RGB")
    axes[0, 0].axis("off")

    im_dem = axes[0, 1].imshow(dem, cmap="terrain")
    axes[0, 1].set_title("DEM Elevation (m)")
    axes[0, 1].axis("off")
    plt.colorbar(im_dem, ax=axes[0, 1], fraction=0.046)

    im_slope = axes[0, 2].imshow(slope, cmap="hot_r", vmin=0, vmax=60)
    axes[0, 2].set_title("Slope (degrees)")
    axes[0, 2].axis("off")
    plt.colorbar(im_slope, ax=axes[0, 2], fraction=0.046)

    # Row 2: Outputs
    water_cmap = mcolors.LinearSegmentedColormap.from_list("wb", ["#c2945e", "#1a5276"])
    axes[1, 0].imshow(water_mask, cmap=water_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title("Water Mask (NDWI)")
    axes[1, 0].axis("off")

    poi_cmap = mcolors.LinearSegmentedColormap.from_list("poi", ["#1a1a2e", "#e94560", "#ffff00"])
    axes[1, 1].imshow(heatmap_true, cmap=poi_cmap, vmin=0, vmax=1)
    axes[1, 1].set_title("Ground Truth POI Heatmap")
    axes[1, 1].axis("off")

    # Overlay prediction on RGB
    overlay = rgb.copy()
    pred_norm = np.clip(heatmap_pred, 0, 1)
    # Red-yellow overlay for high POI regions
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_norm * 0.6, 0, 1)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + pred_norm * 0.3, 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] * (1 - pred_norm * 0.5), 0, 1)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f"Predicted POI Overlay (score: {poi_score:.2f})")
    axes[1, 2].axis("off")

    plt.suptitle(f"Cliff-Water POI Detection — Score: {poi_score:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_poi_ranking(results: list, top_n: int = 10, save_path: str = None):
    """Visualize top-N POI-ranked satellite images with their heatmaps."""
    results_sorted = sorted(results, key=lambda x: x["poi_score"], reverse=True)[:top_n]
    n = len(results_sorted)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    poi_cmap = mcolors.LinearSegmentedColormap.from_list("poi", ["#1a1a2e", "#e94560", "#ffff00"])

    for i, item in enumerate(results_sorted):
        rgb = item["rgb"]
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))

        axes[0, i].imshow(np.clip(rgb, 0, 1))
        axes[0, i].set_title(f"#{i+1} ({item['poi_score']:.2f})", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(item["heatmap_pred"], cmap=poi_cmap, vmin=0, vmax=1)
        axes[1, i].axis("off")

        axes[2, i].imshow(item["dem"], cmap="terrain")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("POI Heatmap", fontsize=10)
    axes[2, 0].set_ylabel("Elevation", fontsize=10)

    plt.suptitle("Top Cliff-Water POI Locations (Ranked)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
