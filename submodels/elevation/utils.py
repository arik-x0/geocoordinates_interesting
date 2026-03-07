"""
Utility functions for topographic terrain beauty detection.

Pseudo-label strategy:
    Pure terrain ruggedness — rewards landscapes with high local relief,
    variable slope (rugged texture), and sharp ridgeline/cliff curvature.
    Based on Scenic Beauty Estimation (SBE) research showing topographic
    heterogeneity independently drives scenic quality regardless of water.

    High scores: mountain ridges, canyons, volcanic calderas, eroded mesas,
                 alpine passes, glacial cirques.
    Low scores:  flat plains, uniform hillsides, featureless terrain.

Kept for compatibility: SRTM download/read infrastructure, compute_slope_aspect,
normalize_channel, extract_rgb, generate_synthetic_dem, compute_ndwi,
detect_water_mask, generate_poi_heatmap (legacy).
"""

import math
import urllib.request
from pathlib import Path
from typing import Tuple, Optional

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import (
    gaussian_filter, uniform_filter,
    maximum_filter, minimum_filter,
)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import (  # noqa: E402
    BAND_RED, BAND_GREEN, BAND_BLUE,
    NDWI_WATER_THRESHOLD,
    POI_TOP_PERCENTILE,
    TERRAIN_RELIEF_WINDOW,
)

SRTM_CACHE_DIR = Path("data/srtm_cache")


# ── SRTM Elevation Data ───────────────────────────────────────────────────────

def get_srtm_tile_id(lat: float, lon: float) -> str:
    """Get SRTM tile ID string for a given latitude/longitude."""
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_int = int(abs(math.floor(lat)))
    lon_int = int(abs(math.floor(lon)))
    return f"{lat_prefix}{lat_int:02d}{lon_prefix}{lon_int:03d}"


def download_srtm_tile(lat: float, lon: float,
                       cache_dir: Path = SRTM_CACHE_DIR) -> Optional[Path]:
    """Download an SRTM 90m tile for the given coordinate."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_id = get_srtm_tile_id(lat, lon)
    hgt_filename = f"{tile_id}.hgt"
    cached_path = cache_dir / hgt_filename

    if cached_path.exists():
        return cached_path

    url = f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile_id[:3]}/{hgt_filename}"
    try:
        print(f"  Downloading SRTM tile {tile_id} ...")
        urllib.request.urlretrieve(url, cached_path)
        if cached_path.exists() and cached_path.stat().st_size > 0:
            return cached_path
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass
    return None


def read_srtm_hgt(hgt_path: Path) -> Tuple[np.ndarray, float, float]:
    """Read an SRTM .hgt file into a numpy elevation array."""
    file_size = hgt_path.stat().st_size
    if file_size == 3601 * 3601 * 2:
        dim = 3601
    elif file_size == 1201 * 1201 * 2:
        dim = 1201
    else:
        dim = int(math.sqrt(file_size / 2))

    data = np.fromfile(str(hgt_path), dtype=">i2")
    elevation = data.reshape((dim, dim)).astype(np.float32)
    elevation[elevation == -32768] = np.nan

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
    """Extract and resample a DEM subset matching geographic bounds."""
    west, south, east, north = target_bounds
    pixel_size = 1.0 / (dem_dim - 1)

    col_start = max(0, int((west - dem_lon) / pixel_size))
    col_end   = min(dem_dim, int((east - dem_lon) / pixel_size) + 1)
    row_start = max(0, int((dem_lat + 1 - north) / pixel_size))
    row_end   = min(dem_dim, int((dem_lat + 1 - south) / pixel_size) + 1)

    subset = full_dem[row_start:row_end, col_start:col_end]
    if subset.size == 0:
        return np.zeros(target_size, dtype=np.float32)

    from PIL import Image
    img = Image.fromarray(subset)
    img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def generate_synthetic_dem(shape: Tuple[int, int] = (64, 64),
                           seed: int = None) -> np.ndarray:
    """Generate synthetic high-relief elevation data for testing."""
    if seed is not None:
        np.random.seed(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    terrain = (
        200.0
        + 80.0 * np.sin(2 * np.pi * x / w * 1.5)
        + 60.0 * np.cos(2 * np.pi * y / h * 2.0)
        + 40.0 * np.sin(2 * np.pi * (x + y) / (w + h) * 3.0)
    )
    cliff_y = h // 3
    terrain[cliff_y:cliff_y + 3, :] += 120.0
    terrain[cliff_y + 3:, :] += 80.0
    for i in range(h):
        j = int(w * 0.6 + i * 0.3)
        if 0 <= j < w - 2:
            terrain[i, j:j + 2] += 100.0
    terrain += np.random.normal(0, 5.0, shape).astype(np.float32)
    return terrain.astype(np.float32)


# ── Terrain Analysis ──────────────────────────────────────────────────────────

def compute_slope_aspect(dem: np.ndarray,
                         cell_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (radians) from a DEM."""
    dem_clean = np.nan_to_num(dem, nan=0.0)
    dy, dx = np.gradient(dem_clean, cell_size)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_deg = np.degrees(slope_rad)
    aspect = np.arctan2(-dy, dx)
    aspect = np.mod(aspect, 2 * np.pi)
    return slope_deg.astype(np.float32), aspect.astype(np.float32)


def generate_terrain_label(
    dem: np.ndarray,
    slope: np.ndarray,
    aspect: Optional[np.ndarray] = None,
    window: int = TERRAIN_RELIEF_WINDOW,
) -> np.ndarray:
    """Topographic beauty heatmap based on terrain ruggedness.

    Three complementary terrain signals (all independent of water):

    1. Local relief (40%): max - min elevation in a local window.
       Mountains, canyons, and ridgelines produce high relief.

    2. Slope ruggedness (40%): local std of slope angles.
       Terrain with heterogeneous inclinations (cliffs mixed with ledges)
       scores high; uniform slopes (a single hillside) score low.

    3. Ridgeline/cliff curvature (20%): gradient magnitude of slope.
       Captures sharp terrain transitions — ridge crests, cliff tops,
       erosion gullies, and crater rims.

    Based on SBE (Scenic Beauty Estimation) research showing topographic
    heterogeneity is a primary independent predictor of scenic quality.

    Returns:
        (H, W) float32 label in [0, 1].
    """
    # -- 1. Local relief -------------------------------------------------
    dem_norm = normalize_channel(dem)
    local_max   = maximum_filter(dem_norm, size=window)
    local_min   = minimum_filter(dem_norm, size=window)
    local_relief = local_max - local_min          # already in [0, 1]

    # -- 2. Slope ruggedness (local std of slope) ------------------------
    slope_norm    = slope / 90.0                  # degrees → [0, 1]
    slope_mean    = uniform_filter(slope_norm, size=window)
    slope_sq_mean = uniform_filter(slope_norm ** 2, size=window)
    slope_var     = np.sqrt(np.clip(slope_sq_mean - slope_mean ** 2, 0.0, None))
    mx = slope_var.max()
    slope_rugged  = (slope_var / mx) if mx > 0 else slope_var

    # -- 3. Ridgeline curvature (slope gradient magnitude) ---------------
    dy_s, dx_s = np.gradient(slope_norm)
    curvature    = np.sqrt(dx_s ** 2 + dy_s ** 2)
    mx = curvature.max()
    curvature    = (curvature / mx) if mx > 0 else curvature

    label = 0.4 * local_relief + 0.4 * slope_rugged + 0.2 * curvature
    label = np.clip(label, 0.0, 1.0)
    return gaussian_filter(label, sigma=1.5).astype(np.float32)


def compute_terrain_score(heatmap: np.ndarray) -> float:
    """Aggregate terrain beauty score: mean of top-10% pixel intensities."""
    flat  = heatmap.flatten()
    top_k = max(1, int(len(flat) * POI_TOP_PERCENTILE))
    return float(np.partition(flat, -top_k)[-top_k:].mean())


# ── Legacy (kept for backward compatibility) ──────────────────────────────────

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    green = green.astype(np.float32)
    nir   = nir.astype(np.float32)
    denom = green + nir
    return np.where(denom > 0, (green - nir) / denom, 0.0).astype(np.float32)


def detect_water_mask(green: np.ndarray, nir: np.ndarray,
                      threshold: float = NDWI_WATER_THRESHOLD) -> np.ndarray:
    return (compute_ndwi(green, nir) >= threshold).astype(np.float32)


def generate_poi_heatmap(slope, water_mask, cliff_threshold=15.0,
                         proximity_sigma=5.0, aspect=None):
    """Legacy cliff-near-water heatmap (no longer used by ElevationPOIDataset)."""
    from scipy.ndimage import distance_transform_edt
    cliff_strength = np.clip((slope - cliff_threshold) / (90.0 - cliff_threshold), 0.0, 1.0)
    if water_mask.sum() > 0:
        dist = distance_transform_edt(1.0 - water_mask)
        water_proximity = np.exp(-0.5 * (dist / proximity_sigma) ** 2)
    else:
        water_proximity = np.zeros_like(slope)
    poi = cliff_strength * water_proximity
    poi = gaussian_filter(poi, sigma=1.0)
    if poi.max() > 0:
        poi = poi / poi.max()
    return poi.astype(np.float32)


# Kept name for old imports
compute_poi_score = compute_terrain_score


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_channel(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a single channel to [0, 1]."""
    arr = arr.astype(np.float32)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    warnings.warn(
        f"normalize_channel: zero-variance input (all values={vmin:.4f}), "
        f"returning zeros (shape={arr.shape}).",
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
    rgb = np.stack(
        [all_bands[BAND_RED], all_bands[BAND_GREEN], all_bands[BAND_BLUE]], axis=0
    )
    return normalize_rgb(rgb)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_terrain(
    rgb: np.ndarray,
    dem: np.ndarray,
    slope: np.ndarray,
    local_relief: np.ndarray,
    heatmap_true: np.ndarray,
    heatmap_pred: np.ndarray,
    terrain_score: float,
    save_path: str = None,
):
    """6-panel terrain beauty result: RGB | DEM | Slope | Relief | True | Predicted."""
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        ax.set_facecolor("#0d0d0d")

    axes[0, 0].imshow(np.clip(rgb, 0, 1))
    axes[0, 0].set_title("Satellite RGB", color="white")
    axes[0, 0].axis("off")

    im_dem = axes[0, 1].imshow(dem, cmap="terrain")
    axes[0, 1].set_title("DEM Elevation (m)", color="white")
    axes[0, 1].axis("off")
    plt.colorbar(im_dem, ax=axes[0, 1], fraction=0.046)

    im_slope = axes[0, 2].imshow(slope, cmap="hot_r", vmin=0, vmax=60)
    axes[0, 2].set_title("Slope (degrees)", color="white")
    axes[0, 2].axis("off")
    plt.colorbar(im_slope, ax=axes[0, 2], fraction=0.046)

    relief_cmap = mcolors.LinearSegmentedColormap.from_list(
        "relief", ["#0d0d0d", "#6c3483", "#e74c3c", "#f39c12"])
    axes[1, 0].imshow(local_relief, cmap=relief_cmap, vmin=0, vmax=1)
    axes[1, 0].set_title("Local Relief", color="white")
    axes[1, 0].axis("off")

    terrain_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ter", ["#0d0d0d", "#1a5276", "#c0392b", "#f9e79f"])
    axes[1, 1].imshow(heatmap_true, cmap=terrain_cmap, vmin=0, vmax=1)
    axes[1, 1].set_title("Ground Truth Terrain Label", color="white")
    axes[1, 1].axis("off")

    overlay = rgb.copy()
    pred = np.clip(heatmap_pred, 0, 1)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred * 0.6, 0, 1)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + pred * 0.2, 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] * (1 - pred * 0.4), 0, 1)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(
        f"Predicted Terrain Overlay (score: {terrain_score:.2f})", color="white")
    axes[1, 2].axis("off")

    plt.suptitle(
        f"Topographic Terrain Beauty — Score: {terrain_score:.3f}",
        fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


# Legacy visualization alias
def visualize_poi(rgb, dem, slope, water_mask, heatmap_true, heatmap_pred,
                  poi_score, save_path=None):
    """Legacy wrapper — calls visualize_terrain, ignores water_mask."""
    dem_norm = normalize_channel(dem)
    local_max  = maximum_filter(dem_norm, size=TERRAIN_RELIEF_WINDOW)
    local_min  = minimum_filter(dem_norm, size=TERRAIN_RELIEF_WINDOW)
    visualize_terrain(
        rgb=rgb, dem=dem, slope=slope,
        local_relief=local_max - local_min,
        heatmap_true=heatmap_true, heatmap_pred=heatmap_pred,
        terrain_score=poi_score, save_path=save_path,
    )


def visualize_poi_ranking(results: list, top_n: int = 10,
                          save_path: str = None):
    """3-row overview: RGB | terrain heatmap | DEM."""
    results_sorted = sorted(
        results, key=lambda x: x.get("terrain_score", x.get("poi_score", 0)),
        reverse=True)[:top_n]
    n = len(results_sorted)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ter", ["#0d0d0d", "#1a5276", "#c0392b", "#f9e79f"])

    for i, item in enumerate(results_sorted):
        rgb = item["rgb"]
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        score = item.get("terrain_score", item.get("poi_score", 0))

        axes[0, i].imshow(np.clip(rgb, 0, 1))
        axes[0, i].set_title(f"#{i+1} ({score:.2f})", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(item["heatmap_pred"], cmap=cmap, vmin=0, vmax=1)
        axes[1, i].axis("off")

        axes[2, i].imshow(item["dem"], cmap="terrain")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("Terrain Label", fontsize=10)
    axes[2, 0].set_ylabel("Elevation", fontsize=10)

    plt.suptitle("Top Terrain Beauty Locations (Ranked)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
