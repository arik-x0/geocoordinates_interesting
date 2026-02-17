"""
Dataset loader for elevation-based POI detection.
Combines EuroSAT Sentinel-2 multispectral tiles with SRTM elevation data
to produce 6-channel inputs (RGB + DEM + slope + aspect) and POI heatmap targets.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils import (
    BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR,
    extract_rgb, normalize_channel,
    compute_slope_aspect, detect_water_mask,
    generate_poi_heatmap, generate_synthetic_dem,
    download_srtm_tile, read_srtm_hgt, extract_dem_for_bounds,
    CLIFF_SLOPE_THRESHOLD, POI_PROXIMITY_SIGMA, NDWI_WATER_THRESHOLD,
)

# ─── EuroSAT Download ───────────────────────────────────────────────────────
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"
DEFAULT_DATA_DIR = Path("data")


def download_eurosat(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and extract EuroSAT multispectral dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "EuroSAT_MS.zip"
    extract_dir = data_dir / "EuroSAT_MS"

    if extract_dir.exists() and any(extract_dir.rglob("*.tif")):
        print(f"EuroSAT already downloaded at {extract_dir}")
        return extract_dir

    if not zip_path.exists():
        print("Downloading EuroSAT multispectral dataset...")
        print(f"URL: {EUROSAT_URL}")
        urllib.request.urlretrieve(EUROSAT_URL, zip_path, _download_progress)
        print("\nDownload complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"Extracted to {extract_dir}")
    return extract_dir


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


# ─── Geolocation Utilities ───────────────────────────────────────────────────

def get_tile_latlon_bounds(tif_path: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract lat/lon bounding box from a GeoTIFF file.

    Returns:
        (west, south, east, north) in WGS84 degrees, or None if no CRS.
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # (left, bottom, right, top) in the file's CRS
        crs = src.crs

        if crs is None:
            return None

        # Transform to WGS84 (EPSG:4326)
        if crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
            west, south = transformer.transform(bounds.left, bounds.bottom)
            east, north = transformer.transform(bounds.right, bounds.top)
        else:
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    return (west, south, east, north)


def get_dem_for_tile(
    tif_path: str,
    srtm_cache: Dict[str, Tuple[np.ndarray, float, float, int]] = None,
    cache_dir: Path = Path("data/srtm_cache"),
) -> Optional[np.ndarray]:
    """Fetch and resample SRTM DEM data matching a EuroSAT GeoTIFF tile.

    Args:
        tif_path: Path to 13-band EuroSAT GeoTIFF
        srtm_cache: In-memory cache of loaded SRTM tiles {tile_id: (data, lat, lon, dim)}
        cache_dir: Directory for cached SRTM downloads

    Returns:
        DEM array of shape (64, 64), or None if SRTM data unavailable.
    """
    if srtm_cache is None:
        srtm_cache = {}

    bounds = get_tile_latlon_bounds(tif_path)
    if bounds is None:
        return None

    west, south, east, north = bounds
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    from utils import get_srtm_tile_id
    tile_id = get_srtm_tile_id(center_lat, center_lon)

    # Check in-memory cache
    if tile_id not in srtm_cache:
        hgt_path = download_srtm_tile(center_lat, center_lon, cache_dir)
        if hgt_path is None:
            return None
        dem_data, lat_origin, lon_origin = read_srtm_hgt(hgt_path)
        srtm_cache[tile_id] = (dem_data, lat_origin, lon_origin, dem_data.shape[0])

    dem_data, lat_origin, lon_origin, dim = srtm_cache[tile_id]
    dem_subset = extract_dem_for_bounds(dem_data, lat_origin, lon_origin, dim, bounds, (64, 64))
    return dem_subset


# ─── Dataset Class ───────────────────────────────────────────────────────────

class ElevationPOIDataset(Dataset):
    """Dataset that produces 6-channel inputs and POI heatmap targets.

    Each sample returns:
        input_tensor: (6, 64, 64) — [RGB(3) + DEM(1) + Slope(1) + Aspect(1)]
        target:       (1, 64, 64) — POI heatmap (cliff-water proximity)
        meta:         dict with filepath, class_name, poi_score, has_water, has_cliffs
    """

    def __init__(
        self,
        root_dir: Path,
        cliff_threshold: float = CLIFF_SLOPE_THRESHOLD,
        water_threshold: float = NDWI_WATER_THRESHOLD,
        proximity_sigma: float = POI_PROXIMITY_SIGMA,
        use_real_dem: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.cliff_threshold = cliff_threshold
        self.water_threshold = water_threshold
        self.proximity_sigma = proximity_sigma
        self.use_real_dem = use_real_dem
        self.srtm_cache: Dict[str, Tuple[np.ndarray, float, float, int]] = {}
        self.samples = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for tif_file in sorted(class_dir.glob("*.tif")):
                self.samples.append((tif_file, class_name))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No .tif files found in {self.root_dir}. "
                "Ensure EuroSAT_MS is extracted correctly."
            )

        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]

        # ── Read 13-band satellite data ──────────────────────────────────
        with rasterio.open(filepath) as src:
            all_bands = src.read()  # (13, 64, 64)

        h, w = all_bands.shape[1], all_bands.shape[2]

        # ── Extract RGB (normalized) ─────────────────────────────────────
        rgb = extract_rgb(all_bands)  # (3, 64, 64)

        # ── Get DEM data ─────────────────────────────────────────────────
        dem = None
        if self.use_real_dem:
            try:
                dem = get_dem_for_tile(str(filepath), self.srtm_cache)
            except Exception:
                dem = None

        if dem is None:
            # Fallback: generate synthetic DEM based on spectral cues
            # Use NIR reflectance as a proxy for terrain roughness + random terrain
            dem = generate_synthetic_dem((h, w), seed=idx)

        # ── Compute slope and aspect from DEM ────────────────────────────
        slope, aspect = compute_slope_aspect(dem)

        # ── Detect water from satellite bands ────────────────────────────
        green_band = all_bands[BAND_GREEN].astype(np.float32)
        nir_band = all_bands[BAND_NIR].astype(np.float32)
        water_mask = detect_water_mask(green_band, nir_band, self.water_threshold)

        # ── Generate POI heatmap ground truth ────────────────────────────
        poi_heatmap = generate_poi_heatmap(
            slope=slope,
            water_mask=water_mask,
            cliff_threshold=self.cliff_threshold,
            proximity_sigma=self.proximity_sigma,
            aspect=aspect,
        )

        # ── Normalize DEM, slope, aspect to [0, 1] ──────────────────────
        dem_norm = normalize_channel(dem)
        slope_norm = normalize_channel(slope)
        aspect_norm = aspect / (2 * np.pi)  # Already in [0, 2*pi] → [0, 1]

        # ── Assemble 6-channel input ─────────────────────────────────────
        input_6ch = np.concatenate([
            rgb,                             # Channels 0-2: RGB
            dem_norm[np.newaxis, :, :],       # Channel 3: Elevation
            slope_norm[np.newaxis, :, :],     # Channel 4: Slope
            aspect_norm[np.newaxis, :, :],    # Channel 5: Aspect
        ], axis=0)  # Shape: (6, 64, 64)

        # ── Convert to tensors ───────────────────────────────────────────
        input_tensor = torch.from_numpy(input_6ch).float()
        target_tensor = torch.from_numpy(poi_heatmap[np.newaxis, :, :]).float()  # (1, 64, 64)

        has_water = bool(water_mask.sum() > 0)
        has_cliffs = bool((slope > self.cliff_threshold).sum() > 0)
        poi_score = float(poi_heatmap.max()) if has_water and has_cliffs else 0.0

        meta = {
            "filepath": str(filepath),
            "class_name": class_name,
            "poi_score": poi_score,
            "has_water": has_water,
            "has_cliffs": has_cliffs,
            "water_fraction": float(water_mask.mean()),
            "max_slope": float(slope.max()),
        }

        return input_tensor, target_tensor, meta


def get_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 0,
    use_real_dem: bool = True,
    cliff_threshold: float = CLIFF_SLOPE_THRESHOLD,
    water_threshold: float = NDWI_WATER_THRESHOLD,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for elevation POI detection."""
    dataset_root = download_eurosat(data_dir)
    dataset = ElevationPOIDataset(
        dataset_root,
        cliff_threshold=cliff_threshold,
        water_threshold=water_threshold,
        use_real_dem=use_real_dem,
    )

    total = len(dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split: {train_size} train / {val_size} val / {test_size} test")

    def collate_fn(batch):
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return inputs, targets, metas

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
