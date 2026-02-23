"""
Shared EuroSAT dataset module for all geo_interesting POI models.

All three models consume the same EuroSAT Multispectral (13-band Sentinel-2)
dataset. This file centralises:
  - Dataset download / extraction  (download_eurosat)
  - EuroSATGreeneryDataset         (vegetation_poi)
  - ElevationPOIDataset            (elevation_poi)
  - EuroSATHousingDataset          (housing_poi)
  - get_vegetation_dataloaders()
  - get_elevation_dataloaders()
  - get_housing_dataloaders()

Sub-package utility functions are loaded at import time via importlib so that
each model's utils.py stays in its own directory without polluting sys.path.
"""

import importlib.util
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

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent


# ── Utility loader ────────────────────────────────────────────────────────────

def _load_subutils(subdir: str):
    """Load <subdir>/utils.py as an isolated module (no sys.path side-effects)."""
    path = _ROOT / subdir / "utils.py"
    spec = importlib.util.spec_from_file_location(f"{subdir}_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_veg_utils   = _load_subutils("vegetation_poi")
_elev_utils  = _load_subutils("elevation_poi")
_house_utils = _load_subutils("housing_poi")

# ── Shared band indices (identical in all three utils) ────────────────────────
BAND_RED   = _elev_utils.BAND_RED    # Band 4 — 665 nm
BAND_GREEN = _elev_utils.BAND_GREEN  # Band 3 — 560 nm
BAND_BLUE  = _elev_utils.BAND_BLUE   # Band 2 — 490 nm
BAND_NIR   = _elev_utils.BAND_NIR    # Band 8 — 842 nm

# ── Vegetation helpers ────────────────────────────────────────────────────────
extract_rgb             = _elev_utils.extract_rgb
compute_ndvi            = _veg_utils.compute_ndvi
ndvi_to_mask            = _veg_utils.ndvi_to_mask
NDVI_GREENERY_THRESHOLD = _veg_utils.NDVI_GREENERY_THRESHOLD

# ── Elevation helpers ─────────────────────────────────────────────────────────
normalize_channel      = _elev_utils.normalize_channel
compute_slope_aspect   = _elev_utils.compute_slope_aspect
detect_water_mask      = _elev_utils.detect_water_mask
generate_poi_heatmap   = _elev_utils.generate_poi_heatmap
generate_synthetic_dem = _elev_utils.generate_synthetic_dem
download_srtm_tile     = _elev_utils.download_srtm_tile
read_srtm_hgt          = _elev_utils.read_srtm_hgt
extract_dem_for_bounds = _elev_utils.extract_dem_for_bounds
get_srtm_tile_id       = _elev_utils.get_srtm_tile_id
CLIFF_SLOPE_THRESHOLD  = _elev_utils.CLIFF_SLOPE_THRESHOLD
POI_PROXIMITY_SIGMA    = _elev_utils.POI_PROXIMITY_SIGMA
NDWI_WATER_THRESHOLD   = _elev_utils.NDWI_WATER_THRESHOLD

# ── Housing helpers ───────────────────────────────────────────────────────────
generate_structure_label = _house_utils.generate_structure_label


# ── EuroSAT download ──────────────────────────────────────────────────────────
EUROSAT_URL      = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"
DEFAULT_DATA_DIR = Path("data")


def download_eurosat(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and extract the EuroSAT multispectral dataset if not present.

    Returns:
        Path to the extracted EuroSAT_MS root directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path    = data_dir / "EuroSAT_MS.zip"
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
        pct      = min(100, downloaded * 100 // total_size)
        mb_done  = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)


# ── Geo utilities (used by ElevationPOIDataset) ───────────────────────────────

def get_tile_latlon_bounds(
    tif_path: str,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (west, south, east, north) in WGS84 degrees for a GeoTIFF."""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs    = src.crs

    if crs is None:
        return None

    if crs != CRS.from_epsg(4326):
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        west, south = transformer.transform(bounds.left,  bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
    else:
        west, south, east, north = (
            bounds.left, bounds.bottom, bounds.right, bounds.top
        )

    return (west, south, east, north)


def get_dem_for_tile(
    tif_path: str,
    srtm_cache: Dict[str, Tuple[np.ndarray, float, float, int]] = None,
    cache_dir: Path = Path("data/srtm_cache"),
) -> Optional[np.ndarray]:
    """Fetch and resample SRTM DEM data matching a EuroSAT tile.

    Returns DEM array of shape (64, 64), or None if unavailable.
    """
    if srtm_cache is None:
        srtm_cache = {}

    bounds = get_tile_latlon_bounds(tif_path)
    if bounds is None:
        return None

    west, south, east, north = bounds
    center_lat = (south + north) / 2
    center_lon = (west  + east)  / 2

    tile_id = get_srtm_tile_id(center_lat, center_lon)

    if tile_id not in srtm_cache:
        hgt_path = download_srtm_tile(center_lat, center_lon, cache_dir)
        if hgt_path is None:
            return None
        dem_data, lat_origin, lon_origin = read_srtm_hgt(hgt_path)
        srtm_cache[tile_id] = (dem_data, lat_origin, lon_origin, dem_data.shape[0])

    dem_data, lat_origin, lon_origin, dim = srtm_cache[tile_id]
    return extract_dem_for_bounds(dem_data, lat_origin, lon_origin, dim, bounds, (64, 64))


# ── Dataset classes ───────────────────────────────────────────────────────────

class EuroSATGreeneryDataset(Dataset):
    """Produces (RGB, NDVI mask, meta) triples for vegetation segmentation.

    Each sample:
        rgb:  Tensor (3, 64, 64) — normalised RGB
        mask: Tensor (1, 64, 64) — binary NDVI greenery mask
        meta: dict — filepath, class_name, ndvi_mean
    """

    def __init__(self, root_dir: Path,
                 ndvi_threshold: float = NDVI_GREENERY_THRESHOLD,
                 transform=None):
        self.root_dir       = Path(root_dir)
        self.ndvi_threshold = ndvi_threshold
        self.transform      = transform
        self.samples        = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for tif_file in sorted(class_dir.glob("*.tif")):
                self.samples.append((tif_file, class_dir.name))

        if not self.samples:
            raise FileNotFoundError(
                f"No .tif files found in {self.root_dir}. "
                "Ensure the EuroSAT_MS dataset is correctly extracted."
            )
        print(f"[Greenery]  Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]

        with rasterio.open(filepath) as src:
            all_bands = src.read()  # (13, 64, 64)

        rgb  = extract_rgb(all_bands)  # (3, 64, 64)
        red  = all_bands[BAND_RED].astype(np.float32)
        nir  = all_bands[BAND_NIR].astype(np.float32)
        ndvi = compute_ndvi(red, nir)
        mask = ndvi_to_mask(ndvi, self.ndvi_threshold)  # (64, 64)

        rgb_tensor  = torch.from_numpy(rgb).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            rgb_tensor, mask_tensor = self.transform(rgb_tensor, mask_tensor)

        meta = {
            "filepath":   str(filepath),
            "class_name": class_name,
            "ndvi_mean":  float(ndvi.mean()),
        }
        return rgb_tensor, mask_tensor, meta


class ElevationPOIDataset(Dataset):
    """Produces (6-channel input, POI heatmap, meta) for cliff-water detection.

    Each sample:
        input_tensor: Tensor (6, 64, 64) — RGB + DEM + Slope + Aspect
        target:       Tensor (1, 64, 64) — cliff-near-water POI heatmap
        meta:         dict — filepath, class_name, poi_score, has_water,
                             has_cliffs, water_fraction, max_slope
    """

    def __init__(
        self,
        root_dir: Path,
        cliff_threshold: float = CLIFF_SLOPE_THRESHOLD,
        water_threshold: float = NDWI_WATER_THRESHOLD,
        proximity_sigma: float = POI_PROXIMITY_SIGMA,
        use_real_dem:    bool  = True,
    ):
        self.root_dir       = Path(root_dir)
        self.cliff_threshold = cliff_threshold
        self.water_threshold = water_threshold
        self.proximity_sigma = proximity_sigma
        self.use_real_dem    = use_real_dem
        self.srtm_cache: Dict[str, Tuple[np.ndarray, float, float, int]] = {}
        self.samples = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for tif_file in sorted(class_dir.glob("*.tif")):
                self.samples.append((tif_file, class_dir.name))

        if not self.samples:
            raise FileNotFoundError(
                f"No .tif files found in {self.root_dir}. "
                "Ensure EuroSAT_MS is extracted correctly."
            )
        print(f"[Elevation] Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]

        with rasterio.open(filepath) as src:
            all_bands = src.read()  # (13, 64, 64)

        h, w = all_bands.shape[1], all_bands.shape[2]
        rgb  = extract_rgb(all_bands)  # (3, 64, 64)

        dem = None
        if self.use_real_dem:
            try:
                dem = get_dem_for_tile(str(filepath), self.srtm_cache)
            except Exception:
                dem = None
        if dem is None:
            dem = generate_synthetic_dem((h, w), seed=idx)

        slope, aspect = compute_slope_aspect(dem)

        green_band = all_bands[BAND_GREEN].astype(np.float32)
        nir_band   = all_bands[BAND_NIR].astype(np.float32)
        water_mask = detect_water_mask(green_band, nir_band, self.water_threshold)

        poi_heatmap = generate_poi_heatmap(
            slope=slope,
            water_mask=water_mask,
            cliff_threshold=self.cliff_threshold,
            proximity_sigma=self.proximity_sigma,
            aspect=aspect,
        )

        dem_norm    = normalize_channel(dem)
        slope_norm  = normalize_channel(slope)
        aspect_norm = aspect / (2 * np.pi)

        input_6ch = np.concatenate([
            rgb,
            dem_norm[np.newaxis, :, :],
            slope_norm[np.newaxis, :, :],
            aspect_norm[np.newaxis, :, :],
        ], axis=0)  # (6, 64, 64)

        input_tensor  = torch.from_numpy(input_6ch).float()
        target_tensor = torch.from_numpy(poi_heatmap[np.newaxis, :, :]).float()

        has_water  = bool(water_mask.sum() > 0)
        has_cliffs = bool((slope > self.cliff_threshold).sum() > 0)
        poi_score  = float(poi_heatmap.max()) if has_water and has_cliffs else 0.0

        meta = {
            "filepath":       str(filepath),
            "class_name":     class_name,
            "poi_score":      poi_score,
            "has_water":      has_water,
            "has_cliffs":     has_cliffs,
            "water_fraction": float(water_mask.mean()),
            "max_slope":      float(slope.max()),
        }
        return input_tensor, target_tensor, meta


class EuroSATHousingDataset(Dataset):
    """Produces (RGB, structure label, meta) for housing density detection.

    Ground-truth labels are generated from NDBI + Sobel gradient — no manual
    annotations required.

    Each sample:
        rgb:   Tensor (3, 64, 64) — normalised RGB
        label: Tensor (1, 64, 64) — binary built-up / structure mask
        meta:  dict — filepath, class_name, housing_score, is_residential,
                      ndbi_mean
    """

    BUILT_UP_CLASSES = {"Residential", "Industrial", "Highway"}

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for tif_file in sorted(class_dir.glob("*.tif")):
                self.samples.append((tif_file, class_dir.name))

        if not self.samples:
            raise FileNotFoundError(
                f"No .tif files found in {self.root_dir}. "
                "Ensure the EuroSAT_MS dataset is correctly extracted."
            )
        print(f"[Housing]   Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]

        with rasterio.open(filepath) as src:
            all_bands = src.read()  # (13, 64, 64)

        rgb           = extract_rgb(all_bands)
        label         = generate_structure_label(all_bands)  # (64, 64)
        housing_score = float(label.mean())

        rgb_tensor   = torch.from_numpy(rgb).float()
        label_tensor = torch.from_numpy(label).unsqueeze(0).float()

        if self.transform:
            rgb_tensor, label_tensor = self.transform(rgb_tensor, label_tensor)

        nir  = all_bands[7].astype(np.float32)
        swir = all_bands[10].astype(np.float32)
        denom = swir + nir
        ndbi_mean = float(
            np.where(denom > 0, (swir - nir) / denom, 0.0).mean()
        )

        meta = {
            "filepath":       str(filepath),
            "class_name":     class_name,
            "housing_score":  housing_score,
            "is_residential": class_name in self.BUILT_UP_CLASSES,
            "ndbi_mean":      ndbi_mean,
        }
        return rgb_tensor, label_tensor, meta


# ── DataLoader factories ──────────────────────────────────────────────────────

def _split_and_load(dataset, batch_size, val_split, test_split, num_workers, collate_fn):
    """Shared train/val/test random split + DataLoader construction."""
    total      = len(dataset)
    test_size  = int(total * test_split)
    val_size   = int(total * val_split)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split: {train_size} train / {val_size} val / {test_size} test")

    make = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    return make(train_set, True), make(val_set, False), make(test_set, False)


def get_vegetation_dataloaders(
    data_dir:       Path  = DEFAULT_DATA_DIR,
    batch_size:     int   = 32,
    val_split:      float = 0.15,
    test_split:     float = 0.10,
    num_workers:    int   = 0,
    ndvi_threshold: float = NDVI_GREENERY_THRESHOLD,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the vegetation segmentation model."""
    root = download_eurosat(data_dir)
    dataset = EuroSATGreeneryDataset(root, ndvi_threshold=ndvi_threshold)

    def collate_fn(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    return _split_and_load(dataset, batch_size, val_split, test_split, num_workers, collate_fn)


def get_elevation_dataloaders(
    data_dir:        Path  = DEFAULT_DATA_DIR,
    batch_size:      int   = 16,
    val_split:       float = 0.15,
    test_split:      float = 0.10,
    num_workers:     int   = 0,
    use_real_dem:    bool  = True,
    cliff_threshold: float = CLIFF_SLOPE_THRESHOLD,
    water_threshold: float = NDWI_WATER_THRESHOLD,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the elevation POI model."""
    root = download_eurosat(data_dir)
    dataset = ElevationPOIDataset(
        root,
        cliff_threshold=cliff_threshold,
        water_threshold=water_threshold,
        use_real_dem=use_real_dem,
    )

    def collate_fn(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    return _split_and_load(dataset, batch_size, val_split, test_split, num_workers, collate_fn)


def get_housing_dataloaders(
    data_dir:    Path  = DEFAULT_DATA_DIR,
    batch_size:  int   = 16,
    val_split:   float = 0.15,
    test_split:  float = 0.10,
    num_workers: int   = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the housing edge detection model."""
    root = download_eurosat(data_dir)
    dataset = EuroSATHousingDataset(root)

    def collate_fn(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    return _split_and_load(dataset, batch_size, val_split, test_split, num_workers, collate_fn)
