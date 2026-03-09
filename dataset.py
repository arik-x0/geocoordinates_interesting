"""
Shared EuroSAT dataset module for all geo_interesting POI models.

All models consume the same EuroSAT Multispectral (13-band Sentinel-2) dataset.
This file centralises dataset download, all Dataset classes, and DataLoader
factories for every submodel.

Structural submodels (POI detection):
  - EuroSATGreeneryDataset        get_vegetation_dataloaders()
  - ElevationPOIDataset           get_elevation_dataloaders()
  - EuroSATHousingDataset         get_housing_dataloaders()

Aesthetic submodels (beauty/perception):
  - FractalPatternDataset         get_fractal_dataloaders()
  - WaterGeometryDataset          get_water_dataloaders()
  - ColorHarmonyDataset           get_color_harmony_dataloaders()
  - SymmetryOrderDataset          get_symmetry_dataloaders()
  - ScaleSublimeDataset           get_sublime_dataloaders()
  - ComplexityBalanceDataset      get_complexity_dataloaders()

Sub-package utility functions are loaded at import time via importlib so that
each model's utils.py stays in its own directory without polluting sys.path.
"""

import importlib.util
import os
import sys
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

# Ensure the project root is on sys.path so sub-utils can import constants
_ROOT_STR = str(_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

from constants import (  # noqa: E402
    BAND_RED, BAND_GREEN, BAND_BLUE, BAND_NIR,
    NDVI_GREENERY_THRESHOLD,
    PRITHVI_BAND_INDICES, PRITHVI_MEAN, PRITHVI_STD,
)


# ── Utility loader ────────────────────────────────────────────────────────────

def _load_subutils(subdir: str):
    """Load <subdir>/utils.py as an isolated module (no sys.path side-effects)."""
    path = _ROOT / subdir / "utils.py"
    spec = importlib.util.spec_from_file_location(f"{subdir}_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_veg_utils   = _load_subutils("submodels/vegetation")
_elev_utils  = _load_subutils("submodels/elevation")
_house_utils = _load_subutils("submodels/housing")

# ── Prithvi band extractor ────────────────────────────────────────────────────

def extract_prithvi_bands(all_bands: np.ndarray) -> np.ndarray:
    """Extract and normalise the 6 HLS bands expected by Prithvi-EO-1.0-100M.

    Args:
        all_bands: (13, H, W) raw EuroSAT Sentinel-2 raster (DN values ×10000).

    Returns:
        (6, H, W) float32 tensor normalised by Prithvi's per-band mean/std.
        Channel order: [B02(Blue), B03(Green), B04(Red), B8A(NIR-N), B11(SWIR1), B12(SWIR2)]
    """
    bands = np.stack(
        [all_bands[i] for i in PRITHVI_BAND_INDICES], axis=0
    ).astype(np.float32)                                         # (6, H, W)
    mean = np.array(PRITHVI_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(PRITHVI_STD,  dtype=np.float32)[:, None, None]
    return (bands - mean) / std


# ── Vegetation helpers ────────────────────────────────────────────────────────
extract_rgb             = _elev_utils.extract_rgb
compute_ndvi            = _veg_utils.compute_ndvi
ndvi_to_mask            = _veg_utils.ndvi_to_mask

# ── Elevation helpers ─────────────────────────────────────────────────────────
normalize_channel       = _elev_utils.normalize_channel
compute_slope_aspect    = _elev_utils.compute_slope_aspect
generate_terrain_label  = _elev_utils.generate_terrain_label
generate_synthetic_dem  = _elev_utils.generate_synthetic_dem
download_srtm_tile      = _elev_utils.download_srtm_tile
read_srtm_hgt           = _elev_utils.read_srtm_hgt
extract_dem_for_bounds  = _elev_utils.extract_dem_for_bounds
get_srtm_tile_id        = _elev_utils.get_srtm_tile_id

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
    """Produces (prithvi_6band, NDVI mask, meta) triples for vegetation segmentation.

    Each sample:
        prithvi: Tensor (6, 64, 64) — Prithvi-normalised 6-band HLS input
        mask:    Tensor (1, 64, 64) — binary NDVI greenery mask
        meta:    dict — filepath, class_name, ndvi_mean
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

        prithvi = extract_prithvi_bands(all_bands)   # (6, 64, 64)
        red  = all_bands[BAND_RED].astype(np.float32)
        nir  = all_bands[BAND_NIR].astype(np.float32)
        ndvi = compute_ndvi(red, nir)
        mask = ndvi_to_mask(ndvi, self.ndvi_threshold)  # (64, 64)

        prithvi_tensor = torch.from_numpy(prithvi).float()
        mask_tensor    = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            prithvi_tensor, mask_tensor = self.transform(prithvi_tensor, mask_tensor)

        meta = {
            "filepath":   str(filepath),
            "class_name": class_name,
            "ndvi_mean":  float(ndvi.mean()),
        }
        return prithvi_tensor, mask_tensor, meta


class ElevationPOIDataset(Dataset):
    """Produces (9-channel input, terrain heatmap, meta) for terrain beauty detection.

    Pseudo-label: topographic ruggedness (local relief + slope variance +
    ridgeline curvature) — independent of water, based on SBE research.

    Each sample:
        input_tensor: Tensor (9, 64, 64) — 6 Prithvi bands + DEM + Slope + Aspect
                          ch 0-5: Prithvi-normalised HLS bands (passed to backbone)
                          ch 6:   DEM (normalised)
                          ch 7:   Slope (normalised)
                          ch 8:   Aspect (normalised 0-1)
        target:       Tensor (1, 64, 64) — terrain beauty heatmap
        meta:         dict — filepath, class_name, terrain_score,
                             max_slope, dem_source
    """

    def __init__(self, root_dir: Path, use_real_dem: bool = True):
        self.root_dir    = Path(root_dir)
        self.use_real_dem = use_real_dem
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
        prithvi = extract_prithvi_bands(all_bands)   # (6, 64, 64)

        import warnings
        dem = None
        dem_source = "synthetic"
        if self.use_real_dem:
            try:
                dem = get_dem_for_tile(str(filepath), self.srtm_cache)
            except Exception:
                dem = None
        if dem is None:
            warnings.warn(
                f"DEM unavailable for tile, using synthetic fallback: {filepath}",
                stacklevel=2,
            )
            dem = generate_synthetic_dem((h, w), seed=idx)
        else:
            dem_source = "real"

        slope, aspect = compute_slope_aspect(dem)
        terrain_heatmap = generate_terrain_label(dem, slope, aspect)

        dem_norm    = normalize_channel(dem)
        slope_norm  = normalize_channel(slope)
        aspect_norm = aspect / (2 * np.pi)

        input_9ch = np.concatenate([
            prithvi,
            dem_norm[np.newaxis, :, :],
            slope_norm[np.newaxis, :, :],
            aspect_norm[np.newaxis, :, :],
        ], axis=0)  # (9, 64, 64): ch0-5=Prithvi, ch6=DEM, ch7=Slope, ch8=Aspect

        input_tensor  = torch.from_numpy(input_9ch).float()
        target_tensor = torch.from_numpy(terrain_heatmap[np.newaxis, :, :]).float()

        meta = {
            "filepath":      str(filepath),
            "class_name":    class_name,
            "terrain_score": float(terrain_heatmap.max()),
            "max_slope":     float(slope.max()),
            "dem_source":    dem_source,
        }
        return input_tensor, target_tensor, meta


class EuroSATHousingDataset(Dataset):
    """Produces (prithvi_6band, structure label, meta) for housing density detection.

    Ground-truth labels are generated from NDBI + Sobel gradient — no manual
    annotations required.

    Each sample:
        prithvi: Tensor (6, 64, 64) — Prithvi-normalised 6-band HLS input
        label:   Tensor (1, 64, 64) — binary built-up / structure mask
        meta:    dict — filepath, class_name, housing_score, is_residential,
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

        prithvi       = extract_prithvi_bands(all_bands)     # (6, 64, 64)
        label         = generate_structure_label(all_bands)  # (64, 64)
        housing_score = float(label.mean())

        prithvi_tensor = torch.from_numpy(prithvi).float()
        label_tensor   = torch.from_numpy(label).unsqueeze(0).float()

        if self.transform:
            prithvi_tensor, label_tensor = self.transform(prithvi_tensor, label_tensor)

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
        return prithvi_tensor, label_tensor, meta


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
    data_dir:     Path  = DEFAULT_DATA_DIR,
    batch_size:   int   = 16,
    val_split:    float = 0.15,
    test_split:   float = 0.10,
    num_workers:  int   = 0,
    use_real_dem: bool  = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the terrain beauty elevation model."""
    root = download_eurosat(data_dir)
    dataset = ElevationPOIDataset(root, use_real_dem=use_real_dem)

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


# ── Generic aesthetic dataset base ────────────────────────────────────────────

class _AestheticDataset(Dataset):
    """Base for all 6 aesthetic submodels — reads EuroSAT, generates pseudo-labels.

    Subclasses implement _make_label(all_bands, idx) -> np.ndarray (H, W) in [0,1].
    Input tensor is always (6, 64, 64) Prithvi-normalised HLS bands.
    Target tensor is (1, 64, 64) soft heatmap.
    """

    def __init__(self, root_dir: Path, tag: str):
        self.root_dir = Path(root_dir)
        self.samples  = []
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
        print(f"[{tag:<14}] Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _make_label(self, all_bands: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        filepath, class_name = self.samples[idx]
        with rasterio.open(filepath) as src:
            all_bands = src.read()   # (13, 64, 64)

        prithvi = extract_prithvi_bands(all_bands)   # (6, 64, 64) float32
        label   = self._make_label(all_bands, idx)   # (64, 64)    float32 [0,1]

        meta = {
            "filepath":    str(filepath),
            "class_name":  class_name,
            "label_mean":  float(label.mean()),
        }
        return (
            torch.from_numpy(prithvi).float(),
            torch.from_numpy(label).unsqueeze(0).float(),
            meta,
        )


def _aesthetic_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        [b[2] for b in batch],
    )


def _aesthetic_loaders(dataset, batch_size, val_split, test_split, num_workers):
    return _split_and_load(
        dataset, batch_size, val_split, test_split, num_workers, _aesthetic_collate
    )


# ── 1. Fractal & Pattern ──────────────────────────────────────────────────────

class FractalPatternDataset(_AestheticDataset):
    """Pseudo-label: multi-scale Laplacian energy ratio peaking at mid-range detail."""

    def __init__(self, root_dir: Path):
        super().__init__(root_dir, "Fractal")
        self._utils = _load_subutils("submodels/fractal")

    def _make_label(self, all_bands, idx):
        return self._utils.generate_fractal_label(all_bands)


def get_fractal_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 32,
    val_split: float = 0.15, test_split: float = 0.10, num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the fractal pattern submodel."""
    root = download_eurosat(data_dir)
    return _aesthetic_loaders(FractalPatternDataset(root), batch_size, val_split, test_split, num_workers)


# ── 2. Water Presence & Geometry ──────────────────────────────────────────────

class WaterGeometryDataset(_AestheticDataset):
    """Pseudo-label: NDWI-based soft water mask with shoreline emphasis."""

    def __init__(self, root_dir: Path):
        super().__init__(root_dir, "Water")
        self._utils = _load_subutils("submodels/water")

    def _make_label(self, all_bands, idx):
        return self._utils.generate_water_label(all_bands)


def get_water_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 32,
    val_split: float = 0.15, test_split: float = 0.10, num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the water geometry submodel."""
    root = download_eurosat(data_dir)
    return _aesthetic_loaders(WaterGeometryDataset(root), batch_size, val_split, test_split, num_workers)


# ── 3. Color Harmony & Vegetation ─────────────────────────────────────────────

class ColorHarmonyDataset(_AestheticDataset):
    """Pseudo-label: HSV saturation blended with NDVI presence."""

    def __init__(self, root_dir: Path):
        super().__init__(root_dir, "ColorHarmony")
        self._utils = _load_subutils("submodels/color_harmony")

    def _make_label(self, all_bands, idx):
        return self._utils.generate_color_harmony_label(all_bands)


def get_color_harmony_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 32,
    val_split: float = 0.15, test_split: float = 0.10, num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the color harmony submodel."""
    root = download_eurosat(data_dir)
    return _aesthetic_loaders(ColorHarmonyDataset(root), batch_size, val_split, test_split, num_workers)


# ── 4. Symmetry & Geometric Order ─────────────────────────────────────────────

class SymmetryOrderDataset(_AestheticDataset):
    """Pseudo-label: local gradient orientation consistency (low variance = ordered)."""

    def __init__(self, root_dir: Path):
        super().__init__(root_dir, "Symmetry")
        self._utils = _load_subutils("submodels/symmetry")

    def _make_label(self, all_bands, idx):
        return self._utils.generate_symmetry_label(all_bands)


def get_symmetry_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 32,
    val_split: float = 0.15, test_split: float = 0.10, num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the symmetry/order submodel."""
    root = download_eurosat(data_dir)
    return _aesthetic_loaders(SymmetryOrderDataset(root), batch_size, val_split, test_split, num_workers)


# ── 5. Scale & Sublime ────────────────────────────────────────────────────────

class ScaleSublimeDataset(_AestheticDataset):
    """Pseudo-label: large-scale contrast (macro formations visible from above)."""

    def __init__(self, root_dir: Path):
        super().__init__(root_dir, "Sublime")
        self._utils = _load_subutils("submodels/sublime")

    def _make_label(self, all_bands, idx):
        return self._utils.generate_sublime_label(all_bands)


def get_sublime_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 32,
    val_split: float = 0.15, test_split: float = 0.10, num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the scale/sublime submodel."""
    root = download_eurosat(data_dir)
    return _aesthetic_loaders(ScaleSublimeDataset(root), batch_size, val_split, test_split, num_workers)


# ── 6. Complexity Balance ─────────────────────────────────────────────────────

class ComplexityBalanceDataset(_AestheticDataset):
    """Pseudo-label: local gradient entropy peaking at mid-range complexity."""

    def __init__(self, root_dir: Path):
        super().__init__(root_dir, "Complexity")
        self._utils = _load_subutils("submodels/complexity")

    def _make_label(self, all_bands, idx):
        return self._utils.generate_complexity_label(all_bands)


def get_complexity_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR, batch_size: int = 32,
    val_split: float = 0.15, test_split: float = 0.10, num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders for the complexity balance submodel."""
    root = download_eurosat(data_dir)
    return _aesthetic_loaders(ComplexityBalanceDataset(root), batch_size, val_split, test_split, num_workers)
