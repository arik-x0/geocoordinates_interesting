"""
Sentinel-2 band indices and model thresholds shared across all three POI pipelines.

All three models consume EuroSAT 13-band Sentinel-2 tiles (.tif).
Band indices are 0-based into the 13-channel raster stack.
"""

# ── Sentinel-2 band indices (0-based, 13-band EuroSAT .tif) ─────────────────
BAND_RED   = 3   # Band 4  — 665 nm
BAND_GREEN = 2   # Band 3  — 560 nm
BAND_BLUE  = 1   # Band 2  — 490 nm
BAND_NIR   = 7   # Band 8  — 842 nm
BAND_SWIR  = 10  # Band 11 — 1610 nm

# ── Vegetation thresholds ────────────────────────────────────────────────────
NDVI_GREENERY_THRESHOLD = 0.3   # NDVI above this → greenery pixel

# ── Elevation / cliff-water thresholds ───────────────────────────────────────
NDWI_WATER_THRESHOLD  = 0.3    # NDWI above this → water pixel
CLIFF_SLOPE_THRESHOLD = 15.0   # slope (degrees) above this → cliff candidate
POI_PROXIMITY_SIGMA   = 5.0    # Gaussian decay sigma for water proximity (pixels)

# ── Housing density thresholds ───────────────────────────────────────────────
HOUSING_DENSITY_MIN = 0.05   # at least 5% built-up coverage
HOUSING_DENSITY_MAX = 0.20   # at most 20% built-up coverage (not overly urban)
NDBI_THRESHOLD      = 0.0    # NDBI > 0 → built-up surface
GRADIENT_WEIGHT     = 0.4    # gradient magnitude contribution to structure label
CLOSING_SIZE        = 3      # morphological closing kernel size (pixels)

# ── Scoring ──────────────────────────────────────────────────────────────────
POI_TOP_PERCENTILE = 0.10   # top-10% pixel mean used by compute_poi_score()
GREENERY_THRESHOLD = 0.5    # binarisation threshold for greenery_score_from_prediction()
