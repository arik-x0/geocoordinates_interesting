"""
Sentinel-2 band indices and model thresholds shared across all three POI pipelines.

All three models consume EuroSAT 13-band Sentinel-2 tiles (.tif).
Band indices are 0-based into the 13-channel raster stack.
"""

# ── Sentinel-2 band indices (0-based, 13-band EuroSAT .tif) ─────────────────
BAND_RED        = 3   # Band 4   — 665 nm
BAND_GREEN      = 2   # Band 3   — 560 nm
BAND_BLUE       = 1   # Band 2   — 490 nm
BAND_NIR        = 7   # Band 8   — 842 nm  (broad NIR, used for NDVI/NDWI)
BAND_NIR_NARROW = 8   # Band 8A  — 865 nm  (narrow NIR, used by Prithvi)
BAND_SWIR       = 10  # Band 11  — 1610 nm (SWIR 1, used for NDBI)
BAND_SWIR2      = 12  # Band 12  — 2190 nm (SWIR 2, used by Prithvi)

# ── Prithvi-EO-1.0-100M backbone input ───────────────────────────────────────
# Six HLS bands expected by Prithvi in this exact order (matching pre-training).
# Indices into the 13-band EuroSAT raster stack (0-based).
#   ch0 = B02 (Blue,       490 nm) — EuroSAT index 1
#   ch1 = B03 (Green,      560 nm) — EuroSAT index 2
#   ch2 = B04 (Red,        665 nm) — EuroSAT index 3
#   ch3 = B8A (NIR-Narrow, 865 nm) — EuroSAT index 8
#   ch4 = B11 (SWIR-1,    1610 nm) — EuroSAT index 11
#   ch5 = B12 (SWIR-2,    2190 nm) — EuroSAT index 12
PRITHVI_BAND_INDICES = [1, 2, 3, 8, 11, 12]

# Per-band normalization constants for Prithvi-EO-1.0-100M.
# Values are in HLS reflectance × 10000 scale (matching EuroSAT raw DNs).
# Source: ibm-nasa-geospatial/Prithvi-EO-1.0-100M model card.
PRITHVI_MEAN = [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0]
PRITHVI_STD  = [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]

# Channel indices within the 6-band Prithvi input for RGB display (R, G, B).
# Prithvi order is Blue-first; display needs Red first for correct colours.
PRITHVI_RGB_DISPLAY = [2, 1, 0]   # ch2=Red(B04), ch1=Green(B03), ch0=Blue(B02)

# ── Vegetation thresholds ────────────────────────────────────────────────────
NDVI_GREENERY_THRESHOLD = 0.3   # NDVI above this → greenery pixel

# ── Elevation / terrain thresholds ───────────────────────────────────────────
NDWI_WATER_THRESHOLD  = 0.3    # NDWI above this → water pixel (used by Water model)
CLIFF_SLOPE_THRESHOLD = 15.0   # kept for backward compatibility
POI_PROXIMITY_SIGMA   = 5.0    # kept for backward compatibility
TERRAIN_RELIEF_WINDOW = 8      # local window (px) for relief & ruggedness computation

# ── Housing density thresholds ───────────────────────────────────────────────
HOUSING_DENSITY_MIN = 0.05   # at least 5% built-up coverage
HOUSING_DENSITY_MAX = 0.20   # at most 20% built-up coverage (not overly urban)
NDBI_THRESHOLD      = 0.0    # NDBI > 0 → built-up surface
GRADIENT_WEIGHT     = 0.4    # gradient magnitude contribution to structure label
CLOSING_SIZE        = 3      # morphological closing kernel size (pixels)

# ── Scoring ──────────────────────────────────────────────────────────────────
POI_TOP_PERCENTILE = 0.10   # top-10% pixel mean used by compute_poi_score()
GREENERY_THRESHOLD = 0.5    # binarisation threshold for greenery_score_from_prediction()

# ── Aesthetic submodel constants ─────────────────────────────────────────────
FRACTAL_TARGET_SIGMA  = 0.25   # width of Gaussian bell around target fractal richness
SYMMETRY_LOCAL_SIZE   = 8      # patch size (px) for local gradient orientation stats
SUBLIME_COARSE_SIGMA  = 10.0   # Gaussian sigma for large-scale feature smoothing
COMPLEXITY_TARGET_LEVEL = 0.45 # target normalised local std for optimal complexity
