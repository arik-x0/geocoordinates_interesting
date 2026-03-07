# Topophilia

A satellite imagery analysis system that detects and scores geographically interesting locations from EuroSAT Sentinel-2 imagery. Nine independent ML submodels each score a different visual dimension of a 64×64 tile. All nine scores are fused by a learned aggregator (`meta/`) into a single **aesthetic heatmap** — the final output of the system.

---

## System Diagram

```
  EuroSAT Sentinel-2 tiles  (B, 13, 64, 64)
              |
              v
  ┌───────────────────────────────────────┐
  │          CoreSatelliteModel           │   core/model.py
  │                                       │
  │  DINO ViT-S/16  (FROZEN)             │
  │  ├── patch embedding (16x16 patches)  │
  │  └── hooks at blocks 2/5/8/11        │
  │        └── (B, 384, 14, 14) maps     │
  │                  |                    │
  │  _SharedDecoder  (TRAINABLE)          │
  │  └── UNet upsample 14→32→64          │
  │        └── (B, 128, 64, 64) feat map │
  │                  |                    │
  │  CLS token → (B, 384) embedding ─────┼──→  VectorDB (FAISS)
  └───────────────────────────────────────┘
              |
     (B, 128, 64, 64) shared feature map
              |
    ┌─────────┴──────────────────────────────────────────────────┐
    │               Nine task heads (all equal)                   │
    │                                                             │
    │  ├── fractal/       FractalPatternNet  → (B,1,64,64)      │
    │  ├── water/         WaterGeometryNet   → (B,1,64,64)      │
    │  ├── color_harmony/ ColorHarmonyNet    → (B,1,64,64)      │
    │  ├── symmetry/      SymmetryOrderNet   → (B,1,64,64)      │
    │  ├── sublime/       ScaleSublimeNet    → (B,1,64,64)      │
    │  ├── complexity/    ComplexityBalance  → (B,1,64,64)      │
    │  ├── vegetation/    TransUNet          → (B,1,64,64)      │
    │  ├── elevation/     ElevPOITransUNet   → (B,1,64,64)      │
    │  │                    + optional topo (DEM/slope/aspect)  │
    │  └── housing/       HousingEdgeCNN    → (B,1,64,64)      │
    └─────────────────────────┬───────────────────────────────────┘
                              │
         9 heatmaps stacked → (B, 9, 64, 64)
         (housing inverted to urban_openness before stacking)
                              │
              ┌───────────────▼──────────────┐
              │       AestheticAggregator     │   meta/model.py
              │                               │
              │  Channel SE attention         │
              │  (which dimension matters?)   │
              │         ↓                     │
              │  Spatial CNN fusion           │
              │  (where does it matter?)      │
              └───────────────┬───────────────┘
                              │
                    (B, 1, 64, 64)
              FINAL AESTHETIC HEATMAP
```

---

## Project Structure

```
geo_interesting/
│
├── core/
│   └── model.py              ← CoreSatelliteModel
│                                DINO ViT-S/16 backbone (frozen)
│                                _SharedDecoder (trainable, shared by all submodels)
│
├── base/                     ← shared abstract infrastructure
│   ├── submodel.py           ← BaseSubmodel, _ConvBlock, count_parameters
│   ├── trainer.py            ← BaseTrainer  (Template Method pattern)
│   ├── predictor.py          ← BasePredictor (Template Method pattern)
│   └── utils.py              ← VectorDB (FAISS cosine similarity index)
│
├── submodels/                ← all nine task heads — structurally equal
│   │
│   ├── fractal/              ← self-similar repeating patterns
│   │   ├── model.py          ← FractalPatternNet (multi-scale dilated, ~0.18M params)
│   │   ├── trainer.py        ← FractalTrainer
│   │   ├── predictor.py      ← FractalPredictor
│   │   ├── utils.py          ← Laplacian pyramid fractal label
│   │   └── __main__.py       ← CLI entry: python -m submodels.fractal
│   │
│   ├── water/                ← water body geometry and reflectance
│   │   ├── model.py          ← WaterGeometryNet (NDWI path + edge path, ~0.14M params)
│   │   ├── trainer.py        ← WaterTrainer
│   │   ├── predictor.py      ← WaterPredictor
│   │   ├── utils.py          ← NDWI water label, visualization
│   │   └── __main__.py       ← CLI entry: python -m submodels.water
│   │
│   ├── color_harmony/        ← chromatic richness and spectral diversity
│   │   ├── model.py          ← ColorHarmonyNet (chroma SE + spectral path, ~0.15M params)
│   │   ├── trainer.py        ← ColorHarmonyTrainer
│   │   ├── predictor.py      ← ColorHarmonyPredictor
│   │   ├── utils.py          ← HSV saturation + cross-band spectral spread label
│   │   └── __main__.py       ← CLI entry: python -m submodels.color_harmony
│   │
│   ├── symmetry/             ← local structural order and directional regularity
│   │   ├── model.py          ← SymmetryOrderNet (4-directional conv, ~0.12M params)
│   │   ├── trainer.py        ← SymmetryTrainer
│   │   ├── predictor.py      ← SymmetryPredictor
│   │   ├── utils.py          ← gradient circular variance label
│   │   └── __main__.py       ← CLI entry: python -m submodels.symmetry
│   │
│   ├── sublime/              ← scale contrast and multi-scale drama
│   │   ├── model.py          ← ScaleSublimeNet (3-scale residual contrast, ~0.10M params)
│   │   ├── trainer.py        ← SublimeTrainer
│   │   ├── predictor.py      ← SublimePredictor
│   │   ├── utils.py          ← coarse/fine luminance contrast label
│   │   └── __main__.py       ← CLI entry: python -m submodels.sublime
│   │
│   ├── complexity/           ← information density at perceptual optimum
│   │   ├── model.py          ← ComplexityBalanceNet (order + chaos paths, ~0.13M params)
│   │   ├── trainer.py        ← ComplexityTrainer
│   │   ├── predictor.py      ← ComplexityPredictor
│   │   ├── utils.py          ← local gradient std Gaussian-bell label
│   │   └── __main__.py       ← CLI entry: python -m submodels.complexity
│   │
│   ├── vegetation/           ← greenery density (biophilia signal)
│   │   ├── model.py          ← TransUNet (SE channel-attention, ~0.2M params)
│   │   ├── trainer.py        ← VegetationTrainer
│   │   ├── predictor.py      ← VegetationPredictor
│   │   ├── utils.py          ← NDVI, greenery scoring, visualization
│   │   └── __main__.py       ← CLI entry: python -m submodels.vegetation
│   │
│   ├── elevation/            ← topographic terrain beauty (SBE-grounded)
│   │   ├── model.py          ← ElevationPOITransUNet (topo-fusion head, ~0.35M params)
│   │   │                        topo=None defaults to zeros (usable without DEM)
│   │   ├── trainer.py        ← ElevationTrainer
│   │   ├── predictor.py      ← ElevationPredictor
│   │   ├── utils.py          ← SRTM DEM, slope/aspect, terrain labels, visualization
│   │   └── __main__.py       ← CLI entry: python -m submodels.elevation
│   │
│   └── housing/              ← built-up structure detection (urban density)
│       ├── model.py          ← HousingEdgeCNN (HED-style dual path, ~0.07M params)
│       ├── trainer.py        ← HousingTrainer
│       ├── predictor.py      ← HousingPredictor
│       ├── utils.py          ← NDBI, edge labels, visualization
│       └── __main__.py       ← CLI entry: python -m submodels.housing
│
├── meta/                     ← final aesthetic aggregation (NOT a submodel)
│   ├── model.py              ← AestheticAggregator (SE attention + spatial CNN, ~0.05M)
│   ├── predictor.py          ← AestheticPredictor (runs full 9-submodel pipeline)
│   └── __main__.py           ← CLI entry: python -m meta
│
├── vectordb_shared_index.py  ← builds one unified FAISS index for the whole project
├── constants.py              ← band indices, thresholds, scoring constants
├── training_utils.py         ← shared losses, metrics, FAISS index builder
└── dataset.py                ← dataset loaders for all submodel tasks
```

---

## Core Model

The `CoreSatelliteModel` ([core/model.py](core/model.py)) is the shared backbone that all submodels build on. It runs once per tile and produces a rich spatial feature map consumed by every task head.

### DINO ViT-S/16 backbone — always frozen

Pretrained with self-supervised DINO on ImageNet. Its weights are never updated.

- Processes `(B, 3, 64, 64)` RGB input via 16×16 patches → 4×4 = 16 patch tokens
- `embed_dim = 384`, spatial token grid `14×14` (with position interpolation to 64px)
- Forward hooks at transformer blocks **2, 5, 8, 11** capture intermediate `(B, 384, 14, 14)` token maps
- The CLS token `(B, 384)` is L2-normalised and stored in the **VectorDB** for similarity search

### Shared UNet decoder — trainable, shared by all submodels

Lives inside `CoreSatelliteModel`. All nine submodels share its weights — training any one submodel also improves the shared representation.

```
block11  (B, 384, 14×14) → proj → (B, 256, 14×14)
                                          ↓ upsample 2×
block8   (B, 384, 14×14) → proj → concat → dec_b → (B, 128, 32×32)
                                                           ↓ upsample 2×
block2   (B, 384, 14×14) → proj → concat → dec_c → (B, 128, 64×64)
                                                           ↓
                                             feature map  (B, 128, 64×64)
```

Output `(B, 128, 64, 64)` is the input to every task head.

---

## Submodels

All nine submodels are **structurally equal** — they share the same `BaseSubmodel` base class, the same training loop, and the same `(B, 128, 64, 64)` → `(B, 1, 64, 64)` interface. All nine feed into the `AestheticAggregator`. The distinction between "visual aesthetic" and "geographic/structural" is semantic, not architectural.

| Submodel | What it detects | Pseudo-label signal | Architecture | Params |
|---|---|---|---|---|
| `fractal` | Self-similar patterns across scales | Laplacian pyramid inter-scale correlation | Multi-scale dilated conv (d=1,2,4) | ~0.18M |
| `water` | Water body geometry and reflectance | NDWI (bands 3+8) mask + edge geometry | NDWI spectral path + edge CNN path | ~0.14M |
| `color_harmony` | Chromatic richness and spectral diversity | 60% HSV saturation + 40% cross-band spectral std | Chroma SE-attention + depthwise spectral path | ~0.15M |
| `symmetry` | Structural order and directional regularity | Gradient circular variance (mean resultant R) | 4-directional convs (H, V, D1, D2) | ~0.12M |
| `sublime` | Multi-scale contrast and large-scale drama | \|fine detail − coarse structure\| luminance diff | 3-scale residual contrast (4×, 8×, 16× pool) | ~0.10M |
| `complexity` | Information density at perceptual optimum | Local gradient std → Gaussian bell at 0.45 | Order path (smooth) × chaos path (local std) | ~0.13M |
| `vegetation` | Greenery density — biophilia signal | NDVI (NIR − R)/(NIR + R) threshold at 0.3 | ConvBlock(128→64) + SE channel-attention | ~0.20M |
| `elevation` | Terrain ruggedness and scenic beauty | Local relief + slope variance + ridgeline curvature (SBE) | proj(128→64) + topo CNN(3→32→64) fusion | ~0.35M |
| `housing` | Built-up structure density | NDBI (SWIR − NIR)/(SWIR + NIR) + Sobel + morph close | Direct path + depthwise edge path → fusion | ~0.07M |

### Research basis for the structural submodels in the aesthetic pipeline

**Vegetation** feeds a direct positive aesthetic signal. The biophilia hypothesis (Wilson, 1984) posits an innate genetic drive to connect with nature. Attention Restoration Theory (Kaplan, 1995) shows greenery directly reduces directed-attention fatigue — the same mechanism underlying aesthetic pleasure in natural landscapes.

**Elevation** feeds a direct positive aesthetic signal. Scenic Beauty Estimation research (Daniel & Boster) shows topographic heterogeneity (local relief, slope ruggedness, ridgeline curvature) independently predicts scenic quality regardless of other visual factors. The model captures terrain structure visible in RGB that the visual-only `sublime` model cannot fully capture.

**Housing** feeds an **inverted** signal to the aggregator (`urban_openness = 1 − housing_pred`). Attention Restoration Theory establishes that high building density increases cognitive load, reducing restorative experience and aesthetic pleasure. The aggregator's SE channel attention learns the per-scene weight of this signal: a natural landscape is boosted by high urban openness; a geometrically beautiful urban scene can still score high via `color_harmony` or `symmetry` despite low urban openness.

Note: `housing_score` stored in the VectorDB metadata reflects raw building presence (not inverted) for standalone use. Only the aggregation inverts it.

### Elevation model — optional topo input

`ElevationPOITransUNet.forward()` accepts `topo=None`. When `topo` is not provided, the model creates a zero tensor automatically and operates using the RGB feature stream only. Terrain structure visible in RGB (ridge shadows, texture contrasts) still contributes meaningfully. Pass real `topo (B, 3, H, W)` during training and standalone prediction for full accuracy.

### Signal ownership — no overlap

| Signal | Owner |
|---|---|
| NDVI (vegetation index) | `vegetation` exclusively |
| NDWI (water index) | `water` exclusively |
| NDBI (built-up index) | `housing` exclusively |
| DEM / slope / aspect | `elevation` exclusively |
| HSV saturation + spectral spread | `color_harmony` |
| Gradient orientation variance | `symmetry` |
| Coarse/fine luminance contrast | `sublime` |
| Local gradient standard deviation | `complexity` |
| Laplacian pyramid correlation | `fractal` |

---

## Meta — AestheticAggregator

The `meta/` directory is **not a submodel** — it sits downstream of all nine submodels and fuses their heatmaps into the final result. It does not receive the shared feature map; it receives nine pre-computed heatmaps.

### What it does

Given the nine heatmaps stacked as `(B, 9, 64, 64)`, the `AestheticAggregator` learns:

1. **Which dimension matters most** for this image (channel SE attention)
2. **Where** in the image those dimensions align spatially (spatial CNN fusion)

The SE channel attention naturally specialises per scene type: for a forest tile it weights vegetation and fractal heavily; for a mountain it weights elevation and sublime; for an urban architectural scene it weights color harmony and symmetry while discounting urban openness.

### Architecture (`meta/model.py`)

```
input:  (B, 9, 64, 64)  — 9 stacked heatmaps
                           [fractal, water, color_harmony, symmetry, sublime,
                            complexity, vegetation, elevation, urban_openness]
           |
  Channel attention (SE):
    GlobalAvgPool → flatten → Linear(9→4) → ReLU → Linear(4→9) → Sigmoid
    → per-channel weights  (B, 9)
           |
  Element-wise weight:  heatmaps × weights  →  (B, 9, 64, 64)
           |
  Spatial fusion:
    Conv(9→16, 3×3) → BN → ReLU → Conv(16→1, 1×1) → Sigmoid
           |
output: (B, 1, 64, 64)  — unified aesthetic heatmap  ← FINAL OUTPUT
```

~0.05M params — intentionally tiny, since inputs are already high-level semantic maps.

### Why it is separate from `submodels/`

The submodels all receive the shared feature map `(B, 128, 64, 64)` as input and are trained independently. `meta/` operates on a completely different input (pre-computed heatmaps from all nine trained submodels) and requires all nine to have been trained first. It is an aggregator, not a feature extractor.

### Decoder strategy at inference

All nine submodels run against a single shared decoder state in the meta pipeline. The decoder is loaded from the first available checkpoint in registry order. This is a deliberate single-pass trade-off: running nine separate forward passes with nine different decoders would be 9× more expensive. In practice, all decoders converge toward the same general satellite feature extraction goal, so the shared decoder produces sufficiently accurate feature maps for all task heads.

---

## Final Output — The Aesthetic Heatmap

The `(B, 1, 64, 64)` tensor produced by `AestheticAggregator` is the system's primary answer to the question: **where in this satellite tile is something aesthetically interesting?**

- Values in `[0, 1]` — higher = more aesthetically compelling
- Spatially aligned with the original tile — every pixel has a score
- Learned, not hand-crafted — the aggregator discovers which combination of nine dimensions makes a location interesting
- Interpretable — `channel_weights()` reveals which dimension drove the score for any given tile

The `AestheticPredictor` (`meta/predictor.py`) saves a **3×4 panel** visualization per image:

```
Row 0:  RGB satellite | fractal | water | color_harmony
Row 1:  symmetry | sublime | complexity | vegetation
Row 2:  elevation | urban_openness | [blank] | aesthetic fusion overlay
```

Each submodel panel shows its learned channel weight `w=…` as a subtitle.

---

## VectorDB — Similarity Search

The `VectorDB` ([base/utils.py](base/utils.py)) is a FAISS-backed cosine similarity index. The project uses a single **shared index** built by `vectordb_shared_index.py` that covers every EuroSAT tile with a complete scoring profile.

### Shared index — one index for the whole project

```
vectordb_shared_index.py  (5 passes, single CoreSatelliteModel in memory)
    |
    Pass 1: all 9 submodel heads run → per-tile scores + CLS embeddings
    Pass 2: vegetation model with its own trained decoder (more accurate score)
    Pass 3: housing model with its own trained decoder
    Pass 4: elevation model with its own trained decoder + synthetic DEM
    Pass 5: FAISS IndexFlatIP built from CLS embeddings
    |
    └── checkpoints/shared/shared_index.faiss
    └── checkpoints/shared/shared_index_meta.json
```

Each entry in the index stores:

```json
{
  "filepath": "...", "class_name": "Forest",
  "aesthetic_score": 0.74, "aesthetic_weights": [0.12, 0.08, ...],
  "fractal_score": 0.81, "water_score": 0.12, "color_score": 0.67,
  "symmetry_score": 0.43, "sublime_score": 0.55, "complexity_score": 0.60,
  "vegetation_score": 0.89, "terrain_score": 0.34, "housing_score": 0.03
}
```

### Loading and querying

```python
from base.utils import VectorDB
from pathlib import Path

vdb = VectorDB.load_shared(Path("checkpoints/shared"))

# Visually similar tiles
vdb.query(embedding, top_k=10)

# Similar tiles that are also highly aesthetic
vdb.query(embedding, top_k=10,
          filter_fn=lambda m: m["aesthetic_score"] > 0.7)

# Similar water landscapes
vdb.query(embedding, top_k=10,
          filter_fn=lambda m: m["water_score"] > 0.65)

# High fractal + high color harmony (natural patterned landscapes)
vdb.query(embedding, top_k=10,
          filter_fn=lambda m: m["fractal_score"] > 0.5
                          and m["color_score"] > 0.5)

# High vegetation + low housing (undeveloped green areas)
vdb.query(embedding, top_k=10,
          filter_fn=lambda m: m["vegetation_score"] > 0.7
                          and m["housing_score"] < 0.1)
```

The `filter_fn` over-retrieves `top_k × 10` candidates and filters before returning, so the result set is always exactly `top_k` (or fewer if not enough pass the filter).

### How the index works

```
Training / index build
    |
    └── for each tile:
            CLS token (B, 384)  ← from DINO backbone (frozen — same for all submodels)
            L2-normalise  →  unit vector
            FAISS IndexFlatIP   (inner product on L2-normed = cosine similarity)

Query
    |
    └── for any tile's CLS embedding:
            retrieve top-K most cosine-similar tiles
            optionally filter by any score field via filter_fn
```

---

## Ground Truth — Pseudo-labels

No manual annotations. All training targets are derived algorithmically from spectral bands or topographic data.

| Submodel | Label generation |
|---|---|
| `vegetation` | NDVI = (NIR − R) / (NIR + R), threshold at 0.3 |
| `housing` | NDBI = (SWIR − NIR) / (SWIR + NIR) + Sobel edge → morph closing |
| `elevation` | SRTM DEM → 40% local relief + 40% slope ruggedness + 20% ridgeline curvature |
| `fractal` | Laplacian pyramid → inter-scale correlation → Gaussian smoothing |
| `water` | NDWI = (G − NIR) / (G + NIR) → threshold + distance-weighted mask |
| `color_harmony` | HSV saturation (60%) + cross-band spectral std across all 13 bands (40%) |
| `symmetry` | Local gradient circular variance → mean resultant length R in [0,1] |
| `sublime` | \|fine − coarse\| luminance difference at macro scale |
| `complexity` | Local gradient std → Gaussian bell at optimal complexity 0.45 |

---

## Training

All submodels share the same training loop from `BaseTrainer`:

- `Adam(decoder.parameters() + head.parameters())`, `weight_decay=1e-4`
- `ReduceLROnPlateau` scheduler (`patience=3`, `factor=0.5`)
- IoU + Dice metrics, best-model checkpoint saved automatically
- Per-epoch JSON training log

### Loss functions

| Submodel | Loss |
|---|---|
| `vegetation` | DiceBCE (`dice_weight=0.5`) |
| `housing` | DiceBCE (`dice_weight=0.5`) |
| `elevation` | MSE + soft-Dice (equal weight) |
| `fractal` | MSE + soft-Dice |
| `water` | MSE + soft-Dice |
| `color_harmony` | MSE + soft-Dice |
| `symmetry` | MSE + soft-Dice |
| `sublime` | MSE + soft-Dice |
| `complexity` | MSE + soft-Dice |

### Checkpoint format

```python
{
    "epoch":                int,
    "decoder_state_dict":   ...,   # shared decoder weights
    "submodel_state_dict":  ...,   # task head weights
    "optimizer_state_dict": ...,
    "val_iou":              float,
    "val_dice":             float,
    "val_loss":             float,
}
```

---

## Running

All commands run from the project root. Train all nine submodels before running `meta predict` or building the shared VectorDB.

### Train all submodels

```bash
# Visual aesthetic submodels
python -m submodels.fractal       train --epochs 25 --data-dir data
python -m submodels.water         train --epochs 25 --data-dir data
python -m submodels.color_harmony train --epochs 25 --data-dir data
python -m submodels.symmetry      train --epochs 25 --data-dir data
python -m submodels.sublime       train --epochs 25 --data-dir data
python -m submodels.complexity    train --epochs 25 --data-dir data

# Structural submodels
python -m submodels.vegetation train --epochs 25 --data-dir data
python -m submodels.housing    train --epochs 25 --data-dir data
python -m submodels.elevation  train --epochs 25 --data-dir data              # synthetic DEM
python -m submodels.elevation  train --epochs 25 --data-dir data --use-real-dem  # real SRTM DEM
```

Order does not matter — all nine are independent.

### Run individual submodel prediction

```bash
python -m submodels.fractal    predict --checkpoint checkpoints/fractal/best_model.pth
python -m submodels.vegetation predict --checkpoint checkpoints/vegetation/best_model.pth
python -m submodels.elevation  predict --checkpoint checkpoints/elevation/best_model.pth
# (same pattern for all nine)
```

### Meta aggregator — full 9-submodel aesthetic pipeline

Run the full pipeline (loads all nine submodels + aggregator, produces final heatmaps):

```bash
python -m meta predict \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --output-dir output/aesthetic \
    --top-n 20
```

Expected checkpoint layout:

```
checkpoints/
├── fractal/best_model.pth
├── water/best_model.pth
├── color_harmony/best_model.pth
├── symmetry/best_model.pth
├── sublime/best_model.pth
├── complexity/best_model.pth
├── vegetation/best_model.pth
├── elevation/best_model.pth
├── housing/best_model.pth
└── meta/best_model.pth          (optional — equal weights used if absent)
```

### Build the shared VectorDB

Run after all nine submodels have been trained:

```bash
python vectordb_shared_index.py \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --output-dir checkpoints/shared \
    --batch-size 32
```

Output: `checkpoints/shared/shared_index.faiss` + `checkpoints/shared/shared_index_meta.json`

### Key CLI arguments

| Argument | Default | Applies to |
|---|---|---|
| `--epochs` | 25 | all submodels |
| `--batch-size` | 32 | all |
| `--lr` | 1e-3 | all |
| `--data-dir` | `data` | all |
| `--checkpoint` | — | submodel predict |
| `--checkpoint-dir` | `checkpoints` | meta predict, vectordb |
| `--output-dir` | `output/<task>` | predict commands |
| `--top-n` | 20 | predict commands |
| `--ndvi-threshold` | 0.3 | vegetation only |
| `--use-real-dem` | false | elevation only |

---

## Outputs

Each submodel predictor saves to `output/<task>/`:

- Per-image visualizations ranked by score (top-N)
- Console table with score statistics and per-class average bars
- VectorDB similarity results for top predictions (if shared index is present)

The `meta` predictor saves a **3×4 panel** per image showing all nine submodel heatmaps with learned channel weights, plus the fused aesthetic overlay.

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
Pillow>=9.5.0
scipy>=1.10.0
faiss-cpu>=1.7.0   # optional — required for VectorDB similarity search
```

```bash
pip install -r requirements.txt
```
