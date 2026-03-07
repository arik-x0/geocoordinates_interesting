# geo_interesting

A satellite imagery analysis system that detects and scores geographically interesting locations from EuroSAT Sentinel-2 imagery. Nine independent ML submodels each score a different visual/aesthetic dimension of a 64×64 tile. Six of those scores are fused by a learned aggregator (`meta/`) into a single **aesthetic heatmap** — the final output of the system.

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
    ┌─────────┴──────────────────────────────────────────────┐
    │         Nine independent task heads                     │
    │                                                         │
    │  GEOGRAPHIC / STRUCTURAL                                │
    │  ├── vegetation/    TransUNet        → (B,1,64,64)     │
    │  ├── housing/       HousingEdgeCNN  → (B,1,64,64)     │
    │  └── elevation/     ElevPOITransUNet→ (B,1,64,64) ←─  │
    │                        + topo input (DEM/slope/aspect) │
    │                                                         │
    │  AESTHETIC                                              │
    │  ├── fractal/       FractalPatternNet → (B,1,64,64)   │
    │  ├── water/         WaterGeometryNet  → (B,1,64,64)   │
    │  ├── color_harmony/ ColorHarmonyNet   → (B,1,64,64)   │
    │  ├── symmetry/      SymmetryOrderNet  → (B,1,64,64)   │
    │  ├── sublime/       ScaleSublimeNet   → (B,1,64,64)   │
    │  └── complexity/    ComplexityBalance → (B,1,64,64)   │
    └─────────────────────────┬───────────────────────────────┘
                              │
              6 aesthetic heatmaps stacked → (B, 6, 64, 64)
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
├── submodels/
│   │
│   │  ── Geographic / Structural ──────────────────────────────────────
│   ├── vegetation/           ← greenery density detection
│   │   ├── model.py          ← TransUNet (SE channel-attention, ~0.2M params)
│   │   ├── trainer.py        ← VegetationTrainer
│   │   ├── predictor.py      ← VegetationPredictor
│   │   ├── utils.py          ← NDVI, greenery scoring, visualization
│   │   └── __main__.py       ← CLI entry: python -m submodels.vegetation
│   │
│   ├── housing/              ← built-up structure / urban density detection
│   │   ├── model.py          ← HousingEdgeCNN (HED-style dual path, ~0.07M params)
│   │   ├── trainer.py        ← HousingTrainer
│   │   ├── predictor.py      ← HousingPredictor
│   │   ├── utils.py          ← NDBI, edge labels, visualization
│   │   └── __main__.py       ← CLI entry: python -m submodels.housing
│   │
│   ├── elevation/            ← topographic terrain beauty detection
│   │   ├── model.py          ← ElevationPOITransUNet (topo-fusion head, ~0.35M params)
│   │   ├── trainer.py        ← ElevationTrainer
│   │   ├── predictor.py      ← ElevationPredictor
│   │   ├── utils.py          ← SRTM DEM, slope/aspect, terrain labels, visualization
│   │   └── __main__.py       ← CLI entry: python -m submodels.elevation
│   │
│   │  ── Aesthetic ─────────────────────────────────────────────────────
│   ├── fractal/              ← self-similar repeating patterns
│   │   ├── model.py          ← FractalPatternNet (multi-scale laplacian, ~0.18M params)
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
│   └── complexity/           ← information density at perceptual optimum
│       ├── model.py          ← ComplexityBalanceNet (order + chaos paths, ~0.13M params)
│       ├── trainer.py        ← ComplexityTrainer
│       ├── predictor.py      ← ComplexityPredictor
│       ├── utils.py          ← local gradient std Gaussian-bell label
│       └── __main__.py       ← CLI entry: python -m submodels.complexity
│
├── meta/                     ← final aesthetic aggregation (NOT a submodel)
│   ├── model.py              ← AestheticAggregator (SE attention + spatial CNN, ~0.04M)
│   ├── predictor.py          ← AestheticPredictor (runs full 6-submodel pipeline)
│   └── __main__.py           ← CLI entry: python -m meta
│
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

### Geographic / Structural submodels

These three models detect structured geographic features and are trained and used independently of the aesthetic pipeline.

#### Vegetation — `submodels/vegetation/`

Detects greenery and vegetation coverage.

- **Pseudo-label**: NDVI from bands 4 (Red) and 8 (NIR), thresholded at 0.3
- **Architecture**: `TransUNet` — ConvBlock(128→64) + SE channel-attention + Conv(64→1), ~0.2M params
- **Score**: mean NDVI over the predicted heatmap, weighted by confidence

#### Housing — `submodels/housing/`

Detects built-up structures and urban density.

- **Pseudo-label**: NDBI from bands 11 (SWIR) and 8 (NIR) + Sobel edge gradient → threshold + morphological closing
- **Architecture**: `HousingEdgeCNN` — direct intensity path + depthwise edge path → fusion Conv(2→1), ~0.07M params
- **Score**: mean predicted structure density

#### Elevation — `submodels/elevation/`

Detects topographic terrain beauty — mountains, ridges, canyons, glacial cirques.

Grounded in **Scenic Beauty Estimation (SBE)** research (Daniel & Boster, US Forest Service), which shows topographic heterogeneity independently predicts scenic quality. The model is entirely water-independent.

- **Pseudo-label**: three terrain signals derived from SRTM DEM + computed slope:
  - **Local relief** (40%) — `max - min` elevation in an 8px window: rewards mountains and canyons
  - **Slope ruggedness** (40%) — local std of slope angles: rewards heterogeneous inclines (cliffs mixed with ledges)
  - **Ridgeline curvature** (20%) — gradient magnitude of slope: rewards sharp ridge crests, cliff tops, erosion gullies
- **Architecture**: `ElevationPOITransUNet` — proj(128→64) + topo CNN(3→32→64) fusing DEM/slope/aspect + Gaussian blur, ~0.35M params
- **Extra input**: `topo (B, 3, 64, 64)` containing DEM, slope, and aspect channels (synthetic or real SRTM)
- **Score**: mean terrain ruggedness weighted by predicted heatmap confidence

---

### Aesthetic submodels

Six models each capture an independent aesthetic dimension of a landscape. Their heatmaps are fed to the `meta/` aggregator for the final score. Each model has no overlap with the others — the dimensions are designed to be orthogonal.

| Submodel | What it detects | Pseudo-label signal | Architecture | Params |
|---|---|---|---|---|
| `fractal` | Self-similar repeating patterns across scales | Laplacian pyramid inter-scale correlation | Multi-scale Laplacian paths | ~0.18M |
| `water` | Water body geometry, clarity, and reflectance | NDWI (bands 3+8) water mask + edge geometry | NDWI spectral path + edge CNN path | ~0.14M |
| `color_harmony` | Chromatic richness and spectral diversity | 60% HSV saturation + 40% cross-band spectral std (all 13 bands) | Chroma SE-attention + depthwise spectral path | ~0.15M |
| `symmetry` | Structural order and directional regularity | Gradient circular variance (mean resultant length R) | 4-directional convs (H, V, D1, D2) | ~0.12M |
| `sublime` | Multi-scale contrast and large-scale drama | \|fine detail − coarse structure\| luminance difference | 3-scale residual contrast (4×, 8×, 16× pool) | ~0.10M |
| `complexity` | Information density at perceptual optimum | Local gradient std → Gaussian bell centred at 0.45 | Order path (smooth) × chaos path (local) | ~0.13M |

**Signal ownership — no overlap:**

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

The `meta/` directory is **not a submodel** — it sits downstream of the aesthetic submodels and fuses their outputs into the final result. It does not receive the shared feature map; it receives six pre-computed heatmaps.

### What it does

Given the six aesthetic heatmaps stacked as `(B, 6, 64, 64)`, the `AestheticAggregator` learns:

1. **Which aesthetic dimension matters most** for this image (channel SE attention)
2. **Where** in the image those dimensions align spatially (spatial CNN fusion)

### Architecture (`meta/model.py`)

```
input:  (B, 6, 64, 64)  — 6 stacked aesthetic heatmaps
           |
  Channel attention (SE):
    GlobalAvgPool → flatten → Linear(6→3) → ReLU → Linear(3→6) → Sigmoid
    → per-channel weights  (B, 6)
           |
  Element-wise weight:  heatmaps × weights  →  (B, 6, 64, 64)
           |
  Spatial fusion:
    Conv(6→16, 3×3) → BN → ReLU → Conv(16→1, 1×1) → Sigmoid
           |
output: (B, 1, 64, 64)  — unified aesthetic heatmap  ← FINAL OUTPUT
```

~0.04M params — intentionally tiny, since inputs are already high-level semantic maps.

### Why it is separate from `submodels/`

The submodels all share the same `CoreSatelliteModel` feature map as input. `meta/` operates on a completely different input (pre-computed heatmaps) and requires all six submodels to have been trained first. It is an aggregator, not a feature extractor.

---

## Final Output — The Aesthetic Heatmap

The `(B, 1, 64, 64)` tensor produced by `AestheticAggregator` is the system's primary answer to the question: **where in this satellite tile is something aesthetically interesting?**

- Values in `[0, 1]` — higher = more aesthetically compelling
- Spatially aligned with the original tile — every pixel has a score
- Learned, not hand-crafted — the aggregator discovers which combination of fractal patterns, water, colour, symmetry, scale drama, and complexity makes a location interesting
- Interpretable — `channel_weights()` reveals which aesthetic dimension drove the score for any given tile

The `AestheticPredictor` (`meta/predictor.py`) saves a 2×4 panel visualization per image showing all six submodel heatmaps alongside the fused result and the per-submodel learned weights as a bar chart.

---

## VectorDB — Similarity Search

The `VectorDB` ([base/utils.py](base/utils.py)) is a FAISS-backed cosine similarity index built automatically at the end of training for each submodel.

### How it works

```
Training loop
    |
    └── for each tile:
            CLS token (B, 384)  ← from DINO backbone (frozen)
            L2-normalise  →  unit vector
            store in FAISS IndexFlatIP + metadata JSON

Inference (predict)
    |
    └── for each top-scoring tile:
            query VectorDB with its CLS embedding
            retrieve top-K most cosine-similar tiles from training set
            display as "visually similar" recommendations
```

### What it enables

- **Content-based retrieval**: given an interesting tile, find visually similar tiles without re-running the model
- **Cluster inspection**: understand what the model learned by seeing which tiles cluster together
- **Cross-submodel reuse**: because CLS embeddings come from the shared frozen backbone, the VectorDB built during vegetation training is semantically consistent with embeddings from elevation or water training

### Files

| File | Contents |
|---|---|
| `embedding_index.faiss` | FAISS IndexFlatIP of L2-normalised CLS embeddings |
| `embedding_meta.json` | Per-vector metadata (file path, score, coordinates) aligned with index |

VectorDB is optional — the system runs without it if `faiss-cpu` is not installed. Missing index files produce a warning, not an error.

---

## Ground Truth — Pseudo-labels

No manual annotations. All training targets are derived algorithmically from the spectral bands or topographic data.

| Submodel | Label generation |
|---|---|
| `vegetation` | NDVI = (NIR − R) / (NIR + R), threshold at 0.3 |
| `housing` | NDBI = (SWIR − NIR) / (SWIR + NIR) + Sobel edge → morph closing |
| `elevation` | SRTM DEM → local relief + slope ruggedness + ridgeline curvature |
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
- FAISS VectorDB built from all CLS embeddings at the end of training

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
| `meta` (aggregator) | MSE against mean of input heatmaps as self-supervised target |

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

All commands run from the project root.

### Geographic / Structural submodels

```bash
# Vegetation
python -m submodels.vegetation train   --epochs 25 --data-dir data
python -m submodels.vegetation predict --checkpoint checkpoints/vegetation/best_model.pth

# Housing
python -m submodels.housing train   --epochs 25 --data-dir data
python -m submodels.housing predict --checkpoint checkpoints/housing/best_model.pth

# Elevation (synthetic DEM)
python -m submodels.elevation train   --epochs 25 --data-dir data
# Elevation (real SRTM DEM — downloads tiles automatically)
python -m submodels.elevation train   --epochs 25 --data-dir data --use-real-dem
python -m submodels.elevation predict --checkpoint checkpoints/elevation/best_model.pth
```

### Aesthetic submodels

Train each independently. Order does not matter.

```bash
python -m submodels.fractal       train --epochs 25 --data-dir data
python -m submodels.water         train --epochs 25 --data-dir data
python -m submodels.color_harmony train --epochs 25 --data-dir data
python -m submodels.symmetry      train --epochs 25 --data-dir data
python -m submodels.sublime       train --epochs 25 --data-dir data
python -m submodels.complexity    train --epochs 25 --data-dir data
```

Run predict on any individual aesthetic submodel:

```bash
python -m submodels.fractal predict --checkpoint checkpoints/fractal/best_model.pth
# (same pattern for water, color_harmony, symmetry, sublime, complexity)
```

### Meta aggregator — full aesthetic pipeline

Train the aggregator after all six aesthetic submodels have been trained:

```bash
python -m meta train \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --epochs 10
```

Run the full pipeline (loads all six submodels + aggregator, produces final heatmaps):

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
└── meta/best_model.pth
```

### Key CLI arguments

| Argument | Default | Applies to |
|---|---|---|
| `--epochs` | 25 | all |
| `--batch-size` | 32 (veg) / 16 (others) | all |
| `--lr` | 1e-3 | all |
| `--data-dir` | `data` | all |
| `--checkpoint` | — | predict commands |
| `--output-dir` | `output/<task>` | predict commands |
| `--top-n` | 20 | predict commands |
| `--ndvi-threshold` | 0.3 | vegetation only |
| `--use-real-dem` | false | elevation only |

---

## Outputs

Each predictor saves to `output/<task>/`:

- Per-image visualizations ranked by score (top-N and bottom-5)
- Ranking overview grid image
- Console table with score statistics
- VectorDB similarity results for top predictions

The `meta` predictor additionally saves a **2×4 panel** per image showing all six submodel heatmaps, the fused final heatmap, and a bar chart of learned per-dimension weights.

---

## Dependencies

```
torch>=2.0.0
timm>=0.9.2        # DINO ViT-S/16 pretrained backbone
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
