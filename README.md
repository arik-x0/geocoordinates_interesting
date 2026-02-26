# geo_interesting

Satellite imagery analysis project with three independent ML pipelines, each detecting a different type of Point of Interest (POI) from EuroSAT Sentinel-2 64×64 tiles.

---

## Project Structure

```
geo_interesting/
├── pretrained_model.py     ← EVA-02 ViT-UNet base class (shared by all 3 models)
├── constants.py            ← band indices, thresholds, scoring constants
├── training_utils.py       ← shared loss functions, metrics, augmentation
├── dataset.py              ← shared dataset module (all 3 models)
├── requirements.txt
├── vegetation_poi/         ← NDVI greenery segmentation
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── elevation_poi/          ← cliff-near-water heatmap detection
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
└── housing_poi/            ← low-density building edge detection
    ├── model.py
    ├── train.py
    ├── predict.py
    └── utils.py
```

---

## Model Architectures

All three models share the same pretrained backbone defined in `pretrained_model.py`. Each task-specific `model.py` subclasses the base class and installs a custom output head.

### Shared Backbone — EVA-02 ViT-UNet (~24M parameters, MIT license)

```
Input (B, C, 64, 64)
  ↓  Channel adapter  Conv2d(C → 3)  [identity if C=3]
  ↓  Bilinear resize → 224×224
  ↓  EVA-02 ViT-S/14  (pretrained, MIT)
  │    patch_size=14 → 16×16 = 256 spatial tokens + 1 CLS
  │    12 transformer blocks, embed_dim=384, 6 heads
  │    Forward hooks tap blocks 2, 5, 8, 11 for skip features
  ↓  U-Net Decoder
  │    Block A  cat(block-11: 384, proj(block-8): 256) → ConvBlock → 256ch @ 16×16
  │    Block B  upsample 16→32  + proj(block-5): 128ch  → ConvBlock → 128ch @ 32×32
  │    Block C  upsample 32→64  + proj(block-2):  64ch  → ConvBlock →  64ch @ 64×64
  ↓  Task-specific head  (see below)
  ↓  (B, 1, 64, 64) — per-pixel POI probability heatmap [0, 1]
```

EVA-02 is pretrained with masked image modelling on ImageNet-22K (BAAI, MIT license). Loaded via `timm>=0.9.2`.

`encode()` runs only the backbone (skips the decoder) and returns a 512-dim L2-normalised CLS-token embedding for FAISS similarity search.

### Task-Specific Output Heads

| Model | Head | Rationale |
|---|---|---|
| **Vegetation** (`TransUNet`) | SE channel-attention: AdaptiveAvgPool → Linear 64→16→64 → Sigmoid → re-weight 64ch → Conv2d(64→1) → Sigmoid | NDVI is spectral; SE re-weights decoder channels that track green-leaf signatures |
| **Elevation** (`ElevationPOITransUNet`) | Soft activation `σ(x)·(1+0.1x)` + fixed 7×7 Gaussian blur (σ=1.5) | Cliff-water POIs are sparse, smooth heatmaps — soft activation avoids hard saturation; Gaussian enforces spatial coherence |
| **Housing** (`HousingEdgeCNN`) | Depthwise Conv2d(64,64, 3×3) + BN + ReLU → Conv2d(64→1) → Sigmoid | Building boundaries are hard and rectangular; depthwise 3×3 learns per-channel Laplacian-like edge filters |

The Elevation model takes **6 input channels** (RGB + DEM elevation + Slope + Aspect). A `Conv2d(6→3)` channel adapter fuses all topographic cues before the ViT so the full self-attention depth sees terrain information.

---

## Ground Truth (Pseudo-labels)

None of the three models use manual annotations:

| Model | Label Generation |
|---|---|
| Vegetation | NDVI from bands 4 & 8 → threshold at 0.3 |
| Elevation | SRTM DEM → slope/aspect → NDWI water mask → Gaussian-weighted POI heatmap |
| Housing | NDBI from bands 11 & 8 + Sobel gradient → threshold + morphological closing |

---

## Training

All three models share the same training loop structure: Adam optimizer, `ReduceLROnPlateau` scheduler (patience=3, factor=0.5), IoU/Dice metrics, best-model checkpoint, and per-epoch JSON logs.

### Loss Functions

| Model | Loss |
|---|---|
| Vegetation | DiceBCE + λ·NT-Xent (λ=0.1) |
| Elevation | MSE+Dice + λ·NT-Xent (λ=0.1) |
| Housing | DiceBCE only |

**NT-Xent (SimCLR contrastive loss):** Two augmented views of the same tile are used as the positive pair. Augmentations are random horizontal/vertical flips and 90° rotations (applied in-batch with pure PyTorch, no torchvision required). This shapes the CLS-token embeddings to be invariant to spatial orientation while remaining discriminative across scene types. The `return_embedding=True` flag returns the prediction and embedding in a single forward pass, avoiding a redundant backbone run during training.

After training, vegetation and elevation build a **FAISS IndexFlatIP** over all splits. Inner product on L2-normalized vectors equals cosine similarity, enabling fast nearest-neighbour retrieval by visual and spectral/topographic similarity.

### Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | 32 (veg) / 16 (elev, housing) | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--contrastive-weight` | 0.1 | λ: weight of NT-Xent loss (0 = disabled) |
| `--temperature` | 0.07 | τ: NT-Xent softmax temperature |
| `--ndvi-threshold` | 0.3 | Vegetation pseudo-label threshold |
| `--cliff-threshold` | 15.0° | Slope angle for cliff classification |
| `--water-threshold` | 0.3 | NDWI threshold for water detection |

### Running Training

```bash
# Vegetation
cd vegetation_poi
python train.py --data-dir data --checkpoint-dir checkpoints

# Elevation
cd elevation_poi
python train.py --data-dir data --checkpoint-dir checkpoints --use-real-dem

# Housing
cd housing_poi
python train.py --data-dir data --checkpoint-dir checkpoints
```

### Running Inference

```bash
# Vegetation
cd vegetation_poi
python predict.py --checkpoint checkpoints/best_model.pth --data-dir data

# Elevation
cd elevation_poi
python predict.py --checkpoint checkpoints/best_model.pth --data-dir data --top-k-similar 5

# Housing
cd housing_poi
python predict.py --checkpoint checkpoints/best_model.pth --data-dir data
```

---

## Outputs

Each `predict.py` saves to an `output/` directory:

- Per-image visualizations ranked by POI score (top-N and bottom-5)
- A ranking overview grid image
- Console tables with per-class POI statistics
- VectorDB similarity search results for the top predictions (all three models)

Checkpoints saved by `train.py`:

| File | Contents |
|---|---|
| `best_model.pth` | Model + optimizer state at best validation IoU |
| `final_model.pth` | Model + optimizer state after last epoch |
| `training_log.json` | Per-epoch metrics, timing, and `is_best` flag |
| `embedding_index.faiss` | FAISS index built from best model (all three models) |
| `embedding_meta.json` | Per-vector metadata aligned with FAISS index |

---

## Dependencies

```
torch>=2.0.0
timm>=0.9.2        # EVA-02 pretrained backbone (MIT license)
numpy>=1.24.0
rasterio>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
Pillow>=9.5.0
scipy>=1.10.0
pyproj>=3.5.0
faiss-cpu>=1.7.0   # optional — required for VectorDB similarity search
```

The EVA-02 ViT-S/14 weights are downloaded automatically by `timm` on first run (MIT license, BAAI).

Install with:

```bash
pip install -r requirements.txt
```

---

## TODO

### Critical

- [x] **Centralise band indices** — `BAND_RED`, `BAND_NIR`, etc. are declared identically in all three `utils.py` files and again in `dataset.py`. Extract into a single `constants.py` to eliminate the risk of silent divergence across four locations.

- [x] **Improve contrastive positives** — NT-Xent currently uses EuroSAT class names as positive pairs. This is semantically weak: the `"Industrial"` class contains highways, mines, and factories. Replace with augmentation-based pairs (two augmented views of the same tile) or learned similarity, rather than coarse class labels.

- [x] **Add `encode()` and VectorDB to the Housing model** — The housing pipeline has no embedding extraction, making cross-model or intra-model similarity search impossible. Implement `encode()` on `HousingEdgeCNN` and build a FAISS index post-training, consistent with the other two models.

### Moderate

- [x] **Log and track synthetic DEM fallback** — When SRTM download fails, `generate_synthetic_dem()` is used silently. Add a warning and record the DEM source (`"real"` / `"synthetic"`) in the per-sample metadata and training log so embedding quality degradation can be traced.

- [x] **Extract shared training boilerplate** — `train_one_epoch()`, `validate()`, `compute_iou()`, and `compute_dice()` are largely copy-pasted across all three `train.py` files. Move into a shared `training_utils.py` to reduce maintenance surface.

- [x] **Handle all-zero channel normalisation** — `normalize_channel()` returns an all-zeros array when a tile has zero variance (e.g. a fully flat DEM tile). This corrupts the 6-channel input silently. Add a warning and a fallback (e.g. return the raw channel or a small noise floor).

### Minor

- [x] **Centralise magic-number thresholds** — `compute_poi_score()` hardcodes the top-10% pixel cutoff; housing density thresholds (5%–20%) are scattered across `utils.py`. Move to named constants or expose as CLI arguments.

- [x] **Remove unused dependencies** — `torchvision` and `scikit-learn` are listed in `requirements.txt` but not used anywhere in the codebase.

- [x] **Tune NT-Xent temperature** — τ=0.07 is the ViT paper default and has not been ablated for satellite imagery. Consider adding a short sweep over τ ∈ {0.05, 0.07, 0.1, 0.2} to confirm the default is appropriate.
