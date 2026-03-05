# geo_interesting

Satellite imagery analysis project with three independent ML pipelines, each detecting a different type of Point of Interest (POI) from EuroSAT Sentinel-2 64×64 tiles.

---

## Project Structure

```
geo_interesting/
├── core/
│   ├── __init__.py
│   └── model.py            ← CoreSatelliteModel (EVA-02 ViT-S/14, frozen)
├── constants.py            ← band indices, thresholds, scoring constants
├── training_utils.py       ← shared loss functions, metrics, augmentation
├── dataset.py              ← shared dataset module (all 3 models)
├── requirements.txt
├── vegetation_poi/         ← NDVI greenery segmentation
│   ├── model.py            ← TransUNet submodel (trained from scratch)
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── elevation_poi/          ← cliff-near-water heatmap detection
│   ├── model.py            ← ElevationPOITransUNet submodel (trained from scratch)
│   ├── train.py
│   ├── predict.py
│   └── utils.py
└── housing_poi/            ← low-density building edge detection
    ├── model.py            ← HousingEdgeCNN submodel (trained from scratch)
    ├── train.py
    ├── predict.py
    └── utils.py
```

---

## Model Architectures

### Core — EVA-02 ViT-S/14 (`core/model.py`, frozen, MIT license)

Pretrained with masked image modelling on ImageNet-22K (BAAI). Loaded via `timm`.
Provides multi-scale satellite image features to all three submodels.

```
Input (B, 3, 64, 64) RGB satellite tile
  ↓  Bilinear resize → 224×224
  ↓  EVA-02 ViT-S/14  (12 blocks, embed_dim=384, 6 heads, patch_size=14)
     Forward hooks capture block outputs at depths 2, 5, 8, 11
     Each reshaped from (B, 257, 384) → (B, 384, 16, 16)

  extract_features() returns dict:
      blk2   (B, 384, 16×16)  — low-level visual features
      blk5   (B, 384, 16×16)  — mid-level features
      blk8   (B, 384, 16×16)  — deep features
      blk11  (B, 384, 16×16)  — deepest semantic features
      cls    (B, 384)          — global CLS descriptor

  encode() → Linear(384→512) + L2-norm → (B, 512) FAISS embedding
```

The core runs under `torch.no_grad()` during all submodel training.
Its weights are never updated.

### Submodels — all trained entirely from scratch

Each submodel receives the core feature dict and adds its own trainable decoder + head.

| Model | Architecture | Task Head | Params |
|---|---|---|---|
| **Vegetation** `TransUNet` | 3-stage U-Net: proj(384→256/128/64) + bilinear 16→32→64 | SE channel-attention + Conv(64→1) + Sigmoid | ~2.5M |
| **Housing** `HousingEdgeCNN` | 3-stage U-Net + side outputs at 16/32/64 | Depthwise edge conv + HED fusion | ~2.2M |
| **Elevation** `ElevationPOITransUNet` | 2-stage RGB decoder + topo CNN (3→32→64ch) | Gaussian heatmap head | ~1.5M |

#### Elevation dual-stream architecture

```
RGB stream (frozen core)               Topo stream (from scratch)
blk11 (384ch, 16×16)                   DEM + Slope + Aspect (3ch, 64×64)
  → proj+dec_a → 128ch @ 16×16           → topo_enc1 → 32ch @ 64×64
  → dec_b+skip → 64ch  @ 32×32           → topo_enc2 → 64ch @ 64×64
  → upsample   → 64ch  @ 64×64    ↘
                                          cat → fusion(128→64) → heatmap head
                                               → (B, 1, 64, 64)
```

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
| Vegetation | DiceBCE |
| Elevation | MSE+Dice |
| Housing | DiceBCE only |

Because the core is frozen during submodel training, contrastive loss is not applied — gradients cannot propagate to the embedding backbone. FAISS similarity search uses EVA-02 CLS-token embeddings directly, which are shaped by the backbone's masked image modelling pretraining.

After training, vegetation and elevation build a **FAISS IndexFlatIP** over all splits. Inner product on L2-normalized vectors equals cosine similarity, enabling fast nearest-neighbour retrieval by visual and spectral/topographic similarity.

### Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | 32 (veg) / 16 (elev, housing) | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
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
