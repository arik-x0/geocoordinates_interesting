# geo_interesting

Satellite imagery analysis project with three independent ML pipelines, each detecting a different type of Point of Interest (POI) from EuroSAT Sentinel-2 64x64 tiles.

---

## Architecture Overview

The project uses a **shared representation + task-specific heads** design. A single pretrained backbone + shared decoder produces a rich spatial feature map that all three submodels consume. Each submodel adds only a thin, task-specific head on top.

```
Input (B, 3, 64, 64) RGB
        |
        v
  CoreSatelliteModel              [core/model.py]
  ├── DINO ViT-S/16  (FROZEN)     — pretrained backbone, never updated
  │   └── hooks at blocks 2/5/8/11 → (B, 384, 14, 14) token maps
  └── _SharedDecoder (TRAINABLE)  — trained jointly with each task head
      14x14 → 32x32 → 64x64 via skip connections
      output: (B, 128, 64, 64) feature map
        |
        ├──> TransUNet head         (vegetation)   → (B, 1, 64, 64) heatmap
        ├──> HousingEdgeCNN head    (housing)      → (B, 1, 64, 64) heatmap
        └──> ElevationPOITransUNet  (elevation)    → (B, 1, 64, 64) heatmap
             + topo CNN (DEM/slope/aspect)
```

Training flow per submodel:
1. Backbone runs under `torch.no_grad()` — no graph, no gradient
2. Shared decoder runs normally — gradients flow, weights update
3. Task head runs normally — gradients flow, weights update
4. `optimizer = Adam(decoder.parameters() + head.parameters())`

---

## Project Structure

```
geo_interesting/
|
├── core/
│   └── model.py                  ← CoreSatelliteModel
│                                    - DINO ViT-S/16 backbone (frozen)
│                                    - _SharedDecoder (trainable, shared)
│                                    - extract_features() / decode() / encode()
│
├── base/                         ← shared abstract infrastructure
│   ├── submodel.py               ← BaseSubmodel, _ConvBlock, count_parameters
│   ├── trainer.py                ← BaseTrainer  (Template Method pattern)
│   ├── predictor.py              ← BasePredictor (Template Method pattern)
│   └── utils.py                  ← VectorDB (FAISS cosine similarity index)
│
├── submodels/
│   ├── vegetation/               ← greenery coverage detection
│   │   ├── model.py              ← TransUNet (SE channel-attention head, ~0.2M params)
│   │   ├── trainer.py            ← VegetationTrainer
│   │   ├── predictor.py          ← VegetationPredictor
│   │   ├── utils.py              ← NDVI, greenery scoring, visualization
│   │   └── __main__.py           ← CLI entry point
│   │
│   ├── housing/                  ← built-up structure detection
│   │   ├── model.py              ← HousingEdgeCNN (HED-style head, ~0.07M params)
│   │   ├── trainer.py            ← HousingTrainer
│   │   ├── predictor.py          ← HousingPredictor
│   │   ├── utils.py              ← NDBI, structure labels, visualization
│   │   └── __main__.py           ← CLI entry point
│   │
│   └── elevation/                ← cliff-near-water POI detection
│       ├── model.py              ← ElevationPOITransUNet (topo-fusion head, ~0.35M params)
│       ├── trainer.py            ← ElevationTrainer
│       ├── predictor.py          ← ElevationPredictor
│       ├── utils.py              ← DEM, slope/aspect, water detection, visualization
│       └── __main__.py           ← CLI entry point
│
├── constants.py                  ← band indices, thresholds, scoring constants
├── training_utils.py             ← shared losses, metrics, FAISS index builder
└── dataset.py                    ← dataset loaders for all three tasks
```

### Design principles

| Principle | Applied via |
|---|---|
| Single Responsibility | Each file has exactly one job (model / trainer / predictor / utils) |
| Open/Closed | Adding a new submodel = new folder only, zero changes to existing code |
| Template Method | `BaseTrainer` and `BasePredictor` define the algorithm skeleton; subclasses fill in hooks |
| Dependency Inversion | All trainers/predictors depend on abstract base classes, not each other |
| Separation of Concerns | Domain logic (NDVI, NDBI, DEM) is isolated from ML infrastructure |

---

## Core Model

### DINO ViT-S/16 backbone (`vit_small_patch16_224.dino` via timm)

Pretrained with self-supervised DINO on ImageNet. Always frozen during training.

- `embed_dim = 384`, `patch_size = 16`, spatial grid = `14x14`
- Forward hooks at blocks 2, 5, 8, 11 capture `(B, 384, 14, 14)` token maps
- CLS token `(B, 384)` used for VectorDB embeddings via L2-normalisation

### Shared UNet decoder (`_SharedDecoder`)

Trainable. Lives inside `CoreSatelliteModel`. Shared across all task heads.

```
blk11 (B, 384, 14x14) → proj11 → (B, 256, 14x14)
                                        ↓ dec_a
                              (B, 256, 14x14) + upsample
blk8  (B, 384, 14x14) → proj8  → concat → dec_b → (B, 128, 32x32)
                                                          ↓ + upsample
blk2  (B, 384, 14x14) → proj2  → concat → dec_c → (B, 128, 64x64)
                                                          ↓
                                               feature_map (B, 128, 64x64)
```

Output `_DEC_CHANNELS = 128` is the shared constant consumed by all task heads.

---

## Submodels

All three submodels receive `(B, 128, 64, 64)` from the shared decoder and output `(B, 1, 64, 64)`.

| Submodel | Head architecture | Params | Task |
|---|---|---|---|
| `TransUNet` | ConvBlock(128→64) + SE channel-attention + Conv(64→1) | ~0.2M | Vegetation segmentation |
| `HousingEdgeCNN` | Direct side + depthwise edge side + fusion Conv(2→1) | ~0.07M | Structure density |
| `ElevationPOITransUNet` | proj(128→64) + topo CNN(3→32→64) + fusion + Gaussian blur | ~0.35M | Cliff-water POI heatmap |

The elevation submodel is unique in taking a second input: `topo (B, 3, 64, 64)` containing DEM elevation, slope, and aspect channels.

---

## Ground Truth (Pseudo-labels)

No manual annotations are used. Labels are derived from spectral/topographic signals:

| Task | Label generation |
|---|---|
| Vegetation | NDVI from bands 4 & 8, threshold at 0.3 |
| Housing | NDBI from bands 11 & 8 + Sobel gradient → threshold + morphological closing |
| Elevation | SRTM DEM → slope/aspect → NDWI water mask → Gaussian-weighted POI heatmap |

---

## Training

All three submodels share the same training loop from `BaseTrainer`:
- `Adam(decoder.parameters() + head.parameters())`, weight_decay=1e-4
- `ReduceLROnPlateau` scheduler (patience=3, factor=0.5)
- IoU + Dice metrics, best-model checkpoint, per-epoch JSON log
- FAISS VectorDB built from L2-normalised CLS embeddings at end of training

### Loss Functions

| Task | Loss |
|---|---|
| Vegetation | DiceBCE (dice_weight=0.5) |
| Housing | DiceBCE (dice_weight=0.5) |
| Elevation | MSE + soft-Dice (equal weight) |

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 25 | Training epochs |
| `--batch-size` | 32 (veg) / 16 (housing, elev) | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--ndvi-threshold` | 0.3 | Vegetation pseudo-label threshold |
| `--use-real-dem` | false | Use real SRTM DEM (elevation only) |

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

All commands run from the project root:

```bash
# Vegetation
python -m submodels.vegetation train   --epochs 25 --data-dir data
python -m submodels.vegetation predict --checkpoint checkpoints/vegetation/best_model.pth

# Housing
python -m submodels.housing train   --epochs 25 --data-dir data
python -m submodels.housing predict --checkpoint checkpoints/housing/best_model.pth

# Elevation (synthetic DEM)
python -m submodels.elevation train   --epochs 25 --data-dir data
# Elevation (real SRTM DEM)
python -m submodels.elevation train   --epochs 25 --data-dir data --use-real-dem
python -m submodels.elevation predict --checkpoint checkpoints/elevation/best_model.pth
```

---

## Outputs

Each predictor saves to `output/<task>/`:

- Per-image visualizations ranked by score (top-N and bottom-5)
- Ranking overview grid image
- Console table with per-class statistics
- VectorDB similarity search results for the top predictions

Checkpoint files:

| File | Contents |
|---|---|
| `best_model.pth` | Decoder + head + optimizer at best validation IoU |
| `final_model.pth` | Decoder + head + optimizer after last epoch |
| `training_log.json` | Per-epoch metrics, timing, `is_best` flag |
| `embedding_index.faiss` | FAISS cosine index over L2-normalised CLS embeddings |
| `embedding_meta.json` | Per-vector metadata aligned with FAISS index |

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
