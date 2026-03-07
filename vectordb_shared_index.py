"""
Unified VectorDB builder — scores every EuroSAT tile with all 9 submodels
+ the meta aggregator and stores everything in a single FAISS index.

Replaces the per-submodel indexes with one rich shared index where each entry
contains the full scoring profile of a tile:

    filepath, class_name
    vegetation_score, housing_score, terrain_score          (structural)
    fractal_score, water_score, color_score,                (aesthetic)
    symmetry_score, sublime_score, complexity_score
    aesthetic_score, aesthetic_weights                      (meta)

Usage:
    python vectordb_shared_index.py \\
        --data-dir data \\
        --checkpoint-dir checkpoints \\
        --output-dir checkpoints/shared \\
        --batch-size 32

Output:
    checkpoints/shared/shared_index.faiss   — FAISS IndexFlatIP (cosine sim)
    checkpoints/shared/shared_index_meta.json — per-tile metadata list

After building, load and query with VectorDB.load_shared():

    from base.utils import VectorDB
    from pathlib import Path

    vdb = VectorDB.load_shared(Path("checkpoints/shared"))

    # Visually similar tiles with high water score
    results = vdb.query(embedding, top_k=10,
                        filter_fn=lambda m: m["water_score"] > 0.65)

    # Tiles with similar aesthetic character (high fractal + high color)
    results = vdb.query(embedding, top_k=10,
                        filter_fn=lambda m: m["fractal_score"] > 0.5
                                        and m["color_score"] > 0.5)

Notes:
  - Pass 1 (aesthetic) runs all 9 submodels with a single shared decoder (from
    the first available aesthetic checkpoint). Geo heads use their own trained
    weights but share this decoder — same design as meta/predictor.py.
    Housing is inverted (1 − pred) before aggregation; raw housing_score stored
    in metadata still reflects building presence.
  - Passes 2-4 re-score vegetation / housing / elevation with their own trained
    decoders, overwriting the Pass 1 estimates with more accurate values.
  - Any submodel whose checkpoint is missing is skipped gracefully.
  - Elevation uses synthetic DEM (--use-real-dem not supported here for speed).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from core.model import CoreSatelliteModel                       # noqa: E402
from dataset import (                                           # noqa: E402
    get_fractal_dataloaders,
    get_vegetation_dataloaders,
    get_housing_dataloaders,
    get_elevation_dataloaders,
)
from submodels.vegetation.model    import TransUNet             # noqa: E402
from submodels.housing.model       import HousingEdgeCNN        # noqa: E402
from submodels.elevation.model     import ElevationPOITransUNet # noqa: E402
from submodels.fractal.model       import FractalPatternNet     # noqa: E402
from submodels.water.model         import WaterGeometryNet      # noqa: E402
from submodels.color_harmony.model import ColorHarmonyNet       # noqa: E402
from submodels.symmetry.model      import SymmetryOrderNet      # noqa: E402
from submodels.sublime.model       import ScaleSublimeNet       # noqa: E402
from submodels.complexity.model    import ComplexityBalanceNet  # noqa: E402
from meta.model                    import AestheticAggregator   # noqa: E402


# Mirrors _SUBMODEL_REGISTRY in meta/predictor.py.
# (score_key, class, checkpoint_subdir, invert_for_aggregator)
# invert=True → aggregator receives (1 − pred); stored score_key stays raw.
_SUBMODEL_REGISTRY = [
    ("fractal_score",    FractalPatternNet,    "fractal",       False),
    ("water_score",      WaterGeometryNet,     "water",         False),
    ("color_score",      ColorHarmonyNet,      "color_harmony", False),
    ("symmetry_score",   SymmetryOrderNet,     "symmetry",      False),
    ("sublime_score",    ScaleSublimeNet,      "sublime",       False),
    ("complexity_score", ComplexityBalanceNet, "complexity",    False),
    ("vegetation_score", TransUNet,            "vegetation",    False),
    ("terrain_score",    ElevationPOITransUNet,"elevation",     False),
    ("housing_score",    HousingEdgeCNN,       "housing",       True),  # → urban_openness
]

_ALL_SCORE_KEYS = [
    "fractal_score", "water_score", "color_score",
    "symmetry_score", "sublime_score", "complexity_score",
    "vegetation_score", "terrain_score", "housing_score",
    "aesthetic_score",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_decoder(ckpt_path: Path, core, device) -> bool:
    """Load decoder weights from a checkpoint into core. Returns True on success."""
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "decoder_state_dict" in ckpt:
        core.decoder.load_state_dict(ckpt["decoder_state_dict"])
        return True
    return False


def _load_head(model_class, ckpt_path: Path, device):
    """Instantiate a task head and load its weights. Returns None if missing."""
    if not ckpt_path.exists():
        print(f"    WARNING: {ckpt_path} not found — skipping")
        return None
    model = model_class(out_channels=1).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["submodel_state_dict"])
    model.eval()
    return model


# ── Passes ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _aesthetic_pass(core, ckpt_dir: Path, records: Dict,
                    loaders, device) -> None:
    """Pass 1 — CLS embeddings + all 9 submodel scores + meta aggregator score.

    All 9 submodels are treated equally. The decoder is loaded from the first
    available checkpoint in registry order. Housing is inverted (urban_openness
    = 1 − pred) before aggregation; housing_score stored in records stays raw.
    Passes 2–4 overwrite the geo scores with more accurate per-decoder values.
    """
    print("\n[Pass 1] All 9 submodels + CLS embeddings + meta aggregator")

    # Load decoder from first available checkpoint (registry order = priority)
    decoder_loaded = False
    for _, _, subdir, _ in _SUBMODEL_REGISTRY:
        if _load_decoder(ckpt_dir / subdir / "best_model.pth", core, device):
            print(f"    Decoder loaded from: {subdir}/best_model.pth")
            decoder_loaded = True
            break
    if not decoder_loaded:
        print("    WARNING: no checkpoint found — using uninitialised decoder")

    # Load all 9 task heads
    heads = []
    for score_key, cls, subdir, _ in _SUBMODEL_REGISTRY:
        head = _load_head(cls, ckpt_dir / subdir / "best_model.pth", device)
        heads.append((score_key, head))

    # Load meta aggregator (9 inputs)
    aggregator = AestheticAggregator(n_submodels=9).to(device)
    agg_path   = ckpt_dir / "meta" / "best_model.pth"
    if agg_path.exists():
        agg_ckpt = torch.load(agg_path, map_location=device, weights_only=True)
        try:
            aggregator.load_state_dict(agg_ckpt["aggregator_state_dict"])
            print("    Aggregator loaded from: meta/best_model.pth")
        except RuntimeError:
            print("    WARNING: meta checkpoint incompatible — using equal weights")
    else:
        print("    WARNING: no meta checkpoint — using equal weights")
    aggregator.eval()
    core.decoder.eval()

    for split_name, loader in zip(["train", "val", "test"], loaders):
        for inputs, _targets, metas in tqdm(loader,
                                            desc=f"  aesthetic [{split_name}]"):
            inputs  = inputs.to(device)
            B       = inputs.size(0)

            features    = core.extract_features(inputs[:, :3])
            feature_map = core.decode(features)
            embs        = F.normalize(features["cls"], p=2, dim=1)  # (B, 384)

            # All 9 heads — uniform call; housing inverted for aggregation
            raw_preds = []
            maps      = []
            for (_, _, _, invert), (_, head) in zip(_SUBMODEL_REGISTRY, heads):
                pred = head(feature_map) if head is not None else \
                       torch.zeros(B, 1, 64, 64, device=device)
                raw_preds.append(pred)
                maps.append(1.0 - pred if invert else pred)

            heatmaps  = torch.cat(maps, dim=1)               # (B, 9, H, W)
            aesthetic = aggregator(heatmaps)                  # (B, 1, H, W)
            weights   = aggregator.channel_weights(heatmaps)  # (B, 9)

            embs_np      = embs.cpu().numpy()
            aesthetic_np = aesthetic.cpu().numpy()
            weights_np   = weights.cpu().numpy()

            for i in range(B):
                fp = metas[i]["filepath"]
                records[fp] = {
                    "filepath":          fp,
                    "class_name":        metas[i]["class_name"],
                    "cls_embedding":     embs_np[i].tolist(),
                    "aesthetic_score":   float(aesthetic_np[i, 0].mean()),
                    "aesthetic_weights": weights_np[i].tolist(),   # 9 weights
                }
                # Store raw scores (housing stored as building presence, not inverted)
                for j, (score_key, _, _, _) in enumerate(_SUBMODEL_REGISTRY):
                    records[fp][score_key] = float(raw_preds[j][i, 0].mean())


@torch.no_grad()
def _rgb_submodel_pass(core, score_key: str, model_class, subdir: str,
                       ckpt_dir: Path, records: Dict, loaders, device) -> None:
    """Score all tiles with a single RGB-input submodel using its own decoder."""
    print(f"\n[Pass] {score_key}")
    ckpt_path = ckpt_dir / subdir / "best_model.pth"

    if not _load_decoder(ckpt_path, core, device):
        print(f"    WARNING: no decoder in {ckpt_path} — skipping")
        return

    head = _load_head(model_class, ckpt_path, device)
    if head is None:
        return

    core.decoder.eval()
    for split_name, loader in zip(["train", "val", "test"], loaders):
        for inputs, _targets, metas in tqdm(loader,
                                            desc=f"  {score_key} [{split_name}]"):
            inputs = inputs.to(device)
            rgb_in = inputs[:, :3]

            features    = core.extract_features(rgb_in)
            feature_map = core.decode(features)
            pred        = head(feature_map)   # (B, 1, H, W)
            pred_np     = pred.cpu().numpy()

            for i in range(inputs.size(0)):
                fp = metas[i]["filepath"]
                if fp in records:
                    records[fp][score_key] = float(pred_np[i, 0].mean())


@torch.no_grad()
def _elevation_pass(core, ckpt_dir: Path, records: Dict,
                    data_dir: Path, batch_size: int, device) -> None:
    """Score all tiles with the elevation model (6-channel RGB + topo input).

    Uses synthetic DEM for speed. Pass --use-real-dem to the submodel trainer
    for the most accurate terrain scores during training.
    """
    print("\n[Pass] terrain_score (elevation)")
    ckpt_path = ckpt_dir / "elevation" / "best_model.pth"

    if not _load_decoder(ckpt_path, core, device):
        print(f"    WARNING: no decoder in {ckpt_path} — skipping")
        return

    head = _load_head(ElevationPOITransUNet, ckpt_path, device)
    if head is None:
        return

    loaders = get_elevation_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_real_dem=False,   # synthetic DEM — fast, no internet required
        num_workers=0,
    )
    core.decoder.eval()
    for split_name, loader in zip(["train", "val", "test"], loaders):
        for inputs, _targets, metas in tqdm(loader,
                                            desc=f"  terrain [{split_name}]"):
            inputs = inputs.to(device)
            rgb_in = inputs[:, :3]
            topo   = inputs[:, 3:]   # DEM, slope, aspect  (B, 3, H, W)

            features    = core.extract_features(rgb_in)
            feature_map = core.decode(features)
            pred        = head(feature_map, topo)  # (B, 1, H, W)
            pred_np     = pred.cpu().numpy()

            for i in range(inputs.size(0)):
                fp = metas[i]["filepath"]
                if fp in records:
                    records[fp]["terrain_score"] = float(pred_np[i, 0].mean())


# ── FAISS index ───────────────────────────────────────────────────────────────

def _build_faiss_index(records: Dict, output_dir: Path) -> None:
    """Assemble and save the unified FAISS index from collected records."""
    try:
        import faiss
    except ImportError:
        print("\nERROR: faiss-cpu not installed.  Run: pip install faiss-cpu")
        return

    all_records = list(records.values())

    # Pull embeddings out before saving metadata JSON
    embeddings = np.array(
        [r.pop("cls_embedding") for r in all_records], dtype=np.float32
    )
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product on L2-normed = cosine sim
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "shared_index.faiss"
    meta_path  = output_dir / "shared_index_meta.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n{'='*58}")
    print(f"  Shared VectorDB built successfully")
    print(f"  Tiles indexed  : {index.ntotal}")
    print(f"  Embedding dim  : {dim}")
    print(f"  Index          : {index_path}")
    print(f"  Metadata       : {meta_path}")
    print(f"{'='*58}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a unified shared VectorDB from all EuroSAT tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",       default="data",
                        help="EuroSAT data root")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Root dir containing all submodel checkpoint subdirs")
    parser.add_argument("--output-dir",     default="checkpoints/shared",
                        help="Output directory for the shared index")
    parser.add_argument("--batch-size",     type=int, default=32)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    ckpt_dir   = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Data    : {data_dir}")
    print(f"Ckpts   : {ckpt_dir}")
    print(f"Output  : {output_dir}")

    # Single CoreSatelliteModel — frozen backbone stays in memory across all
    # passes; only the decoder weights are swapped between submodels.
    core = CoreSatelliteModel().freeze().to(device)

    # RGB dataloaders (fractal dataset = all EuroSAT tiles, RGB only)
    # Used for Pass 1 and reused for vegetation / housing passes.
    rgb_loaders = get_fractal_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size, num_workers=0
    )

    records: Dict[str, dict] = {}

    # ── Pass 1: CLS embeddings + all 9 submodel scores + meta aggregator ────────
    _aesthetic_pass(core, ckpt_dir, records, rgb_loaders, device)
    print(f"\n  Tiles collected: {len(records)}")

    # ── Pass 2: Vegetation ────────────────────────────────────────────────────
    veg_loaders = get_vegetation_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size, num_workers=0
    )
    _rgb_submodel_pass(
        core, "vegetation_score", TransUNet, "vegetation",
        ckpt_dir, records, veg_loaders, device,
    )

    # ── Pass 3: Housing ───────────────────────────────────────────────────────
    house_loaders = get_housing_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size, num_workers=0
    )
    _rgb_submodel_pass(
        core, "housing_score", HousingEdgeCNN, "housing",
        ckpt_dir, records, house_loaders, device,
    )

    # ── Pass 4: Elevation (6-channel topo input, own loader) ──────────────────
    _elevation_pass(core, ckpt_dir, records, data_dir, args.batch_size, device)

    # ── Coverage report ───────────────────────────────────────────────────────
    n = len(records)
    print(f"\n  Total tiles: {n}")
    for key in _ALL_SCORE_KEYS:
        covered = sum(1 for r in records.values() if key in r)
        status  = "OK" if covered == n else f"PARTIAL ({covered}/{n})"
        print(f"    {key:<22}: {status}")

    # ── Pass 5: Build and save FAISS index ────────────────────────────────────
    _build_faiss_index(records, output_dir)

    print("\nDone.  Load with:")
    print("    from base.utils import VectorDB")
    print(f"    vdb = VectorDB.load_shared(Path('{output_dir}'))")


if __name__ == "__main__":
    main()
