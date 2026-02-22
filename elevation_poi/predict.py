"""
Inference and POI ranking script for elevation-based cliff-water detection.
Loads a trained TransUNet, runs segmentation, ranks images by POI intensity,
queries the FAISS VectorDB for visually similar training images, and saves
heatmap visualizations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import ElevationPOITransUNet
from dataset import get_dataloaders
from utils import (
    compute_poi_score,
    visualize_poi,
    visualize_poi_ranking,
    normalize_channel,
)


# ── VectorDB helpers ─────────────────────────────────────────────────────────

def load_index(checkpoint_dir: Path):
    """Load the FAISS index and metadata written by train.py.

    Returns:
        (index, meta) if the index exists, else (None, None).
    """
    try:
        import faiss
    except ImportError:
        print("WARNING: faiss-cpu not installed — similarity search disabled.")
        return None, None

    index_path = checkpoint_dir / "embedding_index.faiss"
    meta_path  = checkpoint_dir / "embedding_meta.json"

    if not index_path.exists():
        print(f"NOTE: No VectorDB index found at {index_path}")
        print("      Run train.py to build it.")
        return None, None

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loaded VectorDB index: {index.ntotal} vectors (dim={index.d})")
    return index, meta


def query_similar(embedding: np.ndarray, index, meta: list, top_k: int = 5):
    """Return the top_k most cosine-similar entries from the VectorDB.

    Args:
        embedding: (512,) L2-normalised float32 query vector.
        index:     FAISS IndexFlatIP (inner product = cosine on unit vectors).
        meta:      metadata list aligned with index rows.
        top_k:     number of neighbours to retrieve.

    Returns:
        list of dicts — each has the stored metadata fields plus 'similarity'
        (cosine score in [-1, 1]; 1.0 = identical).
    """
    scores, indices = index.search(
        embedding.reshape(1, -1).astype(np.float32), top_k
    )
    return [
        {**meta[idx], "similarity": float(score)}
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, test_loader, device, output_dir: Path, top_n: int = 20,
                  index=None, index_meta=None, top_k_similar: int = 5):
    """Run inference, rank by POI score, query VectorDB, and save visualizations."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("Running inference on test set...")
    for input_batch, target_batch, meta_batch in tqdm(test_loader, desc="Inference"):
        input_batch = input_batch.to(device)
        predictions = model(input_batch)      # (B, 1, 64, 64)
        embeddings  = model.encode(input_batch)  # (B, 512) for VectorDB

        for i in range(input_batch.size(0)):
            input_np  = input_batch[i].cpu().numpy()         # (6, 64, 64)
            target_np = target_batch[i, 0].cpu().numpy()     # (64, 64)
            pred_np   = predictions[i, 0].cpu().numpy()      # (64, 64)

            rgb   = input_np[:3]   # (3, 64, 64)
            dem   = input_np[3]    # (64, 64)
            slope = input_np[4]    # (64, 64)

            poi_score = compute_poi_score(pred_np)

            all_results.append({
                "rgb":            rgb,
                "dem":            dem,
                "slope":          slope,
                "heatmap_true":   target_np,
                "heatmap_pred":   pred_np,
                "poi_score":      poi_score,
                "class_name":     meta_batch[i]["class_name"],
                "filepath":       meta_batch[i]["filepath"],
                "has_water":      meta_batch[i]["has_water"],
                "has_cliffs":     meta_batch[i]["has_cliffs"],
                "water_fraction": meta_batch[i]["water_fraction"],
                "max_slope":      meta_batch[i]["max_slope"],
                "embedding":      embeddings[i].cpu().numpy(),  # (512,)
            })

    # Sort by POI score (strongest cliff-water features first)
    all_results.sort(key=lambda x: x["poi_score"], reverse=True)

    # ── Print ranking table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  CLIFF-WATER POI RANKING — Top {min(top_n, len(all_results))} of {len(all_results)}")
    print(f"{'='*80}")
    print(f"  {'Rank':<6} {'POI':<8} {'Water%':<9} {'MaxSlope':<10} {'Class':<18} {'File'}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*10} {'-'*18} {'-'*25}")

    for rank, r in enumerate(all_results[:top_n], 1):
        filename = Path(r["filepath"]).name
        print(f"  {rank:<6} {r['poi_score']:<8.3f} {r['water_fraction']:<9.1%} "
              f"{r['max_slope']:<10.1f}° {r['class_name']:<18} {filename}")

    print(f"{'='*80}")

    # ── Summary statistics ───────────────────────────────────────────────
    scores   = [r["poi_score"] for r in all_results]
    n_water  = sum(1 for r in all_results if r["has_water"])
    n_cliffs = sum(1 for r in all_results if r["has_cliffs"])
    n_both   = sum(1 for r in all_results if r["has_water"] and r["has_cliffs"])

    print(f"\n  Summary:")
    print(f"  Total images:           {len(all_results)}")
    print(f"  With water detected:    {n_water} ({n_water/len(all_results):.0%})")
    print(f"  With cliffs detected:   {n_cliffs} ({n_cliffs/len(all_results):.0%})")
    print(f"  With BOTH (POI zones):  {n_both} ({n_both/len(all_results):.0%})")
    print(f"  Mean POI score:         {np.mean(scores):.4f}")
    print(f"  Max POI score:          {np.max(scores):.4f}")

    # ── Per-class breakdown ──────────────────────────────────────────────
    class_scores = {}
    for r in all_results:
        cls = r["class_name"]
        class_scores.setdefault(cls, []).append(r["poi_score"])

    print(f"\n  {'Class':<25} {'Avg POI':<12} {'Max POI':<12} {'Count'}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
        avg   = np.mean(class_scores[cls])
        mx    = np.max(class_scores[cls])
        count = len(class_scores[cls])
        bar   = "#" * int(avg * 40)
        print(f"  {cls:<25} {avg:<12.4f} {mx:<12.4f} {count:<10} |{bar}")

    # ── VectorDB similarity search ────────────────────────────────────────
    if index is not None:
        show_n = min(5, len(all_results))
        print(f"\n{'='*80}")
        print(f"  VECTORDB SIMILARITY SEARCH — top-{top_k_similar} neighbours"
              f" for top-{show_n} results")
        print(f"{'='*80}")
        for rank, r in enumerate(all_results[:show_n], 1):
            similar = query_similar(r["embedding"], index, index_meta, top_k_similar)
            print(f"\n  Rank #{rank}  {r['class_name']}  poi={r['poi_score']:.4f}"
                  f"  ({Path(r['filepath']).name})")
            print(f"  {'Sim':>6}  {'Split':<6}  {'Water':<6}  {'Cliffs':<7}  {'Class':<20}  File")
            for sim in similar:
                water_flag  = "yes" if sim.get("has_water")  else "no"
                cliffs_flag = "yes" if sim.get("has_cliffs") else "no"
                print(f"  {sim['similarity']:>6.4f}  {sim['split']:<6}  "
                      f"{water_flag:<6}  {cliffs_flag:<7}  "
                      f"{sim['class_name']:<20}  {Path(sim['filepath']).name}")
        print(f"{'='*80}")

    # ── Save top-N POI visualizations ────────────────────────────────────
    print(f"\nSaving top-{min(top_n, len(all_results))} POI visualizations...")
    for rank, r in enumerate(all_results[:top_n], 1):
        water_approx = np.zeros_like(r["dem"])  # Simplified for viz
        save_path = output_dir / f"poi_rank_{rank:02d}_{r['class_name']}.png"
        visualize_poi(
            rgb=r["rgb"],
            dem=r["dem"],
            slope=r["slope"],
            water_mask=water_approx,
            heatmap_true=r["heatmap_true"],
            heatmap_pred=r["heatmap_pred"],
            poi_score=r["poi_score"],
            save_path=str(save_path),
        )

    # ── Save ranking overview ────────────────────────────────────────────
    ranking_path = output_dir / "poi_ranking_overview.png"
    visualize_poi_ranking(
        all_results,
        top_n=min(10, len(all_results)),
        save_path=str(ranking_path),
    )
    print(f"Ranking overview saved to: {ranking_path}")

    # ── Save lowest-POI images for comparison ────────────────────────────
    print(f"\nSaving bottom-5 (lowest POI) for comparison...")
    for rank, r in enumerate(all_results[-5:], 1):
        water_approx = np.zeros_like(r["dem"])
        save_path = output_dir / f"low_poi_{rank:02d}_{r['class_name']}.png"
        visualize_poi(
            rgb=r["rgb"],
            dem=r["dem"],
            slope=r["slope"],
            water_mask=water_approx,
            heatmap_true=r["heatmap_true"],
            heatmap_pred=r["heatmap_pred"],
            poi_score=r["poi_score"],
            save_path=str(save_path),
        )

    print(f"\nAll outputs saved to: {output_dir}/")
    return all_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ElevationPOITransUNet(in_channels=6, out_channels=1).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run train.py first to train the model.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")
    if "val_iou" in checkpoint:
        print(f"  Checkpoint IoU: {checkpoint['val_iou']:.4f} (epoch {checkpoint['epoch']})")

    # Load VectorDB index from the same checkpoint directory
    checkpoint_dir = checkpoint_path.parent
    index, index_meta = load_index(checkpoint_dir)

    _, _, test_loader = get_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=0,
        use_real_dem=args.use_real_dem,
    )

    run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        index=index,
        index_meta=index_meta,
        top_k_similar=args.top_k_similar,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run elevation POI detection and ranking")
    parser.add_argument("--checkpoint",     type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir",       type=str, default="data")
    parser.add_argument("--output-dir",     type=str, default="output")
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--top-n",          type=int, default=20)
    parser.add_argument("--use-real-dem",   action="store_true", default=False)
    parser.add_argument("--top-k-similar",  type=int, default=5,
                        help="Number of VectorDB nearest neighbours to retrieve per image")
    args = parser.parse_args()
    main(args)
