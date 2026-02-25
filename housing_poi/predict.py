"""
Inference and housing density ranking script.

Loads a trained HousingEdgeCNN, runs structure detection on test images,
computes per-image housing density scores, and identifies images in the
low-density residential range (5%–20% built-up coverage).

Images are ranked in three groups:
  1. Low-density residential (5–20%) — the target POI zone
  2. Undeveloped / rural (<5%)
  3. Dense / industrial (>20%)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Shared dataset module lives at the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_housing_dataloaders as get_dataloaders  # noqa: E402

from model import HousingEdgeCNN
from utils import (
    compute_housing_score,
    housing_score_from_tensor,
    is_low_density_residential,
    density_label,
    visualize_housing_detection,
    visualize_housing_ranking,
    HOUSING_DENSITY_MIN,
    HOUSING_DENSITY_MAX,
)


def load_index(checkpoint_dir: Path):
    """Load FAISS embedding index and metadata from a checkpoint directory.

    Returns (index, meta_list) or (None, None) if unavailable.
    """
    try:
        import faiss
    except ImportError:
        return None, None

    index_path = checkpoint_dir / "embedding_index.faiss"
    meta_path  = checkpoint_dir / "embedding_meta.json"

    if not index_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loaded VectorDB index: {index.ntotal} vectors from {index_path}")
    return index, meta


def query_similar(embedding: np.ndarray, index, meta: List[dict],
                  top_k: int = 5) -> List[dict]:
    """Return the top_k most similar training images for a query embedding."""
    query = embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k + 1)   # +1 to skip self if present

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = {**meta[idx], "similarity": float(score)}
        results.append(entry)
        if len(results) == top_k:
            break
    return results


@torch.no_grad()
def run_inference(model, test_loader, device, output_dir: Path, top_n: int = 20,
                  index=None, index_meta: Optional[List[dict]] = None,
                  top_k_similar: int = 5):
    """Run inference, rank by housing density, and save visualisations."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("Running inference on test set...")
    for rgb_batch, label_batch, meta_batch in tqdm(test_loader, desc="Inference"):
        rgb_batch = rgb_batch.to(device)
        predictions = model(rgb_batch)   # (B, 1, 64, 64)
        embeddings  = model.encode(rgb_batch).cpu().numpy()  # (B, 512)

        for i in range(rgb_batch.size(0)):
            rgb_np       = rgb_batch[i].cpu().numpy()         # (3, 64, 64)
            label_true   = label_batch[i, 0].cpu().numpy()    # (64, 64)
            label_pred   = predictions[i, 0].cpu().numpy()    # (64, 64)

            score = compute_housing_score(label_pred)

            all_results.append({
                "rgb":            rgb_np,
                "label_true":     label_true,
                "label_pred":     label_pred,
                "housing_score":  score,
                "class_name":     meta_batch[i]["class_name"],
                "filepath":       meta_batch[i]["filepath"],
                "is_residential": meta_batch[i]["is_residential"],
                "ndbi_mean":      meta_batch[i]["ndbi_mean"],
                "embedding":      embeddings[i],
            })

    # ── Partition results ────────────────────────────────────────────────────
    low_density  = [r for r in all_results if is_low_density_residential(r["housing_score"])]
    undeveloped  = [r for r in all_results if r["housing_score"] < HOUSING_DENSITY_MIN]
    high_density = [r for r in all_results if r["housing_score"] > HOUSING_DENSITY_MAX]

    # Sort low-density by score (closest to midpoint first)
    midpoint = (HOUSING_DENSITY_MIN + HOUSING_DENSITY_MAX) / 2
    low_density.sort(key=lambda r: abs(r["housing_score"] - midpoint))
    undeveloped.sort(key=lambda r: r["housing_score"], reverse=True)
    high_density.sort(key=lambda r: r["housing_score"])

    # ── Print ranking table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  HOUSING DENSITY ANALYSIS — {len(all_results)} test images")
    print(f"  Target zone: {HOUSING_DENSITY_MIN:.0%} – {HOUSING_DENSITY_MAX:.0%} built-up coverage")
    print(f"{'='*80}")
    print(f"  Low-density residential:  {len(low_density):>5}  ({len(low_density)/len(all_results):.0%})")
    print(f"  Undeveloped / rural:      {len(undeveloped):>5}  ({len(undeveloped)/len(all_results):.0%})")
    print(f"  Dense / industrial:       {len(high_density):>5}  ({len(high_density)/len(all_results):.0%})")
    print(f"{'='*80}")

    print(f"\n  --- Top low-density residential ({min(top_n, len(low_density))} shown) ---")
    print(f"  {'Rank':<6} {'Score':<9} {'NDBI':<8} {'Residential?':<14} {'Class':<18} File")
    print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*14} {'-'*18} {'-'*25}")

    for rank, r in enumerate(low_density[:top_n], 1):
        filename = Path(r["filepath"]).name
        print(f"  {rank:<6} {r['housing_score']:<9.1%} {r['ndbi_mean']:<8.3f} "
              f"{'yes' if r['is_residential'] else 'no':<14} {r['class_name']:<18} {filename}")

    # ── Per-class statistics ─────────────────────────────────────────────────
    class_scores: dict = {}
    for r in all_results:
        class_scores.setdefault(r["class_name"], []).append(r["housing_score"])

    print(f"\n  {'Class':<25} {'Avg Density':<14} {'In Range%':<12} {'Count'}")
    print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*10}")
    for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
        scores = class_scores[cls]
        avg = np.mean(scores)
        in_range = sum(1 for s in scores if is_low_density_residential(s)) / len(scores)
        bar = "#" * int(avg * 30)
        print(f"  {cls:<25} {avg:<14.1%} {in_range:<12.0%} {len(scores):<10} |{bar}")

    print(f"\n  Overall mean housing score: {np.mean([r['housing_score'] for r in all_results]):.2%}")
    print(f"{'='*80}")

    # ── VectorDB similarity search ───────────────────────────────────────────
    if index is not None and index_meta is not None:
        print(f"\n  --- Top-{top_k_similar} similar training images for top-3 low-density results ---")
        for rank, r in enumerate(low_density[:3], 1):
            similar = query_similar(r["embedding"], index, index_meta, top_k=top_k_similar)
            print(f"\n  Query #{rank}: {Path(r['filepath']).name}  "
                  f"(density={r['housing_score']:.1%})")
            for j, s in enumerate(similar, 1):
                print(f"    {j}. sim={s['similarity']:.3f}  "
                      f"{s['class_name']:<20}  {Path(s['filepath']).name}")

    # ── Save visualisations ──────────────────────────────────────────────────
    print(f"\nSaving top-{min(top_n, len(low_density))} low-density residential visualisations...")
    for rank, r in enumerate(low_density[:top_n], 1):
        save_path = output_dir / f"low_density_{rank:02d}_{r['class_name']}.png"
        visualize_housing_detection(
            rgb=r["rgb"],
            label_true=r["label_true"],
            label_pred=r["label_pred"],
            housing_score=r["housing_score"],
            save_path=str(save_path),
        )

    # Ranking overview grid
    ranking_path = output_dir / "housing_ranking_overview.png"
    visualize_housing_ranking(
        all_results,
        top_n=min(10, len(low_density) or len(all_results)),
        save_path=str(ranking_path),
    )
    print(f"Ranking overview saved to: {ranking_path}")

    # Bottom examples: highest-density images for contrast
    print(f"\nSaving bottom-5 (densest / most industrial) for comparison...")
    for rank, r in enumerate(sorted(all_results,
                                    key=lambda r: r["housing_score"],
                                    reverse=True)[:5], 1):
        save_path = output_dir / f"dense_{rank:02d}_{r['class_name']}.png"
        visualize_housing_detection(
            rgb=r["rgb"],
            label_true=r["label_true"],
            label_pred=r["label_pred"],
            housing_score=r["housing_score"],
            save_path=str(save_path),
        )

    print(f"\nAll outputs saved to: {output_dir}/")
    return all_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = HousingEdgeCNN(in_channels=3, out_channels=1).to(device)

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

    # Load VectorDB index if available
    checkpoint_dir = checkpoint_path.parent
    index, index_meta = load_index(checkpoint_dir)

    # Load test data
    _, _, test_loader = get_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=0,
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
    parser = argparse.ArgumentParser(
        description="Run housing density detection and low-density residential ranking"
    )
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir",    type=str, default="data")
    parser.add_argument("--output-dir",  type=str, default="output")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--top-n",          type=int, default=20,
                        help="Number of low-density residential images to visualise")
    parser.add_argument("--top-k-similar",  type=int, default=5,
                        help="Number of similar training images to show via VectorDB")
    args = parser.parse_args()
    main(args)
