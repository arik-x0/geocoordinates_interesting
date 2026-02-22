"""
Inference and greenery ranking script for the vegetation TransUNet.
Loads a trained model, runs segmentation on test images, ranks images by
greenery score, queries the FAISS VectorDB for visually similar training
images, and saves visualizations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import TransUNet
from dataset import get_dataloaders
from utils import (
    greenery_score_from_prediction,
    visualize_prediction,
    visualize_ranking,
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
def run_inference(model, test_loader, device, output_dir: Path,
                  top_n: int = 20, index=None, index_meta=None,
                  top_k_similar: int = 5):
    """Run inference, rank by greenery score, query VectorDB, save visualizations."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("Running inference on test set...")
    for rgb_batch, mask_batch, meta_batch in tqdm(test_loader, desc="Inference"):
        rgb_batch = rgb_batch.to(device)
        predictions = model(rgb_batch)              # (B, 1, 64, 64)
        embeddings  = model.encode(rgb_batch)       # (B, 512) for VectorDB

        for i in range(rgb_batch.size(0)):
            rgb_np    = rgb_batch[i].cpu().numpy()          # (3, 64, 64)
            mask_true = mask_batch[i, 0].cpu().numpy()      # (64, 64)
            mask_pred = predictions[i, 0].cpu().numpy()     # (64, 64)
            score     = greenery_score_from_prediction(predictions[i])

            all_results.append({
                "rgb":            rgb_np,
                "mask_true":      mask_true,
                "mask_pred":      mask_pred,
                "greenery_score": score,
                "class_name":     meta_batch[i]["class_name"],
                "filepath":       meta_batch[i]["filepath"],
                "embedding":      embeddings[i].cpu().numpy(),  # (512,)
            })

    # Sort by greenery score descending
    all_results.sort(key=lambda x: x["greenery_score"], reverse=True)

    # ── Ranking table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GREENERY RANKING — Top {min(top_n, len(all_results))} of {len(all_results)} images")
    print(f"{'='*70}")
    print(f"  {'Rank':<6} {'Score':<10} {'Class':<20} {'File'}")
    print(f"  {'-'*6} {'-'*10} {'-'*20} {'-'*30}")

    for rank, r in enumerate(all_results[:top_n], 1):
        filename = Path(r["filepath"]).name
        print(f"  {rank:<6} {r['greenery_score']:<10.1%} {r['class_name']:<20} {filename}")

    print(f"{'='*70}")

    scores = [r["greenery_score"] for r in all_results]
    print(f"\n  Mean greenery score:   {np.mean(scores):.1%}")
    print(f"  Median greenery score: {np.median(scores):.1%}")
    print(f"  Images > 50% green:    {sum(1 for s in scores if s > 0.5)}/{len(scores)}")
    print(f"  Images < 10% green:    {sum(1 for s in scores if s < 0.1)}/{len(scores)}")

    class_scores: dict = {}
    for r in all_results:
        class_scores.setdefault(r["class_name"], []).append(r["greenery_score"])

    print(f"\n  {'Class':<25} {'Avg Greenery':<15} {'Count'}")
    print(f"  {'-'*25} {'-'*15} {'-'*10}")
    for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
        avg   = np.mean(class_scores[cls])
        count = len(class_scores[cls])
        bar   = "#" * int(avg * 20)
        print(f"  {cls:<25} {avg:<15.1%} {count:<10} |{bar}")

    # ── VectorDB similarity search ────────────────────────────────────────
    if index is not None:
        show_n = min(5, len(all_results))
        print(f"\n{'='*70}")
        print(f"  VECTORDB SIMILARITY SEARCH — top-{top_k_similar} neighbours"
              f" for top-{show_n} results")
        print(f"{'='*70}")
        for rank, r in enumerate(all_results[:show_n], 1):
            similar = query_similar(r["embedding"], index, index_meta, top_k_similar)
            print(f"\n  Rank #{rank}  {r['class_name']}  score={r['greenery_score']:.1%}"
                  f"  ({Path(r['filepath']).name})")
            print(f"  {'Sim':>6}  {'Split':<6}  {'Class':<20}  File")
            for sim in similar:
                print(f"  {sim['similarity']:>6.4f}  {sim['split']:<6}  "
                      f"{sim['class_name']:<20}  {Path(sim['filepath']).name}")
        print(f"{'='*70}")

    # ── Save visualizations ───────────────────────────────────────────────
    print(f"\nSaving top-{min(top_n, len(all_results))} greenery visualizations...")
    for rank, r in enumerate(all_results[:top_n], 1):
        save_path = output_dir / f"rank_{rank:02d}_{r['class_name']}.png"
        visualize_prediction(
            rgb=r["rgb"],
            mask_true=r["mask_true"],
            mask_pred=r["mask_pred"],
            greenery_score=r["greenery_score"],
            save_path=str(save_path),
        )

    ranking_path = output_dir / "greenery_ranking_overview.png"
    visualize_ranking(all_results, top_n=min(10, len(all_results)),
                      save_path=str(ranking_path))
    print(f"Ranking overview saved to: {ranking_path}")

    print(f"\nSaving bottom-{min(5, len(all_results))} desert-dominant visualizations...")
    for rank, r in enumerate(all_results[-5:], 1):
        save_path = output_dir / f"desert_{rank:02d}_{r['class_name']}.png"
        visualize_prediction(
            rgb=r["rgb"],
            mask_true=r["mask_true"],
            mask_pred=r["mask_pred"],
            greenery_score=r["greenery_score"],
            save_path=str(save_path),
        )

    print(f"\nAll outputs saved to: {output_dir}/")
    return all_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransUNet(in_channels=3, out_channels=1).to(device)

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
        description="Run greenery segmentation, ranking, and VectorDB similarity search"
    )
    parser.add_argument("--checkpoint",     type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir",       type=str, default="data")
    parser.add_argument("--output-dir",     type=str, default="output")
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--top-n",          type=int, default=20)
    parser.add_argument("--top-k-similar",  type=int, default=5,
                        help="Number of VectorDB nearest neighbours to retrieve per image")
    args = parser.parse_args()
    main(args)
