"""
Inference and greenery ranking script.
Loads a trained TransUNet, runs segmentation on test images, computes
greenery scores, ranks images by greenery dominance, and saves
visualizations.
"""

import argparse
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


@torch.no_grad()
def run_inference(model, test_loader, device, output_dir: Path, top_n: int = 20):
    """Run inference on the test set, rank by greenery, and save visualizations."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("Running inference on test set...")
    for rgb_batch, mask_batch, meta_batch in tqdm(test_loader, desc="Inference"):
        rgb_batch = rgb_batch.to(device)
        predictions = model(rgb_batch)  # (B, 1, 64, 64)

        for i in range(rgb_batch.size(0)):
            rgb_np = rgb_batch[i].cpu().numpy()        # (3, 64, 64)
            mask_true = mask_batch[i, 0].cpu().numpy()  # (64, 64)
            mask_pred = predictions[i, 0].cpu().numpy()  # (64, 64)

            score = greenery_score_from_prediction(predictions[i])

            all_results.append({
                "rgb": rgb_np,
                "mask_true": mask_true,
                "mask_pred": mask_pred,
                "greenery_score": score,
                "class_name": meta_batch[i]["class_name"],
                "filepath": meta_batch[i]["filepath"],
            })

    # Sort by greenery score (descending — greenery areas ranked higher)
    all_results.sort(key=lambda x: x["greenery_score"], reverse=True)

    # Print ranking table
    print(f"\n{'='*70}")
    print(f"  GREENERY RANKING — Top {min(top_n, len(all_results))} of {len(all_results)} images")
    print(f"{'='*70}")
    print(f"  {'Rank':<6} {'Score':<10} {'Class':<20} {'File'}")
    print(f"  {'-'*6} {'-'*10} {'-'*20} {'-'*30}")

    for rank, result in enumerate(all_results[:top_n], 1):
        filename = Path(result["filepath"]).name
        print(f"  {rank:<6} {result['greenery_score']:<10.1%} "
              f"{result['class_name']:<20} {filename}")

    print(f"{'='*70}")

    # Print summary statistics
    scores = [r["greenery_score"] for r in all_results]
    print(f"\n  Mean greenery score:   {np.mean(scores):.1%}")
    print(f"  Median greenery score: {np.median(scores):.1%}")
    print(f"  Images > 50% green:   {sum(1 for s in scores if s > 0.5)}/{len(scores)}")
    print(f"  Images < 10% green:   {sum(1 for s in scores if s < 0.1)}/{len(scores)} (desert-dominant)")

    # Class-level greenery breakdown
    class_scores = {}
    for r in all_results:
        cls = r["class_name"]
        class_scores.setdefault(cls, []).append(r["greenery_score"])

    print(f"\n  {'Class':<25} {'Avg Greenery':<15} {'Count'}")
    print(f"  {'-'*25} {'-'*15} {'-'*10}")
    for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
        avg = np.mean(class_scores[cls])
        count = len(class_scores[cls])
        bar = "#" * int(avg * 20)
        print(f"  {cls:<25} {avg:<15.1%} {count:<10} |{bar}")

    # Save individual visualizations for top-N greenest images
    print(f"\nSaving top-{min(top_n, len(all_results))} greenery visualizations...")
    for rank, result in enumerate(all_results[:top_n], 1):
        save_path = output_dir / f"rank_{rank:02d}_{result['class_name']}.png"
        visualize_prediction(
            rgb=result["rgb"],
            mask_true=result["mask_true"],
            mask_pred=result["mask_pred"],
            greenery_score=result["greenery_score"],
            save_path=str(save_path),
        )

    # Save ranking overview
    ranking_path = output_dir / "greenery_ranking_overview.png"
    visualize_ranking(all_results, top_n=min(10, len(all_results)),
                      save_path=str(ranking_path))
    print(f"Ranking overview saved to: {ranking_path}")

    # Save bottom-N (most desert-like) for comparison
    print(f"\nSaving bottom-{min(5, len(all_results))} desert-dominant visualizations...")
    for rank, result in enumerate(all_results[-5:], 1):
        save_path = output_dir / f"desert_{rank:02d}_{result['class_name']}.png"
        visualize_prediction(
            rgb=result["rgb"],
            mask_true=result["mask_true"],
            mask_pred=result["mask_pred"],
            greenery_score=result["greenery_score"],
            save_path=str(save_path),
        )

    print(f"\nAll outputs saved to: {output_dir}/")
    return all_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
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

    # Load test data
    _, _, test_loader = get_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Run inference and ranking
    run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run greenery segmentation and ranking")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing EuroSAT dataset")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save visualizations")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top greenery images to visualize")
    args = parser.parse_args()
    main(args)
