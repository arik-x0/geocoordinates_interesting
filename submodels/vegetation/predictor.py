"""
VegetationPredictor: runs inference with TransUNet and ranks tiles by greenery score.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.predictor import BasePredictor                           # noqa: E402
from dataset import get_vegetation_dataloaders as _get_veg_loaders  # noqa: E402

from .model import TransUNet
from .utils import visualize_prediction, visualize_ranking as viz_rank


def _greenery_score(pred_np: np.ndarray, threshold: float = 0.5) -> float:
    return float((pred_np > threshold).sum()) / pred_np.size


class VegetationPredictor(BasePredictor):
    """Runs inference with TransUNet and ranks tiles by greenery coverage."""

    score_key = "greenery_score"

    def get_test_loader(self):
        args = self.args
        _, _, test_loader = _get_veg_loaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=0,
        )
        return test_loader

    def build_submodel(self):
        return TransUNet(out_channels=1)

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu):
        return {
            "rgb":            inputs_cpu[i, [2, 1, 0]],   # R=ch2(B04), G=ch1(B03), B=ch0(B02)
            "mask_true":      targets_cpu[i, 0],
            "mask_pred":      preds_cpu[i, 0],
            "greenery_score": _greenery_score(preds_cpu[i, 0]),
            "class_name":     metas[i]["class_name"],
            "filepath":       metas[i]["filepath"],
            "embedding":      embs_cpu[i],
        }

    def print_ranking(self, all_results, top_n):
        scores = [r["greenery_score"] for r in all_results]
        n = len(all_results)
        print(f"\n{'='*70}")
        print(f"  GREENERY RANKING -- Top {min(top_n, n)} of {n} images")
        print(f"{'='*70}")
        print(f"  {'Rank':<6} {'Score':<10} {'Class':<20} {'File'}")
        print(f"  {'-'*6} {'-'*10} {'-'*20} {'-'*30}")
        for rank, r in enumerate(all_results[:top_n], 1):
            print(f"  {rank:<6} {r['greenery_score']:<10.1%} "
                  f"{r['class_name']:<20} {Path(r['filepath']).name}")
        print(f"{'='*70}")
        print(f"\n  Mean: {np.mean(scores):.1%}  Median: {np.median(scores):.1%}  "
              f">50%: {sum(s > 0.5 for s in scores)}/{n}  "
              f"<10%: {sum(s < 0.1 for s in scores)}/{n}")
        class_scores: dict = {}
        for r in all_results:
            class_scores.setdefault(r["class_name"], []).append(r["greenery_score"])
        print(f"\n  {'Class':<25} {'Avg':<15} {'Count'}")
        print(f"  {'-'*25} {'-'*15} {'-'*10}")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            avg = np.mean(class_scores[cls])
            print(f"  {cls:<25} {avg:<15.1%} {len(class_scores[cls]):<10} "
                  f"|{'#' * int(avg * 20)}")

    def save_visualizations(self, all_results, output_dir, top_n):
        print(f"\nSaving top-{min(top_n, len(all_results))} greenery visualizations...")
        for rank, r in enumerate(all_results[:top_n], 1):
            visualize_prediction(
                rgb=r["rgb"], mask_true=r["mask_true"], mask_pred=r["mask_pred"],
                greenery_score=r["greenery_score"],
                save_path=str(output_dir / f"rank_{rank:02d}_{r['class_name']}.png"),
            )
        viz_rank(all_results, top_n=min(10, len(all_results)),
                 save_path=str(output_dir / "greenery_ranking_overview.png"))
        print("Ranking overview saved.")
        print(f"\nSaving bottom-5 desert-dominant images...")
        for rank, r in enumerate(all_results[-5:], 1):
            visualize_prediction(
                rgb=r["rgb"], mask_true=r["mask_true"], mask_pred=r["mask_pred"],
                greenery_score=r["greenery_score"],
                save_path=str(output_dir / f"desert_{rank:02d}_{r['class_name']}.png"),
            )
