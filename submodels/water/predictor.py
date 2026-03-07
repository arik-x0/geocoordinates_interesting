"""WaterPredictor: runs inference and ranks tiles by water presence score."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.predictor import BasePredictor                           # noqa: E402
from dataset import get_water_dataloaders as _get_loaders         # noqa: E402

from .model import WaterGeometryNet
from .utils import compute_water_score, visualize_water


class WaterPredictor(BasePredictor):
    score_key = "water_score"

    def get_test_loader(self):
        args = self.args
        _, _, test_loader = _get_loaders(
            data_dir=Path(args.data_dir), batch_size=args.batch_size, num_workers=0,
        )
        return test_loader

    def build_submodel(self):
        return WaterGeometryNet(out_channels=1)

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu):
        return {
            "rgb":          inputs_cpu[i],
            "label_true":   targets_cpu[i, 0],
            "label_pred":   preds_cpu[i, 0],
            "water_score":  compute_water_score(preds_cpu[i, 0]),
            "class_name":   metas[i]["class_name"],
            "filepath":     metas[i]["filepath"],
            "embedding":    embs_cpu[i],
        }

    def print_ranking(self, all_results, top_n):
        n      = len(all_results)
        scores = [r["water_score"] for r in all_results]
        print(f"\n{'='*70}")
        print(f"  WATER PRESENCE & GEOMETRY RANKING -- Top {min(top_n, n)} of {n}")
        print(f"{'='*70}")
        print(f"  {'Rank':<6} {'Score':<10} {'Class':<22} File")
        print(f"  {'-'*6} {'-'*10} {'-'*22} {'-'*28}")
        for rank, r in enumerate(all_results[:top_n], 1):
            print(f"  {rank:<6} {r['water_score']:<10.4f} "
                  f"{r['class_name']:<22} {Path(r['filepath']).name}")
        print(f"\n  Mean: {np.mean(scores):.4f}  Max: {np.max(scores):.4f}")
        class_scores: dict = {}
        for r in all_results:
            class_scores.setdefault(r["class_name"], []).append(r["water_score"])
        print(f"\n  {'Class':<25} {'Avg':>8}  Bar")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            avg = np.mean(class_scores[cls])
            print(f"  {cls:<25} {avg:>8.4f}  |{'#' * int(avg * 30)}")

    def save_visualizations(self, all_results, output_dir, top_n):
        print(f"\nSaving top-{min(top_n, len(all_results))} water visualizations...")
        for rank, r in enumerate(all_results[:top_n], 1):
            visualize_water(
                rgb=r["rgb"], label_true=r["label_true"], label_pred=r["label_pred"],
                score=r["water_score"],
                save_path=str(output_dir / f"water_{rank:02d}_{r['class_name']}.png"),
            )
