"""
HousingPredictor: runs inference with HousingEdgeCNN and ranks tiles by built-up density.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.predictor import BasePredictor                           # noqa: E402
from dataset import get_housing_dataloaders as _get_housing_loaders  # noqa: E402

from .model import HousingEdgeCNN
from .utils import (
    compute_housing_score,
    is_low_density_residential,
    visualize_housing_detection,
    visualize_housing_ranking,
    HOUSING_DENSITY_MIN,
    HOUSING_DENSITY_MAX,
)


class HousingPredictor(BasePredictor):
    """Runs inference with HousingEdgeCNN and ranks tiles by built-up density."""

    score_key = "housing_score"

    def get_test_loader(self):
        args = self.args
        _, _, test_loader = _get_housing_loaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=0,
        )
        return test_loader

    def build_submodel(self):
        return HousingEdgeCNN(out_channels=1)

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu):
        return {
            "rgb":            inputs_cpu[i],
            "label_true":     targets_cpu[i, 0],
            "label_pred":     preds_cpu[i, 0],
            "housing_score":  compute_housing_score(preds_cpu[i, 0]),
            "class_name":     metas[i]["class_name"],
            "filepath":       metas[i]["filepath"],
            "is_residential": metas[i]["is_residential"],
            "ndbi_mean":      metas[i]["ndbi_mean"],
            "embedding":      embs_cpu[i],
        }

    def print_ranking(self, all_results, top_n):
        n = len(all_results)
        low_density  = [r for r in all_results if is_low_density_residential(r["housing_score"])]
        undeveloped  = [r for r in all_results if r["housing_score"] < HOUSING_DENSITY_MIN]
        high_density = [r for r in all_results if r["housing_score"] > HOUSING_DENSITY_MAX]
        midpoint = (HOUSING_DENSITY_MIN + HOUSING_DENSITY_MAX) / 2
        low_density.sort(key=lambda r: abs(r["housing_score"] - midpoint))

        print(f"\n{'='*80}")
        print(f"  HOUSING DENSITY ANALYSIS -- {n} test images")
        print(f"  Target zone: {HOUSING_DENSITY_MIN:.0%} - {HOUSING_DENSITY_MAX:.0%} built-up")
        print(f"{'='*80}")
        print(f"  Low-density residential: {len(low_density):>5}  ({len(low_density)/n:.0%})")
        print(f"  Undeveloped / rural:     {len(undeveloped):>5}  ({len(undeveloped)/n:.0%})")
        print(f"  Dense / industrial:      {len(high_density):>5}  ({len(high_density)/n:.0%})")
        print(f"{'='*80}")
        print(f"\n  --- Top low-density residential ({min(top_n, len(low_density))} shown) ---")
        print(f"  {'Rank':<6} {'Score':<9} {'NDBI':<8} {'Residential?':<14} {'Class':<18} File")
        print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*14} {'-'*18} {'-'*25}")
        for rank, r in enumerate(low_density[:top_n], 1):
            print(f"  {rank:<6} {r['housing_score']:<9.1%} {r['ndbi_mean']:<8.3f} "
                  f"{'yes' if r['is_residential'] else 'no':<14} "
                  f"{r['class_name']:<18} {Path(r['filepath']).name}")

        class_scores: dict = {}
        for r in all_results:
            class_scores.setdefault(r["class_name"], []).append(r["housing_score"])
        print(f"\n  {'Class':<25} {'Avg Density':<14} {'In Range%':<12} {'Count'}")
        print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*10}")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            scores   = class_scores[cls]
            avg      = np.mean(scores)
            in_range = sum(1 for s in scores if is_low_density_residential(s)) / len(scores)
            print(f"  {cls:<25} {avg:<14.1%} {in_range:<12.0%} {len(scores):<10} "
                  f"|{'#' * int(avg * 30)}")
        print(f"\n  Overall mean: {np.mean([r['housing_score'] for r in all_results]):.2%}")
        print(f"{'='*80}")

    def save_visualizations(self, all_results, output_dir, top_n):
        midpoint    = (HOUSING_DENSITY_MIN + HOUSING_DENSITY_MAX) / 2
        low_density = [r for r in all_results if is_low_density_residential(r["housing_score"])]
        low_density.sort(key=lambda r: abs(r["housing_score"] - midpoint))

        print(f"\nSaving top-{min(top_n, len(low_density))} low-density residential visualizations...")
        for rank, r in enumerate(low_density[:top_n], 1):
            visualize_housing_detection(
                rgb=r["rgb"], label_true=r["label_true"], label_pred=r["label_pred"],
                housing_score=r["housing_score"],
                save_path=str(output_dir / f"low_density_{rank:02d}_{r['class_name']}.png"),
            )
        visualize_housing_ranking(
            all_results,
            top_n=min(10, len(low_density) or len(all_results)),
            save_path=str(output_dir / "housing_ranking_overview.png"),
        )
        print("Ranking overview saved.")
        print(f"\nSaving bottom-5 (densest) for comparison...")
        for rank, r in enumerate(
                sorted(all_results, key=lambda r: r["housing_score"], reverse=True)[:5], 1):
            visualize_housing_detection(
                rgb=r["rgb"], label_true=r["label_true"], label_pred=r["label_pred"],
                housing_score=r["housing_score"],
                save_path=str(output_dir / f"dense_{rank:02d}_{r['class_name']}.png"),
            )
