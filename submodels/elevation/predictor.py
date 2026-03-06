"""
ElevationPredictor: runs inference with ElevationPOITransUNet and ranks tiles by POI score.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.predictor import BasePredictor                             # noqa: E402
from dataset import get_elevation_dataloaders as _get_elev_loaders  # noqa: E402

from .model import ElevationPOITransUNet
from .utils import compute_poi_score, visualize_poi, visualize_poi_ranking


class ElevationPredictor(BasePredictor):
    """Runs inference with ElevationPOITransUNet and ranks tiles by POI score."""

    score_key = "poi_score"

    def get_test_loader(self):
        args = self.args
        _, _, test_loader = _get_elev_loaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=0,
            use_real_dem=args.use_real_dem,
        )
        return test_loader

    def build_submodel(self):
        return ElevationPOITransUNet(out_channels=1)

    def rgb_slice(self, inputs):
        return inputs[:, :3]

    def extra_slice(self, inputs):
        return inputs[:, 3:]

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu):
        return {
            "rgb":            inputs_cpu[i, :3],
            "dem":            inputs_cpu[i, 3],
            "slope":          inputs_cpu[i, 4],
            "heatmap_true":   targets_cpu[i, 0],
            "heatmap_pred":   preds_cpu[i, 0],
            "poi_score":      compute_poi_score(preds_cpu[i, 0]),
            "class_name":     metas[i]["class_name"],
            "filepath":       metas[i]["filepath"],
            "has_water":      metas[i]["has_water"],
            "has_cliffs":     metas[i]["has_cliffs"],
            "water_fraction": metas[i]["water_fraction"],
            "max_slope":      metas[i]["max_slope"],
            "embedding":      embs_cpu[i],
        }

    def print_ranking(self, all_results, top_n):
        n      = len(all_results)
        scores = [r["poi_score"] for r in all_results]
        n_water  = sum(1 for r in all_results if r["has_water"])
        n_cliffs = sum(1 for r in all_results if r["has_cliffs"])
        n_both   = sum(1 for r in all_results if r["has_water"] and r["has_cliffs"])
        print(f"\n{'='*80}")
        print(f"  CLIFF-WATER POI RANKING -- Top {min(top_n, n)} of {n}")
        print(f"{'='*80}")
        print(f"  {'Rank':<6} {'POI':<8} {'Water%':<9} {'MaxSlope':<10} {'Class':<18} File")
        print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*10} {'-'*18} {'-'*25}")
        for rank, r in enumerate(all_results[:top_n], 1):
            print(f"  {rank:<6} {r['poi_score']:<8.3f} {r['water_fraction']:<9.1%} "
                  f"{r['max_slope']:<10.1f} {r['class_name']:<18} {Path(r['filepath']).name}")
        print(f"{'='*80}")
        print(f"\n  Water: {n_water}/{n} ({n_water/n:.0%})  Cliffs: {n_cliffs}/{n}  "
              f"Both: {n_both}/{n}  Mean POI: {np.mean(scores):.4f}")
        class_scores: dict = {}
        for r in all_results:
            class_scores.setdefault(r["class_name"], []).append(r["poi_score"])
        print(f"\n  {'Class':<25} {'Avg POI':<12} {'Max POI':<12} {'Count'}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            avg = np.mean(class_scores[cls])
            print(f"  {cls:<25} {avg:<12.4f} {np.max(class_scores[cls]):<12.4f} "
                  f"{len(class_scores[cls]):<10} |{'#' * int(avg * 40)}")

    def print_similarity(self, result, rank, similar):
        print(f"\n  Rank #{rank}  {result['class_name']}  poi={result['poi_score']:.4f}"
              f"  ({Path(result['filepath']).name})")
        print(f"  {'Sim':>6}  {'Split':<6}  {'Water':<6}  {'Cliffs':<7}  {'Class':<20}  File")
        for s in similar:
            print(f"  {s['similarity']:>6.4f}  {s['split']:<6}  "
                  f"{'yes' if s.get('has_water') else 'no':<6}  "
                  f"{'yes' if s.get('has_cliffs') else 'no':<7}  "
                  f"{s['class_name']:<20}  {Path(s['filepath']).name}")

    def save_visualizations(self, all_results, output_dir, top_n):
        print(f"\nSaving top-{min(top_n, len(all_results))} POI visualizations...")
        for rank, r in enumerate(all_results[:top_n], 1):
            visualize_poi(
                rgb=r["rgb"], dem=r["dem"], slope=r["slope"],
                water_mask=np.zeros_like(r["dem"]),
                heatmap_true=r["heatmap_true"], heatmap_pred=r["heatmap_pred"],
                poi_score=r["poi_score"],
                save_path=str(output_dir / f"poi_rank_{rank:02d}_{r['class_name']}.png"),
            )
        visualize_poi_ranking(all_results, top_n=min(10, len(all_results)),
                              save_path=str(output_dir / "poi_ranking_overview.png"))
        print("Ranking overview saved.")
        print(f"\nSaving bottom-5 (lowest POI) for comparison...")
        for rank, r in enumerate(all_results[-5:], 1):
            visualize_poi(
                rgb=r["rgb"], dem=r["dem"], slope=r["slope"],
                water_mask=np.zeros_like(r["dem"]),
                heatmap_true=r["heatmap_true"], heatmap_pred=r["heatmap_pred"],
                poi_score=r["poi_score"],
                save_path=str(output_dir / f"low_poi_{rank:02d}_{r['class_name']}.png"),
            )
