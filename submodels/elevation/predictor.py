"""ElevationPredictor: runs inference and ranks tiles by topographic terrain beauty."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.predictor import BasePredictor                             # noqa: E402
from dataset import get_elevation_dataloaders as _get_elev_loaders  # noqa: E402

from .model import ElevationPOITransUNet
from .utils import compute_terrain_score, visualize_terrain, visualize_poi_ranking
from .utils import normalize_channel, TERRAIN_RELIEF_WINDOW
from scipy.ndimage import maximum_filter, minimum_filter


class ElevationPredictor(BasePredictor):
    """Runs inference with ElevationPOITransUNet and ranks tiles by terrain beauty score."""

    score_key = "terrain_score"

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
        return inputs[:, :6]   # Prithvi 6-band input (ch 0-5)

    def extra_slice(self, inputs):
        return inputs[:, 6:]   # topo channels: DEM, Slope, Aspect (ch 6-8)

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu):
        dem   = inputs_cpu[i, 6]   # ch6: DEM  (after 6 Prithvi bands)
        slope = inputs_cpu[i, 7]   # ch7: Slope
        dem_norm  = normalize_channel(dem)
        local_relief = maximum_filter(dem_norm, size=TERRAIN_RELIEF_WINDOW) \
                     - minimum_filter(dem_norm, size=TERRAIN_RELIEF_WINDOW)
        return {
            "rgb":           inputs_cpu[i, [2, 1, 0]],   # R=ch2(B04), G=ch1(B03), B=ch0(B02)
            "dem":           dem,
            "slope":         slope,
            "local_relief":  local_relief,
            "heatmap_true":  targets_cpu[i, 0],
            "heatmap_pred":  preds_cpu[i, 0],
            "terrain_score": compute_terrain_score(preds_cpu[i, 0]),
            "max_slope":     metas[i]["max_slope"],
            "dem_source":    metas[i]["dem_source"],
            "class_name":    metas[i]["class_name"],
            "filepath":      metas[i]["filepath"],
            "embedding":     embs_cpu[i],
        }

    def print_ranking(self, all_results, top_n):
        n      = len(all_results)
        scores = [r["terrain_score"] for r in all_results]
        print(f"\n{'='*80}")
        print(f"  TERRAIN BEAUTY RANKING -- Top {min(top_n, n)} of {n}")
        print(f"{'='*80}")
        print(f"  {'Rank':<6} {'Score':<8} {'MaxSlope':<10} {'DEM':<8} {'Class':<20} File")
        print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*20} {'-'*25}")
        for rank, r in enumerate(all_results[:top_n], 1):
            print(f"  {rank:<6} {r['terrain_score']:<8.3f} "
                  f"{r['max_slope']:<10.1f} {r['dem_source']:<8} "
                  f"{r['class_name']:<20} {Path(r['filepath']).name}")
        print(f"{'='*80}")
        print(f"\n  Mean: {np.mean(scores):.4f}  Max: {np.max(scores):.4f}")
        class_scores: dict = {}
        for r in all_results:
            class_scores.setdefault(r["class_name"], []).append(r["terrain_score"])
        print(f"\n  {'Class':<25} {'Avg':<10} {'Max':<10} Count")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            avg = np.mean(class_scores[cls])
            print(f"  {cls:<25} {avg:<10.4f} {np.max(class_scores[cls]):<10.4f} "
                  f"{len(class_scores[cls]):<8} |{'#' * int(avg * 40)}")

    def print_similarity(self, result, rank, similar):
        print(f"\n  Rank #{rank}  {result['class_name']}  "
              f"terrain={result['terrain_score']:.4f}  "
              f"({Path(result['filepath']).name})")
        print(f"  {'Sim':>6}  {'Split':<6}  {'Class':<22}  File")
        for s in similar:
            print(f"  {s['similarity']:>6.4f}  {s['split']:<6}  "
                  f"{s['class_name']:<22}  {Path(s['filepath']).name}")

    def save_visualizations(self, all_results, output_dir, top_n):
        print(f"\nSaving top-{min(top_n, len(all_results))} terrain visualizations...")
        for rank, r in enumerate(all_results[:top_n], 1):
            visualize_terrain(
                rgb=r["rgb"], dem=r["dem"], slope=r["slope"],
                local_relief=r["local_relief"],
                heatmap_true=r["heatmap_true"], heatmap_pred=r["heatmap_pred"],
                terrain_score=r["terrain_score"],
                save_path=str(output_dir / f"terrain_{rank:02d}_{r['class_name']}.png"),
            )
        visualize_poi_ranking(
            all_results, top_n=min(10, len(all_results)),
            save_path=str(output_dir / "terrain_ranking_overview.png"),
        )
        print("Ranking overview saved.")
