"""
Housing POI submodel -- two-scale HED-style detection head.

Task: detect per-pixel built-up structure probability from RGB satellite tiles.
Ground truth: NDBI-derived binary mask.

Input to forward():
    feature_map: (B, 128, 64, 64) from CoreSatelliteModel.decode()

Model:
    HousingEdgeCNN(BaseSubmodel) -- direct side output + depthwise edge-sharpened
    side output fused via a 1x1 conv. Thin head; spatial decoding is shared.

Entry points (run from project root):
    python -m submodels.housing train   [--epochs N ...]
    python -m submodels.housing predict [--checkpoint PATH ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseSubmodel
from core.model import _DEC_CHANNELS


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════

class HousingEdgeCNN(BaseSubmodel):
    """Housing structure detection head (~0.07M trainable params).

    Produces two complementary predictions from the shared feature map and
    fuses them: a direct spatial side output and an edge-sharpened side output.

    Architecture:
        feature_map (B, 128, 64x64)
          -> side_main: Conv(128 -> 1)           -- global structure signal
          -> edge_conv: depthwise(128) + side_edge: Conv(128 -> 1)  -- edge signal
          -> fusion: Conv(2 -> 1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        self.side_main = nn.Conv2d(in_channels, 1, 1)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.side_edge = nn.Conv2d(in_channels, 1, 1)
        self.fusion    = nn.Conv2d(2, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        s_main = self.side_main(feature_map)               # (B, 1, 64, 64)
        s_edge = self.side_edge(self.edge_conv(feature_map))  # (B, 1, 64, 64)
        return torch.sigmoid(self.fusion(torch.cat([s_main, s_edge], dim=1)))


# ════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════

from base.train import BaseTrainer
from training_utils import DiceBCELoss
from dataset import get_housing_dataloaders


class HousingTrainer(BaseTrainer):
    """Trains HousingEdgeCNN head on shared decoder feature maps."""

    submodel_name = "HousingEdgeCNN"

    def get_dataloaders(self):
        args = self.args
        return get_housing_dataloaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    def build_submodel(self):
        return HousingEdgeCNN(out_channels=1)

    def build_criterion(self):
        return DiceBCELoss(dice_weight=0.5)


# ════════════════════════════════════════════════════════════════════════════
# Predictor
# ════════════════════════════════════════════════════════════════════════════

from base.predict import BasePredictor
from dataset import get_housing_dataloaders as _get_housing_loaders
from submodels.utils.housing_utils import (
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


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def _build_parser():
    parser = argparse.ArgumentParser(description="Housing structure detection submodel")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train HousingEdgeCNN head on shared decoder features")
    tp.add_argument("--data-dir",       type=str,   default="data")
    tp.add_argument("--checkpoint-dir", type=str,   default="checkpoints/housing")
    tp.add_argument("--epochs",         type=int,   default=25)
    tp.add_argument("--batch-size",     type=int,   default=16)
    tp.add_argument("--lr",             type=float, default=1e-3)
    tp.add_argument("--num-workers",    type=int,   default=0)

    pp = sub.add_parser("predict", help="Run inference and rank by housing density")
    pp.add_argument("--checkpoint",    type=str, default="checkpoints/housing/best_model.pth")
    pp.add_argument("--data-dir",      type=str, default="data")
    pp.add_argument("--output-dir",    type=str, default="output/housing")
    pp.add_argument("--batch-size",    type=int, default=16)
    pp.add_argument("--top-n",         type=int, default=20)
    pp.add_argument("--top-k-similar", type=int, default=5)

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.command == "train":
        HousingTrainer(args).run()
    else:
        HousingPredictor(args).run()
