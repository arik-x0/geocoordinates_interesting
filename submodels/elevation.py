"""
Elevation POI submodel -- dual RGB+topo stream decoder + Gaussian heatmap head.

Task: predict cliff-near-water POI heatmaps from 6-channel satellite + DEM tiles.
Ground truth: slope > threshold AND water proximity, convolved with Gaussian.

Model:
    ElevationPOITransUNet(BaseSubmodel) -- 2-stage RGB decoder fused with topo CNN.

Entry points (run from project root):
    python -m submodels.elevation train   [--epochs N ...]
    python -m submodels.elevation predict [--checkpoint PATH ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseSubmodel, _ConvBlock, _EMBED_DIM

_KERNEL_SIZE = 7
_SIGMA       = 1.5


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0)


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════

class ElevationPOITransUNet(BaseSubmodel):
    """Elevation cliff-water POI submodel (~1.5M trainable params).

    Architecture:
        RGB stream (from core features):
            blk11 (384ch) -> proj11 -> dec_a (128ch @ 14x14)
            _upsample_cat with proj5(blk5) -> dec_b (64ch @ 32x32)
            upsample -> rgb_feat (64ch @ 64x64)

        Topo stream (raw DEM+slope+aspect, trained from scratch):
            (B, 3, 64x64) -> topo_enc1 (32ch) -> topo_enc2 (64ch @ 64x64)

        Fusion:
            cat(rgb_feat, topo_feat) -> fusion ConvBlock (64ch)
            -> head Conv(64->1) -> soft activation -> Gaussian blur -> clamp [0,1]
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()

        # RGB stream
        self.proj11 = nn.Conv2d(_EMBED_DIM, 128, 1)
        self.proj5  = nn.Conv2d(_EMBED_DIM,  64, 1)
        self.dec_a  = _ConvBlock(128,       128)
        self.dec_b  = _ConvBlock(128 +  64,  64)

        # Topo stream
        self.topo_enc1 = _ConvBlock(3,  32)
        self.topo_enc2 = _ConvBlock(32, 64)

        # Fusion + head
        self.fusion = _ConvBlock(64 + 64, 64)
        self.head   = nn.Conv2d(64, out_channels, 1)

        self.register_buffer("_gauss", _gaussian_kernel(_KERNEL_SIZE, _SIGMA))

    def forward(self, features: dict, topo: torch.Tensor) -> torch.Tensor:
        # RGB stream: decode core features 14x14 -> 64x64
        f11 = self.proj11(features["blk11"])            # (B, 128, 14, 14)
        f5  = self.proj5(features["blk5"])              # (B,  64, 14, 14)

        d = self.dec_a(f11)                                       # (B, 128, 14, 14)
        d = self.dec_b(self._upsample_cat(d, f5, (32, 32)))      # (B,  64, 32, 32)
        rgb_feat = F.interpolate(d, size=(64, 64),
                                 mode="bilinear", align_corners=False)  # (B, 64, 64, 64)

        # Topo stream
        t = self.topo_enc2(self.topo_enc1(topo))        # (B,  64, 64, 64)

        # Fuse and predict
        fused  = self.fusion(torch.cat([rgb_feat, t], dim=1))   # (B, 64, 64, 64)
        logits = self.head(fused)                                # (B,  1, 64, 64)

        sig  = torch.sigmoid(logits)
        pred = sig * (1.0 + 0.1 * logits)

        padding = _KERNEL_SIZE // 2
        pred = F.conv2d(pred, self._gauss, padding=padding)
        return pred.clamp(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════

from base.train import BaseTrainer
from dataset import get_elevation_dataloaders


class HeatmapLoss(nn.Module):
    """MSE + soft-Dice loss for sparse Gaussian heatmap targets."""

    def __init__(self, mse_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.mse_weight  = mse_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss  = self.mse(pred, target)
        smooth    = 1e-6
        inter     = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)
        return self.mse_weight * mse_loss + self.dice_weight * dice_loss


class ElevationTrainer(BaseTrainer):
    """Trains ElevationPOITransUNet on frozen DINO features + raw topo channels."""

    submodel_name = "ElevationPOITransUNet"

    def get_dataloaders(self):
        args = self.args
        return get_elevation_dataloaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_real_dem=args.use_real_dem,
        )

    def build_submodel(self):
        return ElevationPOITransUNet(out_channels=1)

    def build_criterion(self):
        return HeatmapLoss(mse_weight=0.5, dice_weight=0.5)

    def rgb_slice(self, inputs):
        return inputs[:, :3]   # channels 0-2: RGB

    def extra_slice(self, inputs):
        return inputs[:, 3:]   # channels 3-5: DEM, slope, aspect

    def get_encode_fn(self, core):
        return lambda x: core.encode(x[:, :3])

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--use-real-dem", action="store_true", default=False,
                            help="Use real SRTM DEM instead of synthetic terrain")


# ════════════════════════════════════════════════════════════════════════════
# Predictor
# ════════════════════════════════════════════════════════════════════════════

from base.predict import BasePredictor
from dataset import get_elevation_dataloaders as _get_elev_loaders
from submodels.utils.elevation_utils import (
    compute_poi_score,
    visualize_poi,
    visualize_poi_ranking,
)


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
        print(f"\n  Total: {n}  |  Water: {n_water} ({n_water/n:.0%})  "
              f"|  Cliffs: {n_cliffs} ({n_cliffs/n:.0%})  "
              f"|  Both: {n_both} ({n_both/n:.0%})")
        print(f"  Mean POI: {np.mean(scores):.4f}  |  Max POI: {np.max(scores):.4f}")

        class_scores: dict = {}
        for r in all_results:
            class_scores.setdefault(r["class_name"], []).append(r["poi_score"])
        print(f"\n  {'Class':<25} {'Avg POI':<12} {'Max POI':<12} {'Count'}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            avg = np.mean(class_scores[cls])
            mx  = np.max(class_scores[cls])
            print(f"  {cls:<25} {avg:<12.4f} {mx:<12.4f} {len(class_scores[cls]):<10} "
                  f"|{'#' * int(avg * 40)}")

    def print_similarity(self, result, rank, similar):
        print(f"\n  Rank #{rank}  {result['class_name']}  poi={result['poi_score']:.4f}"
              f"  ({Path(result['filepath']).name})")
        print(f"  {'Sim':>6}  {'Split':<6}  {'Water':<6}  {'Cliffs':<7}  {'Class':<20}  File")
        for s in similar:
            wf = "yes" if s.get("has_water")  else "no"
            cf = "yes" if s.get("has_cliffs") else "no"
            print(f"  {s['similarity']:>6.4f}  {s['split']:<6}  {wf:<6}  {cf:<7}  "
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
        visualize_poi_ranking(
            all_results, top_n=min(10, len(all_results)),
            save_path=str(output_dir / "poi_ranking_overview.png"),
        )
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


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Elevation cliff-water POI detection submodel"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train ElevationPOITransUNet on frozen core features")
    tp.add_argument("--data-dir",       type=str,   default="data")
    tp.add_argument("--checkpoint-dir", type=str,   default="checkpoints/elevation")
    tp.add_argument("--epochs",         type=int,   default=25)
    tp.add_argument("--batch-size",     type=int,   default=16)
    tp.add_argument("--lr",             type=float, default=1e-3)
    tp.add_argument("--num-workers",    type=int,   default=0)
    ElevationTrainer.add_args(tp)

    pp = sub.add_parser("predict", help="Run inference and rank by POI score")
    pp.add_argument("--checkpoint",    type=str,  default="checkpoints/elevation/best_model.pth")
    pp.add_argument("--data-dir",      type=str,  default="data")
    pp.add_argument("--output-dir",    type=str,  default="output/elevation")
    pp.add_argument("--batch-size",    type=int,  default=16)
    pp.add_argument("--top-n",         type=int,  default=20)
    pp.add_argument("--top-k-similar", type=int,  default=5)
    pp.add_argument("--use-real-dem",  action="store_true", default=False)

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.command == "train":
        ElevationTrainer(args).run()
    else:
        ElevationPredictor(args).run()
