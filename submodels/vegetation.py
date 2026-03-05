"""
Vegetation POI submodel — lightweight 3-stage U-Net decoder + SE head.

Task: segment per-pixel vegetation/greenery from RGB satellite tiles.
Ground truth: NDVI > threshold binary mask.

Model:
    TransUNet(BaseSubmodel) — 3-stage decoder with channel-attention head.

Entry points (run from project root):
    python -m submodels.vegetation train   [--epochs N ...]
    python -m submodels.vegetation predict [--checkpoint PATH ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseSubmodel, _ConvBlock, _EMBED_DIM


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════

class TransUNet(BaseSubmodel):
    """Vegetation greenery submodel (~2.5M trainable params).

    Architecture:
        blk11 (B, 384, 14×14)  deepest ViT features
          proj11 -> dec_a (256ch @ 14×14)
          _upsample_cat with proj5(blk5) -> dec_b (128ch @ 32×32)
          _upsample_cat with proj2(blk2) -> dec_c (64ch @ 64×64)
          SE channel-attention -> head Conv(64->1) + Sigmoid
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()

        self.proj11 = nn.Conv2d(_EMBED_DIM, 256, 1)
        self.proj5  = nn.Conv2d(_EMBED_DIM, 128, 1)
        self.proj2  = nn.Conv2d(_EMBED_DIM,  64, 1)

        self.dec_a = _ConvBlock(256,       256)
        self.dec_b = _ConvBlock(256 + 128, 128)
        self.dec_c = _ConvBlock(128 +  64,  64)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, features: dict) -> torch.Tensor:
        f11 = self.proj11(features["blk11"])   # (B, 256, 14, 14)
        f5  = self.proj5(features["blk5"])     # (B, 128, 14, 14)
        f2  = self.proj2(features["blk2"])     # (B,  64, 14, 14)

        d = self.dec_a(f11)                                      # (B, 256, 14, 14)
        d = self.dec_b(self._upsample_cat(d, f5, (32, 32)))     # (B, 128, 32, 32)
        d = self.dec_c(self._upsample_cat(d, f2, (64, 64)))     # (B,  64, 64, 64)

        weights = self.se(d).unsqueeze(-1).unsqueeze(-1)         # (B, 64, 1, 1)
        return torch.sigmoid(self.head(d * weights))             # (B, 1, 64, 64)


# ════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════

from base.train import BaseTrainer
from training_utils import DiceBCELoss
from dataset import get_vegetation_dataloaders


class VegetationTrainer(BaseTrainer):
    """Trains TransUNet on frozen DINO core features for vegetation segmentation."""

    submodel_name = "TransUNet"

    def get_dataloaders(self):
        args = self.args
        return get_vegetation_dataloaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ndvi_threshold=args.ndvi_threshold,
        )

    def build_submodel(self):
        return TransUNet(out_channels=1)

    def build_criterion(self):
        return DiceBCELoss(dice_weight=0.5)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--ndvi-threshold", type=float, default=0.3,
                            help="NDVI threshold for greenery pseudo-labels")


# ════════════════════════════════════════════════════════════════════════════
# Predictor
# ════════════════════════════════════════════════════════════════════════════

from base.predict import BasePredictor
from dataset import get_vegetation_dataloaders as _get_veg_loaders


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
            "rgb":            inputs_cpu[i],
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
        print(f"\n  Mean greenery: {np.mean(scores):.1%}  "
              f"Median: {np.median(scores):.1%}  "
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
        from submodels.utils.vegetation_utils import visualize_prediction, visualize_ranking as viz_rank
        print(f"\nSaving top-{min(top_n, len(all_results))} greenery visualizations...")
        for rank, r in enumerate(all_results[:top_n], 1):
            visualize_prediction(
                rgb=r["rgb"], mask_true=r["mask_true"], mask_pred=r["mask_pred"],
                greenery_score=r["greenery_score"],
                save_path=str(output_dir / f"rank_{rank:02d}_{r['class_name']}.png"),
            )
        viz_rank(all_results, top_n=min(10, len(all_results)),
                 save_path=str(output_dir / "greenery_ranking_overview.png"))
        print(f"Ranking overview saved.")
        print(f"\nSaving bottom-5 desert-dominant images...")
        for rank, r in enumerate(all_results[-5:], 1):
            visualize_prediction(
                rgb=r["rgb"], mask_true=r["mask_true"], mask_pred=r["mask_pred"],
                greenery_score=r["greenery_score"],
                save_path=str(output_dir / f"desert_{rank:02d}_{r['class_name']}.png"),
            )


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Vegetation greenery segmentation submodel"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    tp = sub.add_parser("train", help="Train TransUNet on frozen core features")
    tp.add_argument("--data-dir",       type=str,   default="data")
    tp.add_argument("--checkpoint-dir", type=str,   default="checkpoints/vegetation")
    tp.add_argument("--epochs",         type=int,   default=25)
    tp.add_argument("--batch-size",     type=int,   default=32)
    tp.add_argument("--lr",             type=float, default=1e-3)
    tp.add_argument("--num-workers",    type=int,   default=0)
    VegetationTrainer.add_args(tp)

    # ── predict ──
    pp = sub.add_parser("predict", help="Run inference and rank by greenery score")
    pp.add_argument("--checkpoint",    type=str, default="checkpoints/vegetation/best_model.pth")
    pp.add_argument("--data-dir",      type=str, default="data")
    pp.add_argument("--output-dir",    type=str, default="output/vegetation")
    pp.add_argument("--batch-size",    type=int, default=32)
    pp.add_argument("--top-n",         type=int, default=20)
    pp.add_argument("--top-k-similar", type=int, default=5)

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.command == "train":
        VegetationTrainer(args).run()
    else:
        VegetationPredictor(args).run()
