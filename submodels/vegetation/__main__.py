"""
CLI entry point for the vegetation submodel.

Usage (run from project root):
    python -m submodels.vegetation train   [--epochs N --data-dir PATH ...]
    python -m submodels.vegetation predict [--checkpoint PATH ...]
"""

import argparse

from .trainer import VegetationTrainer
from .predictor import VegetationPredictor


def _build_parser():
    parser = argparse.ArgumentParser(description="Vegetation greenery segmentation submodel")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train TransUNet head on shared decoder features")
    tp.add_argument("--data-dir",       type=str,   default="data")
    tp.add_argument("--checkpoint-dir", type=str,   default="checkpoints/vegetation")
    tp.add_argument("--epochs",         type=int,   default=25)
    tp.add_argument("--batch-size",     type=int,   default=32)
    tp.add_argument("--lr",             type=float, default=1e-3)
    tp.add_argument("--num-workers",    type=int,   default=0)
    VegetationTrainer.add_args(tp)

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
