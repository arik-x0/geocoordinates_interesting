"""
CLI entry point for the elevation submodel.

Usage (run from project root):
    python -m submodels.elevation train   [--epochs N --use-real-dem ...]
    python -m submodels.elevation predict [--checkpoint PATH ...]
"""

import argparse

from .trainer import ElevationTrainer
from .predictor import ElevationPredictor


def _build_parser():
    parser = argparse.ArgumentParser(description="Elevation cliff-water POI detection submodel")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train ElevationPOITransUNet head on shared decoder features")
    tp.add_argument("--data-dir",       type=str,   default="data")
    tp.add_argument("--checkpoint-dir", type=str,   default="checkpoints/elevation")
    tp.add_argument("--epochs",         type=int,   default=25)
    tp.add_argument("--batch-size",     type=int,   default=16)
    tp.add_argument("--lr",             type=float, default=1e-3)
    tp.add_argument("--num-workers",    type=int,   default=0)
    ElevationTrainer.add_args(tp)

    pp = sub.add_parser("predict", help="Run inference and rank by POI score")
    pp.add_argument("--checkpoint",    type=str,  default="checkpoints/elevation/best_model.pth")
    pp.add_argument("--data-dir",      type=str,  default="output/elevation")
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
