"""
CLI entry point for the housing submodel.

Usage (run from project root):
    python -m submodels.housing train   [--epochs N --data-dir PATH ...]
    python -m submodels.housing predict [--checkpoint PATH ...]
"""

import argparse

from .trainer import HousingTrainer
from .predictor import HousingPredictor


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
