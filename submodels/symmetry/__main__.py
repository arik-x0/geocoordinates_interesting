"""CLI: python -m submodels.symmetry train|predict"""
import argparse
from .trainer import SymmetryTrainer
from .predictor import SymmetryPredictor


def _build_parser():
    parser = argparse.ArgumentParser(description="Symmetry & Geometric Order submodel")
    sub = parser.add_subparsers(dest="command", required=True)
    tp = sub.add_parser("train")
    tp.add_argument("--data-dir",       type=str,   default="data")
    tp.add_argument("--checkpoint-dir", type=str,   default="checkpoints/symmetry")
    tp.add_argument("--epochs",         type=int,   default=25)
    tp.add_argument("--batch-size",     type=int,   default=32)
    tp.add_argument("--lr",             type=float, default=1e-3)
    tp.add_argument("--num-workers",    type=int,   default=0)
    pp = sub.add_parser("predict")
    pp.add_argument("--checkpoint",    type=str, default="checkpoints/symmetry/best_model.pth")
    pp.add_argument("--data-dir",      type=str, default="data")
    pp.add_argument("--output-dir",    type=str, default="output/symmetry")
    pp.add_argument("--batch-size",    type=int, default=32)
    pp.add_argument("--top-n",         type=int, default=20)
    pp.add_argument("--top-k-similar", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.command == "train":
        SymmetryTrainer(args).run()
    else:
        SymmetryPredictor(args).run()
