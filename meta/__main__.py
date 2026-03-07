"""CLI: python -m meta predict"""
import argparse
from .predictor import AestheticPredictor


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Aesthetic Meta-Aggregator: fuses all 9 submodels")
    sub = parser.add_subparsers(dest="command", required=True)
    pp = sub.add_parser("predict", help="Run full aesthetic pipeline")
    pp.add_argument("--data-dir",        type=str, default="data",
                    help="EuroSAT data root")
    pp.add_argument("--checkpoint-dir",  type=str, default="checkpoints",
                    help="Root directory containing per-submodel checkpoint folders")
    pp.add_argument("--output-dir",      type=str, default="output/aesthetic",
                    help="Where to save ranked visualizations")
    pp.add_argument("--batch-size",      type=int, default=32)
    pp.add_argument("--top-n",           type=int, default=20,
                    help="Number of top-ranked images to visualize")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    AestheticPredictor(args).run()
