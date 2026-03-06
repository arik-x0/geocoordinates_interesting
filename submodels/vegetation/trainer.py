"""
VegetationTrainer: trains TransUNet head on shared decoder feature maps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.trainer import BaseTrainer          # noqa: E402
from training_utils import DiceBCELoss        # noqa: E402
from dataset import get_vegetation_dataloaders  # noqa: E402

from .model import TransUNet


class VegetationTrainer(BaseTrainer):
    """Trains TransUNet head on shared decoder feature maps."""

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
