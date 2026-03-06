"""
HousingTrainer: trains HousingEdgeCNN head on shared decoder feature maps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.trainer import BaseTrainer        # noqa: E402
from training_utils import DiceBCELoss      # noqa: E402
from dataset import get_housing_dataloaders  # noqa: E402

from .model import HousingEdgeCNN


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
