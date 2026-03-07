"""ColorHarmonyTrainer: trains ColorHarmonyNet on shared decoder feature maps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.trainer import BaseTrainer              # noqa: E402
from training_utils import DiceBCELoss           # noqa: E402
from dataset import get_color_harmony_dataloaders  # noqa: E402

from .model import ColorHarmonyNet


class ColorHarmonyTrainer(BaseTrainer):
    submodel_name = "ColorHarmonyNet"

    def get_dataloaders(self):
        args = self.args
        return get_color_harmony_dataloaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    def build_submodel(self):
        return ColorHarmonyNet(out_channels=1)

    def build_criterion(self):
        return DiceBCELoss(dice_weight=0.5)
