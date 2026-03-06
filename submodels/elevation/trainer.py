"""
ElevationTrainer: trains ElevationPOITransUNet head on shared decoder features + topo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.trainer import BaseTrainer          # noqa: E402
from dataset import get_elevation_dataloaders  # noqa: E402

from .model import ElevationPOITransUNet, HeatmapLoss


class ElevationTrainer(BaseTrainer):
    """Trains ElevationPOITransUNet head on shared decoder features + topo."""

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
        return inputs[:, :3]

    def extra_slice(self, inputs):
        return inputs[:, 3:]

    def get_encode_fn(self, core):
        return lambda x: core.encode(x[:, :3])

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--use-real-dem", action="store_true", default=False,
                            help="Use real SRTM DEM instead of synthetic terrain")
