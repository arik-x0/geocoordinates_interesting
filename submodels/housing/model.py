"""
Housing structure detection model.

Task: detect per-pixel built-up structure probability from RGB satellite tiles.
Ground truth: NDBI-derived binary mask.

Input to forward():
    feature_map: (B, 128, 64, 64) from CoreSatelliteModel.decode()

Architecture:
    HousingEdgeCNN(BaseSubmodel) -- direct side output + depthwise edge-sharpened
    side output fused via a 1x1 conv. Thin head (~0.07M params); spatial decoding
    is handled by the shared UNet decoder in CoreSatelliteModel.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel  # noqa: E402
from core.model import _DEC_CHANNELS   # noqa: E402


class HousingEdgeCNN(BaseSubmodel):
    """Housing structure detection head (~0.07M trainable params).

    Produces two complementary predictions from the shared feature map and
    fuses them: a direct spatial side output and an edge-sharpened side output.

    Architecture:
        feature_map (B, 128, 64x64)
          -> side_main: Conv(128 -> 1)                       -- global structure
          -> edge_conv: depthwise(128) + side_edge: Conv(128 -> 1)  -- edge signal
          -> fusion: Conv(2 -> 1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        self.side_main = nn.Conv2d(in_channels, 1, 1)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.side_edge = nn.Conv2d(in_channels, 1, 1)
        self.fusion    = nn.Conv2d(2, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        s_main = self.side_main(feature_map)                      # (B, 1, 64, 64)
        s_edge = self.side_edge(self.edge_conv(feature_map))      # (B, 1, 64, 64)
        return torch.sigmoid(self.fusion(torch.cat([s_main, s_edge], dim=1)))
