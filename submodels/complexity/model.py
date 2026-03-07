"""
Complexity Balance submodel.

Detects regions with optimal information complexity — not too ordered,
not too chaotic. Based on Berlyne's (1974) complexity-preference curve
and fractal dimension research (D≈1.3 is most preferred).

Input:  feature_map (B, 128, 64, 64) from CoreSatelliteModel.decode()
Output: (B, 1, 64, 64) optimal-complexity heatmap

Architecture:
    Two-path design: an "order" path (large receptive field, smooth) and
    a "chaos" path (local response, high frequency). The bell-shaped
    interaction between them captures mid-range complexity. (~0.13M params)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class ComplexityBalanceNet(BaseSubmodel):
    """Optimal-complexity detection head (~0.13M params).

    Architecture:
        feature_map (B, 128, 64x64)
          -> order path:  ConvBlock(128->24) + large AvgPool(7x7) + upsample
          -> chaos path:  ConvBlock(128->24) + 3x3 conv with high response
          -> interaction: concat(48) -> ConvBlock(48->32) -> head Conv(32->1)
          -> Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        # Order path — smooth, large-scale structure
        self.order_reduce = _ConvBlock(in_channels, 24)
        self.order_smooth = nn.AvgPool2d(7, stride=1, padding=3)

        # Chaos path — local high-frequency response
        self.chaos_reduce = _ConvBlock(in_channels, 24)
        self.chaos_local  = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # Bell interaction: subtract paths, measure "distance from extreme"
        self.interact = nn.Sequential(
            nn.Conv2d(48, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.decode = _ConvBlock(32, 32)
        self.head   = nn.Conv2d(32, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        order = self.order_smooth(self.order_reduce(feature_map))  # (B, 24, 64, 64)
        chaos = self.chaos_local(self.chaos_reduce(feature_map))   # (B, 24, 64, 64)

        fused  = self.interact(torch.cat([order, chaos], dim=1))   # (B, 32, 64, 64)
        out    = self.decode(fused)
        return torch.sigmoid(self.head(out))                       # (B,  1, 64, 64)
