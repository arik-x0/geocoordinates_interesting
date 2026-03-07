"""
Fractal & Pattern Recognition submodel.

Detects self-similar, multi-scale spatial patterns (river deltas, coastlines,
tree canopies, mountain ridgelines) that correlate with mid-range fractal
dimension (D ~ 1.4) — the most aesthetically pleasing range per neuroscience.

Input:  feature_map (B, 128, 64, 64) from CoreSatelliteModel.decode()
Output: (B, 1, 64, 64) fractal richness heatmap

Architecture:
    Three parallel dilated convolutions (d=1, d=2, d=4) capture structure at
    fine, medium, and coarse scales simultaneously. Their outputs are fused and
    refined through a ConvBlock before the final heatmap head. (~0.25M params)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class FractalPatternNet(BaseSubmodel):
    """Multi-scale dilated head for fractal & pattern detection (~0.25M params).

    Architecture:
        feature_map (B, 128, 64x64)
          -> scale1 Conv(128->32, d=1) |
          -> scale2 Conv(128->32, d=2) | -> cat (B, 96, 64x64)
          -> scale4 Conv(128->32, d=4) |
          -> fuse ConvBlock(96->64) -> head Conv(64->1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        # Parallel dilated convolutions capture structure at different scales
        self.scale1 = nn.Conv2d(in_channels, 32, 3, padding=1,  dilation=1, bias=False)
        self.scale2 = nn.Conv2d(in_channels, 32, 3, padding=2,  dilation=2, bias=False)
        self.scale4 = nn.Conv2d(in_channels, 32, 3, padding=4,  dilation=4, bias=False)
        self.bn     = nn.BatchNorm2d(96)
        self.act    = nn.ReLU(inplace=True)
        self.fuse   = _ConvBlock(96, 64)
        self.head   = nn.Conv2d(64, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        s1 = self.scale1(feature_map)
        s2 = self.scale2(feature_map)
        s4 = self.scale4(feature_map)
        x  = self.act(self.bn(torch.cat([s1, s2, s4], dim=1)))  # (B, 96, 64, 64)
        x  = self.fuse(x)                                        # (B, 64, 64, 64)
        return torch.sigmoid(self.head(x))                       # (B, 1, 64, 64)
