"""
Symmetry & Geometric Order submodel.

Detects agricultural fields, terraces, salt flats, and other landscapes
with strong geometric regularity — a universal cross-cultural beauty signal.

Input:  feature_map (B, 128, 64, 64) from CoreSatelliteModel.decode()
Output: (B, 1, 64, 64) geometric order heatmap

Architecture:
    Orientation-sensitive multi-directional convolutions: 4 directional
    kernels (H, V, D1, D2) capture dominant structure orientations.
    Their activations are pooled and decoded to an order heatmap. (~0.12M params)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class SymmetryOrderNet(BaseSubmodel):
    """Geometric order detection head (~0.12M params).

    Architecture:
        feature_map (B, 128, 64x64)
          -> reduce ConvBlock(128->32)
          -> 4 directional 1D convs: H(1x5), V(5x1), D1(3x3 d=1), D2(3x3 d=2)
             each outputs 16ch -> concat(64ch)
          -> ConvBlock(64->32) -> head Conv(32->1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        self.reduce = _ConvBlock(in_channels, 32)

        # Directional structure detectors
        self.dir_h  = nn.Conv2d(32, 16, (1, 5), padding=(0, 2), bias=False)
        self.dir_v  = nn.Conv2d(32, 16, (5, 1), padding=(2, 0), bias=False)
        self.dir_d1 = nn.Conv2d(32, 16, 3, padding=1, bias=False)
        self.dir_d2 = nn.Conv2d(32, 16, 3, padding=2, dilation=2, bias=False)
        self.dir_bn = nn.BatchNorm2d(64)
        self.dir_act = nn.ReLU(inplace=True)

        self.decode = _ConvBlock(64, 32)
        self.head   = nn.Conv2d(32, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.reduce(feature_map)                     # (B, 32, 64, 64)

        h  = self.dir_h(x)
        v  = self.dir_v(x)
        d1 = self.dir_d1(x)
        d2 = self.dir_d2(x)

        dirs = torch.cat([h, v, d1, d2], dim=1)          # (B, 64, 64, 64)
        dirs = self.dir_act(self.dir_bn(dirs))

        out  = self.decode(dirs)                          # (B, 32, 64, 64)
        return torch.sigmoid(self.head(out))              # (B,  1, 64, 64)
