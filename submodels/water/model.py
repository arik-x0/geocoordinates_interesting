"""
Water Presence & Geometry submodel.

Detects open water bodies (lakes, rivers, coastlines, reservoirs) and their
geometric form — the biophilia hypothesis identifies water as a primary
aesthetic trigger in satellite imagery.

Input:  feature_map (B, 128, 64, 64) from CoreSatelliteModel.decode()
Output: (B, 1, 64, 64) water presence + geometry heatmap

Architecture:
    SE channel-attention head similar to TransUNet but with an added spatial
    smoothing path (avg-pool branch) that captures blob geometry rather than
    just per-pixel water probability. (~0.18M params)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class WaterGeometryNet(BaseSubmodel):
    """Water presence & geometry head (~0.18M params).

    Architecture:
        feature_map (B, 128, 64x64)
          -> reduce ConvBlock(128->48)
          -> SE channel-attention (48->12->48)           -- highlights water features
          -> geometry branch: AvgPool3x3 -> Conv(48->48) -- captures blob shape
          -> fuse concat(48+48) -> ConvBlock(96->48)
          -> head Conv(48->1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        self.reduce = _ConvBlock(in_channels, 48)

        # Channel attention for feature selection
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 48),
            nn.Sigmoid(),
        )

        # Spatial geometry branch — smoothed to capture water body shape
        self.geom = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(48, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.fuse = _ConvBlock(96, 48)
        self.head = nn.Conv2d(48, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x   = self.reduce(feature_map)                      # (B, 48, 64, 64)
        w   = self.se(x).unsqueeze(-1).unsqueeze(-1)        # (B, 48, 1, 1)
        attn = x * w                                        # (B, 48, 64, 64)
        geom = self.geom(x)                                 # (B, 48, 64, 64)
        fused = self.fuse(torch.cat([attn, geom], dim=1))   # (B, 48, 64, 64)
        return torch.sigmoid(self.head(fused))              # (B,  1, 64, 64)
