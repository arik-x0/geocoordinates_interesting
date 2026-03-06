"""
Color Harmony submodel.

Detects pure chromatic richness and spectral diversity — intentionally
decoupled from NDVI so the Vegetation model owns plant-matter and this
model owns spectral/colour variety. High scores on vivid croplands in
bloom, semi-arid ochre terrain, autumn canopies, and turquoise coastlines.

Input:  feature_map (B, 128, 64, 64) from CoreSatelliteModel.decode()
Output: (B, 1, 64, 64) chromatic richness heatmap

Architecture:
    Dual-path attention: chromatic SE-attention path + depthwise spectral-
    spread path. Fused and decoded to heatmap. (~0.15M params)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class ColorHarmonyNet(BaseSubmodel):
    """Color harmony & vegetation density head (~0.15M params).

    Architecture:
        feature_map (B, 128, 64x64)
          -> chromatic path: ConvBlock(128->48) + channel-wise std attention
          -> vegetation path: ConvBlock(128->48) + spatial avg smoothing
          -> fuse concat(96) -> ConvBlock(96->48) -> head Conv(48->1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        # Chromatic richness path
        self.chroma_reduce = _ConvBlock(in_channels, 48)
        self.chroma_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 48),
            nn.Sigmoid(),
        )

        # Vegetation structure path — captures spatial density patterns
        self.veg_reduce = _ConvBlock(in_channels, 48)
        self.veg_smooth = nn.Sequential(
            nn.Conv2d(48, 48, 5, padding=2, groups=48, bias=False),  # depthwise
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.fuse = _ConvBlock(96, 48)
        self.head = nn.Conv2d(48, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # Chromatic path
        c = self.chroma_reduce(feature_map)
        w = self.chroma_attn(c).unsqueeze(-1).unsqueeze(-1)
        c = c * w                                            # (B, 48, 64, 64)

        # Vegetation path
        v = self.veg_reduce(feature_map)
        v = self.veg_smooth(v)                               # (B, 48, 64, 64)

        fused = self.fuse(torch.cat([c, v], dim=1))          # (B, 48, 64, 64)
        return torch.sigmoid(self.head(fused))               # (B,  1, 64, 64)
