"""
Scale & Sublime submodel.

Detects grand geological formations, mountain ridges, canyons, and other
awe-inducing macro contrasts — the cognitive "scale effect" that triggers
a sense of the sublime.

Input:  feature_map (B, 128, 64, 64) from CoreSatelliteModel.decode()
Output: (B, 1, 64, 64) macro-contrast / sublime heatmap

Architecture:
    Multi-scale contrast: large average-pool residuals at 3 scales reveal
    coarse-to-fine tonal variation. Concatenated and decoded. (~0.10M params)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class ScaleSublimeNet(BaseSubmodel):
    """Large-scale contrast detection head (~0.10M params).

    Architecture:
        feature_map (B, 128, 64x64)
          -> reduce ConvBlock(128->24)
          -> 3 residual contrast branches at scales 4x4, 8x8, 16x16
             each: AvgPool -> upsample back -> subtract -> abs -> Conv1x1(24->8)
          -> concat(24ch) + reduced original(24ch) -> ConvBlock(48->32)
          -> head Conv(32->1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        self.reduce = _ConvBlock(in_channels, 24)

        self.contrast_conv4  = nn.Conv2d(24, 8, 1, bias=False)
        self.contrast_conv8  = nn.Conv2d(24, 8, 1, bias=False)
        self.contrast_conv16 = nn.Conv2d(24, 8, 1, bias=False)

        self.decode = _ConvBlock(48, 32)
        self.head   = nn.Conv2d(32, out_channels, 1)

    def _contrast(self, x: torch.Tensor, pool_size: int,
                  conv: nn.Module) -> torch.Tensor:
        """Residual between x and its avg-pooled-and-upsampled version."""
        H, W   = x.shape[2], x.shape[3]
        coarse = F.avg_pool2d(x, pool_size, stride=pool_size)
        coarse = F.interpolate(coarse, size=(H, W), mode="bilinear",
                               align_corners=False)
        diff   = torch.abs(x - coarse)
        return conv(diff)                                    # (B, 8, H, W)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x  = self.reduce(feature_map)                       # (B, 24, 64, 64)
        c4  = self._contrast(x,  4, self.contrast_conv4)
        c8  = self._contrast(x,  8, self.contrast_conv8)
        c16 = self._contrast(x, 16, self.contrast_conv16)

        fused = torch.cat([x, c4, c8, c16], dim=1)          # (B, 48, 64, 64)
        out   = self.decode(fused)                           # (B, 32, 64, 64)
        return torch.sigmoid(self.head(out))                 # (B,  1, 64, 64)
