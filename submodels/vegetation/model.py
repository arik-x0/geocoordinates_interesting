"""
Vegetation greenery segmentation model.

Task: predict per-pixel vegetation probability from RGB satellite tiles.
Ground truth: NDVI > threshold binary mask.

Input to forward():
    feature_map: (B, 128, 64, 64) from CoreSatelliteModel.decode()

Architecture:
    TransUNet(BaseSubmodel) -- ConvBlock(128->64) + SE channel-attention + Conv(64->1).
    Spatial decoding (ViT tokens -> 64x64) is handled by the shared UNet decoder
    inside CoreSatelliteModel, so this head stays thin (~0.2M params).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402


class TransUNet(BaseSubmodel):
    """Vegetation greenery head (~0.2M trainable params).

    Receives the shared 64x64 feature map from the core decoder and applies
    channel-attention re-weighting before the binary segmentation output.

    Architecture:
        feature_map (B, 128, 64x64)
          -> reduce ConvBlock(128 -> 64)
          -> SE channel-attention (64 -> 16 -> 64 -> sigmoid weights)
          -> head Conv(64 -> 1) + Sigmoid
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()
        self.reduce = _ConvBlock(in_channels, 64)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.reduce(feature_map)                        # (B, 64, 64, 64)
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)          # (B, 64, 1, 1)
        return torch.sigmoid(self.head(x * w))              # (B, 1, 64, 64)
