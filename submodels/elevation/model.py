"""
Elevation POI detection model.

Task: predict cliff-near-water POI heatmaps from 6-channel satellite + DEM tiles.
Ground truth: slope > threshold AND water proximity, convolved with Gaussian.

Input to forward():
    feature_map: (B, 128, 64, 64) from CoreSatelliteModel.decode()
    topo:        (B,   3, 64, 64) DEM elevation + slope + aspect (pre-normalised)

Architecture:
    ElevationPOITransUNet(BaseSubmodel) -- projects shared feature map to 64ch,
    encodes raw topo channels with a small CNN, fuses both streams, predicts
    a Gaussian-smoothed heatmap (~0.35M params). Spatial decoding is done by
    the shared UNet decoder in CoreSatelliteModel.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base.submodel import BaseSubmodel, _ConvBlock  # noqa: E402
from core.model import _DEC_CHANNELS                # noqa: E402

_KERNEL_SIZE = 7
_SIGMA       = 1.5


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0)


class ElevationPOITransUNet(BaseSubmodel):
    """Elevation cliff-water POI fusion head (~0.35M trainable params).

    Two input streams at 64x64:
        RGB stream:  proj(feature_map 128ch) -> 64ch
        Topo stream: small CNN on raw DEM+slope+aspect -> 64ch (trained from scratch)

    Architecture:
        feature_map (B, 128, 64x64) -> proj Conv(128->64) -> rgb_feat (B, 64, 64x64)
        topo        (B,   3, 64x64) -> topo_enc1(32) -> topo_enc2(64) -> topo_feat
        cat([rgb_feat, topo_feat])   -> fusion ConvBlock(128->64)
        -> head Conv(64->1) -> soft activation -> Gaussian blur -> clamp [0,1]
    """

    def __init__(self, in_channels: int = _DEC_CHANNELS, out_channels: int = 1):
        super().__init__()

        self.proj      = nn.Conv2d(in_channels, 64, 1)
        self.topo_enc1 = _ConvBlock(3,  32)
        self.topo_enc2 = _ConvBlock(32, 64)
        self.fusion    = _ConvBlock(64 + 64, 64)
        self.head      = nn.Conv2d(64, out_channels, 1)

        self.register_buffer("_gauss", _gaussian_kernel(_KERNEL_SIZE, _SIGMA))

    def forward(self, feature_map: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
        rgb_feat  = self.proj(feature_map)                # (B, 64, 64, 64)
        topo_feat = self.topo_enc2(self.topo_enc1(topo))  # (B, 64, 64, 64)

        fused  = self.fusion(torch.cat([rgb_feat, topo_feat], dim=1))
        logits = self.head(fused)

        sig  = torch.sigmoid(logits)
        pred = sig * (1.0 + 0.1 * logits)
        pred = F.conv2d(pred, self._gauss, padding=_KERNEL_SIZE // 2)
        return pred.clamp(0.0, 1.0)


class HeatmapLoss(nn.Module):
    """MSE + soft-Dice loss for sparse Gaussian heatmap targets."""

    def __init__(self, mse_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.mse_weight  = mse_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss  = self.mse(pred, target)
        smooth    = 1e-6
        inter     = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)
        return self.mse_weight * mse_loss + self.dice_weight * dice_loss
