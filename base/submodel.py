"""
Shared building blocks for all three POI submodels.

All submodels inherit from BaseSubmodel and use _ConvBlock as their
decoder primitive.  The _upsample_cat static helper eliminates the
repeated bilinear-upsample + interpolate-skip + concatenate pattern
that appears at every decoder stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EMBED_DIM = 384   # must match CoreSatelliteModel._EMBED_DIM


class _ConvBlock(nn.Module):
    """Double 3x3 conv with BN+ReLU -- shared decoder primitive."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaseSubmodel(nn.Module):
    """Base class for all POI task submodels.

    Provides _upsample_cat: the common pattern of bilinear-upsampling the
    current decoder state and a skip-connection feature map to a target
    spatial size, then concatenating them on the channel dimension.

    Subclasses define their own projection layers (proj*), decoder
    ConvBlocks (dec_*), and output heads, then compose them using
    _upsample_cat inside forward().
    """

    @staticmethod
    def _upsample_cat(d: torch.Tensor, f_skip: torch.Tensor,
                      size: tuple) -> torch.Tensor:
        """Upsample d and f_skip to size, then cat on the channel dimension."""
        d      = F.interpolate(d,      size=size, mode="bilinear", align_corners=False)
        f_skip = F.interpolate(f_skip, size=size, mode="bilinear", align_corners=False)
        return torch.cat([d, f_skip], dim=1)

    def forward(self, features: dict, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
