"""
Vegetation POI submodel — lightweight decoder + SE head trained from scratch.

Task: segment per-pixel vegetation/greenery from RGB satellite tiles.
Ground truth: NDVI > threshold binary mask; scored as greenery_score
(fraction of high-greenery pixels) to identify parks, forests, farmland.

The model receives multi-depth feature maps from the frozen CoreSatelliteModel
(core/model.py) and applies its own trainable decoder to upsample from
16×16 → 32×32 → 64×64.  A Squeeze-and-Excitation head re-weights the
64 decoder channels by their global importance for the vegetation prediction,
amplifying channels that track green-leaf spectral signatures.

Input to forward():
    features: dict from CoreSatelliteModel.extract_features()
              Keys used: 'blk11' (B,384,16,16), 'blk5' (B,384,16,16),
                         'blk2'  (B,384,16,16)

Output:
    (B, 1, 64, 64) — per-pixel greenery probability [0, 1]

Output metadata populated by predict.py:
    greenery_score  — float [0, 1]: fraction of pixels above threshold
    ndvi_mean       — float: mean NDVI for the tile
    is_vegetated    — bool: EuroSAT class label (Forest / HerbaceousVeg / etc.)
    class_name      — str: EuroSAT land-cover class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EMBED_DIM = 384   # matches CoreSatelliteModel._EMBED_DIM


class _ConvBlock(nn.Module):
    """Double 3×3 conv with BN+ReLU."""

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


class TransUNet(nn.Module):
    """Vegetation greenery submodel — lightweight U-Net decoder + SE head.

    Architecture (trained from scratch on CoreSatelliteModel features):

        blk11 (B, 384, 16×16)  deepest ViT features
          → proj11: Conv2d(384→256)
          → dec_a:  ConvBlock(256→256) @ 16×16
          → upsample × 2   →  32×32
          ↕ cat skip from blk5 projected to 128ch
          → dec_b: ConvBlock(256+128→128) @ 32×32
          → upsample × 2   →  64×64
          ↕ cat skip from blk2 projected to 64ch
          → dec_c: ConvBlock(128+64→64) @ 64×64
          → SE attention: (AdaptiveAvgPool→Linear 64→16→64→Sigmoid)
          → head: Conv2d(64→1) + Sigmoid   →  (B, 1, 64, 64)

    Approximately 2.5M trainable parameters.
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()

        # Project ViT features to U-Net channel sizes
        self.proj11 = nn.Conv2d(_EMBED_DIM, 256, 1)
        self.proj5  = nn.Conv2d(_EMBED_DIM, 128, 1)
        self.proj2  = nn.Conv2d(_EMBED_DIM,  64, 1)

        # Lightweight U-Net decoder (3 stages)
        self.dec_a = _ConvBlock(256, 256)              # 16×16
        self.dec_b = _ConvBlock(256 + 128, 128)        # 32×32  (after upsample + skip)
        self.dec_c = _ConvBlock(128 +  64,  64)        # 64×64  (after upsample + skip)

        # SE channel-attention (reduction ratio = 4: 64 → 16 → 64)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )

        # Final prediction head
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: dict from CoreSatelliteModel.extract_features()

        Returns:
            (B, 1, 64, 64) per-pixel greenery probability [0, 1]
        """
        # Project ViT features to U-Net dimensions
        f11 = self.proj11(features["blk11"])   # (B, 256, 16, 16)
        f5  = self.proj5(features["blk5"])     # (B, 128, 16, 16)
        f2  = self.proj2(features["blk2"])     # (B,  64, 16, 16)

        # Decoder stage A: process deepest features at 16×16
        d = self.dec_a(f11)                    # (B, 256, 16, 16)

        # Decoder stage B: upsample → 32×32, fuse with mid-level skip
        d = F.interpolate(d, size=(32, 32), mode="bilinear", align_corners=False)
        f5_32 = F.interpolate(f5, size=(32, 32), mode="bilinear", align_corners=False)
        d = self.dec_b(torch.cat([d, f5_32], dim=1))   # (B, 128, 32, 32)

        # Decoder stage C: upsample → 64×64, fuse with fine skip
        d = F.interpolate(d, size=(64, 64), mode="bilinear", align_corners=False)
        f2_64 = F.interpolate(f2, size=(64, 64), mode="bilinear", align_corners=False)
        d = self.dec_c(torch.cat([d, f2_64], dim=1))   # (B, 64, 64, 64)

        # SE channel-attention: re-weight channels by global relevance
        weights = self.se(d).unsqueeze(-1).unsqueeze(-1)   # (B, 64, 1, 1)
        d = d * weights

        return torch.sigmoid(self.head(d))   # (B, 1, 64, 64)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch import randn

    # Simulate core feature dict
    B = 2
    dummy_features = {
        "blk2":  randn(B, 384, 16, 16),
        "blk5":  randn(B, 384, 16, 16),
        "blk8":  randn(B, 384, 16, 16),
        "blk11": randn(B, 384, 16, 16),
        "cls":   randn(B, 384),
    }

    model = TransUNet()
    out = model(dummy_features)
    print(f"TransUNet (vegetation submodel)")
    print(f"  Trainable params: {count_parameters(model):,}")
    print(f"  Output shape:     {tuple(out.shape)}")
    print(f"  Output range:     [{out.min():.3f}, {out.max():.3f}]")
