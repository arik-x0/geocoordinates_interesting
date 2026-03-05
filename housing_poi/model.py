"""
Housing POI submodel — lightweight multi-scale decoder + edge head trained from scratch.

Task: detect per-pixel built-up structure probability from RGB satellite tiles.
Ground truth: NDBI-derived binary mask; scored as fraction of built-up pixels
within the low-density residential range (5%–20%) to identify housing zones.

The model receives multi-depth feature maps from the frozen CoreSatelliteModel
(core/model.py) and decodes them through a lightweight U-Net that produces
multi-scale side outputs at 16×16, 32×32, and 64×64 spatial resolution.
A depthwise 3×3 edge-sharpening conv on the finest features sharpens building
boundary detection before the final fusion.  Multi-scale side outputs are
fused by a learnable 1×1 convolution, consistent with the HED principle.

Input to forward():
    features: dict from CoreSatelliteModel.extract_features()
              Keys used: 'blk11' (B,384,16,16), 'blk8' (B,384,16,16),
                         'blk2'  (B,384,16,16)

Output:
    (B, 1, 64, 64) — per-pixel structure probability [0, 1]

Output metadata populated by predict.py:
    housing_score    — float [0, 1]: fraction of pixels in built-up range
    density_label    — str: 'low-density' / 'dense' / 'rural'
    is_residential   — bool: True if score in 5%–20% range
    ndbi_mean        — float: mean NDBI for the tile
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


class HousingEdgeCNN(nn.Module):
    """Housing structure submodel — multi-scale U-Net decoder + HED-style fusion.

    Architecture (trained from scratch on CoreSatelliteModel features):

        blk11 (B, 384, 16×16)  — deepest ViT features (building shapes)
          → proj11: Conv2d(384→256)
          → dec_a:  ConvBlock(256→256) @ 16×16
          → side_16: Conv2d(256→1) → upsample 64×64

          → upsample × 2  →  32×32
          ↕ cat skip: proj8(blk8) at 32×32
          → dec_b: ConvBlock(256+128→128) @ 32×32
          → side_32: Conv2d(128→1) → upsample 64×64

          → upsample × 2  →  64×64
          ↕ cat skip: proj2(blk2) at 64×64
          → dec_c: ConvBlock(128+64→64) @ 64×64
          → edge_conv: DepthwiseConv(64→64, 3×3) + BN + ReLU
          → side_64: Conv2d(64→1) @ 64×64

        Fusion: cat(side_16, side_32, side_64) → Conv2d(3→1) + Sigmoid

    Approximately 2.2M trainable parameters.
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()

        # Project ViT features to U-Net channel sizes
        self.proj11 = nn.Conv2d(_EMBED_DIM, 256, 1)
        self.proj8  = nn.Conv2d(_EMBED_DIM, 128, 1)
        self.proj2  = nn.Conv2d(_EMBED_DIM,  64, 1)

        # Lightweight U-Net decoder
        self.dec_a = _ConvBlock(256, 256)               # 16×16
        self.dec_b = _ConvBlock(256 + 128, 128)         # 32×32
        self.dec_c = _ConvBlock(128 +  64,  64)         # 64×64

        # Side output projections (one per decoder stage)
        self.side_16 = nn.Conv2d(256, 1, 1)
        self.side_32 = nn.Conv2d(128, 1, 1)

        # Depthwise edge-sharpening on finest features — learns per-channel
        # Laplacian-like filters for crisp building boundary detection
        self.edge_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.side_64 = nn.Conv2d(64, 1, 1)

        # Learnable fusion of all three side outputs
        self.fusion = nn.Conv2d(3, out_channels, 1)

    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: dict from CoreSatelliteModel.extract_features()

        Returns:
            (B, 1, 64, 64) per-pixel structure probability [0, 1]
        """
        # Project ViT features to U-Net dimensions
        f11 = self.proj11(features["blk11"])   # (B, 256, 16, 16)
        f8  = self.proj8(features["blk8"])     # (B, 128, 16, 16)
        f2  = self.proj2(features["blk2"])     # (B,  64, 16, 16)

        # Decoder stage A: 16×16
        d = self.dec_a(f11)                    # (B, 256, 16, 16)
        s16 = F.interpolate(self.side_16(d), size=(64, 64),
                            mode="bilinear", align_corners=False)

        # Decoder stage B: upsample → 32×32, fuse with mid-level skip
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        f8_32 = F.interpolate(f8, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec_b(torch.cat([d, f8_32], dim=1))   # (B, 128, 32, 32)
        s32 = F.interpolate(self.side_32(d), size=(64, 64),
                            mode="bilinear", align_corners=False)

        # Decoder stage C: upsample → 64×64, fuse with fine skip
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        f2_64 = F.interpolate(f2, scale_factor=4, mode="bilinear", align_corners=False)
        d = self.dec_c(torch.cat([d, f2_64], dim=1))   # (B, 64, 64, 64)

        # Edge-sharpened side output at full resolution
        d_edge = self.edge_conv(d)
        s64 = self.side_64(d_edge)             # (B, 1, 64, 64)

        # Fuse all three side outputs
        fused = torch.cat([s16, s32, s64], dim=1)   # (B, 3, 64, 64)
        return torch.sigmoid(self.fusion(fused))      # (B, 1, 64, 64)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch import randn

    B = 2
    dummy_features = {
        "blk2":  randn(B, 384, 16, 16),
        "blk5":  randn(B, 384, 16, 16),
        "blk8":  randn(B, 384, 16, 16),
        "blk11": randn(B, 384, 16, 16),
        "cls":   randn(B, 384),
    }

    model = HousingEdgeCNN()
    out = model(dummy_features)
    print(f"HousingEdgeCNN (housing submodel)")
    print(f"  Trainable params: {count_parameters(model):,}")
    print(f"  Output shape:     {tuple(out.shape)}")
    print(f"  Output range:     [{out.min():.3f}, {out.max():.3f}]")
