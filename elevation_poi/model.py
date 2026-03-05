"""
Elevation POI submodel — RGB decoder + topographic encoder + Gaussian heatmap head.
All layers trained from scratch.

Task: predict cliff-near-water POI heatmaps from 6-channel satellite + DEM tiles.
Ground truth: slope > threshold AND water proximity, convolved with a Gaussian kernel.
Scored as mean activation in top-10% of heatmap pixels.

This submodel uniquely fuses two information streams:
  1. RGB stream  — from the frozen CoreSatelliteModel (ViT features of the satellite
                   image), decoded from 16×16 → 64×64 via a lightweight U-Net decoder.
  2. Topo stream — 3-channel input (DEM elevation + slope + aspect) processed by
                   a small CNN encoder that learns topographic patterns from scratch.

The streams are fused at 64×64 and passed through a Gaussian heatmap head that
enforces smooth, spatially coherent output matching the Gaussian ground-truth labels.

Input to forward():
    features: dict from CoreSatelliteModel.extract_features()  (RGB features)
              Key used: 'blk11' (B,384,16,16), 'blk5' (B,384,16,16)
    topo:     (B, 3, 64, 64) — DEM elevation + slope + aspect channels

Output:
    (B, 1, 64, 64) — per-pixel POI probability heatmap [0, 1]

Output metadata populated by predict.py:
    poi_score    — float [0, 1]: mean heatmap in top-10% pixels
    dem_source   — str: 'real' or 'synthetic'
    slope_mean   — float: mean slope angle for the tile
    ndwi_mean    — float: mean NDWI (water index) for the tile
    class_name   — str: EuroSAT land-cover class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EMBED_DIM   = 384   # matches CoreSatelliteModel._EMBED_DIM
_KERNEL_SIZE = 7
_SIGMA       = 1.5


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Create a (1, 1, size, size) Gaussian blur kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0)


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


class ElevationPOITransUNet(nn.Module):
    """Elevation cliff-water POI submodel.

    Architecture (trained from scratch):

        RGB stream (from core features):
            blk11 (B, 384, 16×16)  deepest ViT features
              → proj11: Conv2d(384→128)
              → dec_a: ConvBlock(128→128) @ 16×16
              → upsample × 2  →  32×32
              ↕ cat skip: proj5(blk5) @ 32×32 (64ch)
              → dec_b: ConvBlock(128+64→64) @ 32×32
              → upsample × 2  →  64×64

        Topo stream (raw DEM+slope+aspect):
            (B, 3, 64, 64)
              → topo_enc1: ConvBlock(3→32) @ 64×64
              → topo_enc2: ConvBlock(32→64) @ 64×64

        Fusion:
            cat(rgb_feat: B×64×64×64,  topo_feat: B×64×64×64)
              → fusion: ConvBlock(128→64) @ 64×64
              → head: Conv2d(64→1)
              → soft activation: σ(x)·(1 + 0.1·x)   avoids hard saturation
              → Gaussian blur (7×7, σ=1.5, fixed)    enforces spatial coherence
              → clamp [0, 1]

    Approximately 1.5M trainable parameters.
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()

        # ── RGB stream (processes core features) ──────────────────────────
        self.proj11 = nn.Conv2d(_EMBED_DIM, 128, 1)
        self.proj5  = nn.Conv2d(_EMBED_DIM,  64, 1)

        self.dec_a = _ConvBlock(128, 128)               # 16×16
        self.dec_b = _ConvBlock(128 + 64, 64)           # 32×32  (upsample + skip)

        # ── Topo stream (processes DEM + slope + aspect directly) ──────────
        self.topo_enc1 = _ConvBlock(3, 32)              # 64×64
        self.topo_enc2 = _ConvBlock(32, 64)             # 64×64

        # ── Fusion and heatmap head ────────────────────────────────────────
        self.fusion = _ConvBlock(64 + 64, 64)           # cat(rgb, topo) → 64ch
        self.head   = nn.Conv2d(64, out_channels, 1)

        # Fixed Gaussian blur kernel — non-trainable, matches ground-truth generation
        self.register_buffer("_gauss", _gaussian_kernel(_KERNEL_SIZE, _SIGMA))

    def forward(self, features: dict, topo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: dict from CoreSatelliteModel.extract_features()  (RGB features)
            topo:     (B, 3, 64, 64) — DEM elevation + slope + aspect channels

        Returns:
            (B, 1, 64, 64) per-pixel cliff-water POI heatmap [0, 1]
        """
        # ── RGB stream: decode core features from 16×16 → 64×64 ──────────
        f11 = self.proj11(features["blk11"])   # (B, 128, 16, 16)
        f5  = self.proj5(features["blk5"])     # (B,  64, 16, 16)

        d = self.dec_a(f11)                    # (B, 128, 16, 16)

        d = F.interpolate(d, size=(32, 32), mode="bilinear", align_corners=False)
        f5_32 = F.interpolate(f5, size=(32, 32), mode="bilinear", align_corners=False)
        d = self.dec_b(torch.cat([d, f5_32], dim=1))   # (B, 64, 32, 32)

        d = F.interpolate(d, size=(64, 64), mode="bilinear", align_corners=False)
        rgb_feat = d                            # (B, 64, 64, 64)

        # ── Topo stream: encode DEM + slope + aspect ──────────────────────
        t = self.topo_enc1(topo)               # (B, 32, 64, 64)
        t = self.topo_enc2(t)                  # (B, 64, 64, 64)

        # ── Fuse and predict ──────────────────────────────────────────────
        fused  = self.fusion(torch.cat([rgb_feat, t], dim=1))   # (B, 64, 64, 64)
        logits = self.head(fused)              # (B, 1, 64, 64)

        # Soft activation: avoids hard saturation on sparse heatmap targets
        sig  = torch.sigmoid(logits)
        pred = sig * (1.0 + 0.1 * logits)     # slightly > 1.0 for high-confidence peaks

        # Fixed Gaussian blur: enforces spatial coherence matching label generation
        padding = _KERNEL_SIZE // 2
        pred = F.conv2d(pred, self._gauss, padding=padding)

        return pred.clamp(0.0, 1.0)


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
    topo = randn(B, 3, 64, 64)

    model = ElevationPOITransUNet()
    out = model(dummy_features, topo)
    print(f"ElevationPOITransUNet (elevation submodel)")
    print(f"  Trainable params: {count_parameters(model):,}")
    print(f"  Output shape:     {tuple(out.shape)}")
    print(f"  Output range:     [{out.min():.3f}, {out.max():.3f}]")
    print(f"\n  Input channels: blk11/blk5 from core (RGB) + topo (DEM+slope+aspect)")
