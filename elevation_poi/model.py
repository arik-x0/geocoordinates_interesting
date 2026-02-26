"""
Elevation POI model — EVA-02 ViT-UNet with Gaussian heatmap output head.

Task: predict cliff-near-water POI heatmaps from 6-channel satellite + DEM tiles.
Ground truth: slope > threshold AND water proximity, convolved with a Gaussian
to produce a smooth continuous heatmap rather than a hard binary mask.

Why a Gaussian heatmap head?
Unlike housing or vegetation (binary masks), the elevation POI target is a
sparse, continuous heatmap — most pixels are zero, and the signal appears as
smooth peaks centred on cliff-water interface locations.  A standard sigmoid
head (used for dense binary segmentation) has steep gradients near 0.5 and
pushes predictions towards 0 or 1, which fights the smooth, continuous nature
of the heatmap target and makes it difficult to learn intermediate confidence
values at POI boundaries.

The Gaussian heatmap head addresses this with two changes:
  1. The raw logit is passed through a *soft activation* that avoids
     saturating at 0 for non-POI pixels:  act(x) = sigmoid(x) * (1 + 0.1*x)
     This produces gentler gradients than plain sigmoid, allowing the model
     to maintain a wide, smooth probability hill around POI peaks rather than
     a hard step.
  2. A fixed 7×7 Gaussian blur (σ=1.5) is applied to the activation output,
     enforcing spatial coherence.  This mirrors the way the ground-truth
     heatmap is generated (Gaussian-blurred cliff+water mask) so the model's
     output space matches its training target.

The channel adapter (Conv2d 6→3) fuses RGB + DEM elevation + slope + aspect
before the ViT so all topographic cues are available to the self-attention
mechanism throughout the full transformer depth.

Output metadata (populated by predict.py):
    poi_score    — float [0, 1]: mean of top-10% highest heatmap values
    dem_source   — str: "real" or "synthetic" (DEM availability flag)
    slope_mean   — float: mean terrain slope for the tile (degrees)
    ndwi_mean    — float: mean NDWI (water-body proximity proxy)
    class_name   — str: EuroSAT land-cover class
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pretrained_model import ViTUNetPOI, count_parameters  # noqa: F401


def _gaussian_kernel(kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Return a normalised 2-D Gaussian kernel as a (1, 1, k, k) tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)          # (k, k)
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)


class ElevationPOITransUNet(ViTUNetPOI):
    """EVA-02 ViT-UNet with Gaussian heatmap head for cliff-water POI prediction.

    Head architecture (replaces the base class 1×1 default):
        Pointwise Conv2d(64, 1, 1×1)    — from base class self.head (logits)
        Soft activation: sigmoid(x) * (1 + 0.1·x) — smooth gradients near zero,
            avoids mode collapse to hard 0/1 for sparse heatmap targets
        Fixed Gaussian blur (7×7, σ=1.5) — enforces spatial coherence to match
            the Gaussian-convolved ground-truth heatmap generation process
        clamp(0, 1)                      — ensure output stays in [0, 1]

    The Gaussian kernel is registered as a non-trainable buffer so it moves
    automatically to the correct device with .to(device) and is saved/loaded
    with model checkpoints.

    Input:  (B, 6, 64, 64)  — RGB (3) + DEM elevation (1) + Slope (1) + Aspect (1)
    Output: (B, 1, 64, 64)  — per-pixel POI probability heatmap [0, 1]
    """

    _KERNEL_SIZE = 7
    _SIGMA       = 1.5

    def __init__(self, in_channels: int = 6, out_channels: int = 1,
                 freeze_backbone: bool = False):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         freeze_backbone=freeze_backbone)

        # Register fixed Gaussian smoothing kernel as a non-trainable buffer.
        # It saves/loads with the checkpoint and moves to GPU automatically.
        self.register_buffer(
            "_gauss_kernel",
            _gaussian_kernel(self._KERNEL_SIZE, self._SIGMA),
        )
        # self.head (Conv2d 64→out_channels) inherited unchanged from base class.

    def _apply_head(self, x_c: torch.Tensor) -> torch.Tensor:
        # 1. Project 64 channels → 1 channel (logits)
        logits = self.head(x_c)                          # (B, 1, 64, 64)

        # 2. Soft activation — gentler than plain sigmoid for sparse heatmaps.
        #    sigmoid(x) * (1 + 0.1·x) interpolates smoothly between near-zero
        #    background and near-one peaks while keeping values in ~[0, 1].
        sig = torch.sigmoid(logits)
        pred = sig * (1.0 + 0.1 * logits)               # (B, 1, 64, 64)

        # 3. Gaussian smoothing — enforces spatial coherence of heatmap peaks.
        padding = self._KERNEL_SIZE // 2
        pred = F.conv2d(pred, self._gauss_kernel, padding=padding)

        return pred.clamp(0.0, 1.0)                      # (B, 1, 64, 64)


if __name__ == "__main__":
    model = ElevationPOITransUNet(in_channels=6, out_channels=1)
    print(f"ElevationPOITransUNet (EVA-02 ViT-UNet + Gaussian heatmap head)")
    print(f"  Trainable params: {count_parameters(model):,}")
    x = torch.randn(2, 6, 64, 64)
    pred, emb = model(x, return_embedding=True)
    enc = model.encode(x)
    print(f"  Input:     {tuple(x.shape)}  (RGB + DEM + Slope + Aspect)")
    print(f"  Heatmap:   {tuple(pred.shape)}  range [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  Embedding: {tuple(emb.shape)}")
    print(f"  encode():  {tuple(enc.shape)}")
