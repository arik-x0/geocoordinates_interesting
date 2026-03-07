"""
AestheticAggregator: fuses N submodel heatmaps into a single aesthetic score map.

The aggregator sits downstream of all individual submodels. It receives a
stack of heatmaps (one per aesthetic dimension) and learns a spatial attention
weighting to produce the final "geo-interesting" aesthetic heatmap.

Input:  (B, N, 64, 64) — N stacked heatmaps (default N=9: 6 aesthetic + 3 structural)
Output: (B, 1, 64, 64) — unified aesthetic heatmap

Architecture:
    - Channel attention: a lightweight MLP over global avg-pooled channels
      learns which aesthetic dimensions matter most for this image.
    - Spatial fusion: a small CNN merges the spatially-weighted heatmaps.
    (~0.05M params — intentionally tiny since inputs are already high-level)

Channel ordering (SUBMODEL_NAMES):
    0  fractal        — self-similar fractal patterns (fractal dimension)
    1  water          — water body geometry
    2  color_harmony  — HSV saturation + spectral spread
    3  symmetry       — gradient circular variance (low = ordered)
    4  sublime        — coarse/fine contrast (scale awe)
    5  complexity     — local gradient std Gaussian bell
    6  vegetation     — NDVI greenery (biophilia signal, direct positive)
    7  elevation      — terrain ruggedness/relief (awe signal, direct positive)
    8  urban_openness — 1 − housing_heatmap (absence of built clutter, positive)

Notes on structural channels (6-8):
    Vegetation (ch 6): Research on biophilia (Wilson, 1984) and Attention
    Restoration Theory (Kaplan, 1995) establishes green vegetation as a
    primary driver of aesthetic preference and restorative experience.
    High NDVI → high aesthetic contribution.

    Elevation (ch 7): Topographic heterogeneity (relief, ruggedness) drives
    scenic quality independent of other visual signals per SBE research.
    The elevation model captures what the visual-only sublime model cannot —
    physical terrain variation from DEM data.

    Urban openness (ch 8): The housing submodel predicts building *presence*;
    this channel feeds (1 − housing_heatmap) to the aggregator so the signal
    represents absence of built clutter. Per Attention Restoration Theory,
    high building density increases cognitive load and reduces restorative/
    aesthetic experience. The aggregator learns the weight of this signal
    per scene type: a dense urban scene may still score high via color/symmetry,
    while a natural landscape is boosted by high urban openness.
    The raw housing_score stored in VectorDB metadata still reflects building
    presence (useful for standalone use); only the meta aggregation inverts it.
"""

import torch
import torch.nn as nn


# Default submodel ordering (indices into the stacked heatmap input)
SUBMODEL_NAMES = [
    "fractal",         # 0 — aesthetic
    "water",           # 1 — aesthetic
    "color_harmony",   # 2 — aesthetic
    "symmetry",        # 3 — aesthetic
    "sublime",         # 4 — aesthetic
    "complexity",      # 5 — aesthetic
    "vegetation",      # 6 — structural (positive, biophilia)
    "elevation",       # 7 — structural (positive, terrain beauty)
    "urban_openness",  # 8 — structural (= 1 − housing_heatmap, anti-clutter)
]


class AestheticAggregator(nn.Module):
    """Learns a weighted fusion of N aesthetic heatmaps. (~0.05M params)

    Architecture:
        x (B, N, H, W)
          -> channel SE attention: global avg-pool -> MLP(N->N//2->N) -> Sigmoid
          -> weighted x: x * weights
          -> spatial fusion: Conv(N->16) -> ReLU -> Conv(16->1) -> Sigmoid
    """

    def __init__(self, n_submodels: int = 9):
        super().__init__()
        self.n = n_submodels

        # Channel-wise importance (which aesthetic dimension dominates)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_submodels, max(2, n_submodels // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(2, n_submodels // 2), n_submodels),
            nn.Sigmoid(),
        )

        # Spatial fusion over weighted heatmaps
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(n_submodels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmaps: (B, N, H, W) — stacked submodel heatmaps, values in [0,1]
        Returns:
            (B, 1, H, W) aesthetic heatmap in [0, 1]
        """
        w = self.channel_attn(heatmaps)              # (B, N)
        w = w.unsqueeze(-1).unsqueeze(-1)            # (B, N, 1, 1)
        weighted = heatmaps * w                      # (B, N, H, W)
        return torch.sigmoid(self.spatial_fuse(weighted))  # (B, 1, H, W)

    def channel_weights(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Return the learned channel importances for inspection.

        Returns:
            (B, N) tensor of per-submodel weights.
        """
        return self.channel_attn(heatmaps)
