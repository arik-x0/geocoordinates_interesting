"""
AestheticAggregator: fuses N submodel heatmaps into a single aesthetic score map.

The aggregator sits downstream of all individual submodels. It receives a
stack of heatmaps (one per aesthetic dimension) and learns a spatial attention
weighting to produce the final "geo-interesting" aesthetic heatmap.

Input:  (B, N, 64, 64) — N stacked heatmaps (default N=6 aesthetic submodels)
Output: (B, 1, 64, 64) — unified aesthetic heatmap

Architecture:
    - Channel attention: a lightweight MLP over global avg-pooled channels
      learns which aesthetic dimensions matter most for this image.
    - Spatial fusion: a small CNN merges the spatially-weighted heatmaps.
    (~0.04M params — intentionally tiny since inputs are already high-level)
"""

import torch
import torch.nn as nn


# Default submodel ordering (indices into the stacked input)
SUBMODEL_NAMES = [
    "fractal",       # 0
    "water",         # 1
    "color_harmony", # 2
    "symmetry",      # 3
    "sublime",       # 4
    "complexity",    # 5
]


class AestheticAggregator(nn.Module):
    """Learns a weighted fusion of N aesthetic heatmaps. (~0.04M params)

    Architecture:
        x (B, N, H, W)
          -> channel SE attention: global avg-pool -> MLP(N->N//2->N) -> Sigmoid
          -> weighted x: x * weights
          -> spatial fusion: Conv(N->16) -> ReLU -> Conv(16->1) -> Sigmoid
    """

    def __init__(self, n_submodels: int = 6):
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
