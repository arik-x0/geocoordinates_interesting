"""
Housing POI model — EVA-02 ViT-UNet with edge-sharpening output head.

Task: detect per-pixel built-up structure probability from RGB satellite tiles.
Ground truth: NDBI-derived binary mask; output is scored as fraction of
built-up pixels (housing_score) to identify low-density residential zones (5–20%).

Why an edge-sharpening head?
Buildings, roads, and impervious surfaces have hard, rectangular boundaries.
The default 1×1 projection head blurs these boundaries because it has no
spatial context.  The edge-sharpening head adds a 3×3 depthwise conv before
the final 1×1 projection, giving each output pixel access to its 8 neighbours.
This lets the head learn Laplacian-like spatial filters that amplify high-
frequency boundary signals already present in the EVA-02 decoder features,
producing crisper building outlines with no additional backbone compute.

Output metadata (populated by predict.py):
    housing_score   — float [0, 1]: fraction of predicted built-up pixels
    density_label   — str: "low-density", "undeveloped", or "dense/industrial"
    is_residential  — bool: EuroSAT class label
    ndbi_mean       — float: mean NDBI (pseudo-label strength indicator)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pretrained_model import ViTUNetPOI, count_parameters  # noqa: F401


class HousingEdgeCNN(ViTUNetPOI):
    """EVA-02 ViT-UNet with edge-sharpening head for structure density detection.

    Head architecture (replaces the base class 1×1 default):
        Depthwise Conv2d(64, 64, 3×3) — spatial neighbourhood mixing per channel
            └─ learns Laplacian-like edge filters within the 64 feature channels
        BatchNorm + ReLU
        Pointwise Conv2d(64, 1, 1×1)  — final per-pixel classification
        Sigmoid

    The depthwise conv preserves the 64-channel depth while adding spatial
    context.  Because each channel is processed independently, it can learn
    a different edge-detection kernel per feature channel — some may fire on
    horizontal rooftop edges, others on vertical wall lines, etc.

    Input:  (B, 3, 64, 64)  — RGB satellite tile
    Output: (B, 1, 64, 64)  — per-pixel built-up probability heatmap [0, 1]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 freeze_backbone: bool = False):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         freeze_backbone=freeze_backbone)

        # Replace the default 1×1 head with an edge-sharpening head.
        # Depthwise (groups=64) so each feature channel gets its own 3×3 kernel.
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def _apply_head(self, x_c: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x_c))


if __name__ == "__main__":
    model = HousingEdgeCNN(in_channels=3, out_channels=1)
    print(f"HousingEdgeCNN (EVA-02 ViT-UNet + edge head)")
    print(f"  Trainable params: {count_parameters(model):,}")
    x = torch.randn(2, 3, 64, 64)
    pred, emb = model(x, return_embedding=True)
    print(f"  Input:     {tuple(x.shape)}")
    print(f"  Heatmap:   {tuple(pred.shape)}  range [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  Embedding: {tuple(emb.shape)}")
