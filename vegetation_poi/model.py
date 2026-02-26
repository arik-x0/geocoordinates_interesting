"""
Vegetation POI model — EVA-02 ViT-UNet with SE channel-attention output head.

Task: segment per-pixel vegetation/greenery from RGB satellite tiles.
Ground truth: NDVI > threshold binary mask; output scored as greenery_score
(fraction of high-NDVI pixels) to identify parks, forests, irrigated farmland.

Why an SE channel-attention head?
NDVI is fundamentally a spectral ratio: (NIR - Red) / (NIR + Red).  Without
a NIR band, the model must approximate it from RGB colour relationships.
Deep ViT decoder features encode spectral patterns, but the 64 channels carry
a mix of spatial, textural, and spectral cues.  The Squeeze-and-Excitation
(SE) block explicitly re-weights these channels by their global relevance for
the vegetation prediction before the final 1×1 projection — letting the model
amplify channels that track green-leaf spectral signatures and suppress channels
that respond to roads or bare soil, without adding spatial parameters.

Output metadata (populated by predict.py):
    greenery_score  — float [0, 1]: fraction of pixels above greenery threshold
    ndvi_mean       — float: mean NDVI for the tile (pseudo-label strength)
    is_vegetated    — bool: EuroSAT class label (Forest / HerbaceousVeg / etc.)
    class_name      — str: EuroSAT land-cover class
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pretrained_model import ViTUNetPOI, count_parameters  # noqa: F401


class TransUNet(ViTUNetPOI):
    """EVA-02 ViT-UNet with SE channel-attention head for greenery segmentation.

    Head architecture (wraps the base class 1×1 head):
        SE block — Squeeze-and-Excitation channel attention:
            AdaptiveAvgPool2d(1)           — global average pool → (B, 64, 1, 1)
            Linear(64 → 16) + ReLU         — squeeze: compress channel descriptor
            Linear(16 → 64) + Sigmoid      — excite: per-channel weight in [0, 1]
            x_c * weights                  — re-weight 64 decoder channels
        Pointwise Conv2d(64, 1, 1×1)      — from base class self.head
        Sigmoid

    The SE block learns which decoder channels correlate with greenery across
    the training set and amplifies them globally at inference time.  The 64→16→64
    bottleneck (reduction ratio = 4) keeps the attention module lightweight
    (~2 048 parameters) while still capturing inter-channel dependencies.

    Input:  (B, 3, 64, 64)  — RGB satellite tile
    Output: (B, 1, 64, 64)  — per-pixel greenery probability [0, 1]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 freeze_backbone: bool = False):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         freeze_backbone=freeze_backbone)

        # SE channel-attention block (reduction ratio = 4: 64 → 16 → 64)
        self.se_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, 64, 1, 1)
            nn.Flatten(),                     # (B, 64)
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.Sigmoid(),
        )
        # self.head (Conv2d 64→out_channels) inherited unchanged from base class.

    def _apply_head(self, x_c: torch.Tensor) -> torch.Tensor:
        # Compute per-channel attention weights from global average pooling
        weights = self.se_attention(x_c)              # (B, 64)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)

        # Re-weight feature channels: amplify spectral vegetation cues
        x_c = x_c * weights                           # (B, 64, 64, 64)

        return torch.sigmoid(self.head(x_c))           # (B, 1, 64, 64)


if __name__ == "__main__":
    model = TransUNet(in_channels=3, out_channels=1)
    print(f"TransUNet (EVA-02 ViT-UNet + SE attention head)")
    print(f"  Trainable params: {count_parameters(model):,}")
    x = torch.randn(2, 3, 64, 64)
    pred, emb = model(x, return_embedding=True)
    print(f"  Input:     {tuple(x.shape)}")
    print(f"  Heatmap:   {tuple(pred.shape)}  range [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  Embedding: {tuple(emb.shape)}")
