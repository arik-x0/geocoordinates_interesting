"""
Multi-scale Edge Detection CNN for low-density residential housing detection.

HED-inspired architecture (Holistically-nested Edge Detection) with four encoder
stages and a side output at each scale, fused by a learnable 1x1 convolution.

The model is designed to detect structural edges (building outlines, pavements,
rooftops) in 64x64 Sentinel-2 satellite tiles. Dilated convolutions in the
deeper stages provide a large receptive field without losing spatial resolution.

Input:  (B, 3, 64, 64)  — RGB satellite image
Output: (B, 1, 64, 64)  — per-pixel structure probability [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU, with optional dilation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        padding = dilation  # keeps spatial size when kernel_size=3
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderStage(nn.Module):
    """Two ConvBNReLU blocks with an optional MaxPool downsampling step before them."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True, dilation: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.conv1 = ConvBNReLU(in_ch, out_ch, dilation=1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SideOutput(nn.Module):
    """1x1 conv projection + bilinear upsample to the original image size."""

    def __init__(self, in_ch: int, target_size: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        return x


class HousingEdgeCNN(nn.Module):
    """Multi-scale Edge Detection CNN for residential housing detection.

    Architecture (for 64x64 input):
        Stage 1: ConvBNReLU(3->32)  x2, d=1   — 64x64  fine local edges
        Stage 2: Pool + Conv(32->64)  x2, d=1  — 32x32  building-scale structure
        Stage 3: Pool + Conv(64->128) x2, d=1  — 16x16  block-scale patterns
        Stage 4: Conv(128->256)   x2, d=2      — 16x16  wider context (no pool)

        Side output at each stage  → bilinear upsample to 64x64
        Fusion: Cat(4 side outputs) → Conv2d(4->1) → Sigmoid

    Side outputs allow each scale to contribute to the final prediction,
    providing multi-resolution edge supervision. The dilated stage 4 expands
    the receptive field to capture building-block context without downsampling.

    Input:  (B, 3, 64, 64)  — RGB satellite image
    Output: (B, 1, 64, 64)  — per-pixel structure probability [0, 1]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder stages
        self.stage1 = EncoderStage(in_channels, 32, pool=False, dilation=1)
        self.stage2 = EncoderStage(32, 64,  pool=True,  dilation=1)
        self.stage3 = EncoderStage(64, 128, pool=True,  dilation=1)
        self.stage4 = EncoderStage(128, 256, pool=False, dilation=2)

        # Side output projections (one per stage)
        self.side1 = SideOutput(32)
        self.side2 = SideOutput(64)
        self.side3 = SideOutput(128)
        self.side4 = SideOutput(256)

        # Learnable fusion of all four side outputs
        self.fusion = nn.Conv2d(4, out_channels, kernel_size=1)

        # Linear projection for VectorDB embeddings: 256 → 512 (matches other models)
        self.embed_proj = nn.Linear(256, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale encoder
        f1 = self.stage1(x)    # (B,  32, 64, 64)
        f2 = self.stage2(f1)   # (B,  64, 32, 32)
        f3 = self.stage3(f2)   # (B, 128, 16, 16)
        f4 = self.stage4(f3)   # (B, 256, 16, 16)

        # Side outputs — each upsampled to full resolution
        s1 = self.side1(f1)    # (B, 1, 64, 64)
        s2 = self.side2(f2)    # (B, 1, 64, 64)
        s3 = self.side3(f3)    # (B, 1, 64, 64)
        s4 = self.side4(f4)    # (B, 1, 64, 64)

        # Fuse all side outputs
        fused = torch.cat([s1, s2, s3, s4], dim=1)   # (B, 4, 64, 64)
        out = self.fusion(fused)                       # (B, 1, 64, 64)
        return torch.sigmoid(out)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a 512-dim L2-normalised embedding from the stage-4 bottleneck.

        Mean-pools the (B, 256, 16, 16) stage-4 feature map to (B, 256), then
        projects to 512-dim for consistency with the TransUNet-based models.
        The resulting vectors are L2-normalised and directly usable with a
        FAISS IndexFlatIP index (inner product = cosine similarity).

        Call within torch.no_grad() during indexing and retrieval.
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)              # (B, 256, 16, 16)
        emb = f4.mean(dim=[2, 3])         # (B, 256) spatial mean-pool
        emb = self.embed_proj(emb)        # (B, 512) projected
        return F.normalize(emb, p=2, dim=1)  # L2-normalise


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HousingEdgeCNN(in_channels=3, out_channels=1)
    print(f"HousingEdgeCNN Parameters: {count_parameters(model):,}")

    dummy = torch.randn(2, 3, 64, 64)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
