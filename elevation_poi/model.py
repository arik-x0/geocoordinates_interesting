"""
U-Net CNN for cliff-water POI detection from satellite + elevation data.
6-channel input architecture: RGB (3) + DEM elevation (1) + slope (1) + aspect (1).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution: Conv2d -> BatchNorm -> ReLU, repeated twice."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder: ConvBlock + MaxPool for downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder: Upsample + skip concatenation + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ElevationPOIUNet(nn.Module):
    """U-Net for cliff-water POI heatmap prediction from multi-channel satellite + DEM input.

    Architecture:
        Input:      (B, 6, 64, 64)  — RGB + DEM + Slope + Aspect
        Encoder:    6 -> 64 -> 128 -> 256 -> 512  (4 downsampling blocks)
        Bottleneck: 512 -> 1024
        Decoder:    1024 -> 512 -> 256 -> 128 -> 64  (4 upsampling blocks + skip connections)
        Output:     (B, 1, 64, 64)  — per-pixel POI probability heatmap

    The 6-channel input allows the convolutional filters to learn joint features
    across spectral (RGB), topographic (elevation, slope), and directional (aspect)
    information — enabling detection of where steep terrain faces water bodies.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 1):
        super().__init__()

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder path
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)   # skip1: (B, 64, 64, 64),  x: (B, 64, 32, 32)
        skip2, x = self.enc2(x)   # skip2: (B, 128, 32, 32), x: (B, 128, 16, 16)
        skip3, x = self.enc3(x)   # skip3: (B, 256, 16, 16), x: (B, 256, 8, 8)
        skip4, x = self.enc4(x)   # skip4: (B, 512, 8, 8),   x: (B, 512, 4, 4)

        # Bottleneck
        x = self.bottleneck(x)     # (B, 1024, 4, 4)

        # Decoder with skip connections
        x = self.dec4(x, skip4)    # (B, 512, 8, 8)
        x = self.dec3(x, skip3)    # (B, 256, 16, 16)
        x = self.dec2(x, skip2)    # (B, 128, 32, 32)
        x = self.dec1(x, skip1)    # (B, 64, 64, 64)

        # Output with sigmoid for [0, 1] heatmap
        x = self.final_conv(x)     # (B, 1, 64, 64)
        x = torch.sigmoid(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ElevationPOIUNet(in_channels=6, out_channels=1)
    print(f"ElevationPOI U-Net Parameters: {count_parameters(model):,}")

    # Test with dummy 6-channel input
    dummy = torch.randn(2, 6, 64, 64)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Verify channel breakdown
    print("\nInput channels:")
    print("  0-2: RGB (satellite imagery)")
    print("  3:   DEM elevation (normalized)")
    print("  4:   Slope (normalized)")
    print("  5:   Aspect (normalized)")
