"""
U-Net Convolutional Neural Network for satellite greenery segmentation.
Encoder-decoder architecture with skip connections, designed for
64x64 Sentinel-2 satellite image inputs.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv2d -> BatchNorm -> ReLU, repeated twice."""

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
    """Encoder block: ConvBlock followed by MaxPool2d for downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)       # Skip connection output
        pooled = self.pool(features)   # Downsampled for next level
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block: Upsample via ConvTranspose2d, concatenate skip connection, ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # *2 for skip concatenation

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch (if input dimensions aren't perfectly divisible)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                          align_corners=False)

        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for binary segmentation of satellite imagery.

    Architecture:
        Encoder: 4 downsampling blocks (3 -> 64 -> 128 -> 256 -> 512)
        Bottleneck: 512 -> 1024
        Decoder: 4 upsampling blocks (1024 -> 512 -> 256 -> 128 -> 64)
        Output: 1 channel sigmoid (greenery probability per pixel)

    Input:  (B, 3, 64, 64)  — RGB satellite image
    Output: (B, 1, 64, 64)  — per-pixel greenery probability
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder path (downsampling)
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder path (upsampling with skip connections)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Final 1x1 convolution to produce single-channel output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)   # skip1: (B, 64, 64, 64),  x: (B, 64, 32, 32)
        skip2, x = self.enc2(x)   # skip2: (B, 128, 32, 32), x: (B, 128, 16, 16)
        skip3, x = self.enc3(x)   # skip3: (B, 256, 16, 16), x: (B, 256, 8, 8)
        skip4, x = self.enc4(x)   # skip4: (B, 512, 8, 8),   x: (B, 512, 4, 4)

        # Bottleneck
        x = self.bottleneck(x)     # (B, 1024, 4, 4)

        # Decoder
        x = self.dec4(x, skip4)    # (B, 512, 8, 8)
        x = self.dec3(x, skip3)    # (B, 256, 16, 16)
        x = self.dec2(x, skip2)    # (B, 128, 32, 32)
        x = self.dec1(x, skip1)    # (B, 64, 64, 64)

        # Final output with sigmoid activation
        x = self.final_conv(x)     # (B, 1, 64, 64)
        x = torch.sigmoid(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    print(f"U-Net Parameters: {count_parameters(model):,}")

    dummy_input = torch.randn(2, 3, 64, 64)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
