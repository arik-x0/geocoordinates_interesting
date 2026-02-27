"""
Models for cliff-water POI detection from satellite + elevation data.
6-channel input: RGB (3) + DEM elevation (1) + slope (1) + aspect (1).

Two architectures are provided:

  ElevationPOIViT    — pure Vision Transformer (original).
      Patch-embeds the 6-channel 64x64 image into 64 tokens, processes them
      with 6 Transformer blocks, then progressively upsamples back to full
      resolution. No spatial inductive bias.

  ElevationPOITransUNet — hybrid Transformer + U-Net (active model).
      CNN encoder builds hierarchical spatial features at 4 scales from the
      full 6-channel input. A Transformer bottleneck applies global self-attention
      over the 8x8 feature map (64 tokens, dim=512). A U-Net decoder restores
      full resolution with skip connections from each encoder stage.
      The 6 input channels are processed jointly from the first conv layer,
      so spectral + topographic cues are fused at every spatial scale.

"""

import torch
import torch.nn as nn


# ── Shared Transformer primitives ────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly embed each.

    Uses a Conv2d with kernel_size=stride=patch_size, equivalent to extracting
    patches and applying a shared linear projection over all input channels.
    """

    def __init__(self, img_size: int = 64, patch_size: int = 8,
                 in_channels: int = 6, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size        # 8
        self.num_patches = self.grid_size ** 2         # 64
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, grid, grid) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """Feed-forward sub-layer inside each Transformer block (GELU activation)."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm.

    LayerNorm -> MHSA -> residual -> LayerNorm -> MLP -> residual
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SegmentationDecoder(nn.Module):
    """Progressively upsample patch token grid to full image resolution.

    For img_size=64, patch_size=8: grid 8x8 -> 16x16 -> 32x32 -> 64x64
    """

    def __init__(self, embed_dim: int = 256, out_channels: int = 1):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, grid_size: int) -> torch.Tensor:
        B, N, D = x.shape
        # Reshape tokens to spatial grid: (B, D, grid, grid)
        x = x.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        x = self.up1(x)   # (B, 128, grid*2, grid*2)
        x = self.up2(x)   # (B,  64, grid*4, grid*4)
        x = self.up3(x)   # (B,  32, grid*8, grid*8) = (B, 32, 64, 64)
        x = self.head(x)  # (B,   1, 64, 64)
        return x


# ── Original ViT model ───────────────────────────────────────────────────────

class ElevationPOIViT(nn.Module):
    """Vision Transformer (ViT) for cliff-water POI heatmap from multi-channel satellite + DEM.

    Architecture:
        Patch Embedding:      (B, 6, 64, 64) -> (B, 64 patches, embed_dim)
        Positional Embedding: learnable, added to all tokens including [CLS]
        Transformer Encoder:  6 blocks of Multi-Head Self-Attention + MLP
        Heatmap Decoder:      reshape patch tokens -> 3x 2x conv upsampling
        Output:               (B, 1, 64, 64) — per-pixel POI probability heatmap

    The 6-channel patch embedding fuses spectral (RGB), topographic (elevation,
    slope), and directional (aspect) features into each patch token. Self-attention
    then models global spatial relationships — e.g., whether a steep slope patch
    (high-slope token) faces a water body patch (high-NDWI token) — across all 64
    spatial locations simultaneously.

    Input:  (B, 6, 64, 64)  — RGB + DEM + Slope + Aspect
    Output: (B, 1, 64, 64)  — per-pixel POI probability heatmap [0, 1]
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 1,
                 img_size: int = 64, patch_size: int = 8,
                 embed_dim: int = 256, depth: int = 6,
                 num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Heatmap decoder
        self.decoder = SegmentationDecoder(embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend [CLS] token and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, num_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Discard [CLS] token — only patch tokens carry spatial information
        patch_tokens = x[:, 1:, :]               # (B, num_patches, embed_dim)

        # Decode back to full image resolution heatmap
        x = self.decoder(patch_tokens, self.grid_size)   # (B, out_channels, 64, 64)
        x = torch.sigmoid(x)
        return x


# ── CNN building blocks for the hybrid encoder / decoder ─────────────────────

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


class DecoderBlock(nn.Module):
    """Decoder block: ConvTranspose2d upsample + skip concatenation + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)   # *2 for skip cat

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:],
                                          mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Transformer Bottleneck ───────────────────────────────────────────────────

class TransformerBottleneck(nn.Module):
    """Global self-attention over the CNN bottleneck feature map.

    Flattens a (B, C, H, W) feature map to (B, H*W, C) tokens, adds learnable
    positional embeddings, runs N Transformer blocks, then reshapes back to
    (B, C, H, W). Channel dimension C is preserved so no projection is needed.

    For a 64x64 image with 3 max-pooling stages the bottleneck is 8x8 = 64 tokens
    of dimension 512 — compact enough for efficient self-attention.
    """

    def __init__(self, channels: int, bottleneck_size: int = 8,
                 depth: int = 4, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        num_tokens = bottleneck_size ** 2
        self.bottleneck_size = bottleneck_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(channels, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # (B, H*W, C) -> (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ── Hybrid TransUNet model ───────────────────────────────────────────────────

class ElevationPOITransUNet(nn.Module):
    """Hybrid Transformer + U-Net for cliff-water POI heatmap from satellite + DEM.

    Architecture:
        CNN Encoder (6-channel input — RGB + DEM + Slope + Aspect):
            enc1: ConvBlock(6  ->  64)          64x64  — joint spectral+topo features
            enc2: Pool + ConvBlock(64  -> 128)  32x32  — mid-level patterns
            enc3: Pool + ConvBlock(128 -> 256)  16x16  — semantic structure
            enc4: Pool + ConvBlock(256 -> 512)   8x8   — bottleneck

        Transformer Bottleneck:
            Flatten 8x8 -> 64 tokens (dim=512)
            4 Transformer blocks with learnable positional embeddings
            Reshape back to 8x8x512
            Self-attention allows each bottleneck location to query all others —
            e.g., a steep-slope patch can directly attend to water-body patches.

        CNN Decoder (U-Net style with skip connections):
            dec3: Up(512->256) + cat(skip3,256) + ConvBlock -> 256  16x16
            dec2: Up(256->128) + cat(skip2,128) + ConvBlock -> 128  32x32
            dec1: Up(128-> 64) + cat(skip1, 64) + ConvBlock ->  64  64x64
            head: Conv(64->1)                                ->   1  64x64

    Input:  (B, 6, 64, 64)  — RGB + DEM + Slope + Aspect
    Output: (B, 1, 64, 64)  — per-pixel POI probability heatmap [0, 1]
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 1,
                 transformer_depth: int = 4, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        # CNN Encoder (accepts all 6 channels from the first conv)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Transformer Bottleneck over the 8x8 feature map (64 tokens, dim=512)
        self.transformer = TransformerBottleneck(
            channels=512, bottleneck_size=8,
            depth=transformer_depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        # CNN Decoder with U-Net skip connections
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """Forward pass.

        Args:
            x:                Input tensor (B, 6, H, W) — RGB + DEM + Slope + Aspect.
            return_embedding: When True, also return the L2-normalised (B, 512)
                              bottleneck embedding computed at no extra cost.
                              Use this during training to avoid calling encode()
                              separately (halves encoder compute vs. two passes).

        Returns:
            prediction              when return_embedding=False  (default)
            (prediction, embedding) when return_embedding=True
        """
        # CNN Encoder — save skip connections at each scale
        skip1      = self.enc1(x)                    # (B,  64, 64, 64)
        skip2      = self.enc2(self.pool(skip1))      # (B, 128, 32, 32)
        skip3      = self.enc3(self.pool(skip2))      # (B, 256, 16, 16)
        bottleneck = self.enc4(self.pool(skip3))      # (B, 512,  8,  8)

        # Transformer Bottleneck — global context over 8x8=64 spatial tokens
        bottleneck = self.transformer(bottleneck)     # (B, 512,  8,  8)

        # CNN Decoder — recover full resolution via skip connections
        out = self.dec3(bottleneck, skip3)            # (B, 256, 16, 16)
        out = self.dec2(out, skip2)                   # (B, 128, 32, 32)
        out = self.dec1(out, skip1)                   # (B,  64, 64, 64)
        prediction = torch.sigmoid(self.final_conv(out))  # (B, 1, 64, 64)

        if return_embedding:
            emb = bottleneck.mean(dim=[2, 3])                        # (B, 512)
            emb = nn.functional.normalize(emb, p=2, dim=1)          # L2-norm
            return prediction, emb

        return prediction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a fixed-size image embedding from the Transformer bottleneck.

        Runs the CNN encoder and Transformer bottleneck, then mean-pools the
        8x8 spatial feature map to a single (B, 512) vector per image.
        All 6 input channels (RGB + DEM + Slope + Aspect) contribute to the
        embedding, so similarity search captures spectral and topographic likeness.
        Vectors are L2-normalised so inner product equals cosine similarity,
        making them directly usable with a FAISS IndexFlatIP index.

        Call within torch.no_grad() during indexing and retrieval.

        Returns:
            (B, 512) L2-normalised float32 embedding tensor.
        """
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        bottleneck = self.enc4(self.pool(skip3))          # (B, 512, 8, 8)
        bottleneck = self.transformer(bottleneck)          # (B, 512, 8, 8)
        emb = bottleneck.mean(dim=[2, 3])                 # (B, 512) spatial mean-pool
        return nn.functional.normalize(emb, p=2, dim=1)   # L2-normalise


# ── Utilities ────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(2, 6, 64, 64)

    vit = ElevationPOIViT(in_channels=6, out_channels=1)
    print(f"ElevationPOI ViT Parameters:      {count_parameters(vit):,}")

    hybrid = ElevationPOITransUNet(in_channels=6, out_channels=1)
    out = hybrid(dummy)
    print(f"ElevationPOI TransUNet Parameters: {count_parameters(hybrid):,}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    print("\nInput channels:")
    print("  0-2: RGB (satellite imagery)")
    print("  3:   DEM elevation (normalized)")
    print("  4:   Slope (normalized)")
    print("  5:   Aspect (normalized)")
