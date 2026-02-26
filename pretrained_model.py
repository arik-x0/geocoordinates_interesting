"""
EVA-02 ViT-S/14 backbone + U-Net decoder for dense POI heatmap prediction.

Shared foundation model used by all three POI detection pipelines:
  vegetation_poi  — greenery coverage         (3-channel RGB input)
  housing_poi     — built-up structure density (3-channel RGB input)
  elevation_poi   — cliff-near-water heatmap   (6-channel RGB+DEM input)

Architecture — ViT-UNet with multi-scale skip connections
----------------------------------------------------------
Channel adapter  : Conv2d(in_channels → 3) if in_channels ≠ 3, else Identity

EVA-02 ViT-S/14 Encoder (pretrained, MIT license):
  patch_size = 14, embed_dim = 384, depth = 12, num_heads = 6
  img_size = 224×224  →  16×16 = 256 patch tokens per layer + 1 CLS token

  Forward hooks tap four transformer blocks for U-Net skip connections:
    block  2 → feat_3  (B, 384, 16, 16) — early: edges, colour, texture
    block  5 → feat_6  (B, 384, 16, 16) — mid:   object parts, spectral patterns
    block  8 → feat_9  (B, 384, 16, 16) — late:  semantic structure, land-cover
    block 11 → feat_12 (B, 384, 16, 16) — deep:  global scene context (bottleneck)

U-Net Decoder (randomly initialised, fine-tuned):
  Block A — bottleneck 16×16:
    cat(feat_12: 384, proj9(feat_9): 256) → ConvBlock(640→256) → (B, 256, 16, 16)
  Block B — upsample 16→32:
    ConvTranspose2d(256→128) + proj6(feat_6) bilinear×2
    cat(128+128) → ConvBlock(256→128) → (B, 128, 32, 32)
  Block C — upsample 32→64:
    ConvTranspose2d(128→64) + proj3(feat_3) bilinear×4
    cat(64+64) → ConvBlock(128→64) → (B, 64, 64, 64)

Task-specific output heads (defined in each sub-module's model.py):
  Subclasses override _apply_head(x_c) to replace the default 1×1 head.
  The base class default: Conv2d(64→1) + sigmoid.

VectorDB embed: CLS token (post-norm, block 11) → Linear(384→512) → L2-norm

Input  : (B, in_channels, 64, 64)
Output : (B, 1, 64, 64)  — per-pixel POI probability heatmap [0, 1]

License
-------
EVA-02 weights: MIT license  https://github.com/baaivision/EVA/tree/master/EVA-02
timm library:   Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the pretrained EVA-02 backbone.\n"
        "Install it with:  pip install timm>=0.9.2"
    ) from e

_BACKBONE_ID  = "eva02_small_patch14_224"
_EMBED_DIM    = 384   # ViT-S hidden dimension
_GRID_SIZE    = 16    # 224 // patch_size(14) = 16 patches per spatial dimension
_SKIP_INDICES = (2, 5, 8, 11)  # 0-indexed transformer blocks tapped for skips


# ── Decoder building block ────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv2d(3×3) → BN → ReLU, twice — standard U-Net decoder block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main model ────────────────────────────────────────────────────────────────

class ViTUNetPOI(nn.Module):
    """EVA-02 ViT-S/14 backbone + U-Net decoder.

    Base class shared by all three POI models.  The backbone, skip projectors,
    and decoder blocks are defined here.  The final output step is intentionally
    left as an overridable hook (_apply_head) so each sub-module can install
    a task-appropriate head without duplicating any backbone or decoder code.

    Subclass interface:
        __init__: call super().__init__(...), then replace self.head and/or
                  register additional modules (e.g. attention, smoothing kernels).
        _apply_head(x_c): receives (B, 64, H, W) decoder output; return the
                  final (B, 1, H, W) prediction in [0, 1].  Default: sigmoid(head(x_c)).

    Pretrained backbone: EVA-02 ViT-S — MIT license, ImageNet-22K.

    Args:
        in_channels     : Input channels — 3 for RGB, 6 for RGB+DEM+Slope+Aspect.
        out_channels    : Output channels (1 for binary heatmap).
        freeze_backbone : Freeze all backbone weights; channel adapter, skip
                          projectors, decoder, and head always remain trainable.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 freeze_backbone: bool = False):
        super().__init__()

        # ── Channel adapter ───────────────────────────────────────────────────
        self.channel_adapter: nn.Module = (
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            if in_channels != 3 else nn.Identity()
        )

        # ── Pretrained EVA-02 ViT-S/14 backbone ──────────────────────────────
        self.backbone = timm.create_model(
            _BACKBONE_ID,
            pretrained=True,
            num_classes=0,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Intermediate feature cache + forward hooks ────────────────────────
        self._feat_cache: dict = {}
        for idx in _SKIP_INDICES:
            self.backbone.blocks[idx].register_forward_hook(
                lambda _m, _inp, out, i=idx: self._feat_cache.__setitem__(i, out)
            )

        D = _EMBED_DIM  # 384

        # ── Skip connection projectors ─────────────────────────────────────────
        self.proj9 = nn.Conv2d(D, 256, kernel_size=1, bias=False)   # block 8
        self.proj6 = nn.Conv2d(D, 128, kernel_size=1, bias=False)   # block 5
        self.proj3 = nn.Conv2d(D, 64,  kernel_size=1, bias=False)   # block 2

        # ── U-Net decoder ──────────────────────────────────────────────────────
        self.dec_a = _ConvBlock(D + 256, 256)   # 16×16 bottleneck
        self.up1   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_b = _ConvBlock(128 + 128, 128)  # 32×32
        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_c = _ConvBlock(64 + 64, 64)     # 64×64

        # ── Default output head (subclasses replace or wrap this) ─────────────
        # Produces (B, out_channels, 64, 64) logits; _apply_head adds the
        # activation.  Subclasses that need a richer head should reassign
        # self.head in their own __init__ and override _apply_head().
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

        # ── VectorDB embedding projector ──────────────────────────────────────
        self.embed_proj = nn.Linear(D, 512)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _encode_backbone(self, x: torch.Tensor):
        """Channel-adapt → resize → ViT forward → multi-scale features + CLS.

        Returns:
            feat_3   : (B, 384, 16, 16) — block-2  (early, pre-norm)
            feat_6   : (B, 384, 16, 16) — block-5  (mid,   pre-norm)
            feat_9   : (B, 384, 16, 16) — block-8  (late,  pre-norm)
            feat_12  : (B, 384, 16, 16) — block-11 (bottleneck, post-norm)
            cls_token: (B, 384)         — CLS token, post-norm
        """
        x = self.channel_adapter(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        self._feat_cache.clear()
        final = self.backbone.forward_features(x)   # (B, 257, 384) post-norm

        cls_token = final[:, 0]                     # (B, 384)
        B = x.shape[0]

        def to_spatial(tokens: torch.Tensor) -> torch.Tensor:
            return (tokens[:, 1:]
                    .transpose(1, 2)
                    .reshape(B, _EMBED_DIM, _GRID_SIZE, _GRID_SIZE))

        feat_3  = to_spatial(self._feat_cache[2])
        feat_6  = to_spatial(self._feat_cache[5])
        feat_9  = to_spatial(self._feat_cache[8])
        feat_12 = to_spatial(final)

        return feat_3, feat_6, feat_9, feat_12, cls_token

    def _decode(self, x: torch.Tensor):
        """Full backbone + U-Net decoder.

        Returns:
            x_c      : (B, 64, 64, 64) — 64-channel feature map before the head.
                        Passed to _apply_head() in forward().
            cls_token: (B, 384) — used for the VectorDB embedding.
        """
        feat_3, feat_6, feat_9, feat_12, cls_token = self._encode_backbone(x)

        s9  = self.proj9(feat_9)                                          # (B, 256, 16, 16)
        x_a = self.dec_a(torch.cat([feat_12, s9], dim=1))                # (B, 256, 16, 16)

        s6  = F.interpolate(self.proj6(feat_6), scale_factor=2.0,
                            mode="bilinear", align_corners=False)         # (B, 128, 32, 32)
        x_b = self.dec_b(torch.cat([self.up1(x_a), s6], dim=1))          # (B, 128, 32, 32)

        s3  = F.interpolate(self.proj3(feat_3), scale_factor=4.0,
                            mode="bilinear", align_corners=False)         # (B, 64, 64, 64)
        x_c = self.dec_c(torch.cat([self.up2(x_b), s3], dim=1))          # (B, 64, 64, 64)

        return x_c, cls_token

    # ── Overridable head hook ─────────────────────────────────────────────────

    def _apply_head(self, x_c: torch.Tensor) -> torch.Tensor:
        """Apply the output head to the 64-channel decoder feature map.

        Default: self.head (Conv2d 64→1) followed by sigmoid.
        Subclasses override this to install task-specific behaviour
        (edge sharpening, channel attention, Gaussian smoothing, etc.)
        without touching any backbone or decoder logic.

        Args:
            x_c: (B, 64, H, W) decoder output.

        Returns:
            (B, out_channels, H, W) prediction in [0, 1].
        """
        return torch.sigmoid(self.head(x_c))

    # ── Public interface ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                return_embedding: bool = False):
        """Dense POI heatmap via ViT-UNet.

        Args:
            x               : (B, in_channels, 64, 64) satellite tile.
            return_embedding: Also return the (B, 512) L2-normalised CLS
                              embedding for SimCLR contrastive training.

        Returns:
            prediction              : (B, 1, 64, 64)  when return_embedding=False
            (prediction, embedding) : tuple            when return_embedding=True
        """
        x_c, cls_token = self._decode(x)
        pred = self._apply_head(x_c)

        if return_embedding:
            emb = F.normalize(self.embed_proj(cls_token), p=2, dim=1)
            return pred, emb
        return pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """512-dim L2-normalised VectorDB embedding from the EVA-02 CLS token.

        Runs only the backbone (skips the decoder) for efficiency during
        VectorDB indexing and retrieval.

        Returns:
            (B, 512) float32 embedding tensor.
        """
        _, _, _, _, cls_token = self._encode_backbone(x)
        return F.normalize(self.embed_proj(cls_token), p=2, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
