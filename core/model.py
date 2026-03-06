"""
Core satellite model: frozen DINO ViT-S/16 backbone + trainable shared UNet decoder.

The backbone is pretrained (DINO self-supervised) and always kept frozen.
The UNet decoder is shared across all tasks and trains alongside each submodel,
producing a rich (B, 128, 64, 64) spatial feature map from raw ViT tokens.
Submodels receive this feature map and add only a thin task-specific head.

Usage:
    core = CoreSatelliteModel().freeze().to(device)   # freezes backbone only
    # decoder + submodel params go into the optimizer together

    with torch.no_grad():
        features = core.extract_features(rgb)   # raw ViT token maps (backbone)
    feature_map = core.decode(features)          # 64x64 spatial map (decoder, with grad)
    embedding   = core.encode(rgb)               # 384-dim CLS for FAISS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKBONE_ID  = "vit_small_patch16_224.dino"
_EMBED_DIM    = 384   # ViT-S hidden dimension
_GRID_SIZE    = 14    # 224 / patch_size(16) = 14 spatial tokens per axis
_HOOK_BLOCKS  = (2, 5, 8, 11)
_DEC_CHANNELS = 128   # output channels of the shared decoder (consumed by submodels)


# ── Shared decoder building block ────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Double 3x3 conv with BN+ReLU — decoder primitive."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Shared UNet decoder ───────────────────────────────────────────────────────

class _SharedDecoder(nn.Module):
    """Trainable 3-stage UNet decoder shared across all POI submodels.

    Progressively upsamples multi-scale ViT token maps from 14x14 to 64x64,
    producing a spatially rich feature map that submodel heads consume.

    Input:  features dict from extract_features() — blk11, blk8, blk2
    Output: (B, _DEC_CHANNELS, 64, 64)

    Stage resolution:
        proj11(blk11)                → (B, 256, 14, 14)  deepest semantics
        dec_a                        → (B, 256, 14, 14)
        upsample + cat(proj8(blk8)) → (B, 128, 32, 32)  mid semantics
        dec_b                        → (B, 128, 32, 32)
        upsample + cat(proj2(blk2)) → (B,  64, 64, 64)  low-level texture
        dec_c                        → (B, 128, 64, 64)  <- DEC_CHANNELS
    """

    def __init__(self, embed_dim: int = _EMBED_DIM, out_channels: int = _DEC_CHANNELS):
        super().__init__()
        self.proj11 = nn.Conv2d(embed_dim, 256, 1)
        self.proj8  = nn.Conv2d(embed_dim, 128, 1)
        self.proj2  = nn.Conv2d(embed_dim,  64, 1)

        self.dec_a = _ConvBlock(256,       256)
        self.dec_b = _ConvBlock(256 + 128, 128)
        self.dec_c = _ConvBlock(128 +  64, out_channels)

    @staticmethod
    def _up_cat(d: torch.Tensor, skip: torch.Tensor, size: tuple) -> torch.Tensor:
        d    = F.interpolate(d,    size=size, mode="bilinear", align_corners=False)
        skip = F.interpolate(skip, size=size, mode="bilinear", align_corners=False)
        return torch.cat([d, skip], dim=1)

    def forward(self, features: dict) -> torch.Tensor:
        f11 = self.proj11(features["blk11"])   # (B, 256, 14, 14)
        f8  = self.proj8(features["blk8"])     # (B, 128, 14, 14)
        f2  = self.proj2(features["blk2"])     # (B,  64, 14, 14)

        d = self.dec_a(f11)                              # (B, 256, 14, 14)
        d = self.dec_b(self._up_cat(d, f8, (32, 32)))   # (B, 128, 32, 32)
        d = self.dec_c(self._up_cat(d, f2, (64, 64)))   # (B, _DEC_CHANNELS, 64, 64)
        return d


# ── Core model ────────────────────────────────────────────────────────────────

class CoreSatelliteModel(nn.Module):
    """DINO ViT-S/16 (frozen) + shared UNet decoder (trainable).

    Architecture overview:
        Input  (B, 3, 64, 64) RGB satellite tile
          |
          v  Bilinear resize -> 224x224
          |
          v  DINO ViT-S/16  [FROZEN]
             Forward hooks capture block outputs at layers 2, 5, 8, 11
             -> blkN: (B, 384, 14, 14) spatial token maps
             -> cls:  (B, 384)         global CLS descriptor
          |
          v  _SharedDecoder  [TRAINABLE, shared across all tasks]
             14x14 -> 32x32 -> 64x64 via skip connections
             -> feature_map: (B, 128, 64, 64)
          |
          v  Task submodel head  [TRAINABLE, task-specific]
             -> heatmap: (B, 1, 64, 64)

    Training strategy:
        core.freeze()        — freezes backbone only; decoder stays trainable
        optimizer = Adam(list(core.decoder.parameters()) + list(submodel.parameters()))
        checkpoint saves both decoder_state_dict and submodel_state_dict

    FAISS VectorDB:
        core.encode(rgb)  — returns 384-dim L2-normalised CLS embedding
        Independent of the decoder; purely from the frozen backbone.
    """

    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            _BACKBONE_ID, pretrained=True, num_classes=0,
        )
        self.decoder = _SharedDecoder()

        # Per-block feature cache — populated by forward hooks on the backbone
        self._feat_cache: dict = {}
        for idx in _HOOK_BLOCKS:
            self.backbone.blocks[idx].register_forward_hook(
                lambda _m, _inp, out, i=idx: self._feat_cache.__setitem__(i, out)
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N+1, D) token sequence -> (B, D, 14, 14)."""
        B = tokens.shape[0]
        return (
            tokens[:, 1:]
            .transpose(1, 2)
            .reshape(B, _EMBED_DIM, _GRID_SIZE, _GRID_SIZE)
        )

    def _run_backbone(self, x_224: torch.Tensor):
        """Run backbone; return (final_tokens, cls_token). Clears the hook cache."""
        self._feat_cache.clear()
        final = self.backbone.forward_features(x_224)   # (B, 197, 384)
        return final, final[:, 0]                        # tokens, cls

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> dict:
        """Run the frozen backbone and return raw multi-scale ViT token maps.

        Call this inside torch.no_grad() during training since the backbone
        is frozen. The returned dict is then passed to decode().

        Returns:
            dict with keys blk2, blk5, blk8, blk11: (B, 384, 14, 14)
                          cls: (B, 384) CLS token
        """
        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        final, cls_token = self._run_backbone(x_224)
        sp = self._tokens_to_spatial
        return {
            "blk2":  sp(self._feat_cache[2]),
            "blk5":  sp(self._feat_cache[5]),
            "blk8":  sp(self._feat_cache[8]),
            "blk11": sp(self._feat_cache[11]),
            "cls":   cls_token,
        }

    def decode(self, features: dict) -> torch.Tensor:
        """Run the trainable shared decoder on ViT features.

        Produces a (B, _DEC_CHANNELS, 64, 64) feature map consumed by task heads.
        Call this OUTSIDE torch.no_grad() so the decoder receives gradients.

        Args:
            features: dict returned by extract_features().

        Returns:
            (B, 128, 64, 64) spatial feature map.
        """
        return self.decoder(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a 384-dim L2-normalised CLS embedding for FAISS.

        Uses only the frozen backbone — no decoder pass needed.
        During inference after extract_features(), prefer:
            F.normalize(features['cls'], p=2, dim=1)
        to avoid a redundant backbone pass.
        """
        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        self._feat_cache.clear()
        final = self.backbone.forward_features(x_224)
        return F.normalize(final[:, 0], p=2, dim=1)

    def freeze(self):
        """Freeze the ViT backbone only. Decoder remains trainable.

        Call before building the optimizer so only decoder + submodel params
        are passed to Adam.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        return self

    def freeze_all(self):
        """Freeze both backbone and decoder (pure inference mode)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


if __name__ == "__main__":
    core = CoreSatelliteModel()
    total    = sum(p.numel() for p in core.parameters())
    backbone = sum(p.numel() for p in core.backbone.parameters())
    decoder  = sum(p.numel() for p in core.decoder.parameters())
    print(f"CoreSatelliteModel total: {total:,}")
    print(f"  backbone (frozen):  {backbone:,}")
    print(f"  decoder (trainable): {decoder:,}")

    core.freeze()
    dummy = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = core.extract_features(dummy)
    fmap = core.decode(feats)
    emb  = core.encode(dummy)
    print(f"\n  blk11: {tuple(feats['blk11'].shape)}")
    print(f"  decode: {tuple(fmap.shape)}")
    print(f"  encode: {tuple(emb.shape)}")
