"""
Core satellite model: frozen Prithvi-EO-1.0-100M backbone + trainable shared UNet decoder.

The backbone is a geospatial foundation model pre-trained on 1 TB+ of Sentinel-2/HLS
multispectral imagery (IBM/NASA).  It ingests 6-band HLS-normalised input and is always
kept frozen.  The UNet decoder is shared across all tasks and trains alongside each
submodel, producing a rich (B, 128, 64, 64) spatial feature map from Prithvi tokens.
Submodels receive this feature map and add only a thin task-specific head.

Input to extract_features / encode:
    (B, 6, H, W) — six HLS bands in Prithvi order:
        ch0 = B02 (Blue,       490 nm)
        ch1 = B03 (Green,      560 nm)
        ch2 = B04 (Red,        665 nm)
        ch3 = B8A (NIR-Narrow, 865 nm)
        ch4 = B11 (SWIR-1,    1610 nm)
        ch5 = B12 (SWIR-2,    2190 nm)
    Values must be pre-normalised using PRITHVI_MEAN / PRITHVI_STD from constants.py.

Usage:
    core = CoreSatelliteModel().freeze().to(device)

    with torch.no_grad():
        features = core.extract_features(prithvi_6band)  # raw token maps (backbone)
    feature_map = core.decode(features)                   # 64×64 spatial map (decoder)
    embedding   = core.encode(prithvi_6band)              # 768-dim CLS for FAISS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKBONE_ID  = "prithvi_eo_v1_100"
_EMBED_DIM    = 768   # Prithvi-100M hidden dimension
_IN_CHANS     = 6     # HLS bands fed to Prithvi
_GRID_SIZE    = 14    # 224 / patch_size(16) = 14 spatial tokens per axis
_OUT_INDICES  = (2, 5, 8, 11)   # transformer blocks to tap (0-indexed, depth=12)
_DEC_CHANNELS = 128   # output channels of the shared decoder (consumed by submodels)


# ── Shared decoder building block ────────────────────────────────────────────

class _ConvBlockCore(nn.Module):
    """Double 3×3 conv with BN+ReLU — decoder primitive."""

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

    Progressively upsamples multi-scale Prithvi token maps from 14×14 to 64×64,
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

        self.dec_a = _ConvBlockCore(256,       256)
        self.dec_b = _ConvBlockCore(256 + 128, 128)
        self.dec_c = _ConvBlockCore(128 +  64, out_channels)

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
    """Prithvi-EO-1.0-100M (frozen) + shared UNet decoder (trainable).

    Architecture overview:
        Input  (B, 6, 64, 64) HLS-normalised Prithvi bands
          |
          v  Bilinear resize → 224×224, add temporal dim → (B, 6, 1, 224, 224)
          |
          v  Prithvi-EO-1.0-100M  [FROZEN]
             Returns token sequences at transformer blocks 2, 5, 8, 11
             -> blkN: (B, 768, 14, 14) spatial token maps
             -> cls:  (B, 768)         global CLS descriptor
          |
          v  _SharedDecoder  [TRAINABLE, shared across all tasks]
             14×14 → 32×32 → 64×64 via skip connections
             -> feature_map: (B, 128, 64, 64)
          |
          v  Task submodel head  [TRAINABLE, task-specific]
             -> heatmap: (B, 1, 64, 64)

    Training strategy:
        core.freeze()        — freezes backbone only; decoder stays trainable
        optimizer = Adam(list(core.decoder.parameters()) + list(submodel.parameters()))
        checkpoint saves both decoder_state_dict and submodel_state_dict

    FAISS VectorDB:
        core.encode(x)  — returns 768-dim L2-normalised CLS embedding
        Independent of the decoder; purely from the frozen backbone.
    """

    def __init__(self):
        super().__init__()
        from terratorch import BACKBONE_REGISTRY
        self.backbone = BACKBONE_REGISTRY.build(
            _BACKBONE_ID,
            pretrained=True,
            num_frames=1,
            out_indices=list(_OUT_INDICES),
        )
        self.decoder = _SharedDecoder()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Resize (B, 6, H, W) → (B, 6, 1, 224, 224) for Prithvi."""
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x.unsqueeze(2)   # add temporal dimension T=1

    @staticmethod
    def _tokens_to_spatial(tokens: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N+1, D) token sequence → (B, D, 14, 14).

        Strips the CLS token (index 0) and reshapes patch tokens to a spatial grid.
        """
        patch_tokens = tokens[:, 1:, :]        # (B, 196, 768) — drop CLS
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)                  # 14
        return patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)   # (B, D, H, W)

    def _run_backbone(self, x_prithvi: torch.Tensor) -> list:
        """Run backbone on (B, 6, 1, 224, 224); return list of 4 token tensors."""
        return self.backbone(x_prithvi)        # [blk2, blk5, blk8, blk11], each (B, 197, 768)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> dict:
        """Run the frozen backbone and return raw multi-scale Prithvi token maps.

        Call this inside torch.no_grad() during training since the backbone
        is frozen. The returned dict is then passed to decode().

        Args:
            x: (B, 6, H, W) HLS-normalised Prithvi input — bands in the order
               [B02, B03, B04, B8A, B11, B12], pre-normalised by PRITHVI_MEAN/STD.

        Returns:
            dict with keys blk2, blk5, blk8, blk11: (B, 768, 14, 14)
                          cls: (B, 768) CLS token from the deepest block
        """
        tokens_list = self._run_backbone(self._prepare_input(x))  # 4 × (B, 197, 768)
        sp = self._tokens_to_spatial
        return {
            "blk2":  sp(tokens_list[0]),
            "blk5":  sp(tokens_list[1]),
            "blk8":  sp(tokens_list[2]),
            "blk11": sp(tokens_list[3]),
            "cls":   tokens_list[3][:, 0, :],   # CLS from deepest block (B, 768)
        }

    def decode(self, features: dict) -> torch.Tensor:
        """Run the trainable shared decoder on Prithvi features.

        Produces a (B, _DEC_CHANNELS, 64, 64) feature map consumed by task heads.
        Call this OUTSIDE torch.no_grad() so the decoder receives gradients.

        Args:
            features: dict returned by extract_features().

        Returns:
            (B, 128, 64, 64) spatial feature map.
        """
        return self.decoder(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a 768-dim L2-normalised CLS embedding for FAISS.

        Uses only the frozen backbone — no decoder pass needed.
        During inference after extract_features(), prefer:
            F.normalize(features['cls'], p=2, dim=1)
        to avoid a redundant backbone pass.

        Args:
            x: (B, 6, H, W) HLS-normalised Prithvi input.
        """
        tokens_list = self._run_backbone(self._prepare_input(x))
        return F.normalize(tokens_list[-1][:, 0, :], p=2, dim=1)

    def freeze(self):
        """Freeze the Prithvi backbone only. Decoder remains trainable.

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
    print(f"  backbone (frozen):   {backbone:,}")
    print(f"  decoder (trainable): {decoder:,}")

    core.freeze()
    dummy = torch.randn(2, 6, 64, 64)   # 6-band Prithvi input
    with torch.no_grad():
        feats = core.extract_features(dummy)
    fmap = core.decode(feats)
    emb  = core.encode(dummy)
    print(f"\n  blk11: {tuple(feats['blk11'].shape)}")
    print(f"  decode: {tuple(fmap.shape)}")
    print(f"  encode: {tuple(emb.shape)}")
