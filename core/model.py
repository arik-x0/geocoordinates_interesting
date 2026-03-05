"""
Core satellite feature extractor.

Pretrained EVA-02 ViT-S/14 backbone (MIT license, BAAI) used as a FROZEN
feature extractor for all task-specific submodels.

The backbone processes an RGB satellite tile and exposes intermediate
transformer block outputs as 16×16 spatial feature maps. Submodels
receive these maps and add their own lightweight decoders + task heads
trained entirely from scratch.

Usage:
    core = CoreSatelliteModel().freeze().to(device)

    with torch.no_grad():
        features = core.extract_features(rgb)   # dict of spatial maps
        embedding = core.encode(rgb)             # 512-dim for FAISS

EVA-02 weights are downloaded automatically via timm on first run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKBONE_ID = "eva02_small_patch14_224"
_EMBED_DIM   = 384     # EVA-02 ViT-S hidden dimension
_GRID_SIZE   = 16      # 224px / 14px patch = 16 spatial tokens per axis
_HOOK_BLOCKS = (2, 5, 8, 11)  # tap features at 1/4, 2/4, 3/4, full depth


class CoreSatelliteModel(nn.Module):
    """EVA-02 ViT-S/14 frozen backbone for satellite image feature extraction.

    Architecture:
        Input  (B, 3, 64, 64) RGB satellite tile
          ↓  Bilinear resize → 224×224
          ↓  EVA-02 ViT-S/14  (12 blocks, embed_dim=384, 6 heads, patch_size=14)
             Forward hooks capture intermediate outputs at blocks 2, 5, 8, 11
          ↓  _tokens_to_spatial: (B, 257, 384) → (B, 384, 16, 16)

    extract_features() returns a dict of spatial maps, one per hooked block,
    plus the CLS token.  Submodels consume these maps and add their own
    decoders trained from scratch.

    encode() skips the spatial reshape and returns a 512-dim L2-normalised
    CLS-token embedding for FAISS nearest-neighbour retrieval.
    """

    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            _BACKBONE_ID, pretrained=True, num_classes=0,
        )

        # Per-block feature cache — populated by forward hooks
        self._feat_cache: dict = {}
        for idx in _HOOK_BLOCKS:
            self.backbone.blocks[idx].register_forward_hook(
                lambda _m, _inp, out, i=idx: self._feat_cache.__setitem__(i, out)
            )

        # CLS-token projection for FAISS (384 → 512)
        self.embed_proj = nn.Linear(_EMBED_DIM, 512)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N+1, D) token sequence → (B, D, 16, 16) spatial map.

        Drops the CLS token and reshapes the N=256 patch tokens to 2-D.
        """
        B = tokens.shape[0]
        return (
            tokens[:, 1:]                                            # (B, 256, D)
            .transpose(1, 2)                                         # (B, D, 256)
            .reshape(B, _EMBED_DIM, _GRID_SIZE, _GRID_SIZE)         # (B, D, 16, 16)
        )

    def _run_backbone(self, x_224: torch.Tensor):
        """Run backbone on a 224×224 tensor; return (final_tokens, cls_token)."""
        self._feat_cache.clear()
        final = self.backbone.forward_features(x_224)   # (B, 257, 384) post-norm
        return final, final[:, 0]                        # tokens, cls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract multi-depth spatial feature maps from an RGB satellite tile.

        Each feature map is the spatial reshape of intermediate transformer
        block outputs.  All maps are at 16×16 spatial resolution and 384
        channels (the ViT's embed_dim).  Submodels project and upsample
        these maps with their own trainable layers.

        Args:
            x: (B, 3, 64, 64) RGB tile, values nominally in [0, 1].

        Returns:
            dict with keys:
                'blk2':  (B, 384, 16, 16)  — early/low-level features
                'blk5':  (B, 384, 16, 16)  — mid-level features
                'blk8':  (B, 384, 16, 16)  — deep features
                'blk11': (B, 384, 16, 16)  — deepest / most semantic
                'cls':   (B, 384)           — CLS token (global descriptor)
        """
        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        final, cls_token = self._run_backbone(x_224)
        sp = self._tokens_to_spatial
        return {
            "blk2":  sp(self._feat_cache[2]),   # (B, 384, 16, 16)
            "blk5":  sp(self._feat_cache[5]),   # (B, 384, 16, 16)
            "blk8":  sp(self._feat_cache[8]),   # (B, 384, 16, 16)
            "blk11": sp(self._feat_cache[11]),  # (B, 384, 16, 16)
            "cls":   cls_token,                  # (B, 384)
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a 512-dim L2-normalised embedding from the CLS token.

        Faster than extract_features() — skips the spatial reshape.
        Use this for building and querying the FAISS VectorDB index.

        Args:
            x: (B, 3, 64, 64) RGB tile.

        Returns:
            (B, 512) L2-normalised float32 embedding.
        """
        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        self._feat_cache.clear()
        final = self.backbone.forward_features(x_224)   # (B, 257, 384)
        cls   = final[:, 0]                              # (B, 384)
        emb   = self.embed_proj(cls)                     # (B, 512)
        return F.normalize(emb, p=2, dim=1)

    def freeze(self):
        """Freeze all backbone parameters. Call before submodel training."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for optional end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        return self


if __name__ == "__main__":
    core = CoreSatelliteModel()
    total = sum(p.numel() for p in core.parameters())
    print(f"CoreSatelliteModel  total params: {total:,}")
    print(f"  backbone: {sum(p.numel() for p in core.backbone.parameters()):,}")
    print(f"  embed_proj: {sum(p.numel() for p in core.embed_proj.parameters()):,}")

    core.freeze()
    dummy = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = core.extract_features(dummy)
        emb   = core.encode(dummy)

    for k, v in feats.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"  encode: {tuple(emb.shape)}")
