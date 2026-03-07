"""
Shared training utilities used across all nine POI submodel pipelines.

Consumed in two ways:
    - base/trainer.py imports compute_iou, compute_dice, and build_embedding_index,
      making them available to every submodel through BaseTrainer inheritance.
    - Individual trainer files import DiceBCELoss directly (all submodels except
      elevation, which uses its own HeatmapLoss for regression-style heatmap output).
"""

import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """Intersection over Union for binary or thresholded-continuous predictions."""
    pred_binary   = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection  = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred: torch.Tensor, target: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """Dice coefficient for binary or thresholded-continuous predictions."""
    pred_binary   = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection  = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


# ── Losses ────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for segmentation training.

    BCE handles class imbalance; Dice ensures spatial extent matches the label.
    """

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (
            pred.sum() + target.sum() + smooth
        )
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


class NTXentLoss(nn.Module):
    """SimCLR-style NT-Xent loss over two augmented views of the same batch.

    Positive pairs:  (emb1[i], emb2[i]) — two augmented views of the same tile.
    Negative pairs:  all other images in both views (2N − 2 negatives per anchor).

    Both embedding tensors must be L2-normalised so that inner product equals
    cosine similarity. model.encode() and forward(..., return_embedding=True)
    both return L2-normalised vectors.

    This replaces the previous class-name-based approach (which used coarse
    EuroSAT class labels as positive-pair signal). Augmentation-based positives
    are tile-specific and semantically precise: the same image under different
    spatial transforms is a true positive pair regardless of class label noise.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1: (B, D) L2-normalised embeddings from view 1 (original)
            emb2: (B, D) L2-normalised embeddings from view 2 (augmented)
        """
        B      = emb1.size(0)
        device = emb1.device

        # Concatenate both views: (2B, D)
        embeddings = torch.cat([emb1, emb2], dim=0)

        # Pairwise cosine similarities (2B, 2B)
        sim = torch.mm(embeddings, embeddings.T) / self.temperature

        # Mask out self-similarities (diagonal)
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        # Positive pair for row i is i+B; for row i+B it is i
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(B,        device=device),
        ])
        return F.cross_entropy(sim, labels)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Apply random spatial augmentations to a batch of satellite tiles.

    Valid for satellite imagery: only spatial transforms are applied.
    Colour/brightness augmentations are intentionally omitted — Sentinel-2
    bands are radiometrically calibrated and their values carry physical meaning.

    Transforms applied per batch (same transform to all samples in the batch):
        - Random horizontal flip  (prob 0.5)
        - Random vertical flip    (prob 0.5)
        - Random 90° rotation     (one of 0°, 90°, 180°, 270° uniformly)

    All channels (RGB, DEM, slope, aspect for the elevation model) are
    transformed together, preserving spatial correspondence between channels.

    Args:
        x: Tensor of shape (B, C, H, W)

    Returns:
        Augmented tensor of shape (B, C, H, W)
    """
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[3])   # horizontal flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[2])   # vertical flip
    k = torch.randint(0, 4, (1,)).item()
    return torch.rot90(x, k, dims=[2, 3])


# ── FAISS index builder ───────────────────────────────────────────────────────

@torch.no_grad()
def build_embedding_index(encode_fn, loaders, device, checkpoint_dir: Path):
    """Extract embeddings from all splits and build a FAISS IndexFlatIP.

    encode_fn must accept a batch tensor (B, C, H, W) on `device` and return
    (B, D) L2-normalised float32 embeddings.  Pass CoreSatelliteModel.encode
    directly, or a lambda that slices channels first (e.g. for the elevation
    model whose dataloader yields 6-channel batches but core.encode expects 3):

        build_embedding_index(lambda x: core.encode(x[:, :3]), loaders, ...)

    Inner product on L2-normalised vectors equals cosine similarity, so the
    resulting IndexFlatIP supports exact nearest-neighbour retrieval.

    Saved files:
        <checkpoint_dir>/embedding_index.faiss  — FAISS index (exact cosine sim)
        <checkpoint_dir>/embedding_meta.json    — per-vector metadata list
    """
    try:
        import faiss
    except ImportError:
        warnings.warn(
            "faiss-cpu not installed — skipping VectorDB index build. "
            "Install with: pip install faiss-cpu",
            stacklevel=2,
        )
        return

    from tqdm import tqdm

    all_embeddings: List[np.ndarray] = []
    all_meta: List[dict] = []

    print("\n--- Building VectorDB Embedding Index ---")
    for split_name, loader in zip(["train", "val", "test"], loaders):
        for inputs, _targets, metas in tqdm(loader, desc=f"  Encoding {split_name}"):
            inputs = inputs.to(device)
            emb = encode_fn(inputs).cpu().numpy()   # (B, D) L2-normalised
            for i in range(len(emb)):
                all_embeddings.append(emb[i])
                # Store split name plus all metadata keys from the dataset
                entry = {"split": split_name}
                for k, v in metas[i].items():
                    entry[k] = v
                all_meta.append(entry)

    matrix = np.stack(all_embeddings).astype(np.float32)
    dim    = matrix.shape[1]

    # IndexFlatIP on L2-normalised vectors = exact cosine similarity search
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    index_path = checkpoint_dir / "embedding_index.faiss"
    meta_path  = checkpoint_dir / "embedding_meta.json"
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"  Indexed {index.ntotal} vectors (dim={dim})")
    print(f"  Index:    {index_path}")
    print(f"  Metadata: {meta_path}")
