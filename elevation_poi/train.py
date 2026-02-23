"""
Training script for the elevation POI detection TransUNet.
Trains the hybrid Transformer + U-Net model to predict cliff-near-water
heatmaps from 6-channel satellite + DEM input, then builds a FAISS VectorDB
embedding index from the trained model's Transformer bottleneck representations.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Shared dataset module lives at the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_elevation_dataloaders as get_dataloaders  # noqa: E402

from model import ElevationPOITransUNet, count_parameters


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.3) -> float:
    """IoU for continuous heatmaps (thresholded to binary)."""
    pred_binary   = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection  = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.3) -> float:
    """Dice coefficient for continuous heatmaps."""
    pred_binary   = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection  = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


class HeatmapLoss(nn.Module):
    """Combined MSE + Dice loss for heatmap regression."""

    def __init__(self, mse_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.mse        = nn.MSELoss()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return self.mse_weight * mse_loss + self.dice_weight * dice_loss


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE) contrastive loss over a batch of L2-normalised embeddings.

    Positive pairs: images sharing the same EuroSAT class name in the batch.
    Negative pairs: all other images in the batch.

    Because model.encode() already returns L2-normalised vectors, inner product
    equals cosine similarity, so no extra normalisation is needed here.

    Loss for anchor i:
        -1/|P_i| * Σ_{p∈P_i} [ sim(i,p)/τ  −  log Σ_{j≠i} exp(sim(i,j)/τ) ]

    When there are no positive pairs for an anchor (its class appears only once
    in the batch) that anchor is skipped — its loss contribution is zero.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, class_names: list) -> torch.Tensor:
        B      = embeddings.size(0)
        device = embeddings.device

        # Cosine similarity matrix (B, B) — embeddings are L2-normalised
        sim = torch.mm(embeddings, embeddings.T) / self.temperature

        # Build positive mask: True where i and j share the same class
        unique_classes = list(set(class_names))
        class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
        label_ids = torch.tensor(
            [class_to_idx[c] for c in class_names], dtype=torch.long, device=device
        )
        pos_mask = (label_ids.unsqueeze(0) == label_ids.unsqueeze(1))  # (B, B)
        pos_mask.fill_diagonal_(False)

        if not pos_mask.any():
            return embeddings.sum() * 0.0   # differentiable zero

        # Denominator: log Σ_{j≠i} exp(sim(i,j)/τ)  — exclude self
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        log_denom = torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1)

        losses = []
        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            loss_i = -(sim[i, pos_idx] - log_denom[i]).mean()
            losses.append(loss_i)

        return torch.stack(losses).mean()


def train_one_epoch(model, loader, criterion, contrastive_criterion,
                    contrastive_weight, optimizer, device):
    """Train for one epoch.

    Returns:
        (total_loss, seg_loss, contra_loss, iou, dice) — all batch-averaged.
    """
    model.train()
    total_loss = total_seg = total_contra = total_iou = total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for inputs, targets, meta in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # Single forward pass — encoder runs once, embedding comes for free
        predictions, embeddings = model(inputs, return_embedding=True)
        seg_loss    = criterion(predictions, targets)

        # Contrastive loss on the bottleneck embeddings (no extra encoder pass)
        class_names = [m["class_name"] for m in meta]
        contra_loss = contrastive_criterion(embeddings, class_names)

        loss = seg_loss + contrastive_weight * contra_loss
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_seg    += seg_loss.item()
        total_contra += contra_loss.item()
        total_iou    += compute_iou(predictions.detach(), targets)
        total_dice   += compute_dice(predictions.detach(), targets)
        n_batches    += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ctr=f"{contra_loss.item():.4f}")

    n = n_batches
    return total_loss / n, total_seg / n, total_contra / n, total_iou / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    for inputs, targets, _meta in tqdm(loader, desc="  Val  ", leave=False):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        total_loss += loss.item()
        total_iou  += compute_iou(predictions, targets)
        total_dice += compute_dice(predictions, targets)
        n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def build_embedding_index(model, loaders, device, checkpoint_dir: Path):
    """Extract Transformer bottleneck embeddings and build a FAISS VectorDB index.

    Iterates all three data splits (train / val / test), calls model.encode()
    to get L2-normalised 512-dim vectors from the 6-channel input (RGB + DEM
    + Slope + Aspect), and builds a FAISS IndexFlatIP index.
    Similarity search therefore reflects both spectral and topographic likeness.

    Saved files:
        <checkpoint_dir>/embedding_index.faiss  — FAISS index (exact cosine sim)
        <checkpoint_dir>/embedding_meta.json    — per-vector metadata list
    """
    try:
        import faiss
    except ImportError:
        print("WARNING: faiss-cpu not installed — skipping VectorDB index build.")
        print("         Install with: pip install faiss-cpu")
        return

    model.eval()
    all_embeddings = []
    all_meta = []

    print("\n--- Building VectorDB Embedding Index ---")
    for split_name, loader in zip(["train", "val", "test"], loaders):
        for inputs, _targets, metas in tqdm(loader, desc=f"  Encoding {split_name}"):
            inputs = inputs.to(device)
            emb = model.encode(inputs).cpu().numpy()   # (B, 512) L2-normalised
            for i in range(len(emb)):
                all_embeddings.append(emb[i])
                all_meta.append({
                    "split":          split_name,
                    "filepath":       metas[i]["filepath"],
                    "class_name":     metas[i]["class_name"],
                    "has_water":      bool(metas[i]["has_water"]),
                    "has_cliffs":     bool(metas[i]["has_cliffs"]),
                    "water_fraction": float(metas[i]["water_fraction"]),
                    "max_slope":      float(metas[i]["max_slope"]),
                })

    matrix = np.stack(all_embeddings).astype(np.float32)
    dim = matrix.shape[1]   # 512

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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Loading Dataset ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_real_dem=args.use_real_dem,
        cliff_threshold=args.cliff_threshold,
        water_threshold=args.water_threshold,
    )

    model = ElevationPOITransUNet(in_channels=6, out_channels=1).to(device)
    print(f"\nElevation POI TransUNet: {count_parameters(model):,} trainable parameters")
    print(f"Input channels: RGB(3) + DEM(1) + Slope(1) + Aspect(1) = 6")

    criterion             = HeatmapLoss(mse_weight=0.5, dice_weight=0.5)
    contrastive_criterion = NTXentLoss(temperature=args.temperature)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)
    print(f"Contrastive weight λ={args.contrastive_weight}  temperature τ={args.temperature}")

    best_val_iou = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.json"

    epoch_logs: list = []
    training_start = time.time()

    print(f"\n--- Training for {args.epochs} epochs ---\n")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_seg, train_contra, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, contrastive_criterion,
            args.contrastive_weight, optimizer, device,
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration   = time.time() - epoch_start
        elapsed          = time.time() - training_start
        remaining_epochs = args.epochs - epoch
        eta_s            = (elapsed / epoch) * remaining_epochs
        eta_str          = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"

        print(f"  Train — Loss: {train_loss:.4f} | Seg: {train_seg:.4f} | "
              f"Contra: {train_contra:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_duration:.1f}s | Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}")

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            save_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou":              val_iou,
                "val_dice":             val_dice,
                "val_loss":             val_loss,
            }, save_path)
            print(f"  ** New best model saved (IoU: {val_iou:.4f}) **")

        epoch_logs.append({
            "epoch":             epoch,
            "timestamp":         datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch_duration_s":  round(epoch_duration, 2),
            "elapsed_s":         round(elapsed, 2),
            "train_loss":        round(train_loss, 6),
            "train_seg_loss":    round(train_seg, 6),
            "train_contra_loss": round(train_contra, 6),
            "train_iou":         round(train_iou, 6),
            "train_dice":        round(train_dice, 6),
            "val_loss":          round(val_loss, 6),
            "val_iou":           round(val_iou, 6),
            "val_dice":          round(val_dice, 6),
            "lr":                current_lr,
            "is_best":           is_best,
        })
        with open(log_path, "w") as f:
            json.dump(epoch_logs, f, indent=2)

        print()

    total_time = time.time() - training_start
    print(f"Training complete in {total_time/60:.1f}m ({total_time:.0f}s).")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Epoch log saved to:  {log_path}")

    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch":                args.epochs,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    print(f"Models saved to: {checkpoint_dir}")

    # Build VectorDB from the best model's Transformer bottleneck embeddings
    best_ckpt = torch.load(checkpoint_dir / "best_model.pth",
                           map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    build_embedding_index(
        model, [train_loader, val_loader, test_loader], device, checkpoint_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train elevation POI TransUNet and build VectorDB index"
    )
    parser.add_argument("--data-dir",        type=str,   default="data")
    parser.add_argument("--checkpoint-dir",  type=str,   default="checkpoints")
    parser.add_argument("--epochs",          type=int,   default=25)
    parser.add_argument("--batch-size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--cliff-threshold", type=float, default=15.0)
    parser.add_argument("--water-threshold", type=float, default=0.3)
    parser.add_argument("--use-real-dem",       action="store_true", default=False)
    parser.add_argument("--num-workers",        type=int,   default=0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="λ: weight of NT-Xent contrastive loss (0 = disabled)")
    parser.add_argument("--temperature",        type=float, default=0.07,
                        help="τ: NT-Xent softmax temperature (lower = sharper)")
    args = parser.parse_args()
    train(args)
