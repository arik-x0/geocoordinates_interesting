"""
Training script for the satellite greenery segmentation TransUNet.
Downloads EuroSAT, generates NDVI ground-truth masks, trains the hybrid
Transformer + U-Net model, and builds a FAISS VectorDB embedding index
from the trained model's Transformer bottleneck representations.
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
from dataset import get_vegetation_dataloaders as get_dataloaders  # noqa: E402

from model import TransUNet, count_parameters


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary segmentation."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient for binary segmentation."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    total = pred_binary.sum() + target.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for better segmentation training."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss, IoU, and Dice."""
    model.train()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for rgb, masks, _meta in pbar:
        rgb   = rgb.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        predictions = model(rgb)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(predictions.detach(), masks)
        total_dice += compute_dice(predictions.detach(), masks)
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model. Returns average loss, IoU, and Dice."""
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    for rgb, masks, _meta in tqdm(loader, desc="  Val  ", leave=False):
        rgb   = rgb.to(device)
        masks = masks.to(device)
        predictions = model(rgb)
        loss = criterion(predictions, masks)
        total_loss += loss.item()
        total_iou  += compute_iou(predictions, masks)
        total_dice += compute_dice(predictions, masks)
        n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def build_embedding_index(model, loaders, device, checkpoint_dir: Path):
    """Extract Transformer bottleneck embeddings and build a FAISS VectorDB index.

    Iterates all three data splits (train / val / test), calls model.encode()
    to get L2-normalised 512-dim vectors, and builds a FAISS IndexFlatIP index.
    Inner product on unit vectors equals cosine similarity, so the index
    supports nearest-neighbour retrieval by visual + spectral similarity.

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
        for rgb, _masks, metas in tqdm(loader, desc=f"  Encoding {split_name}"):
            rgb = rgb.to(device)
            emb = model.encode(rgb).cpu().numpy()   # (B, 512) L2-normalised
            for i in range(len(emb)):
                all_embeddings.append(emb[i])
                all_meta.append({
                    "split":      split_name,
                    "filepath":   metas[i]["filepath"],
                    "class_name": metas[i]["class_name"],
                    "ndvi_mean":  float(metas[i]["ndvi_mean"]),
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
        ndvi_threshold=args.ndvi_threshold,
    )

    model = TransUNet(in_channels=3, out_channels=1).to(device)
    print(f"\nTransUNet initialized: {count_parameters(model):,} trainable parameters")

    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

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

        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start
        elapsed        = time.time() - training_start
        remaining_epochs = args.epochs - epoch
        eta_s          = (elapsed / epoch) * remaining_epochs
        eta_str        = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"

        print(f"  Train — Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
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
            "epoch":            epoch,
            "timestamp":        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch_duration_s": round(epoch_duration, 2),
            "elapsed_s":        round(elapsed, 2),
            "train_loss":       round(train_loss, 6),
            "train_iou":        round(train_iou, 6),
            "train_dice":       round(train_dice, 6),
            "val_loss":         round(val_loss, 6),
            "val_iou":          round(val_iou, 6),
            "val_dice":         round(val_dice, 6),
            "lr":               current_lr,
            "is_best":          is_best,
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
        description="Train greenery segmentation TransUNet and build VectorDB index"
    )
    parser.add_argument("--data-dir",       type=str,   default="data")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--epochs",         type=int,   default=25)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--ndvi-threshold", type=float, default=0.3)
    parser.add_argument("--num-workers",    type=int,   default=0)
    args = parser.parse_args()
    train(args)
