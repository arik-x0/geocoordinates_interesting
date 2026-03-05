"""Training script for the elevation POI detection submodel.

Loads the frozen CoreSatelliteModel (EVA-02 ViT-S/14) to extract RGB
features, then trains the ElevationPOITransUNet submodel using those
features plus raw topographic channels (DEM, slope, aspect).
After training, builds a FAISS VectorDB index from core embeddings.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_elevation_dataloaders as get_dataloaders  # noqa: E402
import torch.nn as nn
from training_utils import (  # noqa: E402
    compute_iou, compute_dice, build_embedding_index,
)
from core.model import CoreSatelliteModel

from model import ElevationPOITransUNet, count_parameters


class HeatmapLoss(nn.Module):
    def __init__(self, mse_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return self.mse_weight * mse_loss + self.dice_weight * dice_loss


def train_one_epoch(core, submodel, loader, criterion, optimizer, device):
    """Train the submodel for one epoch. Core is frozen — no gradients through it."""
    submodel.train()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for rgb, masks, _meta in pbar:
        rgb   = rgb.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Core runs frozen — extract features with no gradient tracking
        with torch.no_grad():
            features = core.extract_features(rgb)

        # Submodel trains on core features
        predictions = submodel(features)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(predictions.detach(), masks)
        total_dice += compute_dice(predictions.detach(), masks)
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = n_batches
    return total_loss / n, total_iou / n, total_dice / n


@torch.no_grad()
def validate(core, submodel, loader, criterion, device):
    """Validate the submodel. Returns average loss, IoU, Dice."""
    submodel.eval()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    for rgb, masks, _meta in tqdm(loader, desc="  Val  ", leave=False):
        rgb   = rgb.to(device)
        masks = masks.to(device)
        features    = core.extract_features(rgb)
        predictions = submodel(features)
        loss = criterion(predictions, masks)
        total_loss += loss.item()
        total_iou  += compute_iou(predictions, masks)
        total_dice += compute_dice(predictions, masks)
        n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


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

    # Core: frozen pretrained EVA-02 ViT-S/14 — feature extractor only
    print("\n--- Loading Core Model (EVA-02 ViT-S/14, frozen) ---")
    core = CoreSatelliteModel().freeze().to(device)
    core_params = sum(p.numel() for p in core.parameters())
    print(f"  Core params (frozen): {core_params:,}")

    # Submodel: lightweight decoder + SE head trained from scratch
    submodel = TransUNet(out_channels=1).to(device)
    print(f"  TransUNet submodel (trainable): {count_parameters(submodel):,} params")

    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = Adam(submodel.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5,
                                  verbose=True)

    best_val_iou = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.json"
    epoch_logs: list = []
    training_start = time.time()

    print(f"\n--- Training for {args.epochs} epochs ---")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_iou, train_dice = train_one_epoch(
            core, submodel, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou, val_dice = validate(
            core, submodel, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration   = time.time() - epoch_start
        elapsed          = time.time() - training_start
        remaining_epochs = args.epochs - epoch
        eta_s            = (elapsed / epoch) * remaining_epochs
        eta_str          = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"

        print(f"  Train — Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
        print(f"  LR: {current_lr:.6f}  |  Time: {epoch_duration:.1f}s | ETA: {eta_str}")

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save({
                "epoch":                   epoch,
                "submodel_state_dict":     submodel.state_dict(),
                "optimizer_state_dict":    optimizer.state_dict(),
                "val_iou":                 val_iou,
                "val_dice":                val_dice,
                "val_loss":                val_loss,
            }, checkpoint_dir / "best_model.pth")
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
    print(f"Training complete in {total_time / 60:.1f}m.  Best IoU: {best_val_iou:.4f}")

    torch.save({
        "epoch":                args.epochs,
        "submodel_state_dict":  submodel.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_dir / "final_model.pth")
    print(f"Models saved to: {checkpoint_dir}")

    # Build VectorDB using core's CLS-token embeddings (no decoder needed)
    build_embedding_index(
        core.encode,
        [train_loader, val_loader, test_loader],
        device,
        checkpoint_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train vegetation greenery submodel on frozen EVA-02 core features"
    )
    parser.add_argument("--data-dir",        type=str,   default="data")
    parser.add_argument("--checkpoint-dir",  type=str,   default="checkpoints")
    parser.add_argument("--epochs",          type=int,   default=25)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--ndvi-threshold",  type=float, default=0.3)
    parser.add_argument("--num-workers",     type=int,   default=0)
    args = parser.parse_args()
    train(args)
