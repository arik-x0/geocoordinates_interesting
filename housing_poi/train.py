"""
Training script for the HousingEdgeCNN structure detection model.
Trains the model to predict per-pixel built-up / structure masks from
RGB satellite images, using NDBI-derived pseudo-labels as ground truth.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Shared modules live at the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_housing_dataloaders as get_dataloaders  # noqa: E402
from training_utils import (  # noqa: E402
    compute_iou, compute_dice, DiceBCELoss, build_embedding_index,
)

from model import HousingEdgeCNN, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss, IoU, Dice."""
    model.train()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for rgb, labels, _meta in pbar:
        rgb    = rgb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(rgb)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(predictions.detach(), labels)
        total_dice += compute_dice(predictions.detach(), labels)
        n_batches  += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model. Returns average loss, IoU, Dice."""
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    n_batches = 0

    for rgb, labels, _meta in tqdm(loader, desc="  Val  ", leave=False):
        rgb    = rgb.to(device)
        labels = labels.to(device)

        predictions = model(rgb)
        loss = criterion(predictions, labels)

        total_loss += loss.item()
        total_iou  += compute_iou(predictions, labels)
        total_dice += compute_dice(predictions, labels)
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
    )

    # Model
    model = HousingEdgeCNN(in_channels=3, out_channels=1).to(device)
    print(f"\nHousingEdgeCNN initialized: {count_parameters(model):,} trainable parameters")

    # Training setup
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3,
                                  factor=0.5, verbose=True)

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

    # Build VectorDB from the best model's stage-4 bottleneck embeddings
    best_ckpt = torch.load(checkpoint_dir / "best_model.pth",
                           map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    build_embedding_index(
        model, [train_loader, val_loader, test_loader], device, checkpoint_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HousingEdgeCNN for low-density residential detection"
    )
    parser.add_argument("--data-dir",       type=str, default="data",
                        help="Directory to download/find EuroSAT dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs",         type=int, default=25)
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--num-workers",    type=int, default=0)
    args = parser.parse_args()
    train(args)
