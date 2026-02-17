"""
Training script for the satellite greenery segmentation U-Net.
Downloads EuroSAT, generates NDVI ground-truth masks, and trains
the model to segment greenery from desert-like areas.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import UNet, count_parameters
from dataset import get_dataloaders


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary segmentation."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0  # Both empty = perfect match
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

        # Smooth Dice loss
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss, IoU, and Dice."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for rgb, masks, _meta in pbar:
        rgb = rgb.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        predictions = model(rgb)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(predictions.detach(), masks)
        total_dice += compute_dice(predictions.detach(), masks)
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model. Returns average loss, IoU, and Dice."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    for rgb, masks, _meta in tqdm(loader, desc="  Val  ", leave=False):
        rgb = rgb.to(device)
        masks = masks.to(device)

        predictions = model(rgb)
        loss = criterion(predictions, masks)

        total_loss += loss.item()
        total_iou += compute_iou(predictions, masks)
        total_dice += compute_dice(predictions, masks)
        n_batches += 1

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("\n--- Loading Dataset ---")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ndvi_threshold=args.ndvi_threshold,
    )

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(f"\nU-Net initialized: {count_parameters(model):,} trainable parameters")

    # Training setup
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

    best_val_iou = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Training for {args.epochs} epochs ---\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train — Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
                "val_loss": val_loss,
            }, save_path)
            print(f"  ** New best model saved (IoU: {val_iou:.4f}) **")

        print()

    # Save final model
    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    print(f"Training complete. Best validation IoU: {best_val_iou:.4f}")
    print(f"Models saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train greenery segmentation U-Net on EuroSAT")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory to download/find EuroSAT dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--ndvi-threshold", type=float, default=0.3,
                        help="NDVI threshold for greenery classification (0-1)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes")
    args = parser.parse_args()
    train(args)
