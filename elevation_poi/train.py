"""
Training script for the elevation POI detection TransUNet.
Trains the hybrid Transformer + U-Net model to predict cliff-near-water
heatmaps from 6-channel satellite + DEM input.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import ElevationPOIUNet, count_parameters
from dataset import get_dataloaders


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.3) -> float:
    """IoU for continuous heatmaps (thresholded to binary)."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.3) -> float:
    """Dice coefficient for continuous heatmaps."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


class HeatmapLoss(nn.Module):
    """Combined MSE + Dice loss for heatmap regression.

    MSE captures the continuous intensity, while Dice ensures
    the spatial overlap of high-POI regions is preserved.
    """

    def __init__(self, mse_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)

        # Smooth Dice loss on continuous values
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return self.mse_weight * mse_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for inputs, targets, _meta in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(predictions.detach(), targets)
        total_dice += compute_dice(predictions.detach(), targets)
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    for inputs, targets, _meta in tqdm(loader, desc="  Val  ", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        total_loss += loss.item()
        total_iou += compute_iou(predictions, targets)
        total_dice += compute_dice(predictions, targets)
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
        use_real_dem=args.use_real_dem,
        cliff_threshold=args.cliff_threshold,
        water_threshold=args.water_threshold,
    )

    # Model
    model = ElevationPOIUNet(in_channels=6, out_channels=1).to(device)
    print(f"\nElevation POI TransUNet: {count_parameters(model):,} trainable parameters")
    print(f"Input channels: RGB(3) + DEM(1) + Slope(1) + Aspect(1) = 6")

    # Training setup
    criterion = HeatmapLoss(mse_weight=0.5, dice_weight=0.5)
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

    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    print(f"Training complete. Best validation IoU: {best_val_iou:.4f}")
    print(f"Models saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train elevation POI detection ViT")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for EuroSAT + SRTM data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (smaller than greenery model due to 6-ch input)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cliff-threshold", type=float, default=15.0,
                        help="Minimum slope (degrees) for cliff classification")
    parser.add_argument("--water-threshold", type=float, default=0.3,
                        help="NDWI threshold for water detection")
    parser.add_argument("--use-real-dem", action="store_true", default=False,
                        help="Download real SRTM DEM (requires internet). "
                             "Default: synthetic DEM for testing.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(args)
