"""
Training script for the HousingEdgeCNN structure detection model.
Trains the model to predict per-pixel built-up / structure masks from
RGB satellite images, using NDBI-derived pseudo-labels as ground truth.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import HousingEdgeCNN, count_parameters
from dataset import get_dataloaders


def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """IoU for binary structure masks."""
    pred_binary   = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection  = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred: torch.Tensor, target: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """Dice coefficient for binary structure masks."""
    pred_binary   = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection  = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss.

    BCE handles class imbalance well (most satellite pixels are non-structure).
    Dice ensures the spatial extent of predicted structures matches the label.
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
    train_loader, val_loader, _ = get_dataloaders(
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
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou":              val_iou,
                "val_dice":             val_dice,
                "val_loss":             val_loss,
            }, save_path)
            print(f"  ** New best model saved (IoU: {val_iou:.4f}) **")

        print()

    final_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch":                args.epochs,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    print(f"Training complete. Best validation IoU: {best_val_iou:.4f}")
    print(f"Models saved to: {checkpoint_dir}")


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
