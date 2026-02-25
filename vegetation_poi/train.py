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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Shared modules live at the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_vegetation_dataloaders as get_dataloaders  # noqa: E402
from training_utils import (  # noqa: E402
    compute_iou, compute_dice, DiceBCELoss, NTXentLoss,
    augment_batch, build_embedding_index,
)

from model import TransUNet, count_parameters


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
    for rgb, masks, _meta in pbar:
        rgb   = rgb.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        # Single forward pass — encoder runs once, embedding (view 1) comes for free
        predictions, emb1 = model(rgb, return_embedding=True)
        seg_loss = criterion(predictions, masks)

        # SimCLR-style contrastive: second view is a spatially augmented copy
        rgb_aug = augment_batch(rgb)
        emb2 = model.encode(rgb_aug)
        contra_loss = contrastive_criterion(emb1, emb2)

        loss = seg_loss + contrastive_weight * contra_loss
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_seg    += seg_loss.item()
        total_contra += contra_loss.item()
        total_iou    += compute_iou(predictions.detach(), masks)
        total_dice   += compute_dice(predictions.detach(), masks)
        n_batches    += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ctr=f"{contra_loss.item():.4f}")

    n = n_batches
    return total_loss / n, total_seg / n, total_contra / n, total_iou / n, total_dice / n


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

    criterion             = DiceBCELoss(dice_weight=0.5)
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
        description="Train greenery segmentation TransUNet and build VectorDB index"
    )
    parser.add_argument("--data-dir",       type=str,   default="data")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--epochs",         type=int,   default=25)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--ndvi-threshold",    type=float, default=0.3)
    parser.add_argument("--num-workers",       type=int,   default=0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="λ: weight of NT-Xent contrastive loss (0 = disabled)")
    parser.add_argument("--temperature",        type=float, default=0.07,
                        help="τ: NT-Xent softmax temperature (lower = sharper)")
    args = parser.parse_args()
    train(args)
