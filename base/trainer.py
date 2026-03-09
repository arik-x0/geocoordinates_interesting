"""
Base training class for all POI submodel pipelines.

Architecture contract:
    core.freeze()    -- freezes ViT backbone only; decoder stays trainable
    optimizer        -- Adam over core.decoder.parameters() + submodel.parameters()
    forward pass:
        with torch.no_grad():
            features = core.extract_features(rgb)   # frozen backbone
        feature_map = core.decode(features)          # trainable shared decoder
        predictions = submodel(feature_map, [topo])  # task-specific head

Checkpoint format:
    {
        "epoch":                int,
        "decoder_state_dict":   core.decoder.state_dict(),
        "submodel_state_dict":  submodel.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_iou":              float,  (best checkpoint only)
        "val_dice":             float,
        "val_loss":             float,
    }

Subclasses override:
    get_dataloaders()  -> (train_loader, val_loader, test_loader)
    build_submodel()   -> nn.Module  (task head only)
    build_criterion()  -> nn.Module
    rgb_slice()        -> extract RGB from batch          (default: full input)
    extra_slice()      -> extra tensor for submodel head  (default: None)
    get_encode_fn()    -> callable for FAISS              (default: core.encode)
    submodel_name      -> str shown in the training header
    add_args()         -> classmethod to register task-specific CLI flags
"""

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
from training_utils import compute_iou, compute_dice, build_embedding_index  # noqa: E402
from core.model import CoreSatelliteModel                                     # noqa: E402
from base.submodel import count_parameters                                    # noqa: E402


class BaseTrainer:
    """Shared training orchestration for all POI submodels.

    The frozen ViT backbone provides raw token maps; the trainable shared
    UNet decoder (inside CoreSatelliteModel) produces a 64x64 feature map;
    the task submodel head produces the final prediction.

    Both the decoder and the submodel head are optimised together.
    """

    submodel_name: str = "Submodel"

    def __init__(self, args):
        self.args = args

    # -- Abstract interface --------------------------------------------------

    def get_dataloaders(self):
        """Return (train_loader, val_loader, test_loader)."""
        raise NotImplementedError

    def build_submodel(self) -> torch.nn.Module:
        """Instantiate and return the task head (not yet on device)."""
        raise NotImplementedError

    def build_criterion(self) -> torch.nn.Module:
        """Return the loss function for this task."""
        raise NotImplementedError

    # -- Optional overrides --------------------------------------------------

    def rgb_slice(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the RGB channels from a batch tensor (default: full input)."""
        return inputs

    def extra_slice(self, inputs: torch.Tensor):
        """Return extra tensor passed to submodel.forward() (default: None).

        Override for multi-stream tasks:
            def extra_slice(self, inputs):
                return inputs[:, 3:]   # elevation: DEM+slope+aspect channels
        """
        return None

    def get_encode_fn(self, core):
        """Return a callable batch -> L2-normalised embedding for FAISS.

        Default: core.encode (expects RGB input).
        Override if the dataloader yields multi-channel batches:
            def get_encode_fn(self, core):
                return lambda x: core.encode(x[:, :3])
        """
        return core.encode

    @classmethod
    def add_args(cls, parser):
        """Register task-specific CLI arguments. Override as needed."""

    # -- Training loop -------------------------------------------------------

    def _train_one_epoch(self, core, submodel, loader, criterion, optimizer, device):
        core.decoder.train()
        submodel.train()
        total_loss = total_iou = total_dice = 0.0
        n_batches  = 0
        pbar = tqdm(loader, desc="  Train", leave=False)
        for inputs, targets, _meta in pbar:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # Backbone frozen: run under no_grad to skip building its graph
            with torch.no_grad():
                features = core.extract_features(self.rgb_slice(inputs))

            # Decoder + head: both trainable, gradients flow normally
            feature_map = core.decode(features)
            extra = self.extra_slice(inputs)
            if extra is not None:
                predictions = submodel(feature_map, extra)
            else:
                predictions = submodel(feature_map)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iou  += compute_iou(predictions.detach(), targets)
            total_dice += compute_dice(predictions.detach(), targets)
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches

    @torch.no_grad()
    def _validate(self, core, submodel, loader, criterion, device):
        core.decoder.eval()
        submodel.eval()
        total_loss = total_iou = total_dice = 0.0
        n_batches  = 0
        for inputs, targets, _meta in tqdm(loader, desc="  Val  ", leave=False):
            inputs  = inputs.to(device)
            targets = targets.to(device)
            features    = core.extract_features(self.rgb_slice(inputs))
            feature_map = core.decode(features)
            extra = self.extra_slice(inputs)
            if extra is not None:
                predictions = submodel(feature_map, extra)
            else:
                predictions = submodel(feature_map)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            total_iou  += compute_iou(predictions, targets)
            total_dice += compute_dice(predictions, targets)
            n_batches  += 1
        return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches

    # -- Main entry point ----------------------------------------------------

    def run(self):
        args   = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("\n--- Loading Dataset ---")
        train_loader, val_loader, test_loader = self.get_dataloaders()

        print("\n--- Loading Core Model (Prithvi-EO-1.0-100M backbone, frozen) ---")
        core = CoreSatelliteModel().freeze().to(device)
        backbone_params = sum(p.numel() for p in core.backbone.parameters())
        decoder_params  = sum(p.numel() for p in core.decoder.parameters())
        print(f"  Backbone (frozen):           {backbone_params:,}")
        print(f"  Shared decoder (trainable):  {decoder_params:,}")

        submodel  = self.build_submodel().to(device)
        criterion = self.build_criterion()
        print(f"  {self.submodel_name} head (trainable): {count_parameters(submodel):,} params")

        # Decoder and submodel head are optimised jointly
        optimizer = Adam(
            list(core.decoder.parameters()) + list(submodel.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5,
                                      verbose=True)

        best_val_iou   = 0.0
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path    = checkpoint_dir / "training_log.json"
        epoch_logs: list = []
        training_start  = time.time()

        print(f"\n--- Training for {args.epochs} epochs ---")
        print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            print(f"Epoch {epoch}/{args.epochs}")

            train_loss, train_iou, train_dice = self._train_one_epoch(
                core, submodel, train_loader, criterion, optimizer, device)
            val_loss, val_iou, val_dice = self._validate(
                core, submodel, val_loader, criterion, device)

            scheduler.step(val_loss)
            current_lr     = optimizer.param_groups[0]["lr"]
            epoch_duration = time.time() - epoch_start
            elapsed        = time.time() - training_start
            eta_s          = (elapsed / epoch) * (args.epochs - epoch)
            eta_str        = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"

            print(f"  Train -- Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
            print(f"  Val   -- Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
            print(f"  LR: {current_lr:.6f}  |  Time: {epoch_duration:.1f}s | ETA: {eta_str}")

            is_best = val_iou > best_val_iou
            if is_best:
                best_val_iou = val_iou
                torch.save({
                    "epoch":                epoch,
                    "decoder_state_dict":   core.decoder.state_dict(),
                    "submodel_state_dict":  submodel.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou":              val_iou,
                    "val_dice":             val_dice,
                    "val_loss":             val_loss,
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
            "decoder_state_dict":   core.decoder.state_dict(),
            "submodel_state_dict":  submodel.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_dir / "final_model.pth")
        print(f"Models saved to: {checkpoint_dir}")

        build_embedding_index(
            self.get_encode_fn(core),
            [train_loader, val_loader, test_loader],
            device,
            checkpoint_dir,
        )
