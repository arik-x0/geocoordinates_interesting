"""
AestheticPredictor: runs all 6 aesthetic submodels + aggregator in one pipeline.

Usage:
    python -m submodels.meta predict \\
        --data-dir data \\
        --checkpoint-dir checkpoints \\
        --output-dir output/aesthetic \\
        --top-n 20

Checkpoint layout expected:
    checkpoints/fractal/best_model.pth
    checkpoints/water/best_model.pth
    checkpoints/color_harmony/best_model.pth
    checkpoints/symmetry/best_model.pth
    checkpoints/sublime/best_model.pth
    checkpoints/complexity/best_model.pth
    checkpoints/meta/best_model.pth       (optional)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.model import CoreSatelliteModel                       # noqa: E402
from dataset import get_fractal_dataloaders                    # noqa: E402

from submodels.fractal.model    import FractalPatternNet
from submodels.water.model      import WaterGeometryNet
from submodels.color_harmony.model import ColorHarmonyNet
from submodels.symmetry.model   import SymmetryOrderNet
from submodels.sublime.model    import ScaleSublimeNet
from submodels.complexity.model import ComplexityBalanceNet
from .model import AestheticAggregator, SUBMODEL_NAMES

# Submodel registry: (name, class, checkpoint-subdir)
_SUBMODEL_REGISTRY = [
    ("fractal",       FractalPatternNet,    "fractal"),
    ("water",         WaterGeometryNet,     "water"),
    ("color_harmony", ColorHarmonyNet,      "color_harmony"),
    ("symmetry",      SymmetryOrderNet,     "symmetry"),
    ("sublime",       ScaleSublimeNet,      "sublime"),
    ("complexity",    ComplexityBalanceNet, "complexity"),
]


def _load_submodel(cls, ckpt_path: Path, core, device):
    """Instantiate, load weights, and return an eval-mode submodel."""
    model = cls(out_channels=1).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["submodel_state_dict"])
        # Load decoder once from the first valid checkpoint
        if "decoder_state_dict" in ckpt and not getattr(core, "_decoder_loaded", False):
            core.decoder.load_state_dict(ckpt["decoder_state_dict"])
            core._decoder_loaded = True
        print(f"  Loaded {cls.__name__} from {ckpt_path.name}")
    else:
        print(f"  WARNING: {ckpt_path} not found — using random weights for {cls.__name__}")
    model.eval()
    return model


class AestheticPredictor:
    """Run all 6 submodels + aggregator and produce a ranked aesthetic output."""

    def __init__(self, args):
        self.args = args

    @torch.no_grad()
    def run(self):
        args   = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ckpt_dir  = Path(args.checkpoint_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load core model
        core = CoreSatelliteModel().freeze().to(device)
        core._decoder_loaded = False

        # Load all 6 submodels
        submodels = {}
        for name, cls, subdir in _SUBMODEL_REGISTRY:
            ckpt_path = ckpt_dir / subdir / "best_model.pth"
            submodels[name] = _load_submodel(cls, ckpt_path, core, device)

        # Load aggregator (optional)
        aggregator = AestheticAggregator(n_submodels=6).to(device)
        agg_ckpt   = ckpt_dir / "meta" / "best_model.pth"
        if agg_ckpt.exists():
            agg_state = torch.load(agg_ckpt, map_location=device, weights_only=True)
            aggregator.load_state_dict(agg_state["aggregator_state_dict"])
            print(f"  Loaded AestheticAggregator from {agg_ckpt}")
        else:
            print("  No aggregator checkpoint — using equal weights (untrained)")
        aggregator.eval()

        # Use fractal dataloader as proxy (same EuroSAT images, all tasks share it)
        _, _, test_loader = get_fractal_dataloaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=0,
        )

        all_results = []
        print("\nRunning aesthetic pipeline...")
        for inputs, _targets, metas in tqdm(test_loader, desc="Aesthetic"):
            inputs  = inputs.to(device)
            rgb_in  = inputs[:, :3]               # first 3 channels = RGB

            features    = core.extract_features(rgb_in)
            feature_map = core.decode(features)

            # Run all 6 submodels
            maps = []
            for name, _, _ in _SUBMODEL_REGISTRY:
                pred = submodels[name](feature_map)  # (B, 1, H, W)
                maps.append(pred)

            heatmaps  = torch.cat(maps, dim=1)       # (B, 6, H, W)
            aesthetic  = aggregator(heatmaps)         # (B, 1, H, W)

            inputs_cpu  = inputs.cpu().numpy()
            aes_cpu     = aesthetic.cpu().numpy()
            maps_cpu    = heatmaps.cpu().numpy()
            weights_cpu = aggregator.channel_weights(heatmaps).cpu().numpy()

            for i in range(inputs.size(0)):
                score = float(aes_cpu[i, 0].mean())
                all_results.append({
                    "rgb":             inputs_cpu[i, :3],
                    "aesthetic_map":   aes_cpu[i, 0],
                    "submodel_maps":   maps_cpu[i],       # (6, H, W)
                    "channel_weights": weights_cpu[i],    # (6,)
                    "aesthetic_score": score,
                    "class_name":      metas[i]["class_name"],
                    "filepath":        metas[i]["filepath"],
                })

        all_results.sort(key=lambda r: r["aesthetic_score"], reverse=True)

        self._print_ranking(all_results, args.top_n)
        self._save_visualizations(all_results, output_dir, args.top_n)
        print(f"\nAll outputs saved to: {output_dir}/")

    # -------------------------------------------------------------------------

    def _print_ranking(self, results, top_n):
        n      = len(results)
        scores = [r["aesthetic_score"] for r in results]
        print(f"\n{'='*75}")
        print(f"  AESTHETIC RANKING (meta aggregator) -- Top {min(top_n, n)} of {n}")
        print(f"{'='*75}")
        header = f"  {'Rank':<5} {'Score':<8} {'Class':<22}"
        for name in SUBMODEL_NAMES:
            header += f" {name[:7]:>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rank, r in enumerate(results[:top_n], 1):
            row = f"  {rank:<5} {r['aesthetic_score']:<8.4f} {r['class_name']:<22}"
            for w in r["channel_weights"]:
                row += f" {w:>7.3f}"
            print(row)
        print(f"\n  Mean: {np.mean(scores):.4f}  Max: {np.max(scores):.4f}")

        class_scores: dict = {}
        for r in results:
            class_scores.setdefault(r["class_name"], []).append(r["aesthetic_score"])
        print(f"\n  {'Class':<25} {'Avg':>8}  Bar")
        for cls in sorted(class_scores, key=lambda c: np.mean(class_scores[c]), reverse=True):
            avg = np.mean(class_scores[cls])
            print(f"  {cls:<25} {avg:>8.4f}  |{'#' * int(avg * 40)}")

    def _save_visualizations(self, results, output_dir: Path, top_n: int):
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "aes", ["#0d0d0d", "#8e44ad", "#e74c3c", "#f39c12", "#f9e79f"])

        print(f"\nSaving top-{min(top_n, len(results))} aesthetic visualizations...")
        for rank, r in enumerate(results[:top_n], 1):
            rgb = np.transpose(np.clip(r["rgb"], 0, 1), (1, 2, 0))
            aes = r["aesthetic_map"]

            fig, axes = plt.subplots(2, 4, figsize=(22, 11))
            fig.patch.set_facecolor("#0d0d0d")

            # Row 0: RGB + 3 top submodel maps
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("Satellite RGB", color="white")
            axes[0, 0].axis("off")

            for col, (name, w) in enumerate(
                zip(SUBMODEL_NAMES[:3], r["channel_weights"][:3]), 1
            ):
                axes[0, col].imshow(r["submodel_maps"][col - 1], cmap=cmap, vmin=0, vmax=1)
                axes[0, col].set_title(f"{name}\nw={w:.3f}", color="white", fontsize=9)
                axes[0, col].axis("off")

            # Row 1: 3 remaining submodel maps + aesthetic fusion
            for col, (name, w) in enumerate(
                zip(SUBMODEL_NAMES[3:], r["channel_weights"][3:]), 0
            ):
                axes[1, col].imshow(r["submodel_maps"][col + 3], cmap=cmap, vmin=0, vmax=1)
                axes[1, col].set_title(f"{name}\nw={w:.3f}", color="white", fontsize=9)
                axes[1, col].axis("off")

            # Aesthetic overlay on RGB
            overlay = rgb.copy()
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + aes * 0.5, 0, 1)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] * (1 - aes * 0.3), 0, 1)
            axes[1, 3].imshow(overlay)
            axes[1, 3].set_title(
                f"Aesthetic Fusion\nscore={r['aesthetic_score']:.4f}",
                color="white", fontsize=9)
            axes[1, 3].axis("off")

            for ax in axes.flat:
                ax.set_facecolor("#0d0d0d")

            plt.suptitle(
                f"#{rank}  {r['class_name']}  Aesthetic Score: {r['aesthetic_score']:.4f}",
                fontsize=13, fontweight="bold", color="white")
            plt.tight_layout()

            save_path = output_dir / f"aesthetic_{rank:02d}_{r['class_name']}.png"
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()
