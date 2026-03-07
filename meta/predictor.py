"""
AestheticPredictor: runs all 9 submodels + aggregator in one pipeline.

All 9 submodels are treated as equal members of the aesthetic pipeline.
Each is loaded from its own checkpoint and called as submodel(feature_map).
The elevation model accepts topo=None (auto-zeros), so all 9 share the same
call signature at inference time.

Usage:
    python -m meta predict \\
        --data-dir data \\
        --checkpoint-dir checkpoints \\
        --output-dir output/aesthetic \\
        --top-n 20

Checkpoint layout:
    checkpoints/<subdir>/best_model.pth  for each entry in _SUBMODEL_REGISTRY

Shared decoder:
    All 9 submodels run against one decoder state, loaded from the first
    available checkpoint in registry order. This trades per-submodel
    decoder accuracy for a single-pass pipeline — a deliberate design
    choice since all decoders converge toward the same satellite features.

Housing inversion (urban_openness):
    The housing model predicts building *presence*. The registry marks it
    with invert=True, so the pipeline feeds (1 − housing_pred) to the
    aggregator as an "urban openness" signal. Attention Restoration Theory:
    dense built environments increase cognitive load and reduce aesthetic
    experience. The raw housing_score in VectorDB metadata still reflects
    building presence for standalone use.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import CoreSatelliteModel                       # noqa: E402
from dataset import get_fractal_dataloaders                    # noqa: E402

from submodels.fractal.model       import FractalPatternNet
from submodels.water.model         import WaterGeometryNet
from submodels.color_harmony.model import ColorHarmonyNet
from submodels.symmetry.model      import SymmetryOrderNet
from submodels.sublime.model       import ScaleSublimeNet
from submodels.complexity.model    import ComplexityBalanceNet
from submodels.vegetation.model    import TransUNet
from submodels.elevation.model     import ElevationPOITransUNet
from submodels.housing.model       import HousingEdgeCNN
from .model import AestheticAggregator, SUBMODEL_NAMES

# Single unified registry: (name, class, checkpoint_subdir, invert_for_aggregator)
# All 9 submodels are equals. The decoder is loaded from the first checkpoint
# that contains one (registry order = priority).
# invert=True  →  aggregator receives (1 − pred)  [housing → urban_openness]
# invert=False →  aggregator receives pred directly
_SUBMODEL_REGISTRY = [
    ("fractal",       FractalPatternNet,     "fractal",       False),
    ("water",         WaterGeometryNet,      "water",         False),
    ("color_harmony", ColorHarmonyNet,       "color_harmony", False),
    ("symmetry",      SymmetryOrderNet,      "symmetry",      False),
    ("sublime",       ScaleSublimeNet,       "sublime",       False),
    ("complexity",    ComplexityBalanceNet,  "complexity",    False),
    ("vegetation",    TransUNet,             "vegetation",    False),
    ("elevation",     ElevationPOITransUNet, "elevation",     False),
    ("housing",       HousingEdgeCNN,        "housing",       True),  # → urban_openness
]


def _load_submodel(cls, ckpt_path: Path, core, device):
    """Instantiate, load weights, and return an eval-mode submodel.

    Loads the shared decoder from the checkpoint on the first call that
    provides one. All subsequent calls skip decoder loading (flag on core).
    """
    model = cls(out_channels=1).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["submodel_state_dict"])
        if "decoder_state_dict" in ckpt and not getattr(core, "_decoder_loaded", False):
            core.decoder.load_state_dict(ckpt["decoder_state_dict"])
            core._decoder_loaded = True
        print(f"  Loaded {cls.__name__} from {ckpt_path.name}")
    else:
        print(f"  WARNING: {ckpt_path} not found — using random weights for {cls.__name__}")
    model.eval()
    return model


class AestheticPredictor:
    """Run all 9 submodels + aggregator and produce a ranked aesthetic output."""

    def __init__(self, args):
        self.args = args

    @torch.no_grad()
    def run(self):
        args   = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ckpt_dir   = Path(args.checkpoint_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        core = CoreSatelliteModel().freeze().to(device)
        core._decoder_loaded = False

        # Load all 9 submodels — uniform loop, one decoder loaded from first hit
        print("\nLoading all 9 submodels...")
        submodels = {}
        for name, cls, subdir, _ in _SUBMODEL_REGISTRY:
            submodels[name] = _load_submodel(cls, ckpt_dir / subdir / "best_model.pth",
                                              core, device)

        if not core._decoder_loaded:
            print("  WARNING: no checkpoint had a decoder_state_dict — "
                  "using randomly initialised decoder")

        # Load aggregator (9 inputs)
        aggregator = AestheticAggregator(n_submodels=9).to(device)
        agg_ckpt   = ckpt_dir / "meta" / "best_model.pth"
        if agg_ckpt.exists():
            agg_state = torch.load(agg_ckpt, map_location=device, weights_only=True)
            try:
                aggregator.load_state_dict(agg_state["aggregator_state_dict"])
                print(f"\n  Loaded AestheticAggregator from {agg_ckpt}")
            except RuntimeError:
                print("\n  WARNING: meta checkpoint was trained with fewer submodels — "
                      "retrain: python -m meta train")
                print("  Continuing with equal weights (untrained aggregator).")
        else:
            print("\n  No aggregator checkpoint — using equal weights (untrained)")
        aggregator.eval()
        core.decoder.eval()

        # Fractal dataloader = all EuroSAT tiles, RGB only
        _, _, test_loader = get_fractal_dataloaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
            num_workers=0,
        )

        all_results = []
        print("\nRunning aesthetic pipeline (9 submodels)...")
        for inputs, _targets, metas in tqdm(test_loader, desc="Aesthetic"):
            inputs  = inputs.to(device)
            B       = inputs.size(0)

            features    = core.extract_features(inputs[:, :3])
            feature_map = core.decode(features)

            # All 9 submodels — uniform call, housing inverted to urban_openness
            maps = []
            for name, _, _, invert in _SUBMODEL_REGISTRY:
                pred = submodels[name](feature_map)          # (B, 1, H, W)
                maps.append(1.0 - pred if invert else pred)

            heatmaps  = torch.cat(maps, dim=1)        # (B, 9, H, W)
            aesthetic = aggregator(heatmaps)           # (B, 1, H, W)

            inputs_cpu  = inputs.cpu().numpy()
            aes_cpu     = aesthetic.cpu().numpy()
            maps_cpu    = heatmaps.cpu().numpy()
            weights_cpu = aggregator.channel_weights(heatmaps).cpu().numpy()

            for i in range(B):
                all_results.append({
                    "rgb":             inputs_cpu[i, :3],
                    "aesthetic_map":   aes_cpu[i, 0],
                    "submodel_maps":   maps_cpu[i],       # (9, H, W); housing ch = urban_openness
                    "channel_weights": weights_cpu[i],    # (9,)
                    "aesthetic_score": float(aes_cpu[i, 0].mean()),
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
        print(f"\n{'='*80}")
        print(f"  AESTHETIC RANKING (9 submodels) -- Top {min(top_n, n)} of {n}")
        print(f"{'='*80}")
        header = f"  {'Rank':<5} {'Score':<8} {'Class':<22}"
        for name in SUBMODEL_NAMES:
            header += f" {name[:6]:>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rank, r in enumerate(results[:top_n], 1):
            row = f"  {rank:<5} {r['aesthetic_score']:<8.4f} {r['class_name']:<22}"
            for w in r["channel_weights"]:
                row += f" {w:>6.3f}"
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

            # 3×4 grid: RGB + 9 submodel heatmaps + blank + aesthetic overlay
            fig, axes = plt.subplots(3, 4, figsize=(22, 17))
            fig.patch.set_facecolor("#0d0d0d")

            # Row 0: RGB | fractal | water | color_harmony (indices 0-2)
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("Satellite RGB", color="white")
            axes[0, 0].axis("off")
            for col, idx in enumerate([0, 1, 2], 1):
                axes[0, col].imshow(r["submodel_maps"][idx], cmap=cmap, vmin=0, vmax=1)
                axes[0, col].set_title(
                    f"{SUBMODEL_NAMES[idx]}\nw={r['channel_weights'][idx]:.3f}",
                    color="white", fontsize=9)
                axes[0, col].axis("off")

            # Row 1: symmetry | sublime | complexity | vegetation (indices 3-6)
            for col, idx in enumerate([3, 4, 5, 6]):
                axes[1, col].imshow(r["submodel_maps"][idx], cmap=cmap, vmin=0, vmax=1)
                axes[1, col].set_title(
                    f"{SUBMODEL_NAMES[idx]}\nw={r['channel_weights'][idx]:.3f}",
                    color="white", fontsize=9)
                axes[1, col].axis("off")

            # Row 2: elevation | urban_openness | blank | aesthetic overlay (indices 7-8)
            for col, idx in enumerate([7, 8]):
                axes[2, col].imshow(r["submodel_maps"][idx], cmap=cmap, vmin=0, vmax=1)
                axes[2, col].set_title(
                    f"{SUBMODEL_NAMES[idx]}\nw={r['channel_weights'][idx]:.3f}",
                    color="white", fontsize=9)
                axes[2, col].axis("off")

            axes[2, 2].axis("off")   # blank panel

            overlay = rgb.copy()
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + aes * 0.5, 0, 1)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] * (1 - aes * 0.3), 0, 1)
            axes[2, 3].imshow(overlay)
            axes[2, 3].set_title(
                f"Aesthetic Fusion\nscore={r['aesthetic_score']:.4f}",
                color="white", fontsize=9)
            axes[2, 3].axis("off")

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
