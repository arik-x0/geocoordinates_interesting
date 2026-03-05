"""
Base inference class for all POI submodel pipelines.

Subclasses override:
    get_dataloaders()    -> (_, _, test_loader)
    build_submodel()     -> nn.Module
    score_key            -> str  (result dict key used for sorting)
    rgb_slice()          -> extract RGB from batch  (default: full input)
    extra_slice()        -> extra tensor for submodel (default: None)
    build_result()       -> build per-sample result dict
    print_ranking()      -> display ranked table
    save_visualizations()-> write output PNGs

Usage example (submodels/vegetation.py):

    class VegetationPredictor(BasePredictor):
        score_key = "greenery_score"

        def get_dataloaders(self, args):
            _, _, test_loader = get_vegetation_dataloaders(...)
            return test_loader

        def build_submodel(self):
            return TransUNet(out_channels=1)

        def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu):
            return {
                "rgb":            inputs_cpu[i],
                "mask_true":      targets_cpu[i, 0],
                "mask_pred":      preds_cpu[i, 0],
                "greenery_score": greenery_score_from_prediction(preds_cpu[i]),
                "class_name":     metas[i]["class_name"],
                "filepath":       metas[i]["filepath"],
                "embedding":      embs_cpu[i],
            }

        def print_ranking(self, all_results, top_n): ...
        def save_visualizations(self, all_results, output_dir, top_n): ...
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import CoreSatelliteModel  # noqa: E402
from base.utils import load_index, query_similar  # noqa: E402


class BasePredictor:
    """Shared inference orchestration for all POI submodels."""

    score_key: str = "score"

    def __init__(self, args):
        self.args = args

    # ── Abstract interface ──────────────────────────────────────────────────

    def get_test_loader(self):
        """Return the test DataLoader."""
        raise NotImplementedError

    def build_submodel(self) -> torch.nn.Module:
        """Instantiate and return the task submodel (not yet on device)."""
        raise NotImplementedError

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu) -> dict:
        """Build the per-sample result dict for sample index i.

        Must include at least:
            self.score_key  — float used for ranking
            "embedding"     — (D,) numpy array for VectorDB search
        """
        raise NotImplementedError

    def print_ranking(self, all_results: list, top_n: int):
        """Print the ranked results table to stdout."""
        raise NotImplementedError

    def save_visualizations(self, all_results: list, output_dir: Path, top_n: int):
        """Write per-result and overview PNG files to output_dir."""
        raise NotImplementedError

    # ── Optional overrides ──────────────────────────────────────────────────

    def rgb_slice(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the RGB portion of a batch tensor (default: full input)."""
        return inputs

    def extra_slice(self, inputs: torch.Tensor):
        """Return extra tensor for submodel.forward() (default: None)."""
        return None

    def print_similarity(self, result: dict, rank: int, similar: list):
        """Print VectorDB results for one query.  Override for task-specific columns."""
        print(f"\n  Rank #{rank}  {result['class_name']}  "
              f"score={result[self.score_key]:.4f}  "
              f"({Path(result['filepath']).name})")
        print(f"  {'Sim':>6}  {'Split':<6}  {'Class':<20}  File")
        for s in similar:
            print(f"  {s['similarity']:>6.4f}  {s['split']:<6}  "
                  f"{s['class_name']:<20}  {Path(s['filepath']).name}")

    # ── Core inference loop ─────────────────────────────────────────────────

    @torch.no_grad()
    def run_inference(self, core, submodel, test_loader, device,
                      output_dir: Path, top_n: int,
                      index=None, index_meta=None, top_k_similar: int = 5):
        """Run full inference: forward pass, ranking, VectorDB, visualizations."""
        submodel.eval()
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        print("Running inference on test set...")
        for inputs, targets, metas in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(device)

            rgb_in = self.rgb_slice(inputs)
            features = core.extract_features(rgb_in)

            extra = self.extra_slice(inputs)
            if extra is not None:
                preds = submodel(features, extra)
            else:
                preds = submodel(features)

            embs = F.normalize(features["cls"], p=2, dim=1)

            inputs_cpu  = inputs.cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            preds_cpu   = preds.cpu().numpy()
            embs_cpu    = embs.cpu().numpy()

            for i in range(inputs.size(0)):
                result = self.build_result(i, inputs_cpu, targets_cpu,
                                           preds_cpu, metas, embs_cpu)
                all_results.append(result)

        all_results.sort(key=lambda r: r[self.score_key], reverse=True)

        self.print_ranking(all_results, top_n)

        if index is not None and index_meta is not None:
            show_n = min(5, len(all_results))
            print(f"\n  --- VectorDB: top-{top_k_similar} similar training images "
                  f"for top-{show_n} results ---")
            for rank, r in enumerate(all_results[:show_n], 1):
                similar = query_similar(r["embedding"], index, index_meta, top_k_similar)
                self.print_similarity(r, rank, similar)

        self.save_visualizations(all_results, output_dir, top_n)
        print(f"\nAll outputs saved to: {output_dir}/")
        return all_results

    # ── Main entry point ────────────────────────────────────────────────────

    def run(self):
        args   = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        core     = CoreSatelliteModel().freeze().to(device)
        submodel = self.build_submodel().to(device)

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print("Train the submodel first.")
            return

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        submodel.load_state_dict(ckpt["submodel_state_dict"])
        print(f"Loaded submodel from {checkpoint_path}")
        if "val_iou" in ckpt:
            print(f"  Checkpoint val_iou={ckpt['val_iou']:.4f} (epoch {ckpt['epoch']})")

        checkpoint_dir = checkpoint_path.parent
        index, index_meta = load_index(checkpoint_dir)

        test_loader = self.get_test_loader()

        self.run_inference(
            core=core,
            submodel=submodel,
            test_loader=test_loader,
            device=device,
            output_dir=Path(args.output_dir),
            top_n=args.top_n,
            index=index,
            index_meta=index_meta,
            top_k_similar=args.top_k_similar,
        )
