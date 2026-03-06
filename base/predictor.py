"""
Base inference class for all POI submodel pipelines.

Forward pass mirrors training:
    features    = core.extract_features(rgb)   # frozen backbone
    feature_map = core.decode(features)         # trained shared decoder
    preds       = submodel(feature_map, [topo]) # task head

The decoder state is loaded from the checkpoint alongside the submodel.

Subclasses override:
    get_test_loader()    -> DataLoader
    build_submodel()     -> nn.Module  (task head)
    score_key            -> str  (result dict key used for ranking)
    rgb_slice()          -> extract RGB from batch         (default: full input)
    extra_slice()        -> extra tensor for submodel head (default: None)
    build_result()       -> per-sample result dict
    print_ranking()      -> display ranked table
    save_visualizations()-> write output PNGs
    print_similarity()   -> VectorDB neighbour display     (default: generic)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import CoreSatelliteModel  # noqa: E402
from base.utils import VectorDB            # noqa: E402


class BasePredictor:
    """Shared inference orchestration for all POI submodels."""

    score_key: str = "score"

    def __init__(self, args):
        self.args = args

    # -- Abstract interface --------------------------------------------------

    def get_test_loader(self):
        """Return the test DataLoader."""
        raise NotImplementedError

    def build_submodel(self) -> torch.nn.Module:
        """Instantiate and return the task head (not yet on device)."""
        raise NotImplementedError

    def build_result(self, i, inputs_cpu, targets_cpu, preds_cpu, metas, embs_cpu) -> dict:
        """Build the per-sample result dict for sample index i.

        Must include at least:
            self.score_key  -- float used for ranking
            "embedding"     -- (D,) numpy array for VectorDB search
        """
        raise NotImplementedError

    def print_ranking(self, all_results: list, top_n: int):
        """Print the ranked results table to stdout."""
        raise NotImplementedError

    def save_visualizations(self, all_results: list, output_dir: Path, top_n: int):
        """Write per-result and overview PNG files to output_dir."""
        raise NotImplementedError

    # -- Optional overrides --------------------------------------------------

    def rgb_slice(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the RGB portion of a batch tensor (default: full input)."""
        return inputs

    def extra_slice(self, inputs: torch.Tensor):
        """Return extra tensor for submodel.forward() (default: None)."""
        return None

    def print_similarity(self, result: dict, rank: int, similar: list):
        """Print VectorDB neighbours for one query. Override for task columns."""
        print(f"\n  Rank #{rank}  {result['class_name']}  "
              f"score={result[self.score_key]:.4f}  "
              f"({Path(result['filepath']).name})")
        print(f"  {'Sim':>6}  {'Split':<6}  {'Class':<20}  File")
        for s in similar:
            print(f"  {s['similarity']:>6.4f}  {s['split']:<6}  "
                  f"{s['class_name']:<20}  {Path(s['filepath']).name}")

    # -- Core inference loop -------------------------------------------------

    @torch.no_grad()
    def run_inference(self, core, submodel, test_loader, device,
                      output_dir: Path, top_n: int,
                      vdb: "VectorDB | None" = None,
                      top_k_similar: int = 5):
        """Run full inference: forward pass, ranking, VectorDB search, visualizations."""
        core.decoder.eval()
        submodel.eval()
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        print("Running inference on test set...")
        for inputs, targets, metas in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(device)

            features    = core.extract_features(self.rgb_slice(inputs))
            feature_map = core.decode(features)

            extra = self.extra_slice(inputs)
            if extra is not None:
                preds = submodel(feature_map, extra)
            else:
                preds = submodel(feature_map)

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

        if vdb is not None:
            show_n = min(5, len(all_results))
            print(f"\n  --- VectorDB: top-{top_k_similar} similar training images "
                  f"for top-{show_n} results ---")
            for rank, r in enumerate(all_results[:show_n], 1):
                similar = vdb.query(r["embedding"], top_k=top_k_similar)
                self.print_similarity(r, rank, similar)

        self.save_visualizations(all_results, output_dir, top_n)
        print(f"\nAll outputs saved to: {output_dir}/")
        return all_results

    # -- Main entry point ----------------------------------------------------

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

        if "decoder_state_dict" in ckpt:
            core.decoder.load_state_dict(ckpt["decoder_state_dict"])
            print("Loaded shared decoder from checkpoint.")
        else:
            print("WARNING: checkpoint has no decoder_state_dict -- "
                  "using randomly initialised decoder.")

        submodel.load_state_dict(ckpt["submodel_state_dict"])
        print(f"Loaded submodel from {checkpoint_path}")
        if "val_iou" in ckpt:
            print(f"  val_iou={ckpt['val_iou']:.4f} (epoch {ckpt['epoch']})")

        checkpoint_dir = checkpoint_path.parent
        vdb = VectorDB.load(checkpoint_dir)

        test_loader = self.get_test_loader()

        self.run_inference(
            core=core,
            submodel=submodel,
            test_loader=test_loader,
            device=device,
            output_dir=Path(args.output_dir),
            top_n=args.top_n,
            vdb=vdb,
            top_k_similar=args.top_k_similar,
        )
