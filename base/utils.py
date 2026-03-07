"""
Shared utilities for all POI submodel pipelines.

VectorDB: thin wrapper around a FAISS IndexFlatIP for cosine similarity search
over L2-normalised CLS embeddings built during training.
"""

import json
from pathlib import Path

import numpy as np


class VectorDB:
    """FAISS-backed cosine similarity index for satellite tile retrieval.

    Built at the end of training by training_utils.build_embedding_index().
    Loaded at inference time via VectorDB.load(checkpoint_dir).

    Usage:
        vdb = VectorDB.load(Path("checkpoints/vegetation"))
        if vdb is not None:
            similar = vdb.query(embedding, top_k=5)
    """

    def __init__(self, index, meta: list):
        self._index = index
        self._meta  = meta

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def dim(self) -> int:
        return self._index.d

    def query(self, embedding: np.ndarray, top_k: int = 5,
              filter_fn=None) -> list:
        """Return the top_k most cosine-similar entries.

        Args:
            embedding:  (D,) L2-normalised float32 query vector.
            top_k:      number of results to return.
            filter_fn:  optional callable(meta_dict) -> bool.  When provided,
                        the index is over-retrieved (top_k * 10 candidates) and
                        filtered before returning top_k results.  Useful for
                        score-based filtering on the shared index, e.g.:
                            filter_fn=lambda m: m["water_score"] > 0.7

        Returns:
            list of dicts with stored metadata fields plus 'similarity' (float).
        """
        fetch_k = top_k * 10 if filter_fn is not None else top_k
        scores, indices = self._index.search(
            embedding.reshape(1, -1).astype(np.float32), fetch_k
        )
        results = [
            {**self._meta[idx], "similarity": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
        if filter_fn is not None:
            results = [r for r in results if filter_fn(r)]
        return results[:top_k]

    @classmethod
    def load_shared(cls, shared_dir: Path) -> "VectorDB | None":
        """Load the unified shared VectorDB built by vectordb_shared_index.py.

        The shared index contains every EuroSAT tile scored by all 9 submodels
        + the meta aggregator — one index for the whole project.

        Args:
            shared_dir: directory containing shared_index.faiss and
                        shared_index_meta.json (default: checkpoints/shared/).

        Returns:
            VectorDB instance, or None if faiss is not installed / files missing.

        Example queries on the shared index::

            vdb = VectorDB.load_shared(Path("checkpoints/shared"))

            # Visually similar tiles
            vdb.query(emb, top_k=10)

            # Visually similar tiles that are also highly aesthetic
            vdb.query(emb, top_k=10,
                      filter_fn=lambda m: m["aesthetic_score"] > 0.7)

            # Similar water scenes regardless of other aesthetics
            vdb.query(emb, top_k=10,
                      filter_fn=lambda m: m["water_score"] > 0.65)

            # Similar landscapes that score high on fractal AND color
            vdb.query(emb, top_k=10,
                      filter_fn=lambda m: m["fractal_score"] > 0.5
                                      and m["color_score"] > 0.5)
        """
        try:
            import faiss
        except ImportError:
            print("WARNING: faiss-cpu not installed — similarity search disabled.")
            return None

        index_path = shared_dir / "shared_index.faiss"
        meta_path  = shared_dir / "shared_index_meta.json"

        if not index_path.exists() or not meta_path.exists():
            print(f"NOTE: No shared VectorDB found in {shared_dir}")
            print("      Run vectordb_shared_index.py to build it.")
            return None

        index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            meta = json.load(f)

        print(f"Loaded shared VectorDB: {index.ntotal} vectors (dim={index.d})")
        return cls(index, meta)

    @classmethod
    def load(cls, checkpoint_dir: Path) -> "VectorDB | None":
        """Load a VectorDB from a checkpoint directory.

        Returns None if faiss is not installed or the index files are missing.
        """
        try:
            import faiss
        except ImportError:
            print("WARNING: faiss-cpu not installed — similarity search disabled.")
            return None

        index_path = checkpoint_dir / "embedding_index.faiss"
        meta_path  = checkpoint_dir / "embedding_meta.json"

        if not index_path.exists() or not meta_path.exists():
            print(f"NOTE: No VectorDB index found in {checkpoint_dir}")
            print("      Run train first to build it.")
            return None

        index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            meta = json.load(f)

        print(f"Loaded VectorDB: {index.ntotal} vectors (dim={index.d})")
        return cls(index, meta)
