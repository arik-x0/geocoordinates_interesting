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

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list:
        """Return the top_k most cosine-similar entries.

        Args:
            embedding: (D,) L2-normalised float32 query vector.
            top_k:     number of neighbours to retrieve.

        Returns:
            list of dicts with stored metadata fields plus 'similarity' (float).
        """
        scores, indices = self._index.search(
            embedding.reshape(1, -1).astype(np.float32), top_k
        )
        return [
            {**self._meta[idx], "similarity": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

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
