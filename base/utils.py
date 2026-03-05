"""
Shared utilities for all POI submodel pipelines.

Provides FAISS VectorDB helpers used by every predict.py entry point.
"""

import json
from pathlib import Path

import numpy as np


def load_index(checkpoint_dir: Path):
    """Load the FAISS index and metadata written by train.py.

    Returns:
        (index, meta) if the index exists, else (None, None).
    """
    try:
        import faiss
    except ImportError:
        print("WARNING: faiss-cpu not installed — similarity search disabled.")
        return None, None

    index_path = checkpoint_dir / "embedding_index.faiss"
    meta_path  = checkpoint_dir / "embedding_meta.json"

    if not index_path.exists() or not meta_path.exists():
        print(f"NOTE: No VectorDB index found at {index_path}")
        print("      Run train first to build it.")
        return None, None

    import faiss as _faiss
    index = _faiss.read_index(str(index_path))
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loaded VectorDB index: {index.ntotal} vectors (dim={index.d})")
    return index, meta


def query_similar(embedding: np.ndarray, index, meta: list, top_k: int = 5) -> list:
    """Return the top_k most cosine-similar entries from the VectorDB.

    Args:
        embedding: (D,) L2-normalised float32 query vector.
        index:     FAISS IndexFlatIP (inner product = cosine on unit vectors).
        meta:      metadata list aligned with index rows.
        top_k:     number of neighbours to retrieve.

    Returns:
        list of dicts with stored metadata fields plus 'similarity'.
    """
    scores, indices = index.search(
        embedding.reshape(1, -1).astype(np.float32), top_k
    )
    return [
        {**meta[idx], "similarity": float(score)}
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]
