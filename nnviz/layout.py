"""Lay the connectome out in 3-D.

The layout is driven only by the wiring: each neuron's neighbour list from
``connectome.py`` is handed to UMAP as a precomputed k-NN graph (distance =
1 - |cos|), so UMAP never sees the raw vectors -- strongly connected neurons are
pulled together, unconnected ones are pushed apart. Dense regions in the result
are wired-up modules; sparse regions are neurons whose connections are weak or
scattered.

Cached on disk, because it depends only on the weights: every later scan of the
same model reuses the same anatomy, which is what lets you overlay different
prompts on one fixed structure and compare them voxel for voxel.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

__all__ = ["layout_connectome", "orient"]


def orient(xyz: np.ndarray, layer: np.ndarray | None = None) -> np.ndarray:
    """Canonical pose: centred, principal axes (longest = x), later layers toward +."""
    p = xyz.astype(np.float64) - np.median(xyz, axis=0)
    _, vecs = np.linalg.eigh(p.T @ p / len(p))
    p = p @ vecs[:, ::-1]
    lc = None if layer is None else layer.astype(np.float64) - layer.mean()
    for a in range(3):
        ref = np.dot(p[:, a], lc) if lc is not None else np.mean(p[:, a] ** 3)
        if ref < 0:
            p[:, a] *= -1
    return p.astype(np.float32)


def layout_connectome(conn, *, n_neighbors: int | None = None, min_dist: float = 0.1,
                      spread: float = 1.0, n_epochs: int | None = None, seed: int = 0,
                      cache_dir: str | Path | None = None, cache_key: str = "",
                      verbose: bool = True) -> np.ndarray:
    """``Connectome`` -> ``(N, 3)`` neuron positions (oriented, cached)."""
    N = len(conn)
    knn = conn.nbr.shape[1] + 1
    key = hashlib.sha1(json.dumps({"k": cache_key, "n": N, "knn": knn, "md": min_dist,
                                   "sp": spread, "e": n_epochs, "s": seed},
                                  sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"layout_{key}.npy" if cache_dir else None
    if path is not None and path.exists():
        if verbose:
            print(f"[nnviz] layout: cached ({path.name})", flush=True)
        return np.load(path)

    import umap

    # precomputed k-NN: self first (distance 0), then partners by strength
    idx = np.concatenate([np.arange(N)[:, None], conn.nbr], axis=1)
    dist = np.concatenate([np.zeros((N, 1), np.float32), 1.0 - conn.strength], axis=1)
    dist = np.where(idx < 0, np.inf, dist).astype(np.float32)
    idx = np.where(idx < 0, -1, idx).astype(np.int32)
    if verbose:
        print(f"[nnviz] laying out {N:,} neurons from their wiring (UMAP, {knn - 1} partners "
              "each) -- a minute or two", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors or knn, min_dist=min_dist,
                            spread=spread, random_state=seed, n_epochs=n_epochs,
                            precomputed_knn=(idx, dist), metric="euclidean", verbose=False)
        # UMAP only uses X for its shape when a k-NN graph is supplied
        xyz = reducer.fit_transform(np.zeros((N, 1), np.float32))
    xyz = orient(np.asarray(xyz, np.float32), conn.layer)
    if verbose:
        print(f"[nnviz] layout done in {time.time() - t0:.0f}s", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, xyz)
    return xyz
