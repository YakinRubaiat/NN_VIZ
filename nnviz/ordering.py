"""Give the channel axis an order in which neighbours behave alike.

Each channel is described by its profile across layers; channels with similar
profiles are placed next to each other by spectral seriation (Fiedler vector of
a k-NN similarity graph). Deterministic for a given model and probe, so two
scans are voxel-comparable when they share a layout.
"""

from __future__ import annotations

import numpy as np

__all__ = ["profile_from_stack", "spectral_order"]


def profile_from_stack(stack: np.ndarray) -> np.ndarray:
    """``(M, L, C)`` measurement maps -> ``(C, M*L)`` z-scored profiles."""
    s = np.asarray(stack, dtype=np.float64)
    if s.ndim == 2:
        s = s[None]
    m, lyr, c = s.shape
    x = s.reshape(m * lyr, c)
    x = x - x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    x = np.divide(x, sd, out=np.zeros_like(x), where=sd > 1e-12)
    return np.ascontiguousarray(x.T)


def spectral_order(profile: np.ndarray, knn: int = 24) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64)
    c = p.shape[0]
    if c < 4:
        return np.arange(c)
    p = p - p.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(p, axis=1, keepdims=True)
    p = np.divide(p, nrm, out=np.zeros_like(p), where=nrm > 1e-12)
    sim = p @ p.T
    np.fill_diagonal(sim, 0.0)
    k = int(min(max(knn, 2), c - 1))
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(c), k)
    w = np.zeros_like(sim)
    w[rows, idx.ravel()] = np.maximum(sim[rows, idx.ravel()], 0.0)
    w = np.maximum(w, w.T)
    deg = w.sum(axis=1)
    if not np.any(deg > 0):
        return np.arange(c)
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    lap = np.eye(c) - dinv[:, None] * w * dinv[None, :]
    vals, vecs = np.linalg.eigh(lap)
    good = np.where(vals > 1e-9)[0]
    if good.size == 0:
        return np.arange(c)
    return np.argsort(vecs[:, good[0]] * dinv, kind="stable").astype(np.int64)
