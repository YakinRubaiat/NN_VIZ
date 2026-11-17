"""Splat a 3-D point cloud into a voxel grid -- the 'scanner' step.

Each unit of the network is a point with attributes (layer, kind, weight norm,
activation...). A voxel's value is an aggregate over the units that fall in it,
smoothed with a small Gaussian point-spread function. That PSF is the same thing
a real scanner has: it trades a little resolution for a continuous image in
which tissue, not individual points, is what you see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["Grid", "fit_grid", "splat", "smooth"]


@dataclass
class Grid:
    """Maps embedding coordinates to voxel coordinates (isotropic voxels)."""

    origin: np.ndarray   # (3,) embedding-space coordinate of voxel (0,0,0)
    scale: float         # voxels per embedding unit
    shape: tuple[int, int, int]

    def to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float64) - self.origin) * self.scale

    def to_dict(self) -> dict:
        return {"origin": self.origin.tolist(), "scale": float(self.scale),
                "shape": list(self.shape)}


def fit_grid(xyz: np.ndarray, max_dim: int = 128, margin: int = 6,
             clip: float = 0.25) -> Grid:
    """Fit an isotropic grid around ``xyz`` whose longest side is ``max_dim``.

    ``clip`` (percent) trims stray outlier points from the bounding box so a
    handful of strays do not shrink the whole object to a dot.
    """
    lo = np.percentile(xyz, clip, axis=0)
    hi = np.percentile(xyz, 100 - clip, axis=0)
    ext = np.maximum(hi - lo, 1e-9)
    scale = (max_dim - 1 - 2 * margin) / ext.max()
    shape = tuple(int(np.ceil(e * scale)) + 1 + 2 * margin for e in ext)
    origin = lo - margin / scale
    return Grid(origin=origin, scale=float(scale), shape=shape)  # type: ignore[arg-type]


def splat(grid: Grid, xyz: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Trilinear splat of (weighted) points into ``grid.shape``."""
    v = grid.to_voxel(xyz)
    w = np.ones(len(v), dtype=np.float64) if weights is None else np.asarray(weights, np.float64)
    sx, sy, sz = grid.shape
    total = sx * sy * sz
    out = np.zeros(total, dtype=np.float64)
    i0 = np.floor(v).astype(np.int64)
    f = v - i0
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                ok = (ix >= 0) & (ix < sx) & (iy >= 0) & (iy < sy) & (iz >= 0) & (iz < sz)
                flat = (ix[ok] * sy + iy[ok]) * sz + iz[ok]
                out += np.bincount(flat, weights=(w * wx * wy * wz)[ok], minlength=total)
    out = out.reshape(grid.shape)
    return out


def smooth(vol: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Gaussian point-spread function, in voxels."""
    if sigma <= 0:
        return vol.astype(np.float32)
    return gaussian_filter(vol.astype(np.float64), sigma=sigma, mode="constant").astype(np.float32)
