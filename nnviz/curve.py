"""Locality-preserving fold from 1-D to 2-D, for the layer-stack view.

A channel index is arbitrary (permute the hidden dimension and the network is
unchanged), so channels are first seriated (``ordering.py``) and then laid onto
a plane along a space-filling curve, which keeps 1-D neighbours close in 2-D.

This is the *generalised* Hilbert curve (Gilbert), defined on any rectangle.
Hidden sizes are often powers of two but intermediate sizes are not (3072,
4864, 8960...); forcing those into a power-of-two square leaves a quarter of
each slice as dead padding, which looks like structure and is not. With Gilbert
a 3072-wide space is a clean 64x48 plane.
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["gilbert2d", "curve_coords", "best_grid", "fold_to_plane"]


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _gen(x, y, ax, ay, bx, by, out):
    w, h = abs(ax + ay), abs(bx + by)
    dax, day, dbx, dby = _sgn(ax), _sgn(ay), _sgn(bx), _sgn(by)
    if h == 1:
        for _ in range(w):
            out.append((x, y)); x += dax; y += day  # noqa: E702
        return
    if w == 1:
        for _ in range(h):
            out.append((x, y)); x += dbx; y += dby  # noqa: E702
        return
    ax2, ay2, bx2, by2 = ax // 2, ay // 2, bx // 2, by // 2
    w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)
    if 2 * w > 3 * h:
        if w2 % 2 and w > 2:
            ax2 += dax; ay2 += day  # noqa: E702
        _gen(x, y, ax2, ay2, bx, by, out)
        _gen(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, out)
    else:
        if h2 % 2 and h > 2:
            bx2 += dbx; by2 += dby  # noqa: E702
        _gen(x, y, bx2, by2, ax2, ay2, out)
        _gen(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, out)
        _gen(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
             -bx2, -by2, -(ax - ax2), -(ay - ay2), out)


def gilbert2d(width: int, height: int) -> list[tuple[int, int]]:
    """Every cell of a ``width x height`` grid in curve order (unit steps)."""
    out: list[tuple[int, int]] = []
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    if width >= height:
        _gen(0, 0, width, 0, 0, height, out)
    else:
        _gen(0, 0, 0, height, width, 0, out)
    return out


def curve_coords(width: int, height: int) -> np.ndarray:
    return np.asarray(gilbert2d(width, height), dtype=np.int64)


def best_grid(n: int, max_aspect: float = 1.75) -> tuple[int, int, int]:
    """Near-square ``(width, height, padding)`` for ``n`` channels, preferring no padding."""
    def exact(m: int):
        for h in range(int(np.sqrt(m)), 0, -1):
            if m % h == 0 and (m // h) / h <= max_aspect:
                return m // h, h
        return None

    for extra in range(0, max(16, n // 16) + 1):
        fit = exact(n + extra)
        if fit:
            return fit[0], fit[1], extra
    side = int(np.ceil(np.sqrt(n)))
    return side, side, side * side - n


def fold_to_plane(values: np.ndarray, width: int, height: int) -> np.ndarray:
    """``(..., n)`` -> ``(..., width, height)`` along the curve (tail = 0)."""
    n = values.shape[-1]
    coords = curve_coords(width, height)[:n]
    out = np.zeros(values.shape[:-1] + (width, height), dtype=np.float32)
    out[..., coords[:, 0], coords[:, 1]] = values.astype(np.float32, copy=False)
    return out
