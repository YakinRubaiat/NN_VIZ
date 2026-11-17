"""One on-disk volume format shared by every view.

Each volume is a raw ``uint8`` file in ``[z][y][x]`` order (x fastest), which is
the byte order WebGL ``texImage3D`` wants, plus a manifest entry describing how
the 8 bits map back to physical values.

Three intensity transforms, chosen per volume:

* ``linear`` -- plain window, for well-behaved distributions.
* ``asinh``  -- linear near zero, logarithmic in the tail. Transformer
  activations are extremely heavy-tailed (a few "massive activation" channels
  run 50-100x the median); a linear window spends all 256 levels on those few
  voxels and paints everything else black. asinh is the standard fix in
  astronomy for the same problem, and unlike log it is defined at zero.
* ``signed`` -- symmetric window around zero, stored with 128 = 0. For
  contrast maps, where the sign is the finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["quantize", "dequantize", "write_volume", "read_volume", "volume_stats"]


def volume_stats(data: np.ndarray, mask: np.ndarray | None = None) -> dict:
    sel = data[mask] if mask is not None else data.reshape(-1)
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return {"min": 0.0, "max": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0, "mean": 0.0}
    p = np.percentile(sel, [1, 50, 99, 99.9])
    return {"min": float(sel.min()), "max": float(sel.max()), "p01": float(p[0]),
            "p50": float(p[1]), "p99": float(p[2]), "p999": float(p[3]),
            "mean": float(sel.mean()), "std": float(sel.std())}


def quantize(data: np.ndarray, *, transform: str = "auto", mask: np.ndarray | None = None,
             window: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    """``float`` volume -> (``uint8`` volume, transform record).

    ``window`` forces a fixed linear window, for label and fraction maps whose
    range is known in advance (layer 0..L-1, fraction 0..1).
    """
    d = np.asarray(data, dtype=np.float64)
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
        q = np.clip((d - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        return (q * 255.0 + 0.5).astype(np.uint8), {"kind": "linear", "lo": lo, "hi": hi}
    sel = d[mask] if mask is not None else d.reshape(-1)
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        sel = np.zeros(1)

    if transform == "signed":
        m = float(np.percentile(np.abs(sel), 99.5)) or 1e-6
        q = 0.5 + 0.5 * np.clip(d / m, -1, 1)
        rec = {"kind": "signed", "lo": -m, "hi": m}
    else:
        pos = sel[sel > 0]
        p50 = float(np.percentile(pos, 50)) if pos.size else 1.0
        p99 = float(np.percentile(pos, 99)) if pos.size else 1.0
        if transform == "auto":
            transform = "asinh" if (p50 > 0 and p99 / p50 > 6.0) else "linear"
        lo = float(np.percentile(sel, 1.0))
        if transform == "asinh":
            lo = max(lo, 0.0)
            hi = float(sel.max())
            s = max(p50, 1e-12)
            t = np.arcsinh(d / s)
            tlo, thi = np.arcsinh(lo / s), np.arcsinh(hi / s)
            q = (t - tlo) / max(thi - tlo, 1e-12)
            rec = {"kind": "asinh", "lo": lo, "hi": hi, "scale": s}
        else:
            hi = float(np.percentile(sel, 99.9))
            if hi <= lo:
                hi = lo + 1e-6
            q = (d - lo) / (hi - lo)
            rec = {"kind": "linear", "lo": lo, "hi": hi}

    q = np.clip(np.nan_to_num(q, nan=0.0), 0.0, 1.0)
    if mask is not None:
        q = np.where(mask, q, 0.5 if rec["kind"] == "signed" else 0.0)
    return (q * 255.0 + 0.5).astype(np.uint8), rec


def dequantize(q: np.ndarray, rec: dict) -> np.ndarray:
    """Inverse of :func:`quantize` (exact up to the 8-bit step)."""
    u = q.astype(np.float64) / 255.0
    k = rec["kind"]
    if k == "signed":
        return ((u - 0.5) * 2.0 * rec["hi"]).astype(np.float32)
    if k == "asinh":
        s = rec["scale"]
        tlo, thi = np.arcsinh(rec["lo"] / s), np.arcsinh(rec["hi"] / s)
        return (s * np.sinh(tlo + u * (thi - tlo))).astype(np.float32)
    return (rec["lo"] + u * (rec["hi"] - rec["lo"])).astype(np.float32)


def write_volume(out_dir: Path, name: str, data: np.ndarray, *, label: str, description: str,
                 role: str = "paint", cmap: str = "gray", group: str = "",
                 transform: str = "auto", mask: np.ndarray | None = None,
                 window: tuple[float, float] | None = None,
                 extra: dict | None = None) -> dict:
    """Write ``vol_<name>.bin`` and return its manifest entry.

    ``data`` is ``(x, y, z)``. ``role`` tells the viewer how to use it:
    ``shape`` (defines the solid object / opacity), ``paint`` (colours the object),
    ``overlay`` (unsigned functional map drawn on top), ``signed`` (contrast map
    drawn on top with a diverging colour map).
    """
    out_dir = Path(out_dir)
    q, rec = quantize(data, transform=transform, mask=mask, window=window)
    fname = f"vol_{name}.bin"
    np.ascontiguousarray(np.transpose(q, (2, 1, 0))).tofile(out_dir / fname)
    entry = {
        "name": name, "label": label, "description": description, "group": group,
        "role": role, "cmap": cmap, "shape": [int(s) for s in data.shape],
        "file": fname, "dtype": "uint8", "transform": rec,
        "stats": volume_stats(np.asarray(data, np.float64), mask),
    }
    if extra:
        entry.update(extra)
    return entry


def read_volume(root: Path, entry: dict) -> np.ndarray:
    """Read a volume back as ``(x, y, z)`` float32 in physical units."""
    x, y, z = entry["shape"]
    buf = np.fromfile(Path(root) / entry["file"], dtype=np.uint8)
    if buf.size != x * y * z:
        raise ValueError(f"{entry['file']}: expected {x*y*z} bytes, found {buf.size}")
    q = np.transpose(buf.reshape(z, y, x), (2, 1, 0))
    return dequantize(q, entry["transform"])
