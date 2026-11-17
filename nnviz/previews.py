"""PNG sheets of a saved scan, for looking without a browser.

Brain view:  maximum-intensity projections and centre cuts from three sides,
             for the structure and for each paint map, plus a filmstrip of the
             dynamic scan (one MIP per token).
Stack view:  one tile per layer for each volume.
Network:     every layer's weight panels flattened side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .colormaps import lut
from .volio import read_volume

__all__ = ["render_previews", "sheet"]


def _norm(a: np.ndarray, lo=None, hi=None) -> np.ndarray:
    v = a[np.isfinite(a)]
    lo = np.percentile(v, 1) if lo is None else lo
    hi = np.percentile(v, 99.7) if hi is None else hi
    return np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)


def _rgb(n: np.ndarray, cmap: str) -> np.ndarray:
    return lut(cmap)[(np.clip(n, 0, 1) * 255 + 0.5).astype(np.uint8)]


def sheet(tiles: list[np.ndarray], cols: int, scale: int = 3, pad: int = 3,
          labels: list[str] | None = None) -> np.ndarray:
    """Tile RGB images (any sizes) into one image, each centred in its cell."""
    th = max(t.shape[0] for t in tiles) * scale
    tw = max(t.shape[1] for t in tiles) * scale
    rows = int(np.ceil(len(tiles) / cols))
    out = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), 10, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        big = np.repeat(np.repeat(t, scale, 0), scale, 1)
        y = pad + r * (th + pad) + (th - big.shape[0]) // 2
        x = pad + c * (tw + pad) + (tw - big.shape[1]) // 2
        out[y:y + big.shape[0], x:x + big.shape[1]] = big
    return out


def _painted_mip(den: np.ndarray, paint: np.ndarray, axis: int, cmap: str,
                 plo: float, phi: float) -> np.ndarray:
    """Colour = paint at the densest voxel along each ray; brightness = density."""
    idx = den.argmax(axis=axis)
    p = np.take_along_axis(paint, np.expand_dims(idx, axis), axis).squeeze(axis)
    b = _norm(den.max(axis=axis))
    return (_rgb(_norm(p, plo, phi), cmap) * b[..., None]).astype(np.uint8)


def _save(img: np.ndarray, path: Path) -> Path:
    from PIL import Image
    Image.fromarray(img).save(path)
    return path


def _arch_sheet(vdir: Path, man: dict, cmap: str = "magma") -> np.ndarray:
    """Every layer's weight panels flattened side by side (attention | MLP)."""
    w = np.fromfile(vdir / man["weights"]["file"], np.uint8)
    panels = man["weights"]["panels"]

    def tex(p):
        r, c = p["tex_shape"]
        return w[p["offset"]:p["offset"] + r * c].reshape(r, c)

    cols = []
    for layer in range(man["n_layers"]):
        stacks = []
        for fam in ("attn", "mlp"):
            parts = [tex(p) for p in panels if p["layer"] == layer and p["family"] == fam]
            gap = np.zeros((3, parts[0].shape[1]), np.uint8)
            stacks.append(np.concatenate(sum([[x, gap] for x in parts], [])[:-1]))
        h = max(x.shape[0] for x in stacks)
        stacks = [np.pad(x, ((0, h - x.shape[0]), (0, 0))) for x in stacks]
        cols += [stacks[0], np.zeros((h, 2), np.uint8), stacks[1], np.zeros((h, 8), np.uint8)]
    h = max(c.shape[0] for c in cols)
    img = np.concatenate([np.pad(c, ((0, h - c.shape[0]), (0, 0))) for c in cols], axis=1)
    return lut(cmap)[img]


def render_previews(scan_dir, out_dir=None, *, scale: int = 3) -> list[Path]:
    root = Path(scan_dir)
    out = Path(out_dir) if out_dir else Path("out") / root.name
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "index.json").read_text())
    written: list[Path] = []

    for view in index["views"]:
        man_path = root / view["manifest"]
        vdir = man_path.parent
        man = json.loads(man_path.read_text())
        vols = {v["name"]: v for v in man["volumes"]}

        if man["kind"] == "arch":
            written.append(_save(_arch_sheet(vdir, man), out / "network_weights.png"))
            continue
        if man["kind"] == "brain":
            den = read_volume(vdir, vols["density"])
            # structure: MIP + centre cut, three sides
            tiles = [_rgb(_norm(den.max(axis=a).T), "bone") for a in range(3)]
            tiles += [_rgb(_norm(np.take(den, den.shape[a] // 2, axis=a).T), "bone")
                      for a in range(3)]
            written.append(_save(sheet(tiles, 3, scale), out / "brain_structure.png"))
            # each paint map, as colour on the structure
            for name, v in vols.items():
                if v["role"] != "paint":
                    continue
                paint = read_volume(vdir, v)
                lo, hi = v["transform"]["lo"], v["transform"]["hi"]
                tiles = [_painted_mip(den, paint, a, v["cmap"], lo, hi).transpose(1, 0, 2)
                         for a in range(3)]
                written.append(_save(sheet(tiles, 3, scale), out / f"brain_{name}.png"))
            # dynamic scan filmstrip
            fr = man.get("frames")
            if fr:
                T, fx, fy, fz = fr["shape"]
                buf = np.fromfile(vdir / fr["file"], np.uint8).reshape(T, fz, fy, fx)
                pick = np.unique(np.linspace(0, T - 1, min(T, 24)).round().astype(int))
                proj = buf.astype(np.float32).mean(axis=1)            # (T, y, x) mean along z
                hi = np.percentile(proj, 99.8) or 1.0
                tiles = [_rgb(proj[t] / hi, "hot") for t in pick]
                written.append(_save(sheet(tiles, 6, scale), out / "brain_dynamic.png"))
        else:
            for name, v in vols.items():
                d = read_volume(vdir, v)
                nl = v.get("n_layers", d.shape[2])
                zs = np.unique(np.linspace(0, d.shape[2] - 1, min(nl, 32)).round().astype(int))
                n = _norm(d)
                tiles = [_rgb(n[:, :, z].T, v["cmap"]) for z in zs]
                written.append(_save(sheet(tiles, 7, scale + 1), out / f"stack_{name}.png"))
    return written
