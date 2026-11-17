"""The brain view: a fixed structure from the wiring, with activity playing on it.

Structure (prompt-independent, from the weights)
    neurons + synapses          connectome.py
    3-D positions from wiring   layout.py
    voxelised with a PSF        voxelize.py

Function (from running the model)
    each neuron's activity at each token, normalised to its own peak:
        a_t(n) = |x_t(n)| / max over all probe tokens |x(n)|      in [0, 1]
    0 = silent, 1 = firing as hard as it ever does. Most neurons sit near 0
    most of the time; the scan is about where the few ~1 values appear, and how
    that moves from token to token.

Written per scan
    volumes   density (shape) + paint maps + function overlays   vol_*.bin
    neurons   positions + attributes, for the point (calcium-imaging) view
    frames    per-token activity: a volume per token, and per-neuron values
    tracts    the strongest synapses, drawn as fibres
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .activity import Activity
from .connectome import FAMILIES, Connectome
from .volio import write_volume
from .voxelize import Grid, fit_grid, smooth, splat

__all__ = ["neuron_values", "build_brain"]


def neuron_values(conn: Connectome, mix: np.ndarray, mlp: np.ndarray) -> np.ndarray:
    """Per-layer maps -> one value per neuron.

    ``mix`` is ``(L, n_heads*head_dim)`` (attention head outputs before o_proj),
    ``mlp`` is ``(L, intermediate)`` (MLP activations before down_proj). Neuron
    ids in the connectome index straight into them.
    """
    out = np.empty(len(conn), dtype=np.float32)
    a = conn.family == 0
    out[a] = mix[conn.layer[a], conn.index[a]]
    out[~a] = mlp[conn.layer[~a], conn.index[~a]]
    return out


def _fill_outside(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Carry values past the surface so GPU interpolation at the edge stays honest."""
    if mask.all() or not mask.any():
        return vol
    _, idx = distance_transform_edt(~mask, return_indices=True)
    return vol[tuple(idx)]


class _Scanner:
    """Splat helper bound to one grid and one point set."""

    def __init__(self, grid: Grid, xyz: np.ndarray, sigma: float, mask_frac: float = 0.04):
        self.grid, self.xyz, self.sigma = grid, xyz, sigma
        self.den = smooth(splat(grid, xyz), sigma)
        inside = self.den[self.den > 0]
        self.mask = self.den > mask_frac * (np.percentile(inside, 99) if inside.size else 1.0)

    def mean(self, values: np.ndarray, fill: bool = True) -> np.ndarray:
        num = smooth(splat(self.grid, self.xyz, values), self.sigma)
        out = np.where(self.mask, num / np.maximum(self.den, 1e-12), 0.0).astype(np.float32)
        return _fill_outside(out, self.mask) if fill else out


def build_brain(out_dir: str | Path, conn: Connectome, xyz: np.ndarray, *, meta: dict,
                activity: Activity | None = None, conditions: dict[str, Activity] | None = None,
                trace: Activity | None = None, peak: dict | None = None,
                grid_dim: int = 128, sigma: float = 1.2, frame_dim: int = 80,
                fire_at: float = 0.5, verbose: bool = True) -> Path:
    """Write every brain-view file into ``out_dir`` and return the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    L = int(conn.meta["n_layers"])
    N = len(conn)

    grid = fit_grid(xyz, max_dim=grid_dim)
    say(f"[nnviz] brain: grid {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]}, "
        f"{N:,} neurons, PSF sigma {sigma}")
    sc = _Scanner(grid, xyz, sigma)
    vols: list[dict] = []

    def add(name, data, **kw):
        vols.append(write_volume(out, name, data, **kw))

    # ---------------------------------------------------------------- structure
    add("density", sc.den, role="shape", cmap="bone", group="Structure", mask=sc.mask,
        label="Neuron density",
        description="How many neurons sit in each voxel of the wiring layout. Dense regions "
                    "are groups of neurons strongly wired to each other (modules); sparse "
                    "regions are neurons whose connections are weak or scattered.")
    add("depth", sc.mean(conn.layer.astype(np.float32)), role="paint", cmap="turbo",
        group="Structure", window=(0, max(L - 1, 1)), label="Layer depth",
        description="Mean layer of the neurons in each voxel (blue = early, red = late). The "
                    "layout never saw layer numbers, so any gradient here is information flow "
                    "recovered from the wiring alone.", extra={"unit": "layer"})
    add("parts", sc.mean(conn.family.astype(np.float32)), role="paint", cmap="parts2",
        group="Structure", window=(0, 1), label="Attention vs MLP",
        description="Share of MLP neurons (warm) versus attention head dimensions (cool). "
                    "The transformer's two tissue types.", extra={"labels": list(FAMILIES.values())})
    chance = float(conn.meta.get("chance_cos", 0.0)) or 1e-6
    wired = (conn.strength[:, : conn.meta["k"]].mean(1) / chance).astype(np.float32)
    add("wiring", sc.mean(wired), role="paint", cmap="magma", group="Structure",
        label="Wiring strength",
        description="How strongly each neuron is wired to its closest partners, as a multiple "
                    "of what random wiring would give (1.0 = chance). Bright = tightly "
                    "connected module; dark = loosely connected.", extra={"unit": "x chance"})

    # ---------------------------------------------------------------- function
    frames_entry, neuron_frames = None, None
    if activity is not None:
        pk_mix = peak["attn_mix"] if peak else activity.stats["attn_mix"].peak
        pk_mlp = peak["mlp_act"] if peak else activity.stats["mlp_act"].peak
        npeak = np.maximum(neuron_values(conn, pk_mix, pk_mlp), 1e-9)
        rms = neuron_values(conn, activity.stats["attn_mix"].rms, activity.stats["mlp_act"].rms)
        duty = np.clip(rms / npeak, 0, 1)
        add("activity", sc.mean(duty, fill=False), role="overlay", cmap="hot", group="Function",
            mask=sc.mask, label="Typical activity",
            description="Each neuron's RMS activity over all probe prompts as a fraction of its "
                        "own peak (0-1), averaged per voxel. Low almost everywhere: most neurons "
                        "spend most of their time near silent.")

        for name, cond in (conditions or {}).items():
            cv = neuron_values(conn, cond.stats["attn_mix"].rms, cond.stats["mlp_act"].rms)
            ratio = np.log2(np.maximum(cv, 1e-9) / np.maximum(rms, 1e-9)).astype(np.float32)
            add(f"task_{name}", sc.mean(ratio, fill=False), role="signed", cmap="diverge",
                group="Task contrast", transform="signed", mask=sc.mask, label=f"Task: {name}",
                description=f"log2 activity under the '{name}' prompts vs all prompts. Red = "
                            f"recruited by {name}, blue = quieter. Subtracting conditions to "
                            "localise function, as in task fMRI.",
                extra={"prompts": cond.prompts})

        tr = trace.trace if trace is not None else None
        if tr is not None and tr.get("mlp_act") is not None:
            T = len(tr["tokens"])
            a = np.stack([np.clip(neuron_values(conn, np.abs(tr["attn_mix"][t]),
                                                np.abs(tr["mlp_act"][t])) / npeak, 0, 1)
                          for t in range(T)])                                   # (T, N)
            neuron_frames = a
            # Per voxel: the fraction of its neurons firing above half their peak.
            # (Not the mean activity -- that is ~0.1 everywhere and hides the
            # rare near-1 values, which are the whole point of the scan.)
            fgrid = fit_grid(xyz, max_dim=frame_dim)
            fsc = _Scanner(fgrid, xyz, sigma * frame_dim / grid_dim)
            fv = np.stack([fsc.mean((a[t] > fire_at).astype(np.float32), fill=False)
                           for t in range(T)])                                  # (T, x, y, z)
            hi = float(np.percentile(fv[:, fsc.mask], 99.9)) or 1e-6
            q = (np.clip(fv / hi, 0, 1) * 255 + 0.5).astype(np.uint8)
            np.ascontiguousarray(np.transpose(q, (0, 3, 2, 1))).tofile(out / "frames.bin")
            (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8).tofile(out / "neuron_frames.bin")

            # sparsity statistics: the numbers behind "most values stay near 0"
            thr = (0.25, 0.5, 0.75)
            att = conn.family == 0
            per_layer = np.stack([[(a[t, conn.layer == l] > 0.5).mean() for l in range(L)]
                                  for t in range(T)])
            frames_entry = {
                "file": "frames.bin", "neuron_file": "neuron_frames.bin",
                "shape": [T, *fgrid.shape], "dtype": "uint8", "role": "overlay", "cmap": "hot",
                "transform": {"kind": "linear", "lo": 0.0, "hi": hi},
                "tokens": tr["tokens"], "token_index": tr["token_index"], "prompt": tr["prompt"],
                "label": "Dynamic scan",
                "fire_at": fire_at,
                "description": "At each token, the share of neurons in each voxel firing "
                               f"above {fire_at:g} of their own peak. Dark = quiet, bright = a "
                               "cluster of neurons firing together. Play it to watch the "
                               "model read the sentence.",
                "stats": {
                    "thresholds": list(thr),
                    "frac_above": [[float((a[t] > th).mean()) for th in thr] for t in range(T)],
                    "frac_above_attn": [float((a[t, att] > 0.5).mean()) for t in range(T)],
                    "frac_above_mlp": [float((a[t, ~att] > 0.5).mean()) for t in range(T)],
                    "per_layer_above_half": per_layer.round(5).tolist(),
                    "hist": np.histogram(a, bins=20, range=(0, 1))[0].tolist(),
                },
            }
            say(f"[nnviz]   dynamic scan: {T} tokens; neurons above 0.5 of peak: "
                f"{(a > 0.5).mean() * 100:.2f}% on average")

    # ---------------------------------------------------------------- neurons + tracts
    vox = grid.to_voxel(xyz).astype(np.float32)          # neuron positions in voxel units
    vox.tofile(out / "neurons_xyz.bin")
    attrs = np.zeros((N, 4), dtype=np.uint8)
    attrs[:, 0] = np.clip(conn.layer, 0, 255)
    attrs[:, 1] = conn.family
    attrs[:, 2] = np.clip(wired / max(np.percentile(wired, 99.5), 1e-6) * 255, 0, 255)
    attrs[:, 3] = np.clip(conn.head + 1, 0, 255)
    attrs.tofile(out / "neurons_attr.bin")
    e = conn.edges.astype(np.int32)
    e.tofile(out / "tracts.bin")
    ew = conn.edge_w
    (np.clip((ew - ew.min()) / max(ew.max() - ew.min(), 1e-9), 0, 1) * 255).astype(np.uint8) \
        .tofile(out / "tracts_w.bin")

    manifest = {
        "kind": "brain",
        "title": "Brain",
        "meta": {**meta, "connectome": conn.meta, "grid": grid.to_dict(), "psf_sigma": sigma,
                 "occupied_voxels": int(sc.mask.sum())},
        "shape_volume": "density",
        "default": {"paint": "depth", "overlay": None},
        "extent": [int(s) for s in grid.shape],
        "volumes": vols,
        "neurons": {"count": int(N), "xyz": "neurons_xyz.bin", "attr": "neurons_attr.bin",
                    "attr_fields": ["layer", "family", "wiring", "head+1"],
                    "n_layers": L},
        "tracts": {"count": int(len(e)), "file": "tracts.bin", "weight": "tracts_w.bin",
                   "w_range": [float(ew.min()), float(ew.max())],
                   "description": "The strongest synapses in the model (|cos| between a "
                                  "writer and a later reader), drawn as fibres."},
    }
    if frames_entry:
        manifest["frames"] = frames_entry
        manifest["default"]["overlay"] = "@frames"
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
