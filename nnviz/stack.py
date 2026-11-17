"""The layer-stack view: depth on one axis, each layer's channels on a plane.

    Z     = depth in the network (layer 0 -> last)
    X, Y  = that layer's channel axis, seriated (ordering.py) and folded onto a
            plane with a generalised Hilbert curve (curve.py)

So one axial slice is "the whole residual stream at layer k" and walking Z walks
through the network. Complements the brain view: the brain shows *where* things
are wired; the stack shows *when* (at which depth) they happen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .activity import Activity
from .curve import best_grid, curve_coords, fold_to_plane
from .model import block_parts, find_layers
from .ordering import profile_from_stack, spectral_order
from .volio import write_volume

__all__ = ["Space", "build_stack", "structural_maps"]


@dataclass
class Space:
    name: str
    n_channels: int
    width: int
    height: int
    order: np.ndarray
    description: str = ""

    @property
    def padding(self) -> int:
        return self.width * self.height - self.n_channels

    def mask(self) -> np.ndarray:
        m = np.zeros((self.width, self.height), dtype=bool)
        c = curve_coords(self.width, self.height)[: self.n_channels]
        m[c[:, 0], c[:, 1]] = True
        return m

    def channel_at(self) -> np.ndarray:
        """``(height, width)`` map voxel -> original channel index (-1 = padding)."""
        out = np.full((self.width, self.height), -1, dtype=np.int32)
        c = curve_coords(self.width, self.height)[: self.n_channels]
        out[c[:, 0], c[:, 1]] = self.order[: self.n_channels]
        return out.T


def _seriate(name: str, maps: list[np.ndarray], description: str, knn: int = 24) -> Space:
    depth = min(m.shape[0] for m in maps)
    stack = np.stack([m[:depth] for m in maps])
    c = stack.shape[-1]
    w, h, _ = best_grid(c)
    return Space(name, c, w, h, spectral_order(profile_from_stack(stack), knn=knn), description)


def _resample_z(vol: np.ndarray, out_z: int) -> np.ndarray:
    in_z = vol.shape[-1]
    if out_z == in_z or in_z < 2:
        return vol.astype(np.float32)
    src = np.linspace(0.0, in_z - 1.0, out_z)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, in_z - 1)
    f = (src - lo).astype(np.float32)
    return (vol[..., lo] * (1 - f) + vol[..., hi] * f).astype(np.float32)


def _fold(maps: np.ndarray, sp: Space, z_scale: int) -> np.ndarray:
    """``(L, C)`` -> ``(w, h, L*z_scale)``."""
    planes = fold_to_plane(maps[:, sp.order], sp.width, sp.height)       # (L, w, h)
    vol = np.transpose(planes, (1, 2, 0))
    return _resample_z(vol, maps.shape[0] * max(1, z_scale))


def _norms(w: torch.Tensor, dim: int) -> np.ndarray:
    return torch.linalg.vector_norm(w.detach().to(torch.float32), dim=dim).cpu().numpy()


@torch.no_grad()
def structural_maps(model) -> list[tuple]:
    """Weight-only ``(name, space, (L, C) map, label, description)`` tuples."""
    layers, _ = find_layers(model)
    rd, wr, ga, gm, nin, nout = [], [], [], [], [], []
    for blk in layers:
        p = block_parts(blk)
        rd.append(np.sqrt(sum(_norms(p[k].weight, 0) ** 2 for k in ("q", "k", "v", "gate", "up"))))
        wr.append(np.sqrt(sum(_norms(p[k].weight, 1) ** 2 for k in ("o", "down"))))
        ga.append(p["ln_attn"].weight.detach().float().cpu().numpy())
        gm.append(p["ln_mlp"].weight.detach().float().cpu().numpy())
        nin.append(_norms(p["gate"].weight, 1))
        nout.append(_norms(p["down"].weight, 0))
    S = np.stack
    return [
        ("w_read", "residual", S(rd), "Read strength",
         "Norm of all weights that read each residual channel (q,k,v,gate,up)."),
        ("w_write", "residual", S(wr), "Write strength",
         "Norm of all weights that write each residual channel (o, down)."),
        ("ln_attn_gain", "residual", S(ga), "Pre-attn norm gain",
         "Learned RMSNorm gain per channel before attention."),
        ("ln_mlp_gain", "residual", S(gm), "Pre-MLP norm gain",
         "Learned RMSNorm gain per channel before the MLP."),
        ("neuron_w_in", "mlp", S(nin), "Neuron fan-in",
         "Norm of each MLP neuron's gate weights: how sharply tuned its detector is."),
        ("neuron_w_out", "mlp", S(nout), "Neuron fan-out",
         "Norm of each MLP neuron's output weights: how loudly it speaks when it fires."),
    ]


def build_stack(out_dir: str | Path, model, *, meta: dict, activity: Activity | None = None,
                trace: Activity | None = None, peak: dict | None = None,
                z_scale: int = 4, verbose: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)

    maps = [(n, s, d, lb, ds, "Structure") for n, s, d, lb, ds in structural_maps(model)]
    if activity is not None:
        st = activity.stats
        maps += [
            ("resid_energy", "residual", st["resid"].rms, "Residual energy",
             "RMS of each residual channel at each depth over all probe tokens.", "Function"),
            ("resid_delta", "residual", st["delta"].rms, "Layer contribution",
             "RMS of h[l+1]-h[l]: how much each block changes each channel.", "Function"),
            ("attn_write", "residual", st["attn_out"].rms, "Attention write",
             "RMS of the attention block's output per channel.", "Function"),
            ("mlp_write", "residual", st["mlp_out"].rms, "MLP write",
             "RMS of the MLP block's output per channel.", "Function"),
            ("burstiness", "residual", st["resid"].kurtosis, "Burstiness",
             "Kurtosis across tokens: ~3 = always-on, high = quiet with rare spikes.", "Function"),
            ("mlp_act", "mlp", st["mlp_act"].rms, "Neuron activation",
             "RMS of each MLP neuron's activation. One voxel = one neuron.", "Function"),
            ("mlp_duty", "mlp", np.clip(st["mlp_act"].rms / np.maximum(
                (peak or {}).get("mlp_act", st["mlp_act"].peak), 1e-9), 0, 1),
             "Neuron duty (0-1)",
             "RMS activation as a fraction of the neuron's own peak.", "Function"),
        ]

    notes = {"residual": "One voxel = one residual-stream channel.",
             "mlp": "One voxel = one MLP neuron of that layer."}
    spaces: dict[str, Space] = {}
    for sname in ("residual", "mlp"):
        group = [m for m in maps if m[1] == sname]
        pref = [m[2] for m in group if m[5] == "Function"] or [m[2] for m in group]
        spaces[sname] = _seriate(sname, pref, notes[sname])
        sp = spaces[sname]
        say(f"[nnviz] stack: {sname} {sp.n_channels} ch -> {sp.width}x{sp.height}"
            f" ({sp.padding} pad)")

    vols = []
    for name, sname, data, label, desc, group in maps:
        sp = spaces[sname]
        vol = _fold(np.asarray(data, np.float32), sp, z_scale)
        mask = np.repeat(sp.mask()[:, :, None], vol.shape[2], axis=2)
        vols.append(write_volume(
            out, name, vol, label=label, description=desc, role="paint",
            cmap="bone" if group == "Structure" else "hot", group=group, mask=mask,
            extra={"space": sname, "n_layers": int(data.shape[0]),
                   "extent": [sp.width, sp.height, int(data.shape[0])]}))

    frames_entry = None
    tr = trace.trace if trace is not None else None
    if tr is not None and activity is not None:
        sp = spaces["residual"]
        pk = np.maximum(activity.stats["resid"].peak, 1e-9)
        if peak and "resid" in peak:
            pk = np.maximum(peak["resid"], 1e-9)
        a = np.clip(np.abs(tr["resid"]) / pk[None], 0, 1)                    # (T, L+1, H)
        fv = np.stack([_fold(a[t], sp, max(1, z_scale // 2)) for t in range(a.shape[0])])
        hi = float(np.percentile(fv, 99.7)) or 1e-6
        q = (np.clip(fv / hi, 0, 1) * 255 + 0.5).astype(np.uint8)
        np.ascontiguousarray(np.transpose(q, (0, 3, 2, 1))).tofile(out / "frames.bin")
        frames_entry = {
            "file": "frames.bin", "shape": [int(q.shape[0]), *map(int, q.shape[1:])],
            "dtype": "uint8", "role": "overlay", "cmap": "hot",
            "transform": {"kind": "linear", "lo": 0.0, "hi": hi},
            "tokens": tr["tokens"], "token_index": tr["token_index"], "prompt": tr["prompt"],
            "space": "residual", "extent": [sp.width, sp.height, int(a.shape[1])],
            "label": "Dynamic scan",
            "description": "|residual| per channel at each token, as a fraction of the "
                           "channel's peak.",
        }

    manifest = {
        "kind": "stack", "title": "Layer stack", "meta": meta, "shape_volume": None,
        "default": {"paint": "resid_energy" if activity is not None else "w_read",
                    "overlay": None},
        "spaces": {s.name: {"name": s.name, "n_channels": s.n_channels, "width": s.width,
                            "height": s.height, "padding": s.padding,
                            "description": s.description,
                            "channel_at": s.channel_at().ravel().tolist()}
                   for s in spaces.values()},
        "volumes": vols,
    }
    if frames_entry:
        manifest["frames"] = frames_entry
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
