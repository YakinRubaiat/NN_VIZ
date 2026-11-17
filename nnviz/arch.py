"""The network view: the real architecture, every parameter visible.

Every weight matrix of the model becomes a panel laid out where it sits in the
computation -- token embedding, then each layer's attention (q, k, v, o) and MLP
(gate, up, down) and their norm gains, then the output head. A panel's area is
proportional to its parameter count, so you see where the parameters are.

Each texel is the RMS of a block of real weights (8x8 for Qwen3-0.6B), on one
global intensity scale for the whole model: bright = large weights packed
densely, dark = small. The residual-stream axis is horizontal in every panel
(read matrices as stored; write matrices o and down transposed), so a hidden
dimension that carries unusually large weights everywhere shows up as a bright
vertical line running through the whole network.

Activity flows through the same picture: for each token, each panel's rows (its
heads' dimensions or MLP neurons) glow with their activation as a fraction of
peak, and a strip above each layer shows the residual stream itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .activity import Activity
from .model import block_parts, find_layers

__all__ = ["build_arch", "pool_rms"]


def pool_rms(w: torch.Tensor, fr: int, fc: int) -> np.ndarray:
    """RMS over ``fr x fc`` blocks; ragged edges average only real entries."""
    w = w.detach().to(torch.float32)
    R, C = w.shape
    Rp, Cp = math.ceil(R / fr) * fr, math.ceil(C / fc) * fc
    sq = torch.zeros((Rp, Cp), dtype=torch.float32, device=w.device)
    cnt = torch.zeros((Rp, Cp), dtype=torch.float32, device=w.device)
    sq[:R, :C] = w * w
    cnt[:R, :C] = 1.0
    s = sq.reshape(Rp // fr, fr, Cp // fc, fc).sum((1, 3))
    n = cnt.reshape(Rp // fr, fr, Cp // fc, fc).sum((1, 3)).clamp_min(1.0)
    return torch.sqrt(s / n).cpu().numpy()


def _pool_max(v: np.ndarray, f: int) -> np.ndarray:
    """Max over groups of ``f`` along the last axis (spikes survive pooling)."""
    n = v.shape[-1]
    m = math.ceil(n / f) * f
    pad = np.zeros(v.shape[:-1] + (m,), dtype=v.dtype)
    pad[..., :n] = v
    return pad.reshape(v.shape[:-1] + (m // f, f)).max(-1)


def _mstats(w: torch.Tensor) -> dict:
    a = w.detach().to(torch.float32).abs()
    rms = float(torch.sqrt((a * a).mean()))
    return {"params": int(a.numel()), "rms": rms, "mean_abs": float(a.mean()),
            "max_abs": float(a.max()), "outlier_frac": float((a > 4 * rms).float().mean())}


@torch.no_grad()
def build_arch(out_dir: str | Path, model, *, meta: dict, trace: Activity | None = None,
               peak: dict | None = None, embed_rows: int = 512, verbose: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    layers, _ = find_layers(model)
    L = len(layers)
    cfg = model.config
    H = int(cfg.hidden_size)
    f = max(1, H // 128)                                  # texel = f x f weights
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)

    mats: list[dict] = []          # weight panels
    norms: list[dict] = []         # gain strips
    pooled: list[np.ndarray] = []
    npooled: list[np.ndarray] = []

    def add(name, kind, layer, w_rows_by_resid, family, fr=None, note=""):
        """``w_rows_by_resid`` is (rows, H): residual axis = columns."""
        p = pool_rms(w_rows_by_resid, fr or f, f)
        st = _mstats(w_rows_by_resid)
        mats.append({"name": name, "kind": kind, "layer": layer, "family": family,
                     "true_shape": [int(w_rows_by_resid.shape[0]), int(w_rows_by_resid.shape[1])],
                     "tex_shape": [int(p.shape[0]), int(p.shape[1])],
                     "row_factor": int(fr or f), "col_factor": int(f), "note": note, **st})
        pooled.append(p)

    def add_norm(name, layer, gain):
        g = gain.detach().to(torch.float32).abs().cpu().numpy()
        gp = _pool_max(g[None], f)[0]
        norms.append({"name": name, "layer": layer, "true_len": int(g.size), "tex_len": int(gp.size),
                      "mean": float(g.mean()), "max": float(g.max())})
        npooled.append(gp)

    emb = model.get_input_embeddings().weight
    V = int(emb.shape[0])
    fr_e = math.ceil(V / embed_rows)
    add("embed", "embed", -1, emb, "embed", fr=fr_e,
        note=f"{V:,} token vectors; each texel row averages {fr_e} tokens")
    for li, blk in enumerate(layers):
        p = block_parts(blk)
        add_norm("ln_attn", li, p["ln_attn"].weight)
        add("q", "q", li, p["q"].weight, "attn")
        add("k", "k", li, p["k"].weight, "attn")
        add("v", "v", li, p["v"].weight, "attn")
        add("o", "o", li, p["o"].weight.T, "attn", note="transposed: residual axis horizontal")
        add_norm("ln_mlp", li, p["ln_mlp"].weight)
        add("gate", "gate", li, p["gate"].weight, "mlp")
        add("up", "up", li, p["up"].weight, "mlp")
        add("down", "down", li, p["down"].weight.T, "mlp", note="transposed: residual axis horizontal")
    final_norm = getattr(getattr(model, "model", model), "norm", None)
    if final_norm is not None:
        add_norm("ln_final", L, final_norm.weight)
    head = model.get_output_embeddings()
    tied = bool(getattr(cfg, "tie_word_embeddings", False)) or (
        head is not None and head.weight.data_ptr() == emb.data_ptr())
    if head is not None and not tied:
        add("lm_head", "lm_head", L, head.weight, "embed", fr=fr_e)

    # ---- one global intensity scale for every weight panel ------------------
    allv = np.concatenate([p.ravel() for p in pooled])
    s = float(np.median(allv))
    hi = float(np.percentile(allv, 99.9))
    t_hi = np.arcsinh(hi / s)
    blob, off = [], 0
    for m, p in zip(mats, pooled):
        q = np.clip(np.arcsinh(p / s) / t_hi, 0, 1)
        q8 = (q * 255 + 0.5).astype(np.uint8)
        m["offset"] = off
        m["q_lo"], m["q_hi"] = (float(x) for x in np.percentile(q8, [1, 99.5]))
        m["pooled_mean"] = float(p.mean())
        blob.append(q8.ravel())
        off += q8.size
    np.concatenate(blob).tofile(out / "weights.bin")

    nall = np.concatenate(npooled)
    nhi = float(np.percentile(nall, 99.5))
    nblob, off = [], 0
    for n, gp in zip(norms, npooled):
        q8 = (np.clip(gp / nhi, 0, 1) * 255 + 0.5).astype(np.uint8)
        n["offset"] = off
        nblob.append(q8)
        off += q8.size
    np.concatenate(nblob).tofile(out / "norms.bin")
    say(f"[nnviz] network: {len(mats)} weight panels, {len(norms)} norm strips, "
        f"texel = {f}x{f} weights, scale asinh(rms/{s:.4f}) up to {hi:.4f}")

    # ---- activity flowing through the panels --------------------------------
    flow_entry = None
    tr = trace.trace if trace is not None else None
    if tr is not None and tr.get("mlp_act") is not None and tr.get("attn_mix") is not None and peak:
        T = len(tr["tokens"])
        hd = int(tr["attn_mix"].shape[-1]) // n_heads
        res = np.clip(np.abs(tr["resid"]) / np.maximum(peak["resid"], 1e-9)[None], 0, 1)   # (T, L+1, H)
        mix = np.clip(np.abs(tr["attn_mix"]) / np.maximum(peak["attn_mix"], 1e-9)[None], 0, 1)  # (T, L, A)
        mlp = np.clip(np.abs(tr["mlp_act"]) / np.maximum(peak["mlp_act"], 1e-9)[None], 0, 1)    # (T, L, I)
        kv = mix.reshape(T, L, n_kv, n_heads // n_kv, hd).mean(3).reshape(T, L, n_kv * hd)
        parts = {"resid": _pool_max(res, f), "attn": _pool_max(mix, f),
                 "kv": _pool_max(kv, f), "mlp": _pool_max(mlp, f)}
        layout, off = {}, 0
        for k, arr in parts.items():
            layout[k] = {"offset": off, "rows": int(arr.shape[1]), "len": int(arr.shape[2])}
            off += arr.shape[1] * arr.shape[2]
        frame_size = off
        buf = np.concatenate([(np.clip(parts[k], 0, 1) * 255 + 0.5).astype(np.uint8)
                              .reshape(T, -1) for k in parts], axis=1)
        buf.tofile(out / "flow.bin")
        per_layer = (mlp > 0.5).mean(-1) * (mlp.shape[-1] / (mlp.shape[-1] + mix.shape[-1])) + \
                    (mix > 0.5).mean(-1) * (mix.shape[-1] / (mlp.shape[-1] + mix.shape[-1]))
        flow_entry = {
            "file": "flow.bin", "frame_size": int(frame_size), "layout": layout, "T": T,
            "tokens": tr["tokens"], "token_index": tr["token_index"], "prompt": tr["prompt"],
            "label": "Dynamic scan",
            "description": "For each token, every panel's rows glow with how hard that head "
                           "dimension or MLP neuron is firing (fraction of its own peak), and "
                           "the strip above each layer shows the residual stream.",
            "stats": {
                "frac_above_attn": [float((mix[t] > 0.5).mean()) for t in range(T)],
                "frac_above_mlp": [float((mlp[t] > 0.5).mean()) for t in range(T)],
                "per_layer_above_half": per_layer.round(5).tolist(),
                "frac_above": [[float((np.concatenate([mix[t].ravel(), mlp[t].ravel()]) > th).mean())
                                for th in (0.25, 0.5, 0.75)] for t in range(T)],
            },
        }
        say(f"[nnviz]   flow: {T} tokens x {frame_size} values")

    total = sum(m["params"] for m in mats)
    manifest = {
        "kind": "arch", "title": "Network",
        "meta": {**meta, "texel": f, "tied_embeddings": tied, "weight_params_shown": int(total)},
        "hidden": H, "n_layers": L,
        "weights": {"file": "weights.bin", "transform": {"kind": "asinh", "lo": 0.0, "hi": hi, "scale": s},
                    "panels": mats},
        "norms": {"file": "norms.bin", "transform": {"kind": "linear", "lo": 0.0, "hi": nhi},
                  "strips": norms},
        "volumes": [],
    }
    if flow_entry:
        manifest["flow"] = flow_entry
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
