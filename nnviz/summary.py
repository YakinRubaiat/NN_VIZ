"""Headline numbers for a scan, so different models can be compared side by side.

    massive      the bright streaks of the residual-energy scan: a few residual
                 channels running 10-1000x louder than the rest ("massive
                 activations"), where they start, and which token carries them
                 (usually the first one: the "attention sink").
    sparsity     how many neurons fire above half their own peak per token.
    density      which layer holds the largest / densest weights, and how many
                 weights are outliers (> 4x the panel RMS).
    wiring       how far the strongest synapses are above random alignment.
"""

from __future__ import annotations

import numpy as np

from .activity import Activity

__all__ = ["summarize"]


def _massive(combined: Activity, trace: Activity | None) -> dict:
    rms = combined.stats["resid"].rms.astype(np.float64)            # (L+1, H)
    med = float(np.median(rms))
    ratio = rms.max(0) / max(med, 1e-12)                             # per channel
    order = np.argsort(-ratio)
    top = int(order[0])
    hot = np.where(rms[:, top] > 10 * med)[0]
    out = {
        "median_energy": med,
        "n_over_20x": int((ratio > 20).sum()),
        "n_over_100x": int((ratio > 100).sum()),
        "top_channels": [int(c) for c in order[:6]],
        "top_ratio": float(ratio[top]),
        "top_onset_layer": int(hot.min()) if hot.size else None,
        "top_layers_hot": int(hot.size),
        "n_depths": int(rms.shape[0]),
    }
    tr = trace.trace if trace is not None else None
    if tr is not None and tr.get("resid") is not None:
        x = tr["resid"].astype(np.float64)                           # (T, L+1, H)
        chans = [c for c in order[:8] if ratio[c] > 20] or [top]
        e_tok = (x[:, :, chans] ** 2).sum(axis=(1, 2))               # energy per token
        share = e_tok / max(e_tok.sum(), 1e-12)
        k = int(share.argmax())
        out.update({
            "carrier_token_index": int(tr["token_index"][k]),
            "carrier_token": tr["tokens"][k],
            "carrier_share": float(share[k]),
            "first_token": tr["tokens"][0],
            "n_trace_tokens": int(x.shape[0]),
        })
    return out


def _sparsity(trace: Activity | None, peak: dict) -> dict:
    tr = trace.trace if trace is not None else None
    if tr is None or tr.get("mlp_act") is None:
        return {}
    a_m = np.abs(tr["mlp_act"]) / np.maximum(peak["mlp_act"], 1e-9)[None]
    a_a = np.abs(tr["attn_mix"]) / np.maximum(peak["attn_mix"], 1e-9)[None]
    # skip token 0: it is the attention sink and behaves unlike every other token
    a_m, a_a = a_m[1:], a_a[1:]
    allv = np.concatenate([a_m.reshape(len(a_m), -1), a_a.reshape(len(a_a), -1)], 1)
    return {
        "frac_over_half": float((allv > 0.5).mean()),
        "frac_over_half_mlp": float((a_m > 0.5).mean()),
        "frac_over_half_attn": float((a_a > 0.5).mean()),
        "median_activity": float(np.median(allv)),
        "frac_over_0_9": float((allv > 0.9).mean()),
    }


def _density(arch_manifest: dict | None) -> dict:
    if not arch_manifest:
        return {}
    P = [p for p in arch_manifest["weights"]["panels"] if p["layer"] >= 0 and p["kind"] != "lm_head"]
    L = arch_manifest["n_layers"]
    by_layer = np.array([np.average([p["rms"] for p in P if p["layer"] == l],
                                    weights=[p["params"] for p in P if p["layer"] == l]) for l in range(L)])
    med = float(np.median(by_layer))
    dense = int(by_layer.argmax())
    big = max(P, key=lambda p: p["max_abs"])
    fam = {f: float(np.average([p["outlier_frac"] for p in P if p["family"] == f],
                               weights=[p["params"] for p in P if p["family"] == f]))
           for f in ("attn", "mlp")}
    return {
        "densest_layer": dense, "densest_ratio": float(by_layer[dense] / med),
        "layer_rms": by_layer.round(5).tolist(),
        "max_abs_weight": float(big["max_abs"]),
        "max_abs_where": f"layer {big['layer']} {big['kind']}",
        "outlier_frac_attn": fam["attn"], "outlier_frac_mlp": fam["mlp"],
    }


def summarize(*, meta: dict, combined: Activity, trace: Activity | None, peak: dict,
              arch_manifest: dict | None = None, connectome_meta: dict | None = None) -> dict:
    s = {
        "model": meta["model"], "params_m": meta["n_params"] / 1e6, "layers": meta["n_layers"],
        "hidden": meta["hidden_size"], "intermediate": meta["intermediate_size"],
        "massive": _massive(combined, trace),
        "sparsity": _sparsity(trace, peak),
        "density": _density(arch_manifest),
    }
    if connectome_meta:
        s["wiring"] = {"mean_top_cos": connectome_meta["mean_top_cos"],
                       "chance_cos": connectome_meta["chance_cos"],
                       "x_chance": connectome_meta["mean_top_cos"] / connectome_meta["chance_cos"],
                       "n_neurons": connectome_meta["n_neurons"]}
    return s
