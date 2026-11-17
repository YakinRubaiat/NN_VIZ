"""One call that scans a model into every view.

    index.json                  which views exist
    brain/manifest.json ...     structure from the wiring + dynamic activity
    stack/manifest.json ...     depth x channel planes
    cache/                      connectome + layout (weights-only, reused)
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np

from . import prompts as _prompts
from .activity import Activity, record_activity
from .connectome import Connectome, build_connectome
from .model import DEFAULT_MODEL, find_layers, load_model

__all__ = ["scan", "slug", "load_connectome", "save_connectome", "write_catalog"]


def write_catalog(data_root: Path) -> Path:
    """List every scan under ``data_root`` so the viewer can switch models."""
    data_root = Path(data_root)
    scans = []
    for idx in sorted(data_root.glob("*/index.json")):
        try:
            d = json.loads(idx.read_text())
        except ValueError:
            continue
        scans.append({"dir": idx.parent.name, "model": d.get("model"),
                      "params_m": round(d.get("meta", {}).get("n_params", 0) / 1e6, 1),
                      "created": d.get("created")})
    scans.sort(key=lambda e: e["params_m"])
    path = data_root / "catalog.json"
    path.write_text(json.dumps({"scans": scans}, indent=1))
    return path


def slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower().replace(".", "-")


def save_connectome(conn: Connectome, path: Path) -> None:
    np.savez_compressed(path, **{k: getattr(conn, k) for k in (
        "family", "layer", "index", "head", "write_norm", "read_norm", "nbr", "strength",
        "out_strength", "in_strength", "edges", "edge_w")},
        meta=np.array(json.dumps(conn.meta)))


def load_connectome(path: Path) -> Connectome:
    d = np.load(path, allow_pickle=False)
    kw = {k: d[k] for k in d.files if k != "meta"}
    return Connectome(**kw, meta=json.loads(str(d["meta"])))


def _peaks(*acts: Activity) -> dict:
    out = {}
    for key in ("attn_mix", "mlp_act", "resid"):
        arrs = [a.stats[key].peak for a in acts if a is not None and key in a.stats]
        if arrs:
            out[key] = np.maximum.reduce(arrs)
    return out


def scan(model: str = DEFAULT_MODEL, *, out: str | Path | None = None, probe: str = "all",
         text: str | None = None, views: tuple[str, ...] = ("arch", "brain", "stack"),
         grid: int = 128, k: int = 12, device: str = "auto", max_len: int = 128,
         max_frames: int = 64, verbose: bool = True) -> Path:
    """Scan ``model`` and write every requested view. Returns ``index.json``.

    probe  which prompt sets define 'typical' activity and the task contrasts:
           ``"all"`` (every set, one contrast per set), a set name, or names
           joined by commas.
    text   the sentence for the dynamic scan. Default: the longest probe prompt.
    """
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    t_start = time.time()
    root = Path(out) if out else Path("data") / slug(model)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)

    net, tok, dev = load_model(model, device=device, verbose=verbose)
    layers, layer_path = find_layers(net)
    cfg = net.config
    n_params = int(sum(p.numel() for p in net.parameters()))

    # ---------------------------------------------------------------- activity
    names = list(_prompts.PROMPT_SETS) if probe == "all" else [
        p.strip() for p in probe.split(",") if p.strip()]
    per_set: dict[str, Activity] = {}
    for name in names:
        prompts = _prompts.PROMPT_SETS[name]

        def prog(i, n, _p, name=name):
            if verbose:
                print(f"\r[nnviz] probe {name:<13} {i}/{n}", end="", flush=True)
        per_set[name] = record_activity(net, tok, prompts, device=dev, label=name,
                                        max_len=max_len, progress=prog)
    if verbose:
        print()
    combined = None
    for a in per_set.values():
        combined = a if combined is None else combined + a

    if text is None:
        text = _prompts.DEFAULT_TRACE
    trace = record_activity(net, tok, [text], device=dev, label="trace", max_len=max_len,
                            trace_prompt=0, max_trace_tokens=max_frames)
    peak = _peaks(combined, trace)
    say(f"[nnviz] dynamic scan text: {text[:70]!r}  ({trace.n_tokens} tokens)")

    meta = {
        "model": model, "n_params": n_params, "n_layers": len(layers),
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(getattr(cfg, "intermediate_size", 0) or 0),
        "n_heads": int(getattr(cfg, "num_attention_heads", 0) or 0),
        "n_kv_heads": int(getattr(cfg, "num_key_value_heads", 0) or 0),
        "layer_path": layer_path, "device": str(dev),
        "probe": probe, "probe_sets": names, "n_prompts": len(combined.prompts),
        "n_tokens": int(combined.n_tokens), "trace_text": text,
    }

    entries = []
    arch_man, conn_meta = None, None
    # ---------------------------------------------------------------- network
    if "arch" in views:
        from .arch import build_arch
        arch_path = build_arch(root / "arch", net, meta=meta, trace=trace, peak=peak, verbose=verbose)
        arch_man = json.loads(arch_path.read_text())
        entries.append({"kind": "arch", "title": "Network", "manifest": "arch/manifest.json"})

    # ---------------------------------------------------------------- brain
    if "brain" in views:
        from .brain import build_brain
        from .layout import layout_connectome

        cpath = cache / f"connectome_k{k}.npz"
        if cpath.exists():
            say(f"[nnviz] connectome: cached ({cpath.name})")
            conn = load_connectome(cpath)
        else:
            conn = build_connectome(net, k=k, verbose=verbose)
            save_connectome(conn, cpath)
        conn_meta = conn.meta
        xyz = layout_connectome(conn, cache_dir=cache, cache_key=f"{model}|k{k}", verbose=verbose)
        conds = per_set if len(per_set) > 1 else {}
        build_brain(root / "brain", conn, xyz, meta=meta, activity=combined, conditions=conds,
                    trace=trace, peak=peak, grid_dim=grid, verbose=verbose)
        entries.append({"kind": "brain", "title": "Brain", "manifest": "brain/manifest.json"})

    # ---------------------------------------------------------------- stack
    if "stack" in views:
        from .stack import build_stack
        build_stack(root / "stack", net, meta=meta, activity=combined, trace=trace, peak=peak,
                    verbose=verbose)
        entries.append({"kind": "stack", "title": "Layer stack", "manifest": "stack/manifest.json"})

    # keep views from earlier runs that this run did not rebuild
    ipath = root / "index.json"
    if ipath.exists():
        try:
            prev = json.loads(ipath.read_text()).get("views", [])
        except ValueError:
            prev = []
        built = {e["kind"] for e in entries}
        entries += [e for e in prev if e["kind"] not in built and (root / e["manifest"]).exists()]
        order = {"arch": 0, "brain": 1, "stack": 2}
        entries.sort(key=lambda e: order.get(e["kind"], 9))
    from .summary import summarize
    if arch_man is None and (root / "arch" / "manifest.json").exists():
        arch_man = json.loads((root / "arch" / "manifest.json").read_text())
    if conn_meta is None and (cache / f"connectome_k{k}.npz").exists():
        conn_meta = load_connectome(cache / f"connectome_k{k}.npz").meta
    summary = summarize(meta=meta, combined=combined, trace=trace, peak=peak,
                        arch_manifest=arch_man, connectome_meta=conn_meta)
    index = {"nnviz": "0.2.0", "model": model, "meta": meta, "summary": summary, "views": entries,
             "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    ipath.write_text(json.dumps(index, indent=1))
    write_catalog(root.parent)
    size = sum(f.stat().st_size for f in root.rglob("*.bin"))
    say(f"[nnviz] done in {time.time() - t_start:.0f}s -> {root}  ({size / 1e6:.1f} MB)")
    return ipath
