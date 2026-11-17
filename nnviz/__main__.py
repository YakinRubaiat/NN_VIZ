"""Command line.

    python -m nnviz scan  [--model ID] [--text "..."] [--probe all] [--out DIR]
    python -m nnviz view  [DIR]
    python -m nnviz info  [DIR]
    python -m nnviz sheets [DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import DEFAULT_MODEL
from .prompts import PROMPT_SETS


def _latest() -> Path | None:
    found = [p.parent for p in Path("data").glob("*/index.json")] if Path("data").is_dir() else []
    return max(found, key=lambda p: (p / "index.json").stat().st_mtime) if found else None


def _target(arg) -> Path | None:
    t = Path(arg) if arg else _latest()
    if t is None or not (t / "index.json").exists():
        print("[nnviz] no scan found; run: python -m nnviz scan", file=sys.stderr)
        return None
    return t


def cmd_scan(a) -> int:
    from .scan import scan
    views = tuple(v.strip() for v in a.views.split(","))
    idx = scan(a.model, out=a.out, probe=a.probe, text=a.text, views=views, grid=a.grid,
               k=a.k, device=a.device, max_len=a.max_len, max_frames=a.frames)
    root = idx.parent
    if a.sheets:
        from .previews import render_previews
        files = render_previews(root)
        print(f"[nnviz] {len(files)} preview PNGs in out/{root.name}")
    if a.view:
        from .server import serve
        serve(root, port=a.port)
    else:
        print(f"[nnviz] view it:  python -m nnviz view {root}")
    return 0


def cmd_view(a) -> int:
    from .server import serve
    from .scan import write_catalog
    if a.dir is None and Path("data").is_dir() and any(Path("data").glob("*/index.json")):
        write_catalog(Path("data"))          # serve every scan; pick the model in the viewer
        serve(Path("data"), port=a.port, open_browser=not a.no_browser)
        return 0
    t = _target(a.dir)
    if t is None:
        return 1
    serve(t, port=a.port, open_browser=not a.no_browser)
    return 0


def _row(s: dict) -> dict:
    m, sp, d, w = s.get("massive", {}), s.get("sparsity", {}), s.get("density", {}), s.get("wiring", {})
    pct = lambda v: f"{v * 100:.2f}%" if v is not None else "-"
    return {
        "model": s["model"].split("/")[-1],
        "params": f"{s['params_m']:.0f}M",
        "layers x hidden": f"{s['layers']} x {s['hidden']}",
        "loudest channel": f"#{m.get('top_channels', ['?'])[0]} at {m.get('top_ratio', 0):.0f}x",
        "channels >100x": str(m.get("n_over_100x", "-")),
        "streak starts": f"layer {m.get('top_onset_layer')}" if m.get("top_onset_layer") is not None else "-",
        "carried by": (f"{m.get('carrier_share', 0) * 100:.0f}% on token {m.get('carrier_token_index')}"
                       f" ({m.get('carrier_token', '').strip() or 'sp'!s})") if "carrier_share" in m else "-",
        "firing >1/2 peak": pct(sp.get("frac_over_half")),
        "densest layer": f"{d.get('densest_layer')} ({d.get('densest_ratio', 0):.2f}x)" if d else "-",
        "max |w|": f"{d.get('max_abs_weight', 0):.2f} ({d.get('max_abs_where', '')})" if d else "-",
        "outliers attn/mlp": f"{d.get('outlier_frac_attn', 0) * 100:.2f}% / {d.get('outlier_frac_mlp', 0) * 100:.2f}%" if d else "-",
        "wiring": f"{w['x_chance']:.2f}x chance" if w else "-",
    }


def cmd_compare(a) -> int:
    dirs = [Path(d) for d in a.dirs] if a.dirs else sorted(p.parent for p in Path("data").glob("*/index.json"))
    rows = []
    for d in dirs:
        idx = json.loads((d / "index.json").read_text())
        if "summary" not in idx:
            print(f"[nnviz] {d}: no summary (re-scan with this version)", file=sys.stderr)
            continue
        rows.append(_row(idx["summary"]))
    if not rows:
        return 1
    rows.sort(key=lambda r: float(r["params"][:-1]))
    keys = list(rows[0])
    if a.markdown:
        print("| " + " | ".join(keys) + " |")
        print("|" + "---|" * len(keys))
        for r in rows:
            print("| " + " | ".join(r[k] for k in keys) + " |")
        return 0
    width = max(len(k) for k in keys)
    colw = [max(len(r["model"]), 14) for r in rows]
    print(" " * width + "  " + "  ".join(r["model"].ljust(w) for r, w in zip(rows, colw)))
    for k in keys[1:]:
        print(k.ljust(width) + "  " + "  ".join(r[k].ljust(w) for r, w in zip(rows, colw)))
    return 0


def cmd_info(a) -> int:
    t = _target(a.dir)
    if t is None:
        return 1
    idx = json.loads((t / "index.json").read_text())
    m = idx["meta"]
    print(f"{m['model']}  {m['n_layers']} layers  hidden {m['hidden_size']}  "
          f"{m['n_params'] / 1e6:.0f}M params   scanned {idx['created']}")
    print(f"probe: {m['probe']} ({m['n_prompts']} prompts, {m['n_tokens']} tokens)")
    print(f"dynamic scan: {m['trace_text'][:80]!r}")
    for v in idx["views"]:
        man = json.loads((t / v["manifest"]).read_text())
        print(f"\n[{v['title']}]")
        if man["kind"] == "arch":
            w = man["weights"]["panels"]
            print(f"  {len(w)} weight panels, {sum(x['params'] for x in w) / 1e6:.0f}M params, "
                  f"texel {man['meta']['texel']}x{man['meta']['texel']}")
        if man["kind"] == "brain":
            c = man["meta"]["connectome"]
            print(f"  {c['n_neurons']:,} neurons, wiring {c['mean_top_cos']:.3f} vs chance "
                  f"{c['chance_cos']:.3f}, grid {'x'.join(map(str, man['extent']))}")
        for vol in man["volumes"]:
            print(f"  {vol['name']:<18} {vol['role']:<8} {'x'.join(map(str, vol['shape']))}")
        if "frames" in man:
            print(f"  frames             {man['frames']['shape'][0]} tokens")
    return 0


def cmd_sheets(a) -> int:
    from .previews import render_previews
    t = _target(a.dir)
    if t is None:
        return 1
    files = render_previews(t, a.out)
    print(f"[nnviz] wrote {len(files)} PNGs")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nnviz", description="Dynamic brain scan of a language model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="scan a model (structure + activity)")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL)
    p.add_argument("--text", "-t", default=None, help="sentence for the dynamic scan")
    p.add_argument("--probe", "-p", default="all",
                   help=f"prompt sets: all | {','.join(PROMPT_SETS)}")
    p.add_argument("--views", default="arch,brain,stack", help="any of arch,brain,stack")
    p.add_argument("--out", "-o", default=None)
    p.add_argument("--grid", type=int, default=128, help="voxel grid, longest side")
    p.add_argument("--k", type=int, default=12, help="synapses kept per neuron, each direction")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--frames", type=int, default=64, help="max tokens in the dynamic scan")
    p.add_argument("--sheets", action="store_true", help="also write PNG previews")
    p.add_argument("--view", action="store_true", help="open the viewer when done")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("view", help="open the viewer on a scan")
    p.add_argument("dir", nargs="?")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("info", help="summarise a scan")
    p.add_argument("dir", nargs="?")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("compare", help="compare saved scans of different models")
    p.add_argument("dirs", nargs="*")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sheets", help="PNG previews of a scan")
    p.add_argument("dir", nargs="?")
    p.add_argument("--out", "-o", default=None)
    p.set_defaults(func=cmd_sheets)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
