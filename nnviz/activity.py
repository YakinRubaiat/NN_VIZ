"""Run prompts through the model and record what every unit does.

One recorder feeds both views. It hooks every block and keeps streaming
per-channel moments for:

    resid     residual stream at each depth           (L+1, hidden)
    delta     change each block makes to it           (L,   hidden)
    attn_out  attention block output                  (L,   hidden)
    mlp_out   MLP block output                        (L,   hidden)
    attn_mix  attention head outputs, pre-o_proj      (L,   heads*head_dim)
    mlp_act   MLP neuron activations, pre-down_proj   (L,   intermediate)

and, for one chosen prompt, the full per-token tensors (the fMRI time course).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import block_parts, find_layers

__all__ = ["StreamStats", "Activity", "record_activity"]


class StreamStats:
    """Streaming per-channel moments over tokens for a stack of layers.

    Keeps count, sum, sum of squares, sum of 4th powers, positive counts and the
    running |max|, so RMS / kurtosis / firing rate come out without holding
    every token in memory. ``a + b`` merges two recordings.
    """

    def __init__(self, n_layers: int, n_channels: int) -> None:
        shape = (n_layers, n_channels)
        self.n = np.zeros(n_layers, dtype=np.float64)
        self.s1 = np.zeros(shape, dtype=np.float64)
        self.s2 = np.zeros(shape, dtype=np.float64)
        self.s4 = np.zeros(shape, dtype=np.float64)
        self.pos = np.zeros(shape, dtype=np.float64)
        self.amax = np.zeros(shape, dtype=np.float64)

    def update(self, layer: int, x: torch.Tensor) -> None:
        a = x.detach().to(torch.float32).cpu().numpy().astype(np.float64, copy=False)
        a = a.reshape(-1, a.shape[-1])
        sq = a * a
        self.n[layer] += a.shape[0]
        self.s1[layer] += a.sum(axis=0)
        self.s2[layer] += sq.sum(axis=0)
        self.s4[layer] += (sq * sq).sum(axis=0)
        self.pos[layer] += (a > 0).sum(axis=0)
        np.maximum(self.amax[layer], np.abs(a).max(axis=0), out=self.amax[layer])

    def __add__(self, other: "StreamStats") -> "StreamStats":
        out = StreamStats(*self.s1.shape)
        for k in ("n", "s1", "s2", "s4", "pos"):
            setattr(out, k, getattr(self, k) + getattr(other, k))
        out.amax = np.maximum(self.amax, other.amax)
        return out

    def _n(self) -> np.ndarray:
        return np.maximum(self.n, 1.0)[:, None]

    @property
    def rms(self) -> np.ndarray:
        return np.sqrt(self.s2 / self._n()).astype(np.float32)

    @property
    def peak(self) -> np.ndarray:
        return self.amax.astype(np.float32)

    @property
    def fire_rate(self) -> np.ndarray:
        return (self.pos / self._n()).astype(np.float32)

    @property
    def kurtosis(self) -> np.ndarray:
        """m4 / m2^2: ~3 = Gaussian, always-on; high = quiet with rare spikes."""
        m2 = self.s2 / self._n()
        m4 = self.s4 / self._n()
        return np.divide(m4, m2 * m2, out=np.zeros_like(m4), where=m2 > 1e-12).astype(np.float32)


STREAMS = ("resid", "delta", "attn_out", "mlp_out", "attn_mix", "mlp_act")


@dataclass
class Activity:
    """Everything recorded for one set of prompts."""

    label: str
    prompts: list[str]
    n_tokens: int
    stats: dict[str, StreamStats]
    trace: dict | None = None   # per-token tensors for one prompt
    meta: dict = field(default_factory=dict)

    def __add__(self, other: "Activity") -> "Activity":
        return Activity(
            label=f"{self.label}+{other.label}",
            prompts=self.prompts + other.prompts,
            n_tokens=self.n_tokens + other.n_tokens,
            stats={k: self.stats[k] + other.stats[k] for k in self.stats if k in other.stats},
            trace=self.trace or other.trace,
            meta=self.meta,
        )


def _clean_token(tok: str) -> str:
    """Printable BPE token (byte-level BPE uses 'G-dot' for a leading space)."""
    return (tok or "").replace("\u0120", " ").replace("\u010a", "\\n").replace("\u2581", " ")


def _token_strings(tokenizer, ids: list[int]) -> list[str]:
    try:
        toks = tokenizer.convert_ids_to_tokens(ids)
        if toks is not None:
            return [_clean_token(t if isinstance(t, str) else str(t)) for t in toks]
    except Exception:
        pass
    return [tokenizer.decode([i]) for i in ids]


@torch.no_grad()
def record_activity(model, tokenizer, prompts: list[str], *, device, label: str = "",
                    max_len: int = 128, trace_prompt: int | None = None,
                    max_trace_tokens: int = 48, progress=None) -> Activity:
    """Run ``prompts`` and record all streams. ``trace_prompt`` (an index) also
    keeps full per-token tensors for that one prompt."""
    layers, _ = find_layers(model)
    L = len(layers)
    cfg = model.config
    hidden = int(cfg.hidden_size)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0)

    parts = [block_parts(b) for b in layers]
    mix_dim = parts[0]["o"].weight.shape[1] if parts[0]["o"] is not None else 0

    st = {
        "resid": StreamStats(L + 1, hidden),
        "delta": StreamStats(L, hidden),
        "attn_out": StreamStats(L, hidden),
        "mlp_out": StreamStats(L, hidden),
        "attn_mix": StreamStats(L, mix_dim) if mix_dim else None,
        "mlp_act": StreamStats(L, inter) if inter else None,
    }

    cap: dict = {}
    handles = []

    def out_hook(key, i):
        def h(_m, _inp, out):
            cap[(key, i)] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return h

    def in_hook(key, i):
        def h(_m, inp):
            cap[(key, i)] = (inp[0] if isinstance(inp, (tuple, list)) else inp).detach()
        return h

    for i, p in enumerate(parts):
        if p["o"] is not None:
            handles.append(p["o"].register_forward_hook(out_hook("attn_out", i)))
            handles.append(p["o"].register_forward_pre_hook(in_hook("attn_mix", i)))
        if p["down"] is not None:
            handles.append(p["down"].register_forward_hook(out_hook("mlp_out", i)))
            handles.append(p["down"].register_forward_pre_hook(in_hook("mlp_act", i)))

    trace = None
    n_tokens = 0
    try:
        for pi, prompt in enumerate(prompts):
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
            ids = enc["input_ids"].to(device)
            n_tokens += int(ids.shape[1])
            cap.clear()
            res = model(ids, output_hidden_states=True, use_cache=False)
            hs = [h[0] for h in res.hidden_states[: L + 1]]

            for li, h in enumerate(hs):
                st["resid"].update(li, h)
            for li in range(len(hs) - 1):
                st["delta"].update(li, hs[li + 1] - hs[li])
            for li in range(L):
                for key in ("attn_out", "mlp_out", "attn_mix", "mlp_act"):
                    t = cap.get((key, li))
                    if t is not None and st[key] is not None:
                        st[key].update(li, t[0] if t.dim() == 3 else t)

            if trace_prompt is not None and pi == trace_prompt:
                T = int(ids.shape[1])
                keep = np.arange(T) if T <= max_trace_tokens else np.unique(
                    np.linspace(0, T - 1, max_trace_tokens).round().astype(int))
                kt = torch.as_tensor(keep, device=hs[0].device)

                def grab(key):
                    rows = []
                    for li in range(L):
                        t = cap.get((key, li))
                        if t is None:
                            return None
                        t = t[0] if t.dim() == 3 else t
                        rows.append(t.index_select(0, kt).to(torch.float32).cpu().numpy())
                    return np.stack(rows, axis=1)  # (T, L, C)

                resid = torch.stack(hs, 0).index_select(1, kt).to(torch.float32).cpu().numpy()
                trace = {
                    "prompt": prompt,
                    "token_index": [int(k) for k in keep],
                    "tokens": [_token_strings(tokenizer, ids[0].tolist())[k] for k in keep],
                    "resid": np.transpose(resid, (1, 0, 2)),   # (T, L+1, H)
                    "mlp_act": grab("mlp_act"),                # (T, L, I)
                    "attn_mix": grab("attn_mix"),              # (T, L, A)
                }
            if progress is not None:
                progress(pi + 1, len(prompts), prompt)
    finally:
        for h in handles:
            h.remove()

    return Activity(label=label, prompts=list(prompts), n_tokens=n_tokens,
                    stats={k: v for k, v in st.items() if v is not None}, trace=trace,
                    meta={"n_layers": L, "hidden": hidden, "intermediate": inter,
                          "mix_dim": int(mix_dim),
                          "n_heads": int(getattr(cfg, "num_attention_heads", 0) or 0),
                          "n_kv_heads": int(getattr(cfg, "num_key_value_heads", 0) or 0),
                          "head_dim": int(getattr(cfg, "head_dim", 0) or
                                          (mix_dim // max(int(getattr(cfg, "num_attention_heads", 1)), 1)))})
