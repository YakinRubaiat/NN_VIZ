"""The model's wiring diagram, read straight from its weights.

Like a fly-brain connectome, this is a graph of neurons and synapses:

neurons
    * every MLP neuron                      (reads via gate/up, writes via down)
    * every attention head dimension (h,d)  (reads via q/k/v, writes via o)
    Qwen3-0.6B: 28 x (3072 + 2048) = 143,360 neurons.

synapses
    Neuron A writes a fixed vector ``w_A`` into the residual stream; neuron B
    at a later position reads it through ``r_B`` (RMSNorm gain folded in). The
    stream is a sum, so A's contribution to B is proportional to
    ``cos(w_A, r_B)`` -- the "virtual weight" between them. Only forward pairs
    count: A must sit earlier in the computation than B.

All ~2e10 candidate pairs are scored on the GPU in blocks and only the
strongest partners of each neuron are kept, giving a sparse graph. Laid out in
3-D (``layout.py``), strongly wired neurons pull together into dense modules,
and weakly wired neurons drift into sparse territory -- the structure comes
from the model itself, not from a layout we imposed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import block_parts, find_layers

__all__ = ["Connectome", "build_connectome", "FAMILIES"]

FAMILIES = {0: "attention", 1: "mlp"}


@dataclass
class Connectome:
    """Neurons, their attributes and their strongest synapses."""

    # per-neuron attributes (row-aligned, attention neurons first, then MLP)
    family: np.ndarray        # (N,) int8   0 attention, 1 mlp
    layer: np.ndarray         # (N,) int16
    index: np.ndarray         # (N,) int32  neuron id (MLP) or h*head_dim+d (attention)
    head: np.ndarray          # (N,) int16  attention head, -1 for MLP
    write_norm: np.ndarray    # (N,) float32
    read_norm: np.ndarray     # (N,) float32
    # sparse wiring: for every neuron its k strongest partners (either direction)
    nbr: np.ndarray           # (N, k) int64, -1 = none
    strength: np.ndarray      # (N, k) float32 |cos|
    out_strength: np.ndarray  # (N,) mean |cos| of the k strongest targets
    in_strength: np.ndarray   # (N,) mean |cos| of the k strongest sources
    # globally strongest synapses, for drawing fibre tracts
    edges: np.ndarray         # (E, 2) int64 source -> target
    edge_w: np.ndarray        # (E,) float32
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.family.shape[0])


def _pos(family: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Order in the computation: attention(l) = 2l, mlp(l) = 2l+1."""
    return 2 * layer.astype(np.int64) + family.astype(np.int64)


def _unit(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def build_connectome(model, *, k: int = 12, n_edges: int = 40000, device=None,
                     block: int = 1024, verbose: bool = True) -> Connectome:
    """Score every forward write->read pair and keep the strongest synapses."""
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    layers, _ = find_layers(model)
    L = len(layers)
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    p0 = block_parts(layers[0])
    A = int(p0["o"].weight.shape[1])
    hd = int(getattr(cfg, "head_dim", 0) or A // n_heads)
    grp = n_heads // n_kv
    inter = int(p0["down"].weight.shape[1])
    f32 = torch.float32
    cpu = torch.device("cpu")

    def host(t):
        return t.detach().to(cpu, f32)

    # ---- neurons: one write vector + a few read vectors each -----------------
    W_attn, Rq, Rk, Rv, W_mlp, Rg, Ru = [], [], [], [], [], [], []
    for blk in layers:
        p = block_parts(blk)
        ga = host(p["ln_attn"].weight)
        gm = host(p["ln_mlp"].weight)
        W_attn.append(host(p["o"].weight).T)                     # (A, H)
        q = host(p["q"].weight) * ga                             # (A, H)
        kk = host(p["k"].weight) * ga                            # (n_kv*hd, H)
        vv = host(p["v"].weight) * ga
        # node (h, d) reads k/v row (h // grp, d)
        kv_idx = torch.tensor([(h // grp) * hd + d for h in range(n_heads) for d in range(hd)])
        Rq.append(q); Rk.append(kk[kv_idx]); Rv.append(vv[kv_idx])         # noqa: E702
        W_mlp.append(host(p["down"].weight).T)                   # (I, H)
        Rg.append(host(p["gate"].weight) * gm)
        Ru.append(host(p["up"].weight) * gm)

    W = torch.cat([torch.cat(W_attn), torch.cat(W_mlp)])                    # (N, H)
    n_attn, n_mlp = L * A, L * inter
    N = n_attn + n_mlp
    family = np.r_[np.zeros(n_attn, np.int8), np.ones(n_mlp, np.int8)]
    layer = np.r_[np.repeat(np.arange(L), A), np.repeat(np.arange(L), inter)].astype(np.int16)
    index = np.r_[np.tile(np.arange(A), L), np.tile(np.arange(inter), L)].astype(np.int32)
    head = np.r_[np.tile(np.arange(A) // hd, L), -np.ones(n_mlp)].astype(np.int16)
    write_norm = W.norm(dim=1).numpy().astype(np.float32)

    # read sets: attention neurons read via q, k, v; MLP neurons via gate, up.
    # A synapse's strength is the best alignment over the target's read vectors.
    reads_attn = [torch.cat(Rq), torch.cat(Rk), torch.cat(Rv)]
    reads_mlp = [torch.cat(Rg), torch.cat(Ru)]
    read_norm = np.r_[torch.stack([r.norm(dim=1) for r in reads_attn]).amax(0).numpy(),
                      torch.stack([r.norm(dim=1) for r in reads_mlp]).amax(0).numpy()
                      ].astype(np.float32)

    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    dt = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    Wd = _unit(W).to(device, dt)
    RA = [_unit(r).to(device, dt) for r in reads_attn]
    RM = [_unit(r).to(device, dt) for r in reads_mlp]
    pos = torch.as_tensor(_pos(family, layer), device=device)
    say(f"[nnviz] connectome: {N:,} neurons ({n_attn:,} attention dims, {n_mlp:,} MLP), "
        f"scoring {N * N / 2:.2e} forward pairs on {device.type}")

    # ---- score all pairs block by block, keeping top-k both ways --------------
    out_idx = torch.empty((N, k), dtype=torch.int64)
    out_val = torch.empty((N, k), dtype=torch.float32)
    in_val = torch.zeros((k, N), device=device, dtype=torch.float32)
    in_idx = torch.full((k, N), -1, device=device, dtype=torch.int64)
    t0 = time.time()
    n_blocks = math.ceil(N / block)
    for bi in range(n_blocks):
        s, e = bi * block, min(N, (bi + 1) * block)
        wb = Wd[s:e]
        sa = torch.stack([(wb @ r.T).abs() for r in RA]).amax(0)            # (b, n_attn)
        sm = torch.stack([(wb @ r.T).abs() for r in RM]).amax(0)            # (b, n_mlp)
        S = torch.cat([sa, sm], dim=1).float()                              # (b, N)
        S.masked_fill_(pos[s:e, None] >= pos[None, :], 0.0)                 # forward only
        v, i = S.topk(k, dim=1)
        out_val[s:e], out_idx[s:e] = v.cpu(), i.cpu()
        bv, bidx = S.topk(min(k, e - s), dim=0)                             # best sources per target
        cv = torch.cat([in_val, bv]); ci = torch.cat([in_idx, bidx + s])
        top = cv.topk(k, dim=0)
        in_val = top.values
        in_idx = ci.gather(0, top.indices)
        if verbose and (bi % 20 == 0 or bi == n_blocks - 1):
            print(f"\r[nnviz]   block {bi + 1}/{n_blocks}  {time.time() - t0:5.1f}s", end="",
                  flush=True)
    if verbose:
        print()
    in_val, in_idx = in_val.T.cpu(), in_idx.T.cpu()                         # (N, k)

    # chance level: |cos| of random unit vectors in H dims, best of N candidates
    H = W.shape[1]
    chance = float(np.sqrt(2.0 * np.log(N)) / np.sqrt(H))

    # ---- merge each neuron's sources and targets into one neighbour list -----
    cand_i = torch.cat([out_idx, in_idx], 1)
    cand_v = torch.cat([out_val, in_val], 1)
    cand_v = torch.where(cand_i < 0, torch.zeros_like(cand_v), cand_v)
    order = cand_v.argsort(dim=1, descending=True)
    cand_i, cand_v = cand_i.gather(1, order), cand_v.gather(1, order)
    nbr = np.full((N, 2 * k), -1, dtype=np.int64)
    stv = np.zeros((N, 2 * k), dtype=np.float32)
    ci_np, cv_np = cand_i.numpy(), cand_v.numpy()
    for r in range(N):                       # dedupe (a pair can appear both ways)
        _, first = np.unique(ci_np[r], return_index=True)
        keep = np.sort(first)
        keep = keep[(ci_np[r][keep] >= 0) & (cv_np[r][keep] > 0)]
        nbr[r, : keep.size] = ci_np[r][keep]
        stv[r, : keep.size] = cv_np[r][keep]

    # ---- strongest synapses overall, for tracts ------------------------------
    src = np.repeat(np.arange(N), k)
    tgt = out_idx.numpy().ravel()
    w = out_val.numpy().ravel()
    ok = w > 0
    src, tgt, w = src[ok], tgt[ok], w[ok]
    top = np.argsort(-w)[:n_edges]
    edges = np.stack([src[top], tgt[top]], 1)

    out_s = out_val.numpy().mean(1).astype(np.float32)
    in_s = in_val.numpy().mean(1).astype(np.float32)
    meta = {"n_neurons": int(N), "n_attention": int(n_attn), "n_mlp": int(n_mlp),
            "k": int(k), "chance_cos": chance, "hidden": int(H), "head_dim": hd,
            "n_heads": n_heads, "n_kv_heads": n_kv, "n_layers": L,
            "mean_top_cos": float(out_val[:, 0].mean()), "seconds": round(time.time() - t0, 1)}
    say(f"[nnviz]   strongest-partner |cos| mean {meta['mean_top_cos']:.3f} vs chance "
        f"{chance:.3f}  ({meta['mean_top_cos'] / chance:.1f}x)  in {meta['seconds']}s")
    return Connectome(family=family, layer=layer, index=index, head=head,
                      write_norm=write_norm, read_norm=read_norm, nbr=nbr, strength=stv,
                      out_strength=out_s, in_strength=in_s, edges=edges,
                      edge_w=w[top].astype(np.float32), meta=meta)
