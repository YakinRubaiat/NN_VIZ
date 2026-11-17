"""Model loading and architecture discovery, shared by every view."""

from __future__ import annotations

import torch

__all__ = ["DEFAULT_MODEL", "pick_device", "load_model", "find_layers", "block_parts"]

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def pick_device(spec: str = "auto") -> torch.device:
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(name: str = DEFAULT_MODEL, *, device: str = "auto", dtype: str = "float32",
               trust_remote_code: bool = False, verbose: bool = True):
    """Return ``(model, tokenizer, device)`` for any HF decoder-only causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    tdtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}.get(dtype, torch.float32)
    if verbose:
        print(f"[nnviz] loading {name} -> {dev} ({dtype})", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    net = AutoModelForCausalLM.from_pretrained(name, dtype=tdtype, attn_implementation="eager",
                                               trust_remote_code=trust_remote_code)
    net.eval().to(dev)
    return net, tok, dev


def find_layers(model: torch.nn.Module):
    """Locate the decoder block list of a HF causal LM (Qwen/Llama/Mistral...)."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (torch.nn.ModuleList, list)):
            return obj, path
    raise RuntimeError("could not locate decoder layers on this model")


def block_parts(blk) -> dict:
    """Named sub-modules of one decoder block (missing ones are ``None``)."""
    attn = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
    mlp = getattr(blk, "mlp", None) or getattr(blk, "feed_forward", None)
    g = lambda m, n: getattr(m, n, None) if m is not None else None  # noqa: E731
    return {
        "attn": attn, "mlp": mlp,
        "q": g(attn, "q_proj"), "k": g(attn, "k_proj"), "v": g(attn, "v_proj"),
        "o": g(attn, "o_proj"),
        "gate": g(mlp, "gate_proj"), "up": g(mlp, "up_proj"), "down": g(mlp, "down_proj"),
        "ln_attn": getattr(blk, "input_layernorm", None),
        "ln_mlp": getattr(blk, "post_attention_layernorm", None),
    }
