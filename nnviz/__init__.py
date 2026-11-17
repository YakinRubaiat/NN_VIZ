"""nnviz -- a dynamic brain scan of a language model.

Structure first, then activity on it:

1. **Structure, from the weights.** Every MLP neuron and attention head
   dimension of every layer is a neuron. Neuron A in an early layer writes a
   vector into the residual stream; neuron B in a later layer reads it; their
   alignment is the synapse strength. That wiring diagram (the connectome) is
   laid out in 3-D so strongly wired neurons pull together. Dense regions are
   wired-up modules; sparse regions are where no clustering happened.

2. **Activity, from running it.** Every neuron's activation at every token,
   normalised to its own peak (0 = silent, 1 = full). Played token by token on
   the fixed structure: a dynamic scan.

A second view, the *layer stack*, lays depth on one axis and each layer's
channels on a plane, for looking at the same model by depth.

    import nnviz
    nnviz.scan("Qwen/Qwen3-0.6B")            # writes data/qwen3-0-6b/
    nnviz.serve("data/qwen3-0-6b")           # opens the viewer

Building blocks are importable on their own: ``build_connectome``,
``layout_connectome``, ``record_activity``, ``build_brain``, ``build_stack``.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .activity import Activity, StreamStats, record_activity
from .connectome import Connectome, build_connectome
from .model import DEFAULT_MODEL, find_layers, load_model, pick_device
from .prompts import PROMPT_SETS
from .scan import load_connectome, save_connectome, scan, slug
from .volio import dequantize, quantize, read_volume, write_volume

__all__ = [
    "__version__", "scan", "serve", "preview", "slug",
    "load_model", "pick_device", "find_layers", "DEFAULT_MODEL",
    "Connectome", "build_connectome", "save_connectome", "load_connectome",
    "layout_connectome", "build_brain", "build_stack",
    "Activity", "StreamStats", "record_activity",
    "quantize", "dequantize", "write_volume", "read_volume", "PROMPT_SETS",
]


def layout_connectome(*a, **kw):
    """See :func:`nnviz.layout.layout_connectome` (imported lazily: needs umap)."""
    from .layout import layout_connectome as f
    return f(*a, **kw)


def build_brain(*a, **kw):
    from .brain import build_brain as f
    return f(*a, **kw)


def build_stack(*a, **kw):
    from .stack import build_stack as f
    return f(*a, **kw)


def preview(scan_dir, out_dir=None, **kw):
    """Render PNG sheets of a saved scan (see :mod:`nnviz.previews`)."""
    from .previews import render_previews
    return render_previews(scan_dir, out_dir, **kw)


def serve(scan_dir=None, *, port: int = 8765, open_browser: bool = True, block: bool = True):
    """Serve the viewer with ``scan_dir`` mounted at ``/data``."""
    from .server import serve as f
    return f(scan_dir, port=port, open_browser=open_browser, block=block)
