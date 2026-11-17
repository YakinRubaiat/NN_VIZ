# nnviz — look inside a language model

> New to this? Start with **[GUIDE.md](GUIDE.md)**, a plain-language tour with pictures.

A small library that scans a language model (default **Qwen3-0.6B**, works with
any Hugging Face decoder-only model up to a few billion parameters) and lets you
look at it in 3-D, the way you would look at a CT or MRI scan: the whole thing
at once, where it is dense and where it is empty, and what lights up while it
reads a sentence.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

nnviz scan                     # ~2.5 min the first time, ~15 s after (cached)
nnviz view                     # opens http://127.0.0.1:8765

# compare models
nnviz scan -m HuggingFaceTB/SmolLM2-360M
nnviz scan -m Qwen/Qwen3-1.7B
nnviz compare                  # side-by-side table of every scan in data/
nnviz view                     # pick the model from the dropdown
```

The viewer opens on a white background; the ☾/☀ button switches to dark (on
dark, densities glow additively; on white they are drawn like ink, with the
colour maps flipped so strong = dark).

## Three views

### Network — the real architecture, every weight visible
Every weight matrix of the model, laid out as it is computed: token embedding,
28 layers of attention (q, k, v, o) and MLP (gate, up, down) with their norm
gains, then the output head.

- **Panel area ∝ parameter count**: the MLP panels are 3× the attention ones
  because they hold 3× the parameters.
- **Brightness = weight magnitude**: each texel is the RMS of an 8×8 block of
  real weights, on **one global scale for the whole model**, so every panel is
  comparable with every other. Bright = large weights packed densely; dark and
  transparent = small.
- **The residual-stream axis is horizontal in every panel**, so a hidden
  dimension that carries large weights everywhere shows up as a line through the
  whole network.
- **Dynamic scan**: as the sentence plays, each panel's rows (head dimensions,
  MLP neurons) glow with how hard they fire.
- Hover any panel for its shape, parameter count, RMS, outlier fraction, the
  exact neurons/dimensions under the cursor, and their activity right now.

### Brain — structure from the wiring
A connectome, built like a fly-brain wiring diagram:

- **Neurons**: every MLP neuron and attention head dimension, 143,360 in total.
- **Synapses**: neuron A writes a vector into the residual stream and neuron B
  in a later layer reads one. Their alignment `|cos(w_A, r_B)|` is the synapse
  strength. All 10¹⁰ forward pairs are scored on the GPU (~35 s), and each
  neuron keeps its strongest partners.
- **Layout**: the wiring graph is laid out in 3-D with UMAP, so strongly
  connected neurons sit together. Dense regions are wired-up modules; sparse
  regions are where no clustering happened. Nothing else goes into the
  layout. Layer depth comes out as a front-to-back gradient anyway, which means
  information flow is recoverable from the wiring alone.
- **Activity on the structure**: per-token firing, typical activity, and task
  contrasts (code vs. all, math vs. all, …), shown as a volume, as individual
  neurons, or with the strongest synapses drawn as fibres.

### Layer stack — depth × channels
Depth on one axis, each layer's channels folded onto a plane (spectral ordering
plus a generalised Hilbert curve, so similar channels sit together). Residual
energy, layer contribution, burstiness, MLP duty cycle, weight norms.

## What the activity numbers mean
Each neuron's activity at a token is `|activation| / that neuron's peak over all
probe prompts`, so 0 means silent and 1 means firing as hard as it ever does. On
Qwen3-0.6B, **only ~2.2% of neurons are above 0.5 at any given token**: most
values sit near zero, and the scan is about where the near-1 values cluster.

## Python API

```python
import nnviz

nnviz.scan("Qwen/Qwen3-0.6B",
           text="Your sentence for the dynamic scan.",
           probe="all",                      # or "code", "math,story", ...
           views=("arch", "brain", "stack"))
nnviz.serve("data/qwen3-0-6b")

# building blocks
net, tok, dev = nnviz.load_model("Qwen/Qwen3-0.6B")
conn = nnviz.build_connectome(net)          # neurons + strongest synapses
xyz  = nnviz.layout_connectome(conn)        # 3-D positions from wiring
act  = nnviz.record_activity(net, tok, ["some text"], device=dev, trace_prompt=0)
```

## CLI

| command | what it does |
|---|---|
| `nnviz scan [-m MODEL] [-t TEXT] [-p PROBE] [--views arch,brain,stack] [--sheets]` | scan a model into `data/<model>/` |
| `nnviz view [DIR]` | serve the viewer (latest scan by default) |
| `nnviz info [DIR]` | summarise a scan |
| `nnviz sheets [DIR]` | PNG previews into `out/<model>/` |
| `nnviz compare [DIR...] [--markdown]` | compare scans: attention-sink streaks, sparsity, weight density, wiring |

**Supported models**: decoder-only Hugging Face models with separate
`q_proj / k_proj / v_proj / o_proj` and `gate_proj / up_proj / down_proj`
(Qwen 2/2.5/3, Llama, Mistral, SmolLM, TinyLlama, ...). Models with fused
projections (Phi-3, GPT-NeoX/Pythia) or post-norm blocks (OLMo-2) need an adapter
in `nnviz/model.py`.

## Layout

```
nnviz/
  model.py        load any HF causal LM, find its blocks
  activity.py     forward hooks: streaming per-neuron stats + per-token traces
  arch.py         network view: pooled weight panels + activity flow
  connectome.py   brain view: neurons, synapse scoring on GPU
  layout.py       brain view: 3-D layout of the wiring (UMAP, cached)
  brain.py        brain view: voxelise, paint maps, dynamic frames
  voxelize.py     point cloud -> voxel grid (trilinear splat + Gaussian PSF)
  stack.py        layer-stack view (with curve.py, ordering.py)
  volio.py        8-bit volume format (linear / asinh / signed windows)
  scan.py         one call that runs everything
  viewer/         Three.js viewer (three r160 vendored, works offline)
```

## Caveats
- The brain layout uses UMAP: **neighbourhoods are meaningful, global distances
  are not**. "These neurons are wired together" is a real statement; "this lobe
  is twice as far as that one" is not.
- Network-view texels are block averages (RMS of 8×8 weights; the embedding
  averages ~300 tokens per row). A single extreme weight is diluted, so use the
  hover readout's `max |w|` for extremes.
- Activity comes from the probe prompts in `nnviz/prompts.py` plus one traced
  sentence. A different probe set changes "typical" activity, which is the
  point of the task-contrast maps.
