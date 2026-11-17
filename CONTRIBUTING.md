# Contributing to nnviz

Thanks for your interest! All kinds of help are welcome: bug reports, new model
adapters, viewer improvements, documentation, and research ideas.

## Get set up

```bash
git clone https://github.com/YakinRubaiat/NN_VIZ.git && cd NN_VIZ
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

nnviz scan        # scan Qwen3-0.6B into data/
nnviz view        # check the viewer at http://127.0.0.1:8765
```

## Making a change

1. Open an [issue](https://github.com/YakinRubaiat/NN_VIZ/issues) first for
   anything larger than a small fix, so we can agree on the approach.
2. Create a branch from `main` (for example `feat/phi3-adapter` or
   `fix/stack-legend`).
3. Keep the change focused. If it touches the viewer, include a screenshot in
   the pull request.
4. Open a pull request against `main` that describes what changed and how you
   checked it.

## Good first contributions

- **Model adapters** in `nnviz/model.py` for fused projections (Phi-3,
  GPT-NeoX / Pythia) or post-norm blocks (OLMo-2).
- **Comparisons**: scan a new open model and share the `nnviz compare` output.
- **Docs**: make [GUIDE.md](GUIDE.md) clearer for people new to AI.

## Contact

For collaboration, research ideas, or anything else, email
**Yakin Rubaiat** at [yakin.rubaiat@gmail.com](mailto:yakin.rubaiat@gmail.com).

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
