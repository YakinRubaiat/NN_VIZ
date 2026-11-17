"""Colormaps shared by the PNG previews and the WebGL viewer.

Each map is a short list of RGB control points, linearly interpolated. They are
deliberately the radiology set rather than a decorative set:

* ``gray``   -- the neutral default; judge structure, not colour.
* ``bone``   -- cool-tinted grayscale, the CT look.
* ``hot``    -- black-red-yellow-white, the PET/activation look.
* ``magma``  -- perceptually uniform, good for print.
* ``turbo``  -- high-contrast rainbow; useful for finding faint structure, bad
                for judging magnitude. Use it to explore, not to conclude.
* ``diverge``-- blue-white-red, for contrast maps where zero means "no change".
* ``parts``  -- categorical, one colour per unit kind (for the parts map).
"""

from __future__ import annotations

import numpy as np

CONTROL_POINTS: dict[str, list[tuple[float, float, float]]] = {
    "gray":   [(0, 0, 0), (1, 1, 1)],
    "bone":   [(0, 0, 0), (0.33, 0.34, 0.44), (0.64, 0.70, 0.70), (1, 1, 1)],
    "hot":    [(0, 0, 0), (0.55, 0.05, 0.05), (0.90, 0.40, 0.03),
               (0.99, 0.82, 0.25), (1, 1, 0.94)],
    "magma":  [(0.001, 0.000, 0.014), (0.232, 0.059, 0.437), (0.551, 0.161, 0.506),
               (0.868, 0.288, 0.409), (0.993, 0.624, 0.427), (0.987, 0.991, 0.750)],
    "viridis": [(0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
                (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
                (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
                (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)],
    "turbo":  [(0.190, 0.072, 0.232), (0.246, 0.537, 0.983), (0.098, 0.902, 0.717),
               (0.640, 0.993, 0.233), (0.976, 0.740, 0.146), (0.900, 0.267, 0.040),
               (0.480, 0.016, 0.011)],
    "diverge": [(0.129, 0.400, 0.675), (0.573, 0.773, 0.871), (0.969, 0.969, 0.969),
                (0.957, 0.647, 0.510), (0.698, 0.094, 0.169)],
    "parts": [(0.38, 0.65, 0.98), (0.30, 0.82, 0.90), (0.55, 0.50, 0.95), (0.20, 0.36, 0.78),
              (0.98, 0.74, 0.30), (0.96, 0.52, 0.30), (0.80, 0.24, 0.28)],
    # two-tissue map: attention (cool) <-> MLP (warm)
    "parts2": [(0.30, 0.62, 0.98), (0.62, 0.60, 0.70), (0.98, 0.62, 0.28)],
}

ORDER = ["gray", "bone", "hot", "magma", "viridis", "turbo", "diverge", "parts", "parts2"]


def lut(name: str, n: int = 256) -> np.ndarray:
    """Return an ``(n, 3)`` uint8 lookup table."""
    pts = np.array(CONTROL_POINTS[name], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, pts.shape[0])
    q = np.linspace(0.0, 1.0, n)
    rgb = np.stack([np.interp(q, xs, pts[:, c]) for c in range(3)], axis=1)
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def as_js() -> str:
    """Emit the control points as a JS object literal, for the viewer."""
    import json

    return json.dumps({k: [[round(c, 4) for c in p] for p in v]
                       for k, v in CONTROL_POINTS.items()}, indent=1)
