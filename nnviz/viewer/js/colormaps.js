/* Colormaps, kept byte-identical to nnviz/colormaps.py so a PNG contact sheet
   and the 3-D view of the same volume are the same image in two media. */
(function (global) {
  "use strict";

  const CONTROL = {
    gray:   [[0,0,0],[1,1,1]],
    bone:   [[0,0,0],[0.33,0.34,0.44],[0.64,0.70,0.70],[1,1,1]],
    hot:    [[0,0,0],[0.55,0.05,0.05],[0.90,0.40,0.03],[0.99,0.82,0.25],[1,1,0.94]],
    magma:  [[0.001,0.000,0.014],[0.232,0.059,0.437],[0.551,0.161,0.506],
             [0.868,0.288,0.409],[0.993,0.624,0.427],[0.987,0.991,0.750]],
    viridis:[[0.267,0.005,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],
             [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.567,0.551],
             [0.135,0.659,0.518],[0.267,0.749,0.441],[0.478,0.821,0.318],
             [0.741,0.873,0.150],[0.993,0.906,0.144]],
    turbo:  [[0.190,0.072,0.232],[0.246,0.537,0.983],[0.098,0.902,0.717],
             [0.640,0.993,0.233],[0.976,0.740,0.146],[0.900,0.267,0.040],
             [0.480,0.016,0.011]],
    diverge:[[0.129,0.400,0.675],[0.573,0.773,0.871],[0.969,0.969,0.969],
             [0.957,0.647,0.510],[0.698,0.094,0.169]],
    parts:  [[0.38,0.65,0.98],[0.30,0.82,0.90],[0.55,0.50,0.95],[0.20,0.36,0.78],
             [0.98,0.74,0.30],[0.96,0.52,0.30],[0.80,0.24,0.28]],
    parts2: [[0.30,0.62,0.98],[0.62,0.60,0.70],[0.98,0.62,0.28]],
    // light-background activity map: stays vivid on white (no white or black end)
    ember:  [[1,1,1],[1.0,0.84,0.52],[0.98,0.52,0.16],[0.84,0.16,0.08],[0.52,0.02,0.06]]
  };

  // On a white page, maps that run dark -> white put the strongest values at
  // the background colour. In light mode those maps are flipped (or swapped
  // for ember) so strong = dark ink.
  const LIGHT_SWAP = { hot: "ember", magma: "magma_r", bone: "bone_r", gray: "gray_r" };
  function resolve(name) {
    if (global.NNColormaps && global.NNColormaps.light && LIGHT_SWAP[name]) return LIGHT_SWAP[name];
    return name;
  }
  function points(name) {
    name = resolve(name);
    if (name.endsWith("_r")) return (CONTROL[name.slice(0, -2)] || CONTROL.gray).slice().reverse();
    return CONTROL[name] || CONTROL.gray;
  }

  const ORDER = ["gray", "bone", "hot", "magma", "viridis", "turbo", "diverge", "parts", "parts2"];

  /** (n,4) RGBA Uint8Array lookup table. */
  function lut(name, n) {
    n = n || 256;
    const pts = points(name);
    const out = new Uint8Array(n * 4);
    const last = pts.length - 1;
    for (let i = 0; i < n; i++) {
      const q = (n === 1) ? 0 : i / (n - 1);
      const f = q * last;
      const a = Math.min(Math.floor(f), last);
      const b = Math.min(a + 1, last);
      const t = f - a;
      for (let c = 0; c < 3; c++) {
        const v = pts[a][c] * (1 - t) + pts[b][c] * t;
        out[i * 4 + c] = Math.round(Math.max(0, Math.min(1, v)) * 255);
      }
      out[i * 4 + 3] = 255;
    }
    return out;
  }

  /** CSS gradient for the legend ramp. */
  function cssGradient(name) {
    const pts = points(name);
    const stops = pts.map(function (p, i) {
      const pct = (i / (pts.length - 1)) * 100;
      const rgb = p.map(function (c) { return Math.round(c * 255); });
      return "rgb(" + rgb.join(",") + ") " + pct.toFixed(1) + "%";
    });
    return "linear-gradient(90deg," + stops.join(",") + ")";
  }

  /** Sample a single colour as [r,g,b] 0-255, for the 2-D canvases. */
  function sample(name, q) {
    const pts = points(name);
    const last = pts.length - 1;
    const f = Math.max(0, Math.min(1, q)) * last;
    const a = Math.min(Math.floor(f), last);
    const b = Math.min(a + 1, last);
    const t = f - a;
    return [0, 1, 2].map(function (c) {
      return Math.round((pts[a][c] * (1 - t) + pts[b][c] * t) * 255);
    });
  }

  global.NNColormaps = { CONTROL: CONTROL, ORDER: ORDER, lut: lut, cssGradient: cssGradient, sample: sample,
                         resolve: resolve, light: false };
})(window);
