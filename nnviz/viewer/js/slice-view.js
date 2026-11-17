/* Orthogonal slice views (axial / coronal / sagittal), drawn on the CPU from
 * the same byte volumes the GPU renders. Volumes may differ in resolution;
 * everything is addressed in normalised 0..1 coordinates, so a coarse overlay
 * lands on a fine anatomy correctly.
 *
 *   axial     constant z   x -> right, y -> down
 *   coronal   constant y   x -> right, z -> up
 *   sagittal  constant x   y -> right, z -> up
 */
(function (global) {
  "use strict";
  const CM = global.NNColormaps;

  function sampler(vol) {
    if (!vol) return null;
    const [nx, ny, nz] = vol.dims, b = vol.bytes;
    return (u, v, w) => {
      const x = Math.min(nx - 1, Math.max(0, Math.floor(u * nx)));
      const y = Math.min(ny - 1, Math.max(0, Math.floor(v * ny)));
      const z = Math.min(nz - 1, Math.max(0, Math.floor(w * nz)));
      return b[(z * ny + y) * nx + x];
    };
  }

  function lutCache(name) {
    const t = CM.lut(name, 256), out = new Array(256);
    for (let i = 0; i < 256; i++) out[i] = [t[i * 4], t[i * 4 + 1], t[i * 4 + 2]];
    return out;
  }

  class SliceView {
    constructor(canvases, onPick) {
      this.c = canvases; this.onPick = onPick;
      this.cursor = [0.5, 0.5, 0.5];
      this.extent = [1, 1, 1];
      ["ax", "co", "sa"].forEach(k => this._bind(k));
    }

    _bind(k) {
      const cv = this.c[k]; let down = false;
      const pick = e => {
        const r = cv.getBoundingClientRect();
        const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
        const g = this._geom[k]; if (!g) return;
        const u = (fx - g.ox) / g.w, v = (fy - g.oy) / g.h;
        if (u < 0 || u > 1 || v < 0 || v > 1) return;
        const c = this.cursor.slice();
        if (k === "ax") { c[0] = u; c[1] = v; }
        if (k === "co") { c[0] = u; c[2] = 1 - v; }
        if (k === "sa") { c[1] = u; c[2] = 1 - v; }
        this.onPick(c);
      };
      cv.addEventListener("pointerdown", e => { down = true; cv.setPointerCapture(e.pointerId); pick(e); });
      cv.addEventListener("pointermove", e => { if (down) pick(e); });
      cv.addEventListener("pointerup", e => { down = false; cv.releasePointerCapture(e.pointerId); });
    }

    /** layers: {shape, paint, over} with {bytes, dims}; style: luts, window, overlay settings */
    draw(layers, style) {
      this.layers = layers; this.style = style;
      const sh = sampler(layers.shape), pa = sampler(layers.paint), ov = sampler(layers.over);
      const lp = lutCache(style.paintCmap), lo = style.overCmap ? lutCache(style.overCmap) : null;
      const wlo = style.level - style.width / 2, ww = Math.max(style.width, 1e-5);
      const c = this.cursor;
      const e = this.extent;
      const views = {
        ax: { W: e[0], H: e[1], at: (u, v) => [u, v, c[2]] },
        co: { W: e[0], H: e[2], at: (u, v) => [u, c[1], 1 - v] },
        sa: { W: e[1], H: e[2], at: (u, v) => [c[0], u, 1 - v] },
      };
      this._geom = {};
      for (const k of ["ax", "co", "sa"]) {
        const cv = this.c[k], ctx = cv.getContext("2d");
        const { W, H, at } = views[k];
        const res = 160;
        const s = Math.min(res / W, res / H);
        const pw = Math.max(1, Math.round(W * s)), ph = Math.max(1, Math.round(H * s));
        const img = ctx.createImageData(pw, ph), d = img.data;
        for (let j = 0; j < ph; j++) for (let i = 0; i < pw; i++) {
          const p = at((i + 0.5) / pw, (j + 0.5) / ph);
          const raw = sh ? sh(p[0], p[1], p[2]) / 255 : 0;
          const sv = Math.min(1, Math.max(0, (raw - wlo) / ww));
          let col;
          const bg = style.light ? 255 : 0;
          if (pa) { const q = lp[pa(p[0], p[1], p[2])]; col = [bg + (q[0] - bg) * sv, bg + (q[1] - bg) * sv, bg + (q[2] - bg) * sv]; }
          else { const q = lp[Math.round(sv * 255)]; col = [q[0], q[1], q[2]]; }
          if (ov && lo) {
            const o = ov(p[0], p[1], p[2]) / 255;
            let amt, oc;
            if (style.overSigned) { const sg = (o - 0.5) * 2; amt = Math.abs(sg); oc = lo[Math.round((0.5 + 0.5 * sg) * 255)]; }
            else { amt = o; oc = lo[Math.round(o * 255)]; }
            const t = Math.min(1, Math.max(0, (amt - style.overThr) / 0.25)) * style.overGain;
            const k2 = Math.min(1, t);
            col = [col[0] * (1 - k2) + oc[0] * k2, col[1] * (1 - k2) + oc[1] * k2, col[2] * (1 - k2) + oc[2] * k2];
          }
          const o4 = (j * pw + i) * 4;
          d[o4] = col[0]; d[o4 + 1] = col[1]; d[o4 + 2] = col[2]; d[o4 + 3] = 255;
        }
        // fit into the square canvas, keeping aspect
        const CW = cv.width, CH = cv.height;
        const fit = Math.min(CW / pw, CH / ph);
        const dw = pw * fit, dh = ph * fit, ox = (CW - dw) / 2, oy = (CH - dh) / 2;
        ctx.fillStyle = style.light ? "#fff" : "#000"; ctx.fillRect(0, 0, CW, CH);
        const off = this._off || (this._off = document.createElement("canvas"));
        off.width = pw; off.height = ph; off.getContext("2d").putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(off, ox, oy, dw, dh);
        this._geom[k] = { ox: ox / CW, oy: oy / CH, w: dw / CW, h: dh / CH };
        // crosshair
        let cx, cy;
        if (k === "ax") { cx = c[0]; cy = c[1]; }
        if (k === "co") { cx = c[0]; cy = 1 - c[2]; }
        if (k === "sa") { cx = c[1]; cy = 1 - c[2]; }
        ctx.strokeStyle = "rgba(79,209,197,0.55)"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ox + cx * dw, oy); ctx.lineTo(ox + cx * dw, oy + dh);
        ctx.moveTo(ox, oy + cy * dh); ctx.lineTo(ox + dw, oy + cy * dh);
        ctx.stroke();
      }
    }

    /** raw bytes of each layer at the cursor */
    probe() {
      const c = this.cursor, L = this.layers || {}, out = {};
      for (const k of ["shape", "paint", "over"]) { const s = sampler(L[k]); out[k] = s ? s(c[0], c[1], c[2]) : null; }
      return out;
    }
  }

  global.SliceView = SliceView;
})(window);
