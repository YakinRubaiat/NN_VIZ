/* Network view: the model's real architecture in 3-D, every weight visible.
 *
 * Layout (world units, residual width H = 1.0):
 *   z  -> depth: embedding, layer 0 ... layer L-1, final norm, output head
 *   x  -> residual-stream dimension (same axis in every panel)
 *   y  -> the matrix's other dimension (head dims / MLP neurons), downward
 *   Each layer: attention column (q, k, v, o) on the left, MLP column
 *   (gate, up, down) on the right, norm-gain strips on top of each, and a
 *   residual-stream strip above the layer that lights up with activity.
 * Panel area is proportional to parameter count.
 */
import * as THREE from "three";

const CM = window.NNColormaps;
const LZ = 0.95;          // spacing between layers
const GAP = 0.05;
const COLX = { attn: -0.58, mlp: 0.58 };
const TOP = 4.7;

const VS = /* glsl */`
out vec2 vUv;
void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const FS = /* glsl */`
out vec4 fragColor;
in vec2 vUv;
uniform sampler2D uTex, uLut, uAct, uActLut;
uniform bool uHasTex, uHasAct, uSolid;
uniform int uActAxis;                 // 0: rows (y), 1: columns (x)
uniform float uLo, uHi, uThr, uGain, uOpacity, uActGain, uDim;
void main(){
  vec2 st = vec2(vUv.x, 1.0 - vUv.y);   // row 0 at the top
  vec3 col = vec3(0.12, 0.14, 0.18); float a = 0.35;
  if (uHasTex) {
    float q = texture(uTex, st).r;
    float v = clamp((q - uLo) / max(uHi - uLo, 1e-4) * uGain, 0.0, 1.0);
    col = texture(uLut, vec2(v, 0.5)).rgb;
    a = uSolid ? 1.0 : smoothstep(uThr, min(uThr + 0.3, 1.0), v) * uOpacity;
    if (uSolid && v < uThr) col *= 0.18;
  }
  if (uHasAct) {
    float act = texture(uAct, vec2(uActAxis == 0 ? st.y : st.x, 0.5)).r;
    float on = smoothstep(0.2, 0.85, act) * uActGain;
    col = mix(col * uDim, texture(uActLut, vec2(0.3 + 0.7 * act, 0.5)).rgb, clamp(on, 0.0, 1.0));
    a = uSolid ? 1.0 : max(a * uDim, clamp(on, 0.0, 1.0));
  }
  if (!uSolid && a < 0.004) discard;
  fragColor = uSolid ? vec4(col, 1.0) : vec4(col * a, a);
}`;

function tex2D(data, w, h, nearest = true) {
  const t = new THREE.DataTexture(data, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  t.minFilter = t.magFilter = nearest ? THREE.NearestFilter : THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.unpackAlignment = 1; t.needsUpdate = true;
  return t;
}
function lutTex(name) {
  const t = new THREE.DataTexture(CM.lut(name, 256), 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.minFilter = t.magFilter = THREE.LinearFilter; t.needsUpdate = true; return t;
}
function label(text, size = 0.22, color = "#7d8693") {
  const c = document.createElement("canvas"), g = c.getContext("2d");
  const px = 48; g.font = `500 ${px}px -apple-system, Inter, sans-serif`;
  c.width = Math.ceil(g.measureText(text).width) + 16; c.height = px + 16;
  g.font = `500 ${px}px -apple-system, Inter, sans-serif`;
  g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, 8, c.height / 2);
  const t = new THREE.CanvasTexture(c); t.minFilter = THREE.LinearFilter;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: false }));
  s.scale.set(size * c.width / c.height, size, 1); s.renderOrder = 10;
  return s;
}

export class ArchScene {
  constructor(R, man, weights, norms) {
    this.R = R; this.man = man;
    this.group = new THREE.Group();
    this.meshes = [];
    this.lutName = "magma"; this.lut = lutTex("magma"); this.actLut = lutTex("hot");
    this.light = !!R.light;
    this.params = { solid: false, perPanel: false, thr: 0.18, gain: 1.0, opacity: 0.9,
                    actGain: 1.0, dim: 1.0, showAct: false };
    this.act = {};                         // activity textures per (part, layer)
    const L = man.n_layers, P = man.weights.panels;

    const mat = (extra) => new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: VS, fragmentShader: FS,
      uniforms: {
        uTex: { value: null }, uLut: { value: this.lut }, uAct: { value: null }, uActLut: { value: this.actLut },
        uHasTex: { value: false }, uHasAct: { value: false }, uSolid: { value: false }, uActAxis: { value: 0 },
        uLo: { value: 0 }, uHi: { value: 1 }, uThr: { value: 0.18 }, uGain: { value: 1 }, uOpacity: { value: 0.9 },
        uActGain: { value: 1 }, uDim: { value: 1 }, ...extra,
      },
      side: THREE.DoubleSide, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
    });

    const zOf = l => l < 0 ? -1.7 * LZ : l >= L ? (L - 1) * LZ + 1.7 * LZ : l * LZ;
    const place = (info, w, h, x, yTop, z, texture, actKey) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat());
      m.position.set(x, yTop - h / 2, z);
      const u = m.material.uniforms;
      if (texture) { u.uTex.value = texture; u.uHasTex.value = true; }
      m.userData = { ...info, w, h, actKey };
      this.group.add(m); this.meshes.push(m);
      return m;
    };

    // ---- weight panels
    const yCursor = {};
    for (const p of P) {
      const [r, c] = p.tex_shape;
      const data = weights.subarray(p.offset, p.offset + r * c);
      const t = tex2D(data, c, r);
      const info = { type: "weight", panel: p, q: data };
      if (p.kind === "embed" || p.kind === "lm_head") {
        place(info, 1.0, 4.0, 0, TOP, zOf(p.kind === "embed" ? -1 : L), t, null);
        continue;
      }
      const fam = p.family, key = `${fam}:${p.layer}`;
      const h = p.true_shape[0] / p.true_shape[1];       // rows / H  -> area ~ params
      const y = yCursor[key] ?? (TOP - 0.05 - GAP);
      const actKey = fam === "mlp" ? "mlp" : (p.kind === "k" || p.kind === "v") ? "kv" : "attn";
      const m = place(info, 1.0, h, COLX[fam], y, zOf(p.layer), t, `${actKey}:${p.layer}`);
      yCursor[key] = y - h - GAP;
    }
    // tied output head: show the embedding again at the far end, clearly marked
    if (man.meta.tied_embeddings) {
      const e = P.find(p => p.kind === "embed");
      const [r, c] = e.tex_shape;
      const t = tex2D(weights.subarray(e.offset, e.offset + r * c), c, r);
      place({ type: "weight", panel: { ...e, name: "lm_head", kind: "lm_head", note: "tied: same matrix as the embedding" },
              q: weights.subarray(e.offset, e.offset + r * c) }, 1.0, 4.0, 0, TOP, zOf(L), t, null);
    }
    // ---- norm strips
    for (const n of man.norms.strips) {
      const data = norms.subarray(n.offset, n.offset + n.tex_len);
      const t = tex2D(data, n.tex_len, 1);
      const fam = n.name === "ln_mlp" ? "mlp" : "attn";
      const x = n.name === "ln_final" ? 0 : COLX[fam];
      const z = n.name === "ln_final" ? (L - 1) * LZ + 0.8 * LZ : zOf(n.layer);
      const m = place({ type: "norm", strip: n, q: data }, 1.0, 0.05, x, TOP, z, t, null);
      m.material.uniforms.uHi.value = 1;
    }
    // ---- residual stream strips (activity only)
    for (let l = 0; l <= L; l++) {
      for (const fam of ["attn", "mlp"]) {
        if (l === L && fam === "mlp") continue;
        const x = l === L ? 0 : COLX[fam];
        const z = l === L ? (L - 1) * LZ + 0.8 * LZ : zOf(l);
        const m = place({ type: "resid", layer: l }, 1.0, 0.06, x, TOP + 0.16, z, null, `resid:${l}`);
        m.material.uniforms.uActAxis.value = 1;
      }
    }

    // ---- labels
    const lab = (t, x, y, z, s, c) => { const sp = label(t, s, c); sp.position.set(x, y, z); this.group.add(sp); };
    lab("token embedding", 0, TOP + 0.45, zOf(-1), 0.2, "#7d8693");
    lab(man.meta.tied_embeddings ? "output head (tied)" : "output head", 0, TOP + 0.45, zOf(L), 0.2, "#7d8693");
    lab("attention", COLX.attn, TOP + 0.62, zOf(0), 0.2, "#3b82f6");
    lab("MLP", COLX.mlp, TOP + 0.62, zOf(0), 0.2, "#f59e0b");
    for (let l = 0; l < L; l += 4) lab(`layer ${l}`, -1.5, TOP + 0.1, zOf(l), 0.17);
    lab(`layer ${L - 1}`, -1.5, TOP + 0.1, zOf(L - 1), 0.17);
    const k0 = {}; for (const m of this.meshes) {
      const p = m.userData.panel;
      if (p && p.layer === 0 && !k0[p.kind]) { k0[p.kind] = 1; lab(p.kind, COLX[p.family] + (p.family === "attn" ? -0.62 : 0.62), m.position.y, zOf(0), 0.16); }
    }
    this.center = new THREE.Vector3(0, TOP - 4.6, (L - 1) * LZ / 2);
    this._solid = null;
    this.apply();
  }

  /* activity textures, created on first frame */
  _actTex(key, len) {
    if (!this.act[key]) this.act[key] = tex2D(new Uint8Array(len), len, 1, false);
    return this.act[key];
  }
  setFlow(flow, frame) {
    this.flow = flow; this.flowFrame = frame;
    const lay = flow.layout;
    for (const part of ["resid", "attn", "kv", "mlp"]) {
      const { offset, rows, len } = lay[part];
      for (let l = 0; l < rows; l++) {
        const t = this._actTex(`${part}:${l}`, len);
        t.image.data.set(frame.subarray(offset + l * len, offset + (l + 1) * len));
        t.needsUpdate = true;
      }
    }
    if (!this._actBound) {
      for (const m of this.meshes) {
        const k = m.userData.actKey;
        if (k && this.act[k]) m.material.uniforms.uAct.value = this.act[k];
      }
      this._actBound = true;
      this.apply();
    }
    this.R.dirty = true;
  }

  set(p) { Object.assign(this.params, p); this.apply(); }
  apply() {
    const p = this.params;
    for (const m of this.meshes) {
      const u = m.material.uniforms, d = m.userData;
      const hasAct = p.showAct && !!u.uAct.value;
      u.uHasAct.value = hasAct;
      u.uSolid.value = p.solid;
      u.uThr.value = d.type === "weight" ? p.thr : 0.0;
      u.uGain.value = p.gain; u.uOpacity.value = p.opacity;
      u.uActGain.value = p.actGain; u.uDim.value = hasAct ? p.dim : 1.0;
      if (d.type === "weight") {
        const pp = d.panel;
        u.uLo.value = p.perPanel ? pp.q_lo / 255 : 0; u.uHi.value = p.perPanel ? pp.q_hi / 255 : 1;
      }
      if (d.type === "resid") m.visible = hasAct;
      if (this._solid !== p.solid || this._light !== this.light) {
        m.material.depthTest = p.solid; m.material.depthWrite = p.solid;
        m.material.blending = p.solid ? THREE.NormalBlending : THREE.CustomBlending;
        m.material.blendDst = this.light ? THREE.OneMinusSrcAlphaFactor : THREE.OneFactor;
        m.material.needsUpdate = true;
      }
    }
    this._solid = p.solid; this._light = this.light;
    this.R.dirty = true;
  }
  setLut(name) { this.lutName = name; this.lut.image.data.set(CM.lut(name, 256)); this.lut.needsUpdate = true; this.R.dirty = true; }
  setTheme(light) {
    this.light = light;
    this.setLut(this.lutName);
    this.actLut.image.data.set(CM.lut("hot", 256)); this.actLut.needsUpdate = true;
    this.apply();
  }

  attach() {
    const R = this.R;
    R.scene.add(this.group);
    R.volMesh.visible = false; R.edges.visible = false; R.planes.visible = false;
    R.controls.target.copy(this.center);
    R.camera.position.set(this.center.x + 19, this.center.y + 8, this.center.z - 15);
    R.controls.maxDistance = 60; R.controls.update(); R.dirty = true;
  }
  detach() {
    const R = this.R;
    R.scene.remove(this.group);
    R.volMesh.visible = R.params.showVolume; R.edges.visible = true;
    R.controls.maxDistance = 12; R.dirty = true;
  }

  /* what is under the mouse */
  pick(ndc) {
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.R.camera);
    const hits = rc.intersectObjects(this.meshes.filter(m => m.visible), false);
    if (!hits.length) return null;
    const h = hits[0], d = h.object.userData, uv = h.uv;
    const out = { ...d, u: uv.x, v: 1 - uv.y };
    if (d.type === "weight") {
      const [r, c] = d.panel.tex_shape;
      const row = Math.min(r - 1, Math.floor(out.v * r)), col = Math.min(c - 1, Math.floor(out.u * c));
      out.row = row; out.col = col; out.q = d.q[row * c + col];
    }
    if (d.type === "norm") { const n = d.strip.tex_len; out.col = Math.min(n - 1, Math.floor(out.u * n)); out.q = d.q[out.col]; }
    if (d.actKey && this.act[d.actKey] && this.params.showAct) {
      const a = this.act[d.actKey].image.data, n = a.length;
      const i = Math.min(n - 1, Math.floor((d.type === "resid" ? out.u : out.v) * n));
      out.act = a[i] / 255;
    }
    return out;
  }
}
