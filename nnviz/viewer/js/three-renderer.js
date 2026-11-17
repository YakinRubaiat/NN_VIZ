/* Three.js scene for nnviz scans.
 *
 *   volume   a box mesh whose fragment shader ray-marches three 3-D textures:
 *              shape   -> opacity and shading (the structure)
 *              paint   -> colour (layer depth, attention/MLP, wiring); optional
 *              overlay -> activity, drawn where it passes threshold
 *   fibres   THREE.LineSegments, the strongest synapses, additive
 *   neurons  THREE.Points, one per neuron, optionally coloured by live activity
 *   planes   three orthogonal cut planes through the cursor
 *
 * Volume modes: 0 Volume (emission/absorption) | 1 MIP | 2 Surface (iso, with
 * the activity just beneath it projected onto it) | 3 X-ray (mean).
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CM = window.NNColormaps;

/* ---------------------------------------------------------------- shaders */
const VOL_VS = /* glsl */`
out vec3 vLocal;
void main() {
  vLocal = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const VOL_FS = /* glsl */`
out vec4 fragColor;
precision highp sampler3D;
in vec3 vLocal;
uniform sampler3D uShape, uPaint, uOver;
uniform sampler2D uLutPaint, uLutOver;
uniform bool  uHasPaint, uHasOver, uOverSigned, uLight;
uniform vec3  uEye, uBox, uClipLo, uClipHi, uInk;
uniform int   uMode;
uniform float uLevel, uWidth, uDensity, uCutoff, uIso, uSteps, uOverThr, uOverGain;

float win(float v){ float lo = uLevel - uWidth*0.5; return clamp((v-lo)/max(uWidth,1e-5),0.0,1.0); }
vec3  tc(vec3 p){ return (p / uBox) * 0.5 + 0.5; }
float shapeAt(vec3 t){ return texture(uShape, t).r; }
vec3  lutP(float v){ return texture(uLutPaint, vec2(clamp(v,0.0,1.0),0.5)).rgb; }
vec3  lutO(float v){ return texture(uLutOver,  vec2(clamp(v,0.0,1.0),0.5)).rgb; }
vec3  paintAt(vec3 t, float s){ return uHasPaint ? lutP(texture(uPaint, t).r) : lutP(s); }

float overAt(vec3 t, out vec3 col){
  float o = texture(uOver, t).r;
  if (uOverSigned) {
    float s = (o - 0.5) * 2.0;
    col = lutO(0.5 + 0.5 * s);
    return smoothstep(uOverThr, min(uOverThr + 0.25, 1.0), abs(s));
  }
  col = lutO(o);
  return smoothstep(uOverThr, min(uOverThr + 0.25, 1.0), o);
}
vec3 grad(vec3 t, vec3 h){
  return vec3(shapeAt(t+vec3(h.x,0,0))-shapeAt(t-vec3(h.x,0,0)),
              shapeAt(t+vec3(0,h.y,0))-shapeAt(t-vec3(0,h.y,0)),
              shapeAt(t+vec3(0,0,h.z))-shapeAt(t-vec3(0,0,h.z)));
}

void main(){
  vec3 dir = normalize(vLocal - uEye);
  vec3 bmin = -uBox + 2.0 * uBox * uClipLo, bmax = -uBox + 2.0 * uBox * uClipHi;
  vec3 inv = 1.0 / dir, ta = (bmin - uEye) * inv, tb = (bmax - uEye) * inv;
  vec3 tsm = min(ta, tb), tbg = max(ta, tb);
  float t0 = max(max(tsm.x, tsm.y), max(tsm.z, 0.0)), t1 = min(min(tbg.x, tbg.y), tbg.z);
  if (t1 <= t0) discard;
  float dt = (t1 - t0) / uSteps;
  vec3 h = vec3(1.0) / vec3(textureSize(uShape, 0));
  vec3 L = normalize(vec3(0.5, 0.8, 0.45));
  vec3 oc;

  if (uMode == 1) {                                          // MIP
    float m = 0.0; vec3 mt = vec3(0.5); float om = 0.0; vec3 ocol = vec3(0.0);
    for (float i = 0.0; i < 1024.0; i += 1.0) {
      if (i >= uSteps) break;
      vec3 t = tc(uEye + dir * (t0 + (i + 0.5) * dt));
      float s = win(shapeAt(t));
      if (s > m) { m = s; mt = t; }
      if (uHasOver) { float a = overAt(t, oc); if (a > om) { om = a; ocol = oc; } }
    }
    float a = m > uCutoff ? clamp(m * uDensity * 1.5, 0.0, 1.0) : 0.0;
    float k = clamp(om * uOverGain, 0.0, 1.0);
    float A = max(a, k);
    if (A <= 0.0) discard;
    fragColor = vec4(paintAt(mt, m) * a * (1.0 - k) + ocol * k, A);
    return;
  }
  if (uMode == 3) {                                          // X-ray
    float acc = 0.0, oa = 0.0, n = 0.0; vec3 ocol = vec3(0.0);
    for (float i = 0.0; i < 1024.0; i += 1.0) {
      if (i >= uSteps) break;
      vec3 t = tc(uEye + dir * (t0 + (i + 0.5) * dt));
      acc += win(shapeAt(t));
      if (uHasOver) { float a = overAt(t, oc); oa += a; ocol += oc * a; }
      n += 1.0;
    }
    float v = clamp(acc / max(n, 1.0) * uDensity * 2.5, 0.0, 1.0);
    vec3 base = uHasPaint ? uInk : lutP(v);
    float k = clamp(oa / max(n, 1.0) * 6.0 * uOverGain, 0.0, 1.0);
    float A = max(v, k);
    if (A <= 0.0) discard;
    fragColor = vec4(base * v * (1.0 - k) + (ocol / max(oa, 1e-5)) * k, A);
    return;
  }
  if (uMode == 2) {                                          // surface
    for (float i = 0.0; i < 1024.0; i += 1.0) {
      if (i >= uSteps) break;
      float tt = t0 + (i + 0.5) * dt;
      if (win(shapeAt(tc(uEye + dir * tt))) >= uIso) {
        float lo = tt - dt, hi = tt;
        for (int k = 0; k < 5; k++) {
          float mid = 0.5 * (lo + hi);
          if (win(shapeAt(tc(uEye + dir * mid))) >= uIso) hi = mid; else lo = mid;
        }
        vec3 t = tc(uEye + dir * hi);
        vec3 g = grad(t, h); float gl = length(g);
        vec3 n = gl > 1e-6 ? -g / gl : -dir;
        float diff = max(dot(n, L), 0.0);
        float spec = pow(max(dot(reflect(-L, n), -dir), 0.0), 28.0);
        float rim = pow(1.0 - max(dot(n, -dir), 0.0), 2.5);
        vec3 base = uHasPaint ? paintAt(t, uIso) : vec3(0.78, 0.80, 0.84);
        vec3 col = base * (0.25 + 0.75 * diff) + vec3(spec * 0.3) + base * rim * 0.35;
        if (uHasOver) {
          float om = 0.0; vec3 ocol = vec3(0.0);
          for (float j = 0.0; j < 40.0; j += 1.0) {
            float a = overAt(tc(uEye + dir * (hi + j * dt * 0.5)), oc);
            if (a > om) { om = a; ocol = oc; }
          }
          col = mix(col, ocol * (0.55 + 0.45 * diff) + ocol * 0.35, clamp(om * uOverGain, 0.0, 1.0));
        }
        fragColor = vec4(col, 1.0);
        return;
      }
    }
    discard;
  }
  // 0: emission / absorption
  vec3 acc = vec3(0.0); float alpha = 0.0;
  float norm = 220.0 * dt / max(length(uBox), 1e-4);
  for (float i = 0.0; i < 1024.0; i += 1.0) {
    if (i >= uSteps || alpha > 0.995) break;
    vec3 t = tc(uEye + dir * (t0 + (i + 0.5) * dt));
    float s = win(shapeAt(t));
    float sa = 0.0; vec3 col = vec3(0.0);
    if (s > uCutoff) {
      float q = (s - uCutoff) / max(1.0 - uCutoff, 1e-5);
      sa = clamp((uLight ? q * q * q * 0.6 : q * q) * uDensity * norm, 0.0, 1.0);
      col = paintAt(t, s);
      vec3 g = grad(t, h); float gl = length(g);
      if (gl > 1e-5) col *= 0.6 + 0.55 * max(dot(-g / gl, L), 0.0);
    }
    float oa = 0.0; oc = vec3(0.0);
    if (uHasOver) oa = clamp(overAt(t, oc) * uOverGain * norm * 3.0, 0.0, 1.0);
    float a = sa + oa - sa * oa;
    if (a <= 0.0) continue;
    vec3 c = (col * sa + oc * oa * 1.6) / max(sa + oa, 1e-5);
    acc += (1.0 - alpha) * a * c;
    alpha += (1.0 - alpha) * a;
  }
  if (alpha <= 0.0) discard;
  fragColor = vec4(acc, alpha);
}`;

const PT_VS = /* glsl */`
in vec4 aAttr; in float aAct;
uniform vec3 uBox, uClipLo, uClipHi;
uniform float uSize, uNL, uThr; uniform int uColorBy; uniform bool uHasAct;
uniform sampler2D uLut, uLutAct;
out vec3 vCol; out float vA; out float vClip;
void main(){
  vec3 t = (position + uBox) / (2.0 * uBox);
  vClip = (any(lessThan(t, uClipLo)) || any(greaterThan(t, uClipHi))) ? 1.0 : 0.0;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float size = uSize;
  if (uColorBy == 3 && uHasAct) {
    float a = aAct / 255.0;
    float on = smoothstep(uThr - 0.15, 1.0, a);
    vCol = mix(vec3(0.18, 0.22, 0.30), texture(uLutAct, vec2(0.35 + 0.65 * a, 0.5)).rgb, on);
    vA = mix(0.04, 1.0, on);
    size *= mix(0.8, 2.6, on);
  } else {
    float v = uColorBy == 0 ? aAttr.x / max(uNL - 1.0, 1.0) : uColorBy == 1 ? aAttr.y : aAttr.z / 255.0;
    vCol = texture(uLut, vec2(v, 0.5)).rgb; vA = 0.55;
  }
  gl_PointSize = max(size / max(-mv.z, 0.05), 1.0);
}`;
const PT_FS = /* glsl */`
out vec4 fragColor;
in vec3 vCol; in float vA; in float vClip;
void main(){
  if (vClip > 0.5) discard;
  vec2 d = gl_PointCoord * 2.0 - 1.0; float r = dot(d, d);
  if (r > 1.0) discard;
  float a = vA * (1.0 - r);
  fragColor = vec4(vCol * a, a);
}`;

const LN_VS = /* glsl */`
in float aW; in float aLayer;
uniform vec3 uBox, uClipLo, uClipHi; uniform float uNL;
out float vW; out float vL; out float vClip;
void main(){
  vec3 t = (position + uBox) / (2.0 * uBox);
  vClip = (any(lessThan(t, uClipLo)) || any(greaterThan(t, uClipHi))) ? 1.0 : 0.0;
  vW = aW; vL = aLayer / max(uNL - 1.0, 1.0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const LN_FS = /* glsl */`
out vec4 fragColor;
in float vW; in float vL; in float vClip;
uniform sampler2D uLut; uniform float uAlpha;
void main(){
  if (vClip > 0.5) discard;
  float a = uAlpha * (0.25 + 0.75 * vW);
  fragColor = vec4(texture(uLut, vec2(vL, 0.5)).rgb * a, a);
}`;

const PL_VS = /* glsl */`
in vec3 aUvw; out vec3 vUvw;
void main(){ vUvw = aUvw; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const PL_FS = /* glsl */`
out vec4 fragColor;
precision highp sampler3D;
in vec3 vUvw;
uniform sampler3D uShape, uPaint; uniform sampler2D uLutPaint; uniform bool uHasPaint;
uniform float uLevel, uWidth;
void main(){
  float s = clamp((texture(uShape, vUvw).r - (uLevel - uWidth*0.5)) / max(uWidth,1e-5), 0.0, 1.0);
  vec3 c = uHasPaint ? texture(uLutPaint, vec2(texture(uPaint, vUvw).r, 0.5)).rgb * s
                     : texture(uLutPaint, vec2(s, 0.5)).rgb;
  fragColor = vec4(c, 0.92);
}`;

/* ---------------------------------------------------------------- helpers */
function tex3D(bytes, dims) {
  const t = new THREE.Data3DTexture(bytes, dims[0], dims[1], dims[2]);
  t.format = THREE.RedFormat; t.type = THREE.UnsignedByteType;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
  t.unpackAlignment = 1; t.needsUpdate = true;
  return t;
}
function lutTex(name) {
  const t = new THREE.DataTexture(CM.lut(name, 256), 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}
const ADD = { blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
              transparent: true, depthWrite: false, depthTest: false };

export class ScanRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.gl.setClearColor(0x07080a, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    this.camera.position.set(1.55, 0.75, 1.75);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.autoRotateSpeed = 0.7; this.controls.minDistance = 0.3; this.controls.maxDistance = 12;
    this.controls.addEventListener("change", () => (this.dirty = true));

    this.box = new THREE.Vector3(0.5, 0.5, 0.5);
    this.extent = [1, 1, 1];
    this.params = {
      mode: 0, level: 0.5, width: 1.0, density: 1.0, cutoff: 0.12, iso: 0.35, steps: 300,
      overThr: 0.3, overGain: 1.0, clipLo: [0, 0, 0], clipHi: [1, 1, 1],
      showVolume: true, showNeurons: false, showTracts: false, showPlanes: false,
      neuronColorBy: 0, pointSize: 5, tractAlpha: 0.35, cursor: [0.5, 0.5, 0.5]
    };
    this.lutNames = { paint: "bone", over: "hot", pts: "turbo", act: "hot", line: "turbo" };
    this.luts = {};
    for (const [k, n] of Object.entries(this.lutNames)) this.luts[k] = lutTex(n);
    this.light = false;
    const empty = () => tex3D(new Uint8Array([0]), [1, 1, 1]);
    this.tex = { shape: empty(), paint: empty(), over: empty() };

    // ---- volume
    this.volU = {
      uShape: { value: this.tex.shape }, uPaint: { value: this.tex.paint }, uOver: { value: this.tex.over },
      uLutPaint: { value: this.luts.paint }, uLutOver: { value: this.luts.over },
      uHasPaint: { value: false }, uHasOver: { value: false }, uOverSigned: { value: false },
      uEye: { value: new THREE.Vector3() }, uBox: { value: this.box },
      uClipLo: { value: new THREE.Vector3(0, 0, 0) }, uClipHi: { value: new THREE.Vector3(1, 1, 1) },
      uMode: { value: 0 }, uLevel: { value: 0.5 }, uWidth: { value: 1 }, uDensity: { value: 1 },
      uCutoff: { value: 0.12 }, uIso: { value: 0.35 }, uSteps: { value: 300 },
      uOverThr: { value: 0.3 }, uOverGain: { value: 1 }, uInk: { value: new THREE.Vector3(1, 1, 1) }, uLight: { value: false },
    };
    this.volMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms: this.volU, vertexShader: VOL_VS, fragmentShader: VOL_FS,
      side: THREE.BackSide, transparent: true, depthWrite: false, depthTest: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    }));
    this.volMesh.renderOrder = 0;
    this.scene.add(this.volMesh);

    this.edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0x4fd1c5, transparent: true, opacity: 0.18, depthTest: false }));
    this.edges.renderOrder = 3;
    this.scene.add(this.edges);

    // ---- cut planes
    this.planeU = { uShape: this.volU.uShape, uPaint: this.volU.uPaint, uLutPaint: this.volU.uLutPaint,
                    uHasPaint: this.volU.uHasPaint, uLevel: this.volU.uLevel, uWidth: this.volU.uWidth };
    this.planeGeo = new THREE.BufferGeometry();
    this.planes = new THREE.Mesh(this.planeGeo, new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms: this.planeU, vertexShader: PL_VS, fragmentShader: PL_FS,
      side: THREE.DoubleSide, transparent: true, depthTest: true, depthWrite: true }));
    this.planes.renderOrder = 1; this.planes.visible = false;
    this.scene.add(this.planes);

    this.shared = { uBox: { value: this.box }, uClipLo: this.volU.uClipLo, uClipHi: this.volU.uClipHi };
    this.dirty = true; this.spin = false; this._fps = 0; this._n = 0; this._t = performance.now();
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
  }

  /* ---- data ---- */
  setExtent(ext) {
    this.extent = ext.slice();
    const m = Math.max(...ext);
    this.box.set(0.5 * ext[0] / m, 0.5 * ext[1] / m, 0.5 * ext[2] / m);
    this.volMesh.geometry.dispose();
    this.volMesh.geometry = new THREE.BoxGeometry(2 * this.box.x, 2 * this.box.y, 2 * this.box.z);
    this.edges.geometry.dispose();
    this.edges.geometry = new THREE.EdgesGeometry(this.volMesh.geometry);
    this.dirty = true;
  }
  _swap(key, bytes, dims) {
    this.tex[key].dispose();
    this.tex[key] = bytes ? tex3D(bytes, dims) : tex3D(new Uint8Array([0]), [1, 1, 1]);
    const u = { shape: "uShape", paint: "uPaint", over: "uOver" }[key];
    this.volU[u].value = this.tex[key];
  }
  setShape(bytes, dims) { this._swap("shape", bytes, dims); this.dirty = true; }
  setPaint(bytes, dims) { this._swap("paint", bytes, dims); this.volU.uHasPaint.value = !!bytes; this.dirty = true; }
  setOverlay(bytes, dims, signed) {
    if (bytes && this._overDims && this._overDims.join() === dims.join() && this.tex.over.image.data.length === bytes.length) {
      this.tex.over.image.data.set(bytes); this.tex.over.needsUpdate = true;      // fast path: next frame
    } else {
      this._swap("over", bytes, dims);
    }
    this._overDims = bytes ? dims.slice() : null;
    this.volU.uHasOver.value = !!bytes; this.volU.uOverSigned.value = !!signed;
    this.dirty = true;
  }
  setLut(which, name) {
    this.lutNames[which] = name;
    this.luts[which].image.data.set(CM.lut(name, 256));
    this.luts[which].needsUpdate = true; this.dirty = true;
  }
  setTheme(light) {
    this.light = light;
    this.gl.setClearColor(light ? 0xffffff : 0x07080a, 1);
    this.volU.uInk.value.set(...(light ? [0.10, 0.12, 0.16] : [1, 1, 1]));
    this.volU.uLight.value = light;
    for (const [k, n] of Object.entries(this.lutNames)) this.setLut(k, n);
    this.edges.material.color.set(light ? 0x0f9d8f : 0x4fd1c5);
    this.edges.material.opacity = light ? 0.35 : 0.18;
    for (const o of [this.points, this.lines]) if (o) { o.material.blendDst = this._glowDst(); o.material.needsUpdate = true; }
    this.dirty = true;
  }
  // additive glow on a dark page; ordinary over-blending on a white one
  _glowDst() { return this.light ? THREE.OneMinusSrcAlphaFactor : THREE.OneFactor; }
  _toBox(v) {          // voxel coordinate -> box (object) space
    const e = this.extent, b = this.box;
    return [-b.x + 2 * b.x * (v[0] + 0.5) / e[0], -b.y + 2 * b.y * (v[1] + 0.5) / e[1], -b.z + 2 * b.z * (v[2] + 0.5) / e[2]];
  }
  setNeurons(xyz, attr, nLayers) {
    if (this.points) { this.scene.remove(this.points); this.points.geometry.dispose(); }
    const N = xyz.length / 3, pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) pos.set(this._toBox([xyz[3*i], xyz[3*i+1], xyz[3*i+2]]), 3 * i);
    this.neuronPos = pos; this.nLayers = nLayers;
    this.neuronLayer = new Uint8Array(N);
    for (let i = 0; i < N; i++) this.neuronLayer[i] = attr[4 * i];
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aAttr", new THREE.BufferAttribute(attr, 4));
    this.actAttr = new THREE.BufferAttribute(new Uint8Array(N), 1);
    this.actAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("aAct", this.actAttr);
    this.ptU = { ...this.shared, uLut: { value: this.luts.pts }, uLutAct: { value: this.luts.act },
                 uSize: { value: 5 }, uNL: { value: nLayers }, uThr: { value: 0.3 },
                 uColorBy: { value: 0 }, uHasAct: { value: false } };
    this.points = new THREE.Points(g, new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms: this.ptU, vertexShader: PT_VS, fragmentShader: PT_FS, ...ADD, blendDst: this._glowDst() }));
    this.points.renderOrder = 2; this.points.frustumCulled = false;
    this.points.visible = this.params.showNeurons;
    this.scene.add(this.points);
    this.set({}); this.dirty = true;
  }
  setNeuronActivity(bytes) {
    if (!this.actAttr) return;
    if (bytes) { this.actAttr.array.set(bytes); this.actAttr.needsUpdate = true; }
    this.ptU.uHasAct.value = !!bytes; this.dirty = true;
  }
  setTracts(edges, weights) {
    if (!this.neuronPos) return;
    if (this.lines) { this.scene.remove(this.lines); this.lines.geometry.dispose(); }
    const E = edges.length / 2, P = this.neuronPos;
    const pos = new Float32Array(E * 6), w = new Float32Array(E * 2), lay = new Float32Array(E * 2);
    for (let e = 0; e < E; e++) {
      const a = edges[2 * e], b = edges[2 * e + 1];
      pos[6*e] = P[3*a]; pos[6*e+1] = P[3*a+1]; pos[6*e+2] = P[3*a+2];
      pos[6*e+3] = P[3*b]; pos[6*e+4] = P[3*b+1]; pos[6*e+5] = P[3*b+2];
      w[2*e] = w[2*e+1] = weights[e] / 255;
      lay[2*e] = this.neuronLayer[a]; lay[2*e+1] = this.neuronLayer[b];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aW", new THREE.BufferAttribute(w, 1));
    g.setAttribute("aLayer", new THREE.BufferAttribute(lay, 1));
    this.lnU = { ...this.shared, uLut: { value: this.luts.line }, uNL: { value: this.nLayers || 1 },
                 uAlpha: { value: 0.35 } };
    this.lines = new THREE.LineSegments(g, new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms: this.lnU, vertexShader: LN_VS, fragmentShader: LN_FS, ...ADD, blendDst: this._glowDst() }));
    this.lines.renderOrder = 2; this.lines.frustumCulled = false;
    this.lines.visible = this.params.showTracts;
    this.scene.add(this.lines);
    this.set({}); this.dirty = true;
  }
  clearNeurons() {
    for (const k of ["points", "lines"]) if (this[k]) { this.scene.remove(this[k]); this[k].geometry.dispose(); this[k] = null; }
    this.neuronPos = null; this.actAttr = null; this.dirty = true;
  }

  _buildPlanes() {
    const b = this.box, c = this.params.cursor;
    const px = -b.x + 2*b.x*c[0], py = -b.y + 2*b.y*c[1], pz = -b.z + 2*b.z*c[2];
    const P = [], U = [];
    const quad = (pts, uvw) => [0, 1, 2, 0, 2, 3].forEach(i => { P.push(...pts[i]); U.push(...uvw[i]); });
    quad([[-b.x,-b.y,pz],[b.x,-b.y,pz],[b.x,b.y,pz],[-b.x,b.y,pz]], [[0,0,c[2]],[1,0,c[2]],[1,1,c[2]],[0,1,c[2]]]);
    quad([[-b.x,py,-b.z],[b.x,py,-b.z],[b.x,py,b.z],[-b.x,py,b.z]], [[0,c[1],0],[1,c[1],0],[1,c[1],1],[0,c[1],1]]);
    quad([[px,-b.y,-b.z],[px,b.y,-b.z],[px,b.y,b.z],[px,-b.y,b.z]], [[c[0],0,0],[c[0],1,0],[c[0],1,1],[c[0],0,1]]);
    this.planeGeo.setAttribute("position", new THREE.Float32BufferAttribute(P, 3));
    this.planeGeo.setAttribute("aUvw", new THREE.Float32BufferAttribute(U, 3));
  }

  set(p) {
    Object.assign(this.params, p);
    const q = this.params, u = this.volU;
    u.uMode.value = q.mode; u.uLevel.value = q.level; u.uWidth.value = q.width;
    u.uDensity.value = q.density; u.uCutoff.value = q.cutoff; u.uIso.value = q.iso;
    u.uSteps.value = q.steps; u.uOverThr.value = q.overThr; u.uOverGain.value = q.overGain;
    u.uClipLo.value.fromArray(q.clipLo); u.uClipHi.value.fromArray(q.clipHi);
    this.volMesh.visible = q.showVolume;
    if (this.points) {
      this.points.visible = q.showNeurons;
      this.ptU.uSize.value = q.pointSize * this.gl.getPixelRatio() * 0.6;
      this.ptU.uThr.value = q.overThr; this.ptU.uColorBy.value = q.neuronColorBy;
    }
    if (this.lines) { this.lines.visible = q.showTracts; this.lnU.uAlpha.value = q.tractAlpha; }
    this.planes.visible = q.showPlanes;
    if (q.showPlanes) this._buildPlanes();
    this.controls.autoRotate = this.spin;
    this.dirty = true;
  }
  setSpin(on) { this.spin = on; this.controls.autoRotate = on; this.dirty = true; }
  resetCamera() {
    const r = this.box.length();
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(1.9 * r * 1.05, 0.9 * r * 1.05, 2.1 * r * 1.05);
    this.controls.update(); this.dirty = true;
  }

  _resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.gl.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.dirty = true;
  }
  get fps() { return this._fps; }
  start() {
    this.gl.setAnimationLoop(() => {
      if (this.controls.update()) this.dirty = true;
      if (!this.dirty) return;
      this.dirty = false;
      this.volU.uEye.value.copy(this.volMesh.worldToLocal(this.camera.position.clone()));
      this.gl.render(this.scene, this.camera);
      this._n++; const now = performance.now();
      if (now - this._t > 500) { this._fps = Math.round(this._n * 1000 / (now - this._t)); this._n = 0; this._t = now; }
    });
  }
}
