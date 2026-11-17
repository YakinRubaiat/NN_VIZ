/* nnviz viewer controller.
 * Loads /data/index.json (written by `python -m nnviz scan`), then lets you
 * switch between the Brain and Layer-stack views, choose what colours the
 * structure and what activity is drawn on it, and play the dynamic scan.
 */
import { ScanRenderer } from "./three-renderer.js";
import { ArchScene } from "./arch-scene.js";

const CM = window.NNColormaps;
const $ = id => document.getElementById(id);
const MODES = ["Volume", "MIP", "Surface", "X-ray"];

const S = {
  root: "data/", catalog: null,
  index: null, view: null, man: null, base: "",
  bytes: new Map(), paint: null, overlay: null,
  frames: null, neuronFrames: null, frame: 0, playing: false, timer: null,
  neurons: null, preset: null,
};

/* ------------------------------------------------------------ io */
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function getBytes(url, Type = Uint8Array) {
  const key = url + Type.name;
  if (S.bytes.has(key)) return S.bytes.get(key);
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const arr = new Type(await r.arrayBuffer());
  S.bytes.set(key, arr);
  return arr;
}
const vol = name => S.man.volumes.find(v => v.name === name);
const volBytes = v => getBytes(S.base + v.file);

/* uint8 -> physical value, mirroring nnviz/volio.py */
function dequant(rec, q) {
  const u = q / 255;
  if (rec.kind === "signed") return (u - 0.5) * 2 * rec.hi;
  if (rec.kind === "asinh") {
    const s = rec.scale, tlo = Math.asinh(rec.lo / s), thi = Math.asinh(rec.hi / s);
    return s * Math.sinh(tlo + u * (thi - tlo));
  }
  return rec.lo + u * (rec.hi - rec.lo);
}
const fmt = v => (v == null || !isFinite(v)) ? "—" :
  Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) :
  Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(3) : v.toExponential(1);

/* ------------------------------------------------------------ renderer + slices */
const R = new ScanRenderer($("gl"));
const SV = new window.SliceView({ ax: $("cAx"), co: $("cCo"), sa: $("cSa") }, c => {
  SV.cursor = c; R.set({ cursor: c }); drawSlices(); probe();
});
R.start();
window.__nnviz = { R, S };   // handle for debugging from the console

setInterval(() => { $("roFps").textContent = R.fps || "idle"; }, 500);

/* ------------------------------------------------------------ boot */
async function boot() {
  try {
    applyTheme(savedTheme(), false);
    let cat = null;
    try { cat = await getJSON("data/catalog.json"); } catch { /* single-scan mount */ }
    if (cat && cat.scans && cat.scans.length) {
      S.catalog = cat.scans;
      const sel = $("modelSelect"); sel.innerHTML = "";
      cat.scans.forEach(c => {
        const o = document.createElement("option");
        o.value = c.dir; o.textContent = `${c.model.split("/").pop()}  ·  ${c.params_m >= 1000 ? (c.params_m / 1000).toFixed(1) + "B" : Math.round(c.params_m) + "M"}`;
        sel.appendChild(o);
      });
      let pick = null;
      try { pick = localStorage.getItem("nnviz.model"); } catch { /* storage unavailable */ }
      if (!cat.scans.some(c => c.dir === pick)) pick = (cat.scans.find(c => c.dir === "qwen3-0-6b") || cat.scans[0]).dir;
      sel.value = pick; S.root = `data/${pick}/`;
      $("modelBlock").hidden = false;
      sel.onchange = async () => {
        try { localStorage.setItem("nnviz.model", sel.value); } catch { /* ignore */ }
        S.root = `data/${sel.value}/`; S.bytes.clear(); S.archCache = null;
        await openScan(S.view);
      };
    }
    await openScan();
    $("boot").classList.add("gone");
  } catch (e) {
    console.error(e);
    $("bootMsg").textContent = "Could not load a scan.";
    $("bootHint").innerHTML = "Run <code>nnviz scan</code>, then <code>nnviz view</code>.<br>" + e.message;
  }
}

async function openScan(keepKind) {
  stop();
  if (S.arch) { S.arch.detach(); S.arch = null; }
  try { S.index = await getJSON(S.root + "index.json"); }
  catch { const m = await getJSON(S.root + "manifest.json"); S.index = { views: [{ kind: m.kind, title: m.title, manifest: "manifest.json" }], meta: m.meta }; }
  const tabs = $("viewTabs");
  tabs.innerHTML = "";
  S.index.views.forEach(v => {
    const b = document.createElement("button");
    b.textContent = v.title; b.dataset.kind = v.kind;
    b.onclick = () => loadView(v.kind);
    tabs.appendChild(b);
  });
  const kind = keepKind && S.index.views.some(v => v.kind === keepKind) ? keepKind : S.index.views[0].kind;
  await loadView(kind);
}

/* ------------------------------------------------------------ theme */
function savedTheme() {
  try { return localStorage.getItem("nnviz.theme") || "light"; } catch { return "light"; }
}
function applyTheme(t, redraw = true) {
  const light = t === "light";
  document.documentElement.dataset.theme = t;
  CM.light = light;
  R.setTheme(light);
  if (S.arch) S.arch.setTheme(light);
  if (S.archCache && S.archCache.scene !== S.arch) S.archCache.scene.setTheme(light);
  $("themeBtn").textContent = light ? "☾" : "☀";
  $("themeBtn").title = light ? "Dark background" : "White background";
  try { localStorage.setItem("nnviz.theme", t); } catch { /* storage unavailable */ }
  if (redraw && S.man) { legend(); drawSlices(); if (S.frames) drawCharts(); }
}
$("themeBtn").onclick = () => applyTheme(CM.light ? "dark" : "light");

async function loadView(kind) {
  stop();
  const entry = S.index.views.find(v => v.kind === kind);
  S.view = kind;
  S.man = await getJSON(S.root + entry.manifest);
  S.base = S.root + entry.manifest.replace(/manifest\.json$/, "");
  [...$("viewTabs").children].forEach(b => b.classList.toggle("on", b.dataset.kind === kind));
  $("viewBadge").textContent = S.man.title;
  $("modelBadge").textContent = (S.index.meta || S.man.meta).model;

  const arch = kind === "arch", brain = kind === "brain";
  if (S.arch) { S.arch.detach(); S.arch = null; }
  ["renderBlock", "structBlock", "actWinBlock", "clipBlock", "sliceBlock"].forEach(id => $(id).hidden = arch);
  $("archBlock").hidden = !arch;
  ["tNeuronsL", "tTractsL", "ncRow", "psRow", "taRow"].forEach(id => $(id).hidden = !brain);
  $("tip").hidden = true;
  R.clearNeurons(); R.setOverlay(null); R.setPaint(null);
  S.neurons = null; S.frames = S.neuronFrames = null; S.frame = 0; S.shapeBytes = null;
  S.paint = null; S.overlay = null;

  if (arch) {
    const [w, n] = await Promise.all([getBytes(S.base + S.man.weights.file), getBytes(S.base + S.man.norms.file)]);
    if (!S.archCache || S.archCache.base !== S.base) S.archCache = { base: S.base, scene: new ArchScene(R, S.man, w, n) };
    S.arch = S.archCache.scene; S.arch.attach();
    $("roGrid").textContent = `${S.man.weights.panels.length} panels`;
    $("roMode").textContent = "network";
    if (S.man.flow) {
      const f = S.man.flow;
      S.frames = { ...f, arch: true, shape: [f.T], fire_at: 0.5, transform: { hi: 1 }, bytes: await getBytes(S.base + f.file) };
    }
  } else {
    if (brain) {
      const shape = vol(S.man.shape_volume);
      R.setExtent(S.man.extent);
      S.shapeVol = shape; S.shapeBytes = await volBytes(shape);
      R.setShape(S.shapeBytes, shape.shape);
      SV.extent = S.man.extent;
      $("roGrid").textContent = S.man.extent.join("×");
      const n = S.man.neurons;
      const [xyz, attr] = await Promise.all([getBytes(S.base + n.xyz, Float32Array), getBytes(S.base + n.attr)]);
      S.neurons = { xyz, attr, count: n.count, nLayers: n.n_layers };
      R.setNeurons(xyz, attr, n.n_layers);
      const t = S.man.tracts;
      const [edges, w] = await Promise.all([getBytes(S.base + t.file, Int32Array), getBytes(S.base + t.weight)]);
      R.setTracts(edges, w);
    }
    if (S.man.frames) {
      const f = S.man.frames;
      S.frames = { ...f, bytes: await getBytes(S.base + f.file) };
      if (f.neuron_file) S.neuronFrames = await getBytes(S.base + f.neuron_file);
    }
  }
  buildLists(); buildMeta(); buildDock(); buildPresets();
  await applyPreset(S.presets[0]);
  if (!arch) R.resetCamera();
  if (S.frames) play();
}

/* ------------------------------------------------------------ lists */
function item(parent, label, tag, on, click) {
  const b = document.createElement("button");
  b.innerHTML = `<span>${label}</span>${tag ? `<span class="tag">${tag}</span>` : ""}`;
  if (on) b.classList.add("on");
  b.onclick = click; parent.appendChild(b); return b;
}
function group(parent, name) {
  const g = document.createElement("div"); g.className = "grp"; g.textContent = name; parent.appendChild(g);
}
function buildLists() {
  const pl = $("paintList"), ol = $("overList");
  pl.innerHTML = ""; ol.innerHTML = "";
  if (S.view === "arch") {
    item(pl, "Weight density", "global", S.paint === "global", () => selectPaint("global")).dataset.name = "global";
    item(pl, "Weight density", "per panel", S.paint === "perpanel", () => selectPaint("perpanel")).dataset.name = "perpanel";
    item(ol, "None", "", !S.overlay, () => selectOverlay(null)).dataset.name = "";
    if (S.frames) item(ol, "Dynamic scan", `${S.frames.shape[0]} tok`, S.overlay === "@frames", () => selectOverlay("@frames")).dataset.name = "@frames";
    return;
  }
  const brain = S.view === "brain";
  const paints = S.man.volumes.filter(v => brain ? (v.role === "paint" || v.role === "shape") : true);
  let lastG = null;
  paints.forEach(v => {
    if (!brain && v.group !== lastG) { group(pl, v.group); lastG = v.group; }
    item(pl, v.label, v.role === "shape" ? "plain" : "", S.paint === v.name, () => selectPaint(v.name)).dataset.name = v.name;
  });
  item(ol, "None", "", S.overlay === null, () => selectOverlay(null)).dataset.name = "";
  if (S.frames) item(ol, "Dynamic scan", `${S.frames.shape[0]} tok`, S.overlay === "@frames", () => selectOverlay("@frames")).dataset.name = "@frames";
  if (brain) {
    lastG = null;
    S.man.volumes.filter(v => v.role === "overlay" || v.role === "signed").forEach(v => {
      if (v.group !== lastG) { group(ol, v.group); lastG = v.group; }
      item(ol, v.label, v.role === "signed" ? "±" : "", S.overlay === v.name, () => selectOverlay(v.name)).dataset.name = v.name;
    });
  }
}
function markLists() {
  [...$("paintList").querySelectorAll("button")].forEach(b => b.classList.toggle("on", b.dataset.name === S.paint));
  [...$("overList").querySelectorAll("button")].forEach(b => b.classList.toggle("on", b.dataset.name === (S.overlay || "")));
}

/* ------------------------------------------------------------ selection */
async function selectPaint(name) {
  if (S.view === "arch") {
    S.paint = name; S.arch.set({ perPanel: name === "perpanel" });
    [...$("archScale").children].forEach(b => b.classList.toggle("on", (+b.dataset.pp === 1) === (name === "perpanel")));
    markLists(); legend(); describe(); return;
  }
  const v = vol(name); if (!v) return;
  S.paint = name;
  if (S.view === "brain") {
    if (v.role === "shape") { R.setPaint(null); R.setLut("paint", "bone"); S.paintBytes = null; }
    else { S.paintBytes = await volBytes(v); R.setPaint(S.paintBytes, v.shape); R.setLut("paint", v.cmap); }
  } else {
    // stack: the chosen volume is the structure itself
    const ext = v.extent || v.shape;
    R.setExtent(ext); SV.extent = ext;
    S.shapeVol = v; S.shapeBytes = await volBytes(v);
    R.setShape(S.shapeBytes, v.shape); R.setPaint(null); R.setLut("paint", v.cmap); S.paintBytes = null;
    $("roGrid").textContent = v.shape.join("×");
  }
  markLists(); legend(); describe(); drawSlices(); probe();
}

async function selectOverlay(key) {
  S.overlay = key;
  if (S.view === "arch") {
    S.arch.set({ showAct: key === "@frames" });
    if (key === "@frames") pushFrame();
    markLists(); legend(); describe(); return;
  }
  if (!key) { R.setOverlay(null); $("lgOver").hidden = true; }
  else if (key === "@frames") { R.setLut("over", "hot"); pushFrame(); }
  else {
    const v = vol(key);
    R.setLut("over", v.cmap);
    S.overBytes = await volBytes(v); S.overVol = v;
    R.setOverlay(S.overBytes, v.shape, v.role === "signed");
  }
  markLists(); legend(); describe(); drawSlices(); probe();
}

/* ------------------------------------------------------------ dynamic scan */
function frameBytes(t) {
  if (S.frames.arch) { const n = S.frames.frame_size; return S.frames.bytes.subarray(t * n, (t + 1) * n); }
  const [T, x, y, z] = S.frames.shape, n = x * y * z;
  return S.frames.bytes.subarray(t * n, (t + 1) * n);
}
function pushFrame() {
  if (!S.frames) return;
  const t = S.frame, f = S.frames;
  if (f.arch) S.arch.setFlow(f, frameBytes(t));
  else if (S.overlay === "@frames") R.setOverlay(frameBytes(t), f.shape.slice(1), false);
  if (S.neuronFrames && S.neurons) {
    const N = S.neurons.count;
    R.setNeuronActivity(S.neuronFrames.subarray(t * N, (t + 1) * N));
  }
  $("frame").value = t;
  $("frameOut").textContent = `${t + 1} / ${f.shape[0]}`;
  [...$("tokenStrip").children].forEach((s, i) => s.classList.toggle("on", i === t));
  const on = $("tokenStrip").children[t]; if (on) on.scrollIntoView({ block: "nearest", inline: "nearest" });
  drawCharts();
  if (S.overlay === "@frames" && !f.arch) { drawSlices(); probe(); }
  if (f.arch && S.lastNdc && !$("tip").hidden) { S.lastPick = S.arch.pick(S.lastNdc); inspect(S.lastPick); }
}
function seek(t) { S.frame = Math.max(0, Math.min(S.frames.shape[0] - 1, t)); pushFrame(); }
function play() {
  if (!S.frames) return;
  S.playing = true; $("playBtn").textContent = "❚❚";
  clearInterval(S.timer);
  S.timer = setInterval(() => seek((S.frame + 1) % S.frames.shape[0]), +$("speed").value);
}
function stop() { S.playing = false; clearInterval(S.timer); $("playBtn").textContent = "▶"; }
$("playBtn").onclick = () => (S.playing ? stop() : play());
$("frame").oninput = e => { stop(); seek(+e.target.value); };
$("speed").onchange = () => { if (S.playing) play(); };
document.addEventListener("keydown", e => {
  if (!S.frames || ["INPUT", "SELECT"].includes(document.activeElement.tagName) && e.key !== " ") return;
  if (e.key === " ") { e.preventDefault(); S.playing ? stop() : play(); }
  if (e.key === "ArrowRight") { stop(); seek(S.frame + 1); }
  if (e.key === "ArrowLeft") { stop(); seek(S.frame - 1); }
});

function buildDock() {
  $("dock").hidden = !S.frames;
  if (!S.frames) return;
  const f = S.frames, strip = $("tokenStrip");
  $("frame").max = f.shape[0] - 1;
  strip.innerHTML = "";
  f.tokens.forEach((tok, i) => {
    const s = document.createElement("span");
    s.textContent = tok.replace(/ /g, "·") || "∅"; s.title = `token ${f.token_index[i]}`;
    s.onclick = () => { stop(); seek(i); };
    strip.appendChild(s);
  });
  ["chartFire", "chartLayer"].forEach(id => {
    $(id).onclick = e => {
      const r = e.target.getBoundingClientRect();
      stop(); seek(Math.floor((e.clientX - r.left) / r.width * f.shape[0]));
    };
  });
}

function drawCharts() {
  const f = S.frames;
  document.querySelector(".charts").hidden = !(f && f.stats);
  if (!f || !f.stats) return;
  const st = f.stats, T = f.shape[0];
  // firing fraction lines
  const c = $("chartFire"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  if (st.frac_above_attn) {
    const ys = [st.frac_above_attn, st.frac_above_mlp];
    const max = Math.max(1e-6, ...ys.flat()) * 1.1;
    g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--grid").trim() || "#1b2028"; g.lineWidth = 1;
    for (let k = 1; k < 4; k++) { g.beginPath(); g.moveTo(0, H * k / 4); g.lineTo(W, H * k / 4); g.stroke(); }
    [[CM.light ? "#3b82f6" : "#6aa7ff", ys[0]], [CM.light ? "#f59e0b" : "#ffb454", ys[1]]].forEach(([col, y]) => {
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
      y.forEach((v, i) => { const X = (i + 0.5) / T * W, Y = H - 4 - v / max * (H - 10); i ? g.lineTo(X, Y) : g.moveTo(X, Y); });
      g.stroke();
    });
    g.fillStyle = "#5d6675"; g.font = "11px ui-monospace, monospace";
    g.fillText(`${(max * 100).toFixed(1)}%`, 4, 12);
  }
  marker(g, W, H, T);
  // per-layer heatmap
  const c2 = $("chartLayer"), g2 = c2.getContext("2d"), W2 = c2.width, H2 = c2.height;
  g2.clearRect(0, 0, W2, H2);
  const pl = st.per_layer_above_half;
  if (pl) {
    const L = pl[0].length, max = Math.max(1e-6, ...pl.flat());
    const lut = CM.lut("hot", 256);
    for (let t = 0; t < T; t++) for (let l = 0; l < L; l++) {
      const q = Math.round(Math.min(1, pl[t][l] / max) * 255) * 4;
      g2.fillStyle = `rgb(${lut[q]},${lut[q + 1]},${lut[q + 2]})`;
      g2.fillRect(Math.floor(t / T * W2), Math.floor(l / L * H2), Math.ceil(W2 / T), Math.ceil(H2 / L));
    }
  }
  marker(g2, W2, H2, T);
}
function marker(g, W, H, T) {
  const x = (S.frame + 0.5) / T * W;
  g.strokeStyle = "rgba(79,209,197,0.9)"; g.lineWidth = 2;
  g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
}

/* ------------------------------------------------------------ legend, text */
function legend() {
  if (S.view === "arch") {
    const tr = S.man.weights.transform;
    $("rampPaint").style.background = CM.cssGradient($("archCmap").value || "magma");
    $("lgPaintName").textContent = S.paint === "perpanel" ? "weight RMS (each panel stretched)" : "weight RMS per 8×8 block";
    $("rampPaintLo").textContent = "0"; $("rampPaintHi").textContent = fmt(tr.hi);
    $("lgOver").hidden = S.overlay !== "@frames";
    $("rampOver").style.background = CM.cssGradient("hot");
    $("lgOverName").textContent = "firing (fraction of own peak)";
    $("rampOverLo").textContent = "0"; $("rampOverHi").textContent = "1";
    return;
  }
  const v = vol(S.paint);
  if (v) {
    const cm = (S.view === "brain" && v.role === "shape") ? "bone" : v.cmap;
    $("rampPaint").style.background = CM.cssGradient(cm);
    $("lgPaintName").textContent = v.label;
    if (v.labels && v.labels.length === 2) { $("rampPaintLo").textContent = v.labels[0]; $("rampPaintHi").textContent = v.labels[1]; }
    else { $("rampPaintLo").textContent = fmt(v.transform.lo); $("rampPaintHi").textContent = fmt(v.transform.hi) + (v.unit ? " " + v.unit : ""); }
  }
  const o = S.overlay;
  $("lgOver").hidden = !o;
  if (o === "@frames") {
    $("rampOver").style.background = CM.cssGradient("hot");
    const stk = S.view === "stack";
    $("lgOverName").textContent = stk ? "|residual| ÷ channel's peak" : `firing > ${S.frames.fire_at ?? 0.5} of peak`;
    $("rampOverLo").textContent = "0%"; $("rampOverHi").textContent = (S.frames.transform.hi * 100).toFixed(0) + (stk ? "% of peak" : "% of voxel");
  } else if (o) {
    const ov = vol(o);
    $("rampOver").style.background = CM.cssGradient(ov.cmap);
    $("lgOverName").textContent = ov.label;
    $("rampOverLo").textContent = fmt(ov.transform.lo); $("rampOverHi").textContent = fmt(ov.transform.hi);
  }
}
function describe() {
  if (S.view === "arch") {
    const m = S.man.meta;
    let h = `<b>Weight density.</b> Every weight matrix of the model, laid out as it is computed: token embedding, then ${S.man.n_layers} layers of attention (q, k, v, o) and MLP (gate, up, down), then the output head. Panel area is proportional to parameter count. Each texel is the RMS of an ${m.texel}×${m.texel} block of real weights; bright = large weights packed densely, dark/transparent = small. The residual-stream axis runs horizontally in every panel, so a hidden dimension with large weights everywhere appears as a line through the whole network.`;
    if (S.paint === "perpanel") h += " <i>Per-panel scale: each panel is stretched to its own range — good for structure inside a panel, not for comparing panels.</i>";
    if (S.overlay === "@frames") h += `<br><br><b>Dynamic scan.</b> ${S.frames.description}<br><span class="muted">“${S.frames.prompt}”</span>`;
    $("desc").innerHTML = h; return;
  }
  const parts = [];
  const v = vol(S.paint); if (v) parts.push(`<b>${v.label}.</b> ${v.description}`);
  if (S.overlay === "@frames") parts.push(`<b>${S.frames.label}.</b> ${S.frames.description}<br><span class="muted">“${S.frames.prompt}”</span>`);
  else if (S.overlay) { const o = vol(S.overlay); parts.push(`<b>${o.label}.</b> ${o.description}`); }
  $("desc").innerHTML = parts.join("<br><br>");
}
function buildMeta() {
  const m = S.index.meta || S.man.meta, rows = [
    ["model", m.model], ["params", `${(m.n_params / 1e6).toFixed(0)} M`],
    ["layers", `${m.n_layers} · hidden ${m.hidden_size}`],
  ];
  if (S.view === "arch") {
    rows.push(["shown", `${(S.man.meta.weight_params_shown / 1e6).toFixed(0)} M weights in ${S.man.weights.panels.length} panels`]);
    rows.push(["texel", `${S.man.meta.texel}×${S.man.meta.texel} weights (RMS)`]);
  }
  const c = S.man.meta && S.man.meta.connectome;
  if (c) {
    rows.push(["neurons", c.n_neurons.toLocaleString()]);
    rows.push(["wiring", `${c.mean_top_cos.toFixed(3)} vs chance ${c.chance_cos.toFixed(3)}`]);
  }
  rows.push(["probe", `${m.n_prompts} prompts · ${m.n_tokens} tok`]);
  if (S.frames && S.frames.stats) {
    const fa = S.frames.stats.frac_above.map(r => r[1]);
    rows.push(["firing", `${(fa.reduce((a, b) => a + b, 0) / fa.length * 100).toFixed(2)}% of neurons > ½ peak`]);
  }
  $("metaList").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

/* ------------------------------------------------------------ slices + probe */
function overLayer() {
  if (S.overlay === "@frames" && S.frames) return { bytes: frameBytes(S.frame), dims: S.frames.shape.slice(1) };
  if (S.overlay && S.overBytes && S.overVol) return { bytes: S.overBytes, dims: S.overVol.shape };
  return null;
}
function drawSlices() {
  if (!S.shapeBytes || S.view === "arch") return;
  const p = R.params, v = vol(S.paint);
  const paintLayer = (S.view === "brain" && S.paintBytes && v) ? { bytes: S.paintBytes, dims: v.shape } : null;
  const ov = overLayer();
  SV.draw({ shape: { bytes: S.shapeBytes, dims: S.shapeVol.shape }, paint: paintLayer, over: ov }, {
    paintCmap: paintLayer ? v.cmap : (S.view === "brain" ? "bone" : S.shapeVol.cmap),
    overCmap: ov ? (S.overlay === "@frames" ? "hot" : vol(S.overlay).cmap) : null,
    overSigned: !!(S.overlay && S.overlay !== "@frames" && vol(S.overlay).role === "signed"),
    overThr: p.overThr, overGain: p.overGain, level: p.level, width: p.width, light: CM.light,
  });
}
function nearestNeuron(c) {
  if (!S.neurons) return null;
  const e = S.man.extent, X = S.neurons.xyz;
  const px = c[0] * e[0] - 0.5, py = c[1] * e[1] - 0.5, pz = c[2] * e[2] - 0.5;
  let best = -1, bd = Infinity;
  for (let i = 0; i < S.neurons.count; i++) {
    const dx = X[3*i] - px, dy = X[3*i+1] - py, dz = X[3*i+2] - pz, d = dx*dx + dy*dy + dz*dz;
    if (d < bd) { bd = d; best = i; }
  }
  return best < 0 ? null : { i: best, dist: Math.sqrt(bd) };
}
function probe() {
  if (S.view === "arch") return;
  const p = SV.probe(), rows = [];
  const c = SV.cursor;
  if (S.view === "stack" && S.shapeVol) {
    const nl = S.shapeVol.n_layers || S.shapeVol.shape[2];
    rows.push(["layer", Math.min(nl - 1, Math.floor(c[2] * nl))]);
    const sp = S.man.spaces[S.shapeVol.space];
    if (sp) {
      const x = Math.min(sp.width - 1, Math.floor(c[0] * sp.width)), y = Math.min(sp.height - 1, Math.floor(c[1] * sp.height));
      const ch = sp.channel_at[y * sp.width + x];
      rows.push([sp.name === "mlp" ? "neuron" : "channel", ch < 0 ? "padding" : ch]);
    }
  }
  if (p.shape != null) rows.push([S.shapeVol.label.toLowerCase(), fmt(dequant(S.shapeVol.transform, p.shape))]);
  const v = vol(S.paint);
  if (S.view === "brain" && p.paint != null && S.paintBytes && v) rows.push([v.label.toLowerCase(), fmt(dequant(v.transform, p.paint))]);
  if (p.over != null && S.overlay) {
    if (S.overlay === "@frames") rows.push(["firing here", (p.over / 255 * S.frames.transform.hi * 100).toFixed(1) + "%"]);
    else { const o = vol(S.overlay); rows.push([o.label.toLowerCase(), fmt(dequant(o.transform, p.over))]); }
  }
  let html = rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
  const nn = nearestNeuron(c);
  if (nn) {
    const a = S.neurons.attr, i = nn.i, fam = a[4*i+1] ? "MLP neuron" : `attn head ${a[4*i+3] - 1}`;
    let act = "";
    if (S.neuronFrames) act = (S.neuronFrames[S.frame * S.neurons.count + i] / 255).toFixed(2);
    html += `<div class="sep"></div><div><span>nearest</span><b>#${i}</b></div><div><span>kind</span><b>${fam}</b></div>`
         + `<div><span>layer</span><b>${a[4*i]}</b></div>` + (act ? `<div><span>activity now</span><b>${act}</b></div>` : "");
  }
  $("probe").innerHTML = html;
}

/* ------------------------------------------------------------ controls */
const sliders = [["density", 2], ["cutoff", 2], ["level", 2], ["width", 2], ["iso", 2], ["overThr", 2],
                 ["overGain", 2], ["pointSize", 1], ["tractAlpha", 2]];
sliders.forEach(([id, d]) => {
  $(id).addEventListener("input", e => {
    const v = +e.target.value; $(id + "Out").textContent = v.toFixed(d);
    R.set({ [id]: v }); if (["level", "width", "overThr", "overGain"].includes(id)) { drawSlices(); }
  });
});
["X", "Y", "Z"].forEach((a, i) => $("clip" + a).addEventListener("input", e => {
  const hi = R.params.clipHi.slice(); hi[i] = +e.target.value; $("clip" + a + "Out").textContent = hi[i].toFixed(2);
  R.set({ clipHi: hi });
}));
[...$("modeSegs").children].forEach(b => b.onclick = () => setMode(+b.dataset.mode));
function setMode(m) {
  R.set({ mode: m }); [...$("modeSegs").children].forEach(b => b.classList.toggle("on", +b.dataset.mode === m));
  $("roMode").textContent = MODES[m];
}
$("tVolume").onchange = e => R.set({ showVolume: e.target.checked });
$("tNeurons").onchange = e => R.set({ showNeurons: e.target.checked });
$("tTracts").onchange = e => R.set({ showTracts: e.target.checked });
$("tPlanes").onchange = e => R.set({ showPlanes: e.target.checked });
$("tSpin").onchange = e => R.setSpin(e.target.checked);
$("neuronColor").onchange = e => { R.set({ neuronColorBy: +e.target.value }); setNeuronLut(+e.target.value); };
function setNeuronLut(by) { R.setLut("pts", by === 1 ? "parts2" : by === 2 ? "magma" : "turbo"); }

function setControl(id, v, d = 2) { $(id).value = v; const o = $(id + "Out"); if (o) o.textContent = (+v).toFixed(d); }

/* ------------------------------------------------------------ presets */
function buildPresets() {
  if (S.view === "arch") {
    S.presets = [
      S.frames && { id: "adyn", name: "Dynamic scan", sub: "neurons light up per token", arch: { solid: false, perPanel: false, thr: 0.3, gain: 1.0, opacity: 0.22, dim: 0.25, actGain: 1.0 }, paint: "global", overlay: "@frames" },
      { id: "axray", name: "X-ray", sub: "see through to dense weights", arch: { solid: false, perPanel: false, thr: 0.25, gain: 1.0, opacity: 0.3 }, paint: "global", overlay: null },
      { id: "asolid", name: "Solid panels", sub: "every matrix, opaque", arch: { solid: true, perPanel: false, thr: 0.0, gain: 1.0, opacity: 1 }, paint: "global", overlay: null },
      { id: "adense", name: "Densest only", sub: "only the largest weights", arch: { solid: false, perPanel: false, thr: 0.62, gain: 1.0, opacity: 0.9 }, paint: "global", overlay: null },
      { id: "astruct", name: "Inner structure", sub: "each panel stretched", arch: { solid: true, perPanel: true, thr: 0.0, gain: 1.0, opacity: 1 }, paint: "perpanel", overlay: null },
    ].filter(Boolean);
    const box = $("presets"); box.innerHTML = "";
    S.presets.forEach(p => { const b = document.createElement("button"); b.innerHTML = `${p.name}<small>${p.sub}</small>`; b.dataset.id = p.id; b.onclick = () => applyPreset(p); box.appendChild(b); });
    return;
  }
  const brain = S.view === "brain", tasks = S.man.volumes.filter(v => v.group === "Task contrast");
  S.presets = brain ? [
    S.frames && { id: "dyn", name: "Dynamic scan", sub: "activity playing on the structure",
      paint: "density", overlay: "@frames", mode: 0, density: 0.3, cutoff: 0.1, overThr: 0.45, overGain: 0.8,
      vol: true, neurons: false, tracts: false },
    S.frames && { id: "fire", name: "Neurons firing", sub: "every neuron, lit when active",
      paint: "density", overlay: "@frames", mode: 0, density: 0.08, cutoff: 0.1, overThr: 0.3, overGain: 0.0001,
      vol: true, neurons: true, tracts: false, ncolor: 3, psize: 4 },
    { id: "wire", name: "Wiring", sub: "strongest synapses as fibres",
      paint: "depth", overlay: null, mode: 0, density: 0.12, cutoff: 0.1, vol: true, neurons: true, tracts: true,
      ncolor: 0, psize: 2.5, talpha: 0.3 },
    { id: "mri", name: "Anatomy", sub: "structure coloured by layer depth",
      paint: "depth", overlay: null, mode: 0, density: 1.0, cutoff: 0.12, vol: true, neurons: false, tracts: false },
    { id: "surf", name: "Surface", sub: "outer surface, activity beneath",
      paint: "depth", overlay: S.frames ? "@frames" : null, mode: 2, iso: 0.3, overThr: 0.25, overGain: 1.2,
      vol: true, neurons: false, tracts: false },
    tasks.length && { id: "task", name: "Task contrast", sub: `${tasks[0].label.replace("Task: ", "")} vs all`,
      paint: "density", overlay: tasks[0].name, mode: 0, density: 0.2, cutoff: 0.1, overThr: 0.25, overGain: 1.3,
      vol: true, neurons: false, tracts: false },
  ].filter(Boolean) : [
    S.frames && { id: "sdyn", name: "Dynamic scan", sub: "|residual| per token",
      paint: S.man.default.paint, overlay: "@frames", mode: 0, density: 0.35, cutoff: 0.15, overThr: 0.3, overGain: 1.3, vol: true },
    { id: "senergy", name: "Energy", sub: "residual stream by depth", paint: "resid_energy", overlay: null, mode: 0, density: 1, cutoff: 0.12, vol: true },
    { id: "sneur", name: "MLP neurons", sub: "duty cycle, one voxel per neuron", paint: "mlp_duty", overlay: null, mode: 1, cutoff: 0.05, density: 1, vol: true },
    { id: "sweights", name: "Weights", sub: "read strength", paint: "w_read", overlay: null, mode: 0, density: 1, cutoff: 0.1, vol: true },
  ].filter(p => p && (!p.paint || vol(p.paint)));
  const box = $("presets"); box.innerHTML = "";
  S.presets.forEach(p => {
    const b = document.createElement("button");
    b.innerHTML = `${p.name}<small>${p.sub}</small>`; b.dataset.id = p.id;
    b.onclick = () => applyPreset(p); box.appendChild(b);
  });
}
async function applyPreset(p) {
  if (!p) return;
  S.preset = p.id;
  [...$("presets").children].forEach(b => b.classList.toggle("on", b.dataset.id === p.id));
  if (S.view === "arch") {
    const a = { solid: false, perPanel: false, thr: 0.18, gain: 1, opacity: 0.9, dim: 0.35, actGain: 1, ...p.arch };
    S.arch.set(a);
    setControl("archThr", a.thr); setControl("archGain", a.gain); setControl("archOpacity", a.opacity);
    setControl("archDim", a.dim); setControl("archAct", a.actGain);
    [...$("archDraw").children].forEach(b => b.classList.toggle("on", (+b.dataset.solid === 1) === a.solid));
    await selectPaint(p.paint); await selectOverlay(p.overlay);
    return;
  }
  const set = {
    mode: p.mode ?? 0, density: p.density ?? 1, cutoff: p.cutoff ?? 0.12, iso: p.iso ?? 0.35,
    overThr: p.overThr ?? 0.3, overGain: p.overGain ?? 1, level: 0.5, width: 1,
    showVolume: p.vol !== false, showNeurons: !!p.neurons, showTracts: !!p.tracts,
    neuronColorBy: p.ncolor ?? 0, pointSize: p.psize ?? 5, tractAlpha: p.talpha ?? 0.35,
  };
  R.set(set); setMode(set.mode);
  setControl("density", set.density); setControl("cutoff", set.cutoff); setControl("iso", set.iso);
  setControl("overThr", set.overThr); setControl("overGain", set.overGain);
  setControl("level", 0.5); setControl("width", 1);
  setControl("pointSize", set.pointSize, 1); setControl("tractAlpha", set.tractAlpha);
  $("tVolume").checked = set.showVolume; $("tNeurons").checked = set.showNeurons; $("tTracts").checked = set.showTracts;
  $("neuronColor").value = String(set.neuronColorBy); setNeuronLut(set.neuronColorBy);
  await selectPaint(p.paint || S.man.default.paint);
  await selectOverlay(p.overlay === undefined ? null : p.overlay);
}

/* ------------------------------------------------------------ network view controls */
CM.ORDER.filter(n => !["parts", "parts2", "diverge"].includes(n)).forEach(n => {
  const o = document.createElement("option"); o.value = n; o.textContent = n; if (n === "magma") o.selected = true;
  $("archCmap").appendChild(o);
});
$("archCmap").onchange = e => { if (S.arch) { S.arch.setLut(e.target.value); legend(); } };
[["archThr", "thr"], ["archGain", "gain"], ["archOpacity", "opacity"], ["archAct", "actGain"], ["archDim", "dim"]].forEach(([id, key]) => {
  $(id).addEventListener("input", e => { const v = +e.target.value; $(id + "Out").textContent = v.toFixed(2); if (S.arch) S.arch.set({ [key]: v }); });
});
[...$("archDraw").children].forEach(b => b.onclick = () => {
  const solid = +b.dataset.solid === 1; S.arch && S.arch.set({ solid });
  [...$("archDraw").children].forEach(x => x.classList.toggle("on", x === b));
});
[...$("archScale").children].forEach(b => b.onclick = () => selectPaint(+b.dataset.pp ? "perpanel" : "global"));
$("archSpin").onchange = e => R.setSpin(e.target.checked);

function inspect(hit) {
  const tip = $("tip");
  if (!hit) { tip.hidden = true; return; }
  const tr = S.man.weights.transform;
  const asinhVal = q => tr.scale * Math.sinh((q / 255) * Math.asinh(tr.hi / tr.scale));
  let h = "";
  if (hit.type === "weight") {
    const p = hit.panel, fr = p.row_factor, fc = p.col_factor;
    const where = p.layer < 0 ? "" : p.kind === "lm_head" ? "" : ` · layer ${p.layer}`;
    h += `<b>${p.kind}${where}</b> <span class="k"></span><br>`;
    h += `<span class="k">shape</span>${p.true_shape[0].toLocaleString()} × ${p.true_shape[1].toLocaleString()} · ${(p.params / 1e6).toFixed(2)} M params<br>`;
    h += `<span class="k">panel RMS</span>${p.rms.toFixed(4)} · max |w| ${p.max_abs.toFixed(3)}<br>`;
    h += `<span class="k">outliers</span>${(p.outlier_frac * 100).toFixed(3)}% of weights > 4×RMS<br>`;
    const r0 = hit.row * fr, c0 = hit.col * fc;
    const rowName = p.kind === "embed" || p.kind === "lm_head" ? "tokens" : (p.family === "mlp" ? "neurons" : "dims");
    h += `<span class="k">here</span>${rowName} ${r0}–${Math.min(r0 + fr, p.true_shape[0]) - 1}, residual ${c0}–${c0 + fc - 1}<br>`;
    h += `<span class="k">block RMS</span><span class="hl">${asinhVal(hit.q).toFixed(4)}</span>`;
    if (p.family === "attn" && p.kind !== "k" && p.kind !== "v") h += ` · head ${Math.floor(r0 / (p.true_shape[0] / (S.index.meta.n_heads || 16)))}`;
    if (p.note) h += `<br><span class="k"></span><span class="muted">${p.note}</span>`;
  } else if (hit.type === "norm") {
    const n = hit.strip, nt = S.man.norms.transform;
    h += `<b>${n.name}${n.name === "ln_final" ? "" : " · layer " + n.layer}</b><br>`;
    h += `<span class="k">RMSNorm gain</span>mean ${n.mean.toFixed(3)} · max ${n.max.toFixed(2)}<br>`;
    h += `<span class="k">here (max)</span><span class="hl">${(hit.q / 255 * nt.hi).toFixed(3)}</span>`;
  } else if (hit.type === "resid") {
    h += `<b>residual stream · ${hit.layer === S.man.n_layers ? "output" : "into layer " + hit.layer}</b>`;
  }
  if (hit.act != null) h += `<br><span class="k">firing now</span><span class="hl">${hit.act.toFixed(2)}</span> of peak · “${(S.frames.tokens[S.frame] || "").trim()}”`;
  tip.innerHTML = h; tip.hidden = false;
}
$("gl").addEventListener("pointermove", e => {
  if (S.view !== "arch" || !S.arch || e.buttons) { if (S.view === "arch") $("tip").hidden = true; return; }
  const r = e.target.getBoundingClientRect();
  const ndc = { x: (e.clientX - r.left) / r.width * 2 - 1, y: -((e.clientY - r.top) / r.height * 2 - 1) };
  S.lastNdc = ndc;
  const hit = S.arch.pick(ndc); S.lastPick = hit; inspect(hit);
  const tip = $("tip"), sr = $("gl").parentElement.getBoundingClientRect();
  let x = e.clientX - sr.left + 14, y = e.clientY - sr.top + 14;
  if (x + 300 > sr.width) x -= 330;
  tip.style.left = x + "px"; tip.style.top = y + "px";
});
$("gl").addEventListener("pointerleave", () => { $("tip").hidden = true; S.lastPick = null; });

boot();
