// Headless harness: boots the real viewer.js (a createViewer(rootEl, opts)
// factory) against compact THREE + DOM stubs so we can drive it in Node. The
// auto-boot runs first (proving the production boot survives the stubs), then a
// boot-capture shim builds one more viewer through the gated opts.expose hook so
// the suites can read instance-scoped internals. Stubs are deliberately thin —
// BufferAttribute stores the real typed array it's given, so geometry-attribute
// loops in viewer code operate on correctly-sized buffers.
const fs = require("fs");
const vm = require("vm");

function makeEl() {
  const e = {
    style: { setProperty() {}, removeProperty() {}, display: "", left: "", top: "", fontSize: "", cursor: "", background: "", color: "", width: "", height: "" },
    dataset: {}, files: [], value: "", checked: false, textContent: "", title: "", _html: "",
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {}, removeEventListener() {}, appendChild(c) { return c; }, removeChild(c) { return c; }, click() {},
    // Every element resolves selectors to a fresh stub element so viewer.js's
    // q(attr,id) and chained rootEl.querySelector(...) calls never hit undefined.
    querySelector() { return makeEl(); }, querySelectorAll() { return []; },
    getContext() { return ctx2d(); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 150 }; },
    // The factory sizes itself from clientWidth/clientHeight (offsetWidth fallback).
    clientWidth: 1200, clientHeight: 800, offsetWidth: 1200, offsetHeight: 800,
    toDataURL() { return "data:image/png;base64,iVBORw0KGgo="; },
    width: 300, height: 150, length: 0,
  };
  Object.defineProperty(e, "innerHTML", { set(v) { this._html = v; }, get() { return this._html; } });
  return e;
}
function ctx2d() { return new Proxy({}, { get() { return () => {}; } }); }

let downloads = [];
const document = {
  getElementById: () => makeEl(),
  querySelector: () => makeEl(),
  querySelectorAll: () => [],
  createElement: (tag) => { const e = makeEl(); e.tagName = tag; e.click = () => downloads.push({ href: e.href, download: e.download }); return e; },
  // body + documentElement are real element stubs (NOT bare {classList}) so the
  // factory can createViewer(document.body) and call rootEl.querySelector(...).
  documentElement: makeEl(),
  body: makeEl(),
};

// ── THREE stub ────────────────────────────────────────────────────────────
function vec3(x = 0, y = 0, z = 0) {
  return {
    x, y, z,
    set(a, b, c) { this.x = a; this.y = b; this.z = c; return this; },
    copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; },
    clone() { return vec3(this.x, this.y, this.z); },
    add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; },
    sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; },
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; },
    addScaledVector(v, s) { this.x += v.x * s; this.y += v.y * s; this.z += v.z * s; return this; },
    normalize() { const m = Math.hypot(this.x, this.y, this.z) || 1; this.x /= m; this.y /= m; this.z /= m; return this; },
    dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; },
    distanceTo(v) { return Math.hypot(this.x - v.x, this.y - v.y, this.z - v.z); },
    length() { return Math.hypot(this.x, this.y, this.z); },
    lerpVectors(a, b, t) { this.x = a.x + (b.x - a.x) * t; return this; },
    project() { return this; },
    unproject() { return this; },
    setFromUnitVectors() { return this; },
  };
}
function node3() {
  return {
    children: [], visible: true, geometry: null, material: null,
    position: vec3(), scale: { setScalar() {} }, quaternion: { setFromUnitVectors() {} },
    add(c) { this.children.push(c); }, remove(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); },
    traverse(f) { f(this); this.children.forEach((c) => c.traverse && c.traverse(f)); },
  };
}
function geometry() {
  return {
    attrs: {},
    setAttribute(n, a) { this.attrs[n] = a; return this; },
    getAttribute(n) { return this.attrs[n]; },
    setFromPoints(p) { this._pts = p; return this; }, computeBoundingSphere() {}, dispose() {},
  };
}
function bufAttr(array) { return { array, needsUpdate: false, setXYZ() {} }; }
const THREE = {
  WebGLRenderer: function () { return { setPixelRatio() {}, setSize() {}, render() {} }; },
  WebGLRenderTarget: function () { return { setSize() {}, dispose() {}, texture: {} }; },
  Scene: function () { return node3(); },
  Color: function () { return { r: 1, g: 1, b: 1, getHex() { return 0; } }; },
  PerspectiveCamera: function () { const c = node3(); c.position = vec3(1, 1, 1); c.getWorldDirection = (v) => v.set(0, 0, -1); c.updateProjectionMatrix = () => {}; c.aspect = 1; return c; },
  OrbitControls: function () { const c = node3(); c.target = vec3(); c.update = () => {}; c.__l = {}; c.addEventListener = (t, f) => { (c.__l[t] || (c.__l[t] = [])).push(f); }; c.removeEventListener = (t, f) => { const a = c.__l[t]; if (a) { const i = a.indexOf(f); if (i >= 0) a.splice(i, 1); } }; c.minDistance = 0; c.maxDistance = 1e9; c.zoomSpeed = 0.5; c.enableDamping = true; c.dampingFactor = 0; c.autoRotate = false; c.autoRotateSpeed = 0; return c; },
  AmbientLight: function () { return node3(); },
  DirectionalLight: function () { const n = node3(); n.position = vec3(); return n; },
  Raycaster: function () { return { params: { Points: { threshold: 0 } }, setFromCamera() {}, intersectObject() { return []; } }; },
  Vector2: function () { return { x: 0, y: 0 }; },
  Vector3: function (x, y, z) { return vec3(x, y, z); },
  Group: function () { return node3(); },
  Mesh: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  Line: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  LineSegments: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  Points: function (g, m) { const n = node3(); n.geometry = g; n.material = m; n.frustumCulled = false; return n; },
  Sprite: function (m) { const n = node3(); n.material = m; return n; },
  SphereGeometry: function () { return geometry(); },
  RingGeometry: function () { return geometry(); },
  PlaneGeometry: function () { return geometry(); },
  BufferGeometry: function () { return geometry(); },
  ShaderMaterial: function (opts) { return { uniforms: (opts && opts.uniforms) || { opacity: { value: 1 } }, dispose() {} }; },
  MeshBasicMaterial: function () { return { dispose() {} }; },
  MeshStandardMaterial: function () { return { dispose() {} }; },
  LineBasicMaterial: function () { return { dispose() {} }; },
  SpriteMaterial: function () { return { dispose() {}, map: { dispose() {} } }; },
  CanvasTexture: function () { return { dispose() {} }; },
  BufferAttribute: function (array) { return bufAttr(array); },
  Float32BufferAttribute: function (array) { return bufAttr(Float32Array.from(array)); },
  BackSide: 0, DoubleSide: 2,
};

function run(viewerPath, D, opts) {
  const src = fs.readFileSync(viewerPath, "utf8");
  downloads = [];
  const location = { hash: (opts && opts.hash) || "", href: "http://x/" };
  const winListeners = {};
  const parentPosts = [];
  let exposed = null; // filled by opts.expose via the boot-capture shim
  const sandbox = {
    THREE, document, console, location,
    history: { replaceState(s, t, u) { const i = String(u).indexOf("#"); location.hash = i >= 0 ? String(u).slice(i) : ""; location.href = "http://x/" + String(u); } },
    navigator: {},
    parent: { postMessage: (m) => parentPosts.push(m) },
    matchMedia: () => ({ matches: false }),
    innerWidth: 1200, innerHeight: 800, devicePixelRatio: 1,
    requestAnimationFrame: () => 0, cancelAnimationFrame: () => {}, setTimeout: () => 0, clearTimeout: () => {},
    URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
    Blob: function () {}, FileReader: function () { this.readAsText = () => {}; },
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    escape, unescape, encodeURIComponent, decodeURIComponent,
    window: { addEventListener: (t, f) => { (winListeners[t] || (winListeners[t] = [])).push(f); } },
    Math, JSON, Set, Map, Array, Object, String, Number, isFinite, parseFloat, parseInt, Float32Array, Infinity, NaN,
    // Streaming-client primitives need these intrinsics; expose the OUTER realm's
    // copies so typed arrays/buffers built in a test are identity-compatible.
    Promise, Uint8Array, Uint16Array, Uint32Array, Int32Array, DataView, ArrayBuffer,
    structuredClone: typeof structuredClone === "function" ? structuredClone : (x) => x,
    // The factory registers all DOM listeners through an AbortController (its
    // dispose() calls ac.abort()); supply the outer realm's global.
    AbortController,
    ResizeObserver: undefined, // factory guards `typeof ResizeObserver` — leave it absent
  };
  sandbox.window.innerWidth = 1200;
  sandbox.globalThis = sandbox;
  sandbox.D = D;
  // Tests may inject extra globals (e.g. a `fetch`/`indexedDB`/`Worker` stub for
  // the ServerSource / TileCache suites) via opts.globals.
  if (opts && opts.globals) Object.assign(sandbox, opts.globals);
  const ctx = vm.createContext(sandbox);
  ctx.__captureExpose = (o) => { exposed = o; };
  ctx.__dl = downloads;

  // Boot-capture shim: after the source defines createViewer + runs its
  // auto-boot (which sets window.viewer), build ONE more viewer through the
  // expose hook on the same rootEl. We assert on this instance; its handle
  // methods + exposed internals are what __vt forwards.
  const bootShim =
    "\n;globalThis.__vtHandle=createViewer(document.body,{expose:globalThis.__captureExpose});" +
    "globalThis.__vtHandle.rebuild(typeof D!=='undefined'&&D?D:{points:[]});" +
    "if(globalThis.__vtHandle.applyViewHash)globalThis.__vtHandle.applyViewHash();";

  // Export shim: __vt = { module symbols } ∪ { handle methods }. The exposed
  // instance internals are merged from the host side via defineProperties
  // (below) so their live getters/setters keep their descriptors.
  const shim =
    "\n;globalThis.__vt=Object.assign({}," +
    "{parseScene,deriveStrength," +
    "decodeTile:(typeof decodeTile!=='undefined'?decodeTile:undefined)," +
    "catOrder:(typeof catOrder!=='undefined'?catOrder:undefined)," +
    "stratify:(typeof stratify!=='undefined'?stratify:undefined)," +
    "tileQuery:(typeof tileQuery!=='undefined'?tileQuery:undefined)," +
    "safeColor:(typeof safeColor!=='undefined'?safeColor:undefined)," +
    "InlineSource:(typeof InlineSource!=='undefined'?InlineSource:undefined)," +
    "ServerSource:(typeof ServerSource!=='undefined'?ServerSource:undefined)," +
    "TileCache:(typeof TileCache!=='undefined'?TileCache:undefined)," +
    "makeWorkerDecoder:(typeof makeWorkerDecoder!=='undefined'?makeWorkerDecoder:undefined)," +
    "TileStreamer:(typeof TileStreamer!=='undefined'?TileStreamer:undefined)," +
    "tileMeshSink:(typeof tileMeshSink!=='undefined'?tileMeshSink:undefined)," +
    "connectToServer:(typeof connectToServer!=='undefined'?connectToServer:undefined)," +
    "disconnectServer:(typeof disconnectServer!=='undefined'?disconnectServer:undefined)," +
    "selectStreamRow:(typeof selectStreamRow!=='undefined'?selectStreamRow:undefined)," +
    "renderDiagnostics:(typeof renderDiagnostics!=='undefined'?renderDiagnostics:undefined)}," +
    "globalThis.__vtHandle);";

  vm.runInContext(src + bootShim + shim, ctx, { filename: "viewer.js" });
  // Merge the LIVE exposed getters/setters onto __vt — copy the property
  // *descriptors*, not their snapshotted values, so vt.N / vt.radialG=… stay
  // bound to the current closure state (Object.assign would snapshot them).
  if (exposed) Object.defineProperties(ctx.__vt, Object.getOwnPropertyDescriptors(exposed));
  ctx.__vt.downloads = downloads;

  // Test helpers for the compare-embed path (unchanged contract).
  ctx.__vt.parentPosts = parentPosts;
  ctx.__vt.fireWindow = (type, ev) => {
    // Default a message event's source to the parent stub (the compare host), so
    // viewer.js's `e.source !== parent` guard accepts it. Tests can set ev.source
    // explicitly to simulate a foreign window.
    if (type === "message" && ev && ev.source === undefined) ev.source = sandbox.parent;
    (winListeners[type] || []).forEach((f) => f(ev));
  };
  ctx.__vt.parentStub = sandbox.parent;
  ctx.__vt.fireCtrlChange = () => { const c = ctx.__vt.controls; (c.__l && c.__l.change || []).forEach((f) => f()); };
  return ctx.__vt;
}
module.exports = { run };
