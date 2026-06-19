// Headless harness: boots the real viewer.js against compact THREE + DOM stubs
// so we can drive parseScene()/rebuild() in Node. Stubs are deliberately thin —
// BufferAttribute stores the real typed array it's given, so geometry-attribute
// loops in viewer code operate on correctly-sized buffers.
const fs = require("fs");
const vm = require("vm");

function makeEl() {
  const e = {
    style: { setProperty() {}, display: "", left: "", top: "", fontSize: "", cursor: "", background: "", color: "", width: "" },
    dataset: {}, files: [], value: "", checked: false, textContent: "", title: "", _html: "",
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {}, removeEventListener() {}, appendChild() {}, removeChild() {}, click() {},
    querySelector() { return makeEl(); }, querySelectorAll() { return []; },
    getContext() { return ctx2d(); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 150 }; },
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
  querySelectorAll: () => [],
  createElement: (tag) => { const e = makeEl(); e.tagName = tag; e.click = () => downloads.push({ href: e.href, download: e.download }); return e; },
  documentElement: { style: { setProperty() {} } },
  body: { classList: { toggle() {} } },
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
  Scene: function () { return node3(); },
  Color: function () { return { r: 1, g: 1, b: 1, getHex() { return 0; } }; },
  PerspectiveCamera: function () { const c = node3(); c.position = vec3(1, 1, 1); c.getWorldDirection = (v) => v.set(0, 0, -1); c.updateProjectionMatrix = () => {}; c.aspect = 1; return c; },
  OrbitControls: function () { const c = node3(); c.target = vec3(); c.update = () => {}; c.__l = {}; c.addEventListener = (t, f) => { (c.__l[t] || (c.__l[t] = [])).push(f); }; c.minDistance = 0; c.maxDistance = 1e9; c.zoomSpeed = 0.5; c.enableDamping = true; c.dampingFactor = 0; c.autoRotate = false; c.autoRotateSpeed = 0; return c; },
  AmbientLight: function () { return node3(); },
  DirectionalLight: function () { const n = node3(); n.position = vec3(); return n; },
  Raycaster: function () { return { params: { Points: { threshold: 0 } }, setFromCamera() {}, intersectObject() { return []; } }; },
  Vector2: function () { return { x: 0, y: 0 }; },
  Vector3: function (x, y, z) { return vec3(x, y, z); },
  Group: function () { return node3(); },
  Mesh: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  Line: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  LineSegments: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  Points: function (g, m) { const n = node3(); n.geometry = g; n.material = m; return n; },
  SphereGeometry: function () { return geometry(); },
  RingGeometry: function () { return geometry(); },
  PlaneGeometry: function () { return geometry(); },
  BufferGeometry: function () { return geometry(); },
  ShaderMaterial: function (opts) { return { uniforms: (opts && opts.uniforms) || { opacity: { value: 1 } }, dispose() {} }; },
  MeshBasicMaterial: function () { return { dispose() {} }; },
  MeshStandardMaterial: function () { return { dispose() {} }; },
  LineBasicMaterial: function () { return { dispose() {} }; },
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
  const sandbox = {
    THREE, document, console, location,
    history: { replaceState(s, t, u) { const i = String(u).indexOf("#"); location.hash = i >= 0 ? String(u).slice(i) : ""; location.href = "http://x/" + String(u); } },
    navigator: {},
    parent: { postMessage: (m) => parentPosts.push(m) },
    matchMedia: () => ({ matches: false }),
    innerWidth: 1200, innerHeight: 800, devicePixelRatio: 1,
    requestAnimationFrame: () => 0, setTimeout: () => 0,
    URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
    Blob: function () {}, FileReader: function () { this.readAsText = () => {}; },
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    escape, unescape, encodeURIComponent, decodeURIComponent,
    window: { addEventListener: (t, f) => { (winListeners[t] || (winListeners[t] = [])).push(f); } },
    Math, JSON, Set, Map, Array, Object, String, Number, isFinite, parseFloat, parseInt, Float32Array, Infinity, NaN,
    // Streaming-client primitives need these intrinsics; expose the OUTER
    // realm's copies so typed arrays/buffers built in a test are identity-
    // compatible with what viewer.js produces.
    Promise, Uint8Array, Uint16Array, Uint32Array, Int32Array, DataView, ArrayBuffer,
  };
  sandbox.window.innerWidth = 1200;
  sandbox.globalThis = sandbox;
  sandbox.D = D;
  // Tests may inject extra globals (e.g. a `fetch`/`indexedDB`/`Worker` stub for
  // the ServerSource / TileCache suites) via opts.globals.
  if (opts && opts.globals) Object.assign(sandbox, opts.globals);
  const ctx = vm.createContext(sandbox);
  // Expose internals for assertions by appending an export shim.
  const shim = "\n;globalThis.__vt={parseScene,rebuild,setRuler,rulerAddPick,shareLink,applyViewHash,exportPNG,applyScale,currentSettings," +
    "get N(){return N;},get pts(){return pts;},get catSet(){return catSet;},get SR(){return SR;},get overlayKinds(){return [...overlayKinds];}," +
    "get rulerOn(){return rulerOn;},get rulerPicks(){return rulerPicks;},get rulerLast(){return rulerLast;}," +
    "get curScale(){return curScale;},get radialG(){return radialG;},get spreadF(){return spreadF;}," +
    "set radialG(v){radialG=v;},set spreadF(v){spreadF=v;},get camera(){return camera;},get controls(){return controls;},get downloads(){return globalThis.__dl;},get rulerGroup(){return rulerGroup;}," +
    "get catDir(){return catDir;},get uniforms(){return pointsMat?pointsMat.uniforms:null;},applyTransform,pickEncode,pickDecode,getHovered," +
    "highlightByIds,clearQuery,get queryGroup(){return queryGroup;},get idCount(){return idToIndex.size;}," +
    "setMorphTarget,applyMorph,clearMorph,curPos,get morphT(){return morphT;}," +
    "setPinMode,addPin,clearPins,currentSettings,applySettings,get pins(){return pins;},get pinOn(){return pinOn;},get pinGroup(){return pinGroup;},get zoomLocked(){return zoomLocked;}," +
    "decodeTile,catOrder,stratify,tileQuery,InlineSource,ServerSource,TileCache,makeWorkerDecoder,TileStreamer,tileMeshSink,connectToServer,disconnectServer,safeColor,selectStreamRow,renderDiagnostics,get streamStreamer(){return streamStreamer;},get dataSource(){return dataSource;}," +
    "get densityArr(){return pointsGeo&&pointsGeo.getAttribute('density')&&pointsGeo.getAttribute('density').array;},get densityOn(){return pointsMat?pointsMat.uniforms.densityOn.value:0;}};";
  ctx.__dl = downloads;
  vm.runInContext(src + shim, ctx, { filename: "viewer.js" });
  // Test helpers for the compare-embed path.
  ctx.__vt.parentPosts = parentPosts;
  ctx.__vt.fireWindow = (type, ev) => {
    // Default a message event's source to the parent stub (the compare host),
    // so viewer.js's `e.source !== parent` guard accepts it. Tests can set
    // ev.source explicitly to simulate a foreign window.
    if (type === "message" && ev && ev.source === undefined) ev.source = sandbox.parent;
    (winListeners[type] || []).forEach((f) => f(ev));
  };
  ctx.__vt.parentStub = sandbox.parent;
  ctx.__vt.fireCtrlChange = () => { const c = ctx.__vt.controls; (c.__l && c.__l.change || []).forEach((f) => f()); };
  return ctx.__vt;
}
module.exports = { run };
