// Headless test of worker.js message protocol: stub importScripts + the
// no-modules `wasm_bindgen` global, drive the worker, and assert the
// postMessage shapes (ready, ok, error, queue-before-ready).
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const WORKER = path.join(__dirname, "..", "sphereql-wasm", "studio", "worker.js");

let fails = 0;
const ok = (c, m) => { if (!c) { console.log("FAIL:", m); fails++; } else console.log("ok  :", m); };

const posted = [];
let onmsg = null;

function LinguaStudio() {}
LinguaStudio.prototype.process = function (text) {
  return JSON.stringify({
    title: "Lingua studio",
    points: text ? [{ x: 1, y: 0, z: 0, cat: "c", label: "a", id: "a" }] : [],
    overlays: [], stats: { projection_kind: "lingua", evr: 0, evr_label: "mean salience" },
    surface_radius: 1, show_axes: false,
  });
};
function Pipeline(corpus) { this.corpus = corpus; }
Pipeline.prototype.buildSceneJson = function (title) {
  return JSON.stringify({
    title, points: [{ x: 1, y: 0, z: 0, cat: "s", label: "p", id: "s-0" }],
    overlays: [], stats: { projection_kind: "pca", evr: 0.5, evr_label: "PCA variance" },
    surface_radius: 1, show_axes: false,
  });
};
Pipeline.prototype.free = function () { this.freed = true; };
Pipeline.prototype.nearest = function (query, k) {
  return [
    { id: "s-0", category: "science", distance: 0.10 },
    { id: "s-1", category: "cooking", distance: 0.31 },
    { id: "s-2", category: "science", distance: 0.42 },
  ].slice(0, k);
};
Pipeline.newWithConfig = function (corpus) { return new Pipeline(corpus); };

const wasm_bindgen = function () { return Promise.resolve(); };
wasm_bindgen.LinguaStudio = LinguaStudio;
wasm_bindgen.Pipeline = Pipeline;

const sandbox = {
  importScripts: () => {},
  wasm_bindgen,
  postMessage: (m) => posted.push(m),
  get onmessage() { return onmsg; },
  set onmessage(f) { onmsg = f; },
  Promise, String, JSON, Error, setTimeout, console,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(WORKER, "utf8"), sandbox, { filename: "worker.js" });

(async () => {
  // Before wasm is ready, a message must be queued (not dropped, not crash).
  ok(typeof onmsg === "function", "worker registered onmessage");
  onmsg({ data: { id: 1, kind: "lingua", text: "hello world" } });
  ok(posted.length === 0, "message before ready is queued, nothing posted yet");

  // Flush microtasks so the wasm-init .then runs.
  for (let i = 0; i < 5; i++) await Promise.resolve();

  ok(posted.some((m) => m.type === "ready"), "posts {type:'ready'} after init");
  const replay = posted.find((m) => m.id === 1);
  ok(replay && replay.ok === true, "queued lingua request is replayed on ready");
  ok(replay && JSON.parse(replay.json).title === "Lingua studio", "replayed json is a Scene");

  // Corpus request, handled live.
  onmsg({ data: { id: 2, kind: "corpus", corpus: "{}", config: null, title: "Corpus" } });
  const corp = posted.find((m) => m.id === 2);
  ok(corp && corp.ok === true, "corpus request handled");
  ok(corp && JSON.parse(corp.json).points.length === 1, "corpus scene has points");

  // Corpus with config exercises the newWithConfig path.
  onmsg({ data: { id: 3, kind: "corpus", corpus: "{}", config: '{"projection_kind":"UmapSphere"}', title: "C" } });
  ok(posted.find((m) => m.id === 3 && m.ok), "corpus+config request handled (newWithConfig)");

  // Unknown kind → graceful per-request error, not a crash.
  onmsg({ data: { id: 4, kind: "bogus" } });
  const err = posted.find((m) => m.id === 4);
  ok(err && err.ok === false && /unknown kind/.test(err.error), "unknown kind → {ok:false,error}");

  // Query against the corpus built by id:2 (the worker keeps that pipeline).
  onmsg({ data: { id: 5, kind: "query", query: "[0.9,0.1,0,0.2]", k: 2 } });
  const q = posted.find((m) => m.id === 5);
  ok(q && q.ok === true && Array.isArray(q.neighbors), "query → {ok:true,neighbors:[…]}");
  ok(q && q.neighbors.length === 2 && q.neighbors[0].id === "s-0", "neighbors honor k and carry ids");

  // 'load' primes a pipeline without returning a scene; query then works.
  onmsg({ data: { id: 6, kind: "load", corpus: "{}", config: null } });
  const ld = posted.find((m) => m.id === 6);
  ok(ld && ld.ok === true && ld.loaded === true && ld.json === undefined, "load → {ok:true,loaded:true} (no scene)");
  onmsg({ data: { id: 7, kind: "query", query: "[1]", k: 1 } });
  ok(posted.find((m) => m.id === 7 && m.ok && Array.isArray(m.neighbors)), "query works after a load (primed)");

  console.log("--- fresh worker: query before any corpus ---");
  await freshQueryFirst();

  console.log(fails === 0 ? "\nALL PASS" : "\n" + fails + " FAILURES");
  process.exit(fails === 0 ? 0 : 1);
})();

// A second worker instance with no corpus yet must reject a query gracefully.
async function freshQueryFirst() {
  const posted2 = [];
  let onmsg2 = null;
  const sb = {
    importScripts: () => {},
    wasm_bindgen,
    postMessage: (m) => posted2.push(m),
    get onmessage() { return onmsg2; },
    set onmessage(f) { onmsg2 = f; },
    Promise, String, JSON, Error, setTimeout, console,
  };
  sb.globalThis = sb;
  vm.createContext(sb);
  vm.runInContext(fs.readFileSync(WORKER, "utf8"), sb, { filename: "worker.js" });
  for (let i = 0; i < 5; i++) await Promise.resolve();
  onmsg2({ data: { id: 9, kind: "query", query: "[0.1]", k: 3 } });
  const e = posted2.find((m) => m.id === 9);
  ok(e && e.ok === false && /run a corpus first/.test(e.error), "query before corpus → graceful error");
}
