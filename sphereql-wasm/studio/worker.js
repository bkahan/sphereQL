// SphereQL Studio worker — runs the wasm pipeline OFF the main thread so a
// large corpus or a long paste never freezes the UI. Classic worker (so it can
// `importScripts` the `--target no-modules` wasm glue, which exposes a global
// `wasm_bindgen` init + the exported classes). It posts back Scene JSON; the
// main thread hands that to the viewer's rebuild().
//
// Message protocol
//   main → worker: { id, kind:'lingua', text }
//                  { id, kind:'corpus', corpus, config?, title? }
//                  { id, kind:'load',   corpus, config? }  build pipeline, no scene
//                  { id, kind:'query',  query, k? }     against the last corpus/load
//   worker → main: { type:'ready' }                     once wasm is initialized
//                  { id, ok:true, json }                a Scene JSON string
//                  { id, ok:true, neighbors }           nearest hits [{id,…}]
//                  { id, ok:true, loaded:true }         pipeline primed (no scene)
//                  { id, ok:false, error }              per-request failure
//                  { type:'fatal', error }              wasm failed to load

/* global wasm_bindgen, importScripts */
importScripts("pkg/sphereql_wasm.js");

let studio = null; // reused LinguaStudio instance
let pipeline = null; // last corpus Pipeline, kept alive so queries can reuse it
let ready = false;
let queued = null; // newest message received before wasm finished loading

wasm_bindgen("pkg/sphereql_wasm_bg.wasm")
  .then(() => {
    studio = new wasm_bindgen.LinguaStudio();
    ready = true;
    postMessage({ type: "ready" });
    if (queued) {
      const q = queued;
      queued = null;
      handle(q);
    }
  })
  .catch((e) => postMessage({ type: "fatal", error: String((e && e.message) || e) }));

function handle(msg) {
  const { id, kind } = msg;
  try {
    if (kind === "lingua") {
      postMessage({ id, ok: true, json: studio.process(msg.text || "") });
    } else if (kind === "corpus") {
      if (pipeline && pipeline.free) pipeline.free(); // release the previous corpus
      pipeline = msg.config
        ? wasm_bindgen.Pipeline.newWithConfig(msg.corpus, msg.config)
        : new wasm_bindgen.Pipeline(msg.corpus);
      postMessage({ id, ok: true, json: pipeline.buildSceneJson(msg.title || "Corpus") });
    } else if (kind === "load") {
      // Prime the pipeline (for queries) without producing a scene — keeps the
      // currently displayed scene (e.g. the baked demo) untouched.
      if (pipeline && pipeline.free) pipeline.free();
      pipeline = msg.config
        ? wasm_bindgen.Pipeline.newWithConfig(msg.corpus, msg.config)
        : new wasm_bindgen.Pipeline(msg.corpus);
      postMessage({ id, ok: true, loaded: true });
    } else if (kind === "query") {
      if (!pipeline) throw new Error("run a corpus first");
      const neighbors = pipeline.nearest(msg.query, msg.k || 8);
      postMessage({ id, ok: true, neighbors });
    } else {
      throw new Error("unknown kind: " + kind);
    }
  } catch (e) {
    postMessage({ id, ok: false, error: String((e && e.message) || e) });
  }
}

onmessage = (e) => {
  const msg = e.data;
  if (!ready) {
    queued = msg; // keep only the newest pending request
    return;
  }
  handle(msg);
};
