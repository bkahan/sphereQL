# SphereQL Studio

A live, in-browser projection studio. Paste prose and watch
[`sphereql-lingua`](../../sphereql-lingua) place each concept on the sphere as
you type; or switch to **Corpus JSON** mode to run the full embedding pipeline
(`sphereql-embed`) on category + vector data and pick a projection. Everything
runs client-side via WebAssembly — no server, no network round-trips.

It is the same viewer as the offline `sphereql-vis` export: the studio shell is
emitted from `Scene::to_html()` (three.js + the shared `viewer.js` inlined), so
the live studio and the baked HTML can never drift. The studio adds an input
panel and a **Web Worker** that runs the wasm off the main thread and feeds the
viewer's `rebuild()` with the resulting Scene.

## Build & run

```sh
./build.sh                          # → dist/index.html + worker.js + studio.js + pkg/
(cd dist && python -m http.server 8080)
# open http://localhost:8080/
```

A worker + wasm must be served over `http(s)` — opening `dist/index.html` via
`file://` will not work.

`build.sh` uses `wasm-pack` if it is installed (it applies `wasm-opt -Oz`
itself); otherwise it falls back to `cargo build --target wasm32-unknown-unknown
--release` + `wasm-bindgen --target no-modules` (+ `wasm-opt` if available). The
worker is a **classic** worker that `importScripts` the no-modules glue, so it
needs `wasm_bindgen` exposed as a global — hence `--target no-modules`.

## Architecture

```
index.html ── inlined viewer.js (global rebuild/parseScene) + studio chrome
   └─ studio.js (main thread)  ──postMessage──▶  worker.js (classic worker)
        ▲                                            └─ importScripts pkg/sphereql_wasm.js
        └────────────── Scene JSON ◀── LinguaStudio.process / Pipeline.buildSceneJson
```

- **studio.js** debounces input (≈320 ms), tags each request with an id, and
  drops stale worker responses (`m.id < latestId`) so the freshest paste wins.
- **worker.js** keeps one reused `LinguaStudio`; corpus runs build a `Pipeline`
  per request and `free()` it. Requests received before the wasm finishes
  loading are queued (newest only) and replayed on `ready`.
- The worker returns **Scene JSON strings**; the main thread runs them through
  the viewer's own `parseScene()` before `rebuild()`, so a studio-built scene
  gets the same validation as a dropped file.

## Packaging

Separate-file is the default (smaller `index.html`, the `.wasm` loads
alongside). A single-file export (worker + base64-inlined `.wasm` folded into
one HTML) is a planned opt-in for sharing a studio as one attachment.

`dist/` is generated — it is git-ignored.
