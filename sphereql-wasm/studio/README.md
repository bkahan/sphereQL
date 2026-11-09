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

## Compare & morph

In **Corpus JSON** mode the studio adds two ways to compare projections:

- **Morph slider** — pick a second projection under *morph →*; the studio
  builds it as a target and the slider interpolates every point from its
  position in the current projection (A) to its id-matched position in the
  target (B) along the shell. t=0 is A, t=1 is B. Points with no id match stay
  put. This is one viewport, so it reads as an animation between projections.

- **Side by side** (`compare.html`) — two viewer iframes (`embed.html#embed`),
  each fed a Scene for a different projection of the same corpus by one shared
  worker. The panes broadcast their camera moves to the parent, which relays
  them to the other pane, so orbiting one orbits both. The relay can't echo
  into a feedback loop: the viewer epsilon-gates its own broadcasts so the
  damping that follows an applied camera update is swallowed.

Both align points by **stable id** (`s-0000…`), so morph/relay only ever match
the same item across projections.

## Packaging

Separate-file is the default (smaller `index.html`, the `.wasm` loads
alongside). A single-file export (worker + base64-inlined `.wasm` folded into
one HTML) is a planned opt-in for sharing a studio as one attachment.

`dist/` is generated — it is git-ignored.
