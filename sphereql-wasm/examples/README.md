# sphereQL WASM — browser demo

Self-contained HTML page exercising the full pipeline + metalearning
surface in the browser. No build system, no bundler — just `wasm-pack`
+ any static file server.

## Build

From the `sphereql-wasm/` directory:

```bash
wasm-pack build --target web
```

This produces `sphereql-wasm/pkg/` with the `.wasm` + JS glue. The demo
imports from `./pkg/sphereql_wasm.js` relative to `index.html`, so copy
or symlink the generated `pkg/` next to `examples/index.html`:

```bash
# one-time copy
cp -r pkg examples/pkg
```

Then serve the `examples/` directory. Any static server works — one
option is Python's built-in:

```bash
cd examples
python3 -m http.server 8000
```

Open <http://localhost:8000/>.

## What the demo covers

- **Pipeline construction** — `new Pipeline(inputJson)` with the default
  PCA projection, plus k-NN search.
- **`newWithConfig`** — build with an explicit `PipelineConfig`
  (projection family, routing, inner sphere, bridges, Laplacian).
- **`corpusFeatures` + `autoTune`** — extract a profile then tune over a
  custom `search_space`. The report dict is displayed verbatim.
- **`hierarchicalNearest`** side-by-side with plain `nearest`.
- **`domainGroups` + `projectionWarnings`** from the tuned pipeline.
- **`MetaModel`** — fit `NearestNeighborMetaModel` on a synthesized
  record set, then `predict` a config for a new corpus profile.
- **`FeedbackAggregator`** — record + summarize user satisfaction
  events. Persistence is caller-owned (browsers have no filesystem —
  use `localStorage`, `IndexedDB`, or ship JSON to a server).

The corpus is synthesized inline: four categories × eight 16-d vectors,
with one dominant axis plus a shared cross-talk axis per category.
Small enough to reason about, large enough to produce non-trivial
bridges and domain groups.
