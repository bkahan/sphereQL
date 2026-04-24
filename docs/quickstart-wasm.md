# WASM quickstart

WebAssembly bindings for running the embedding pipeline (including
category enrichment) in the browser.

As with the Python bindings, the WASM build is PCA / Kernel-PCA-only
in 0.1.x — Laplacian projection, `auto_tune`, and the `MetaModel`
layer are Rust-only for now.

## Build

```bash
cd sphereql-wasm
wasm-pack build --target web
```

## Use

```javascript
import init, { Pipeline } from './pkg/sphereql_wasm.js';

await init();

const pipeline = new Pipeline(JSON.stringify({
  categories: ["science", "cooking", "sports"],
  embeddings: [[0.1, 0.9, 0.3], [0.9, 0.1, 0.0], [0.4, 0.4, 0.8]]
}));

// k-NN search (returns JSON string)
const results = pipeline.nearest(
  JSON.stringify([0.15, 0.85, 0.35]),
  3
);
console.log(JSON.parse(results));

// Category enrichment
const catPath = JSON.parse(pipeline.category_concept_path("science", "cooking"));
const neighbors = JSON.parse(pipeline.category_neighbors("science", 2));
const stats = JSON.parse(pipeline.category_stats());
const drillDown = JSON.parse(pipeline.drill_down(
  "science", 5, JSON.stringify([0.15, 0.85, 0.35])
));
```
