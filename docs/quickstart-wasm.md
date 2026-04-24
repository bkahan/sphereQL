# WASM quickstart

WebAssembly bindings for running the embedding pipeline (including
category enrichment, Laplacian eigenmap, `auto_tune`, the `MetaModel`
layer, and `FeedbackAggregator`) in the browser or Node.

All pipeline methods now return **typed TypeScript values** via
[`tsify`](https://github.com/madonoharu/tsify) — no `JSON.parse` step
on the JS side. `wasm-pack` emits a `.d.ts` with a named interface
for every payload.

## Build

```bash
cd sphereql-wasm
wasm-pack build --target web
```

## Use

```typescript
import init, {
  Pipeline,
  LaplacianEigenmapProjection,
  autoTune,
} from './pkg/sphereql_wasm.js';

await init();

const pipeline = new Pipeline(JSON.stringify({
  categories: ["science", "cooking", "sports"],
  embeddings: [[0.1, 0.9, 0.3], [0.9, 0.1, 0.0], [0.4, 0.4, 0.8]]
}));

// k-NN search — returns NearestOut[]
const results = pipeline.nearest(
  JSON.stringify([0.15, 0.85, 0.35]),
  3
);
for (const r of results) {
  console.log(`${r.id} ${r.category} d=${r.distance.toFixed(4)}`);
}

// Category enrichment — all typed
const catPath = pipeline.category_concept_path("science", "cooking");
const neighbors = pipeline.category_neighbors("science", 2);
const stats = pipeline.category_stats();
const drillDown = pipeline.drill_down(
  "science", 5, JSON.stringify([0.15, 0.85, 0.35])
);
```

## Laplacian eigenmap

```typescript
import { LaplacianEigenmapProjection } from './pkg/sphereql_wasm.js';

const proj = new LaplacianEigenmapProjection(
  JSON.stringify(embeddings),
  '"magnitude"', // radial
  15,            // k_neighbors
  0.05,          // active_threshold
);
console.log('connectivity_ratio:', proj.connectivityRatio);
const point = JSON.parse(proj.project(JSON.stringify(query)));

// Or build a pipeline with Laplacian as the outer projection:
const pipeline = Pipeline.newWithConfig(
  JSON.stringify({ categories, embeddings }),
  JSON.stringify({ projection_kind: "LaplacianEigenmap" })
);
```

## Auto-tune + MetaModel

```typescript
import {
  autoTune,
  NearestNeighborMetaModel,
  corpusFeatures,
} from './pkg/sphereql_wasm.js';

const report = autoTune(
  JSON.stringify({ categories, embeddings }),
  JSON.stringify({ metric: "default_composite", budget: 16 })
);
console.log("best score:", report.best_score);

// Build the tuned pipeline from the returned config.
const tuned = Pipeline.newWithConfig(
  JSON.stringify({ categories, embeddings }),
  JSON.stringify(report.best_config)
);

// MetaModel for warm-start predictions
const model = new NearestNeighborMetaModel();
model.fit(/* JSON array of MetaTrainingRecord */);
const features = corpusFeatures(JSON.stringify({ categories, embeddings }));
const predicted = model.predict(JSON.stringify(features));
```
