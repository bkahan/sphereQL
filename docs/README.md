# sphereQL Documentation

Full documentation for the [sphereQL](../README.md) project. The top-level
README carries an elevator pitch and a minimal Rust quickstart; every deeper
topic lives here.

## Getting started

- [Rust quickstart](quickstart-rust.md) — full workspace tour: core math,
  indexing, layout, embedding, auto-tuning, GraphQL.
- [Python quickstart](quickstart-python.md) — `pip install sphereql`, semantic
  search, 3D visualization, vector database bridges.
- [WASM quickstart](quickstart-wasm.md) — in-browser pipeline via
  `wasm-bindgen`.

## Architecture & concepts

- [Architecture](architecture.md) — workspace crates, dependency graph,
  feature-flag composition.
- [Coordinate system](coordinate-system.md) — physics convention for
  (r, θ, φ); geographic conversion.
- [Projections](projections.md) — how PCA, Kernel PCA, Laplacian eigenmap,
  and UMAP-on-sphere map embeddings onto S² (plus the low-level random
  projection).
- [Empirical findings](empirical-findings.md) — when each projection wins,
  with measured scores.
- [Visualization](visualization.md) — running the 3D viewer (offline file,
  WASM studio, streaming server), the `Scene`/`Manifest`/`SQT1` wire contracts,
  the `viewer.js` runtime, the HTTP API, and the generic-vs-coupled extraction
  map.
- [Use cases](use-cases.md) — what problem each feature is meant to solve.

## Guides

- [Auto-tuning & meta-learning](auto-tuning.md) — `PipelineConfig`,
  `auto_tune`, `MetaModel`, `FeedbackEvent` — the full metalearning framework.
- [Lingua pipeline](../sphereql-lingua/README.md) — six-stage text →
  `ConceptGraph` pipeline that places every concept at a SphereQL
  `(r, θ, φ)` position. Python-side skeleton lives in
  [`lingua-spherica`](../lingua-spherica/README.md).

## Reference

- [Performance](performance.md) — index internals, benchmark numbers,
  speed/precision tradeoffs.
- [Examples](examples.md) — catalog of runnable examples across Rust, Python,
  and WASM.
- [Testing](testing.md) — running tests locally; CI pipeline.
- [Project status](project-status.md) — release status, known limitations,
  roadmap.
- [Benchmark analysis](benchmark-analysis.md) — annotated results on a
  10k-point search benchmark.
- [Search precision roadmap](search-precision-roadmap.md) — tracked
  improvements to search quality.
