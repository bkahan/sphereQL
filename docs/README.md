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
  and random projection map embeddings onto S².
- [Empirical findings](empirical-findings.md) — when each projection wins,
  with measured scores.
- [Use cases](use-cases.md) — what problem each feature is meant to solve.

## Guides

- [Auto-tuning & meta-learning](auto-tuning.md) — `PipelineConfig`,
  `auto_tune`, `MetaModel`, `FeedbackEvent` — the full metalearning framework.

## Reference

- [Performance](performance.md) — index internals, benchmark numbers,
  speed/precision tradeoffs.
- [Examples](examples.md) — catalog of runnable examples across Rust, Python,
  and WASM.
- [Testing](testing.md) — running tests locally; CI pipeline.
- [Project status](project-status.md) — alpha surface, known limitations,
  roadmap.
- [Benchmark analysis](benchmark-analysis.md) — annotated results on a
  10k-point search benchmark.
- [Search precision roadmap](search-precision-roadmap.md) — tracked
  improvements to search quality.
