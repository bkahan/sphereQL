# sphereQL

[![CI](https://github.com/bkahan/sphereQL/actions/workflows/ci.yml/badge.svg)](https://github.com/bkahan/sphereQL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/sphereql.svg)](https://crates.io/crates/sphereql)
[![PyPI](https://img.shields.io/pypi/v/sphereql.svg)](https://pypi.org/project/sphereql/)

**Project high-dimensional embeddings onto a 3D sphere for fast semantic
search, spatial queries, category-aware exploration, and interactive
visualization.**

sphereQL maps vectors from any embedding model (OpenAI, Cohere,
sentence-transformers, etc.) onto spherical coordinates via one of four
projection families — linear PCA, kernel PCA with a Gaussian (RBF)
kernel, Laplacian eigenmap over a k-NN similarity graph, or random
projection — then indexes them with shell/sector partitioning for fast
nearest-neighbor lookups. A Category Enrichment Layer computes
inter-category relationships, classifies bridges (`Genuine` /
`OverlapArtifact` / `Weak`), and builds inner spheres for
high-resolution within-category search. sphereQL auto-tunes its
pipeline per corpus against a scalar `QualityMetric`; a meta-model
recalls winning configs from past tuner runs when a new corpus arrives.
Callable from Rust, Python, or the browser via WASM.

## Documentation

Full documentation lives under [`docs/`](docs/README.md).

- [Quickstart — Rust](docs/quickstart-rust.md) · [Python](docs/quickstart-python.md) · [WASM](docs/quickstart-wasm.md)
- [Architecture](docs/architecture.md) — workspace crates and feature flags
- [Projections](docs/projections.md) — how the four projection families work
- [Auto-tuning & meta-learning](docs/auto-tuning.md) — the metalearning framework
- [Empirical findings](docs/empirical-findings.md) — when does each projection win?
- [Examples catalog](docs/examples.md) · [Performance](docs/performance.md) · [Project status](docs/project-status.md)

## Install

```toml
# Cargo.toml
[dependencies]
sphereql = { version = "0.1", features = ["full"] }
```

```bash
# Python
pip install sphereql
```

See [architecture.md](docs/architecture.md) for feature-flag details.

## Rust — minimal example

```rust
use sphereql::embed::*;

// 1. Build a pipeline from categorized embeddings.
let input = PipelineInput {
    categories: vec![
        "science".into(), "science".into(),
        "cooking".into(), "cooking".into(),
    ],
    embeddings: vec![
        vec![0.1, 0.9, 0.3, 0.0],
        vec![0.2, 0.8, 0.4, 0.1],
        vec![0.9, 0.1, 0.0, 0.5],
        vec![0.8, 0.2, 0.1, 0.4],
    ],
};
let pipeline = SphereQLPipeline::new(input).unwrap();

// 2. Query nearest neighbors.
let query = PipelineQuery { embedding: vec![0.15, 0.85, 0.35, 0.05] };
let results = pipeline.query(SphereQLQuery::Nearest { k: 3 }, &query);
```

See the [Rust quickstart](docs/quickstart-rust.md) for spatial indexing,
the layout engine, GraphQL, and the full embedding pipeline.
[`auto-tuning.md`](docs/auto-tuning.md) covers the `PipelineConfig` +
`auto_tune` + `MetaModel` workflow end-to-end.

## Python — minimal example

```python
import sphereql

categories = ["science", "science", "cooking", "cooking"]
embeddings = [
    [0.1, 0.9, 0.3, 0.0],
    [0.2, 0.8, 0.4, 0.1],
    [0.9, 0.1, 0.0, 0.5],
    [0.8, 0.2, 0.1, 0.4],
]

pipeline = sphereql.Pipeline(categories, embeddings)
results = pipeline.nearest([0.15, 0.85, 0.35, 0.05], k=3)

# Interactive 3D visualization in your browser
sphereql.visualize(categories, embeddings, title="My Embeddings")
```

The Python bindings expose PCA and Kernel PCA today; Laplacian
projection, `auto_tune`, and the `MetaModel` layer are Rust-only in
0.1.x. See the [Python quickstart](docs/quickstart-python.md) for
semantic search, 3D visualization, vector database bridges, and the
core type surface.

## WASM — minimal example

```bash
cd sphereql-wasm && wasm-pack build --target web
```

```javascript
import init, { Pipeline } from './pkg/sphereql_wasm.js';
await init();

const pipeline = new Pipeline(JSON.stringify({
  categories: ["science", "cooking"],
  embeddings: [[0.1, 0.9, 0.3], [0.9, 0.1, 0.0]],
}));
const results = pipeline.nearest(JSON.stringify([0.15, 0.85, 0.35]), 1);
```

Same bindings coverage as Python. See the
[WASM quickstart](docs/quickstart-wasm.md) for category enrichment in
the browser.

## Workspace layout

| Crate | Role |
|---|---|
| `sphereql` | Umbrella crate with feature flags for selective imports. |
| `sphereql-core` | Spherical math — points, conversions, distance metrics, region types. |
| `sphereql-index` | Spatial indexing with shell + sector partitioning. |
| `sphereql-layout` | Layout engines (Fibonacci spiral, k-means, force-directed). |
| `sphereql-embed` | Projections, query pipeline, Category Enrichment Layer, metalearning framework. |
| `sphereql-graphql` | async-graphql schema with cone/shell/band/wedge queries + subscriptions. |
| `sphereql-vectordb` | Vector store bridge (InMemory, Qdrant, Pinecone) with hybrid search. |
| `sphereql-python` | Python bindings via PyO3/maturin. |
| `sphereql-wasm` | WASM bindings via wasm-bindgen. |
| `sphereql-corpus` | Shared example corpora (775-concept built-in + 300-concept stress). |

Full dependency graph and crate-by-crate description in
[architecture.md](docs/architecture.md).

## Project status

sphereQL is at **v0.1.0-alpha**. The core API is functional and covered
by 400+ tests, but may change before 1.0. Known limitations and roadmap
are in [project-status.md](docs/project-status.md).

## Contributing

1. Fork the repo and create a feature branch.
2. Run `cargo test --workspace --all-features` and
   `cargo clippy --workspace --all-features --all-targets`.
3. For Python changes, run `cd sphereql-python && maturin develop && pytest -v`.
4. Open a PR against `main`.

The codebase uses Rust 2024 edition. All CI checks must pass before
merge. See [testing.md](docs/testing.md) for the full pipeline.

## License

[MIT](LICENSE)
