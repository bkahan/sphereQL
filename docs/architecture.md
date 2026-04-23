# Architecture

sphereQL is a Rust workspace of focused crates that compose via feature
flags. The umbrella `sphereql` crate re-exports subsets behind named
features; you depend on it with the flags you need.

## Dependency graph

```text
                    sphereql (umbrella crate, feature-gated)
                    |
    +---------------+------+------------------+
    |               |      |                  |
sphereql-graphql  sphereql-vectordb           |
    |               |                         |
    |           sphereql-embed                |
    |               |                         |
    |           sphereql-layout               |
    |               |                         |
    +----------sphereql-index                 |
                    |                         |
                sphereql-core ----------------+

    sphereql-python  (PyO3 bindings via maturin)
    sphereql-wasm    (wasm-bindgen bindings)

    sphereql-corpus  (shared example data, no runtime deps)
```

## Crates

| Crate | Description |
|---|---|
| `sphereql-core` | Spherical math primitives: points (`SphericalPoint`, `CartesianPoint`, `GeoPoint`), coordinate conversions, distance metrics (angular, great-circle, chord, cosine), interpolation (slerp, nlerp), and region types (cone, cap, shell, band, wedge). |
| `sphereql-index` | Spatial indexing with composite shell + sector partitioning, k-NN search, cone/cap/shell/band/wedge/region queries, and cached Cartesian vectors for fast angular-distance proxy. |
| `sphereql-layout` | Layout engines for distributing items on S²: Fibonacci spiral (uniform), k-means clustering, force-directed simulation, and incremental managed layouts with quality metrics. |
| `sphereql-embed` | Embedding projection (PCA / Kernel PCA / Laplacian eigenmap / random), query pipeline (k-NN, similarity threshold, concept paths, glob detection, local manifold fitting), Category Enrichment Layer (inter-category graph, bridge classification, inner spheres, drill-down, hierarchical domain-group routing), and a metalearning framework (`PipelineConfig`, `QualityMetric`, `auto_tune`, `MetaModel`, `FeedbackAggregator`). |
| `sphereql-graphql` | `async-graphql` schema with cone/shell/band/wedge/region queries, k-NN search, distance calculations, and real-time subscriptions via a broadcast event bus. |
| `sphereql-vectordb` | Vector store bridge for InMemory, Qdrant (gRPC), and Pinecone backends. Handles sync, PCA fitting, projection, and hybrid search with cosine re-ranking. |
| `sphereql-python` | Python bindings via PyO3/maturin. Exposes Pipeline (including category enrichment), projections, vector store bridges, and interactive 3D visualization. |
| `sphereql-wasm` | WebAssembly bindings via `wasm-bindgen` for running the embedding pipeline (including category enrichment) in the browser. |
| `sphereql` | Umbrella crate with feature flags for selective imports. |
| `sphereql-corpus` | Shared test corpora for examples: 775-concept built-in across 31 academic domains, plus a 300-concept low-SNR stress corpus (via `build_stress_corpus` / `embed_with_noise`). Used by `ai_knowledge_navigator`, `spatial_analysis`, `auto_tune`, and the meta-learning examples. |

## Feature flags

| Feature | Includes | Dependencies |
|---|---|---|
| `core` (default) | Math primitives, conversions, distances, regions | — |
| `index` | Spatial indexing and queries | `core` |
| `layout` | Layout strategies and quality metrics | `core`, `index` |
| `embed` | Embedding projection, pipeline, auto-tuner, meta-model | `core`, `index`, `layout` |
| `graphql` | GraphQL schema, subscriptions, event bus | `core`, `index` |
| `vectordb` | Vector store bridge and hybrid search | `embed` |
| `pinecone` | Pinecone backend for vectordb | `vectordb` |
| `full` | All of the above except `pinecone` | All non-backend features |

`full` does not activate `pinecone` because it pulls in `reqwest`. Enable it
explicitly if you need the Pinecone backend:

```toml
sphereql = { version = "0.1", features = ["full", "pinecone"] }
```

The `qdrant` feature is available on `sphereql-vectordb` and `sphereql-python`
directly but is not re-exported through the umbrella crate. Use
`sphereql-vectordb` with `features = ["qdrant"]` for Rust, or
`pip install sphereql[qdrant]` for Python.
