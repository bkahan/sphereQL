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
| `sphereql-graphql` | `async-graphql` schema with spatial queries (cone/shell/band/wedge/region, k-NN, distances), the full category enrichment surface (concept paths, drill-down, domain groups, stats), real-time subscriptions, and a pluggable `TextEmbedder` trait for natural-language query inputs. |
| `sphereql-vectordb` | Vector store bridge for InMemory, Qdrant (gRPC), and Pinecone backends. Handles sync, PCA fitting, projection, and hybrid search with cosine re-ranking. |
| `sphereql-python` | Python bindings via PyO3/maturin. Exposes Pipeline (with category enrichment + Laplacian), every projection family, vector store bridges, `auto_tune`, the `MetaModel` layer, `FeedbackAggregator`, and interactive 3D visualization. Type stubs (`.pyi`) auto-generated via `pyo3-stub-gen`. |
| `sphereql-wasm` | WebAssembly bindings via `wasm-bindgen`. Typed return values via `tsify` — every pipeline / category / metalearning method returns a TypeScript-typed value, no `JSON.parse` required on the JS side. |
| `scripts/check-drift` | CI tool that `syn`-parses `sphereql-embed` + `sphereql-layout` public APIs and fails when a new public item isn't bound in Python/WASM and isn't in `.bindings-ignore.toml`. |
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

## GraphQL category schema

`sphereql-graphql` exposes the full category-enrichment surface through
a `MergedQueryRoot` that combines the spatial-only resolvers
(`withinCone`, `nearestTo`, …) with seven category resolvers:
`conceptPath`, `categoryConceptPath`, `categoryNeighbors`, `drillDown`,
`hierarchicalNearest`, `categoryStats`, `domainGroups`.

Resolvers that take a `queryText: String` argument embed the text
server-side through a `TextEmbedder` trait — pluggable at schema
construction time, with `NoEmbedder` (error-descriptively) as the
default:

```rust
use std::sync::Arc;
use sphereql_embed::{Embedding, EmbedderError, FnEmbedder};
use sphereql_graphql::{
    build_unified_schema, build_pipeline_handle_from_items,
    create_default_index, SpatialEventBus, CategorizedItemInput,
};

// 1. Your embedder — here a deterministic closure for tests; in
//    production, wrap your sentence-transformers / OpenAI client.
let embedder = Arc::new(FnEmbedder::new(|text: &str| {
    Ok::<_, EmbedderError>(Embedding::new(embed_text(text)))
}));

// 2. A fitted pipeline wrapped for concurrent reads.
let items: Vec<CategorizedItemInput> = load_your_corpus();
let pipeline = build_pipeline_handle_from_items(&items)?;

// 3. Unified schema — spatial + category + subscriptions.
let schema = build_unified_schema(
    create_default_index(),
    SpatialEventBus::new(256),
    pipeline,
    embedder,
);
```

Users who only want the spatial surface keep the existing
`build_schema(index, event_bus)` entry point — the pipeline and
embedder context entries are optional.
