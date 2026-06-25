# Architecture

sphereQL is a Rust workspace of focused crates that compose via feature
flags. The umbrella `sphereql` crate re-exports subsets behind named
features; you depend on it with the flags you need.

## Dependency graph

Intra-workspace dependency edges (normal `[dependencies]`, derived
directly from each `Cargo.toml` — kept as an edge list rather than ASCII
art so it can't drift out of sync):

```text
  sphereql-core      →  (no workspace deps)
  sphereql-index     →  core
  sphereql-layout    →  core, index
  sphereql-embed     →  core, index, layout
  sphereql-graphql   →  core, index, embed
  sphereql-vectordb  →  core, embed
  sphereql-lingua    →  core
  sphereql-corpus    →  core  (re-exports the synthetic embedder)
  sphereql-vis       →  core
  sphereql (umbrella, feature-gated)
                     →  core, index, layout, embed, graphql, vectordb, vis

  Bindings / tooling (consume the above; not part of the core graph):
  sphereql-python      →  core, index, layout, embed, graphql, vectordb, lingua, vis  (all optional)
  sphereql-wasm        →  core, index, embed
  sphereql-vis-server  →  core, index, embed, vis, corpus
  sphereql-examples    →  sphereql, core, corpus, embed, lingua
  scripts/check-drift, scripts/check-versions  →  CI tooling (no library deps)
```

Both `sphereql-graphql` and `sphereql-vectordb` depend on
`sphereql-embed` — the GraphQL layer serves the category-enrichment
surface and embeds query text through the `TextEmbedder` trait — and
both also depend directly on `sphereql-core`. The Python skeleton
`lingua-spherica` is a separate package, not a workspace crate.

## Crates

| Crate | Description |
|---|---|
| `sphereql-core` | Spherical math primitives: points (`SphericalPoint`, `CartesianPoint`, `GeoPoint`), coordinate conversions, distance metrics (angular, great-circle, chord, cosine), interpolation (slerp, nlerp), and region types (cone, cap, shell, band, wedge). |
| `sphereql-index` | Spatial indexing with composite shell + sector partitioning, k-NN search, cone/cap/shell/band/wedge/region queries, and cached Cartesian vectors for fast angular-distance proxy. |
| `sphereql-layout` | Layout engines for distributing items on S²: Fibonacci spiral (uniform), k-means clustering, force-directed simulation, and incremental managed layouts with quality metrics. |
| `sphereql-embed` | Embedding projection (PCA / Kernel PCA / Laplacian eigenmap / UMAP-on-sphere, plus a low-level random-projection baseline), query pipeline (k-NN, similarity threshold, concept paths, glob detection, local manifold fitting), Category Enrichment Layer (inter-category graph, bridge classification, inner spheres, drill-down, hierarchical domain-group routing), and a metalearning framework (`PipelineConfig`, `QualityMetric`, `auto_tune`, `MetaModel`, `FeedbackAggregator`). |
| `sphereql-graphql` | `async-graphql` schema with spatial queries (cone/shell/band/wedge/region, k-NN, distances), the full category enrichment surface (concept paths, drill-down, domain groups, stats), real-time subscriptions, and a pluggable `TextEmbedder` trait for natural-language query inputs. |
| `sphereql-vectordb` | Vector store bridge for InMemory, Qdrant (gRPC), and Pinecone backends. Handles sync, PCA fitting, projection, and hybrid search with cosine re-ranking. |
| `sphereql-vis` | Self-contained 3D visualization. A serializable `Scene` (point cloud + `SceneStats` + an `#[non_exhaustive]` `Overlay` set: centroids, classified bridges, geodesic concept paths, Voronoi caps, antipodes, coverage maps, domain-group spokes, globs, manifold slices) and a hardened emitter that inlines the Three.js + OrbitControls runtime so every HTML file is offline-portable. Also ships the shared `viewer.js` runtime (one `createViewer` implementation behind the offline file, the WASM studio, and the server front-end) and the out-of-core streaming contract — a bounded `Manifest` + binary `SQT1` tile format (`manifest` / `tile` modules) consumed by `sphereql-vis-server`. Pure: depends only on `sphereql-core`. Consumed by the Python bindings, the examples, the umbrella `vis` feature, and `sphereql-vis-server`. See [visualization.md](visualization.md). |
| `sphereql-lingua` | Six-stage text → `ConceptGraph` pipeline that maps natural language onto SphereQL `(r, θ, φ)` coordinates: concept extraction (pluggable `ConceptExtractor`, regex default), domain taxonomy θ assignment, abstraction φ resolution, salience-driven r weighting, relation encoding (typed geodesic arcs), graph assembly. Built on `sphereql-core` so coordinate convention and distance math match the rest of the workspace. **Rust is the source of truth** — the Python `lingua-spherica` package is a types/math skeleton only. |
| `sphereql-vis-server` | Out-of-core query server (axum + tokio) for the streaming viewer. Loads a corpus, embeds + projects it (gating O(n²) families by size), and holds the projected positions (spatial-indexed for cone/viewport tiling), the raw embeddings (ANN-indexed for semantic neighbors), and per-point metadata in memory. Serves a bounded `Manifest` up front, then streams binary `SQT1` point tiles by viewport + LOD, with lazy `POST /points` metadata and trace queries on top (`/nearest` ANN neighbors, `/path` category-graph routes, `/globs` cluster detection, `/drill_down` within-category k-NN). Reuses the pure `sphereql-vis` manifest/tile contract. |
| `sphereql-python` | Python bindings via PyO3/maturin. Exposes Pipeline (with category enrichment + Laplacian), every projection family, vector store bridges, `auto_tune`, the `MetaModel` layer, `FeedbackAggregator`, and interactive 3D visualization. Type stubs (`.pyi`) auto-generated via `pyo3-stub-gen`. |
| `sphereql-wasm` | WebAssembly bindings via `wasm-bindgen`. Typed return values via `tsify` — every pipeline / category / metalearning method returns a TypeScript-typed value, no `JSON.parse` required on the JS side. |
| `scripts/check-drift` | CI tool that `syn`-parses `sphereql-embed` + `sphereql-layout` public APIs and fails when a new public item isn't bound in Python/WASM and isn't in `.bindings-ignore.toml`. |
| `scripts/check-versions` | CI tool that checks every release-version string across manifests, READMEs, and docs against the canonical workspace + pyproject versions. |
| `scripts/check-docs` | CI tool that checks doc/code consistency: every workspace crate appears in the README + architecture crate tables, and the test-count floors stated in the docs still hold. |
| `scripts/check-doc-snippets` | CI tool that compile-checks the `rust` fenced blocks in the prose docs against the `sphereql` umbrella crate, so API drift in a quickstart snippet fails CI. |
| `sphereql` | Umbrella crate with feature flags for selective imports. |
| `sphereql-corpus` | Shared test corpora for examples: 775-concept built-in across 31 academic domains, plus a 300-concept low-SNR stress corpus (via `build_stress_corpus` / `embed_with_noise`). Bulk-ingested parquet corpora (DBpedia 500K, Wikidata 50K) live in `sphereql-corpus/data/` and are produced by the `bulk_ingest` example. Used by `ai_knowledge_navigator`, `spatial_analysis`, `auto_tune`, `meta_learn`, and the full_e2e example. |
| `sphereql-examples` | Runnable examples for the whole workspace (`auto_tune`, `meta_learn`, `bulk_ingest`, `corpus_self_tune`, `full_e2e`, `graphql_server`, …). Workspace member; not published. |

## Feature flags

| Feature | Includes | Dependencies |
|---|---|---|
| `core` (default) | Math primitives, conversions, distances, regions | — |
| `index` | Spatial indexing and queries | `core` |
| `layout` | Layout strategies and quality metrics | `core`, `index` |
| `embed` | Embedding projection, pipeline, auto-tuner, meta-model | `core`, `index`, `layout` |
| `graphql` | GraphQL schema, subscriptions, event bus | `core`, `index` (+ `embed` via the crate) |
| `vectordb` | Vector store bridge and hybrid search | `embed` |
| `pinecone` | Pinecone backend for vectordb | `vectordb` |
| `retain-embeddings` | Keep the original high-dimensional embeddings on the pipeline (`raw_embeddings()`, `pairwise_similarities()`, `nearest_by_embedding()`) | `sphereql-embed/retain-embeddings` |
| `full` | All of the above except `pinecone` (includes `retain-embeddings`) | All non-backend features |

`full` does not activate `pinecone` because it pulls in `reqwest`. Enable it
explicitly if you need the Pinecone backend:

```toml
sphereql = { version = "0.3.0", features = ["full", "pinecone"] }
```

The `qdrant` feature is available on `sphereql-vectordb` and `sphereql-python`
directly but is not re-exported through the umbrella crate. Use
`sphereql-vectordb` with `features = ["qdrant"]` for Rust, or build the
Python wheel from source with `maturin develop --features qdrant` for
Python (it is not a PyPI extra).

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

```rust,ignore
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
