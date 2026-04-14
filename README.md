# sphereQL

[![CI](https://github.com/bkahan/sphereQL/actions/workflows/ci.yml/badge.svg)](https://github.com/bkahan/sphereQL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/sphereql.svg)](https://crates.io/crates/sphereql)
[![PyPI](https://img.shields.io/pypi/v/sphereql.svg)](https://pypi.org/project/sphereql/)

**Project high-dimensional embeddings onto a 3D sphere for fast semantic search,
spatial queries, and interactive visualization.**

sphereQL maps vectors from any embedding model (OpenAI, Cohere, sentence-transformers,
etc.) onto spherical coordinates via PCA or Kernel PCA, then indexes them with
shell/sector spatial partitioning for fast nearest-neighbor lookups.
The result: you can search, cluster, trace concept paths, and visualize hundreds
of thousands of embeddings on a 3D sphere -- from Rust, Python, or the browser
via WASM.

## Use Cases

- **Semantic search** -- project embeddings to S^2 and query nearest neighbors
  in microseconds, with optional cosine-similarity re-ranking in the original
  space for precision
- **Knowledge visualization** -- render your entire embedding corpus as an
  interactive 3D sphere, colored by category, explorable in the browser
- **Concept path tracing** -- find the shortest semantic path between two
  concepts through projected space
- **Cluster detection** -- automatically discover concept "globs" (dense regions)
  on the sphere surface
- **Geospatial indexing** -- use the core library for pure spherical geometry:
  coordinate conversions, great-circle distances, region queries (cone, cap,
  shell, band, wedge)
- **Vector database bridge** -- connect Qdrant or Pinecone collections to
  sphereQL's pipeline for hybrid search and spherical coordinate enrichment

## Architecture

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
```

| Crate | Description |
|---|---|
| `sphereql-core` | Spherical math primitives: points (`SphericalPoint`, `CartesianPoint`, `GeoPoint`), coordinate conversions, distance metrics (angular, great-circle, chord, cosine), interpolation (slerp, nlerp), and region types (cone, cap, shell, band, wedge) |
| `sphereql-index` | Spatial indexing with composite shell + sector partitioning, k-NN search, cone/cap/shell/band/wedge/region queries, and cached Cartesian vectors for fast angular distance proxy |
| `sphereql-layout` | Layout engines for distributing items on S^2: Fibonacci spiral (uniform), k-means clustering, force-directed simulation, and incremental managed layouts with quality metrics |
| `sphereql-embed` | Embedding projection via PCA, Kernel PCA (Gaussian/RBF), or random projection. Query pipeline with k-NN, similarity threshold, concept paths, glob detection, and local manifold fitting |
| `sphereql-graphql` | async-graphql schema with cone/shell/band/wedge/region queries, k-NN search, distance calculations, and real-time subscriptions via a broadcast event bus |
| `sphereql-vectordb` | Vector store bridge for InMemory, Qdrant (gRPC), and Pinecone backends. Handles sync, PCA fitting, projection, and hybrid search with cosine re-ranking |
| `sphereql-python` | Python bindings via PyO3/maturin. Exposes Pipeline, projections, vector store bridges, and interactive 3D visualization |
| `sphereql-wasm` | WebAssembly bindings via wasm-bindgen for running the embedding pipeline in the browser |
| `sphereql` | Umbrella crate with feature flags for selective imports |

## Quick Start (Rust)

Add to your `Cargo.toml`:

```toml
[dependencies]
sphereql = { version = "0.1", features = ["full"] }
```

### Feature Flags

| Feature | Includes | Dependencies |
|---|---|---|
| `core` (default) | Math primitives, conversions, distances, regions | -- |
| `index` | Spatial indexing and queries | `core` |
| `layout` | Layout strategies and quality metrics | `core`, `index` |
| `embed` | Embedding projection (PCA, Kernel PCA, random) and pipeline | `core`, `index`, `layout` |
| `graphql` | GraphQL schema, subscriptions, event bus | `core`, `index` |
| `vectordb` | Vector store bridge and hybrid search | `embed` |
| `pinecone` | Pinecone backend for vectordb | `vectordb` |
| `full` | All of the above except `pinecone` | All non-backend features |

> **Note:** `full` does not activate the `pinecone` feature because it pulls in
> `reqwest`. Enable it explicitly if you need the Pinecone backend:
> `features = ["full", "pinecone"]`.
>
> The `qdrant` feature is available on `sphereql-vectordb` and `sphereql-python`
> directly but is not re-exported through the umbrella crate. Use
> `sphereql-vectordb` with `features = ["qdrant"]` for Rust, or
> `pip install sphereql[qdrant]` for Python.

### Spherical Math

```rust
use sphereql::core::*;

// Create spherical points (r, theta, phi)
let p1 = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let p2 = SphericalPoint::new(1.0, 1.2, 1.5).unwrap();

// Convert to Cartesian
let cart = spherical_to_cartesian(&p1);

// Convert to geographic (lat/lon/alt)
let geo = spherical_to_geo(&p1);

// Compute distances
let angle = angular_distance(&p1, &p2);
let arc = great_circle_distance(&p1, &p2, 6371.0); // Earth radius in km
let chord = chord_distance(&p1, &p2);

// Interpolate along a great circle
let midpoint = slerp(&p1, &p2, 0.5);
```

### Spatial Indexing

```rust
use sphereql::core::*;
use sphereql::index::*;

// Define your item type
#[derive(Debug, Clone)]
struct Star { id: u64, pos: SphericalPoint }

impl SpatialItem for Star {
    type Id = u64;
    fn id(&self) -> &u64 { &self.id }
    fn position(&self) -> &SphericalPoint { &self.pos }
}

// Build a spatial index
let mut index = SpatialIndex::<Star>::builder()
    .uniform_shells(5, 10.0)
    .theta_divisions(12)
    .phi_divisions(6)
    .build();

// Insert items
index.insert(Star {
    id: 1,
    pos: SphericalPoint::new_unchecked(1.0, 0.5, 0.8),
});

// Query: find items within a cone
let apex = SphericalPoint::origin();
let axis = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let cone = Cone::new(apex, axis, 0.3).unwrap();
let result = index.query_cone(&cone);

// Find k nearest neighbors
let target = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let neighbors = index.nearest(&target, 5);
```

### Layout Engine

```rust
use sphereql::core::*;
use sphereql::layout::*;

// Uniform distribution via Fibonacci spiral
let layout = UniformLayout::new();

// Clustered layout with k-means
let layout = ClusteredLayout::new()
    .with_clusters(4)
    .with_spread(0.3);

// Force-directed simulation
let layout = ForceDirectedLayout::new()
    .with_iterations(100)
    .with_repulsion(1.0)
    .with_cooling(0.95);
```

### Embedding Projection

```rust
use sphereql::embed::*;

// Prepare embeddings (e.g., 384-dimensional sentence-transformer output)
let corpus: Vec<Embedding> = vectors
    .into_iter()
    .map(Embedding::new)
    .collect();

// Fit PCA projection from a corpus
let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude);

// Or use Kernel PCA for non-linear manifold structure
let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Magnitude);

// Project a single embedding to the sphere
let point = pca.project(&corpus[0]);

// Use the full pipeline for search, concept paths, and more
let input = PipelineInput {
    categories: categories,     // Vec<String>, one per embedding
    embeddings: raw_vectors,    // Vec<Vec<f64>>
};
let pipeline = SphereQLPipeline::new(input).unwrap();

// k-NN search
let query = PipelineQuery { embedding: query_vec };
let results = pipeline.query(
    SphereQLQuery::Nearest { k: 5 },
    &query,
);

// Concept path between two items
let path = pipeline.query(
    SphereQLQuery::ConceptPath {
        source_id: "s-0001",
        target_id: "s-0042",
        graph_k: 10,
    },
    &query,
);

// Detect clusters on the sphere
let globs = pipeline.query(
    SphereQLQuery::DetectGlobs { k: None, max_k: 10 },
    &query,
);

// Export for visualization
let points = pipeline.exported_points();
let evr = pipeline.explained_variance_ratio();
```

### GraphQL Integration

```rust
use sphereql::graphql::*;
use std::sync::Arc;
use tokio::sync::RwLock;

// Build schema with sensible defaults
let schema = create_schema_with_defaults();

// Or configure manually
let index = create_default_index();
let event_bus = SpatialEventBus::new(256);
let schema = build_schema(index, event_bus);

// Execute queries
let result = schema.execute(r#"{
    withinCone(cone: {
        apex: { r: 0, theta: 0, phi: 0 },
        axis: { r: 1, theta: 0.5, phi: 0.8 },
        halfAngle: 0.3
    }) { items { r theta phi } totalScanned }
}"#).await;
```

## Quick Start (Python)

### Install

```bash
pip install sphereql
```

For Qdrant vector database support:

```bash
pip install sphereql[qdrant]
```

### Semantic Search

```python
import sphereql

categories = ["science", "science", "cooking", "cooking", "sports"]
embeddings = [
    [0.1, 0.9, 0.3, 0.0],
    [0.2, 0.8, 0.4, 0.1],
    [0.9, 0.1, 0.0, 0.5],
    [0.8, 0.2, 0.1, 0.4],
    [0.4, 0.4, 0.8, 0.2],
]

pipeline = sphereql.Pipeline(categories, embeddings)

# k-nearest neighbors
query = [0.15, 0.85, 0.35, 0.05]
results = pipeline.nearest(query, k=3)
for r in results:
    print(f"{r.id}  {r.category}  distance={r.distance:.4f}")

# Similarity threshold search
similar = pipeline.similar_above(query, min_cosine=0.8)

# Concept path between items
path = pipeline.concept_path("s-0000", "s-0003", graph_k=10)

# Cluster detection
globs = pipeline.detect_globs(max_k=10)

# Local manifold fitting
manifold = pipeline.local_manifold(query, neighborhood_k=10)

# Export projected coordinates
points = pipeline.exported_points()
print(f"Explained variance ratio: {pipeline.explained_variance_ratio:.4f}")
```

### Interactive 3D Visualization

```python
import sphereql

# Opens an interactive WebGL sphere in your browser
sphereql.visualize(categories, embeddings, title="My Embeddings")

# Or visualize from an existing pipeline
sphereql.visualize_pipeline(pipeline, title="Pipeline View")
```

### Vector Database Bridge

```python
import sphereql

# In-memory store (for testing and small datasets)
store = sphereql.InMemoryStore("my-collection", dimension=384)
store.upsert([
    {"id": "doc-1", "vector": embedding_1, "metadata": {"category": "science"}},
    {"id": "doc-2", "vector": embedding_2, "metadata": {"category": "cooking"}},
])

bridge = sphereql.VectorStoreBridge(store)
bridge.build_pipeline(category_key="category")

# Hybrid search: angular candidates + cosine re-ranking
results = bridge.hybrid_search(query_vec, final_k=5, recall_k=20)
```

### Core Types in Python

```python
import sphereql

# Spherical/Cartesian/Geo point types
p = sphereql.SphericalPoint(1.0, 0.5, 0.8)
c = sphereql.spherical_to_cartesian(p)
g = sphereql.spherical_to_geo(p)

# Distance functions
d = sphereql.angular_distance(p1, p2)
gc = sphereql.great_circle_distance(p1, p2, radius=6371.0)

# Projection classes
pca = sphereql.PcaProjection.fit(embeddings, radial="magnitude")
kpca = sphereql.KernelPcaProjection.fit(embeddings, radial="magnitude")
rp = sphereql.RandomProjection(dimension=384, radial=1.0, seed=42)
```

## Quick Start (WASM)

```bash
# Build the WASM package
cd sphereql-wasm
wasm-pack build --target web
```

```javascript
import init, { Pipeline } from './pkg/sphereql_wasm.js';

await init();

const pipeline = new Pipeline(JSON.stringify({
  categories: ["science", "cooking", "sports"],
  embeddings: [[0.1, 0.9, 0.3], [0.9, 0.1, 0.0], [0.4, 0.4, 0.8]]
}));

// Returns JSON string
const results = pipeline.nearest(
  JSON.stringify({ embedding: [0.15, 0.85, 0.35] }),
  3
);
console.log(JSON.parse(results));
```

## Coordinate System

sphereQL uses the **physics convention** for spherical coordinates:

- **r** -- radial distance from origin (r >= 0)
- **theta** -- azimuthal angle in the xy-plane from the x-axis, range [0, 2*pi)
- **phi** -- polar angle from the z-axis, range [0, pi]

Geographic coordinates use standard (lat, lon, alt) with latitude in [-90, 90]
and longitude in [-180, 180].

## How Embedding Projection Works

sphereQL projects high-dimensional vectors (e.g., 384-d sentence-transformer
output) down to 3D spherical coordinates:

1. **Normalize** -- all embeddings are L2-normalized to the unit hypersphere
2. **Center** -- subtract the corpus mean
3. **Reduce** -- find the top 3 principal components via PCA (linear) or
   Kernel PCA with Gaussian/RBF kernel (non-linear manifold preservation)
4. **Map** -- the 3 components become Cartesian (x, y, z), which convert to
   spherical (r, theta, phi)

The **radial coordinate** is configurable via `RadialStrategy`:
- `Magnitude` (default): r = pre-normalization L2 magnitude of the embedding,
  encoding "confidence" or specificity
- `Fixed(value)`: constant radius for all points (pure angular projection)
- `MagnitudeTransform(fn)`: custom transform (e.g., log-scaling)

**Important:** This projection is inherently lossy. The explained variance ratio
(EVR) indicates how much angular structure is preserved. For typical transformer
embeddings, expect 2-5% EVR at 3 dimensions. sphereQL compensates with **hybrid
search**: fast angular-distance candidate retrieval in projected space, followed
by cosine similarity re-ranking in the original embedding space.

**Kernel PCA** (new) captures non-linear manifold structure (curved clusters,
rings, spirals) that linear PCA crushes flat. It uses the Gaussian kernel
k(x, y) = exp(-||x-y||^2 / 2*sigma^2) with automatic sigma selection via the
median heuristic. See the [kernel PCA source](sphereql-embed/src/kernel_pca.rs)
for mathematical details and references.

## Performance

The spatial index uses a two-tier partitioning scheme:

- **Shell partitioning** -- radial shells for fast r-range filtering
- **Sector partitioning** -- angular sectors (theta x phi grid) for spatial
  locality
- **Cosine proxy** -- k-NN uses precomputed unit Cartesian vectors and
  `1 - dot(a, b)` instead of the full Vincenty formula, reducing per-item
  cost to 3 multiplications + 2 additions

Benchmark results (10,000 points, 384 dimensions, 20 clusters, 200 queries):

| Method | k | Precision@k | nDCG@k | Mean latency |
|---|---|---|---|---|
| Brute-force ANN | 5 | 1.000 | 1.000 | 154 ms |
| SphereQL PCA | 1 | 1.000 | 1.000 | 1.7 ms |
| SphereQL PCA | 5 | 0.205 | 0.745 | 1.6 ms |
| SphereQL KPCA | 5 | 0.204 | 0.746 | 84 ms |
| Hybrid (r=k*2) | 5 | 1.000 | 1.000 | 155 ms |

SphereQL PCA queries run **~90x faster** than brute-force with perfect precision
at k=1. Precision degrades at higher k due to the lossy 384-d to 3-d projection
(~2.8% explained variance). The hybrid approach recovers full precision via
cosine re-ranking in the original space, at near brute-force latency. Improving
the speed/precision tradeoff at higher k is an active development priority.

For full results see [`docs/benchmark-analysis.md`](docs/benchmark-analysis.md) and
[`docs/search-precision-roadmap.md`](docs/search-precision-roadmap.md).

## Examples

```bash
# Basic spherical math
cargo run --example basic_positioning -p sphereql --features core

# Spatial indexing and geospatial queries
cargo run --example geospatial -p sphereql --features index

# GraphQL server
cargo run --example graphql_server -p sphereql --features full

# Embedding projection
cargo run --example word_embeddings -p sphereql --features embed
cargo run --example semantic_search -p sphereql --features embed
cargo run --example auto_categorize -p sphereql --features embed

# End-to-end transformer embedding pipeline
cargo run --example e2e_transformer -p sphereql --features embed

# Benchmarks
cargo run --example benchmark -p sphereql --features full
```

Python examples are in [`sphereql-python/examples/`](sphereql-python/examples/):

```bash
cd sphereql-python
pip install maturin numpy
maturin develop

python examples/quickstart.py
python examples/kernel_pca.py
python examples/dataset.py
```

## Running Tests

```bash
# All workspace tests
cargo test --workspace

# All features (including qdrant/pinecone compile checks)
cargo test --workspace --all-features

# Clippy lint pass
cargo clippy --workspace --all-features --all-targets

# Python tests
cd sphereql-python
maturin develop
pytest -v

# Benchmarks
cargo bench -p sphereql-core
cargo bench -p sphereql-index
```

## CI

The [CI pipeline](.github/workflows/ci.yml) runs on every push and PR to `main`:

- `cargo test --workspace --all-features` + doc-tests
- `cargo clippy` with `-Dwarnings`
- `cargo fmt --check`
- Per-feature compilation matrix (core, index, layout, embed, graphql, vectordb,
  full, no-default-features)
- Python build + `pytest` on Python 3.12

A separate [release workflow](.github/workflows/python-publish.yml) builds
cross-platform wheels (Linux x86_64/aarch64, macOS x86_64/aarch64, Windows
x86_64) and publishes to PyPI on GitHub release.

## Project Status

sphereQL is at **v0.1.0** (alpha). The API is functional and tested but may
change before 1.0. Current priorities:

- Improving search precision at higher k values
- HNSW or VP-tree indexing for better recall without brute-force fallback
- Streaming/incremental PCA for large-scale datasets
- Published crate on crates.io

## Contributing

Contributions are welcome. To get started:

1. Fork the repo and create a feature branch
2. Run `cargo test --workspace --all-features` and `cargo clippy --workspace --all-features --all-targets`
3. For Python changes, run `cd sphereql-python && maturin develop && pytest -v`
4. Open a PR against `main`

The codebase uses Rust 2024 edition. All CI checks must pass before merge.

## License

[MIT](LICENSE)
