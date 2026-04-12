# sphereQL

Production-grade spherical coordinate operations library for Rust with GraphQL integration.

sphereQL provides a layered architecture for working with spherical coordinates — from core math primitives through spatial indexing, layout engines, embedding projection, and a ready-to-use GraphQL API.

## Crates

| Crate              | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `sphereql-core`    | Spherical math primitives: points, conversions, distances, interpolation, regions |
| `sphereql-index`   | Spatial indexing with shell/sector partitioning for fast queries                  |
| `sphereql-layout`  | Layout engine: uniform (Fibonacci), clustered (k-means), force-directed           |
| `sphereql-embed`   | Vector embedding projection: PCA and random projection from high-D to S²          |
| `sphereql-graphql` | async-graphql integration with queries, subscriptions, and event bus              |
| `sphereql-vectordb`| Vector store integration (in-memory, Pinecone, Qdrant) with hybrid search        |
| `sphereql-wasm`    | WebAssembly bindings for the embedding pipeline                                   |
| `sphereql-python`  | Python bindings via PyO3/maturin                                                  |
| `sphereql`         | Umbrella crate with feature flags for selective imports                           |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
sphereql = { version = "0.1", features = ["full"] }
```

### Feature Flags

| Feature          | Includes                                         | Dependencies              |
| ---------------- | ------------------------------------------------ | ------------------------- |
| `core` (default) | Math primitives, conversions, distances, regions | —                         |
| `index`          | Spatial indexing and queries                     | `core`                    |
| `layout`         | Layout strategies and quality metrics            | `core`, `index`           |
| `embed`          | Embedding projection (PCA, random) and pipeline  | `core`, `index`, `layout` |
| `graphql`        | GraphQL schema, subscriptions, event bus         | `core`, `index`           |
| `vectordb`       | Vector store bridge and hybrid search            | `embed`                   |
| `pinecone`       | Pinecone backend for vectordb                    | `vectordb`                |
| `qdrant`         | Qdrant gRPC backend for vectordb                 | `vectordb`                |
| `full`           | All of the above except `pinecone` and `qdrant`  | All non-backend features  |

> **Note:** `full` does not activate the `pinecone` or `qdrant` features because
> they pull in heavy external dependencies (`reqwest` and `qdrant-client`
> respectively). Enable them explicitly if you need a specific backend:
> `features = ["full", "qdrant"]`.

### Basic Usage

```rust
use sphereql::core::*;

// Create spherical points (r, theta, phi)
let p1 = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let p2 = SphericalPoint::new(1.0, 1.2, 1.5).unwrap();

// Convert to Cartesian
let cart = spherical_to_cartesian(&p1);

// Compute distances
let angle = angular_distance(&p1, &p2);
let arc = great_circle_distance(&p1, &p2, 6371.0); // Earth radius in km

// Interpolate along great circle
let midpoint = slerp(&p1, &p2, 0.5);
```

### Spatial Indexing

```rust
use sphereql::core::*;
use sphereql::index::*;

// Build a spatial index
let mut index = SpatialIndexBuilder::new()
    .uniform_shells(5, 10.0)
    .theta_divisions(12)
    .phi_divisions(6)
    .build();

// Insert items (must implement SpatialItem trait)
index.insert(my_item);

// Query: find items within a cone
let cone = Cone::new(apex, axis, half_angle).unwrap();
let result = index.query_cone(&cone);

// Find k nearest neighbors
let neighbors = index.nearest(&target_point, 5);
```

### Layout Engine

```rust
use sphereql::core::*;
use sphereql::layout::*;

// Uniform distribution via Fibonacci spiral
let layout = UniformLayout::new();
let result = layout.layout(&items, &mapper);

// Clustered layout with k-means
let layout = ClusteredLayout::new().with_clusters(4).with_spread(0.3);

// Force-directed simulation
let layout = ForceDirectedLayout::default();
```

### Embedding Projection

```rust
use sphereql::embed::*;

// Fit PCA projection from a corpus of embeddings
let corpus: Vec<Embedding> = load_embeddings();
let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude);

// Project high-dimensional vectors to the sphere
let point = pca.project(&query_embedding);

// Build a spatial index over embeddings
let mut index = EmbeddingIndex::builder(pca)
    .uniform_shells(10, 1.0)
    .theta_divisions(12)
    .phi_divisions(6)
    .build();

// Search: k nearest neighbors in projected space
let results = index.search_nearest(&query, 5);
```

### GraphQL Integration

```rust
use sphereql::graphql::*;
use std::sync::Arc;
use tokio::sync::RwLock;

// Build schema with defaults
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

### Python

```bash
cd sphereql-python
pip install maturin
maturin develop
```

```python
import sphereql

# Build a pipeline from embeddings
pipeline = sphereql.Pipeline(
    categories=["science", "cooking", "sports"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
)

# Query nearest neighbors
results = pipeline.nearest([0.15, 0.25, ...], k=3)
for r in results:
    print(f"{r.category}: {r.distance:.4f}")

# Export projected points
points = pipeline.exported_points()
print(pipeline.explained_variance_ratio)
```

## Coordinate System

sphereQL uses the physics convention for spherical coordinates:

- **r** — radial distance from origin (r >= 0)
- **theta** — azimuthal angle in the xy-plane from the x-axis, range [0, 2*pi*)
- **phi** — polar angle from the z-axis, range [0, pi]

## Architecture

```text
sphereql (umbrella, feature-gated)
  |
  +-- sphereql-graphql (async-graphql schema, subscriptions)
  |     |
  +-- sphereql-vectordb (vector store bridge, hybrid search)
  |     |
  +-- sphereql-embed (PCA/random projection, semantic pipeline)
  |     |
  +-- sphereql-layout (layout strategies, quality metrics)
  |     |
  +-----+-- sphereql-index (spatial partitioning, queries)
  |           |
  +-----------+-- sphereql-core (math primitives, zero dependencies*)

  sphereql-wasm (WASM bindings for sphereql-embed pipeline)
  sphereql-python (PyO3 bindings via maturin)
```

\* Core depends only on `serde`, `thiserror`, and `approx`.

## Examples

```bash
# Basic spherical math
cargo run --example basic_positioning -p sphereql --features core

# Spatial indexing demo
cargo run --example geospatial -p sphereql --features index

# GraphQL schema demo
cargo run --example graphql_server -p sphereql --features full

# Embedding projection demos
cargo run --example word_embeddings -p sphereql --features embed
cargo run --example semantic_search -p sphereql --features embed
cargo run --example auto_categorize -p sphereql --features embed
```

## Running Tests

```bash
# All tests
cargo test --workspace

# With clippy
cargo clippy --workspace --all-features --all-targets

# Benchmarks
cargo bench -p sphereql-core
cargo bench -p sphereql-index
```

## Performance Notes

The embedding pipeline projects high-dimensional vectors (e.g., 384-d) down to 3D
spherical coordinates via PCA. This projection is inherently lossy — the explained
variance ratio (EVR) indicates how much information is preserved. For typical
transformer embeddings, expect 2-5% EVR at 3D.

sphereQL compensates with hybrid search: fast angular-distance candidate retrieval
in projected space, followed by cosine similarity re-ranking in the original
embedding space. See [`docs/benchmark-analysis.md`](docs/benchmark-analysis.md) and
[`docs/search-precision-roadmap.md`](docs/search-precision-roadmap.md) for details.

## License

MIT
