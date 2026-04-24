# Rust quickstart

A tour of each workspace capability. The top-level [README](../README.md)
has a minimal version of this; the sections below fill it in.

```toml
[dependencies]
sphereql = { version = "0.1", features = ["full"] }
```

See [architecture.md](architecture.md) for feature-flag details.

## Spherical math

```rust
use sphereql::core::*;

// Create spherical points (r, θ, φ)
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

## Spatial indexing

```rust
use sphereql::core::*;
use sphereql::index::*;

#[derive(Debug, Clone)]
struct Star { id: u64, pos: SphericalPoint }

impl SpatialItem for Star {
    type Id = u64;
    fn id(&self) -> &u64 { &self.id }
    fn position(&self) -> &SphericalPoint { &self.pos }
}

let mut index = SpatialIndex::<Star>::builder()
    .uniform_shells(5, 10.0)
    .theta_divisions(12)
    .phi_divisions(6)
    .build();

index.insert(Star {
    id: 1,
    pos: SphericalPoint::new_unchecked(1.0, 0.5, 0.8),
});

// Cone query
let apex = SphericalPoint::origin();
let axis = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let cone = Cone::new(apex, axis, 0.3).unwrap();
let result = index.query_cone(&cone);

// k nearest neighbors
let target = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let neighbors = index.nearest(&target, 5);
```

## Layout engine

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

## Embedding projection

```rust
use sphereql::embed::*;

// Prepare embeddings (e.g., 384-dimensional sentence-transformer output)
let corpus: Vec<Embedding> = vectors.into_iter().map(Embedding::new).collect();

// Fit a projection from a corpus — PCA, Kernel PCA, or Laplacian eigenmap.
let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude);
let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Magnitude);

// Project a single embedding to the sphere
let point = pca.project(&corpus[0]);

// Full pipeline for search, concept paths, category enrichment, etc.
let input = PipelineInput {
    categories: categories,    // Vec<String>, one per embedding
    embeddings: raw_vectors,   // Vec<Vec<f64>>
};
let pipeline = SphereQLPipeline::new(input).unwrap();

// k-NN search
let query = PipelineQuery { embedding: query_vec };
let results = pipeline.query(SphereQLQuery::Nearest { k: 5 }, &query);

// Concept path between two items
let path = pipeline.query(
    SphereQLQuery::ConceptPath {
        source_id: "s-0001",
        target_id: "s-0042",
        graph_k: 10,
    },
    &query,
);

// Cluster detection
let globs = pipeline.query(
    SphereQLQuery::DetectGlobs { k: None, max_k: 10 },
    &query,
);

// --- Category Enrichment Layer ---

// Category-level concept path
let cat_path = pipeline.query(
    SphereQLQuery::CategoryConceptPath {
        source_category: "science",
        target_category: "cooking",
    },
    &query,
);

// Nearest neighbor categories
let neighbors = pipeline.query(
    SphereQLQuery::CategoryNeighbors { category: "science", k: 3 },
    &query,
);

// Drill down into a category (uses inner sphere if available)
let drill = pipeline.query(
    SphereQLQuery::DrillDown { category: "science", k: 5 },
    &query,
);

// Category stats
let stats = pipeline.query(SphereQLQuery::CategoryStats, &query);

// Export for visualization
let points = pipeline.exported_points();
let evr = pipeline.explained_variance_ratio();
```

See [projections.md](projections.md) for a tour of the four projection
families and [auto-tuning.md](auto-tuning.md) for the `PipelineConfig` +
`auto_tune` workflow.

## GraphQL

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
let result = schema
    .execute(r#"{
        withinCone(cone: {
            apex: { r: 0, theta: 0, phi: 0 },
            axis: { r: 1, theta: 0.5, phi: 0.8 },
            halfAngle: 0.3
        }) { items { r theta phi } totalScanned }
    }"#)
    .await;
```
