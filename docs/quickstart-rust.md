# Rust quickstart

A tour of each workspace capability. The top-level [README](../README.md)
has a minimal version of this; the sections below fill it in.

```toml
[dependencies]
sphereql = { version = "0.3.0", features = ["full"] }
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

```rust,ignore
use sphereql::embed::*;

// Prepare embeddings (e.g., 384-dimensional sentence-transformer output)
let corpus: Vec<Embedding> = vectors.into_iter().map(Embedding::new).collect();

// Fit a projection from a corpus — PCA, Kernel PCA, Laplacian eigenmap,
// or UMAP-on-sphere. All `fit` constructors return a Result.
let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude).unwrap();
let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Magnitude).unwrap();

// Project a single embedding to the sphere
let point = pca.project(&corpus[0]);

// Full pipeline for search, concept paths, category enrichment, etc.
let input = PipelineInput {
    categories,                // Vec<String>, one per embedding
    embeddings: raw_vectors,   // Vec<Vec<f64>>
};
let pipeline = SphereQLPipeline::new(input).unwrap();

// Or pick a non-default projection family (Pca is the default; KernelPca,
// LaplacianEigenmap, and UmapSphere are the others):
//
//   let config = PipelineConfig {
//       projection_kind: ProjectionKind::UmapSphere,
//       ..Default::default()
//   };
//   let pipeline = SphereQLPipeline::new_with_config(input, config).unwrap();

// k-NN search — `query` returns Result<SphereQLOutput, PipelineError>;
// unknown ids/categories surface as errors, not empty results.
let query = PipelineQuery { embedding: query_vec };
let results = pipeline.query(SphereQLQuery::Nearest { k: 5 }, &query).unwrap();

// Concept path between two items
let path = pipeline.query(
    SphereQLQuery::ConceptPath {
        source_id: "s-0001",
        target_id: "s-0042",
        graph_k: 10,
    },
    &query,
).unwrap();

// Cluster detection
let globs = pipeline.query(
    SphereQLQuery::DetectGlobs { k: None, max_k: 10 },
    &query,
).unwrap();

// --- Category Enrichment Layer ---

// Category-level concept path
let cat_path = pipeline.query(
    SphereQLQuery::CategoryConceptPath {
        source_category: "science",
        target_category: "cooking",
    },
    &query,
).unwrap();

// Nearest neighbor categories
let neighbors = pipeline.query(
    SphereQLQuery::CategoryNeighbors { category: "science", k: 3 },
    &query,
).unwrap();

// Drill down into a category (uses inner sphere if available)
let drill = pipeline.query(
    SphereQLQuery::DrillDown { category: "science", k: 5 },
    &query,
).unwrap();

// Category stats
let stats = pipeline.query(SphereQLQuery::CategoryStats, &query).unwrap();

// Export for visualization. The quality score is EVR for PCA / kernel
// PCA, a connectivity ratio for Laplacian, and kNN-recall
// trustworthiness for UMAP — same [0, 1] scale, different meaning.
let points = pipeline.exported_points();
let quality = pipeline.explained_variance_ratio();
```

See [projections.md](projections.md) for a tour of the four projection
families and [auto-tuning.md](auto-tuning.md) for the `PipelineConfig` +
`auto_tune` workflow.

## Visualization

Enable the `vis` feature (included in `full`) to render a corpus to a
self-contained, interactive 3D sphere. The emitted HTML inlines the Three.js
runtime, so it opens offline.

```rust
use sphereql::vis::{Scene, ScenePoint, SceneStats};

// Map projected points into a scene (here, literals; in practice convert
// `pipeline.exported_points()` into `ScenePoint`s — see the
// `visualize_corpus` example for the full overlay set).
let scene = Scene::builder()
    .title("My corpus")
    .points(vec![
        ScenePoint::from_spherical("science", "atom", 1.0, 0.3, 1.2),
        ScenePoint::from_spherical("cooking", "bread", 1.0, 2.7, 0.9),
        ScenePoint::from_spherical("science", "energy", 1.0, 0.5, 1.0),
    ])
    // The label is set per projection family, not hardcoded.
    .stats(SceneStats::new("pca", 0.83).with_label("PCA variance"))
    .build();

let html = scene.to_html(); // offline; `to_html_cdn()` for a smaller file
std::fs::write("sphere_viz.html", html).unwrap();
```

The `sphereql-examples` crate's `visualize_corpus` example shows the full
workflow: load a corpus, auto-tune a projection, and emit a scene with
category centroids, classified bridges, geodesic concept paths, Voronoi caps,
antipodes, and domain-group spokes.

For corpora past a few hundred thousand points, inlining one JSON blob stops
scaling: `sphereql-vis-server` instead streams binary `SQT1` tiles by viewport.
The full runbook, architecture, and HTTP API are in
[visualization.md](visualization.md).

## GraphQL

```rust,ignore
use sphereql::graphql::*;

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
