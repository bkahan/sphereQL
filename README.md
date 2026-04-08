# sphereQL

Production-grade spherical coordinate operations library for Rust with GraphQL integration.

sphereQL provides a layered architecture for working with spherical coordinates — from core math primitives through spatial indexing, layout engines, and a ready-to-use GraphQL API.

## Crates

| Crate              | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `sphereql-core`    | Spherical math primitives: points, conversions, distances, interpolation, regions |
| `sphereql-index`   | Spatial indexing with shell/sector partitioning for fast queries                  |
| `sphereql-layout`  | Layout engine: uniform (Fibonacci), clustered (k-means), force-directed           |
| `sphereql-graphql` | async-graphql integration with queries, subscriptions, and event bus              |
| `sphereql`         | Umbrella crate with feature flags for selective imports                           |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
sphereql = { version = "0.1", features = ["full"] }
```

### Feature Flags

| Feature          | Includes                                         | Dependencies     |
| ---------------- | ------------------------------------------------ | ---------------- |
| `core` (default) | Math primitives, conversions, distances, regions | --               |
| `index`          | Spatial indexing and queries                     | `core`           |
| `layout`         | Layout strategies and quality metrics            | `core`, `index`  |
| `graphql`        | GraphQL schema, subscriptions, event bus         | `core`, `index`  |
| `full`           | Everything                                       | All of the above |

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
let layout = UniformLayout::new(1.0);
let result = layout.layout(&items, &mapper);

// Clustered layout with k-means
let layout = ClusteredLayout::new(4, 1.0, 0.3);

// Force-directed simulation
let layout = ForceDirectedLayout::default();
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
  +-- sphereql-layout (layout strategies, quality metrics)
  |     |
  +-----+-- sphereql-index (spatial partitioning, queries)
  |           |
  +-----------+-- sphereql-core (math primitives, zero dependencies*)
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

## License

MIT
