# sphereql-layout

Layout engines for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Given a set of items with affinities, produce positions on S² that
respect those affinities. Used by sphereQL's category enrichment
layer to lay out concepts and by visualization frontends to render
coherent point clouds.

## What's here

- **`UniformLayout`** — Fibonacci spiral over the sphere. Best when
  you have no affinity information and just want a visually even
  spread. Min-pair distance scan parallelized via rayon above n=128.
- **`ClusteredLayout`** — k-means on S² with weighted-mean centroid
  updates and a parallel silhouette score that pre-buckets cluster
  members so cost is O(n²) instead of O(n² · k).
- **`ForceDirectedLayout`** — repulsive simulation with great-circle
  distances. Replaces the `cartesian_to_spherical → angular_distance`
  hot path with `dot.clamp(-1, 1).acos()` since points are already
  unit Cartesian. Per-iteration force computation is parallelized
  above n=128.
- **`ManagedLayout`** — incremental updates: insert / remove / move
  with quality-metric-driven re-layout decisions.
- **Quality metrics** — silhouette, packing density, neighborhood
  preservation; consumed by the auto-tuner to pick layout
  hyperparameters per corpus.

All layouts produce `Vec<LayoutEntry<T>>` with item id, position, and
metadata so they round-trip through the index and visualization
crates.

## Versioning

Pre-1.0. Builder types are `#[must_use]`. See the workspace
[CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md).

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and
[performance.md](https://github.com/bkahan/sphereQL/blob/main/docs/performance.md).
