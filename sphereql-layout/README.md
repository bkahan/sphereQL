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

All strategies implement `LayoutStrategy<T>` and return a
`LayoutResult<T>`: the positioned entries (`LayoutEntry { item,
position }`) plus a `LayoutQuality` block (dispersion, overlap,
silhouette) so callers can compare layouts without recomputing
metrics.

## Example

```rust
use sphereql_core::SphericalPoint;
use sphereql_layout::{DimensionMapper, LayoutStrategy, UniformLayout};

// UniformLayout places items on a Fibonacci lattice and ignores the
// mapper's semantic positions, so a stub mapper is fine here. The
// affinity-driven strategies (ClusteredLayout, ForceDirectedLayout)
// use the mapper to seed each item's natural position.
struct NoMapper;
impl DimensionMapper for NoMapper {
    type Item = &'static str;
    fn map(&self, _: &Self::Item) -> SphericalPoint {
        SphericalPoint::new_unchecked(1.0, 0.0, 0.0)
    }
}

let items = ["alpha", "beta", "gamma", "delta"];
let result = UniformLayout::new().layout(&items, &NoMapper);
for e in &result.entries {
    println!("{} -> theta {:.2}, phi {:.2}", e.item, e.position.theta, e.position.phi);
}
println!("dispersion: {:.2}", result.quality.dispersion_score);
```

## Versioning

Part of the sphereQL workspace, currently `0.2.0-alpha`; API may change
before 1.0. See the workspace
[CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md).

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and
[performance.md](https://github.com/bkahan/sphereQL/blob/main/docs/performance.md).
