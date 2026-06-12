# sphereql-index

Spatial indexing for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Partitions S² into a two-tier index — radial shells × angular sectors
— so spatial queries scan only the buckets that overlap the query
region instead of the whole point set.

## What's here

- **`SectorIndex<T>`** — angular sector grid (`theta_divisions ×
  phi_divisions`) for fast in-region lookups (`Cone`, `Cap`, `Band`,
  `Wedge`, …) defined in `sphereql-core`.
- **`ShellIndex<T>`** — radial shell partitioning for r-range
  filtering. Composes with `SectorIndex` via `SpatialIndex`.
- **`SpatialIndex<T>`** — composite of shell + sector. Built through
  `SpatialIndexBuilder` (chained config: shell boundaries, theta /
  phi divisions, then `.build()`).
- **`CachedIndex<T>`** — wraps `SpatialIndex` with an LRU cache
  (IndexMap-backed, O(1) touch and evict) over recent query keys,
  invalidated by a generation counter on mutation. `CachedIndexBuilder`
  exposes `.cache_capacity(n)`.
- **k-NN** — uses precomputed unit Cartesian vectors and `1 − dot`
  as a cosine proxy instead of full Vincenty per pair, so the inner
  loop is 3 muls + 2 adds per item.

The `SpatialItem` trait is the integration point: implement it on your
record type to plug into any of the indexes above.

## Example

```rust
use sphereql_core::SphericalPoint;
use sphereql_index::{SpatialIndex, SpatialItem};

#[derive(Debug, Clone)]
struct Star { id: u64, pos: SphericalPoint }

impl SpatialItem for Star {
    type Id = u64;
    fn id(&self) -> &u64 { &self.id }
    fn position(&self) -> &SphericalPoint { &self.pos }
}

let mut idx = SpatialIndex::<Star>::builder()
    .uniform_shells(3, 10.0)
    .theta_divisions(8)
    .phi_divisions(4)
    .build();

idx.insert(Star { id: 1, pos: SphericalPoint::new_unchecked(1.0, 0.5, 0.8) });
idx.insert(Star { id: 2, pos: SphericalPoint::new_unchecked(1.0, 3.0, 2.0) });

let hits = idx.nearest(&SphericalPoint::new_unchecked(1.0, 0.6, 0.9), 1);
assert_eq!(*hits[0].item.id(), 1);
```

## Versioning

Part of the sphereQL workspace, currently `0.2.0-alpha`; API may change
before 1.0. `IndexError` is `#[non_exhaustive]`. Builder types are
`#[must_use]` — forgetting `.build()` warns at the use site.

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and
[performance.md](https://github.com/bkahan/sphereQL/blob/main/docs/performance.md).
