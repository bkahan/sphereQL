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

## Versioning

Pre-1.0. `IndexError` is `#[non_exhaustive]`. Builder types are
`#[must_use]` — forgetting `.build()` warns at the use site.

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and
[performance.md](https://github.com/bkahan/sphereQL/blob/main/docs/performance.md).
