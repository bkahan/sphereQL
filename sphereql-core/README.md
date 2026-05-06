# sphereql-core

Spherical math primitives for the [sphereQL](https://github.com/bkahan/sphereQL) project.

This is the foundational crate. Every other crate in the workspace
re-exports types from here, so changes to this surface ripple
everywhere — keep them deliberate.

## What's here

- **Coordinate types** — `SphericalPoint { r, theta, phi }` (physics
  convention: theta in [0, 2π), phi in [0, π]), `CartesianPoint`,
  `GeoPoint` for lat/lon/alt input.
- **Distance metrics** — `angular_distance` (Vincenty, numerically
  stable across the antipodal seam), great-circle, chord, and a
  cosine-similarity helper that returns `Result` so caller code
  handles zero vectors explicitly.
- **Interpolation** — `slerp` (spherical linear) and `nlerp` (normalized
  lerp) with an antipodal branch in slerp for numerically safe
  interpolation through the pole.
- **Region primitives** — `Cone`, `Cap`, `Shell`, `Band`, `Wedge`,
  plus boolean `Region::Intersection` / `Region::Union`. All implement
  the `Contains<SphericalPoint>` trait.
- **Errors** — `SphereQlError` is `#[non_exhaustive]`; new variants
  can be added without breaking downstream `match` arms.

## Versioning

Pre-1.0. Public API is stable enough to ship against, but reserve the
right to break on minor bumps. See the workspace
[CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md).

## Documentation

Crate-level rustdoc lives at the top of `src/lib.rs`. For algorithmic
detail and tradeoff discussion, see the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and [coordinate-system.md](https://github.com/bkahan/sphereQL/blob/main/docs/coordinate-system.md).
