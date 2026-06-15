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
  the `Contains` trait (containment test against a `SphericalPoint`).
- **Errors** — `SphereQlError` is `#[non_exhaustive]`; new variants
  can be added without breaking downstream `match` arms.

## Example

```rust
use sphereql_core::{Cap, Contains, SphereQlError, SphericalPoint, angular_distance, slerp};

fn main() -> Result<(), SphereQlError> {
    let a = SphericalPoint::new(1.0, 0.3, 1.2)?;
    let b = SphericalPoint::new(1.0, 2.1, 0.8)?;

    println!("angular distance: {:.3} rad", angular_distance(&a, &b));

    let mid = slerp(&a, &b, 0.5);
    let cap = Cap::new(a, 1.0)?;
    println!("midpoint inside cap around a: {}", cap.contains(&mid));
    Ok(())
}
```

## Versioning

Part of the sphereQL workspace, currently `0.3.0`. Public API is
stable enough to ship against, but reserve the right to break on minor
bumps before 1.0. See the workspace
[CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md).

## Documentation

Crate-level rustdoc lives at the top of `src/lib.rs`. For algorithmic
detail and tradeoff discussion, see the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and [coordinate-system.md](https://github.com/bkahan/sphereQL/blob/main/docs/coordinate-system.md).
