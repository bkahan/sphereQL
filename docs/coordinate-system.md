# Coordinate system

sphereQL uses the **physics convention** for spherical coordinates:

- **r** — radial distance from origin (r ≥ 0)
- **θ** (theta) — azimuthal angle in the xy-plane from the x-axis,
  range [0, 2π)
- **φ** (phi) — polar angle from the z-axis, range [0, π]

Geographic coordinates use standard (latitude, longitude, altitude) with
latitude in [-90, 90] and longitude in [-180, 180].

## Conversion examples

```rust
use sphereql::core::*;

let p = SphericalPoint::new(1.0, 0.5, 0.8).unwrap();
let cart = spherical_to_cartesian(&p);             // → CartesianPoint(x, y, z)
let geo = spherical_to_geo(&p);                    // → GeoPoint(lat, lon, alt)
```

## Why physics convention

Most machine-learning libraries default to Cartesian. We use physics
convention (θ azimuthal, φ polar) because it pairs naturally with the
latitude/longitude geographic convention for the radial coordinate *r*,
and because spherical Voronoi + cap geometry (used by the auto-tuner's
spatial quality metrics) reads cleanly with this parametrization.

If you come from the mathematics convention (θ polar, φ azimuthal), be
aware the convention here is not just a labeling choice: `SphericalPoint::new`
validates θ ∈ [0, 2π) and φ ∈ [0, π], and every conversion
(`spherical_to_cartesian`, `spherical_to_geo` with lat = 90° − φ°) treats φ as
the polar angle. Map your angles into this convention at construction time —
there is no toggle.
