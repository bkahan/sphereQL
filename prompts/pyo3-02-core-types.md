# Prompt 02: Implement core geometry types for Python

## Context

The `sphereql-python` crate was scaffolded in prompt 01. This prompt implements
`sphereql-python/src/core_types.rs` — the Python-facing wrappers for the core
geometry types and distance functions.

## Source Types (in `sphereql-core/src/types.rs`)

```rust
pub struct SphericalPoint { pub r: f64, pub theta: f64, pub phi: f64 }
pub struct CartesianPoint { pub x: f64, pub y: f64, pub z: f64 }
pub struct GeoPoint { pub lat: f64, pub lon: f64, pub alt: f64 }
```

And from `sphereql-embed/src/types.rs`:

```rust
pub struct ProjectedPoint {
    pub position: SphericalPoint,
    pub certainty: f64,       // 0.0–1.0
    pub intensity: f64,       // pre-norm magnitude
    pub projection_magnitude: f64,
}
```

## Task

Implement `sphereql-python/src/core_types.rs` with:

### 1. `PySphericalPoint`

```python
# Python usage:
p = sphereql.SphericalPoint(r=1.0, theta=0.5, phi=0.7)
p.r          # 1.0
p.theta      # 0.5
p.phi        # 0.7
repr(p)      # "SphericalPoint(r=1.0, theta=0.5, phi=0.7)"
p.to_cartesian()  # -> CartesianPoint
p.to_geo()        # -> GeoPoint
```

- `#[pyclass(name = "SphericalPoint")]`
- Constructor validates ranges (r >= 0, theta in [0, 2pi), phi in [0, pi]),
  raising `ValueError` on invalid inputs.
- Read-only properties: `r`, `theta`, `phi`.
- Methods: `to_cartesian() -> PyCartesianPoint`, `to_geo() -> PyGeoPoint`.
- `__repr__`, `__eq__`.
- Store the inner `sphereql_core::SphericalPoint` and expose a
  `pub(crate) fn inner(&self) -> &SphericalPoint` for use by other modules.

### 2. `PyCartesianPoint`

```python
c = sphereql.CartesianPoint(x=1.0, y=0.0, z=0.0)
c.magnitude()       # 1.0
c.normalize()       # -> CartesianPoint
c.to_spherical()    # -> SphericalPoint
```

- `#[pyclass(name = "CartesianPoint")]`
- Constructor takes `x, y, z: f64`.
- Properties: `x`, `y`, `z`.
- Methods: `magnitude() -> f64`, `normalize() -> PyCartesianPoint`,
  `to_spherical() -> PySphericalPoint`.
- `__repr__`, `__eq__`.

### 3. `PyGeoPoint`

```python
g = sphereql.GeoPoint(lat=40.7, lon=-74.0, alt=0.0)
g.to_spherical()    # -> SphericalPoint
```

- `#[pyclass(name = "GeoPoint")]`
- Constructor validates ranges (lat in [-90, 90], lon in [-180, 180],
  alt >= 0), raising `ValueError`.
- Properties: `lat`, `lon`, `alt`.
- Methods: `to_spherical() -> PySphericalPoint`,
  `to_cartesian() -> PyCartesianPoint`.
- `__repr__`, `__eq__`.

### 4. `PyProjectedPoint`

```python
pp = result[0]        # from pipeline.nearest()
pp.position           # -> SphericalPoint
pp.certainty          # 0.95
pp.intensity          # 5.2
pp.projection_magnitude  # 0.87
```

- `#[pyclass(name = "ProjectedPoint")]`
- Properties: `position -> PySphericalPoint`, `certainty`, `intensity`,
  `projection_magnitude`.
- `__repr__`.
- `pub(crate) fn from_inner(inner: ProjectedPoint) -> Self` constructor.

### 5. Distance functions

Expose as module-level `#[pyfunction]`:

```python
d = sphereql.angular_distance(a, b)          # radians
d = sphereql.great_circle_distance(a, b, radius=6371.0)
d = sphereql.chord_distance(a, b)
```

Each takes two `PySphericalPoint` references. Use `&self` pattern for the
point args — extract the inner `SphericalPoint` via `.inner()`.

### 6. Conversion functions

```python
c = sphereql.spherical_to_cartesian(p)       # SphericalPoint -> CartesianPoint
p = sphereql.cartesian_to_spherical(c)       # CartesianPoint -> SphericalPoint
g = sphereql.spherical_to_geo(p)             # SphericalPoint -> GeoPoint
p = sphereql.geo_to_spherical(g)             # GeoPoint -> SphericalPoint
```

## Implementation Notes

- All Py* types should `#[derive(Clone)]` so they can be returned from methods.
- Use `PyResult<Self>` constructors with `PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)`.
- For `__eq__`, compare the inner types using their `PartialEq` impl.
- Do NOT implement `__hash__` (f64 fields make hashing unreliable).

## Verification

After implementing, `cargo check -p sphereql-python` should still fail (other
submodules are stubs), but this file should have no errors of its own. Verify
with: `cargo check -p sphereql-python 2>&1 | grep -v "projection\|pipeline"`
