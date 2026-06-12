# lingua-spherica

Python coordinate-type primitives for the [sphereQL](https://github.com/bkahan/sphereQL) project's lingua surface.

This package is a **thin skeleton** of types and spherical math
helpers used by example notebooks and downstream Python code. It is a
standalone pure-Python package (hatchling build, no Rust toolchain
required) living alongside, but not inside, the Cargo workspace. The
full text → `ConceptGraph` pipeline lives in the Rust crate
[`sphereql-lingua`](https://github.com/bkahan/sphereQL/tree/main/sphereql-lingua)
and is exposed to Python via the `sphereql-python` bindings behind
the `lingua` Cargo feature (`maturin develop --features lingua`; the
default `sphereql` PyPI wheel does not include it yet).

**Rust is the source of truth.** Don't rebuild the pipeline here —
extend `sphereql-lingua` and re-export through the bindings.

## What's here

- `SphericalPoint(r, theta, phi)` — matches the canonical Rust
  `sphereql_core::SphericalPoint::new(r, theta, phi)` signature so
  positional construction round-trips between languages.
- `Concept`, `Relation`, `RelationType`, `ConceptGraph`,
  `DomainAnchor` — dataclasses for hand-built or
  downstream-produced graphs.
- `coordinates.py` — `angular_distance` (Vincenty), `slerp` with
  antipodal branch, weighted spherical centroid, theta/phi distance
  helpers, semantic-distance combiner. Same algorithms as the Rust
  side; this package's pytest suite covers the math, but parity with
  Rust is maintained by hand — there is no automated cross-language
  check.

## Coordinate convention

Physics convention, identical across the Rust and Python surfaces:

- `θ` (theta) ∈ [0, 2π) — azimuthal angle (domain / longitude)
- `φ` (phi)   ∈ [0, π]  — polar angle (abstraction / colatitude)
- `r`         ∈ (0, ∞)  — radius (epistemic weight)

`SphericalPoint.__post_init__` clamps `phi`, wraps `theta` mod 2π,
and floors `r` at `1e-6` so degenerate inputs don't propagate
silently.

## Status

Pre-1.0 (`0.2.0a0`, tracking the workspace's `0.2.0-alpha`). Public
surface kept small and stable; new functionality ships in
`sphereql-lingua` first and reaches Python through the bindings.
