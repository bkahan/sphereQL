# Prompt 01: Scaffold the sphereql-python crate

## Context

The sphereQL workspace at `/Users/benkahan/Documents/Code/sphereQL` is a Rust
monorepo with these crates:

- `sphereql-core` — types (`SphericalPoint`, `CartesianPoint`, `GeoPoint`),
  conversions, distance functions, regions
- `sphereql-index` — spatial index with shell/sector partitioning
- `sphereql-embed` — embedding projection engine (PCA, random projection),
  pipeline, query, `ProjectedPoint`, `EmbeddingIndex`, `SphereQLPipeline`
- `sphereql-layout` — layout strategies
- `sphereql-vectordb` — vector database connectors (in-memory, qdrant)
- `sphereql-wasm` — wasm-bindgen bindings for JS
- `sphereql-graphql` — GraphQL schema

We need a new `sphereql-python` crate that exposes Python bindings via PyO3.
The module should be importable as `import sphereql` in Python.

## Task

1. Create `sphereql-python/Cargo.toml` with:
   ```toml
   [package]
   name = "sphereql-python"
   description = "Python bindings for the sphereQL spherical coordinate engine"
   edition.workspace = true
   version.workspace = true
   license.workspace = true
   repository.workspace = true

   [lib]
   name = "sphereql"
   crate-type = ["cdylib"]

   [dependencies]
   sphereql-core = { path = "../sphereql-core" }
   sphereql-embed = { path = "../sphereql-embed" }
   sphereql-index = { path = "../sphereql-index" }
   pyo3 = { version = "0.24", features = ["extension-module"] }
   numpy = "0.24"
   serde_json = "1"

   [build-dependencies]
   pyo3-build-config = "0.24"
   ```

2. Create `sphereql-python/pyproject.toml` for maturin:
   ```toml
   [build-system]
   requires = ["maturin>=1.0,<2.0"]
   build-backend = "maturin"

   [project]
   name = "sphereql"
   requires-python = ">=3.9"
   classifiers = [
       "Programming Language :: Rust",
       "Programming Language :: Python :: Implementation :: CPython",
   ]

   [tool.maturin]
   features = ["pyo3/extension-module"]
   module-name = "sphereql"
   ```

3. Create `sphereql-python/src/lib.rs` with the PyO3 module entry point:
   ```rust
   use pyo3::prelude::*;

   mod core_types;
   mod projection;
   mod pipeline;

   #[pymodule]
   fn sphereql(m: &Bound<'_, PyModule>) -> PyResult<()> {
       // core geometry types
       m.add_class::<core_types::PySphericalPoint>()?;
       m.add_class::<core_types::PyCartesianPoint>()?;
       m.add_class::<core_types::PyGeoPoint>()?;
       m.add_class::<core_types::PyProjectedPoint>()?;

       // projection
       m.add_class::<projection::PyPcaProjection>()?;
       m.add_class::<projection::PyRandomProjection>()?;

       // pipeline
       m.add_class::<pipeline::PyPipeline>()?;

       // standalone functions
       m.add_function(wrap_pyfunction!(core_types::angular_distance, m)?)?;
       m.add_function(wrap_pyfunction!(core_types::great_circle_distance, m)?)?;
       m.add_function(wrap_pyfunction!(core_types::chord_distance, m)?)?;
       m.add_function(wrap_pyfunction!(core_types::spherical_to_cartesian, m)?)?;
       m.add_function(wrap_pyfunction!(core_types::cartesian_to_spherical, m)?)?;
       m.add_function(wrap_pyfunction!(core_types::spherical_to_geo, m)?)?;
       m.add_function(wrap_pyfunction!(core_types::geo_to_spherical, m)?)?;

       Ok(())
   }
   ```

4. Create stub files for the three submodules:
   - `sphereql-python/src/core_types.rs` — empty, will be filled in prompt 02
   - `sphereql-python/src/projection.rs` — empty, will be filled in prompt 03
   - `sphereql-python/src/pipeline.rs` — empty, will be filled in prompt 04

5. Add `"sphereql-python"` to the workspace members list in the root
   `Cargo.toml`.

6. Create `sphereql-python/.gitignore`:
   ```
   /target
   *.so
   *.pyd
   *.egg-info
   __pycache__
   ```

## Verification

Run `cargo check -p sphereql-python` — it will fail because the submodules are
stubs. That's expected. The goal is just to have the scaffold in place.

## Do NOT

- Write any `#[pyclass]` or `#[pymethods]` implementations yet.
- Add the vectordb crate as a dependency (async complicates things; we'll add
  it in a later prompt if needed).
