# Prompt 03: Implement projection bindings for Python

## Context

The `sphereql-python` crate has core types implemented (prompt 02). This prompt
implements `sphereql-python/src/projection.rs` — Python bindings for PCA and
random projections.

## Source Types (in `sphereql-embed/src/projection.rs`)

```rust
pub trait Projection: Send + Sync {
    fn project(&self, embedding: &Embedding) -> SphericalPoint;
    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint;
    fn dimensionality(&self) -> usize;
}

pub struct PcaProjection { /* fields private */ }
impl PcaProjection {
    pub fn fit(embeddings: &[Embedding], radial: RadialStrategy) -> Self;
    pub fn fit_default(embeddings: &[Embedding]) -> Self;
    pub fn with_volumetric(self, enabled: bool) -> Self;
    pub fn explained_variance_ratio(&self) -> f64;
}

pub struct RandomProjection { /* fields private */ }
impl RandomProjection {
    pub fn new(dim: usize, radial: RadialStrategy, seed: u64) -> Self;
    pub fn new_default(dim: usize) -> Self;
}
```

And from `sphereql-embed/src/types.rs`:

```rust
pub enum RadialStrategy {
    Fixed(f64),
    Magnitude,
    MagnitudeTransform(Arc<dyn Fn(f64) -> f64 + Send + Sync>),
}
```

## Task

Implement `sphereql-python/src/projection.rs` with:

### 1. `PyPcaProjection`

```python
import numpy as np
import sphereql

embeddings = np.random.randn(100, 384).astype(np.float64)

# Fit PCA from a corpus
pca = sphereql.PcaProjection.fit(embeddings)
pca = sphereql.PcaProjection.fit(embeddings, radial="magnitude")
pca = sphereql.PcaProjection.fit(embeddings, radial=2.5)  # fixed
pca = sphereql.PcaProjection.fit(embeddings, volumetric=True)

# Query
pca.dimensionality        # 384
pca.explained_variance_ratio  # 0.87

# Project a single embedding
point = pca.project(query_vec)           # -> SphericalPoint
rich = pca.project_rich(query_vec)       # -> ProjectedPoint

# Project a batch
points = pca.project_batch(embeddings)   # -> list[SphericalPoint]
rich_batch = pca.project_rich_batch(embeddings)  # -> list[ProjectedPoint]
```

- `#[pyclass(name = "PcaProjection")]`
- `#[classmethod] fn fit(...)` — accepts embeddings as:
  - `numpy.ndarray` (2D, float64) via `PyReadonlyArray2<f64>` from the `numpy`
    crate
  - Convert rows to `Vec<Embedding>`.
  - Optional kwarg `radial`: `str` (`"magnitude"`) or `f64` (fixed value).
    Default: `"magnitude"`. Do NOT support `MagnitudeTransform` from Python —
    the closure can't cross FFI. Raise `ValueError` if an unrecognized string
    is passed.
  - Optional kwarg `volumetric: bool`, default `false`.
- Properties: `dimensionality -> usize`, `explained_variance_ratio -> f64`.
- `fn project(&self, embedding)` — accepts a 1D numpy array or `list[float]`,
  returns `PySphericalPoint`.
- `fn project_rich(&self, embedding)` — same input, returns `PyProjectedPoint`.
- `fn project_batch(&self, embeddings)` — accepts 2D numpy array, returns
  `Vec<PySphericalPoint>`.
- `fn project_rich_batch(&self, embeddings)` — same, returns
  `Vec<PyProjectedPoint>`.
- Wrap the inner `PcaProjection` in the struct. Expose
  `pub(crate) fn inner(&self) -> &PcaProjection` for use by the pipeline module.
- `__repr__`.

### 2. `PyRandomProjection`

```python
rp = sphereql.RandomProjection(dim=384, seed=42)
rp = sphereql.RandomProjection(dim=384, radial=1.0, seed=42)

point = rp.project(query_vec)
```

- `#[pyclass(name = "RandomProjection")]`
- Constructor: `dim: usize`, optional `radial` (same parsing as PCA),
  optional `seed: u64` (default 42).
- Same `project`, `project_rich`, `project_batch`, `project_rich_batch`
  methods as PCA.
- Property: `dimensionality -> usize`.
- `__repr__`.

### Helper: Embedding extraction

Write a `pub(crate)` helper function to convert Python input to `Embedding`:

```rust
pub(crate) fn extract_embedding(obj: &Bound<'_, PyAny>) -> PyResult<Embedding> {
    // Try numpy array first, then list[float]
}
```

And a batch version:

```rust
pub(crate) fn extract_embeddings(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Embedding>> {
    // 2D numpy array -> Vec<Embedding>
}
```

These should handle both `numpy.ndarray` and `list[list[float]]`. Use
`numpy::PyReadonlyArray1` / `PyReadonlyArray2` for the numpy path, fall back
to iterating Python sequences for lists.

## Implementation Notes

- `RadialStrategy::MagnitudeTransform` is intentionally unsupported from
  Python. The closure can't cross FFI. Document this in the docstring.
- Store the Rust projection type directly (not behind `Arc` or `Box<dyn>`).
  PyO3 wraps `#[pyclass]` types in its own reference-counted container.
- For batch methods, release the GIL during the Rust computation using
  `py.allow_threads(|| ...)` since projection is CPU-bound and we don't need
  Python objects during computation.

## Verification

`cargo check -p sphereql-python` should now only fail on the pipeline stub.
