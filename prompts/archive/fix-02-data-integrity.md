# Fix: Data Integrity & Correctness Issues

## Priority: HIGH — silent wrong results or data loss

## Context

Review found several places where invalid input is silently accepted or
where algorithmic correctness is compromised. These produce wrong results
with no error signal to callers.

---

## 1. Silent data coercion in `from_json` (`sphereql-python`, `sphereql-wasm`)

**Files:**
- `sphereql-python/src/pipeline.rs:76-81`
- `sphereql-wasm/src/lib.rs` (same pattern in `Pipeline::new`)

Non-string categories silently become `"unknown"`, non-numeric values
become `0.0`, malformed rows become empty `Vec`s. The pipeline accepts
structurally invalid data with no error.

**Fix:** Return explicit errors on type mismatch:

```rust
// Instead of:
.map(|v| v.as_str().unwrap_or("unknown").to_string())

// Do:
.map(|v| v.as_str()
    .ok_or_else(|| PyValueError::new_err("category must be a string"))
    .map(|s| s.to_string()))
.collect::<Result<Vec<_>, _>>()?
```

Apply the same pattern to embedding values (reject non-numeric) and
embedding rows (reject non-array).

---

## 2. No NaN/Inf validation at FFI boundaries

**Files:**
- `sphereql-python/src/lib.rs:22-26` (`PyEmbedding::new`)
- `sphereql-wasm/src/lib.rs` (`insert`, `search_nearest`)

NaN and Inf values propagate silently through geometry code. NaN in
angular distance makes all comparisons false. Inf in normalization
produces NaN (inf/inf).

**Fix (Python):**
```rust
#[new]
fn new(values: Vec<f64>) -> PyResult<Self> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "embedding values must be finite (no NaN or Inf)"
        ));
    }
    Ok(Self { inner: Embedding::new(values) })
}
```

**Fix (WASM):** Validate in `insert` and `search_nearest` before
constructing `Embedding`, return `JsError`.

---

## 3. `cosine_similarity` length mismatch — silent truncation

**File:** `sphereql-core/src/distance.rs:123`

`for i in 0..a.len()` with `b[i]` access: if `a` is shorter than `b`,
the result is silently truncated (extra `b` elements ignored). Only a
`debug_assert` guards this — release builds get wrong results.

**Fix:**
```rust
assert_eq!(a.len(), b.len(), "vectors must have equal length for cosine similarity");
let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
```

Use `.zip()` iterator pattern and a release-mode `assert_eq!`.

---

## 4. `Wedge::new` allows zero-width wedge

**File:** `sphereql-core/src/regions.rs:115`

`Band::new` correctly rejects `phi_min >= phi_max`. `Wedge::new` does
not — `theta_min == theta_max` creates a zero-width wedge that only
matches exact boundary values. Inconsistent validation.

**Fix:** Either reject `theta_min == theta_max` (consistent with Band),
or explicitly document it as a valid single-meridian wedge.

---

## 5. `normalize()` returns zero vector for zero-magnitude input

**File:** `sphereql-core/src/geometry.rs:80-86`

Callers that expect a unit vector get all zeros. Downstream dot products
produce NaN or zero, which propagates silently.

**Fix:** Either return `Option<Vec<f64>>` (breaking change), or document
the zero-vector contract in the doc comment and ensure all callers
(especially kernel PCA's `e.normalized()`) handle it.

---

## 6. `build_pipeline` zero-vector check uses wrong metric

**File:** `sphereql-vectordb/src/bridge.rs:99-104`

Per-component `abs() < f64::EPSILON` doesn't catch near-zero vectors
like `[1e-10, 1e-10, ...]`. These cause numerical instability in PCA.

**Fix:** Check L2 norm instead:
```rust
let mag: f64 = r.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
if mag < 1e-9 {
    return Err(VectorStoreError::InvalidConfig("near-zero vector".into()));
}
```

---

## 7. `GeoPoint` validation bypass via `new_unchecked`

**File:** `sphereql-python/src/core_types.rs:294-300`

`py_geo_to_spherical` reconstructs a `GeoPoint` with `new_unchecked`
from getter values. Safe today because `PyGeoPoint` is frozen, but
fragile to future changes.

**Fix:** Add `pub(crate) fn inner_geo(&self) -> &GeoPoint` to
`PyGeoPoint` and use it directly instead of decomposing/reconstructing.

---

## 8. `nlerp` antipodal edge case

**File:** `sphereql-core/src/interpolation.rs:54`

Antipodal inputs produce a zero-radius point (degenerate origin) instead
of a valid midpoint. `slerp` has a guard for this but `nlerp` does not.

**Fix:** Check if the lerped Cartesian point is near-zero magnitude and
fall back to returning `a` (matching slerp's convention).

---

## 9. `.pyi` stub missing `KernelPcaProjection`

**File:** `sphereql-python/python/sphereql/sphereql.pyi`

The class is fully implemented and registered but invisible to Python
type checkers. Users cannot discover or type-check the API.

**Fix:** Add the full class stub matching the pattern for `PcaProjection`:
- `fit(embeddings, radial)`, `fit_with_sigma(embeddings, sigma, radial)`
- `project(embedding)`, `project_rich(embedding)`
- `project_batch(embeddings)`, `project_rich_batch(embeddings)`
- `dimensionality`, `explained_variance_ratio()`, `sigma()`, `num_training_points()`

---

## Verification

- NaN test: `Embedding([float('nan'), 1.0, 2.0])` should raise `ValueError`.
- Cosine test: `cosine_similarity([1.0], [1.0, 2.0])` should panic in release.
- Zero-vector test: fit PCA with a near-zero vector, confirm error not NaN output.
- Stub test: `mypy --strict` on a script using `KernelPcaProjection` should pass.
