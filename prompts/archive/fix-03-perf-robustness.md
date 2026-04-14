# Fix: Performance & Robustness Issues

## Priority: MEDIUM — correctness under load, API robustness

## Context

Review found performance bottlenecks under scale, missing error handling
at FFI boundaries, and API design issues that become problems as usage
grows.

---

## 1. Python GIL held during expensive operations

**Files:**
- `sphereql-python/src/lib.rs:76-98` (`search_nearest`)
- `sphereql-python/src/lib.rs:183-206` (`KernelPcaProjection::fit`)

Both operations are pure Rust with no Python object access but hold the
GIL throughout. `search_nearest` is O(n) for large indices. Kernel PCA
fit is O(n^2 * d).

**Fix:** Wrap compute-heavy calls in `py.allow_threads(|| ...)`:

```rust
fn search_nearest(&self, py: Python<'_>, query: PyEmbedding, k: usize) -> Vec<PySearchResult> {
    let inner = &self.inner;
    let query_inner = &query.inner;
    let results = py.allow_threads(|| inner.search_nearest(query_inner, k));
    results.items.iter().map(|item| PySearchResult { ... }).collect()
}
```

Apply to: `search_nearest`, `project`, `project_rich`, `project_batch`,
`project_rich_batch`, `KernelPcaProjection::fit`, `fit_with_sigma`.

---

## 2. WASM panics become unrecoverable traps

**File:** `sphereql-wasm/src/lib.rs:73-88`

Dimension mismatch in `search_nearest`/`insert`/`project` triggers
`assert_eq!` which panics, killing the entire WASM instance with a
cryptic "unreachable executed" error.

**Fix:** Validate dimensions at the FFI boundary and return `JsError`:

```rust
#[wasm_bindgen]
pub fn search_nearest(&self, query: &[f64], k: usize) -> Result<Vec<JsValue>, JsError> {
    let idx = self.inner.borrow();
    if query.len() != idx.dimensionality() {
        return Err(JsError::new(&format!(
            "expected {} dimensions, got {}", idx.dimensionality(), query.len()
        )));
    }
    // ... safe to proceed
}
```

Apply to every public method that touches projection: `insert`,
`search_nearest`, `project`, `project_rich`.

---

## 3. `SimilarAbove` recomputes projection in inner loop

**File:** `sphereql-embed/src/pipeline.rs:227`

`self.pca.project(&emb)` is called once per result item inside the
`.map()` closure. Trivial one-line hoist.

**Fix:** Move `let sp_q = self.pca.project(&emb);` before the `.map()`.

---

## 4. `concept_path` O(n^2) with no caching

**File:** `sphereql-embed/src/query.rs:176`

For each of n items, `self.index.nearest(item, k+1)` is O(n). Total
O(n^2 * k). No memoization. Repeated `ConceptPath` queries rebuild
the entire graph.

**Fix:** Either:
- Add a doc comment warning about O(n^2) complexity, or
- Cache the k-NN graph and invalidate on insert/remove, or
- Add a `max_n` guard that errors if the index is too large.

---

## 5. O(n^2) quality metrics unbounded — DoS vector

**Files:**
- `sphereql-layout/src/clustered.rs:276-348`
- `sphereql-layout/src/force.rs:77-97`
- `sphereql-layout/src/quality.rs:7-26`

Silhouette scores and force simulation compute all pairwise distances
with no guard. Reachable from GraphQL query path. 10K items = 50M pairs.

**Fix:** Add a `max_quality_n` threshold (e.g., 5000). Above it, either
sample pairs or skip quality computation and return `None`.

---

## 6. `sector_diagonal` overestimates near poles

**File:** `sphereql-index/src/sector.rs:246`

`sqrt(d_theta^2 + d_phi^2)` treats spherical angles as Euclidean. Near
poles, `d_theta` extent maps to near-zero arc length but the diagonal
stays large. Causes `items_in_nearby_sectors` to scan too many sectors
for pole-adjacent queries.

**Fix:** Use `sin(phi) * d_theta` for the theta component:
```rust
let effective_d_theta = d_theta * (phi_center).sin();
let diagonal = (effective_d_theta * effective_d_theta + d_phi * d_phi).sqrt();
```

---

## 7. Tokio runtime per store instance

**File:** `sphereql-python/src/vectordb.rs:65-69`

Each `PyInMemoryStore` creates a new multi-thread Tokio runtime.
`block_on` inside an existing async context panics.

**Fix:** Use `tokio::runtime::Handle::try_current()` to detect existing
runtime, only create a new one if none exists:
```rust
let rt = tokio::runtime::Handle::try_current()
    .map(|h| h.clone())
    .unwrap_or_else(|_| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .handle()
            .clone()
    });
```

---

## 8. `UniformLayout` discards mapper results

**File:** `sphereql-layout/src/uniform.rs:79-81`

`DimensionMapper` is called and results discarded. The trait contract
implies the mapper informs placement, but `UniformLayout` ignores it
entirely.

**Fix:** Either:
- Change the trait signature so mapper is optional, or
- Document on `UniformLayout` that it ignores semantic positions, or
- Remove the mapper call entirely (don't call then discard).

---

## 9. `ManagedLayout::remove` dirty-set stale indices

**File:** `sphereql-layout/src/managed.rs:32-48`

If `add` + `remove` interleaves without `reflow`, stale indices can
remain in the dirty set. `reflow_incremental` would then index out of
bounds.

**Fix:** In `remove`, after shifting indices, add a bounds check:
```rust
dirty.retain(|&i| i < items.len());
```

---

## 10. Facade glob re-exports expose internals

**File:** `sphereql/src/lib.rs:1-11`

`pub use sphereql_core::*` etc. exports `SplitMix64`, `normalize_vec`,
`dot` at the top level.

**Fix:** Use explicit re-exports for the public API:
```rust
pub use sphereql_core::{SphericalPoint, CartesianPoint, GeoPoint, angular_distance, ...};
pub use sphereql_embed::{Embedding, PcaProjection, KernelPcaProjection, ...};
```

Or add a `pub mod prelude` with the curated set.

---

## 11. Bridge code duplication (`sphereql-python`)

**File:** `sphereql-python/src/vectordb.rs:238-714`

Three bridge types repeat five identical query methods. Any fix must be
applied 3 times.

**Fix:** Extract a `impl_bridge_queries!` macro or a shared trait impl.

---

## 12. Non-contiguous numpy arrays take slow path silently

**File:** `sphereql-python/src/projection.rs:49-80`

`as_slice()` fails on Fortran-order/transposed arrays, falling through
to Python-level `extract::<Vec<Vec<f64>>>()` with no diagnostic.

**Fix:** Call `as_array().as_standard_layout()` to normalize contiguity
before `as_slice`, or emit a Python warning when falling back.

---

## Verification

- GIL: benchmark `search_nearest` from two Python threads; confirm they
  run concurrently after fix.
- WASM: call `search_nearest` with wrong-dimension array from JS; confirm
  `JsError` thrown, not `unreachable`.
- Quality DoS: call quality metric with 10K items; confirm it either
  samples or returns None within reasonable time.
- Dirty set: `add(A), add(B), remove(A), reflow_incremental()` should
  not panic.
