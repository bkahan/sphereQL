# Prompt: Add Data Export Methods to SphereQLPipeline

## Context

The `SphereQLPipeline` in `sphereql-embed/src/pipeline.rs` builds an
internal spatial index over projected embeddings, but there's no way to
extract the projected data for external use (visualization, export,
debugging, serialization). Several downstream features need this:

- The visualization tool needs (id, category, x, y, z, certainty) per item.
- Users want to export projected coordinates to CSV/JSON.
- The Python bindings need to expose projected data without re-projecting.

## Deliverables

### 1. Add export methods to `SphereQLPipeline` in `sphereql-embed/src/pipeline.rs`

```rust
/// Projected data for a single item, suitable for export or visualization.
#[derive(Debug, Clone)]
pub struct ExportedPoint {
    pub id: String,
    pub category: String,
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub certainty: f64,
    pub intensity: f64,
}
```

Add these methods to `SphereQLPipeline`:

```rust
/// Export all projected points with their Cartesian and spherical coordinates.
///
/// Returns one `ExportedPoint` per indexed item, in insertion order.
/// Useful for visualization, CSV export, and debugging the projection.
pub fn exported_points(&self) -> Vec<ExportedPoint> { ... }

/// The PCA projection's explained variance ratio (0.0–1.0).
/// Higher means the 3D projection preserves more of the original
/// embedding structure.
pub fn explained_variance_ratio(&self) -> f64 { ... }

/// Number of categories in the corpus.
pub fn num_categories(&self) -> usize { ... }

/// Unique category names in insertion order.
pub fn unique_categories(&self) -> Vec<String> { ... }
```

Implementation notes:
- `exported_points` should iterate over `self.ids`, look up each item
  in `self.index` (via the `EmbeddingIndex::get` method), and combine
  with `self.categories` and `self.cart_points`.
- For certainty/intensity, use the `EmbeddingItem::certainty()` and
  `EmbeddingItem::intensity()` methods that already exist.
- The pipeline already stores `cart_points: Vec<[f64; 3]>`, so x/y/z
  come from there. Spherical coords come from the item's position.

### 2. Add `to_json` and `to_csv` export methods

```rust
/// Serialize all projected points as a JSON array string.
///
/// Each element: {"id", "category", "r", "theta", "phi", "x", "y", "z", "certainty", "intensity"}
pub fn to_json(&self) -> String { ... }

/// Serialize all projected points as CSV with a header row.
pub fn to_csv(&self) -> String { ... }
```

For `to_json`, use `serde_json`. Add `#[derive(serde::Serialize)]` to
`ExportedPoint`.

For `to_csv`, format manually (no CSV library needed for simple flat
data): header row, then one row per point with comma separation and
6 decimal places for floats.

### 3. Expose in Python bindings

In `sphereql-python/src/pipeline.rs`, add to the existing `Pipeline`
class:

```python
# Returns list of dicts
pipeline.exported_points()
# -> [{"id": "s-0000", "category": "science", "r": 1.2, "theta": 0.5, "phi": 0.8, "x": 0.3, "y": 0.7, "z": 0.6, "certainty": 0.95, "intensity": 1.1}, ...]

pipeline.explained_variance_ratio()  # -> 0.85

pipeline.to_json()  # -> JSON string
pipeline.to_csv()   # -> CSV string

pipeline.unique_categories()  # -> ["science", "cooking", ...]
```

### 4. Expose in WASM bindings

In `sphereql-wasm/src/lib.rs`, add to the existing `Pipeline`:

```rust
/// Export all projected points as JSON.
pub fn export_json(&self) -> Result<String, JsError> { ... }

/// PCA explained variance ratio.
pub fn explained_variance(&self) -> f64 { ... }
```

### 5. Tests

Add tests in `sphereql-embed/src/pipeline.rs` (in the existing test
module):

- `test_exported_points_count` — build pipeline with 20 items, verify
  `exported_points().len() == 20`.
- `test_exported_points_fields` — verify each point has valid coordinates
  (r >= 0, theta in [0, 2π), phi in [0, π]).
- `test_exported_points_categories` — verify categories match input.
- `test_to_json_parseable` — verify `to_json()` output parses as valid
  JSON array with correct length.
- `test_to_csv_lines` — verify `to_csv()` has header + N data lines.
- `test_explained_variance` — verify ratio is in (0.0, 1.0].
- `test_unique_categories` — verify correct set of unique categories.

## Constraints

- Add `serde::Serialize` derive to `ExportedPoint` only. Do NOT add serde
  derives to existing types that don't already have them.
- Do NOT change the `SphereQLPipeline::new` or `with_projection`
  constructors.
- Do NOT change the `SphereQLQuery` enum or `query()` method.
- Run `cargo clippy --workspace --all-features --all-targets` and
  `cargo test --workspace` before done.
