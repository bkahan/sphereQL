# Prompt: Package the 3D Sphere Visualization as a CLI + Python Command

## Context

SphereQL already has an example (`sphereql/examples/e2e_transformer.rs`)
that reads a JSON file of embeddings, projects them through PCA onto a 3D
sphere, and writes an interactive HTML visualization using Three.js. The
output file `sphere_viz.html` already exists in the repo root as a working
example.

The problem: this visualization is buried inside a Rust example that
requires manually preparing a JSON file and running `cargo run --example`.
Nobody will discover or use it this way.

The goal is to make this a first-class tool available from both the CLI and
Python, so a developer can go from "I have embeddings" to "I'm looking at
an interactive 3D sphere" in one command.

## Deliverables

### 1. Python function: `sphereql.visualize()`

Add to `sphereql-python/src/viz.rs` (new file) and register in lib.rs:

```python
import sphereql

# From raw data
sphereql.visualize(
    categories=["science", "science", "cooking", "cooking", ...],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    output="sphere.html",         # default: "sphere_viz.html"
    labels=None,                  # optional list[str], one per embedding
    title="My Knowledge Sphere",  # optional
    open_browser=True,            # default: True, opens in default browser
)

# From a Pipeline (already built)
pipeline = sphereql.Pipeline(categories, embeddings)
sphereql.visualize_pipeline(
    pipeline,
    output="sphere.html",
    title="My Knowledge Sphere",
    open_browser=True,
)
```

The function should:
1. Fit a PCA projection (volumetric mode) on the embeddings.
2. Project each embedding to (x, y, z) Cartesian coordinates.
3. Generate a self-contained HTML file with an embedded Three.js scene.
4. Optionally open it in the default browser.

### 2. HTML Template

Create `sphereql-python/src/viz_template.html` (or embed it as a Rust
string constant). The template should be based on the existing
`sphere_viz.html` but generalized with placeholder tokens that get
string-replaced at generation time.

The visualization must include:
- **3D scatter plot** on a unit sphere using Three.js (r128, loaded from
  cdnjs.cloudflare.com).
- **Color-coded points** by category, with a legend.
- **Orbit controls** for rotation/zoom (implement manually since
  OrbitControls isn't on the CDN — use mouse event handlers for
  rotate/zoom/pan, same approach as the existing sphere_viz.html).
- **Hover tooltips** showing the item's label (or id), category, and
  spherical coordinates (θ, φ, r).
- **Click-to-highlight** that dims all other points and shows the 5
  nearest neighbors connected by lines.
- **Category filter toggles** in a sidebar to show/hide categories.
- **Search box** that highlights points whose labels match a substring.
- **Stats panel** showing: number of points, number of categories,
  PCA explained variance ratio, and coordinate ranges.

Style: dark background (#1a1a2e), points as small spheres with category
colors, subtle grid lines on the sphere surface.

### 3. Rust-side implementation

In `sphereql-python/src/viz.rs`:

```rust
#[pyfunction]
#[pyo3(signature = (categories, embeddings, output="sphere_viz.html", labels=None, title=None, open_browser=true))]
fn visualize(
    categories: Vec<String>,
    embeddings: Vec<Vec<f64>>,
    output: &str,
    labels: Option<Vec<String>>,
    title: Option<&str>,
    open_browser: bool,
) -> PyResult<String> { ... }
```

Steps inside:
1. Convert to `Vec<Embedding>`, fit `PcaProjection::fit_default` with
   volumetric mode.
2. Project each embedding, convert to Cartesian (x, y, z).
3. Build a JSON data blob: `[{x, y, z, r, theta, phi, category, label, certainty}, ...]`
4. String-replace the placeholder in the HTML template with the JSON blob
   and the title.
5. Write to `output` path.
6. If `open_browser`, use Python's `webbrowser.open` via PyO3 (call
   `Python::import(py, "webbrowser")?.call_method1("open", (url,))?`).
7. Return the absolute path of the output file.

### 4. Also expose `visualize_pipeline`

```rust
#[pyfunction]
#[pyo3(signature = (pipeline, output="sphere_viz.html", title=None, open_browser=true))]
fn visualize_pipeline(
    pipeline: &Pipeline,  // existing PyO3 Pipeline class
    output: &str,
    title: Option<&str>,
    open_browser: bool,
) -> PyResult<String> { ... }
```

This reuses the projection already fitted inside the pipeline, avoiding
re-fitting PCA. Extract the internal data (categories, projected points)
from the pipeline and generate the same HTML.

This requires adding a method to the Rust `SphereQLPipeline` to export
projected coordinates. If the pipeline doesn't expose this today, add a
`pub fn projected_points(&self) -> Vec<(String, String, SphericalPoint)>`
method that returns (id, category, position) triples. This change goes in
`sphereql-embed/src/pipeline.rs`.

### 5. Tests

Add `sphereql-python/tests/test_viz.py`:

- `test_visualize_creates_file` — call with 20 synthetic embeddings,
  assert file exists and contains `<script>` and `THREE`.
- `test_visualize_pipeline` — build a Pipeline, call visualize_pipeline,
  assert file exists.
- `test_visualize_categories_in_output` — verify the category names appear
  in the HTML output.
- `test_visualize_stats_panel` — verify the explained variance ratio
  appears in the output.

All tests should use `open_browser=False` and write to a temp directory.

## Constraints

- The HTML must be fully self-contained (inline JS/CSS, external scripts
  only from cdnjs.cloudflare.com).
- No new Rust dependencies outside `sphereql-python`. You can use
  `serde_json` (already a dep) for generating the data blob.
- The only change allowed outside `sphereql-python` is adding the
  `projected_points()` method to `SphereQLPipeline` in `sphereql-embed`.
- Keep the Three.js scene performant for up to 50,000 points (use
  `BufferGeometry` with `Points` material, not individual mesh spheres).
- Run `cargo clippy --workspace --all-features --all-targets` and
  `cargo test --workspace` before done.
