# Prompt 04: Implement the pipeline binding for Python

## Context

The `sphereql-python` crate has core types (prompt 02) and projections (prompt
03). This prompt implements `sphereql-python/src/pipeline.rs` — the main
`Pipeline` class that wraps `SphereQLPipeline`.

## Source Types (in `sphereql-embed/src/pipeline.rs`)

```rust
pub struct PipelineInput {
    pub categories: Vec<String>,
    pub embeddings: Vec<Vec<f64>>,
}

pub struct SphereQLPipeline { /* private */ }
impl SphereQLPipeline {
    pub fn new(input: PipelineInput) -> Self;
    pub fn with_projection(categories: Vec<String>, embeddings: Vec<Embedding>, pca: PcaProjection) -> Self;
    pub fn query(&self, q: SphereQLQuery, query_embedding: &PipelineQuery) -> SphereQLOutput;
    pub fn num_items(&self) -> usize;
    pub fn categories(&self) -> &[String];
}

pub enum SphereQLQuery<'a> {
    Nearest { k: usize },
    SimilarAbove { min_cosine: f64 },
    ConceptPath { source_id: &'a str, target_id: &'a str, graph_k: usize },
    DetectGlobs { k: Option<usize>, max_k: usize },
    LocalManifold { neighborhood_k: usize },
}

pub enum SphereQLOutput {
    Nearest(Vec<NearestResult>),
    KNearest(Vec<NearestResult>),
    ConceptPath(Option<PathResult>),
    Globs(Vec<GlobSummary>),
    LocalManifold(ManifoldResult),
}

pub struct NearestResult {
    pub id: String,
    pub category: String,
    pub distance: f64,
    pub certainty: f64,
    pub intensity: f64,
}

pub struct PathResult {
    pub steps: Vec<PipelinePathStep>,
    pub total_distance: f64,
}

pub struct PipelinePathStep {
    pub id: String,
    pub category: String,
    pub cumulative_distance: f64,
}

pub struct GlobSummary {
    pub id: usize,
    pub centroid: [f64; 3],
    pub member_count: usize,
    pub radius: f64,
    pub top_categories: Vec<(String, usize)>,
}

pub struct ManifoldResult {
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub variance_ratio: f64,
}
```

## Task

Implement `sphereql-python/src/pipeline.rs` with:

### 1. `PyPipeline`

```python
import numpy as np
import sphereql

categories = ["science", "cooking", "sports", ...]
embeddings = np.random.randn(100, 384).astype(np.float64)

# Create pipeline (fits PCA internally)
pipeline = sphereql.Pipeline(categories, embeddings)

# Or with a pre-fitted PCA
pca = sphereql.PcaProjection.fit(embeddings)
pipeline = sphereql.Pipeline(categories, embeddings, projection=pca)

pipeline.num_items    # 100
pipeline.categories   # ["science", "cooking", ...]
```

- `#[pyclass(name = "Pipeline")]`
- Constructor accepts:
  - `categories: list[str]`
  - `embeddings: numpy.ndarray` (2D float64)
  - Optional `projection: PyPcaProjection` — if provided, use
    `SphereQLPipeline::with_projection` to avoid fitting a second PCA.
    If not provided, use `SphereQLPipeline::new`.
- Properties: `num_items -> usize`, `categories -> Vec<String>`.

### 2. Query methods

Each query method should accept the query embedding as a 1D numpy array or
`list[float]`, and return typed Python objects.

#### `nearest(query, k=5) -> list[NearestHit]`

```python
hits = pipeline.nearest(query_vec, k=10)
hits[0].id           # "s-0042"
hits[0].category     # "science"
hits[0].distance     # 0.123
hits[0].certainty    # 0.95
hits[0].intensity    # 4.7
```

#### `similar_above(query, min_cosine=0.8) -> list[NearestHit]`

```python
hits = pipeline.similar_above(query_vec, min_cosine=0.7)
```

#### `concept_path(source_id, target_id, graph_k=10, query=None) -> PathResult | None`

```python
path = pipeline.concept_path("s-0000", "s-0050", graph_k=15, query=dummy_vec)
if path:
    path.total_distance   # 1.23
    path.steps[0].id      # "s-0000"
    path.steps[-1].id     # "s-0050"
```

Note: `SphereQLPipeline.query()` requires a `PipelineQuery` even for
`ConceptPath`. Accept an optional `query` kwarg; if not provided, use a
zero vector of the correct dimensionality.

#### `detect_globs(k=None, max_k=10, query=None) -> list[GlobInfo]`

```python
globs = pipeline.detect_globs(k=3, max_k=10, query=dummy_vec)
globs[0].id              # 0
globs[0].centroid        # [0.1, 0.2, 0.3]
globs[0].member_count    # 15
globs[0].radius          # 0.45
globs[0].top_categories  # [("science", 8), ("tech", 5)]
```

#### `local_manifold(query, neighborhood_k=10) -> ManifoldInfo`

```python
m = pipeline.local_manifold(query_vec, neighborhood_k=15)
m.centroid         # [0.1, 0.2, 0.3]
m.normal           # [0.0, 0.0, 1.0]
m.variance_ratio   # 0.92
```

### 3. Result types

Define lightweight `#[pyclass]` types for query results. These are returned
by the query methods, not constructed by users.

#### `PyNearestHit`

- `#[pyclass(name = "NearestHit")]`
- Properties: `id: String`, `category: String`, `distance: f64`,
  `certainty: f64`, `intensity: f64`.
- `__repr__`.
- No constructor exposed to Python — only created internally.

#### `PyPathResult`

- `#[pyclass(name = "PathResult")]`
- Properties: `total_distance: f64`, `steps: Vec<PyPathStep>`.
- `__repr__`.

#### `PyPathStep`

- `#[pyclass(name = "PathStep")]`
- Properties: `id: String`, `category: String`, `cumulative_distance: f64`.
- `__repr__`.

#### `PyGlobInfo`

- `#[pyclass(name = "GlobInfo")]`
- Properties: `id: usize`, `centroid: [f64; 3]` (as Python list),
  `member_count: usize`, `radius: f64`,
  `top_categories: Vec<(String, usize)>` (as Python list of tuples).
- `__repr__`.

#### `PyManifoldInfo`

- `#[pyclass(name = "ManifoldInfo")]`
- Properties: `centroid: [f64; 3]` (as list), `normal: [f64; 3]` (as list),
  `variance_ratio: f64`.
- `__repr__`.

### 4. Register result types in the module

Go back to `lib.rs` and register all result types:

```rust
m.add_class::<pipeline::PyNearestHit>()?;
m.add_class::<pipeline::PyPathResult>()?;
m.add_class::<pipeline::PyPathStep>()?;
m.add_class::<pipeline::PyGlobInfo>()?;
m.add_class::<pipeline::PyManifoldInfo>()?;
```

## Implementation Notes

- For methods that take a query embedding, use the `extract_embedding` helper
  from the projection module.
- Release the GIL during `SphereQLPipeline::new()` and `query()` calls using
  `py.allow_threads(|| ...)` — these are pure Rust computation.
- The `PipelineQuery` wrapper just holds `Vec<f64>` — construct it inline.
- For `concept_path` and `detect_globs`, if no query is provided, create a
  zero vector: `vec![0.0; pipeline.num_items()]` won't work because we need
  the embedding dimension, not item count. Instead, store the dimensionality
  in `PyPipeline` during construction and use `vec![0.0; dim]`.
- Return Python `None` from `concept_path` if no path is found (use
  `Option<PyPathResult>` which PyO3 maps to `None`).

## Verification

`cargo check -p sphereql-python` should now pass with zero errors. Run
`cargo clippy -p sphereql-python` and fix any warnings.
