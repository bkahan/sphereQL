# Prompt: Extend Python Bindings to Expose VectorStoreBridge

## Context

SphereQL is a Rust workspace at the repo root. The `sphereql-python` crate
(`sphereql-python/`) already exposes core types and the embedding pipeline
via PyO3. The `sphereql-vectordb` crate (`sphereql-vectordb/`) implements a
`VectorStoreBridge<S: VectorStore>` that pulls vectors from a store, fits a
PCA projection, builds a `SphereQLPipeline`, and supports `query()`,
`hybrid_search()`, and `sync_projections()`. There is also an
`InMemoryStore` (always available) and a `QdrantStore` (behind the `qdrant`
feature flag).

The Python bindings today do NOT expose the vectordb bridge. The goal is to
make this possible from Python:

```python
import sphereql

# In-memory path (no external DB needed)
store = sphereql.InMemoryStore("my-collection", dimension=768)
store.upsert([
    {"id": "doc-1", "vector": [...], "metadata": {"topic": "science"}},
    {"id": "doc-2", "vector": [...], "metadata": {"topic": "cooking"}},
])

bridge = sphereql.VectorStoreBridge(store, batch_size=100, max_records=500_000)
bridge.build_pipeline(category_key="topic")

# Query
results = bridge.query_nearest(query_vector, k=5)
results = bridge.hybrid_search(query_vector, final_k=5, recall_k=20)

# Write spherical coords back to store
count = bridge.sync_projections()

# Qdrant path (when feature enabled)
qdrant_store = sphereql.QdrantStore(
    url="http://localhost:6334",
    collection="my-embeddings",
    dimension=768,
)
bridge = sphereql.VectorStoreBridge(qdrant_store)
bridge.build_pipeline(category_key="topic")
```

## Requirements

### 1. Add `vectordb` feature flag to `sphereql-python/Cargo.toml`

- Add an optional dependency on `sphereql-vectordb`.
- Add a `vectordb` feature that enables it.
- Add a `qdrant` feature that enables `sphereql-vectordb/qdrant`.
- The `pyproject.toml` should build with `vectordb` enabled by default. The
  `qdrant` feature should be opt-in.

### 2. Create `sphereql-python/src/vectordb.rs`

Expose these PyO3 classes:

**`PyInMemoryStore`**
- `#[new] fn new(collection: String, dimension: usize) -> Self`
- `fn upsert(&self, records: Vec<PyDict>) -> PyResult<()>` — each dict has
  keys `id` (str), `vector` (list[float]), optionally `metadata` (dict).
  Runs the async method using `pyo3_asyncio` or `tokio::runtime::Runtime`
  block_on.
- `fn count(&self) -> PyResult<usize>`

**`PyVectorStoreBridge`** (wraps `VectorStoreBridge<InMemoryStore>`)
- `#[new] fn new(store: PyInMemoryStore, batch_size: Option<usize>, max_records: Option<usize>) -> Self`
- `fn build_pipeline(&mut self, category_key: String) -> PyResult<()>` —
  uses `category_key` to extract the category from each record's metadata.
  Blocks on the async method.
- `fn query_nearest(&self, embedding: Vec<f64>, k: usize) -> PyResult<Vec<PyDict>>` —
  returns list of `{id, category, distance, certainty, intensity}`.
- `fn query_similar(&self, embedding: Vec<f64>, min_cosine: f64) -> PyResult<Vec<PyDict>>`
- `fn query_concept_path(&self, source_id: String, target_id: String, graph_k: usize, embedding: Vec<f64>) -> PyResult<Option<PyDict>>`
- `fn query_detect_globs(&self, embedding: Vec<f64>, k: Option<usize>, max_k: usize) -> PyResult<Vec<PyDict>>`
- `fn hybrid_search(&self, embedding: Vec<f64>, final_k: usize, recall_k: usize) -> PyResult<Vec<PyDict>>`
- `fn sync_projections(&self) -> PyResult<usize>`
- `fn len(&self) -> usize`

**Async handling**: The VectorStore trait is async. Since PyO3 doesn't
natively await Rust futures, create a shared `tokio::runtime::Runtime` and
use `runtime.block_on(...)` for each async call. Store the runtime in the
bridge struct. This is the same pattern used by qdrant-client's own Python
wrapper.

**`PyQdrantBridge`** (behind `#[cfg(feature = "qdrant")]`, wraps
`VectorStoreBridge<QdrantStore>`)
- `#[new] fn new(url: String, collection: String, dimension: usize, api_key: Option<String>) -> PyResult<Self>`
- Same query/search/sync methods as `PyVectorStoreBridge`.

### 3. Register in `sphereql-python/src/lib.rs`

- Add `#[cfg(feature = "vectordb")] mod vectordb;`
- Register `PyInMemoryStore`, `PyVectorStoreBridge` in the module.
- Conditionally register `PyQdrantBridge` when `qdrant` feature is on.

### 4. Add Python tests in `sphereql-python/tests/test_vectordb.py`

Test cases:
- `test_inmemory_upsert_and_count` — insert 20 records, verify count.
- `test_bridge_build_and_nearest` — build pipeline, query nearest 5, verify
  results have expected fields and are distance-sorted.
- `test_bridge_hybrid_search` — build, hybrid search, verify results are
  score-sorted descending.
- `test_bridge_sync_projections` — build, sync, verify count returned.
- `test_bridge_concept_path` — build, query path between first and last
  item, verify path starts and ends correctly.
- `test_bridge_detect_globs` — build with 2 clusters of data, detect
  globs with k=2, verify 2 globs returned.
- `test_bridge_before_build_raises` — query before build_pipeline raises
  an exception.

Use synthetic embeddings (same pattern as the Rust tests in
`sphereql-vectordb/src/bridge.rs`: half the records with strong dim-0,
half with strong dim-1, 10 dimensions).

### 5. Update `pyproject.toml`

- Ensure `maturin` builds with `--features vectordb` by default.
- Add `qdrant` as an optional extra: `pip install sphereql[qdrant]`.

## Constraints

- Do NOT modify any existing crate outside `sphereql-python/`.
- Do NOT add new dependencies to `sphereql-vectordb` or other crates.
- All new Python-facing types must have docstrings.
- Match existing code style: no comments unless the WHY is non-obvious.
- Run `cargo clippy --workspace --all-features --all-targets` and
  `cargo test --workspace` before considering the task done.
