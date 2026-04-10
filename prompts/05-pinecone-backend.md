# Prompt: Add Pinecone Backend to sphereql-vectordb

## Context

`sphereql-vectordb` defines a `VectorStore` async trait with these methods:
`upsert`, `get`, `delete`, `search`, `list`, `count`, `set_payload`,
`dimension`, `collection_name`. There are two implementations today:

- `InMemoryStore` (always available) — in `sphereql-vectordb/src/memory.rs`
- `QdrantStore` (behind `qdrant` feature) — in `sphereql-vectordb/src/qdrant.rs`

The `VectorStoreBridge` is generic over `S: VectorStore`, so adding a new
backend automatically gives it `build_pipeline`, `query`, `hybrid_search`,
and `sync_projections` for free.

The goal is to add a Pinecone backend so users with existing Pinecone
indexes can plug SphereQL in as a cache/re-ranking layer.

## Study the Existing QdrantStore First

Before writing any code, read `sphereql-vectordb/src/qdrant.rs` thoroughly.
It is the template for how a backend should be structured:

- Config struct with builder pattern.
- `ensure_collection` for auto-creating the collection.
- Conversion helpers between the store's native types and SphereQL's
  `VectorRecord`/`SearchResult`/`VectorPage`.
- `set_payload` merges metadata without replacing vectors.
- `list` uses cursor-based pagination with an offset token.
- Tests use `#[cfg(test)]` with a helper to check if a live instance is
  available, skipping gracefully if not.

Match this pattern exactly for Pinecone.

## Deliverables

### 1. Add `pinecone` feature to `sphereql-vectordb/Cargo.toml`

```toml
[features]
default = []
qdrant = ["dep:qdrant-client"]
pinecone = ["dep:reqwest", "dep:serde_json"]

[dependencies]
reqwest = { version = "0.12", features = ["json"], optional = true }
# serde_json is already a dependency
```

Pinecone doesn't have an official Rust SDK, so use their REST API directly
via `reqwest`. This keeps the dependency lightweight.

### 2. Create `sphereql-vectordb/src/pinecone.rs`

**`PineconeConfig`**
```rust
pub struct PineconeConfig {
    pub api_key: String,
    pub host: String,        // e.g. "my-index-abc123.svc.us-east1-gcp.pinecone.io"
    pub namespace: String,   // Pinecone namespace, default ""
    pub dimension: usize,
    pub top_k_limit: usize,  // Pinecone caps at 10,000, default 10,000
}
```

Builder methods: `new(api_key, host, dimension)`, `with_namespace`,
`with_top_k_limit`.

**`PineconeStore`**
```rust
pub struct PineconeStore {
    config: PineconeConfig,
    client: reqwest::Client,
}
```

Implement `VectorStore` using Pinecone's REST API v1:

- **upsert**: `POST /vectors/upsert`
  Body: `{"vectors": [{"id": "...", "values": [...], "metadata": {...}}], "namespace": "..."}`
  Batch in chunks of 100 (Pinecone recommends ≤100 vectors per upsert).

- **get** (fetch): `GET /vectors/fetch?ids=id1&ids=id2&namespace=...`
  Returns `{"vectors": {"id1": {"id", "values", "metadata"}, ...}}`

- **delete**: `POST /vectors/delete`
  Body: `{"ids": ["id1", "id2"], "namespace": "..."}`

- **search** (query): `POST /query`
  Body: `{"vector": [...], "topK": k, "namespace": "...", "includeValues": true, "includeMetadata": true}`
  Returns `{"matches": [{"id", "score", "values", "metadata"}, ...]}`

- **list**: `GET /vectors/list?namespace=...&limit=N&paginationToken=...`
  Note: Pinecone's list endpoint only returns IDs, not vectors. So `list`
  must first call list to get IDs, then call fetch in batches to get the
  full records. This is a known Pinecone limitation.

- **count**: `POST /describe_index_stats`
  Body: `{}` → response has `{"namespaces": {"": {"vectorCount": N}}}`

- **set_payload**: Pinecone calls this "update". `POST /vectors/update`
  Body: `{"id": "...", "setMetadata": {...}, "namespace": "..."}`
  Must be called per-vector (Pinecone doesn't support batch metadata
  updates). Loop through updates sequentially.

- **dimension**: return from config.
- **collection_name**: return `config.namespace` (or `config.host` if
  namespace is empty).

**Error mapping**: Map reqwest errors and Pinecone API error responses to
`VectorStoreError`. Pinecone returns `{"code": N, "message": "..."}` on
errors.

**Headers**: All requests need `Api-Key: {api_key}` header and
`Content-Type: application/json`.

**Base URL**: `https://{config.host}`

### 3. Register in `sphereql-vectordb/src/lib.rs`

```rust
#[cfg(feature = "pinecone")]
pub mod pinecone;
#[cfg(feature = "pinecone")]
pub use pinecone::*;
```

### 4. Add tests in `sphereql-vectordb/src/pinecone.rs`

Add `#[cfg(test)] mod tests` with:

- Unit tests that mock the HTTP responses (use a local test helper that
  checks `PINECONE_API_KEY` and `PINECONE_HOST` env vars — skip if not
  set, just like the Qdrant tests likely do).
- `test_config_builder` — verify defaults.
- `test_upsert_and_fetch` — round-trip test.
- `test_search` — insert, search, verify ordering.
- `test_set_payload` — insert, set metadata, fetch, verify merged.
- `test_delete` — insert, delete, verify gone.
- `test_list_pagination` — insert 15 records with batch_size 5, list all.

For tests without a live Pinecone instance, create an integration test
file at `sphereql-vectordb/tests/pinecone_integration.rs` that's gated
behind `#[ignore]` and can be run explicitly with
`cargo test --features pinecone -- --ignored`.

### 5. Update workspace Cargo.toml if needed

Ensure the `pinecone` feature propagates correctly through the umbrella
`sphereql` crate's feature flags.

### 6. Add to Python bindings (minimal)

In `sphereql-python/src/vectordb.rs`, add a `PyPineconeBridge` behind
`#[cfg(feature = "pinecone")]`:

```rust
#[pyclass]
pub struct PyPineconeBridge {
    inner: VectorStoreBridge<PineconeStore>,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PyPineconeBridge {
    #[new]
    fn new(api_key: String, host: String, dimension: usize, namespace: Option<String>) -> PyResult<Self> { ... }
    // Same query/search/sync methods as PyVectorStoreBridge
}
```

## Constraints

- Do NOT modify the `VectorStore` trait or any existing implementations.
- All HTTP calls must use `reqwest` — do not shell out or use other HTTP
  libraries.
- Respect Pinecone's rate limits: add a brief note in the doc comments
  about set_payload being sequential and potentially slow for large
  datasets.
- The Pinecone REST API docs are at https://docs.pinecone.io/reference/api
  — consult them for exact request/response shapes. If in doubt, match
  the shapes from the Pinecone docs exactly.
- Run `cargo clippy --workspace --all-features --all-targets` and
  `cargo test --workspace` (non-integration tests) before done.
