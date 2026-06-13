# sphereql-vectordb

Vector database connectors for the [sphereQL](https://github.com/bkahan/sphereQL) project.

A small, opinionated trait (`VectorStore`) over external vector
stores so the rest of the workspace can sync vectors, project them
onto S², and run hybrid spherical-then-cosine search uniformly.

## Backends

- **`InMemoryStore`** (default) — pure-Rust, no network. Caches
  per-vector L2 norms at insert time so cosine queries don't recompute
  them per scan; parallelizes search via rayon above n=256.
- **`QdrantStore`** (feature `qdrant`) — gRPC against a Qdrant
  instance. Supports payload-only updates and scroll pagination.
- **`PineconeStore`** (feature `pinecone`) — REST against Pinecone.
  Honors 429 `Retry-After` headers and caps response size to defend
  against OOM from misbehaving servers.

There's also a product-quantization layer (`PqCodebook`, `PqIndex`,
`InMemoryPqStore`) for compressing full-dimensional vectors; the
`pq-lmdb` feature adds on-disk persistence via heed/LMDB (opt-in
because it pulls in a C build dependency).

## Example

`VectorStore` is async — the snippet assumes a tokio runtime.

```rust
use sphereql_vectordb::{InMemoryStore, VectorRecord, VectorStore};

let store = InMemoryStore::new("demo", 4);

store.upsert(&[
    VectorRecord::new("a", vec![0.1, 0.9, 0.3, 0.0]),
    VectorRecord::new("b", vec![0.9, 0.1, 0.0, 0.5]),
]).await?;

let hits = store.search(&[0.15, 0.85, 0.35, 0.05], 1).await?;
println!("{} (score {:.3})", hits[0].id, hits[0].score);
```

## Hybrid search

The crate is most useful when paired with `sphereql-embed`. The
typical flow:

1. Sync raw embeddings into a backing store via `VectorStore::upsert`.
2. Fit a sphereQL projection on the corpus.
3. Spherical-prune candidates with the spatial index.
4. Re-rank survivors using full-dimensional cosine similarity from
   the backing store.

This yields near-brute-force precision at near-spherical-index
latency. See
[`full_e2e.rs`](https://github.com/bkahan/sphereQL/blob/main/sphereql-examples/examples/full_e2e.rs)
in the workspace examples for a runnable demo.

## Errors and forward compat

`VectorStoreError` and `DistanceMetric` are both `#[non_exhaustive]`
— new variants ship as minor bumps without breaking downstream
matches.

## Versioning

Part of the sphereQL workspace, currently `0.3.0`; API may
change before 1.0. See the workspace
[CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md).

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and
[empirical-findings.md](https://github.com/bkahan/sphereQL/blob/main/docs/empirical-findings.md).
