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

## Hybrid search

The crate is most useful when paired with `sphereql-embed`. The
typical flow:

1. Sync raw embeddings into a backing store via `VectorStore::upsert`.
2. Fit a sphereQL projection on the corpus.
3. Spherical-prune candidates with the spatial index.
4. Re-rank survivors using full-dimensional cosine similarity from
   the backing store.

This yields near-brute-force precision at near-spherical-index
latency. See `examples/full-e2e/` in the repo for a runnable demo.

## Errors and forward compat

`VectorStoreError` and `DistanceMetric` are both `#[non_exhaustive]`
— new variants ship as minor bumps without breaking downstream
matches.

## Documentation

See the workspace
[architecture.md](https://github.com/bkahan/sphereQL/blob/main/docs/architecture.md)
and
[empirical-findings.md](https://github.com/bkahan/sphereQL/blob/main/docs/empirical-findings.md).
