# Prompt: Build a Retrieval Benchmark Comparing Vanilla Vector Search vs SphereQL Hybrid

## Context

SphereQL (`sphereql-vectordb` crate) has a `VectorStoreBridge` that
supports `hybrid_search`: it first queries a vector store for a broad
`recall_k` set via ANN, then re-ranks through SphereQL's angular distance
in projected spherical space, returning `final_k` results.

The claim is that this hybrid approach improves precision (better top-k
quality) while the SphereQL projection adds negligible latency overhead.
We need hard numbers to back this up.

The goal is a self-contained benchmark binary that:
1. Uses a real-ish dataset (not toy synthetic data).
2. Compares vanilla ANN search vs. SphereQL hybrid search.
3. Measures precision@k, recall@k, nDCG@k, and latency.
4. Outputs results as a Markdown table and a JSON file.

## Deliverables

### 1. Create `sphereql-vectordb/benches/hybrid_benchmark.rs`

This is NOT a Criterion benchmark (those are micro-benchmarks). This is a
`[[bin]]` or a standalone Rust file in `examples/` that runs the full
retrieval evaluation. Put it at `sphereql/examples/benchmark.rs` so it
can use the umbrella crate with all features.

### 2. Dataset: Synthetic Clustered Embeddings

Generate a reproducible synthetic dataset that mimics real embedding
distributions. Do NOT use external files or downloads — everything must
be self-contained and deterministic.

```
Parameters:
  - num_clusters: 20
  - points_per_cluster: 500
  - total: 10,000 points
  - dimension: 384 (matches all-MiniLM-L6-v2)
  - num_queries: 200
```

Generation procedure:
1. For each cluster, sample a random centroid on the unit hypersphere
   (normalize a random Gaussian vector).
2. For each point in the cluster, add Gaussian noise (σ=0.15) to the
   centroid and re-normalize. This gives tight but overlapping clusters.
3. Assign category labels: `"cluster-{i}"` for cluster i.
4. For queries, pick 200 random points from the dataset. The ground truth
   for each query is the set of all points from the same cluster, ranked
   by true cosine similarity to the query.

Use a fixed seed (42) via the SplitMix64 PRNG already in the codebase
(or just use a simple LCG — no external dependency needed).

### 3. Evaluation Protocol

For each query, run two searches:

**Baseline: Vanilla ANN (InMemoryStore)**
- `store.search(query_vec, k)` — returns top-k by the store's internal
  cosine similarity.

**SphereQL Hybrid**
- `bridge.hybrid_search(query_vec, final_k=k, recall_k=k*4)` — the
  bridge queries the store for 4x candidates, then re-ranks through the
  PCA projection.

Evaluate at k = 1, 5, 10, 20.

### 4. Metrics

For each method and each k, compute:

- **Precision@k**: fraction of returned items that are in the ground-truth
  top-k (by true cosine similarity across the FULL dataset, not just the
  store's ANN approximation).
- **Recall@k**: fraction of ground-truth top-k that appear in the returned
  results.
- **nDCG@k**: normalized discounted cumulative gain using true cosine
  similarity as relevance score.
- **Mean latency (μs)**: wall-clock time per query, averaged over all 200
  queries. Use `std::time::Instant` for timing.
- **p99 latency (μs)**: 99th percentile latency.

Also compute and report:
- PCA explained variance ratio (how much information the projection
  preserves).
- Total index build time (fitting PCA + inserting all 10k points).

### 5. Output Format

Print a Markdown table to stdout:

```
## SphereQL Hybrid Search Benchmark
Dataset: 10,000 points, 384-d, 20 clusters
Queries: 200
PCA explained variance: XX.X%
Index build time: XXX ms

| Method          | k  | Precision@k | Recall@k | nDCG@k | Mean μs | p99 μs |
|-----------------|----|-------------|----------|--------|---------|--------|
| Vanilla ANN     |  1 | ...         | ...      | ...    | ...     | ...    |
| Vanilla ANN     |  5 | ...         | ...      | ...    | ...     | ...    |
| ...             |    |             |          |        |         |        |
| SphereQL Hybrid |  1 | ...         | ...      | ...    | ...     | ...    |
| SphereQL Hybrid |  5 | ...         | ...      | ...    | ...     | ...    |
| ...             |    |             |          |        |         |        |
```

Also write the full results to `benchmark_results.json` with the same
data in structured form, plus per-query breakdowns.

### 6. Additional comparisons to include

Add two more configurations to make the benchmark more informative:

**SphereQL-only (no ANN fallback)**
- `bridge.query(SphereQLQuery::Nearest { k }, &query_vec)` — searches
  only in projected spherical space, no vector store involved.
- This isolates the quality of the 3D projection alone.

**Varying recall_k for hybrid**
- Run hybrid search with recall_k = k*2, k*4, k*8.
- This shows the precision/latency tradeoff curve.

### 7. Warm-up

Run 50 warm-up queries before timing begins to eliminate JIT/cache effects.
Do not include warm-up queries in the metrics.

## Constraints

- No external dependencies beyond what's already in the workspace.
- No file I/O for the dataset — everything generated in-memory.
- Must compile and run with: `cargo run --example benchmark -p sphereql --features full --release`
- The benchmark must complete in under 60 seconds on a modern laptop in
  release mode.
- Do NOT use `criterion` — this is an end-to-end retrieval benchmark, not
  a micro-benchmark. Simple `Instant::now()` timing is correct here.
- Run `cargo clippy --workspace --all-features --all-targets` before done.
