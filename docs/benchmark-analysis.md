# Benchmark Analysis

## Setup

- **Harness:** [`sphereql-examples/examples/benchmark.rs`](../sphereql-examples/examples/benchmark.rs)
  (writes `benchmark_results.json` to the working directory)
- **Run date:** 2026-06-12. The raw run is committed verbatim at
  [`docs/benchmarks/2026-06-12.json`](benchmarks/2026-06-12.json) — every
  figure below is traceable to that file. This run is on **current code**
  (post hybrid re-rank fix, post spatial-index query acceleration in
  commit `e169f59`, and post UMAP projection overhaul).
- **Dataset:** 10,000 points, 384 dimensions, 20 synthetic clusters
- **Queries:** 200 random queries with pre-computed ground truth. Ground
  truth is exact k-NN, so **Precision@k == Recall@k** throughout.
- **Metrics:** Precision@k, Recall@k, nDCG@k, latency (mean and p99)
- **Methods tested:**
  - Vanilla ANN (brute-force cosine similarity — the exact baseline)
  - SphereQL PCA / KPCA / UMAP (search in the 3D projected space)
  - Hybrid (ANN recall + **exact cosine** re-ranking, recall multipliers
    2x/4x/8x — the current, fixed hybrid behavior)
  - PQ-rerank (ANN recall + Product Quantization asymmetric re-ranking,
    recall multipliers 2x/4x/8x)

Build/fit costs for this run: PCA fit **7,834 ms** (EVR 2.81%), Kernel PCA
fit **138,160 ms ≈ 2.3 min** (EVR 1.68%), UMAP-on-sphere fit **12,208 ms**
(kNN-recall 5.80%), PQ index build **11,918 ms**.

## Results

```
PCA explained variance ratio: 2.81%   (fit 7,834 ms)
KPCA explained variance ratio: 1.68%  (fit 138,160 ms)
UMAP kNN-recall: 5.80%                (fit 12,208 ms)
PQ index build time: 11,918 ms

| Method               |  k | Precision@k | Recall@k | nDCG@k |   Mean us |   p99 us |
|----------------------|----|-------------|----------|--------|-----------|----------|
| Vanilla ANN          |  1 |      1.0000 |   1.0000 | 1.0000 |   12676.6 |  18671.0 |
| Vanilla ANN          |  5 |      1.0000 |   1.0000 | 1.0000 |   14312.7 |  20769.0 |
| Vanilla ANN          | 10 |      1.0000 |   1.0000 | 1.0000 |   13432.9 |  32686.0 |
| Vanilla ANN          | 20 |      1.0000 |   1.0000 | 1.0000 |   13232.0 |  22677.0 |
| SphereQL PCA         |  1 |      0.0000 |   0.0000 | 0.0000 |     205.2 |    800.0 |
| SphereQL PCA         |  5 |      0.0000 |   0.0000 | 0.0000 |     348.5 |    985.0 |
| SphereQL PCA         | 10 |      0.0000 |   0.0000 | 0.0000 |     315.2 |   1261.0 |
| SphereQL PCA         | 20 |      0.0000 |   0.0000 | 0.0000 |     327.3 |   1240.0 |
| SphereQL KPCA        |  1 |      1.0000 |   1.0000 | 1.0000 |    4739.8 |   7051.0 |
| SphereQL KPCA        |  5 |      0.2050 |   0.2050 | 0.7467 |    4544.1 |   6939.0 |
| SphereQL KPCA        | 10 |      0.1095 |   0.1095 | 0.6506 |    4813.9 |   6975.0 |
| SphereQL KPCA        | 20 |      0.0648 |   0.0648 | 0.5647 |    5336.9 |   8119.0 |
| SphereQL UMAP        |  1 |      1.0000 |   1.0000 | 1.0000 |     222.9 |    897.0 |
| SphereQL UMAP        |  5 |      0.2180 |   0.2180 | 0.8290 |     209.6 |   1576.0 |
| SphereQL UMAP        | 10 |      0.1250 |   0.1250 | 0.7669 |     206.8 |    617.0 |
| SphereQL UMAP        | 20 |      0.0968 |   0.0968 | 0.7172 |     232.4 |    655.0 |
| Hybrid (r=k*2)       |  1 |      1.0000 |   1.0000 | 1.0000 |   13347.7 |  22794.0 |
| Hybrid (r=k*2)       |  5 |      1.0000 |   1.0000 | 1.0000 |   16198.8 |  34957.0 |
| Hybrid (r=k*2)       | 10 |      1.0000 |   1.0000 | 1.0000 |   13656.5 |  20434.0 |
| Hybrid (r=k*2)       | 20 |      1.0000 |   1.0000 | 1.0000 |   13488.5 |  20147.0 |
| Hybrid (r=k*4)       |  1 |      1.0000 |   1.0000 | 1.0000 |   12932.4 |  19873.0 |
| Hybrid (r=k*4)       |  5 |      1.0000 |   1.0000 | 1.0000 |   12520.6 |  26112.0 |
| Hybrid (r=k*4)       | 10 |      1.0000 |   1.0000 | 1.0000 |   12898.1 |  25249.0 |
| Hybrid (r=k*4)       | 20 |      1.0000 |   1.0000 | 1.0000 |   12730.2 |  18319.0 |
| Hybrid (r=k*8)       |  1 |      1.0000 |   1.0000 | 1.0000 |   11734.8 |  18251.0 |
| Hybrid (r=k*8)       |  5 |      1.0000 |   1.0000 | 1.0000 |   11634.0 |  17247.0 |
| Hybrid (r=k*8)       | 10 |      1.0000 |   1.0000 | 1.0000 |   11279.4 |  17272.0 |
| Hybrid (r=k*8)       | 20 |      1.0000 |   1.0000 | 1.0000 |   11252.1 |  19058.0 |
| PQ-rerank (r=k*2)    |  1 |      1.0000 |   1.0000 | 1.0000 |   12746.4 |  19011.0 |
| PQ-rerank (r=k*2)    |  5 |      0.5860 |   0.5860 | 0.9853 |   11772.3 |  16932.0 |
| PQ-rerank (r=k*2)    | 10 |      0.5770 |   0.5770 | 0.9749 |   12723.3 |  19916.0 |
| PQ-rerank (r=k*2)    | 20 |      0.5733 |   0.5733 | 0.9618 |   13351.8 |  23452.0 |
| PQ-rerank (r=k*4)    |  1 |      1.0000 |   1.0000 | 1.0000 |   13204.1 |  22870.0 |
| PQ-rerank (r=k*4)    |  5 |      0.4380 |   0.4380 | 0.9718 |   11106.9 |  15096.0 |
| PQ-rerank (r=k*4)    | 10 |      0.4030 |   0.4030 | 0.9536 |   13263.7 |  24344.0 |
| PQ-rerank (r=k*4)    | 20 |      0.3835 |   0.3835 | 0.9321 |   16251.9 |  41986.0 |
| PQ-rerank (r=k*8)    |  1 |      1.0000 |   1.0000 | 1.0000 |   15406.2 |  27242.0 |
| PQ-rerank (r=k*8)    |  5 |      0.3580 |   0.3580 | 0.9574 |   16487.0 |  26013.0 |
| PQ-rerank (r=k*8)    | 10 |      0.3005 |   0.3005 | 0.9310 |   17061.9 |  33148.0 |
| PQ-rerank (r=k*8)    | 20 |      0.2895 |   0.2895 | 0.9015 |   14635.3 |  21702.0 |
```

## Key Findings

### The EVR problem

Projecting 384 dimensions down to 3 retains only a few percent of the
original variance (PCA EVR 2.81%, KPCA 1.68%), and that loss is the
dominant factor limiting projected-space precision. The 3D sphere simply
cannot represent enough of the embedding structure to distinguish
neighbors beyond the single closest point.

- **PCA collapses entirely on this corpus.** At 2.81% EVR over 20 uniform
  synthetic clusters, PCA-only precision is **0.000 at every k** — it does
  not even hold k=1. The projection scrambles the cluster geometry badly
  enough that the true nearest neighbor rarely survives. This is harsher
  than earlier reads of PCA, which held k=1; on this uniform-cluster
  synthetic data PCA buys speed (~0.3 ms/query) and nothing else.
- **KPCA and UMAP hold k=1 but decay past it.** Both nonlinear projections
  keep perfect precision@1, then fall off: KPCA reaches 0.205 at k=5 and
  0.065 at k=20; UMAP reaches 0.218 at k=5 and 0.097 at k=20. The
  projection scrambles the ordering of points that are
  close-but-not-closest in the original space.

### UMAP is the fastest projection and edges KPCA on quality

UMAP-on-sphere queries at **~0.2 ms** — the lowest latency of any method in
the run and ~20× faster than KPCA (~4.5 ms) — while *beating* KPCA on
precision@5 (0.218 vs 0.205) and nDCG@5 (0.829 vs 0.747). For projected
manifold work it dominates KPCA on both axes here. KPCA still earns its
place as a structure tool, but it is no longer the precision leader among
projections on this corpus.

### Speed vs. accuracy tradeoff

Projected-space search is **3–65× faster** than vanilla ANN (~0.2–4.7 ms
vs ~13 ms): UMAP/PCA at ~0.2–0.35 ms are ~40–65× faster, while Kernel PCA
at ~4.7 ms is only ~3×. The speed advantage is real and comes from
operating in 3 dimensions instead of 384. But at these precision levels it is only useful
for approximate/exploratory queries (and for PCA on this corpus, not even
that) — not as an ANN replacement on its own.

### Hybrid exact-rerank recovers perfect precision

The current hybrid path (`VectorStoreBridge::hybrid_search`, commit
`e169f59`) recalls candidates through the ANN index and re-ranks them by
**original cosine similarity** in the full 384-d embedding space. On this
run it returns **P@k = 1.000 at every k and every recall multiplier**
(r = k×2/4/8) at ~12–16 ms/query — effectively the brute-force result
delivered through the candidate-recall path. This is the fix for the old
behavior (re-rank by 3-d spherical distance), which demoted correct results
and got *worse* with more candidates.

### PQ-rerank is the middle ground — and more candidates hurt it

Re-ranking the ANN candidate set by Product Quantization asymmetric
distance lands between projected-space search and exact rerank: **P@5 ≈
0.586, nDCG@5 ≈ 0.985 at r = k×2** for ~12 ms/query, with precision roughly
flat across k (0.577 at k=10, 0.573 at k=20).

The notable, reportable finding is the **inverse candidate-count trend**:
widening the recall pool makes PQ *worse*, not better. P@5 falls from 0.586
at r = k×2 to 0.438 at r = k×4 to 0.358 at r = k×8 (the same ordering holds
at k=10 and k=20). Because PQ ranks by an *approximate* distance, every
extra candidate is another chance for a distractor to be misranked into the
top-k; the larger the pool, the more such misranks survive into the final
result. This is the opposite of how exact rerank behaves (where more
candidates can only help) and is a property of the PQ approximation, not of
the recall stage.

### Build / fit time

On this run the projection/index fits dominate: PCA **7.8 s**, UMAP
**12.2 s**, PQ index **11.9 s**, and KPCA **138.2 s ≈ 2.3 min** (its n×n
Gram-matrix eigendecomposition is the outlier — far above the ~7.8 s PCA
fit, but nowhere near the ~85-min figure earlier drafts of the docs
quoted). KPCA is the only fit that needs to be amortized via serialization;
the rest are cheap enough to refit routinely at n = 10k.

## What SphereQL is good at

These results confirm that SphereQL's value is in **spatial analysis**,
not search replacement:

- **Visualization** -- 3D projection is purpose-built for this
- **Glob detection** -- coarse cluster structure survives low EVR
- **Concept paths** -- topological connectivity doesn't need metric precision
- **Local manifold analysis** -- directional variance is meaningful even
  in a lossy projection

Search precision improvements are tracked in
[search-precision-roadmap.md](search-precision-roadmap.md).

## Projection choice is corpus-dependent

Separate from the retrieval benchmark above (10k synthetic points, EVR
2.8%), the auto-tuner has been run head-to-head across **PCA vs Laplacian
eigenmap vs UMAP-on-sphere** on both the 775-concept built-in corpus and
the 300-concept low-SNR stress corpus (`build_stress_corpus`), via
[`examples/auto_tune.rs`](../sphereql-examples/examples/auto_tune.rs)
(budget 24, run 2026-06-12). Full per-projection, per-metric scores live
in [empirical-findings.md](empirical-findings.md); the headline:

- **UMAP-on-sphere wins both corpora**, under both `default_composite`
  and `connectivity_composite`.
- **Laplacian eigenmap collapses in this search** on the stress corpus
  (its noise-dominated affinity graph scores near zero) while staying
  roughly level with PCA on the built-in corpus. The collapse is a
  hyperparameter artifact, not a property of the corpus: Laplacian's
  stress-corpus score is bimodal in `laplacian_active_threshold`, and a
  *tuned* Laplacian (threshold `0.10`) recovers the signature at ~0.68.
  `auto_tune` only hits the fragile low-threshold region because adding
  `UmapSphere` shifts its random stream; `examples/meta_learn.rs` sweeps
  the same corpus without UMAP and reports Laplacian winning at ~0.68. See
  [empirical-findings.md](empirical-findings.md) for the full
  reconciliation.
- **PCA is robust on the stress corpus** but still loses to UMAP there.

This supersedes the earlier two-way result (PCA wins the built-in corpus,
Laplacian wins the stress corpus), which predated both UMAP and the
`BridgeDiversity`-weighted composites: PCA still edges Laplacian on the
built-in corpus, but with UMAP in the space the winner is UMAP on both.

The winner depends on the corpus *and* the projection set — the
motivation for the `MetaModel` layer, which predicts the winning
projection (and knob values) from a corpus's 10-feature profile before
running the full tuner. See
[`examples/meta_learn.rs`](../sphereql-examples/examples/meta_learn.rs)
for the end-to-end loop; note it tunes over `SearchSpace::default()`
(PCA + Laplacian) and builds its own training corpora, so its
per-projection scores are measured separately from the three-way sweep
above.
