# Performance

## Index internals

The spatial index uses a two-tier partitioning scheme:

- **Shell partitioning** — radial shells for fast r-range filtering.
- **Sector partitioning** — angular sectors (θ × φ grid) for spatial
  locality.
- **Cosine proxy** — k-NN uses precomputed unit Cartesian vectors and
  `1 − dot(a, b)` instead of the full Vincenty formula, reducing
  per-item cost to 3 multiplications + 2 additions.

## Benchmark (10k points, 384 dims, 20 clusters, 200 queries)

Numbers from the 2026-06-12 run of
[`sphereql-examples/examples/benchmark.rs`](../sphereql-examples/examples/benchmark.rs),
committed verbatim at [`docs/benchmarks/2026-06-12.json`](benchmarks/2026-06-12.json)
and analyzed in full in [benchmark-analysis.md](benchmark-analysis.md).
This run is on current code (post hybrid re-rank fix, post spatial-index
query acceleration in commit `e169f59`, and post UMAP projection overhaul).
In this synthetic k-NN-ground-truth setup Precision@k == Recall@k, so only
Precision@k is shown below.

| Method | k | Precision@k | nDCG@k | Mean latency |
|---|---|---|---|---|
| Brute-force ANN | 5 | 1.000 | 1.000 | 14.3 ms |
| SphereQL PCA | 5 | 0.000 | 0.000 | 0.3 ms |
| SphereQL KPCA | 5 | 0.205 | 0.747 | 4.5 ms |
| SphereQL UMAP | 5 | 0.218 | 0.829 | 0.2 ms |
| Hybrid (r = k × 2) | 5 | 1.000 | 1.000 | 16.2 ms |
| PQ-rerank (r = k × 2) | 5 | 0.586 | 0.985 | 11.8 ms |

**SphereQL PCA collapses on this corpus.** At ~2.8% explained variance the
384-d → 3-d projection retains too little structure to separate the 20
uniform synthetic clusters, so PCA-only precision is ~0 at every k (it does
not even hold k=1 here). Queries still run at ~0.3 ms — the speed is real,
but on this dataset it buys nothing useful.

**UMAP is the fastest projection and edges KPCA on quality.** UMAP-on-sphere
queries at ~0.2 ms (the lowest latency of any method, ~20× faster than KPCA)
while beating KPCA on precision@5 (0.218 vs 0.205) and nDCG@5 (0.829 vs
0.747). Both nonlinear projections still degrade past k=1 because the 3-d
sphere cannot resolve close-but-not-closest neighbors — read precision, not
nDCG alone, when judging retrieval.

KPCA query latency (~4.5 ms) is now well below brute force (~14 ms) on this
run, so KPCA buys nonlinear structure *and* a query-speed win here; it is no
longer "close to brute force" the way the pre-acceleration runs reported.

**The Hybrid exact-rerank path recovers perfect precision.**
`VectorStoreBridge::hybrid_search` re-ranks ANN candidates by **original
cosine similarity** in the full embedding space (commit `e169f59`), giving
P@k = 1.000 at every k and recall multiplier (r = k×2/4/8) at ~12–16 ms/query
— essentially the brute-force result through the candidate-recall path.

**PQ-rerank is the middle ground.** Re-ranking ANN candidates by Product
Quantization asymmetric distance lands at P@5 ≈ 0.586 (nDCG@5 ≈ 0.985) at
r = k×2 for ~12 ms/query. Note the *inverse* candidate-count trend: widening
the pool to r = k×8 *lowers* precision (P@5 drops to 0.358), because PQ's
approximate distance misranks more distractors into the top-k as the
candidate set grows — more candidates is worse, not better, for PQ rerank.

See [benchmark-analysis.md](benchmark-analysis.md) for the full per-k tables
and [search-precision-roadmap.md](search-precision-roadmap.md) for what has
shipped and what is still planned.

## Auto-tuner costs

Per-trial cost in `auto_tune` is dominated by spatial-quality Monte
Carlo sampling, bridge graph construction, and category layer rebuild.
The projection itself is **fit once per distinct fit-affecting
hyperparameter tuple** and reused across trials: PCA and Kernel PCA
key per kind, Laplacian per `(k_neighbors, active_threshold)`, UMAP
per `(n_neighbors, n_epochs, category_weight, min_dist)` — and UMAP's kNN
graph + PCA warm-start are additionally cached per `n_neighbors`, so
epoch/weight sweeps only pay for the Adam optimizer
(`TuneReport::umap_graph_builds` reports how often the cache fired).

`SearchStrategy::Random` and `::Bayesian` accept `max_wall_secs` to
bound a run: the tuner stops *proposing* new trials once the cap is
exceeded, but a trial already in flight runs to completion — it is not
a hard timeout. At 100k+ items, prefer the wall-time cap over a large
trial budget.

At n=775 (built-in corpus), a budget-24 random search runs in **~9.9 s**
release mode (2026-06-12, three-way `[Pca, LaplacianEigenmap,
UmapSphere]` space). That is up from the ~3 s figure measured before
UMAP entered the space: UMAP trials pay the Adam optimizer per
`(n_epochs, category_weight, min_dist)` sweep, with the kNN graph cached
once per `n_neighbors`. The same budget-24 search on the 300-concept
stress corpus runs in **~4.1 s**.

## Projection fit costs

PCA fits in seconds at n = 10k (**~7.8 s** on the 2026-06-12 run) — its
cost is power iteration with deflation for the top-3 eigenvectors,
computed as Xᵀ(Xv) so each iteration is O(N · d) without materializing
the d × d covariance. The category-weighted variant adds one O(N) weight
pass.

**Kernel PCA is materially slower.** It builds an n × n Gram matrix
and runs eigendecomposition on it, both O(n²)–O(n³). On the 10k-point
benchmark above (384-d, 20 clusters), KPCA fitting takes
**~138 s (≈ 2.3 min)** in release mode on a modern laptop CPU — roughly
18× the ~7.8 s PCA fit, but far cheaper than the ~85-min figure earlier
drafts of this doc quoted. Still, only fit KPCA once and serialize the
resulting projection — refitting per request is not feasible. Auto-tuner
runs that include KPCA in the search space should expect a one-shot
prefit cost of similar magnitude per distinct kernel-parameter tuple. See
`sphereql-embed/src/kernel_pca.rs` for the implementation.

Laplacian eigenmap fitting is between PCA and KPCA — graph
construction is O(n · k) and eigendecomposition is O(n²) but on a
sparse Laplacian, so practical cost is closer to PCA than KPCA at
mid-size n.

UMAP-on-sphere fit splits into two phases: kNN-graph construction
(brute force below 2000 items; RP-forest ANN above, O(N · d · log N)
build) plus PCA warm-start, then the Adam optimizer at
O(N · k · epochs). On the 2026-06-12 run it fits the 10k-point corpus in
**~12.2 s** — between PCA and KPCA, and the projection that won on both
query latency (~0.2 ms) and precision in the retrieval benchmark above.
Building the PQ index over the same corpus took **~11.9 s**. Inside the tuner the first phase is cached per
`n_neighbors`, so a sweep over `n_epochs × category_weight × min_dist`
pays the graph cost once. The same RP-forest backs `GraphModularity`'s k-NN
edge construction at ≥ 2000 items, keeping composite-metric scoring
feasible at 100k–500k items.
