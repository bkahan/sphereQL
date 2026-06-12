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

Numbers from the 2026-04-10 run of
[`sphereql-examples/examples/benchmark.rs`](../sphereql-examples/examples/benchmark.rs),
recorded in full in [benchmark-analysis.md](benchmark-analysis.md). That
run predates the hybrid re-rank fix and the spatial-index query
acceleration (commit `e169f59`) and the UMAP projection overhaul — a
re-run on current code is pending.

| Method | k | Precision@k | nDCG@k | Mean latency |
|---|---|---|---|---|
| Brute-force ANN | 5 | 1.000 | 1.000 | 173 ms |
| SphereQL PCA | 1 | 1.000 | 1.000 | 1.9 ms |
| SphereQL PCA | 5 | 0.205 | 0.745 | 2.1 ms |
| SphereQL KPCA | 5 | 0.204 | 0.746 | 84 ms\* |
| Hybrid (r = k × 2) | 5 | 0.574 | 0.982 | 159 ms |

\* KPCA row is from an earlier run whose `benchmark_results.json` is no
longer in the repo (pre-UMAP-overhaul — re-benchmark pending). The
headline stands either way: **KPCA query latency is close to brute
force** (~84 ms vs ~150–175 ms), so KPCA buys nonlinear structure, not
query speed.

SphereQL PCA queries ran **~80–90× faster** than brute-force in that run
(93× at k=1, 82× at k=5) with perfect precision at k=1. Precision
degrades at higher k due to the lossy 384-d → 3-d projection (~2.8%
explained variance) — 20.5% precision@5 even though nDCG@5 stays at
0.745, so don't read the nDCG column alone.

The hybrid row above was measured under the old behavior (re-rank ANN
candidates by 3-d spherical distance), which at low EVR demoted correct
results. That has since been fixed: `VectorStoreBridge::hybrid_search`
now re-ranks by **original cosine similarity** in the full embedding
space (commit `e169f59`). Post-fix hybrid numbers have not been recorded
yet. See [benchmark-analysis.md](benchmark-analysis.md) for the detailed
analysis and [search-precision-roadmap.md](search-precision-roadmap.md)
for what has shipped since and what is still planned.

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

At n=775 (built-in corpus), a random search of budget 24 ran in ~3
seconds release mode — measured before `UmapSphere` joined the default
`SearchSpace` (the default now sweeps PCA + UMAP, and UMAP trials pay
the Adam optimizer per epoch sweep). Expect a budget-24 run to take
longer today; re-measure pending.

## Projection fit costs

PCA fits in milliseconds at n ≤ 10k — its cost is power iteration with
deflation for the top-3 eigenvectors, computed as Xᵀ(Xv) so each
iteration is O(N · d) without materializing the d × d covariance.
The category-weighted variant adds one O(N) weight pass.

**Kernel PCA is materially slower.** It builds an n × n Gram matrix
and runs eigendecomposition on it, both O(n²)–O(n³). On the 10k-point
benchmark above (384-d, 20 clusters), KPCA fitting takes
**~85 minutes** in release mode on a modern laptop CPU. Only fit KPCA
once and serialize the resulting projection — refitting per request
is not feasible. Auto-tuner runs that include KPCA in the search
space should expect a one-shot prefit cost of similar magnitude per
distinct kernel-parameter tuple. See `sphereql-embed/src/kernel_pca.rs`
for the implementation.

Laplacian eigenmap fitting is between PCA and KPCA — graph
construction is O(n · k) and eigendecomposition is O(n²) but on a
sparse Laplacian, so practical cost is closer to PCA than KPCA at
mid-size n.

UMAP-on-sphere fit splits into two phases: kNN-graph construction
(brute force below 2000 items; RP-forest ANN above, O(N · d · log N)
build) plus PCA warm-start, then the Adam optimizer at
O(N · k · epochs). Inside the tuner the first phase is cached per
`n_neighbors`, so a sweep over `n_epochs × category_weight × min_dist`
pays the graph cost once. The same RP-forest backs `GraphModularity`'s k-NN
edge construction at ≥ 2000 items, keeping composite-metric scoring
feasible at 100k–500k items.
