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

| Method | k | Precision@k | nDCG@k | Mean latency |
|---|---|---|---|---|
| Brute-force ANN | 5 | 1.000 | 1.000 | 154 ms |
| SphereQL PCA | 1 | 1.000 | 1.000 | 1.9 ms |
| SphereQL PCA | 5 | 0.205 | 0.745 | 2.1 ms |
| SphereQL KPCA | 5 | 0.204 | 0.746 | 84 ms |
| Hybrid (r = k × 2) | 5 | 0.574 | 0.982 | 159 ms |

SphereQL PCA queries run **~90× faster** than brute-force with perfect
precision at k=1. Precision degrades at higher k due to the lossy
384-d → 3-d projection (~2.8% explained variance).

The hybrid approach re-ranks by spherical distance after ANN retrieval,
which at low EVR can demote correct results. See
[benchmark-analysis.md](benchmark-analysis.md) for the detailed analysis
and [search-precision-roadmap.md](search-precision-roadmap.md) for the
planned fix (invert the hybrid: use angular projection as a pre-filter,
then score survivors by full cosine similarity).

Improving the speed/precision tradeoff at higher k is an active
development priority. For the full results see
[benchmark-analysis.md](benchmark-analysis.md) and
[search-precision-roadmap.md](search-precision-roadmap.md).

## Auto-tuner costs

Per-trial cost in `auto_tune` is dominated by spatial-quality Monte
Carlo sampling, bridge graph construction, and category layer rebuild.
The projection itself is **fit once per distinct fit-affecting
hyperparameter tuple** and reused across trials: PCA and Kernel PCA
key per kind, Laplacian per `(k_neighbors, active_threshold)`, UMAP
per `(n_neighbors, n_epochs, category_weight, min_dist)` — and UMAP's kNN graph
+ PCA warm-start are additionally cached per `n_neighbors`, so
epoch/weight sweeps only pay for the Adam optimizer
(`TuneReport::umap_graph_builds` reports how often the cache fired).

`SearchStrategy::Random` and `::Bayesian` accept `max_wall_secs` to
time-box a run; the tuner stops proposing trials once the cap is
exceeded. At 100k+ items, prefer the wall-time cap over a large trial
budget.

At n=775 (built-in corpus), a random search of budget 24 runs in ~3
seconds release mode.

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
