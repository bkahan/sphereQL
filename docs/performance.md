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
| SphereQL PCA | 1 | 1.000 | 1.000 | 1.7 ms |
| SphereQL PCA | 5 | 0.205 | 0.745 | 1.6 ms |
| SphereQL KPCA | 5 | 0.204 | 0.746 | 84 ms |
| Hybrid (r = k × 2) | 5 | 1.000 | 1.000 | 155 ms |

SphereQL PCA queries run **~90× faster** than brute-force with perfect
precision at k=1. Precision degrades at higher k due to the lossy
384-d → 3-d projection (~2.8% explained variance). The hybrid approach
recovers full precision via cosine re-ranking in the original space,
at near-brute-force latency.

Improving the speed/precision tradeoff at higher k is an active
development priority. For the full results see
[benchmark-analysis.md](benchmark-analysis.md) and
[search-precision-roadmap.md](search-precision-roadmap.md).

## Auto-tuner costs

Per-trial cost in `auto_tune` is dominated by spatial-quality Monte
Carlo sampling, bridge graph construction, and category layer rebuild.
The projection itself is **fit once per distinct fit-affecting
hyperparameter tuple** and reused across trials, so projection fitting
contributes only a one-time prefit cost per unique (`ProjectionKind`,
Laplacian params) combination.

At n=775 (built-in corpus), a random search of budget 24 runs in ~3
seconds release mode.
