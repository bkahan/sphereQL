# Empirical: when does each projection win?

The right projection is corpus-dependent. The numbers below are a
three-way head-to-head — PCA vs Laplacian eigenmap vs UMAP-on-sphere —
measured **2026-06-12** with the current `BridgeDiversity`-weighted
composites, a budget-24 random search per metric, on both corpora. Each
cell is the *best* score that projection reached across its trials:

| Corpus | Metric | PCA | Laplacian | UMAP | Winner |
|---|---|---|---|---|---|
| Built-in 775-concept (31 academic domains, hand-crafted 128-d) | `default_composite` | 0.1271 | 0.1159 | 0.6324 | **UMAP** |
| Built-in 775-concept | `connectivity_composite` | 0.0687 | 0.0611 | 0.6195 | **UMAP** |
| Stress 300-concept (10 categories, 2-axis signatures, high noise) | `default_composite` | 0.6419 | 0.0752 | 0.6513 | **UMAP** |
| Stress 300-concept | `connectivity_composite` | 0.5912 | 0.0193 | 0.6100 | **UMAP** |

With UMAP in the search space, **UMAP wins under both metrics on both
corpora.** This overturns the earlier two-way story (PCA winning the
built-in corpus, Laplacian winning the stress corpus) measured before
`UmapSphere` and `BridgeDiversity` landed: those scores predated the
UMAP overhaul and the current composite weights and are no longer
score-comparable. Two regime effects survive the change, now visible in
the PCA-vs-Laplacian columns:

- On the **built-in** corpus PCA still edges Laplacian (0.1271 vs 0.1159
  under `default_composite`), but both linear/spectral options are far
  behind UMAP — the 128-d hand-crafted signatures have enough usable
  variance for PCA, yet UMAP's neighborhood-preserving embedding
  separates the 31 domains better.
- On the **stress** corpus Laplacian *collapses* (best 0.0752 /
  0.0193): at 0.2 noise amplitude over 2-axis signatures its affinity
  graph is dominated by noise edges. PCA holds up here (0.6419 / 0.5912)
  because the 2 authored axes still carry most of the variance, and UMAP
  edges it out (0.6513 / 0.6100).

> **Laplacian is fragile, not uniformly bad, on the stress corpus.** The
> 0.0752 above is `auto_tune`'s *best* Laplacian trial under its three-way
> search (`[Pca, LaplacianEigenmap, UmapSphere]`, budget 24), but
> Laplacian's score here is bimodal in `laplacian_active_threshold`: at
> `0.10` the 0.2-amplitude noise is filtered enough that the 2 authored
> axes dominate and Laplacian *recovers* the signature (~0.68), while at
> `0.03`/`0.05` it builds the affinity graph on noise and collapses
> (~0.07). `examples/meta_learn.rs` sweeps the same corpus under the plain
> `SearchSpace::default()` (`[Pca, LaplacianEigenmap]`, no UMAP, budget 12)
> and its random draws land on the `0.10` region, so it reports a *tuned*
> Laplacian winning at ~0.68. Both runs are correct for their own search
> settings — adding `UmapSphere` shifts the random stream so `auto_tune`'s
> Laplacian trials only hit the fragile low-threshold region. The
> takeaway: a well-tuned Laplacian is competitive on this regime (~0.68 vs
> PCA ~0.64), but most of its hyperparameter space collapses, so a
> budget-limited search that doesn't sample `active_threshold = 0.10`
> misses the good configuration.

This per-corpus, per-metric variation is the whole motivation for the
auto-tuner + meta-model layer — no single projection is right, so the
pipeline picks one per corpus.

## UMAP's quality metric changed

The projection lineup is four families (`Pca`, `KernelPca`,
`LaplacianEigenmap`, `UmapSphere`); the head-to-head above adds
`UmapSphere` to `SearchSpace::default()`'s `[Pca, LaplacianEigenmap]`
explicitly — see [projections.md](projections.md). One caveat on the
UMAP column:

- UMAP's `explained_variance_ratio` now reports a
  trustworthiness-style kNN-recall score (mean neighborhood overlap),
  not the old fraction-below-median-random-distance proxy. Any UMAP
  quality numbers stored before the overhaul are **not
  score-comparable** with the 2026-06-12 numbers above.
- The PCA / Laplacian columns use unchanged EVR semantics, but were
  re-measured in the same 2026-06-12 run for an apples-to-apples
  comparison.

## Reproduce

```bash
# Built-in corpus (default)
cargo run -p sphereql-examples --example auto_tune --release

# Stress corpus
SPHEREQL_CORPUS=stress \
    cargo run -p sphereql-examples --example auto_tune --release

# Both corpora at once, with MetaModel verification
cargo run -p sphereql-examples --example meta_learn --release
```

**Note:** [`examples/auto_tune.rs`](../sphereql-examples/examples/auto_tune.rs)
overrides `SearchSpace::default()` (`[Pca, LaplacianEigenmap]`) to sweep
`[Pca, LaplacianEigenmap, UmapSphere]` explicitly, so each run produces the
three-way per-projection scores in the table above. `examples/meta_learn.rs`
still uses the plain `SearchSpace::default()`.

[`examples/meta_learn.rs`](../sphereql-examples/examples/meta_learn.rs) also
verifies that a `NearestNeighborMetaModel` fitted on both records can
predict the winning projection family from each corpus's feature
profile without re-running the tuner.

## Metric details

- `default_composite` — 30% `BridgeDiversity` + 25% `TerritorialHealth`
  + 25% `ClusterSilhouette` + 20% `GraphModularity`.
- `connectivity_composite` — 40% `GraphModularity` + 35%
  `BridgeDiversity` + 25% `TerritorialHealth`.

`BridgeDiversity` replaced `BridgeCoherence` in both composites because
`BridgeCoherence` converges to ~0.50 under the quantile-based
classification floor and so carries no signal across projection
configurations. `BridgeCoherence` is still available as a standalone
metric for callers who want to construct their own composite.

See the [`QualityMetric`
docs](https://docs.rs/sphereql-embed/latest/sphereql_embed/quality_metric/)
or [auto-tuning guide](auto-tuning.md) for what each metric measures.
