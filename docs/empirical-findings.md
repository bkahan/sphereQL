# Empirical: when does each projection win?

The right projection is corpus-dependent. Two sanity checks, same
pipeline, same tuner, same metrics, two different corpora, opposite
winners:

| Corpus | Metric | PCA score | Laplacian score | Winner |
|---|---|---|---|---|
| Built-in 775-concept (31 academic domains, hand-crafted 128-d) | `default_composite` | best | lower | **PCA** |
| Built-in 775-concept | `connectivity_composite` | best | lower | **PCA** |
| Stress 300-concept (10 categories, 2-axis signatures, high noise) | `default_composite` | 0.9606 | 1.0000 | **Laplacian** |
| Stress 300-concept | `connectivity_composite` | 0.9265 | 0.9500 | **Laplacian** |

The stress-corpus scores above were measured under the pre-`BridgeDiversity`
composite weights (see *Metric details* below for the current weights).
Absolute numbers will shift slightly under the new composites; the
winner-flip conclusion still holds because both composites still reward
the connectivity structure Laplacian preserves on that corpus.

Dense, low-noise embeddings where variance tracks meaning: PCA wins.
Sparse, noise-heavy regimes where variance is dominated by noise and the
real signal is in the co-activation graph: Laplacian wins. This is the
whole motivation for the auto-tuner + meta-model layer — no single
projection is right, so the pipeline picks one per corpus.

## UMAP-on-sphere: comparison pending

The projection lineup is now four families (`Pca`, `KernelPca`,
`LaplacianEigenmap`, `UmapSphere`), and `SearchSpace::default()` sweeps
**PCA + UMAP** — see [projections.md](projections.md). The table above
predates `UmapSphere`: no UMAP-vs-PCA or UMAP-vs-Laplacian scores have
been recorded on either corpus yet. Two things to know before recording
them:

- UMAP's `explained_variance_ratio` now reports a
  trustworthiness-style kNN-recall score (mean neighborhood overlap),
  not the old fraction-below-median-random-distance proxy. Any UMAP
  quality numbers stored before the overhaul are **not
  score-comparable** with current ones.
- The table's PCA / Laplacian scores are unaffected by that change
  (their EVR semantics are unchanged), but they were measured on
  pre-overhaul builds of the metrics; treat absolute values as
  approximate per the weighting caveat above.

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

**Caveat:** both examples use `SearchSpace::default()`, which now
enumerates PCA + UMAP only. To reproduce the PCA-vs-Laplacian
head-to-head in the table, add `ProjectionKind::LaplacianEigenmap` to
`projection_kinds` in the example's search space first.

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
