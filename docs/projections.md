# Projections

sphereQL projects high-dimensional vectors (e.g. 384-d sentence-transformer
output) down to 3D spherical coordinates via one of four pipeline families —
the `ProjectionKind` enum: PCA, Kernel PCA, Laplacian eigenmap, and
UMAP-on-sphere. (A random-projection baseline also exists as a low-level
`Projection` impl; see below.)

## The core pipeline

1. **Normalize** — embeddings are L2-normalized to the unit hypersphere.
2. **Center** — subtract the corpus mean (for PCA / Kernel PCA).
3. **Reduce** — pick 3 coordinates per the chosen projection family
   (see below).
4. **Map** — the 3 components become Cartesian (x, y, z), which convert
   to spherical (r, θ, φ).

The **radial coordinate** is configurable via `RadialStrategy`:

- `Magnitude` (default) — r = pre-normalization L2 magnitude, encoding
  "confidence" or specificity. Degenerates to a constant when inputs are
  already L2-normalized; pick a projection-side variant in that case.
- `Fixed(value)` — constant radius; pure angular projection.
- `MagnitudeTransform(fn)` — custom transform of the pre-normalization
  magnitude (e.g. log-scaling).
- `ProjectionMagnitude` — r = ‖(x, y, z)‖, how much input variance landed
  in the projected 3-vector. Recommended starting point for normalized
  embeddings.
- `Certainty { scale }` — r = scale × the projection's per-point fidelity
  score in [0, 1].
- `Custom(fn)` — arbitrary per-point logic over the `RadialContext` signals.

## PCA

Linear PCA on the centered corpus. The 3 principal components become the
3D coordinates. Fast, deterministic, zero hyperparameters.

When built through `PipelineConfig` (`fit_projection_for_config`), the
fit is **category-weighted**: each sample carries `w = 1/√|category|`,
so a category of size m contributes √m covariance mass instead of m —
square-root softening of imbalance that keeps singleton-heavy corpora
(DBpedia-style) from collapsing to a large-category-only subspace.
`PcaProjection::fit` remains the unweighted fit for direct callers.

Strength: dense low-noise embeddings where variance tracks meaning.
Failure mode: sparse corpora where most per-item variance comes from
uninformative axes.

## Kernel PCA

PCA in the feature space induced by a Gaussian (RBF) kernel
`k(x, y) = exp(-‖x − y‖² / 2σ²)` with automatic σ selection via the median
heuristic. Captures non-linear manifold structure (curved clusters, rings,
spirals) that linear PCA crushes flat. See
[`sphereql-embed/src/kernel_pca.rs`](../sphereql-embed/src/kernel_pca.rs)
for mathematical details and references.

## Laplacian eigenmap

Targets a different failure mode: embeddings where the signal is which
items are close to which, not which directions vary most. PCA and
kernel PCA get pulled toward whichever noise directions happen to have
the largest variance; a graph-Laplacian spectral decomposition does
not.

The pipeline:

1. Link each item to its `k` most similar items and take the union of
   those links. Similarity is `LaplacianConfig.similarity`:
   - `Cosine` (default): cosine similarity between the embeddings.
     Works on any embedding and has no threshold to tune. Exact scan
     below 2,000 items, RP-forest ANN above.
   - `ActiveAxes`: the **Jaccard similarity** between the items'
     active-axis sets (`|v_i| > active_threshold`). For sparse
     embeddings whose signal is which axes fire. The threshold has to
     sit between the noise floor and the signal: on the 300-concept
     stress corpus (noise amplitude 0.2), 0.10 recovers the categories
     and 0.05 does not. On dense embeddings every axis is active for
     every item and the layout collapses.
2. Form the normalized graph Laplacian `L = I − D⁻¹ᐟ² W D⁻¹ᐟ²`.
3. Keep the bottom 3 non-trivial eigenvectors as coordinates. Fitted
   items sit at their own rows.
4. Out-of-sample points: **Nyström extension** over the query's `k`
   nearest fitted items, the same sparsification the fit used.

A graph that splits into two or three separate groups leaves fewer than
three group-level directions, so the remaining coordinate is variation
*inside* one group and that group spreads out on the sphere. With four
or more groups all three coordinates separate groups.

Hyperparameters live in `LaplacianConfig` (`k_neighbors`, `similarity`,
`active_threshold`). See
[`sphereql-embed/src/laplacian.rs`](../sphereql-embed/src/laplacian.rs)
for construction details.

## UMAP-on-sphere

UMAP optimized directly on S². A kNN graph over the normalized
embeddings supplies the attractive term (brute-force below 2000 items —
an O(n²) build — RP-forest ANN above, O(N log N); see
[`sphereql-embed/src/ann.rs`](../sphereql-embed/src/ann.rs)). Edges
carry fuzzy simplicial weights: per-point ρ/σ calibration (nearest edge
weight 1.0, `Σ exp(−(d−ρ)/σ) = log₂ k`) with fuzzy-union
symmetrization, and both the attraction and each edge's negative draws
scale with the weight. Each Adam step runs in the local tangent space
`T_x S²` with retraction back to the sphere by normalization. PCA
provides the warm start; an opt-in `warm_start_anchor` weight (default
0.0 — a bit-identical no-op) adds a weak pull toward the warm-start
positions so disconnected kNN components on sparse corpora keep their
global arrangement. An optional supervised term (`category_weight > 0`)
draws one same-category cohesion pair and one different-category
separation pair per point per epoch. `min_dist` (default 0.1) controls
how tightly neighbors may pack — the embedding kernel's (a, b) curve is
fit to it by deterministic least squares.

Hyperparameters live in `UmapConfig` (`n_neighbors`, `n_epochs`,
`category_weight`, `min_dist`, plus `warm_start_anchor` and `seed`);
the first four are first-class auto-tuner axes, and the tuner caches
the kNN graph + warm start per `n_neighbors` so epoch/weight/min_dist
sweeps only pay for optimization. UMAP is non-parametric: corpus items
return their exact Adam-optimized positions (a bit-pattern fast path,
certainty 1.0), and unseen embeddings are transformed by kNN-weighted
interpolation over the fitted positions — looked up through the same
RP-forest index the graph build retains at ≥ 2000 items — so
far-from-corpus queries degrade gracefully rather than extrapolating.
See [`sphereql-embed/src/umap.rs`](../sphereql-embed/src/umap.rs).

Strength: preserving local neighborhood structure at scale (O(N log N)
graph construction with ANN). Failure mode: global distances between
far-apart clusters are not meaningful, and transform quality depends on
the fitted corpus covering the query region.

## Random projection (low-level baseline)

The Johnson–Lindenstrauss baseline. Useful for ablations: if PCA doesn't
beat a random 3-axis basis, the corpus has no low-rank structure in 3
dimensions. **Not a pipeline family** — there is no
`ProjectionKind::Random`, so neither `PipelineConfig` nor the auto-tuner
can select it. `RandomProjection`
([`sphereql-embed/src/projection.rs`](../sphereql-embed/src/projection.rs))
implements the low-level `Projection` trait for direct use.

## Projection quality scores (per family)

Every projection family reports a single quality scalar in `[0, 1]`
through the shared `explained_variance_ratio()` accessor — but the four
families do **not** measure the same thing. The name is a historical
artifact (the accessor predates the non-PCA families); only PCA actually
returns an explained-variance ratio. The other three reuse the method
for API compatibility and to feed the same downstream thresholds, while
reporting a quantity appropriate to their own mathematics.

| Family | Quality scalar reported | What it measures | Rough interpretation |
|---|---|---|---|
| **PCA** | Classical explained-variance ratio (EVR) | Fraction of total corpus variance retained in the top-3 principal components | Higher = more linear variance survives the 3D projection. 2–5% is typical for transformer embeddings at 3 dimensions — low in absolute terms, but the surviving axes still rank meaning. |
| **Kernel PCA** | Kernel-space EVR | Fraction of variance in the (infinite-dimensional) Gaussian feature space captured by the top-3 kernel PCs | Same shape as PCA's EVR but computed in RBF feature space; comparable to PCA only loosely, not to the two below. |
| **Laplacian eigenmap** | kNN recall | Mean overlap between each point's neighborhood among the fitted 3D positions and its neighbors in the graph the fit was built from | Same definition as UMAP's, so the two compare directly. **Not** a variance fraction and **not** comparable to PCA EVR. The spectral summary (mean \|μ\| of the 3 kept eigenvalues) is still available as `connectivity_ratio()`; it describes how strongly the graph splits into groups, not how well the layout preserves it. |
| **UMAP-on-sphere** | Trustworthiness-style kNN recall | Mean overlap between each point's neighborhood among the fitted 3D positions and its original-space kNN set | Higher = more local neighborhoods survive the projection. 1.0 means every original neighborhood is preserved; random placements score near `k/n`. Rank-meaningful across corpora. |

All four are bounded in `[0, 1]` and feed the same EVR-adaptive
thresholds downstream — the bridge threshold,
`RoutingConfig::low_evr_threshold`, the high-EVR routing bypass in
`default_nearest`, and confidence scoring all consult this single value.
That shared plumbing is the reason the families expose one accessor; it
is **not** a claim that the scores are interchangeable. Do not compare a
PCA EVR against a UMAP or Laplacian recall as if
they were the same scale — a "low" number means something different in
each column above. (Earlier UMAP versions used a median-distance proxy
that saturated near 1.0 on the sphere; recall replaced it because it
discriminates and is honest across corpora.)

Whatever the family, the 3D projection is inherently lossy. sphereQL
compensates with **hybrid search** (angular candidates in projected
space → cosine re-ranking in the original space) and, when the reported
quality scalar falls below `RoutingConfig::low_evr_threshold`,
`hierarchical_nearest` routes through domain groups and inner spheres
instead of the outer sphere — applied uniformly across families on the
one scalar, regardless of which quantity it represents.

## Choosing a projection

The right choice is corpus-dependent. See
[empirical findings](empirical-findings.md) for measured scores across
both built-in corpora — read those scores through the per-family table in
[Projection quality scores](#projection-quality-scores-per-family),
since a number from one family is not comparable to one from another —
and [auto-tuning](auto-tuning.md) for how the tuner picks for you,
including the PCA + UMAP `SearchSpace::large_corpus()` space used above
10k items.
