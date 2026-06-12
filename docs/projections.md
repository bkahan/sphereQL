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

Targets a different failure mode: sparse, noise-heavy embeddings where the
signal lives in the co-activation structure of a few active axes rather
than in coordinate variance. PCA and kernel PCA get pulled toward whichever
noise directions happen to have the largest variance; a graph-Laplacian
spectral decomposition does not.

The pipeline:

1. For each embedding, extract its **active-axis set** — indices where
   `|v_i| > active_threshold`.
2. Build a k-NN graph whose edges are the **Jaccard similarity** between
   active-axis sets.
3. Form the normalized graph Laplacian `L = I − D⁻¹ᐟ² W D⁻¹ᐟ²`.
4. Keep the bottom 3 non-trivial eigenvectors as coordinates.
5. Out-of-sample points: **Nyström extension** blends the query's Jaccard
   weights against the training graph's eigenvectors, so a fitted
   projection behaves like any other at query time.

Hyperparameters live in `LaplacianConfig` (`k_neighbors`,
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

## Explained variance (EVR)

Every projection reports an `explained_variance_ratio()` in `[0, 1]`.
PCA returns the classical EVR; Kernel PCA returns its kernel-space
EVR; Laplacian returns a compatible connectivity ratio (mean |μ| of the
three retained eigenvalues); UMAP-on-sphere returns a trustworthiness-style
kNN recall (mean overlap between each point's neighborhood among the
fitted 3D positions and its original-space kNN set) — not a variance
ratio, but bounded in `[0, 1]` and rank-meaningful across corpora.
(Earlier versions used a median-distance proxy that saturated near 1.0
on the sphere; recall replaced it.) All of them feed the EVR-adaptive
thresholds downstream — bridge threshold, `RoutingConfig::low_evr_threshold`,
the high-EVR routing bypass in `default_nearest`, and confidence scoring
all consult this value.

**Typical values:** 2–5% EVR for transformer embeddings at 3 dimensions
under PCA; supervised UMAP reports much higher neighborhood-recall
scores on category-structured corpora.
This projection is inherently lossy; sphereQL compensates with **hybrid
search** (angular candidates in projected space → cosine re-ranking in
the original space) and, for low-EVR corpora,
`hierarchical_nearest` which routes through domain groups and inner
spheres instead of the outer sphere.

## Choosing a projection

The right choice is corpus-dependent. See
[empirical findings](empirical-findings.md) for measured scores across
both built-in corpora, and [auto-tuning](auto-tuning.md) for how the
tuner picks for you — including the PCA + UMAP
`SearchSpace::large_corpus()` space used above 10k items.
