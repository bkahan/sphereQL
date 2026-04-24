# Projections

sphereQL projects high-dimensional vectors (e.g. 384-d sentence-transformer
output) down to 3D spherical coordinates via one of four families.

## The core pipeline

1. **Normalize** — embeddings are L2-normalized to the unit hypersphere.
2. **Center** — subtract the corpus mean (for PCA / Kernel PCA).
3. **Reduce** — pick 3 coordinates per the chosen projection family
   (see below).
4. **Map** — the 3 components become Cartesian (x, y, z), which convert
   to spherical (r, θ, φ).

The **radial coordinate** is configurable via `RadialStrategy`:

- `Magnitude` (default) — r = pre-normalization L2 magnitude, encoding
  "confidence" or specificity.
- `Fixed(value)` — constant radius; pure angular projection.
- `MagnitudeTransform(fn)` — custom transform (e.g. log-scaling).

## PCA

Linear PCA on the centered corpus. The 3 principal components become the
3D coordinates. Fast, deterministic, zero hyperparameters.

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

## Random projection

The Johnson–Lindenstrauss baseline. Useful for ablations: if PCA doesn't
beat a random 3-axis basis, the corpus has no low-rank structure in 3
dimensions.

## Explained variance (EVR)

Every projection reports an `explained_variance_ratio()` in `[0, 1]`.
PCA returns the classical EVR; Kernel PCA returns its kernel-space
EVR; Laplacian returns a compatible connectivity ratio (mean of the
retained eigenvalues). All three feed the EVR-adaptive thresholds
downstream — bridge threshold, `RoutingConfig::low_evr_threshold`, and
confidence scoring all consult this value.

**Typical values:** 2–5% EVR for transformer embeddings at 3 dimensions.
This projection is inherently lossy; sphereQL compensates with **hybrid
search** (angular candidates in projected space → cosine re-ranking in
the original space) and, for low-EVR corpora,
`hierarchical_nearest` which routes through domain groups and inner
spheres instead of the outer sphere.

## Choosing a projection

The right choice is corpus-dependent. See
[empirical findings](empirical-findings.md) for measured scores across
both built-in corpora, and [auto-tuning](auto-tuning.md) for how the
tuner picks for you.
