//! UMAP-on-sphere via Adam in the tangent bundle of S².
//!
//! Standard UMAP optimizes 2D embeddings in Euclidean space. Here every
//! fitted point lives on the unit 2-sphere, so each Adam step happens in
//! the local tangent space `T_x S² = { v : x·v = 0 }` and the iterate is
//! retracted back to the sphere via normalization. PCA provides the warm
//! start; the kNN graph supplies the attractive term and uniformly
//! sampled negatives supply the repulsive term. Optional per-point
//! categories add a third term that pulls same-category points together
//! and pushes different-category points apart.
//!
//! `project()` on a fitted training embedding returns its exact
//! optimized position (the Adam output, not an interpolation). For
//! genuinely unseen embeddings it uses a kNN-weighted slerp-ish
//! average over the fitted positions — UMAP itself is non-parametric, so
//! transforms hand new points to their nearest fitted neighbors and
//! interpolate on the sphere.

use std::collections::HashMap;
use std::sync::Arc;

use sphereql_core::SphericalPoint;

use crate::ann::{AnnConfig, AnnIndex};
use crate::projection::{
    Projection, ProjectionError, SplitMix64, dot, normalize_vec, project_xyz_to_spherical,
};
use crate::types::{Embedding, ProjectedPoint, RadialContext, RadialStrategy};

/// Knobs for [`UmapSphereProjection::fit`]. All defaults match the
/// canonical UMAP paper unless noted.
#[derive(Debug, Clone)]
pub struct UmapConfig {
    /// Neighbors per point in the kNN graph that supplies the
    /// attractive term. Higher = preserve global structure, lower =
    /// preserve local clusters. UMAP default 15.
    pub n_neighbors: usize,
    /// Optimizer iterations. ~200 is enough for n<=2000; scale up
    /// roughly logarithmically.
    pub n_epochs: usize,
    /// Adam base learning rate. Tangent-space gradients are bounded so
    /// 0.05 is safe even for tiny corpora.
    pub learning_rate: f64,
    /// Negative samples drawn per attractive edge per epoch.
    pub negative_sample_rate: usize,
    /// Weight on the supervised category term (0.0 = disabled). When
    /// active, every epoch samples for each point one same-category
    /// partner (cohesion, attractive) and one different-category
    /// partner (separation, repulsive) — stratified so the cohesion
    /// half fires regardless of how many categories the corpus has.
    /// Only meaningful when `categories` is supplied to `fit`.
    pub category_weight: f64,
    /// PRNG seed for kNN tie-breaking, negative sampling, and
    /// fallback random init when PCA warm-start is degenerate.
    pub seed: u64,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_epochs: 200,
            learning_rate: 0.05,
            negative_sample_rate: 5,
            category_weight: 0.0,
            seed: 0xA1B2_C3D4,
        }
    }
}

/// Precomputed kNN graph for UMAP. Cacheable across configs that
/// share the same `n_neighbors` but differ in `n_epochs` or
/// `category_weight`.
#[derive(Clone)]
pub struct UmapGraph {
    /// kNN adjacency list: `knn[i]` = indices of k nearest neighbors of item i.
    pub(crate) knn: Vec<Vec<usize>>,
    /// L2-normalized embeddings used for graph construction.
    /// Retained for the Adam optimizer's similarity lookups.
    pub(crate) normalized: Vec<Vec<f64>>,
    /// PCA warm-start positions on S² (unit vectors in ℝ³).
    pub(crate) warm_start: Vec<[f64; 3]>,
    /// Embedding dimensionality.
    pub(crate) dim: usize,
    /// Number of neighbors.
    pub(crate) k: usize,
    /// ANN index retained from kNN-graph construction (only built when
    /// `n >= ANN_BRUTE_FORCE_THRESHOLD`). Depends solely on `normalized`
    /// and the default [`AnnConfig`] — never on `n_neighbors` — so the
    /// tuner's per-`n_neighbors` graph cache stays sound. Carried
    /// through to the projection for transform-time neighbor queries.
    pub(crate) ann: Option<Arc<AnnIndex>>,
}

impl UmapGraph {
    /// Build the kNN graph and PCA warm-start from embeddings.
    ///
    /// This is the expensive part of UMAP fit — O(N·log N·d) for the
    /// ANN-backed graph + O(N·d) for PCA warm-start. The result is
    /// reusable across all UMAP configs that share `n_neighbors`.
    pub fn build(embeddings: &[Embedding], n_neighbors: usize) -> Result<Self, ProjectionError> {
        if embeddings.is_empty() {
            return Err(ProjectionError::EmptyCorpus);
        }
        let dim = embeddings[0].dimension();
        if dim < 3 {
            return Err(ProjectionError::DimensionTooLow {
                got: dim,
                required: 3,
            });
        }
        for (i, e) in embeddings.iter().enumerate() {
            if e.dimension() != dim {
                return Err(ProjectionError::InconsistentDimension {
                    index: i,
                    expected: dim,
                    got: e.dimension(),
                });
            }
        }
        let n = embeddings.len();
        if n < 4 {
            return Err(ProjectionError::TooFewEmbeddings {
                got: n,
                required: 4,
            });
        }

        let normalized: Vec<Vec<f64>> = embeddings.iter().map(|e| e.normalized()).collect();
        let k = n_neighbors.min(n - 1).max(1);
        let (knn, ann) = build_knn_graph(&normalized, k);
        let warm_start = pca_warm_start(embeddings, &normalized)?;

        Ok(Self {
            knn,
            normalized,
            warm_start,
            dim,
            k,
            ann,
        })
    }
}

/// UMAP-style projection that lives on S² and transforms new points by
/// kNN-weighted averaging over the fitted positions.
#[derive(Clone)]
pub struct UmapSphereProjection {
    /// Unit vectors in ℝ³ — one per fitted embedding.
    fitted_points: Vec<[f64; 3]>,
    /// L2-normalized copies of the original embeddings, kept for
    /// kNN lookup at transform time (UMAP is non-parametric).
    fitted_normalized: Vec<Vec<f64>>,
    /// Exact-match lookup: hash of the normalized embedding's bit
    /// pattern → fitted indices with that hash (verified by full vector
    /// comparison on hit, so hash collisions are safe). Projecting a
    /// training embedding returns its exact fitted position with
    /// certainty 1.0 — the optimizer computed that position, so it is
    /// known, not interpolated. Exact duplicates map to the first
    /// fitted index: their post-optimization positions may differ
    /// slightly, and first-index keeps the choice deterministic.
    exact_lookup: HashMap<u64, Vec<usize>>,
    /// ANN index over `fitted_normalized` for transform-time neighbor
    /// queries when the corpus is at or above
    /// `ANN_BRUTE_FORCE_THRESHOLD`; `None` means brute force.
    ann: Option<Arc<AnnIndex>>,
    dim: usize,
    radial: RadialStrategy,
    n_neighbors: usize,
    /// Post-fit quality in `[0, 1]`: trustworthiness-style kNN recall —
    /// mean overlap between each point's neighborhood among the fitted
    /// 3D positions and its original-space kNN set. 1.0 means the
    /// sphere preserves every original neighborhood.
    quality: f64,
}

impl UmapSphereProjection {
    /// Fit with default config and no categories.
    pub fn fit_default(embeddings: &[Embedding]) -> Result<Self, ProjectionError> {
        Self::fit(
            embeddings,
            None,
            RadialStrategy::default(),
            UmapConfig::default(),
        )
    }

    /// Optimize from a prebuilt kNN graph. This is the cheap part of UMAP
    /// fit — O(N·k·epochs) for the Adam optimizer. The graph is not rebuilt.
    ///
    /// Use this when the tuner has already built the graph via
    /// [`UmapGraph::build`] and is sweeping `n_epochs` / `category_weight`.
    pub fn fit_from_graph(
        graph: &UmapGraph,
        categories: Option<&[u32]>,
        radial: RadialStrategy,
        config: UmapConfig,
    ) -> Result<Self, ProjectionError> {
        let n = graph.normalized.len();

        if let Some(cats) = categories
            && cats.len() != n
        {
            return Err(ProjectionError::SliceLengthMismatch {
                expected: n,
                got: cats.len(),
            });
        }

        let mut points = graph.warm_start.clone();
        let mut rng = SplitMix64::new(config.seed);
        // NaN category_weight compares false, exactly like the old
        // partial_cmp().map(is_gt).unwrap_or(false) chain.
        let cat_active = config.category_weight > 0.0 && categories.is_some();

        // Per-category index buckets for stratified sampling in the
        // supervised term. Category ids may be sparse, so each id is
        // compacted to a dense bucket index up front; `bucket_of[i]`
        // is point i's bucket.
        let cat_buckets: Option<(Vec<Vec<usize>>, Vec<usize>)> = cat_active.then(|| {
            // cat_active is only true when categories.is_some().
            let cats = categories.unwrap();
            let mut id_to_bucket: HashMap<u32, usize> = HashMap::new();
            let mut buckets: Vec<Vec<usize>> = Vec::new();
            let mut bucket_of = Vec::with_capacity(n);
            for (i, &c) in cats.iter().enumerate() {
                let b = *id_to_bucket.entry(c).or_insert_with(|| {
                    buckets.push(Vec::new());
                    buckets.len() - 1
                });
                buckets[b].push(i);
                bucket_of.push(b);
            }
            (buckets, bucket_of)
        });

        // Adam state, three components per point.
        let mut m = vec![[0.0f64; 3]; n];
        let mut v = vec![[0.0f64; 3]; n];
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        for epoch in 1..=config.n_epochs {
            // Anneal the learning rate UMAP-style: linear decay to ~0.
            let lr = config.learning_rate * (1.0 - (epoch as f64 / config.n_epochs as f64));
            let mut grads = vec![[0.0f64; 3]; n];

            // Attractive + repulsive in one pass: each kNN edge
            // contributes its own attractive force AND draws
            // `negative_sample_rate` repulsive samples for the source
            // endpoint (UMAP's standard per-edge negative sampling, not
            // per-point).
            for (i, neighbors) in graph.knn.iter().enumerate() {
                for &j in neighbors {
                    let (gi, gj) = attractive_grad(&points[i], &points[j]);
                    add3(&mut grads[i], &gi);
                    add3(&mut grads[j], &gj);

                    for _ in 0..config.negative_sample_rate {
                        let nj = (rng.next_u64() as usize) % n;
                        if nj == i {
                            continue;
                        }
                        let (gi_r, gj_r) = repulsive_grad(&points[i], &points[nj]);
                        add3(&mut grads[i], &gi_r);
                        add3(&mut grads[nj], &gj_r);
                    }
                }
            }

            // Optional category term, stratified per point: one
            // same-category partner (cohesion) and one
            // different-category partner (separation) per epoch. A
            // uniform partner draw would be ~(C-1)/C repulsion at C
            // categories, starving the cohesion half exactly where
            // territorial scores need it.
            if let Some((buckets, bucket_of)) = &cat_buckets {
                let w = config.category_weight;
                for i in 0..n {
                    let bucket = &buckets[bucket_of[i]];
                    if bucket.len() > 1 {
                        // Uniform over the bucket minus i: draw from the
                        // first len-1 slots and remap a self-draw to the
                        // last slot.
                        let idx = (rng.next_u64() as usize) % (bucket.len() - 1);
                        let j = if bucket[idx] == i {
                            bucket[bucket.len() - 1]
                        } else {
                            bucket[idx]
                        };
                        let (gi, gj) = attractive_grad(&points[i], &points[j]);
                        add3_scaled(&mut grads[i], &gi, w);
                        add3_scaled(&mut grads[j], &gj, w);
                    }
                    // Rejection-sample the cross-category partner with
                    // a bounded retry count so a (near-)single-category
                    // corpus can't spin forever — on exhaustion, skip.
                    for _ in 0..MAX_CROSS_CATEGORY_DRAWS {
                        let j = (rng.next_u64() as usize) % n;
                        if bucket_of[j] != bucket_of[i] {
                            let (gi, gj) = repulsive_grad(&points[i], &points[j]);
                            add3_scaled(&mut grads[i], &gi, w);
                            add3_scaled(&mut grads[j], &gj, w);
                            break;
                        }
                    }
                }
            }

            // Adam step in tangent space, retract to S².
            for i in 0..n {
                let g_tan = project_to_tangent(&points[i], &grads[i]);
                for d in 0..3 {
                    m[i][d] = beta1 * m[i][d] + (1.0 - beta1) * g_tan[d];
                    v[i][d] = beta2 * v[i][d] + (1.0 - beta2) * g_tan[d] * g_tan[d];
                }
                let t = epoch as f64;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);
                let mut step = [0.0f64; 3];
                for d in 0..3 {
                    let m_hat = m[i][d] / bc1;
                    let v_hat = v[i][d] / bc2;
                    step[d] = lr * m_hat / (v_hat.sqrt() + eps);
                }
                // Retraction: x_new = (x - step) / |x - step|.
                // Sign: gradient descent ⇒ subtract.
                let mut next = [
                    points[i][0] - step[0],
                    points[i][1] - step[1],
                    points[i][2] - step[2],
                ];
                let mag = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                if mag > f64::EPSILON {
                    next[0] /= mag;
                    next[1] /= mag;
                    next[2] /= mag;
                    points[i] = next;
                }
            }
        }

        let quality = knn_recall_score(&points, &graph.knn);

        let mut exact_lookup: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, vec) in graph.normalized.iter().enumerate() {
            let bucket = exact_lookup.entry(hash_normalized(vec)).or_default();
            if !bucket.iter().any(|&j| graph.normalized[j] == *vec) {
                bucket.push(i);
            }
        }

        Ok(Self {
            fitted_points: points,
            fitted_normalized: graph.normalized.clone(),
            exact_lookup,
            ann: graph.ann.clone(),
            dim: graph.dim,
            radial,
            n_neighbors: graph.k,
            quality,
        })
    }

    /// Fit with custom config. `categories` is parallel to `embeddings`
    /// when supplied; pass `None` to disable the supervised term even
    /// if `config.category_weight > 0`.
    ///
    /// Equivalent to [`UmapGraph::build`] followed by
    /// [`Self::fit_from_graph`] — the tuner calls those two halves
    /// directly so it can reuse graphs across configs that share
    /// `n_neighbors`; this entry point serves every other caller.
    pub fn fit(
        embeddings: &[Embedding],
        categories: Option<&[u32]>,
        radial: RadialStrategy,
        config: UmapConfig,
    ) -> Result<Self, ProjectionError> {
        let graph = UmapGraph::build(embeddings, config.n_neighbors)?;
        Self::fit_from_graph(&graph, categories, radial, config)
    }

    /// Post-fit quality: trustworthiness-style kNN recall. For each
    /// point, its `k` nearest neighbors among the fitted 3D positions
    /// (`k` = the graph's `n_neighbors`) are intersected with its
    /// original-space kNN set; the score is the mean overlap fraction.
    /// Bounded `[0, 1]`, where 1.0 means every original neighborhood
    /// survives the projection.
    ///
    /// Intentionally exposed under the EVR name so the auto-tuner's
    /// `MetaModel` consumers can compare projection kinds on one
    /// scalar. Note the semantics changed: this used to be the fraction
    /// of kNN edges shorter than the median random pairwise distance, a
    /// bar that random spherical pairs (≈90° apart) made trivially
    /// clearable, so scores saturated high and barely discriminated.
    /// Recall is rank-meaningful across corpora and more honest when
    /// compared against other projection kinds.
    pub fn explained_variance_ratio(&self) -> f64 {
        self.quality
    }

    /// Locate the `n_neighbors` fitted points closest to `embedding`
    /// (cosine similarity in the original space) and return their
    /// indices with similarity weights.
    fn nearest_fitted(&self, normalized: &[f64]) -> Vec<(usize, f64)> {
        if let Some(ann) = &self.ann {
            return ann.query(normalized, self.n_neighbors);
        }
        let mut sims: Vec<(usize, f64)> = self
            .fitted_normalized
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(normalized, v)))
            .collect();
        sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        sims.truncate(self.n_neighbors);
        sims
    }

    fn exact_fitted(&self, normalized: &[f64]) -> Option<usize> {
        self.exact_lookup
            .get(&hash_normalized(normalized))?
            .iter()
            .copied()
            .find(|&i| self.fitted_normalized[i] == normalized)
    }

    fn project_xyz(&self, embedding: &Embedding) -> ([f64; 3], f64) {
        let normalized = embedding.normalized();
        if let Some(idx) = self.exact_fitted(&normalized) {
            return (self.fitted_points[idx], 1.0);
        }
        let neighbors = self.nearest_fitted(&normalized);

        // Softmax over similarities to get a stable weighted average.
        let max_sim = neighbors
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = neighbors
            .iter()
            .map(|(_, s)| ((s - max_sim) * 8.0).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if total > f64::EPSILON {
            for w in &mut weights {
                *w /= total;
            }
        } else {
            let n = weights.len() as f64;
            for w in &mut weights {
                *w = 1.0 / n;
            }
        }

        let mut acc = [0.0f64; 3];
        for ((idx, _), w) in neighbors.iter().zip(weights.iter()) {
            let p = self.fitted_points[*idx];
            acc[0] += w * p[0];
            acc[1] += w * p[1];
            acc[2] += w * p[2];
        }
        let mag = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
        let certainty = mag.clamp(0.0, 1.0);
        (acc, certainty)
    }
}

impl Projection for UmapSphereProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        // Caller contract: dimension must match the fitted projection.
        assert_eq!(
            embedding.dimension(),
            self.dim,
            "expected dimension {}, got {}",
            self.dim,
            embedding.dimension()
        );
        let (xyz, certainty) = self.project_xyz(embedding);
        let projection_magnitude = (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt();
        let intensity = embedding.magnitude();
        let r = self.radial.compute_rich(&RadialContext::full(
            intensity,
            projection_magnitude,
            certainty,
        ));
        project_xyz_to_spherical(xyz[0], xyz[1], xyz[2], r)
    }

    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint {
        // Caller contract: dimension must match the fitted projection.
        assert_eq!(
            embedding.dimension(),
            self.dim,
            "expected dimension {}, got {}",
            self.dim,
            embedding.dimension()
        );
        let (xyz, certainty) = self.project_xyz(embedding);
        let projection_magnitude = (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt();
        let intensity = embedding.magnitude();
        let r = self.radial.compute_rich(&RadialContext::full(
            intensity,
            projection_magnitude,
            certainty,
        ));
        let position = project_xyz_to_spherical(xyz[0], xyz[1], xyz[2], r);
        ProjectedPoint::new(position, certainty, intensity, projection_magnitude)
    }

    fn dimensionality(&self) -> usize {
        self.dim
    }
}

// ── Gradients ───────────────────────────────────────────────────────
//
// Loss decomposition mirrors the standard UMAP form, evaluated in ℝ³ on
// the embedded points and projected to the tangent at step time. The
// closed-form gradients below are the Euclidean gradients; the caller
// projects them to the tangent before stepping.

fn attractive_grad(xi: &[f64; 3], xj: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    // L_attr = log(1 + d²) where d = |xi - xj|.
    // ∂L/∂xi = 2(xi - xj) / (1 + d²); ∂L/∂xj = -∂L/∂xi.
    let dx = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
    let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    let coef = 2.0 / (1.0 + d2);
    let g = [coef * dx[0], coef * dx[1], coef * dx[2]];
    (g, [-g[0], -g[1], -g[2]])
}

fn repulsive_grad(xi: &[f64; 3], xj: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    // L_rep = -log(1 - 1/(1 + d²)) = log((1 + d²)/d²).
    // ∂L/∂xi = -2(xi - xj) / (d² (1 + d²)); pushes them apart.
    let dx = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
    let d2 = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).max(1e-6);
    let coef = -2.0 / (d2 * (1.0 + d2));
    let g = [coef * dx[0], coef * dx[1], coef * dx[2]];
    (g, [-g[0], -g[1], -g[2]])
}

fn project_to_tangent(x: &[f64; 3], g: &[f64; 3]) -> [f64; 3] {
    // T_x S² = { v : x·v = 0 }; project g by removing the radial part.
    let radial = x[0] * g[0] + x[1] * g[1] + x[2] * g[2];
    [
        g[0] - radial * x[0],
        g[1] - radial * x[1],
        g[2] - radial * x[2],
    ]
}

fn add3(a: &mut [f64; 3], b: &[f64; 3]) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
}

fn add3_scaled(a: &mut [f64; 3], b: &[f64; 3], s: f64) {
    a[0] += s * b[0];
    a[1] += s * b[1];
    a[2] += s * b[2];
}

// ── Helpers ────────────────────────────────────────────────────────

/// Retry bound for the cross-category rejection sampler in the
/// supervised term. With C ≥ 2 roughly balanced categories the miss
/// probability per draw is ≤ 1/2, so eight draws fail with probability
/// ≤ 1/256; only a near-single-category corpus exhausts this.
const MAX_CROSS_CATEGORY_DRAWS: usize = 8;

/// Corpus size at which the ANN index amortizes its build cost.
/// Below this, brute-force is faster and gives exact answers; above it,
/// the all-pairs O(N²) cost dominates.
const ANN_BRUTE_FORCE_THRESHOLD: usize = 2000;

/// FNV-1a over the bit patterns of the components. Exact bit equality
/// is the right key: training embeddings re-projected through the
/// pipeline pass through the same deterministic `Embedding::normalized`,
/// so they reproduce identical bits.
fn hash_normalized(v: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in v {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn build_knn_graph(normalized: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Option<Arc<AnnIndex>>) {
    let n = normalized.len();
    if n < ANN_BRUTE_FORCE_THRESHOLD {
        let knn = (0..n)
            .map(|i| {
                let mut sims: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, dot(&normalized[i], &normalized[j])))
                    .collect();
                sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                sims.into_iter().take(k).map(|(j, _)| j).collect()
            })
            .collect();
        return (knn, None);
    }

    // AnnConfig defaults (n_trees=8, max_leaf_size=40) give >95% recall
    // at N=500k for cosine kNN — the regime that drives this branch.
    let index = Arc::new(AnnIndex::build_normalized(
        normalized.to_vec(),
        &AnnConfig::default(),
    ));
    let knn = index.knn_graph(k);
    (knn, Some(index))
}

fn pca_warm_start(
    embeddings: &[Embedding],
    normalized: &[Vec<f64>],
) -> Result<Vec<[f64; 3]>, ProjectionError> {
    use crate::projection::PcaProjection;
    use sphereql_core::spherical_to_cartesian;

    let pca = PcaProjection::fit(embeddings, RadialStrategy::Fixed(1.0))?;
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(embeddings.len());
    for (i, e) in embeddings.iter().enumerate() {
        // `project_rich` exposes `projection_magnitude` — the raw 3D
        // magnitude before the radial-strategy override. With
        // `Fixed(1.0)` the SphericalPoint always has r=1, so checking
        // the *spherical* point's cartesian magnitude is meaningless
        // (it's always 1). The pre-radial magnitude is the real signal
        // for "input near corpus mean → degenerate placement."
        let pp = pca.project_rich(e);
        if pp.projection_magnitude > f64::EPSILON {
            let cart = spherical_to_cartesian(&pp.position);
            out.push([cart.x, cart.y, cart.z]);
            continue;
        }
        // Degenerate PCA position (input near corpus mean). Fall back
        // to the first three normalized coords as a direction —
        // stable, deterministic, and independent of any noise that
        // pushed the PCA coordinate to zero.
        let row = &normalized[i];
        let mut v = [row[0], row[1], row[2]];
        normalize_vec(&mut v);
        if v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 {
            v = [1.0, 0.0, 0.0];
        }
        out.push(v);
    }
    Ok(out)
}

/// Trustworthiness-style kNN recall: for each point, compute its kNN
/// set among the fitted 3D positions and intersect it with the
/// original-space kNN set from the graph; the score is the mean overlap
/// fraction. Random placements score near k/n; 1.0 means every original
/// neighborhood survives the projection.
///
/// The fitted points are unit vectors, so cosine order equals angular
/// order — the same kNN machinery used in the original space applies.
/// Above `ANN_BRUTE_FORCE_THRESHOLD` a fresh ANN index is built over
/// the 3D positions (deterministic via the default seed); this is
/// distinct from the retained high-dimensional index in `UmapGraph`.
fn knn_recall_score(points: &[[f64; 3]], knn: &[Vec<usize>]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 1.0;
    }

    let ann = (n >= ANN_BRUTE_FORCE_THRESHOLD).then(|| {
        let coords: Vec<Vec<f64>> = points.iter().map(|p| p.to_vec()).collect();
        AnnIndex::build_normalized(coords, &AnnConfig::default())
    });

    let mut total = 0.0;
    let mut counted = 0usize;
    for (i, original) in knn.iter().enumerate() {
        let k = original.len();
        if k == 0 {
            continue;
        }
        let spherical: Vec<usize> = match &ann {
            Some(index) => index
                .query_by_index(i, k)
                .into_iter()
                .map(|(j, _)| j)
                .collect(),
            None => {
                let mut sims: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, dot(&points[i], &points[j])))
                    .collect();
                sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                sims.into_iter().take(k).map(|(j, _)| j).collect()
            }
        };
        let hits = spherical.iter().filter(|j| original.contains(j)).count();
        total += hits as f64 / k as f64;
        counted += 1;
    }
    if counted == 0 {
        1.0
    } else {
        total / counted as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sphereql_core::angular_distance;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    fn cluster_corpus() -> Vec<Embedding> {
        // Two clear clusters in 6D so neighbor preservation is testable.
        let mut out = Vec::new();
        for i in 0..8 {
            let t = i as f64 * 0.01;
            out.push(emb(&[1.0 + t, 0.5 + t, 0.0, 0.0, 0.0, 0.0]));
        }
        for i in 0..8 {
            let t = i as f64 * 0.01;
            out.push(emb(&[0.0, 0.0, 0.0, 1.0 + t, 0.5 + t, 0.0]));
        }
        out
    }

    #[test]
    fn fit_default_runs_and_produces_valid_points() {
        let corpus = cluster_corpus();
        let proj = UmapSphereProjection::fit_default(&corpus).unwrap();
        for e in &corpus {
            let sp = proj.project(e);
            assert!(sp.r >= 0.0);
            assert!(sp.theta.is_finite());
            assert!(sp.phi.is_finite());
        }
    }

    #[test]
    fn quality_score_in_unit_interval() {
        let corpus = cluster_corpus();
        let proj = UmapSphereProjection::fit_default(&corpus).unwrap();
        let q = proj.explained_variance_ratio();
        assert!((0.0..=1.0).contains(&q), "got {q}");
    }

    #[test]
    fn well_separated_clusters_score_high_recall() {
        let corpus = cluster_corpus();
        let proj = UmapSphereProjection::fit(
            &corpus,
            None,
            RadialStrategy::Fixed(1.0),
            UmapConfig {
                n_neighbors: 5,
                ..UmapConfig::default()
            },
        )
        .unwrap();
        let q = proj.explained_variance_ratio();
        assert!(
            q > 0.5,
            "expected high recall for separated clusters, got {q}"
        );
    }

    #[test]
    fn shuffled_positions_score_lower_recall() {
        let corpus = cluster_corpus();
        let config = UmapConfig {
            n_neighbors: 5,
            ..UmapConfig::default()
        };
        let graph = UmapGraph::build(&corpus, config.n_neighbors).unwrap();
        let proj =
            UmapSphereProjection::fit_from_graph(&graph, None, RadialStrategy::Fixed(1.0), config)
                .unwrap();
        let fitted = proj.explained_variance_ratio();

        // Permuting the fitted positions breaks every neighborhood the
        // optimizer built, so the same scorer must rank them below the
        // real layout.
        let mut shuffled = proj.fitted_points.clone();
        let mut rng = SplitMix64::new(0xD15C);
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        let broken = knn_recall_score(&shuffled, &graph.knn);
        assert!(broken < fitted, "shuffled={broken}, fitted={fitted}");
    }

    #[test]
    fn empty_corpus_errors() {
        assert!(matches!(
            UmapSphereProjection::fit_default(&[]),
            Err(ProjectionError::EmptyCorpus)
        ));
    }

    #[test]
    fn dimension_too_low_errors() {
        let bad = vec![emb(&[1.0, 2.0]); 8];
        assert!(matches!(
            UmapSphereProjection::fit_default(&bad),
            Err(ProjectionError::DimensionTooLow { .. })
        ));
    }

    #[test]
    fn too_few_embeddings_errors() {
        let small = vec![emb(&[1.0, 2.0, 3.0, 4.0]); 3];
        assert!(matches!(
            UmapSphereProjection::fit_default(&small),
            Err(ProjectionError::TooFewEmbeddings {
                got: 3,
                required: 4
            })
        ));
    }

    #[test]
    fn ann_backed_knn_routes_to_correct_cluster() {
        // The 2000-item threshold in build_knn_graph routes this corpus
        // through brute force; here we directly exercise the ANN module
        // to confirm its k-NN graph respects cluster structure.
        //
        // cluster_corpus() builds two orthogonal clusters of 8 items.
        // The leaf size must be >= cluster size so every query routes
        // into a leaf containing all of its true neighbors — at this
        // tiny N, smaller leaves cause sub-leaf fragmentation that
        // would surface as recall holes only at this scale.
        use crate::ann::{AnnConfig, AnnIndex};

        let corpus = cluster_corpus();
        let normalized: Vec<Vec<f64>> = corpus.iter().map(|e| e.normalized()).collect();

        let config = AnnConfig {
            n_trees: 8,
            max_leaf_size: 8,
            seed: 42,
        };
        let index = AnnIndex::build_normalized(normalized.clone(), &config);
        let ann: Vec<Vec<usize>> = index.knn_graph(5);

        for (i, neighbors) in ann.iter().enumerate() {
            let own_cluster = if i < 8 { 0..8 } else { 8..16 };
            for &n in neighbors {
                assert!(
                    own_cluster.contains(&n),
                    "item {i} got neighbor {n} from the wrong cluster"
                );
            }
        }
    }

    #[test]
    fn category_term_pulls_same_class_together() {
        let corpus = cluster_corpus();
        let cats: Vec<u32> = (0..corpus.len())
            .map(|i| if i < 8 { 0 } else { 1 })
            .collect();

        let unsupervised = UmapSphereProjection::fit(
            &corpus,
            None,
            RadialStrategy::Fixed(1.0),
            UmapConfig {
                n_epochs: 100,
                category_weight: 0.0,
                ..UmapConfig::default()
            },
        )
        .unwrap();

        let supervised = UmapSphereProjection::fit(
            &corpus,
            Some(&cats),
            RadialStrategy::Fixed(1.0),
            UmapConfig {
                n_epochs: 100,
                category_weight: 2.0,
                ..UmapConfig::default()
            },
        )
        .unwrap();

        // Mean within-class distance, supervised vs unsupervised.
        let within_unsup = mean_within_class(&unsupervised.fitted_points, &cats);
        let within_sup = mean_within_class(&supervised.fitted_points, &cats);
        assert!(
            within_sup <= within_unsup + 1e-6,
            "supervised within-class={within_sup}, unsupervised={within_unsup}"
        );
    }

    #[test]
    fn category_term_tightens_classes_at_many_categories() {
        // 8 categories x 4 points, each category in its own basis
        // direction of an 8D space. At C=8 a uniform partner draw is
        // same-category only ~3/31 of the time, so the old sampling
        // was almost pure repulsion here — the cohesion half of the
        // term never fired. Stratified sampling must beat the
        // unsupervised baseline on within-class spread.
        let mut corpus = Vec::new();
        let mut cats: Vec<u32> = Vec::new();
        for c in 0..8u32 {
            for i in 0..4 {
                let mut v = vec![0.0; 8];
                v[c as usize] = 1.0 + i as f64 * 0.05;
                v[(c as usize + 1) % 8] = 0.1 + i as f64 * 0.02;
                corpus.push(emb(&v));
                cats.push(c);
            }
        }

        let config = |category_weight: f64| UmapConfig {
            n_neighbors: 3,
            n_epochs: 100,
            category_weight,
            ..UmapConfig::default()
        };

        let unsupervised =
            UmapSphereProjection::fit(&corpus, None, RadialStrategy::Fixed(1.0), config(0.0))
                .unwrap();
        let supervised = UmapSphereProjection::fit(
            &corpus,
            Some(&cats),
            RadialStrategy::Fixed(1.0),
            config(2.0),
        )
        .unwrap();

        let within_unsup = mean_within_class(&unsupervised.fitted_points, &cats);
        let within_sup = mean_within_class(&supervised.fitted_points, &cats);
        assert!(
            within_sup < within_unsup,
            "supervised within-class={within_sup}, unsupervised={within_unsup}"
        );
    }

    fn mean_within_class(points: &[[f64; 3]], cats: &[u32]) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if cats[i] == cats[j] {
                    let pi = SphericalPoint::new_unchecked(
                        1.0,
                        points[i][1]
                            .atan2(points[i][0])
                            .rem_euclid(std::f64::consts::TAU),
                        points[i][2].clamp(-1.0, 1.0).acos(),
                    );
                    let pj = SphericalPoint::new_unchecked(
                        1.0,
                        points[j][1]
                            .atan2(points[j][0])
                            .rem_euclid(std::f64::consts::TAU),
                        points[j][2].clamp(-1.0, 1.0).acos(),
                    );
                    total += angular_distance(&pi, &pj);
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    #[test]
    fn dimensionality_reports_input_dim() {
        let corpus = cluster_corpus();
        let proj = UmapSphereProjection::fit_default(&corpus).unwrap();
        assert_eq!(proj.dimensionality(), 6);
    }

    #[test]
    fn fit_from_graph_matches_full_fit() {
        let corpus = cluster_corpus();
        let config = UmapConfig {
            n_epochs: 50,
            category_weight: 0.0,
            seed: 42,
            ..UmapConfig::default()
        };

        let full =
            UmapSphereProjection::fit(&corpus, None, RadialStrategy::Fixed(1.0), config.clone())
                .unwrap();

        let graph = UmapGraph::build(&corpus, config.n_neighbors).unwrap();
        let split =
            UmapSphereProjection::fit_from_graph(&graph, None, RadialStrategy::Fixed(1.0), config)
                .unwrap();

        assert!(
            (full.explained_variance_ratio() - split.explained_variance_ratio()).abs() < 1e-6,
            "full={}, split={}",
            full.explained_variance_ratio(),
            split.explained_variance_ratio()
        );
    }

    #[test]
    fn graph_reusable_across_configs() {
        let corpus = cluster_corpus();
        let graph = UmapGraph::build(&corpus, 5).unwrap();

        let config1 = UmapConfig {
            n_epochs: 30,
            category_weight: 0.0,
            seed: 1,
            ..UmapConfig::default()
        };
        let config2 = UmapConfig {
            n_epochs: 60,
            category_weight: 1.0,
            seed: 2,
            ..UmapConfig::default()
        };

        let p1 =
            UmapSphereProjection::fit_from_graph(&graph, None, RadialStrategy::Fixed(1.0), config1)
                .unwrap();
        let p2 =
            UmapSphereProjection::fit_from_graph(&graph, None, RadialStrategy::Fixed(1.0), config2)
                .unwrap();

        assert!((0.0..=1.0).contains(&p1.explained_variance_ratio()));
        assert!((0.0..=1.0).contains(&p2.explained_variance_ratio()));
    }

    fn assert_projects_to_fitted(proj: &UmapSphereProjection, e: &Embedding, idx: usize) {
        use sphereql_core::spherical_to_cartesian;

        let pp = proj.project_rich(e);
        assert_eq!(pp.certainty, 1.0, "exact match must report certainty 1.0");
        let cart = spherical_to_cartesian(&pp.position);
        let expected = proj.fitted_points[idx];
        assert!((cart.x - expected[0]).abs() < 1e-12);
        assert!((cart.y - expected[1]).abs() < 1e-12);
        assert!((cart.z - expected[2]).abs() < 1e-12);
    }

    #[test]
    fn projecting_training_embedding_returns_exact_fitted_position() {
        let corpus = cluster_corpus();
        let proj = UmapSphereProjection::fit(
            &corpus,
            None,
            RadialStrategy::Fixed(1.0),
            UmapConfig::default(),
        )
        .unwrap();

        // Default n_neighbors=15 on a 16-point corpus means the old
        // interpolation path averaged nearly every fitted point, so a
        // smeared result would be visibly off the per-point positions
        // checked here.
        for (i, e) in corpus.iter().enumerate() {
            assert_projects_to_fitted(&proj, e, i);
        }
    }

    #[test]
    fn duplicate_training_embedding_maps_to_first_fitted_index() {
        let mut corpus = cluster_corpus();
        corpus.push(corpus[0].clone());
        let proj = UmapSphereProjection::fit(
            &corpus,
            None,
            RadialStrategy::Fixed(1.0),
            UmapConfig::default(),
        )
        .unwrap();

        assert_projects_to_fitted(&proj, &corpus[0], 0);
        assert_projects_to_fitted(&proj, &corpus[16], 0);

        let a = proj.project_rich(&corpus[0]);
        let b = proj.project_rich(&corpus[16]);
        assert_eq!(a.position.theta, b.position.theta);
        assert_eq!(a.position.phi, b.position.phi);
        assert_eq!(a.position.r, b.position.r);
    }

    #[test]
    fn unseen_embedding_interpolates_on_sphere() {
        let corpus = cluster_corpus();
        let proj = UmapSphereProjection::fit(
            &corpus,
            None,
            RadialStrategy::Fixed(1.0),
            UmapConfig::default(),
        )
        .unwrap();

        let unseen = emb(&[1.0, 0.55, 0.02, 0.0, 0.0, 0.0]);
        let pp = proj.project_rich(&unseen);
        assert!(pp.certainty > 0.0 && pp.certainty < 1.0);
        assert!(pp.position.theta.is_finite());
        assert!(pp.position.phi.is_finite());
        assert!((pp.position.r - 1.0).abs() < 1e-12);
        assert!(pp.projection_magnitude > 0.0 && pp.projection_magnitude <= 1.0 + 1e-12);
    }

    #[test]
    fn ann_backed_transform_above_threshold() {
        let mut rng = SplitMix64::new(0x5EED);
        let mut random_emb = |dim: usize| {
            let vals: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
            emb(&vals)
        };
        let corpus: Vec<Embedding> = (0..ANN_BRUTE_FORCE_THRESHOLD)
            .map(|_| random_emb(8))
            .collect();
        let config = UmapConfig {
            n_neighbors: 5,
            n_epochs: 2,
            negative_sample_rate: 1,
            ..UmapConfig::default()
        };
        let proj =
            UmapSphereProjection::fit(&corpus, None, RadialStrategy::Fixed(1.0), config).unwrap();
        assert!(proj.ann.is_some(), "expected ANN index above threshold");

        assert_projects_to_fitted(&proj, &corpus[1234], 1234);

        let unseen = random_emb(8);
        let pp = proj.project_rich(&unseen);
        assert!(pp.certainty > 0.0 && pp.certainty <= 1.0);
        assert!(pp.position.theta.is_finite());
        assert!(pp.position.phi.is_finite());
        assert!(pp.position.r.is_finite());
    }
}
