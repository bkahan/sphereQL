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
//! `project()` for unseen embeddings uses a kNN-weighted slerp-ish
//! average over the fitted positions — UMAP itself is non-parametric, so
//! transforms hand new points to their nearest fitted neighbors and
//! interpolate on the sphere.

use sphereql_core::SphericalPoint;

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
    /// Weight on the category separation term (0.0 = disabled).
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

/// UMAP-style projection that lives on S² and transforms new points by
/// kNN-weighted averaging over the fitted positions.
#[derive(Clone)]
pub struct UmapSphereProjection {
    /// Unit vectors in ℝ³ — one per fitted embedding.
    fitted_points: Vec<[f64; 3]>,
    /// L2-normalized copies of the original embeddings, kept for
    /// kNN lookup at transform time (UMAP is non-parametric).
    fitted_normalized: Vec<Vec<f64>>,
    dim: usize,
    radial: RadialStrategy,
    n_neighbors: usize,
    /// Post-fit quality proxy in `[0, 1]`: fraction of attractive
    /// edges whose final spherical distance is below the median —
    /// higher means the optimizer actually pulled neighbors together.
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

    /// Fit with custom config. `categories` is parallel to `embeddings`
    /// when supplied; pass `None` to disable the supervised term even
    /// if `config.category_weight > 0`.
    pub fn fit(
        embeddings: &[Embedding],
        categories: Option<&[u32]>,
        radial: RadialStrategy,
        config: UmapConfig,
    ) -> Result<Self, ProjectionError> {
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
        if let Some(cats) = categories
            && cats.len() != n
        {
            return Err(ProjectionError::InconsistentDimension {
                index: cats.len(),
                expected: n,
                got: cats.len(),
            });
        }

        let normalized: Vec<Vec<f64>> = embeddings.iter().map(|e| e.normalized()).collect();
        let k = config.n_neighbors.min(n - 1).max(1);
        let knn = build_knn_graph(&normalized, k);

        // PCA warm-start: project to 3D with the existing PCA path,
        // then push each result onto S² as a unit vector. Reusing the
        // crate's PCA keeps the warm start consistent with the rest of
        // the projection family.
        let warm = pca_warm_start(embeddings, &normalized)?;
        let mut points = warm;

        let mut rng = SplitMix64::new(config.seed);
        let cat_active = config
            .category_weight
            .partial_cmp(&0.0)
            .map(|o| o.is_gt())
            .unwrap_or(false)
            && categories.is_some();

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
            for (i, neighbors) in knn.iter().enumerate() {
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

            // Optional category term: pull same-cat together, push others apart.
            if cat_active {
                let cats = categories.unwrap();
                for i in 0..n {
                    let j = (rng.next_u64() as usize) % n;
                    if j == i {
                        continue;
                    }
                    let (gi, gj) = if cats[i] == cats[j] {
                        attractive_grad(&points[i], &points[j])
                    } else {
                        repulsive_grad(&points[i], &points[j])
                    };
                    let w = config.category_weight;
                    add3_scaled(&mut grads[i], &gi, w);
                    add3_scaled(&mut grads[j], &gj, w);
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

        let quality = neighbor_preservation_score(&points, &knn);

        Ok(Self {
            fitted_points: points,
            fitted_normalized: normalized,
            dim,
            radial,
            n_neighbors: k,
            quality,
        })
    }

    /// Post-fit quality: fraction of attractive edges whose final
    /// great-circle distance is below the median pairwise distance.
    /// Bounded `[0, 1]`; comparable to PCA's explained-variance ratio
    /// for the auto-tuner's `MetaModel` consumers.
    pub fn explained_variance_ratio(&self) -> f64 {
        self.quality
    }

    /// Locate the `n_neighbors` fitted points closest to `embedding`
    /// (cosine similarity in the original space) and return their
    /// indices with similarity weights.
    fn nearest_fitted(&self, normalized: &[f64]) -> Vec<(usize, f64)> {
        let mut sims: Vec<(usize, f64)> = self
            .fitted_normalized
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(normalized, v)))
            .collect();
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(self.n_neighbors);
        sims
    }

    fn project_xyz(&self, embedding: &Embedding) -> ([f64; 3], f64) {
        let normalized = embedding.normalized();
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
        assert_eq!(embedding.dimension(), self.dim);
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
        assert_eq!(embedding.dimension(), self.dim);
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

// ── Gradients ─────────────────────────────────────────────────────────
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

// ── Helpers ────────────────────────────────────────────────────────────

fn build_knn_graph(normalized: &[Vec<f64>], k: usize) -> Vec<Vec<usize>> {
    let n = normalized.len();
    (0..n)
        .map(|i| {
            let mut sims: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, dot(&normalized[i], &normalized[j])))
                .collect();
            sims.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            sims.into_iter().take(k).map(|(j, _)| j).collect()
        })
        .collect()
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
        let sp = pca.project(e);
        let cart = spherical_to_cartesian(&sp);
        let mut v = [cart.x, cart.y, cart.z];
        let mag = normalize_vec(&mut v);
        if mag < f64::EPSILON {
            // Degenerate PCA position (input near corpus mean).
            // Fall back to using the first three normalized coords as a
            // direction — anything stable and deterministic.
            let row = &normalized[i];
            v = [row[0], row[1], row[2]];
            normalize_vec(&mut v);
            if v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 {
                v = [1.0, 0.0, 0.0];
            }
        }
        out.push(v);
    }
    Ok(out)
}

fn neighbor_preservation_score(points: &[[f64; 3]], knn: &[Vec<usize>]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 1.0;
    }
    // Estimate the median pairwise distance with a small random sample
    // (cheap proxy for the full O(n²) median).
    let mut rng = SplitMix64::new(0xBEEF);
    let sample = (n * 4).min(2000);
    let mut sample_d2: Vec<f64> = (0..sample)
        .map(|_| {
            let i = (rng.next_u64() as usize) % n;
            let j = (rng.next_u64() as usize) % n;
            sq_dist(&points[i], &points[j])
        })
        .collect();
    sample_d2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sample_d2
        .get(sample_d2.len() / 2)
        .copied()
        .unwrap_or(f64::INFINITY);

    let mut hits = 0usize;
    let mut total = 0usize;
    for (i, neigh) in knn.iter().enumerate() {
        for &j in neigh {
            if sq_dist(&points[i], &points[j]) <= median {
                hits += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        1.0
    } else {
        hits as f64 / total as f64
    }
}

fn sq_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
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
}
