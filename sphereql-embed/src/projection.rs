use std::f64::consts::PI;
use std::sync::Arc;

use sphereql_core::{CartesianPoint, SphericalPoint, cartesian_to_spherical};

use crate::types::{Embedding, ProjectedPoint, RadialStrategy};

/// Reasons a projection fit can fail.
///
/// Every concrete projection's `fit` used to panic via `assert!` on
/// invalid input, which turned typos in Python / WASM bindings into
/// `PanicException`s. These variants classify the same preconditions
/// so callers can surface typed errors instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProjectionError {
    /// The input slice was empty. Fitting needs at least one embedding.
    #[error("need at least one embedding to fit a projection")]
    EmptyCorpus,

    /// Embedding dimensionality below the projection's requirement.
    /// PCA and kernel PCA need `dim >= 3`; Laplacian requires `dim > 0`.
    #[error("embedding dimension {got} is below the minimum {required} for this projection")]
    DimensionTooLow { got: usize, required: usize },

    /// Embeddings disagreed on dimensionality. Every row must match the
    /// first one; the mismatch is reported with the offending index.
    #[error("embedding {index} has dimension {got}, expected {expected}")]
    InconsistentDimension {
        index: usize,
        expected: usize,
        got: usize,
    },

    /// Projection needs more embeddings than were provided. Laplacian
    /// eigenmap's graph construction requires `n >= 4`.
    #[error("need at least {required} embeddings, got {got}")]
    TooFewEmbeddings { got: usize, required: usize },

    /// `fit_with_sigma` was given a non-positive Gaussian bandwidth.
    #[error("kernel bandwidth σ must be positive, got {got}")]
    InvalidSigma { got: f64 },
}

/// Maps high-dimensional embeddings to spherical coordinates.
///
/// The angular coordinates (theta, phi) encode semantic direction via
/// dimensionality reduction from S^{n-1} to S^2. The radial coordinate
/// is controlled by the projection's [`RadialStrategy`].
pub trait Projection: Send + Sync {
    fn project(&self, embedding: &Embedding) -> SphericalPoint;

    /// Project with rich metadata: certainty, intensity, projection magnitude.
    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint {
        let position = self.project(embedding);
        ProjectedPoint::from_position(position, embedding.magnitude())
    }

    fn dimensionality(&self) -> usize;
}

impl<P: Projection> Projection for Arc<P> {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        (**self).project(embedding)
    }
    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint {
        (**self).project_rich(embedding)
    }
    fn dimensionality(&self) -> usize {
        (**self).dimensionality()
    }
}

/// Corpus-fitted projection via spherical PCA.
///
/// Finds the 3 principal directions of maximum angular variance in the
/// embedding space, then projects new embeddings onto them. This preserves
/// angular (cosine similarity) relationships as faithfully as possible
/// in 3 dimensions.
///
/// Fitting: O(N·n·k·iters) where N=corpus size, n=dimension, k=3.
/// Projection: O(n) per embedding.
#[derive(Clone)]
pub struct PcaProjection {
    components: [Vec<f64>; 3],
    mean: Vec<f64>,
    dim: usize,
    radial: RadialStrategy,
    volumetric: bool,
    /// Top-3 eigenvalues from PCA (descending). Used to compute per-point certainty.
    eigenvalues: [f64; 3],
    /// Total variance across all dimensions. eigenvalues[0..3].sum() / total_variance
    /// gives the global explained variance ratio.
    total_variance: f64,
}

impl PcaProjection {
    /// Fit the top-3 principal components on `embeddings`.
    ///
    /// Returns [`ProjectionError::EmptyCorpus`] if the slice is empty,
    /// [`ProjectionError::DimensionTooLow`] if `dim < 3`, and
    /// [`ProjectionError::InconsistentDimension`] if any row's
    /// dimensionality disagrees with the first. Previously these paths
    /// panicked via `assert!`, which surfaced as a `PanicException` in
    /// Python / WASM bindings.
    pub fn fit(
        embeddings: &[Embedding],
        radial: RadialStrategy,
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

        let normalized: Vec<Vec<f64>> = embeddings.iter().map(|e| e.normalized()).collect();
        let n = normalized.len();

        let mut mean = vec![0.0; dim];
        for v in &normalized {
            for (i, &val) in v.iter().enumerate() {
                mean[i] += val;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }

        let centered: Vec<Vec<f64>> = normalized
            .iter()
            .map(|v| {
                v.iter()
                    .zip(mean.iter())
                    .map(|(&val, &m)| val - m)
                    .collect()
            })
            .collect();

        let (components, eigenvalues) = top_k_eigenvectors(&centered, 3, dim);

        // Total variance = sum of all eigenvalues = trace of covariance = sum of squared norms
        let total_variance: f64 = centered
            .iter()
            .map(|row| row.iter().map(|x| x * x).sum::<f64>())
            .sum::<f64>()
            / centered.len() as f64;

        Ok(Self {
            components: [
                components[0].clone(),
                components[1].clone(),
                components[2].clone(),
            ],
            mean,
            dim,
            radial,
            volumetric: false,
            eigenvalues: [
                eigenvalues.first().copied().unwrap_or(0.0),
                eigenvalues.get(1).copied().unwrap_or(0.0),
                eigenvalues.get(2).copied().unwrap_or(0.0),
            ],
            total_variance,
        })
    }

    pub fn fit_default(embeddings: &[Embedding]) -> Result<Self, ProjectionError> {
        Self::fit(embeddings, RadialStrategy::default())
    }

    /// Enable volumetric mode: r comes from the PCA projection magnitude
    /// instead of the embedding magnitude. Points distribute through the
    /// full 3D volume rather than clustering on the sphere surface.
    pub fn with_volumetric(mut self, enabled: bool) -> Self {
        self.volumetric = enabled;
        self
    }

    /// The fraction of total variance captured by the top-3 PCA components.
    /// A global quality metric for the projection — higher means less information lost.
    pub fn explained_variance_ratio(&self) -> f64 {
        if self.total_variance < f64::EPSILON {
            return 1.0;
        }
        let explained: f64 = self.eigenvalues.iter().sum();
        (explained / self.total_variance).clamp(0.0, 1.0)
    }

    /// Allocation-free projection kernel: folds
    /// `normalize(embedding) − mean` into the per-axis dot product
    /// without materializing the intermediate `Vec<f64>`s that the
    /// previous implementation allocated per call.
    ///
    /// Matches the numerics of `project_centered(&centered)` exactly:
    /// each axis sums `(v_i/|v| − mean_i) · component_j[i]` over i,
    /// plus a total-squared accumulator for the residual.
    ///
    /// Called by [`Self::project`] and [`Self::project_rich`]; callers
    /// that want `SphericalPoint` or `ProjectedPoint` should use those.
    fn project_xyz_residual(&self, embedding: &Embedding) -> (f64, f64, f64, f64) {
        let values = &embedding.values;
        debug_assert_eq!(values.len(), self.dim);

        let mag = embedding.magnitude();
        let inv_mag = if mag < f64::EPSILON { 0.0 } else { 1.0 / mag };

        let mut x = 0.0f64;
        let mut y = 0.0f64;
        let mut z = 0.0f64;
        let mut total_sq = 0.0f64;
        let c0 = &self.components[0];
        let c1 = &self.components[1];
        let c2 = &self.components[2];
        for i in 0..self.dim {
            let n = values[i] * inv_mag;
            let c = n - self.mean[i];
            x += c * c0[i];
            y += c * c1[i];
            z += c * c2[i];
            total_sq += c * c;
        }
        let projected_sq = x * x + y * y + z * z;
        let residual_sq = (total_sq - projected_sq).max(0.0);
        (x, y, z, residual_sq)
    }
}

impl Projection for PcaProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        assert_eq!(
            embedding.dimension(),
            self.dim,
            "expected dimension {}, got {}",
            self.dim,
            embedding.dimension()
        );

        let (x, y, z, _) = self.project_xyz_residual(embedding);

        if self.volumetric {
            let sp = cartesian_to_spherical(&CartesianPoint::new(x, y, z));
            if sp.r < f64::EPSILON {
                return SphericalPoint::new_unchecked(0.0, 0.0, 0.0);
            }
            SphericalPoint::new_unchecked(sp.r, sp.theta, sp.phi)
        } else {
            let r = self.radial.compute(embedding.magnitude());
            project_xyz_to_spherical(x, y, z, r)
        }
    }

    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint {
        assert_eq!(
            embedding.dimension(),
            self.dim,
            "expected dimension {}, got {}",
            self.dim,
            embedding.dimension()
        );

        let intensity = embedding.magnitude();
        let (x, y, z, residual_sq) = self.project_xyz_residual(embedding);
        let projection_magnitude = (x * x + y * y + z * z).sqrt();

        // Per-point certainty: fraction of this point's variance captured
        // by the 3 components. The fold below also drops the separate
        // centered-vec allocation the old version materialized.
        let inv_mag = if intensity < f64::EPSILON {
            0.0
        } else {
            1.0 / intensity
        };
        let total_sq: f64 = (0..self.dim)
            .map(|i| {
                let c = embedding.values[i] * inv_mag - self.mean[i];
                c * c
            })
            .sum();
        let certainty = if total_sq < f64::EPSILON {
            0.0
        } else {
            (1.0 - residual_sq / total_sq).clamp(0.0, 1.0)
        };

        let position = if self.volumetric {
            let sp = cartesian_to_spherical(&CartesianPoint::new(x, y, z));
            if sp.r < f64::EPSILON {
                SphericalPoint::new_unchecked(0.0, 0.0, 0.0)
            } else {
                SphericalPoint::new_unchecked(sp.r, sp.theta, sp.phi)
            }
        } else {
            let r = self.radial.compute(intensity);
            project_xyz_to_spherical(x, y, z, r)
        };

        ProjectedPoint::new(position, certainty, intensity, projection_magnitude)
    }

    fn dimensionality(&self) -> usize {
        self.dim
    }
}

/// Fit-free projection via random matrix (Johnson-Lindenstrauss).
///
/// Generates a fixed 3×n random matrix at construction time. Preserves
/// pairwise distances probabilistically without needing a training corpus.
/// Less accurate than PCA for any specific dataset, but useful when
/// no corpus is available or for quick prototyping.
///
/// Deterministic for a given seed.
#[derive(Clone)]
pub struct RandomProjection {
    matrix: [Vec<f64>; 3],
    dim: usize,
    radial: RadialStrategy,
}

impl RandomProjection {
    pub fn new(dim: usize, radial: RadialStrategy, seed: u64) -> Self {
        assert!(dim >= 3, "embedding dimension must be >= 3");
        let mut rng = SplitMix64::new(seed);
        let matrix = std::array::from_fn(|_| (0..dim).map(|_| rng.normal()).collect());
        Self {
            matrix,
            dim,
            radial,
        }
    }

    pub fn new_default(dim: usize) -> Self {
        Self::new(dim, RadialStrategy::default(), 42)
    }
}

impl Projection for RandomProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        assert_eq!(
            embedding.dimension(),
            self.dim,
            "expected dimension {}, got {}",
            self.dim,
            embedding.dimension()
        );

        let magnitude = embedding.magnitude();
        let r = self.radial.compute(magnitude);
        let normalized = embedding.normalized();

        let x = dot(&normalized, &self.matrix[0]);
        let y = dot(&normalized, &self.matrix[1]);
        let z = dot(&normalized, &self.matrix[2]);

        project_xyz_to_spherical(x, y, z, r)
    }

    fn dimensionality(&self) -> usize {
        self.dim
    }
}

// --- Shared projection math (pub(crate) for reuse by kernel_pca) ---

pub(crate) fn project_xyz_to_spherical(x: f64, y: f64, z: f64, r: f64) -> SphericalPoint {
    let cart = CartesianPoint::new(x, y, z).normalize();
    if cart.magnitude() < f64::EPSILON {
        return SphericalPoint::new_unchecked(r, 0.0, 0.0);
    }
    let sp = cartesian_to_spherical(&cart);
    SphericalPoint::new_unchecked(r, sp.theta, sp.phi)
}

// --- Linear algebra primitives (pub(crate) for reuse by kernel_pca) ---

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

pub(crate) fn normalize_vec(v: &mut [f64]) -> f64 {
    let mag = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag > f64::EPSILON {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
    mag
}

/// Power iteration with deflation for the top-k eigenvectors of XᵀX.
///
/// Computes XᵀX·v as Xᵀ(Xv) to avoid forming the n×n matrix,
/// keeping each iteration at O(N·n) instead of O(n²).
///
/// Returns (eigenvectors, eigenvalues) both sorted by decreasing eigenvalue.
fn top_k_eigenvectors(data: &[Vec<f64>], k: usize, dim: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let max_iters = 200;
    let tol = 1e-10;
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut values: Vec<f64> = Vec::with_capacity(k);
    let mut rng = SplitMix64::new(0xDEAD_BEEF);
    let n = data.len() as f64;

    for _ in 0..k {
        let mut v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        normalize_vec(&mut v);
        let mut eigenvalue = 0.0;

        for _ in 0..max_iters {
            // w = Xv ∈ ℝᴺ
            let w: Vec<f64> = data.iter().map(|row| dot(row, &v)).collect();

            // u = Xᵀw ∈ ℝⁿ
            let mut u = vec![0.0; dim];
            for (row, &wi) in data.iter().zip(w.iter()) {
                for (uj, &rj) in u.iter_mut().zip(row.iter()) {
                    *uj += wi * rj;
                }
            }

            // Deflate: remove components along previously found eigenvectors
            for prev in &vectors {
                let proj = dot(&u, prev);
                for (uj, &pj) in u.iter_mut().zip(prev.iter()) {
                    *uj -= proj * pj;
                }
            }

            let mag = normalize_vec(&mut u);
            if mag < f64::EPSILON {
                break;
            }

            // The eigenvalue is vᵀ(XᵀX)v / N = mag / N (before normalization)
            eigenvalue = mag / n;

            // `.max(0.0)` clamps the FP noise that can briefly push
            // `1 - |⟨u,v⟩|` slightly negative near convergence.
            let change = (1.0 - dot(&u, &v).abs()).max(0.0);
            v = u;

            if change < tol {
                break;
            }
        }

        vectors.push(v);
        values.push(eigenvalue);
    }

    // If some components had zero variance, fill with orthogonal random directions
    while vectors.len() < k {
        let mut v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        for prev in &vectors {
            let proj = dot(&v, prev);
            for (vj, &pj) in v.iter_mut().zip(prev.iter()) {
                *vj -= proj * pj;
            }
        }
        normalize_vec(&mut v);
        vectors.push(v);
        values.push(0.0);
    }

    (vectors, values)
}

// --- Deterministic PRNG (SplitMix64 + Box-Muller) ---
// pub(crate) for reuse by kernel_pca module.

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sphereql_core::angular_distance;
    use std::f64::consts::TAU;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    fn corpus_10d() -> Vec<Embedding> {
        vec![
            emb(&[1.0, 0.0, 0.0, 0.1, 0.05, -0.02, 0.03, -0.01, 0.04, 0.02]),
            emb(&[0.0, 1.0, 0.0, -0.05, 0.1, 0.03, -0.02, 0.01, -0.03, 0.04]),
            emb(&[0.0, 0.0, 1.0, 0.02, -0.03, 0.1, 0.05, 0.02, -0.01, -0.04]),
            emb(&[1.0, 1.0, 0.0, 0.05, 0.08, 0.01, 0.01, -0.02, 0.02, 0.03]),
            emb(&[0.0, 1.0, 1.0, -0.02, 0.07, 0.07, 0.01, 0.02, -0.02, 0.01]),
            emb(&[1.0, 0.0, 1.0, 0.06, 0.01, 0.05, -0.03, -0.01, 0.03, -0.02]),
            emb(&[-1.0, 0.0, 0.0, -0.08, 0.02, 0.01, 0.02, 0.03, -0.02, 0.01]),
            emb(&[0.0, -1.0, 0.0, 0.03, -0.09, -0.02, 0.01, -0.01, 0.02, -0.03]),
        ]
    }

    fn assert_valid_spherical(sp: &SphericalPoint) {
        assert!(sp.r >= 0.0, "r must be >= 0, got {}", sp.r);
        assert!(
            sp.theta >= 0.0 && sp.theta < TAU,
            "theta must be in [0, 2π), got {}",
            sp.theta
        );
        assert!(
            sp.phi >= 0.0 && sp.phi <= PI,
            "phi must be in [0, π], got {}",
            sp.phi
        );
    }

    // --- PCA tests ---

    #[test]
    fn pca_produces_valid_spherical_points() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        for e in &corpus {
            assert_valid_spherical(&pca.project(e));
        }
    }

    #[test]
    fn pca_preserves_angular_ordering() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();

        // a and b are both +x-ish, c is -x: a should be closer to b than to c
        let a = emb(&[1.0, 0.1, 0.0, 0.05, 0.02, -0.01, 0.01, 0.0, 0.02, 0.01]);
        let b = emb(&[0.9, 0.2, 0.1, 0.04, 0.03, 0.0, 0.02, -0.01, 0.01, 0.02]);
        let c = emb(&[-1.0, -0.1, 0.0, -0.04, 0.01, 0.02, 0.01, 0.02, -0.01, 0.01]);

        let pa = pca.project(&a);
        let pb = pca.project(&b);
        let pc = pca.project(&c);

        let d_ab = angular_distance(&pa, &pb);
        let d_ac = angular_distance(&pa, &pc);

        assert!(
            d_ab < d_ac,
            "similar items should be closer: d(a,b)={d_ab:.4} should be < d(a,c)={d_ac:.4}"
        );
    }

    #[test]
    fn pca_magnitude_radial() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude).unwrap();

        let short = emb(&[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let long = emb(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let ps = pca.project(&short);
        let pl = pca.project(&long);

        assert!(ps.r < pl.r, "longer vector should have larger radius");
        assert!((ps.r - 0.1).abs() < 1e-10);
        assert!((pl.r - 10.0).abs() < 1e-10);
    }

    #[test]
    fn pca_transform_radial() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(
            &corpus,
            RadialStrategy::MagnitudeTransform(Arc::new(|mag| mag.ln_1p())),
        )
        .unwrap();

        let e = emb(&[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sp = pca.project(&e);
        assert!((sp.r - 5.0_f64.ln_1p()).abs() < 1e-10);
    }

    #[test]
    fn pca_single_embedding() {
        let corpus = vec![emb(&[1.0, 0.0, 0.0, 0.0, 0.0])];
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let sp = pca.project(&corpus[0]);
        assert!((sp.r - 1.0).abs() < 1e-12);
        assert_valid_spherical(&sp);
    }

    #[test]
    fn pca_dimensionality() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        assert_eq!(pca.dimensionality(), 10);
    }

    #[test]
    fn pca_empty_corpus_returns_err() {
        assert!(matches!(
            PcaProjection::fit(&[], RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::EmptyCorpus)
        ));
    }

    #[test]
    fn pca_too_few_dimensions_returns_err() {
        assert!(matches!(
            PcaProjection::fit(&[emb(&[1.0, 2.0])], RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::DimensionTooLow {
                got: 2,
                required: 3
            })
        ));
    }

    #[test]
    #[should_panic(expected = "expected dimension 10")]
    fn pca_dimension_mismatch_panics() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let _ = pca.project(&emb(&[1.0, 2.0, 3.0]));
    }

    // --- Random projection tests ---

    #[test]
    fn random_produces_valid_spherical_points() {
        let rp = RandomProjection::new(10, RadialStrategy::Fixed(1.0), 42);
        for i in 0..20 {
            let e = emb(&[i as f64 * 0.1 + 0.01; 10]);
            assert_valid_spherical(&rp.project(&e));
        }
    }

    #[test]
    fn random_deterministic_with_same_seed() {
        let rp1 = RandomProjection::new(10, RadialStrategy::Fixed(1.0), 42);
        let rp2 = RandomProjection::new(10, RadialStrategy::Fixed(1.0), 42);
        let e = emb(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let sp1 = rp1.project(&e);
        let sp2 = rp2.project(&e);
        assert!((sp1.theta - sp2.theta).abs() < 1e-12);
        assert!((sp1.phi - sp2.phi).abs() < 1e-12);
    }

    #[test]
    fn random_different_seeds_differ() {
        let rp1 = RandomProjection::new(10, RadialStrategy::Fixed(1.0), 42);
        let rp2 = RandomProjection::new(10, RadialStrategy::Fixed(1.0), 999);
        let e = emb(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let d = angular_distance(&rp1.project(&e), &rp2.project(&e));
        assert!(
            d > 1e-6,
            "different seeds should produce different projections"
        );
    }

    #[test]
    fn random_dimensionality() {
        let rp = RandomProjection::new(768, RadialStrategy::Fixed(1.0), 0);
        assert_eq!(rp.dimensionality(), 768);
    }

    #[test]
    #[should_panic(expected = "embedding dimension must be >= 3")]
    fn random_too_few_dimensions_panics() {
        RandomProjection::new(2, RadialStrategy::Fixed(1.0), 0);
    }

    // --- Arc delegation ---

    #[test]
    fn arc_projection_delegates() {
        let rp = Arc::new(RandomProjection::new_default(10));
        let e = emb(&[1.0; 10]);
        let sp = rp.project(&e);
        assert!(sp.r > 0.0);
        assert_eq!(rp.dimensionality(), 10);
    }

    // --- SplitMix64 sanity ---

    #[test]
    fn prng_produces_distinct_values() {
        let mut rng = SplitMix64::new(42);
        let vals: Vec<f64> = (0..100).map(|_| rng.next_f64()).collect();
        for i in 0..vals.len() {
            for j in (i + 1)..vals.len() {
                assert_ne!(vals[i].to_bits(), vals[j].to_bits());
            }
        }
    }

    #[test]
    fn prng_normal_distribution_reasonable() {
        let mut rng = SplitMix64::new(12345);
        let samples: Vec<f64> = (0..10_000).map(|_| rng.normal()).collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(mean.abs() < 0.05, "mean should be near 0, got {mean}");
        assert!(
            (variance - 1.0).abs() < 0.1,
            "variance should be near 1, got {variance}"
        );
    }
}
