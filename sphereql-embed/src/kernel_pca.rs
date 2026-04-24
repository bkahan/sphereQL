use sphereql_core::{CartesianPoint, SphericalPoint, cartesian_to_spherical};

use crate::projection::{
    ProjectionError, SplitMix64, dot, normalize_vec, project_xyz_to_spherical,
};
use crate::types::{Embedding, ProjectedPoint, RadialStrategy};

use crate::projection::Projection;

/// Corpus-fitted projection via kernel PCA with a Gaussian (RBF) kernel.
///
/// # Mathematical background
///
/// Standard PCA finds the 3 directions of maximum *linear* variance.
/// Kernel PCA first maps data into an infinite-dimensional feature space
/// **F** via the kernel trick, then performs PCA there. With the Gaussian
/// kernel k(**x**, **y**) = exp(−‖**x**−**y**‖²/(2σ²)), every data point
/// Φ(**x**) lies on a hypersphere **S** in **F** (since k(**x**,**x**) = 1
/// for all **x**). This is a natural fit for SphereQL's spherical geometry.
///
/// The key advantage over linear PCA: kernel PCA captures non-linear
/// manifold structure (curved clusters, rings, spirals) that linear PCA
/// crushes flat. For embedding spaces with complex semantic geometry,
/// this preserves more meaningful neighborhood relationships.
///
/// # Limit behaviour
///
/// * **σ → ∞**: kernel PCA converges to standard PCA (Hoffmann, Appendix A).
/// * **σ → 0**: all points become orthogonal in **F**; PCA is meaningless.
///
/// # Complexity
///
/// * **Fitting**: O(n²·d) to build the kernel matrix + O(n²·q·iters)
///   for power iteration on the n×n centered kernel matrix.
/// * **Projection**: O(n·d) per embedding (n kernel evaluations).
/// * **Memory**: O(n·d) for training data + O(n) per eigenvector.
///
/// # References
///
/// * Schölkopf, Smola, Müller. "Nonlinear component analysis as a kernel
///   eigenvalue problem." *Neural Computation* 10 (1998) 1299–1319.
/// * Hoffmann. "Kernel PCA for novelty detection." *Pattern Recognition*
///   40 (2007) 863–874.
#[derive(Clone)]
pub struct KernelPcaProjection {
    /// Normalized training data — needed at query time for kernel evaluations.
    /// Each inner Vec has length `dim`.
    training_data: Vec<Vec<f64>>,

    /// Gaussian kernel width.
    sigma: f64,

    /// Precomputed 1/(2σ²) to avoid repeated division.
    inv_two_sigma_sq: f64,

    /// Top-3 eigenvector coefficients in ℝⁿ, scaled so ‖α^l‖² = 1/λ_l.
    /// Each Vec has length n = training_data.len().
    alphas: [Vec<f64>; 3],

    /// Top-3 eigenvalues of the centered kernel matrix (descending).
    eigenvalues: [f64; 3],

    /// Column means of the **raw** kernel matrix K.
    /// col_means\[i\] = (1/n) Σⱼ K_{ij}.
    col_means: Vec<f64>,

    /// Grand mean of the **raw** kernel matrix K.
    /// (1/n²) Σ_{i,j} K_{ij}.
    grand_mean: f64,

    /// Total variance in centred feature space = trace(K̃)/n.
    total_variance: f64,

    /// Embedding dimensionality.
    dim: usize,

    /// Controls the radial coordinate.
    radial: RadialStrategy,

    /// When true, r comes from the projection magnitude rather than
    /// the embedding magnitude.
    volumetric: bool,
}

impl KernelPcaProjection {
    /// Fit kernel PCA with automatic σ selection.
    ///
    /// σ is set to the median pairwise Euclidean distance on the normalised
    /// embeddings divided by √2, so that the kernel value at the median
    /// distance is exp(−1) ≈ 0.37. This is a standard heuristic in the
    /// kernel methods literature.
    pub fn fit(embeddings: &[Embedding], radial: RadialStrategy) -> Result<Self, ProjectionError> {
        Self::fit_impl(embeddings, None, radial)
    }

    /// Fit kernel PCA with an explicit kernel width σ.
    ///
    /// Use this when you have domain knowledge about the appropriate scale,
    /// or when benchmarking different σ values. Returns
    /// [`ProjectionError::InvalidSigma`] if `sigma <= 0.0`.
    pub fn fit_with_sigma(
        embeddings: &[Embedding],
        sigma: f64,
        radial: RadialStrategy,
    ) -> Result<Self, ProjectionError> {
        if sigma <= 0.0 {
            return Err(ProjectionError::InvalidSigma { got: sigma });
        }
        Self::fit_impl(embeddings, Some(sigma), radial)
    }

    /// Convenience: fit with default radial strategy and auto σ.
    pub fn fit_default(embeddings: &[Embedding]) -> Result<Self, ProjectionError> {
        Self::fit(embeddings, RadialStrategy::default())
    }

    /// Enable volumetric mode: r comes from the kernel PCA projection
    /// magnitude instead of the embedding magnitude.
    pub fn with_volumetric(mut self, enabled: bool) -> Self {
        self.volumetric = enabled;
        self
    }

    /// The Gaussian kernel width used for this projection.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Number of training points stored (needed for kernel evaluations).
    pub fn num_training_points(&self) -> usize {
        self.training_data.len()
    }

    /// The fraction of total feature-space variance captured by the top-3
    /// kernel principal components.
    ///
    /// Analogous to `PcaProjection::explained_variance_ratio()` but in the
    /// (infinite-dimensional) Gaussian feature space.
    pub fn explained_variance_ratio(&self) -> f64 {
        if self.total_variance < f64::EPSILON {
            return 1.0;
        }
        let explained: f64 = self.eigenvalues.iter().sum();
        // eigenvalues from power iteration on K̃ are the raw eigenvalues;
        // total_variance = trace(K̃)/n, and eigenvalues relate to variance
        // via λ_l/n (since K̃ = n·Cov in feature space). Divide both by n.
        (explained / (self.total_variance * self.training_data.len() as f64)).clamp(0.0, 1.0)
    }

    /// The top-3 eigenvalues of the centred kernel matrix.
    pub fn eigenvalues(&self) -> [f64; 3] {
        self.eigenvalues
    }

    // ── Implementation ─────────────────────────────────────────────────

    fn fit_impl(
        embeddings: &[Embedding],
        sigma: Option<f64>,
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

        // 1. Normalise to unit sphere (angular similarity only)
        let normalized: Vec<Vec<f64>> = embeddings.iter().map(|e| e.normalized()).collect();
        let n = normalized.len();

        // 2. Choose σ
        let sigma = sigma.unwrap_or_else(|| auto_sigma(&normalized));
        let inv_two_sigma_sq = 0.5 / (sigma * sigma);

        // 3. Build the n×n kernel matrix K
        //    K_{ij} = exp(−‖x_i − x_j‖² / (2σ²))
        //    For normalised data, ‖x_i − x_j‖² = 2 − 2(x_i · x_j)
        //    so K_{ij} = exp(−(1 − x_i·x_j) / σ²)
        let mut kernel_flat = vec![0.0_f64; n * n];
        for i in 0..n {
            kernel_flat[i * n + i] = 1.0; // k(x,x) = 1
            for j in (i + 1)..n {
                let val = gaussian_kernel(&normalized[i], &normalized[j], inv_two_sigma_sq);
                kernel_flat[i * n + j] = val;
                kernel_flat[j * n + i] = val;
            }
        }

        // 4. Centering statistics on the raw kernel matrix
        //    col_means[i] = (1/n) Σ_j K_{ij}
        //    grand_mean   = (1/n²) Σ_{i,j} K_{ij}
        let mut col_means = vec![0.0; n];
        let mut grand_sum = 0.0;
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += kernel_flat[i * n + j];
            }
            col_means[i] = row_sum / n as f64;
            grand_sum += row_sum;
        }
        let grand_mean = grand_sum / (n * n) as f64;

        // 5. Centre the kernel matrix (Schölkopf eq. 4):
        //    K̃_{ij} = K_{ij} − col_means[i] − col_means[j] + grand_mean
        //
        //    We overwrite kernel_flat in place to save memory. K̃ is
        //    symmetric in closed form, but `col_means[i] + col_means[j]
        //    - grand_mean` accumulates FP rounding such that K̃[i,j] can
        //    drift from K̃[j,i] by ~1e-15. Across 300 power-iteration
        //    steps on n ≈ 1000 that drift compounds into eigenvector
        //    error; symmetrize once here to eliminate the source.
        for i in 0..n {
            for j in i..n {
                let v = kernel_flat[i * n + j] - col_means[i] - col_means[j] + grand_mean;
                kernel_flat[i * n + j] = v;
                kernel_flat[j * n + i] = v;
            }
        }

        // 6. Total variance in feature space = trace(K̃) / n
        let trace: f64 = (0..n).map(|i| kernel_flat[i * n + i]).sum();
        let total_variance = trace / n as f64;

        // 7. Top-3 eigenvectors via power iteration on K̃
        let (raw_vectors, raw_values) = top_k_symmetric(&kernel_flat, n, 3);

        // 8. Scale eigenvectors: ‖α^l‖² = 1/λ_l so that principal
        //    components V^l in F have unit norm.
        let mut alphas: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut eigenvalues = [0.0_f64; 3];
        for l in 0..3 {
            let lambda = raw_values.get(l).copied().unwrap_or(0.0);
            eigenvalues[l] = lambda;
            if lambda > f64::EPSILON {
                let scale = 1.0 / lambda.sqrt();
                alphas[l] = raw_vectors[l].iter().map(|&a| a * scale).collect();
            } else {
                alphas[l] = vec![0.0; n];
            }
        }

        Ok(Self {
            training_data: normalized,
            sigma,
            inv_two_sigma_sq,
            alphas,
            eigenvalues,
            col_means,
            grand_mean,
            total_variance,
            dim,
            radial,
            volumetric: false,
        })
    }

    /// Compute centred kernel vector and project onto top-3 components.
    ///
    /// Returns (f₁, f₂, f₃, spherical_potential).
    fn kernel_project(&self, normalized: &[f64]) -> (f64, f64, f64, f64) {
        let n = self.training_data.len();

        // k_z[i] = k(z, x_i)
        let k_z: Vec<f64> = self
            .training_data
            .iter()
            .map(|x_i| gaussian_kernel(normalized, x_i, self.inv_two_sigma_sq))
            .collect();

        // mean(k_z)
        let mean_k_z: f64 = k_z.iter().sum::<f64>() / n as f64;

        // Centre: k̃_z[i] = k_z[i] − mean(k_z) − col_means[i] + grand_mean
        let k_z_centered: Vec<f64> = (0..n)
            .map(|i| k_z[i] - mean_k_z - self.col_means[i] + self.grand_mean)
            .collect();

        // Project: f_l = Σ_i α^l_i · k̃_z[i]
        let f1 = dot(&self.alphas[0], &k_z_centered);
        let f2 = dot(&self.alphas[1], &k_z_centered);
        let f3 = dot(&self.alphas[2], &k_z_centered);

        // Spherical potential (Hoffmann eq. 6):
        //   p_S(z) = k(z,z) − (2/n)Σ k(z,x_i) + (1/n²)ΣΣ k(x_i,x_j)
        //          = 1 − 2·mean_k_z + grand_mean
        //
        // This is ‖Φ̃(z)‖² — the total "energy" of the centred point.
        let spherical_potential = (1.0 - 2.0 * mean_k_z + self.grand_mean).max(0.0);

        (f1, f2, f3, spherical_potential)
    }
}

impl Projection for KernelPcaProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        assert_eq!(
            embedding.dimension(),
            self.dim,
            "expected dimension {}, got {}",
            self.dim,
            embedding.dimension()
        );

        let normalized = embedding.normalized();
        let (x, y, z, _) = self.kernel_project(&normalized);

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
        let normalized = embedding.normalized();
        let (x, y, z, spherical_potential) = self.kernel_project(&normalized);

        let projection_magnitude = (x * x + y * y + z * z).sqrt();

        // Per-point certainty from reconstruction error (Hoffmann eq. 11):
        //   p(z)  = p_S(z) − Σ f_l(z)²
        //   certainty = 1 − p(z)/p_S(z) = (Σ f_l²) / p_S(z)
        let projection_sq = x * x + y * y + z * z;
        let certainty = if spherical_potential > f64::EPSILON {
            (projection_sq / spherical_potential).clamp(0.0, 1.0)
        } else {
            0.0
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

// ── Kernel function ────────────────────────────────────────────────────

/// Gaussian (RBF) kernel: k(**x**, **y**) = exp(−‖**x**−**y**‖² · inv_two_sigma_sq)
#[inline]
fn gaussian_kernel(a: &[f64], b: &[f64], inv_two_sigma_sq: f64) -> f64 {
    let sq_dist: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum();
    (-sq_dist * inv_two_sigma_sq).exp()
}

// ── Automatic σ selection ──────────────────────────────────────────────

/// Median-heuristic σ: σ = median(pairwise distances) / √2.
///
/// At the median distance, k = exp(−median²/(2σ²)) = exp(−1) ≈ 0.37.
/// For small datasets (n ≤ 100) all pairs are used; otherwise 5 000
/// random pairs are sampled.
fn auto_sigma(data: &[Vec<f64>]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    let mut distances = Vec::new();
    let mut rng = SplitMix64::new(0xCAFE_BABE);

    if n <= 100 {
        distances.reserve(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(euclidean_dist(&data[i], &data[j]));
            }
        }
    } else {
        let num_pairs = 5000.min(n * (n - 1) / 2);
        distances.reserve(num_pairs);
        for _ in 0..num_pairs {
            let i = (rng.next_u64() as usize) % n;
            let mut j = (rng.next_u64() as usize) % n;
            if j == i {
                j = (j + 1) % n;
            }
            distances.push(euclidean_dist(&data[i], &data[j]));
        }
    }

    // Median via `select_nth_unstable_by` is O(n) vs the O(n log n)
    // full sort — for the 5000-pair sampled branch above that's the
    // difference between ~65k and ~5k comparisons. `total_cmp` is
    // NaN-safe (previously `.partial_cmp().unwrap()` panicked on
    // degenerate zero-distance input).
    let mid = distances.len() / 2;
    let (_, median_ref, _) = distances.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let median = *median_ref;

    (median * std::f64::consts::FRAC_1_SQRT_2).max(1e-8)
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

// ── Eigensolver for symmetric n×n matrix ───────────────────────────────

/// Power iteration with deflation for the top-k eigenvectors of a
/// symmetric n×n matrix stored as a flat row-major array.
///
/// Returns (eigenvectors, eigenvalues) sorted by decreasing eigenvalue.
fn top_k_symmetric(matrix: &[f64], n: usize, k: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let max_iters = 300;
    let tol = 1e-10;
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut values: Vec<f64> = Vec::with_capacity(k);
    let mut rng = SplitMix64::new(0xBEEF_CAFE);

    // Closure: write `matrix · src` into `dst`. Extracted because the
    // inner loop needs this twice — once for the iterate and once for
    // the Rayleigh quotient — and we cache the second call's result
    // into the next iteration's iterate.
    let matvec = |dst: &mut [f64], src: &[f64]| {
        for (i, dst_i) in dst.iter_mut().enumerate() {
            let row_start = i * n;
            let mut s = 0.0;
            for j in 0..n {
                s += matrix[row_start + j] * src[j];
            }
            *dst_i = s;
        }
    };

    for _ in 0..k {
        let mut v: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        normalize_vec(&mut v);
        let mut eigenvalue = 0.0;

        // Cached `matrix · v` from the previous iteration's Rayleigh
        // step. `None` on first pass; populated afterward. Saves one
        // full mat-vec per iteration.
        //
        // Note: a `.skip(1)` bug in earlier versions left `u[0] = 0`;
        // see the `first_coordinate_is_nonzero` regression test.
        let mut mv_cache: Option<Vec<f64>> = None;

        for _ in 0..max_iters {
            let mut u = mv_cache.take().unwrap_or_else(|| {
                let mut u = vec![0.0; n];
                matvec(&mut u, &v);
                u
            });

            for prev in &vectors {
                let proj = dot(&u, prev);
                for (ui, &pi) in u.iter_mut().zip(prev.iter()) {
                    *ui -= proj * pi;
                }
            }

            let mag = normalize_vec(&mut u);
            if mag < f64::EPSILON {
                break;
            }

            // Rayleigh quotient: λ ≈ uᵀ · (matrix · u). Save the
            // `matrix · u` vector — next iteration starts with
            // `u_next = matrix · v_next = matrix · u_current`.
            let mut mv_next = vec![0.0; n];
            matvec(&mut mv_next, &u);
            eigenvalue = u.iter().zip(mv_next.iter()).map(|(a, b)| a * b).sum();

            // Convergence: |⟨u, v⟩| → 1 as u aligns with v.
            // `.max(0.0)` clamps the FP noise that briefly pushes the
            // value below zero near convergence.
            let change = (1.0 - dot(&u, &v).abs()).max(0.0);
            v = u;
            mv_cache = Some(mv_next);

            if change < tol {
                break;
            }
        }

        vectors.push(v);
        values.push(eigenvalue);
    }

    while vectors.len() < k {
        let mut v: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        for prev in &vectors {
            let proj = dot(&v, prev);
            for (vi, &pi) in v.iter_mut().zip(prev.iter()) {
                *vi -= proj * pi;
            }
        }
        normalize_vec(&mut v);
        vectors.push(v);
        values.push(0.0);
    }

    (vectors, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sphereql_core::angular_distance;
    use std::f64::consts::{PI, TAU};

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

    /// Regression test for the `.skip(1)` bug in `top_k_symmetric`:
    /// the mat-vec loop used to start at row 1, leaving `u[0] = 0.0`
    /// on every power-iteration step. That silently zeroed the first
    /// coordinate of every kernel PCA eigenvector. This test fails
    /// loudly if the skip comes back.
    #[test]
    fn top_k_symmetric_first_coordinate_nonzero() {
        // Symmetric, dominant-diagonal 4×4 matrix whose top eigenvector
        // is approximately the all-ones vector (each coordinate ≈ 0.5).
        let n = 4;
        #[rustfmt::skip]
        let matrix = vec![
            4.0, 1.0, 0.5, 0.2,
            1.0, 4.0, 1.0, 0.5,
            0.5, 1.0, 4.0, 1.0,
            0.2, 0.5, 1.0, 4.0,
        ];
        let (vectors, values) = top_k_symmetric(&matrix, n, 2);
        assert_eq!(vectors.len(), 2);
        for (idx, v) in vectors.iter().enumerate() {
            assert!(
                v[0].abs() > 1e-6,
                "eigenvector {idx} has a suspiciously small first \
                 coordinate ({:.2e}) — the .skip(1) bug is back",
                v[0]
            );
        }
        assert!(values[0] > values[1], "eigenvalues must be sorted");
    }

    #[test]
    fn kernel_pca_fit_auto_sigma() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        assert_eq!(kpca.dimensionality(), 10);
        assert!(kpca.sigma() > 0.0);
        assert_eq!(kpca.num_training_points(), 8);
    }

    #[test]
    fn kernel_pca_fit_explicit_sigma() {
        let corpus = corpus_10d();
        let kpca =
            KernelPcaProjection::fit_with_sigma(&corpus, 0.5, RadialStrategy::Fixed(1.0)).unwrap();
        assert!((kpca.sigma() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn kernel_pca_fit_default() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit_default(&corpus).unwrap();
        assert_eq!(kpca.dimensionality(), 10);
    }

    #[test]
    fn kernel_pca_empty_corpus_returns_err() {
        assert!(matches!(
            KernelPcaProjection::fit(&[], RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::EmptyCorpus)
        ));
    }

    #[test]
    fn kernel_pca_too_few_dims_returns_err() {
        assert!(matches!(
            KernelPcaProjection::fit(&[emb(&[1.0, 2.0])], RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::DimensionTooLow {
                got: 2,
                required: 3
            })
        ));
    }

    #[test]
    fn kernel_pca_zero_sigma_returns_err() {
        let corpus = corpus_10d();
        assert!(matches!(
            KernelPcaProjection::fit_with_sigma(&corpus, 0.0, RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::InvalidSigma { got }) if got == 0.0
        ));
    }

    #[test]
    fn kernel_pca_produces_valid_spherical_points() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        for e in &corpus {
            assert_valid_spherical(&kpca.project(e));
        }
    }

    #[test]
    fn kernel_pca_out_of_sample_produces_valid_points() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let novel = emb(&[0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_valid_spherical(&kpca.project(&novel));
    }

    #[test]
    fn kernel_pca_preserves_angular_ordering() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let a = emb(&[1.0, 0.1, 0.0, 0.05, 0.02, -0.01, 0.01, 0.0, 0.02, 0.01]);
        let b = emb(&[0.9, 0.2, 0.1, 0.04, 0.03, 0.0, 0.02, -0.01, 0.01, 0.02]);
        let c = emb(&[-1.0, -0.1, 0.0, -0.04, 0.01, 0.02, 0.01, 0.02, -0.01, 0.01]);
        let pa = kpca.project(&a);
        let pb = kpca.project(&b);
        let pc = kpca.project(&c);
        let d_ab = angular_distance(&pa, &pb);
        let d_ac = angular_distance(&pa, &pc);
        assert!(
            d_ab < d_ac,
            "similar items should be closer: d(a,b)={d_ab:.4} < d(a,c)={d_ac:.4}"
        );
    }

    #[test]
    fn kernel_pca_fixed_radial() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(2.5)).unwrap();
        let sp = kpca.project(&corpus[0]);
        assert!((sp.r - 2.5).abs() < 1e-12);
    }

    #[test]
    fn kernel_pca_magnitude_radial() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Magnitude).unwrap();
        let short = emb(&[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let long = emb(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let ps = kpca.project(&short);
        let pl = kpca.project(&long);
        assert!(ps.r < pl.r, "longer vector should have larger radius");
        assert!((ps.r - 0.1).abs() < 1e-10);
        assert!((pl.r - 10.0).abs() < 1e-10);
    }

    #[test]
    fn kernel_pca_volumetric() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0))
            .unwrap()
            .with_volumetric(true);
        let sp = kpca.project(&corpus[0]);
        assert!(sp.r >= 0.0);
    }

    #[test]
    fn kernel_pca_project_rich_has_valid_certainty() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        for e in &corpus {
            let rich = kpca.project_rich(e);
            assert!(
                rich.certainty >= 0.0 && rich.certainty <= 1.0,
                "certainty must be in [0,1], got {}",
                rich.certainty
            );
            assert!(rich.intensity > 0.0);
            assert!(rich.projection_magnitude >= 0.0);
        }
    }

    #[test]
    fn kernel_pca_certainty_is_meaningful() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let total_certainty: f64 = corpus.iter().map(|e| kpca.project_rich(e).certainty).sum();
        let mean_certainty = total_certainty / corpus.len() as f64;
        assert!(
            mean_certainty > 0.01,
            "mean certainty of training data should be non-trivial, got {mean_certainty}"
        );
    }

    #[test]
    fn kernel_pca_explained_variance_in_range() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let ratio = kpca.explained_variance_ratio();
        assert!(
            ratio > 0.0 && ratio <= 1.0,
            "explained variance ratio should be in (0, 1], got {ratio}"
        );
    }

    #[test]
    fn kernel_pca_eigenvalues_descending() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let ev = kpca.eigenvalues();
        assert!(ev[0] >= ev[1], "eigenvalues must descend: {ev:?}");
        assert!(ev[1] >= ev[2], "eigenvalues must descend: {ev:?}");
        assert!(ev[0] > 0.0, "first eigenvalue should be positive");
    }

    #[test]
    fn gaussian_kernel_self_similarity() {
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let inv = 0.5;
        assert!((gaussian_kernel(&x, &x, inv) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_kernel_symmetry() {
        let a = vec![1.0, 0.0, 0.5];
        let b = vec![0.0, 1.0, -0.3];
        let inv = 1.0;
        assert!((gaussian_kernel(&a, &b, inv) - gaussian_kernel(&b, &a, inv)).abs() < 1e-15);
    }

    #[test]
    fn gaussian_kernel_range() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let inv = 0.5;
        let k = gaussian_kernel(&a, &b, inv);
        assert!(
            k > 0.0 && k <= 1.0,
            "Gaussian kernel must be in (0, 1], got {k}"
        );
    }

    #[test]
    fn auto_sigma_is_positive() {
        let corpus = corpus_10d();
        let normalized: Vec<Vec<f64>> = corpus.iter().map(|e| e.normalized()).collect();
        let sigma = auto_sigma(&normalized);
        assert!(sigma > 0.0);
    }

    #[test]
    fn auto_sigma_single_point() {
        let data = vec![vec![1.0, 0.0, 0.0]];
        assert!((auto_sigma(&data) - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "expected dimension 10")]
    fn kernel_pca_dimension_mismatch_panics() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let _ = kpca.project(&emb(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn kernel_pca_clone_produces_identical_results() {
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let kpca2 = kpca.clone();
        for e in &corpus {
            let sp1 = kpca.project(e);
            let sp2 = kpca2.project(e);
            assert!((sp1.theta - sp2.theta).abs() < 1e-12);
            assert!((sp1.phi - sp2.phi).abs() < 1e-12);
        }
    }

    #[test]
    fn kernel_pca_works_with_embedding_index() {
        use crate::query::EmbeddingIndex;
        let corpus = corpus_10d();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let mut idx = EmbeddingIndex::builder(kpca)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();
        for (i, e) in corpus.iter().enumerate() {
            idx.insert(format!("item-{i}"), e);
        }
        assert_eq!(idx.len(), corpus.len());
        let query = emb(&[0.9, 0.1, 0.0, 0.05, 0.02, -0.01, 0.01, 0.0, 0.02, 0.01]);
        let results = idx.search_nearest(&query, 3);
        assert_eq!(results.len(), 3);
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn large_sigma_approaches_pca_ordering() {
        use crate::projection::PcaProjection;
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let kpca = KernelPcaProjection::fit_with_sigma(&corpus, 100.0, RadialStrategy::Fixed(1.0))
            .unwrap();
        let query = emb(&[1.0, 0.1, 0.0, 0.05, 0.02, -0.01, 0.01, 0.0, 0.02, 0.01]);
        let pca_pt = pca.project(&query);
        let kpca_pt = kpca.project(&query);
        let mut pca_dists: Vec<(usize, f64)> = corpus
            .iter()
            .enumerate()
            .map(|(i, e)| (i, angular_distance(&pca_pt, &pca.project(e))))
            .collect();
        let mut kpca_dists: Vec<(usize, f64)> = corpus
            .iter()
            .enumerate()
            .map(|(i, e)| (i, angular_distance(&kpca_pt, &kpca.project(e))))
            .collect();
        pca_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        kpca_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        assert_eq!(
            pca_dists[0].0, kpca_dists[0].0,
            "nearest neighbour should match between PCA and kernel PCA with large σ"
        );
    }
}
