//! # sphereql-embed
//!
//! Maps high-dimensional real vectors onto the unit 2-sphere via a
//! seeded, orthonormal random linear projection followed by normalization.
//!
//! The construction is a 3×N matrix `P` whose rows are drawn from an
//! i.i.d. standard Gaussian and then orthonormalized with modified
//! Gram-Schmidt plus one reorthogonalization pass ("twice-is-enough",
//! Kahan-Parlett). This gives orthonormality to machine precision
//! independent of the sampling conditioning, which in turn means that
//! `p = P u / ‖P u‖` is an unbiased unit vector on `S²` whose cosine
//! similarity with another projected vector is a rank-consistent, if
//! noisy, proxy for the input cosine similarity (Johnson-Lindenstrauss,
//! specialized to `k = 3`).
//!
//! The mapper is fully deterministic in the `(input_dim, seed)` pair,
//! which is the only state that gets serialized — the projection
//! matrix is regenerated on deserialize. That keeps persisted artifacts
//! tiny (16 bytes of config) while remaining byte-reproducible across
//! processes and machines.
//!
//! ## Example
//!
//! ```
//! use sphereql_embed::{EmbedConfig, EmbeddingMapper};
//!
//! let mapper = EmbeddingMapper::new(EmbedConfig { input_dim: 384, seed: 0xC0FFEE }).unwrap();
//! let v: Vec<f64> = (0..384).map(|i| (i as f64).sin()).collect();
//! let p = mapper.map(&v).unwrap();
//! assert!((p.r - 1.0).abs() < 1e-12);
//! ```

use serde::{Deserialize, Serialize};
use sphereql_core::{CartesianPoint, SphericalPoint, cartesian_to_spherical};

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum EmbedError {
    #[error("input_dim {0} is too small: must be >= 3")]
    InputDimTooSmall(usize),
    #[error("input length {got} does not match mapper input_dim {expected}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("input vector is zero (or within 2^-500 of zero) and cannot be projected")]
    ZeroVector,
    #[error("input vector contains a non-finite component at index {0}")]
    NonFinite(usize),
    #[error(
        "random projection failed to orthonormalize after {retries} resamples; \
         this is statistically impossible for well-conditioned dimensions and \
         indicates a corrupted PRNG state"
    )]
    OrthonormalizationFailed { retries: usize },
}

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------

/// Configuration for an [`EmbeddingMapper`].
///
/// `(input_dim, seed)` fully determines the projection matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedConfig {
    pub input_dim: usize,
    pub seed: u64,
}

impl EmbedConfig {
    pub fn new(input_dim: usize, seed: u64) -> Result<Self, EmbedError> {
        if input_dim < 3 {
            return Err(EmbedError::InputDimTooSmall(input_dim));
        }
        Ok(Self { input_dim, seed })
    }

    pub fn new_unchecked(input_dim: usize, seed: u64) -> Self {
        Self { input_dim, seed }
    }
}

// --------------------------------------------------------------------------
// EmbeddingMapper
// --------------------------------------------------------------------------

/// Deterministic linear map `R^N -> S²`, N = `input_dim`.
///
/// Internally stores a 3×N row-major orthonormal matrix. Construction is
/// O(N) memory and O(N) time for both matrix generation and per-vector
/// projection. The mapper is `Send + Sync` and can be shared across threads.
#[derive(Debug, Clone)]
pub struct EmbeddingMapper {
    config: EmbedConfig,
    // Row-major: rows[0..N], rows[N..2N], rows[2N..3N].
    // Stored flat to keep the three 2D projections in cache-contiguous runs.
    matrix: Box<[f64]>,
}

impl EmbeddingMapper {
    /// Construct a new mapper from a validated config.
    ///
    /// ```
    /// use sphereql_embed::{EmbedConfig, EmbeddingMapper};
    ///
    /// let m = EmbeddingMapper::new(EmbedConfig::new(128, 42).unwrap()).unwrap();
    /// assert_eq!(m.input_dim(), 128);
    /// ```
    pub fn new(config: EmbedConfig) -> Result<Self, EmbedError> {
        if config.input_dim < 3 {
            return Err(EmbedError::InputDimTooSmall(config.input_dim));
        }
        let matrix = generate_orthonormal_projection(config.input_dim, config.seed)?;
        Ok(Self { config, matrix })
    }

    pub fn config(&self) -> &EmbedConfig {
        &self.config
    }

    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Project a single vector to a point on the unit sphere.
    ///
    /// Returns [`EmbedError::DimensionMismatch`] if the input length is
    /// wrong, [`EmbedError::NonFinite`] if any component is NaN or infinity,
    /// and [`EmbedError::ZeroVector`] if the 3D projection collapses to the
    /// origin (true only for inputs orthogonal to all three projection
    /// rows, which has measure zero for nonzero inputs).
    ///
    /// ```
    /// use sphereql_embed::{EmbedConfig, EmbeddingMapper};
    ///
    /// let m = EmbeddingMapper::new(EmbedConfig::new(16, 7).unwrap()).unwrap();
    /// let p = m.map(&[1.0; 16]).unwrap();
    /// assert!((p.r - 1.0).abs() < 1e-12);
    /// ```
    pub fn map(&self, input: &[f64]) -> Result<SphericalPoint, EmbedError> {
        let n = self.config.input_dim;
        if input.len() != n {
            return Err(EmbedError::DimensionMismatch {
                expected: n,
                got: input.len(),
            });
        }
        let mut input_norm_sq = 0.0;
        for (i, &v) in input.iter().enumerate() {
            if !v.is_finite() {
                return Err(EmbedError::NonFinite(i));
            }
            input_norm_sq += v * v;
        }

        let r0 = &self.matrix[0..n];
        let r1 = &self.matrix[n..2 * n];
        let r2 = &self.matrix[2 * n..3 * n];

        let x = dot(r0, input);
        let y = dot(r1, input);
        let z = dot(r2, input);

        let proj_norm_sq = x * x + y * y + z * z;

        // Reject inputs whose direction is undefined on S². Two cases:
        //   1. ‖u‖ = 0 — literal zero vector.
        //   2. ‖Pu‖² / ‖u‖² < ε — the input lies (to machine precision)
        //      in the null space of P; its projected direction is pure
        //      roundoff and any returned sphere point would be noise.
        // We fold both into the ZeroVector variant because the caller's
        // recovery is the same: supply a different vector.
        let threshold = input_norm_sq * 1e-24;
        if input_norm_sq == 0.0 || proj_norm_sq <= threshold {
            return Err(EmbedError::ZeroVector);
        }
        let inv_norm = proj_norm_sq.sqrt().recip();
        let cart = CartesianPoint::new(x * inv_norm, y * inv_norm, z * inv_norm);
        Ok(cartesian_to_spherical(&cart))
    }

    /// Project many vectors. Returns early on the first error.
    ///
    /// Useful for indexing workloads where a caller has a batch of
    /// precomputed embeddings to insert into a sphereQL index.
    pub fn map_batch(&self, inputs: &[Vec<f64>]) -> Result<Vec<SphericalPoint>, EmbedError> {
        let mut out = Vec::with_capacity(inputs.len());
        for v in inputs {
            out.push(self.map(v)?);
        }
        Ok(out)
    }

    /// Minimum-norm preimage of a spherical point under the projection.
    ///
    /// Given `p ∈ S²`, returns the unique vector `u* ∈ R^N` satisfying
    /// `P u* = p` and having minimum Euclidean norm. Because `P` has
    /// orthonormal rows, `P P^T = I₃` and the Moore-Penrose pseudoinverse
    /// is simply `P^T`, so `u* = P^T p`.
    ///
    /// This is **not** a true inverse: `map(unproject(p))` returns `p`,
    /// but `unproject(map(v)) ≠ v` in general (only its component in the
    /// row space of `P` is recovered; the `N-3`-dimensional null space
    /// is discarded). Use this for visualization, debugging, or pulling
    /// a canonical representative back into input space.
    pub fn unproject(&self, point: &SphericalPoint) -> Vec<f64> {
        let n = self.config.input_dim;
        let cart = CartesianPoint::new(
            point.phi.sin() * point.theta.cos(),
            point.phi.sin() * point.theta.sin(),
            point.phi.cos(),
        );
        let r0 = &self.matrix[0..n];
        let r1 = &self.matrix[n..2 * n];
        let r2 = &self.matrix[2 * n..3 * n];
        let mut out = vec![0.0_f64; n];
        for i in 0..n {
            out[i] = cart.x * r0[i] + cart.y * r1[i] + cart.z * r2[i];
        }
        out
    }

    /// Access the raw 3×N row-major projection matrix. Exposed for
    /// verification and benchmarking; most users should prefer [`map`].
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }
}

// EmbeddingMapper serializes just the config; the matrix is regenerated
// on deserialize. This keeps persisted mappers tiny and avoids shipping
// O(N) floats that are trivially reconstructible from (dim, seed).
impl Serialize for EmbeddingMapper {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.config.serialize(s)
    }
}

impl<'de> Deserialize<'de> for EmbeddingMapper {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let config = EmbedConfig::deserialize(d)?;
        EmbeddingMapper::new(config).map_err(serde::de::Error::custom)
    }
}

// --------------------------------------------------------------------------
// PRNG: xoshiro256**
// --------------------------------------------------------------------------
//
// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
// Generators", ACM TOMS 2021. Period 2^256 - 1, passes BigCrush, and has
// no bad seeds except the all-zero state (which we reject in `seed_state`).

#[derive(Debug, Clone, Copy)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        // SplitMix64 is the canonical way to expand a single u64 seed
        // into four xoshiro state words without correlated initial output.
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut t = z;
            t = (t ^ (t >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            t = (t ^ (t >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = t ^ (t >> 31);
        }
        // All-zero state is a fixed point of xoshiro; reseed if we hit it.
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `(0, 1]` double. Mapping `(x >> 11) + 1` guarantees the
    /// result is strictly positive, which keeps `ln(u)` finite in
    /// Box-Muller. Bias from the `+1` is 2^-53.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

// --------------------------------------------------------------------------
// Gaussian sampling (Box-Muller)
// --------------------------------------------------------------------------
//
// Standard Box-Muller: given U1, U2 ~ U(0,1], set
//   r = sqrt(-2 ln U1),  θ = 2π U2,
//   z1 = r cos θ,  z2 = r sin θ.
// Both z1, z2 are independent N(0,1). We buffer the second output.
// The tail concern (U1 → 0) is handled by `next_f64` clamping away from 0.

struct GaussianStream {
    rng: Xoshiro256,
    cached: Option<f64>,
}

impl GaussianStream {
    fn new(rng: Xoshiro256) -> Self {
        Self { rng, cached: None }
    }

    fn next(&mut self) -> f64 {
        if let Some(z) = self.cached.take() {
            return z;
        }
        let u1 = self.rng.next_f64();
        let u2 = self.rng.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        let (s, c) = angle.sin_cos();
        self.cached = Some(radius * s);
        radius * c
    }
}

// --------------------------------------------------------------------------
// Orthonormal projection matrix construction
// --------------------------------------------------------------------------
//
// Generate 3 independent Gaussian rows in R^N, then apply modified
// Gram-Schmidt with one reorthogonalization pass. For Gaussian rows
// at N >= 4, the probability that any residual falls below our safety
// threshold is smaller than 1e-50 — but we still retry on failure to
// keep the function total.

const ORTHO_THRESHOLD_COEFF: f64 = 16.0 * f64::EPSILON;
const MAX_RESAMPLES: usize = 8;

fn generate_orthonormal_projection(n: usize, seed: u64) -> Result<Box<[f64]>, EmbedError> {
    debug_assert!(n >= 3);
    let mut rng = GaussianStream::new(Xoshiro256::new(seed));
    let mut matrix = vec![0.0_f64; 3 * n].into_boxed_slice();

    // Threshold on row norm below which we consider the residual
    // numerically collapsed. For a χ²_{N-k} distribution this is
    // astronomically unlikely to trigger at N >= 4.
    let threshold = ORTHO_THRESHOLD_COEFF * (n as f64).sqrt();

    for row_idx in 0..3 {
        let mut resamples = 0;
        loop {
            let (before, rest) = matrix.split_at_mut(row_idx * n);
            let (row, _after) = rest.split_at_mut(n);
            for slot in row.iter_mut() {
                *slot = rng.next();
            }

            // Modified Gram-Schmidt against all previous rows.
            for k in 0..row_idx {
                let prev = &before[k * n..(k + 1) * n];
                let alpha = dot(prev, row);
                axpy(-alpha, prev, row);
            }
            // Reorthogonalization pass (twice-is-enough).
            for k in 0..row_idx {
                let prev = &before[k * n..(k + 1) * n];
                let alpha = dot(prev, row);
                axpy(-alpha, prev, row);
            }

            let norm = dot(row, row).sqrt();
            if norm > threshold {
                let inv = 1.0 / norm;
                for slot in row.iter_mut() {
                    *slot *= inv;
                }
                break;
            }

            resamples += 1;
            if resamples > MAX_RESAMPLES {
                return Err(EmbedError::OrthonormalizationFailed {
                    retries: resamples,
                });
            }
        }
    }

    Ok(matrix)
}

// --------------------------------------------------------------------------
// Small BLAS-1 helpers
// --------------------------------------------------------------------------

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// `y <- y + alpha * x`
#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sphereql_core::angular_distance;

    fn mapper(n: usize, seed: u64) -> EmbeddingMapper {
        EmbeddingMapper::new(EmbedConfig::new(n, seed).unwrap()).unwrap()
    }

    // ---- Config validation ----

    #[test]
    fn config_rejects_small_dim() {
        assert!(EmbedConfig::new(2, 0).is_err());
        assert!(EmbedConfig::new(3, 0).is_ok());
    }

    #[test]
    fn mapper_new_rejects_small_dim() {
        let cfg = EmbedConfig::new_unchecked(2, 0);
        assert!(matches!(
            EmbeddingMapper::new(cfg),
            Err(EmbedError::InputDimTooSmall(2))
        ));
    }

    // ---- Orthonormality of the projection matrix ----

    #[test]
    fn projection_rows_are_unit_norm() {
        for &n in &[3usize, 4, 16, 128, 768] {
            let m = mapper(n, 0xABCDEF);
            for k in 0..3 {
                let row = &m.matrix()[k * n..(k + 1) * n];
                let norm_sq: f64 = row.iter().map(|x| x * x).sum();
                assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn projection_rows_are_mutually_orthogonal() {
        for &n in &[3usize, 4, 16, 128, 768] {
            let m = mapper(n, 0x123456);
            let r0 = &m.matrix()[0..n];
            let r1 = &m.matrix()[n..2 * n];
            let r2 = &m.matrix()[2 * n..3 * n];
            assert!(dot(r0, r1).abs() < 1e-13);
            assert!(dot(r0, r2).abs() < 1e-13);
            assert!(dot(r1, r2).abs() < 1e-13);
        }
    }

    // ---- map() invariants ----

    #[test]
    fn map_output_is_on_unit_sphere() {
        let m = mapper(256, 1);
        let v: Vec<f64> = (0..256).map(|i| ((i as f64) * 0.17).sin()).collect();
        let p = m.map(&v).unwrap();
        assert_relative_eq!(p.r, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn map_rejects_wrong_length() {
        let m = mapper(16, 0);
        let err = m.map(&[0.0; 15]).unwrap_err();
        assert!(matches!(
            err,
            EmbedError::DimensionMismatch {
                expected: 16,
                got: 15
            }
        ));
    }

    #[test]
    fn map_rejects_zero_vector() {
        let m = mapper(32, 0);
        assert_eq!(m.map(&vec![0.0; 32]), Err(EmbedError::ZeroVector));
    }

    #[test]
    fn map_rejects_nan() {
        let m = mapper(4, 0);
        let mut v = vec![1.0; 4];
        v[2] = f64::NAN;
        assert_eq!(m.map(&v), Err(EmbedError::NonFinite(2)));
    }

    #[test]
    fn map_rejects_infinity() {
        let m = mapper(4, 0);
        let mut v = vec![1.0; 4];
        v[0] = f64::INFINITY;
        assert_eq!(m.map(&v), Err(EmbedError::NonFinite(0)));
    }

    // ---- Determinism & reproducibility ----

    #[test]
    fn same_seed_same_matrix() {
        let a = mapper(64, 0xDEADBEEF);
        let b = mapper(64, 0xDEADBEEF);
        assert_eq!(a.matrix(), b.matrix());
    }

    #[test]
    fn different_seeds_different_matrices() {
        let a = mapper(64, 1);
        let b = mapper(64, 2);
        assert_ne!(a.matrix(), b.matrix());
    }

    #[test]
    fn same_input_same_output() {
        let m = mapper(128, 42);
        let v: Vec<f64> = (0..128).map(|i| ((i * 3) as f64).cos()).collect();
        let p1 = m.map(&v).unwrap();
        let p2 = m.map(&v).unwrap();
        assert_eq!(p1, p2);
    }

    // ---- Scale invariance ----

    #[test]
    fn map_is_scale_invariant() {
        let m = mapper(64, 9);
        let v: Vec<f64> = (0..64).map(|i| ((i as f64) - 32.0).exp2()).collect();
        let p1 = m.map(&v).unwrap();
        let v2: Vec<f64> = v.iter().map(|x| x * 1e3).collect();
        let p2 = m.map(&v2).unwrap();
        let dist = angular_distance(&p1, &p2);
        assert!(dist < 1e-10, "angular distance after scaling = {dist}");
    }

    // ---- Antipodal symmetry ----

    #[test]
    fn antipodal_input_maps_to_antipodal_point() {
        let m = mapper(128, 17);
        let v: Vec<f64> = (0..128).map(|i| ((i as f64) * 0.3).sin()).collect();
        let p_pos = m.map(&v).unwrap();
        let v_neg: Vec<f64> = v.iter().map(|x| -x).collect();
        let p_neg = m.map(&v_neg).unwrap();
        // Antipodal points on S² are separated by angular distance π.
        let sep = angular_distance(&p_pos, &p_neg);
        assert_relative_eq!(sep, std::f64::consts::PI, epsilon = 1e-10);
    }

    // ---- Isometry property for inputs in the row space of P ----
    //
    // For any u that lies in the row space of the projection P (i.e.,
    // u = P^T w for some w ∈ R³), we have Pu = PP^T w = w, and so
    // normalizing gives back exactly w/‖w‖. This gives a tight
    // correctness check that doesn't depend on the JL bound.

    #[test]
    fn row_space_inputs_are_preserved() {
        let m = mapper(64, 99);
        // Construct u = w₀ r₀ + w₁ r₁ + w₂ r₂ for arbitrary w.
        let w = [0.7_f64, -0.3, 1.1];
        let n = 64;
        let r0 = &m.matrix()[0..n];
        let r1 = &m.matrix()[n..2 * n];
        let r2 = &m.matrix()[2 * n..3 * n];
        let u: Vec<f64> = (0..n)
            .map(|i| w[0] * r0[i] + w[1] * r1[i] + w[2] * r2[i])
            .collect();
        let p = m.map(&u).unwrap();
        let w_norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        let expected = CartesianPoint::new(w[0] / w_norm, w[1] / w_norm, w[2] / w_norm);
        let got = CartesianPoint::new(
            p.phi.sin() * p.theta.cos(),
            p.phi.sin() * p.theta.sin(),
            p.phi.cos(),
        );
        assert_relative_eq!(got.x, expected.x, epsilon = 1e-12);
        assert_relative_eq!(got.y, expected.y, epsilon = 1e-12);
        assert_relative_eq!(got.z, expected.z, epsilon = 1e-12);
    }

    // ---- Null space collapse ----
    //
    // An input orthogonal to every projection row should project to
    // the origin and thus be rejected as ZeroVector.

    #[test]
    fn null_space_input_is_rejected() {
        // With only 3 rows in a 4-dim space, we can always construct
        // a nonzero vector orthogonal to all three (the 4th basis
        // direction of the orthogonal complement).
        let m = mapper(4, 0);
        let n = 4;
        let r0 = &m.matrix()[0..n];
        let r1 = &m.matrix()[n..2 * n];
        let r2 = &m.matrix()[2 * n..3 * n];
        // Solve for v ⊥ r0, r1, r2: use a naive constructive approach —
        // start with e0, subtract projections onto each row (same MGS idea).
        let mut v = vec![0.0; n];
        v[0] = 1.0;
        for row in [r0, r1, r2] {
            let a = dot(row, &v);
            for i in 0..n {
                v[i] -= a * row[i];
            }
        }
        // Now v has nonzero magnitude (generically) but is orthogonal
        // to all rows. Rescale so it's not microscopic.
        let vn = dot(&v, &v).sqrt();
        if vn > 1e-12 {
            for x in v.iter_mut() {
                *x /= vn;
            }
            let err = m.map(&v).unwrap_err();
            assert_eq!(err, EmbedError::ZeroVector);
        }
    }

    // ---- unproject is a left-inverse of map restricted to the row space ----

    #[test]
    fn unproject_then_map_is_identity() {
        let m = mapper(32, 123);
        let v: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.7).cos()).collect();
        let p = m.map(&v).unwrap();
        let u_star = m.unproject(&p);
        let p2 = m.map(&u_star).unwrap();
        assert_relative_eq!(angular_distance(&p, &p2), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn unproject_lives_in_row_space() {
        // u* should satisfy u* = P^T (cart(p)), so ‖u*‖ = ‖cart(p)‖ = 1.
        let m = mapper(64, 7);
        let v: Vec<f64> = (0..64).map(|i| (i as f64) - 30.0).collect();
        let p = m.map(&v).unwrap();
        let u_star = m.unproject(&p);
        let norm_sq: f64 = u_star.iter().map(|x| x * x).sum();
        assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-12);
    }

    // ---- Batch API ----

    #[test]
    fn batch_matches_individual() {
        let m = mapper(32, 5);
        let inputs: Vec<Vec<f64>> = (0..10)
            .map(|k| (0..32).map(|i| ((i * k) as f64).sin() + 0.1).collect())
            .collect();
        let batch = m.map_batch(&inputs).unwrap();
        for (i, v) in inputs.iter().enumerate() {
            let single = m.map(v).unwrap();
            assert_eq!(batch[i], single);
        }
    }

    #[test]
    fn batch_propagates_first_error() {
        let m = mapper(4, 0);
        let inputs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 0.0, 0.0, 0.0]];
        assert_eq!(m.map_batch(&inputs), Err(EmbedError::ZeroVector));
    }

    // ---- Serde ----

    #[test]
    fn serde_roundtrip_regenerates_matrix() {
        let m = mapper(128, 0xFEEDFACE);
        let json = serde_json::to_string(&m).unwrap();
        // Config-only: should be tiny.
        assert!(json.len() < 100);
        let restored: EmbeddingMapper = serde_json::from_str(&json).unwrap();
        assert_eq!(m.config(), restored.config());
        assert_eq!(m.matrix(), restored.matrix());
    }

    #[test]
    fn serde_rejects_invalid_dim() {
        let bad_cfg = EmbedConfig::new_unchecked(2, 0);
        let json = serde_json::to_string(&bad_cfg).unwrap();
        assert!(serde_json::from_str::<EmbeddingMapper>(&json).is_err());
    }

    // ---- Cosine preservation (statistical) ----
    //
    // For k = 3, random projection is a very lossy rank proxy. Two
    // random unit vectors in R^N concentrate their cosine tightly
    // around 0 (variance ~1/N), so any test using purely-random pairs
    // is dominated by projection noise and Spearman collapses.
    //
    // We instead build a deliberately-varied test set by linearly
    // interpolating between two fixed anchor vectors, which sweeps
    // input cosine similarity across a wide range and gives the
    // projection a meaningful signal to preserve.

    fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len();
        let rank = |v: &[f64]| -> Vec<f64> {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
            let mut r = vec![0.0; n];
            for (pos, &i) in idx.iter().enumerate() {
                r[i] = pos as f64;
            }
            r
        };
        let rx = rank(xs);
        let ry = rank(ys);
        let mean_x: f64 = rx.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = ry.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut dx = 0.0;
        let mut dy = 0.0;
        for i in 0..n {
            let a = rx[i] - mean_x;
            let b = ry[i] - mean_y;
            num += a * b;
            dx += a * a;
            dy += b * b;
        }
        num / (dx.sqrt() * dy.sqrt())
    }

    #[test]
    fn cosine_preservation_statistical() {
        let n = 256;
        let m = mapper(n, 2024);
        let mut rng = GaussianStream::new(Xoshiro256::new(99));

        // Two fixed, orthogonal anchor vectors.
        let mut a: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        let a_norm = dot(&a, &a).sqrt();
        for x in a.iter_mut() {
            *x /= a_norm;
        }
        let mut b: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        // Gram-Schmidt b against a.
        let ab = dot(&a, &b);
        for i in 0..n {
            b[i] -= ab * a[i];
        }
        let b_norm = dot(&b, &b).sqrt();
        for x in b.iter_mut() {
            *x /= b_norm;
        }

        // Interpolated vectors v(t) = cos(t) a + sin(t) b for t ∈ [0, π],
        // all unit norm and with ⟨a, v(t)⟩ = cos(t) swept over [-1, 1].
        let samples = 64;
        let mut input_cos = Vec::with_capacity(samples);
        let mut output_cos = Vec::with_capacity(samples);
        let p_a = m.map(&a).unwrap();
        for k in 0..samples {
            let t = std::f64::consts::PI * (k as f64) / ((samples - 1) as f64);
            let (s, c) = t.sin_cos();
            let v: Vec<f64> = (0..n).map(|i| c * a[i] + s * b[i]).collect();
            input_cos.push(c);
            let p_v = m.map(&v).unwrap();
            output_cos.push(angular_distance(&p_a, &p_v).cos());
        }

        let rho = spearman(&input_cos, &output_cos);
        // Realistic threshold for k=3 projection of a swept cosine range:
        // the theoretical upper bound on Pearson correlation of a single
        // 3D random projection is ~sqrt(3/5) ≈ 0.775, and Spearman
        // typically tracks Pearson here because the sweep produces a
        // monotonic ground-truth signal with wide dynamic range.
        assert!(
            rho > 0.6,
            "Spearman correlation {rho} too low — rank preservation degraded"
        );
    }

    // ---- Distance preservation: known-separated inputs ----
    //
    // If two inputs are identical, their projections are identical.
    // If the two inputs differ by a scalar multiple, same.

    #[test]
    fn identical_inputs_map_to_same_point() {
        let m = mapper(64, 3);
        let v: Vec<f64> = (0..64).map(|i| (i as f64).ln_1p()).collect();
        let p1 = m.map(&v).unwrap();
        let p2 = m.map(&v).unwrap();
        assert_eq!(p1, p2);
    }

    // ---- Proptest: any finite nonzero input produces a valid sphere point ----

    proptest::proptest! {
        #[test]
        fn prop_any_finite_input_stays_on_sphere(
            seed in 0u64..1000,
            values in proptest::collection::vec(-100.0_f64..100.0, 8..=8),
        ) {
            let m = mapper(8, seed);
            if let Ok(p) = m.map(&values) {
                proptest::prop_assert!((p.r - 1.0).abs() < 1e-10);
                proptest::prop_assert!(p.theta >= 0.0 && p.theta < std::f64::consts::TAU);
                proptest::prop_assert!(p.phi >= 0.0 && p.phi <= std::f64::consts::PI);
            }
        }

        #[test]
        fn prop_scale_invariance(
            seed in 0u64..100,
            values in proptest::collection::vec(-10.0_f64..10.0, 16..=16),
            scale in 1e-3_f64..1e3,
        ) {
            let m = mapper(16, seed);
            if let (Ok(p1), Ok(p2)) = (
                m.map(&values),
                m.map(&values.iter().map(|x| x * scale).collect::<Vec<_>>()),
            ) {
                proptest::prop_assert!(angular_distance(&p1, &p2) < 1e-9);
            }
        }
    }
}
