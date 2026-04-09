use std::f64::consts::PI;
use std::sync::Arc;

use sphereql_core::{CartesianPoint, SphericalPoint, cartesian_to_spherical};

use crate::types::{Embedding, RadialStrategy};

/// Maps high-dimensional embeddings to spherical coordinates.
///
/// The angular coordinates (theta, phi) encode semantic direction via
/// dimensionality reduction from S^{n-1} to S^2. The radial coordinate
/// is controlled by the projection's [`RadialStrategy`].
pub trait Projection: Send + Sync {
    fn project(&self, embedding: &Embedding) -> SphericalPoint;
    fn dimensionality(&self) -> usize;
}

impl<P: Projection> Projection for Arc<P> {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        (**self).project(embedding)
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
}

impl PcaProjection {
    pub fn fit(embeddings: &[Embedding], radial: RadialStrategy) -> Self {
        assert!(
            !embeddings.is_empty(),
            "need at least one embedding to fit PCA"
        );
        let dim = embeddings[0].dimension();
        assert!(dim >= 3, "embedding dimension must be >= 3");
        for (i, e) in embeddings.iter().enumerate() {
            assert_eq!(
                e.dimension(),
                dim,
                "embedding {i} has dimension {}, expected {dim}",
                e.dimension()
            );
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

        let components = top_k_eigenvectors(&centered, 3, dim);

        Self {
            components: [
                components[0].clone(),
                components[1].clone(),
                components[2].clone(),
            ],
            mean,
            dim,
            radial,
        }
    }

    pub fn fit_default(embeddings: &[Embedding]) -> Self {
        Self::fit(embeddings, RadialStrategy::default())
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

        let magnitude = embedding.magnitude();
        let r = self.radial.compute(magnitude);
        let normalized = embedding.normalized();

        let centered: Vec<f64> = normalized
            .iter()
            .zip(self.mean.iter())
            .map(|(&v, &m)| v - m)
            .collect();

        let x = dot(&centered, &self.components[0]);
        let y = dot(&centered, &self.components[1]);
        let z = dot(&centered, &self.components[2]);

        project_xyz_to_spherical(x, y, z, r)
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
        Self { matrix, dim, radial }
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

// --- Shared projection math ---

fn project_xyz_to_spherical(x: f64, y: f64, z: f64, r: f64) -> SphericalPoint {
    let cart = CartesianPoint::new(x, y, z).normalize();
    if cart.magnitude() < f64::EPSILON {
        return SphericalPoint::new_unchecked(r, 0.0, 0.0);
    }
    let sp = cartesian_to_spherical(&cart);
    SphericalPoint::new_unchecked(r, sp.theta, sp.phi)
}

// --- Linear algebra primitives ---

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn normalize_vec(v: &mut [f64]) -> f64 {
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
fn top_k_eigenvectors(data: &[Vec<f64>], k: usize, dim: usize) -> Vec<Vec<f64>> {
    let max_iters = 200;
    let tol = 1e-10;
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng = SplitMix64::new(0xDEAD_BEEF);

    for _ in 0..k {
        let mut v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        normalize_vec(&mut v);

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

            let change = 1.0 - dot(&u, &v).abs();
            v = u;

            if change < tol {
                break;
            }
        }

        vectors.push(v);
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
    }

    vectors
}

// --- Deterministic PRNG (SplitMix64 + Box-Muller) ---

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
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
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        for e in &corpus {
            assert_valid_spherical(&pca.project(e));
        }
    }

    #[test]
    fn pca_preserves_angular_ordering() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));

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
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude);

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
        );

        let e = emb(&[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sp = pca.project(&e);
        assert!((sp.r - 5.0_f64.ln_1p()).abs() < 1e-10);
    }

    #[test]
    fn pca_single_embedding() {
        let corpus = vec![emb(&[1.0, 0.0, 0.0, 0.0, 0.0])];
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let sp = pca.project(&corpus[0]);
        assert!((sp.r - 1.0).abs() < 1e-12);
        assert_valid_spherical(&sp);
    }

    #[test]
    fn pca_dimensionality() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        assert_eq!(pca.dimensionality(), 10);
    }

    #[test]
    #[should_panic(expected = "need at least one embedding")]
    fn pca_empty_corpus_panics() {
        PcaProjection::fit(&[], RadialStrategy::Fixed(1.0));
    }

    #[test]
    #[should_panic(expected = "embedding dimension must be >= 3")]
    fn pca_too_few_dimensions_panics() {
        PcaProjection::fit(&[emb(&[1.0, 2.0])], RadialStrategy::Fixed(1.0));
    }

    #[test]
    #[should_panic(expected = "expected dimension 10")]
    fn pca_dimension_mismatch_panics() {
        let corpus = corpus_10d();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        pca.project(&emb(&[1.0, 2.0, 3.0]));
    }

    // --- Random projection tests ---

    #[test]
    fn random_produces_valid_spherical_points() {
        let rp = RandomProjection::new(10, RadialStrategy::Fixed(1.0), 42);
        for i in 0..20 {
            let e = emb(&vec![i as f64 * 0.1 + 0.01; 10]);
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
        assert!(d > 1e-6, "different seeds should produce different projections");
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
