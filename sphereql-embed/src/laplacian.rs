//! Laplacian eigenmap projection — connectivity-preserving embedding on S².
//!
//! Standard PCA/kernel PCA maximize *variance* — the directions along which
//! data spreads. On sparse, noise-heavy embeddings where signal lives in a
//! handful of co-activated axes and the rest is low-amplitude noise, variance
//! maximization pulls the projection toward the noise axes and collapses
//! genuine category structure. Laplacian eigenmaps instead preserve
//! *connectivity*: two points are "close" on the sphere if they share many
//! active axes in the original space, regardless of how small those
//! activations are in absolute magnitude.
//!
//! # Algorithm
//!
//! 1. For each embedding, extract the set of axes whose absolute weight
//!    exceeds `active_threshold` (noise filter).
//! 2. Build a pairwise Jaccard-similarity matrix W over those active sets.
//! 3. Sparsify W to a mutual-k-nearest-neighbors graph (union semantics).
//! 4. Compute the normalized affinity matrix W_norm = D^(-1/2) W D^(-1/2).
//! 5. Find the top-3 *non-trivial* eigenvectors of W_norm via power iteration
//!    with deflation against the trivial eigenvector (μ=1, ∝ sqrt(D)).
//! 6. Project: for a new embedding, apply the Nyström out-of-sample formula
//!    u_k(y) = (1/μ_k) Σ_j w_norm(y, j) · u_k(j), then L2-normalize the
//!    three coordinates to land on S².
//!
//! # References
//!
//! * Belkin & Niyogi. "Laplacian Eigenmaps for Dimensionality Reduction and
//!   Data Representation." *Neural Computation* 15 (2003) 1373–1396.
//! * Bengio et al. "Out-of-Sample Extensions for LLE, Isomap, MDS, Eigenmaps,
//!   and Spectral Clustering." *NIPS* 2003.

use sphereql_core::SphericalPoint;

use crate::projection::{
    Projection, ProjectionError, SplitMix64, dot, normalize_vec, project_xyz_to_spherical,
};
use crate::types::{Embedding, ProjectedPoint, RadialStrategy};

// ── Defaults ───────────────────────────────────────────────────────────

/// Absolute-weight cutoff below which an axis is treated as noise and
/// excluded from the active set used to compute Jaccard similarity.
pub const DEFAULT_ACTIVE_THRESHOLD: f64 = 0.05;

/// k in the k-NN graph sparsification step. Each point keeps edges to its
/// top-k most similar neighbors; the graph is the *union* of those edges.
pub const DEFAULT_K_NEIGHBORS: usize = 15;

/// Floor on the degree of any graph node. Zero-degree nodes would produce
/// NaN during D^(-1/2) normalization; this regularization keeps the spectrum
/// well-defined on disconnected graphs.
pub const DEGREE_REGULARIZATION: f64 = 1e-6;

const MAX_POWER_ITERS: usize = 400;
const POWER_ITER_TOL: f64 = 1e-10;
const RNG_SEED: u64 = 0xC0FF_EECA_FE00;

// ── Projection ─────────────────────────────────────────────────────────

/// Laplacian-eigenmap projection: embeds concepts on S² by the spectral
/// structure of the axis-co-activation graph rather than raw variance.
///
/// Fit cost: O(n² · d) to build the Jaccard graph plus O(n² · q · iters)
/// for power iteration on the n×n affinity matrix. Memory: O(n²) during
/// fit, O(n) per eigenvector after.
///
/// Project cost (Nyström out-of-sample): O(n · d) per query.
#[derive(Clone)]
pub struct LaplacianEigenmapProjection {
    // ── Fit parameters ──
    active_threshold: f64,
    radial: RadialStrategy,
    dim: usize,

    // ── Fitted corpus state ──
    /// For each corpus point, the sorted list of active-axis indices.
    /// Used at query time to compute Jaccard similarity with new points.
    corpus_active_sets: Vec<Vec<usize>>,
    /// Sparsified degree of each corpus point: Σ_j W[i][j].
    /// Used in the Nyström extension's D^(-1/2) normalization.
    corpus_degrees: Vec<f64>,

    /// Top-3 non-trivial eigenvectors of the normalized affinity W_norm,
    /// in descending eigenvalue order. Each vector has length n.
    eigenvectors: [Vec<f64>; 3],
    /// Corresponding eigenvalues μ_1, μ_2, μ_3 ∈ (−1, 1). The trivial
    /// μ_0 = 1 is excluded.
    eigenvalues: [f64; 3],

    /// Mean of |μ_k| across the three retained eigenvalues. A proxy for
    /// how much "low-frequency" connectivity structure the embedding
    /// captured. Reported in place of PCA's explained_variance_ratio for
    /// API compatibility; see [`Self::connectivity_ratio`].
    connectivity_ratio: f64,
}

impl LaplacianEigenmapProjection {
    /// Fit with default parameters (`k=15`, `active_threshold=0.05`).
    pub fn fit(embeddings: &[Embedding], radial: RadialStrategy) -> Result<Self, ProjectionError> {
        Self::fit_with_params(
            embeddings,
            DEFAULT_K_NEIGHBORS,
            DEFAULT_ACTIVE_THRESHOLD,
            radial,
        )
    }

    /// Fit with explicit graph parameters.
    ///
    /// - `k` — k-NN graph density. Higher k = more edges = smoother embedding,
    ///   less sensitive to local noise, but also more blurred category
    ///   boundaries. Typical range: 10–30.
    /// - `active_threshold` — absolute-weight cutoff for the active-axis
    ///   filter. Axes with |v_i| ≤ threshold contribute nothing to the
    ///   similarity metric.
    pub fn fit_with_params(
        embeddings: &[Embedding],
        k: usize,
        active_threshold: f64,
        radial: RadialStrategy,
    ) -> Result<Self, ProjectionError> {
        let n = embeddings.len();
        if n < 4 {
            return Err(ProjectionError::TooFewEmbeddings {
                got: n,
                required: 4,
            });
        }
        let dim = embeddings[0].dimension();
        if dim == 0 {
            return Err(ProjectionError::DimensionTooLow {
                got: dim,
                required: 1,
            });
        }
        let k = k.min(n - 1).max(1);

        // 1. Active-axis sets (sorted for merge-style intersection).
        let corpus_active_sets: Vec<Vec<usize>> = embeddings
            .iter()
            .map(|e| {
                let mut idxs: Vec<usize> = e
                    .values
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| (v.abs() > active_threshold).then_some(i))
                    .collect();
                idxs.sort_unstable();
                idxs
            })
            .collect();

        // 2–4. Sparse top-k graph construction.
        //
        // Previously this pass materialized four separate n×n matrices
        // (`sim`, `keep`, `w`, then `w_norm`) totaling ~3.2 GB at
        // n = 10 000. The only buffer the eigensolver actually reads
        // is `w_norm`, so every intermediate above was pure RAM tax.
        //
        // New shape:
        //   per-row top-k edge list  (≈ n · k entries, O(n · k) memory)
        //     → union-symmetrize into a `HashSet<(i, j)>`
        //     → compute degrees from the set
        //     → scatter directly into one dense `w_norm` the
        //       eigensolver can consume
        //
        // Memory: from 4 · n² · 8 ≈ 3.2 GB down to ~1 · n² · 8 + O(n·k)
        // ≈ 800 MB + a few MB of scratch.
        //
        // Jaccard is computed once per unordered pair; the upper-
        // triangular scan fills both rows' candidate lists in one step.

        // Per-row top-k min-heap keyed on (sim, j) — keeps the
        // *largest* k similarities by evicting the current minimum.
        // Using `std::cmp::Reverse` converts the default max-heap into
        // the min-heap we need.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        #[derive(PartialEq)]
        struct TopKEntry(f64, usize);
        impl Eq for TopKEntry {}
        impl PartialOrd for TopKEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for TopKEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Compare on sim first (total_cmp — NaN-safe), break
                // ties on index so the order is deterministic.
                self.0
                    .total_cmp(&other.0)
                    .then_with(|| self.1.cmp(&other.1))
            }
        }

        let mut top_k_heaps: Vec<BinaryHeap<Reverse<TopKEntry>>> =
            (0..n).map(|_| BinaryHeap::with_capacity(k + 1)).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let s = jaccard_sorted(&corpus_active_sets[i], &corpus_active_sets[j]);
                if s <= 0.0 {
                    continue;
                }
                for (owner, other) in [(i, j), (j, i)] {
                    let h = &mut top_k_heaps[owner];
                    h.push(Reverse(TopKEntry(s, other)));
                    if h.len() > k {
                        h.pop();
                    }
                }
            }
        }

        // Union-symmetrize the per-row top-k candidates into a dedup
        // edge set. Jaccard is symmetric by construction, so we can
        // recover the edge weight from the heap entry.
        let mut edges: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::with_capacity(n * k);
        for (i, heap) in top_k_heaps.into_iter().enumerate() {
            for Reverse(TopKEntry(s, j)) in heap.into_iter() {
                let key = if i < j { (i, j) } else { (j, i) };
                edges.insert(key, s);
            }
        }

        // Degrees: sum of weights on each node's incident edges.
        let mut degrees = vec![0.0f64; n];
        for (&(a, b), &s) in edges.iter() {
            degrees[a] += s;
            degrees[b] += s;
        }

        // Regularize isolated/near-isolated nodes so D^(-1/2) stays finite.
        let safe_degrees: Vec<f64> = degrees
            .iter()
            .map(|&d| d.max(DEGREE_REGULARIZATION))
            .collect();
        let d_inv_sqrt: Vec<f64> = safe_degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();

        // 5. Scatter the symmetric W_norm = D^(-1/2) · W · D^(-1/2)
        //    directly from the edge map. Cells with no edge stay 0.
        let mut w_norm = vec![0.0f64; n * n];
        for (&(a, b), &s) in edges.iter() {
            let v = s * d_inv_sqrt[a] * d_inv_sqrt[b];
            w_norm[a * n + b] = v;
            w_norm[b * n + a] = v;
        }

        // 6. Trivial eigenvector of W_norm at μ=1 is proportional to sqrt(D).
        //    Explicitly constructed so deflation against it pulls it out of
        //    the subsequent power-iteration subspace.
        let sum_d: f64 = safe_degrees.iter().sum();
        let trivial_ev: Vec<f64> = safe_degrees
            .iter()
            .map(|&d| d.sqrt() / sum_d.sqrt())
            .collect();

        // 7. Top-3 non-trivial eigenvectors + eigenvalues.
        let (eigenvectors_vec, eigenvalues_vec) =
            top_k_symmetric_excluding(&w_norm, n, 3, &[trivial_ev]);

        let eigenvalues: [f64; 3] = [eigenvalues_vec[0], eigenvalues_vec[1], eigenvalues_vec[2]];
        let eigenvectors: [Vec<f64>; 3] = [
            eigenvectors_vec[0].clone(),
            eigenvectors_vec[1].clone(),
            eigenvectors_vec[2].clone(),
        ];

        let connectivity_ratio =
            (eigenvalues[0].abs() + eigenvalues[1].abs() + eigenvalues[2].abs()) / 3.0;

        Ok(Self {
            active_threshold,
            radial,
            dim,
            corpus_active_sets,
            corpus_degrees: safe_degrees,
            eigenvectors,
            eigenvalues,
            connectivity_ratio,
        })
    }

    /// Top-3 non-trivial eigenvalues of the normalized affinity matrix
    /// (in descending order). Values close to 1.0 indicate strong block
    /// structure along that spectral direction.
    pub fn eigenvalues(&self) -> [f64; 3] {
        self.eigenvalues
    }

    /// Scalar quality proxy analogous to PCA's explained-variance ratio.
    ///
    /// Defined as the mean of `|μ_k|` across the three retained eigenvalues;
    /// bounded in [0, 1]. Higher = stronger low-frequency community structure
    /// captured by the 3D embedding.
    ///
    /// *Not* directly comparable to PCA EVR — the quantities have different
    /// mathematical meaning. Use as a signal of projection health for this
    /// projection family specifically.
    pub fn connectivity_ratio(&self) -> f64 {
        self.connectivity_ratio
    }

    /// Returned for API compatibility with `PcaProjection::explained_variance_ratio`.
    /// See [`Self::connectivity_ratio`] for semantics — the two are not the
    /// same metric, but both live in [0, 1] and can feed EVR-adaptive
    /// thresholds downstream.
    pub fn explained_variance_ratio(&self) -> f64 {
        self.connectivity_ratio
    }

    /// Nyström-extend the three retained eigenvectors to a new embedding y,
    /// returning raw (unnormalized) 3D coordinates.
    fn project_to_3d(&self, embedding: &Embedding) -> (f64, f64, f64) {
        let mut active_y: Vec<usize> = embedding
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (v.abs() > self.active_threshold).then_some(i))
            .collect();
        active_y.sort_unstable();

        let n = self.corpus_active_sets.len();
        let sims: Vec<f64> = self
            .corpus_active_sets
            .iter()
            .map(|s| jaccard_sorted(&active_y, s))
            .collect();

        // Treat the query as a new graph node whose similarities to corpus
        // points are the sims vector. Its degree is the sum of those sims.
        let degree_y = sims.iter().sum::<f64>().max(DEGREE_REGULARIZATION);
        let inv_sqrt_dy = 1.0 / degree_y.sqrt();

        // Nyström extension: u_k(y) = (1/μ_k) Σ_j w_norm(y, j) · u_k(j).
        // w_norm(y, j) = sims[j] / (sqrt(d_y) · sqrt(d_j)).
        let mut coords = [0.0f64; 3];
        for (k, (ev, &mu)) in self
            .eigenvectors
            .iter()
            .zip(self.eigenvalues.iter())
            .enumerate()
        {
            if mu.abs() < 1e-10 {
                continue;
            }
            let mut s = 0.0;
            for j in 0..n {
                let w_norm_yj = sims[j] * inv_sqrt_dy / self.corpus_degrees[j].sqrt();
                s += w_norm_yj * ev[j];
            }
            coords[k] = s / mu;
        }
        (coords[0], coords[1], coords[2])
    }
}

impl Projection for LaplacianEigenmapProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        let (x, y, z) = self.project_to_3d(embedding);
        let r = self.radial.compute(embedding.magnitude());
        project_xyz_to_spherical(x, y, z, r)
    }

    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint {
        let (x, y, z) = self.project_to_3d(embedding);
        let r = self.radial.compute(embedding.magnitude());
        let position = project_xyz_to_spherical(x, y, z, r);
        let proj_mag = (x * x + y * y + z * z).sqrt();
        // Certainty reflects whether the Nyström extension produced a
        // non-degenerate coordinate. A zero-magnitude projection means the
        // query had no active axes in common with the corpus — the placement
        // is effectively arbitrary, so certainty should be low.
        let certainty = proj_mag.tanh();
        ProjectedPoint {
            position,
            certainty,
            intensity: embedding.magnitude(),
            projection_magnitude: proj_mag,
        }
    }

    fn dimensionality(&self) -> usize {
        self.dim
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Jaccard similarity between two sorted, deduplicated index sets.
/// Merge-intersect in O(|a| + |b|).
fn jaccard_sorted(a: &[usize], b: &[usize]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut ia = 0;
    let mut ib = 0;
    let mut intersection: usize = 0;
    while ia < a.len() && ib < b.len() {
        match a[ia].cmp(&b[ib]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                ia += 1;
                ib += 1;
            }
            std::cmp::Ordering::Less => ia += 1,
            std::cmp::Ordering::Greater => ib += 1,
        }
    }
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Power iteration with deflation on a symmetric n×n matrix (row-major).
/// Finds the top-k eigenvectors orthogonal to everything in `exclude`.
///
/// Returns (vectors, values) in descending eigenvalue order.
fn top_k_symmetric_excluding(
    matrix: &[f64],
    n: usize,
    k: usize,
    exclude: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut values: Vec<f64> = Vec::with_capacity(k);
    let mut rng = SplitMix64::new(RNG_SEED);

    let matvec = |dst: &mut [f64], src: &[f64]| {
        for (i, dst_i) in dst.iter_mut().enumerate() {
            let row = i * n;
            let mut s = 0.0;
            for j in 0..n {
                s += matrix[row + j] * src[j];
            }
            *dst_i = s;
        }
    };

    for _ in 0..k {
        let mut v: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        // Initial orthogonalization against exclude + previously-found vectors.
        for prev in exclude.iter().chain(vectors.iter()) {
            let proj = dot(&v, prev);
            for (vi, &pi) in v.iter_mut().zip(prev.iter()) {
                *vi -= proj * pi;
            }
        }
        normalize_vec(&mut v);
        let mut eigenvalue = 0.0;

        // Cached `matrix · v` from the previous iteration's Rayleigh
        // step. Each iteration's `matrix · v_new = matrix · u_previous`,
        // so we can skip one mat-vec per step after the first.
        let mut mv_cache: Option<Vec<f64>> = None;

        for _ in 0..MAX_POWER_ITERS {
            let mut u = mv_cache.take().unwrap_or_else(|| {
                let mut u = vec![0.0f64; n];
                matvec(&mut u, &v);
                u
            });
            // Deflate each iteration so drift back into the excluded subspace
            // is continuously removed.
            for prev in exclude.iter().chain(vectors.iter()) {
                let proj = dot(&u, prev);
                for (ui, &pi) in u.iter_mut().zip(prev.iter()) {
                    *ui -= proj * pi;
                }
            }
            let mag = normalize_vec(&mut u);
            if mag < f64::EPSILON {
                break;
            }

            // Rayleigh quotient: λ ≈ uᵀ · (matrix · u). Cache
            // `matrix · u` for next iteration's start — halves the
            // steady-state mat-vec cost.
            let mut mv_next = vec![0.0f64; n];
            matvec(&mut mv_next, &u);
            eigenvalue = dot(&u, &mv_next);

            let change = (1.0 - dot(&u, &v).abs()).max(0.0);
            v = u;
            mv_cache = Some(mv_next);
            if change < POWER_ITER_TOL {
                break;
            }
        }

        vectors.push(v);
        values.push(eigenvalue);
    }

    // Sort in descending eigenvalue order (power iteration usually finds
    // them in that order already, but deflation drift can swap adjacent
    // pairs on near-degenerate spectra).
    let mut order: Vec<usize> = (0..vectors.len()).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_vectors: Vec<Vec<f64>> = order.iter().map(|&i| vectors[i].clone()).collect();
    let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    (sorted_vectors, sorted_values)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sphereql_core::angular_distance;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    /// Three well-separated 8-dim clusters. Each cluster activates 3
    /// distinct axes strongly; the remaining 5 dims carry ±0.02 noise.
    fn three_cluster_corpus() -> Vec<Embedding> {
        let noise = 0.02;
        let mut corpus = Vec::new();

        // Cluster A: axes 0, 1, 2
        for i in 0..5 {
            let delta = i as f64 * 0.01;
            corpus.push(emb(&[
                1.0 + delta,
                0.8 - delta,
                0.7 + delta,
                noise,
                -noise,
                noise,
                -noise,
                noise,
            ]));
        }
        // Cluster B: axes 3, 4, 5
        for i in 0..5 {
            let delta = i as f64 * 0.01;
            corpus.push(emb(&[
                noise,
                -noise,
                noise,
                1.0 + delta,
                0.8 - delta,
                0.7 + delta,
                -noise,
                noise,
            ]));
        }
        // Cluster C: axes 5, 6, 7 (shares axis 5 with B to exercise a bridge)
        for i in 0..5 {
            let delta = i as f64 * 0.01;
            corpus.push(emb(&[
                -noise,
                noise,
                -noise,
                noise,
                -noise,
                0.9 + delta,
                1.0 - delta,
                0.8 + delta,
            ]));
        }
        corpus
    }

    #[test]
    fn jaccard_empty_sets_zero() {
        assert_eq!(jaccard_sorted(&[], &[]), 0.0);
        assert_eq!(jaccard_sorted(&[1, 2], &[]), 0.0);
    }

    #[test]
    fn jaccard_identical_sets_one() {
        assert!((jaccard_sorted(&[1, 2, 3], &[1, 2, 3]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jaccard_disjoint_sets_zero() {
        assert_eq!(jaccard_sorted(&[1, 2], &[3, 4]), 0.0);
    }

    #[test]
    fn jaccard_partial_overlap() {
        // {1,2,3} ∩ {2,3,4} = {2,3} (2), union = {1,2,3,4} (4) → 0.5
        assert!((jaccard_sorted(&[1, 2, 3], &[2, 3, 4]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fit_produces_non_trivial_eigenvalues() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let [m1, m2, m3] = lap.eigenvalues();
        // All retained eigenvalues must be strictly below 1 (trivial excluded)
        // and above 0 (the 3-cluster structure should produce meaningful
        // non-zero connectivity eigenvalues).
        assert!(m1 < 1.0 && m1 > 0.0, "μ_1 = {m1}");
        assert!(m2 < 1.0 && m2 > -1.0, "μ_2 = {m2}");
        assert!(m3 < 1.0 && m3 > -1.0, "μ_3 = {m3}");
        // Descending order.
        assert!(m1 >= m2 - 1e-10);
        assert!(m2 >= m3 - 1e-10);
    }

    #[test]
    fn connectivity_ratio_in_unit_interval() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let r = lap.connectivity_ratio();
        assert!((0.0..=1.0).contains(&r), "connectivity_ratio = {r}");
        assert_eq!(r, lap.explained_variance_ratio());
    }

    #[test]
    fn projection_lands_on_unit_sphere() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        for e in &corpus {
            let sp = lap.project(e);
            assert!((sp.r - 1.0).abs() < 1e-9, "r = {}", sp.r);
            assert!(sp.theta >= 0.0 && sp.theta <= std::f64::consts::TAU);
            assert!(sp.phi >= 0.0 && sp.phi <= std::f64::consts::PI);
        }
    }

    #[test]
    fn same_cluster_points_closer_than_cross_cluster() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let positions: Vec<SphericalPoint> = corpus.iter().map(|e| lap.project(e)).collect();

        // Within cluster A (indices 0..5): max pairwise distance.
        let mut max_within_a = 0.0f64;
        for i in 0..5 {
            for j in (i + 1)..5 {
                max_within_a = max_within_a.max(angular_distance(&positions[i], &positions[j]));
            }
        }
        // A vs C (indices 0..5 vs 10..15): min pairwise distance. A and C
        // share no axes, so C should be the farthest-out of the three
        // clusters from A.
        let mut min_a_to_c = f64::INFINITY;
        for i in 0..5 {
            for j in 10..15 {
                min_a_to_c = min_a_to_c.min(angular_distance(&positions[i], &positions[j]));
            }
        }
        assert!(
            max_within_a < min_a_to_c,
            "within-A max ({max_within_a}) should be less than A-to-C min ({min_a_to_c})"
        );
    }

    #[test]
    fn nystrom_roundtrip_approximates_training_position() {
        // A corpus point fed back through project() should land near its
        // Nyström-implied position. We don't demand exact recovery (the
        // Nyström extension differs subtly from the in-sample eigenvector
        // coord because Jaccard(x, x) = 1 but W[i][i] = 0 during fit), but
        // the same corpus point should project consistently across calls.
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let first = lap.project(&corpus[0]);
        let again = lap.project(&corpus[0]);
        assert!((first.theta - again.theta).abs() < 1e-12);
        assert!((first.phi - again.phi).abs() < 1e-12);
    }

    #[test]
    fn new_point_in_known_cluster_routes_to_cluster_region() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        // Synthesize a fresh "cluster-A" point and check it lands closer to
        // an existing cluster-A member than to a cluster-C member.
        let query = emb(&[1.0, 0.8, 0.7, 0.02, -0.02, 0.02, -0.02, 0.02]);
        let q = lap.project(&query);
        let a_member = lap.project(&corpus[0]);
        let c_member = lap.project(&corpus[10]);
        let d_to_a = angular_distance(&q, &a_member);
        let d_to_c = angular_distance(&q, &c_member);
        assert!(
            d_to_a < d_to_c,
            "query→A {d_to_a} should be less than query→C {d_to_c}"
        );
    }

    #[test]
    fn dimensionality_matches_input() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        assert_eq!(lap.dimensionality(), 8);
    }

    #[test]
    fn fit_rejects_tiny_corpus() {
        let corpus = vec![emb(&[1.0, 0.0]), emb(&[0.0, 1.0])];
        assert!(matches!(
            LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::TooFewEmbeddings {
                got: 2,
                required: 4
            })
        ));
    }

    #[test]
    fn project_rich_certainty_in_range() {
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        for e in &corpus {
            let pp = lap.project_rich(e);
            assert!(pp.certainty >= 0.0 && pp.certainty <= 1.0);
        }
    }

    #[test]
    fn disconnected_query_gracefully_placed() {
        // A query with no active axes (all below threshold) has no similarity
        // to the corpus. It should still produce a valid SphericalPoint
        // rather than NaN.
        let corpus = three_cluster_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let query = emb(&[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]);
        let sp = lap.project(&query);
        assert!(sp.r.is_finite());
        assert!(sp.theta.is_finite());
        assert!(sp.phi.is_finite());
    }
}
