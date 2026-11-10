//! Laplacian eigenmap projection — connectivity-preserving embedding on S².
//!
//! PCA and kernel PCA maximize *variance*. A Laplacian eigenmap preserves
//! *connectivity* instead: each item is linked to its nearest neighbours,
//! and the layout is the one in which linked items sit closest together.
//!
//! # Algorithm
//!
//! 1. Link each item to its `k` most similar items and take the union of
//!    those links. Similarity is one of two things
//!    ([`LaplacianSimilarity`]):
//!    - **Cosine** (the default) between the embeddings.
//!    - **Active axes**: the Jaccard overlap of the sets of axes whose
//!      absolute weight exceeds `active_threshold`. This suits sparse
//!      embeddings whose signal is *which* axes fire. On dense embeddings
//!      every axis is active for every item and the graph carries no
//!      information.
//! 2. Form the normalized affinity `S = D^(-1/2) W D^(-1/2)`.
//! 3. Take the three eigenvectors of `S` with the largest eigenvalues
//!    below the trivial one (μ = 1, ∝ sqrt(D)). These are the three
//!    smallest non-trivial eigenvectors of the graph Laplacian `I − S`.
//! 4. A fitted item sits at its row of those eigenvectors, normalized
//!    onto S². Any other embedding is placed by the Nyström formula
//!    `u_k(y) = (1/μ_k) Σ_j s(y, j) · u_k(j)` over its `k` nearest
//!    fitted items, the same sparsification the fit used.
//!
//! # References
//!
//! * Belkin & Niyogi. "Laplacian Eigenmaps for Dimensionality Reduction and
//!   Data Representation." *Neural Computation* 15 (2003) 1373–1396.
//! * Bengio et al. "Out-of-Sample Extensions for LLE, Isomap, MDS, Eigenmaps,
//!   and Spectral Clustering." *NIPS* 2003.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use sphereql_core::SphericalPoint;

use crate::ann::{AnnConfig, AnnIndex};
use crate::config::LaplacianSimilarity;
use crate::projection::{
    Projection, ProjectionError, SplitMix64, dot, normalize_vec, project_xyz_to_spherical,
};
use crate::types::{Embedding, ProjectedPoint, RadialStrategy};

// ── Defaults ───────────────────────────────────────────────────────────

/// Absolute-weight cutoff below which an axis is treated as noise and
/// left out of an item's active set ([`LaplacianSimilarity::ActiveAxes`]).
pub const DEFAULT_ACTIVE_THRESHOLD: f64 = 0.05;

/// k in the k-NN graph. Each item links to its k most similar items; the
/// graph is the *union* of those links.
pub const DEFAULT_K_NEIGHBORS: usize = 15;

/// Floor on the degree of any graph node. A zero degree would make
/// `D^(-1/2)` infinite; this keeps the spectrum defined when an item has
/// no neighbour.
pub(crate) const DEGREE_REGULARIZATION: f64 = 1e-6;

/// At or above this many items the cosine graph comes from the RP-forest
/// index instead of an exact all-pairs scan (the same cut-over UMAP uses).
const ANN_THRESHOLD: usize = 2000;

/// Eigenvectors kept as coordinates.
const KEPT: usize = 3;
/// Extra vectors carried through the iteration. The block converges at
/// the rate of the gap to the first eigenvalue *outside* it, so a few
/// spare vectors make the kept three converge much faster.
const GUARD: usize = 3;
const MAX_ITERS: usize = 2000;
/// Iterations between convergence checks.
const CHECK_EVERY: usize = 10;
/// Stop once `‖S·v − μ·v‖` is below this for each kept vector.
const RESIDUAL_TOL: f64 = 1e-8;
const RNG_SEED: u64 = 0xC0FF_EECA_FE00;

// ── Projection ─────────────────────────────────────────────────────────

/// Laplacian-eigenmap projection: places items on S² by the spectral
/// structure of their nearest-neighbour graph.
///
/// Fit cost is the graph build (cosine: O(n²·d) exact below 2,000 items,
/// RP-forest above; active axes: O(n²·a) for `a` active axes per item)
/// plus O(n·k) per solver iteration. Memory is O(n·k) for the graph, and
/// the fitted corpus the queries are compared against: O(n·d) for
/// cosine, O(n·a) for active axes.
///
/// Projecting an embedding that was not fitted costs one nearest-neighbour
/// lookup against the corpus.
#[derive(Clone)]
pub struct LaplacianEigenmapProjection {
    corpus: Corpus,
    k: usize,
    radial: RadialStrategy,
    dim: usize,
    /// Degree of each fitted item in the symmetrized graph.
    degrees: Vec<f64>,
    /// The three kept eigenvectors of the normalized affinity, in
    /// descending eigenvalue order. Each has one entry per fitted item.
    eigenvectors: [Vec<f64>; 3],
    /// Their eigenvalues μ_1 ≥ μ_2 ≥ μ_3, each in [−1, 1].
    eigenvalues: [f64; 3],
    /// Fitted items by a hash of the vector they were fitted from, so
    /// projecting one returns its own row instead of a Nyström estimate.
    exact: HashMap<u64, Vec<u32>>,
    /// A second, independent hash per fitted item, checked on lookup.
    fingerprints: Vec<u64>,
    /// See [`Self::explained_variance_ratio`].
    neighbor_recall: f64,
}

/// What a query is compared against, per similarity.
#[derive(Clone)]
enum Corpus {
    Cosine {
        /// Unit vectors, when the corpus is small enough to scan.
        normalized: Vec<Vec<f64>>,
        /// The index that holds them otherwise.
        ann: Option<Arc<AnnIndex>>,
    },
    ActiveAxes {
        threshold: f64,
        /// Sorted active-axis indices of each fitted item.
        sets: Vec<Vec<usize>>,
    },
}

impl LaplacianEigenmapProjection {
    /// Fit on the cosine k-NN graph with `k = 15`.
    pub fn fit(embeddings: &[Embedding], radial: RadialStrategy) -> Result<Self, ProjectionError> {
        Self::fit_cosine(embeddings, DEFAULT_K_NEIGHBORS, radial)
    }

    /// Fit on the cosine k-NN graph: each item links to the `k` items
    /// whose embeddings point most nearly the same way.
    ///
    /// Higher `k` gives a smoother layout with blurrier boundaries
    /// between groups. Typical range: 10–30.
    pub fn fit_cosine(
        embeddings: &[Embedding],
        k: usize,
        radial: RadialStrategy,
    ) -> Result<Self, ProjectionError> {
        let (dim, k) = validate(embeddings, k)?;
        let normalized: Vec<Vec<f64>> = embeddings.iter().map(|e| e.normalized()).collect();
        let keys: Vec<(u64, u64)> = normalized.iter().map(|v| hashes(v)).collect();
        let (lists, corpus) = if normalized.len() < ANN_THRESHOLD {
            let lists = (0..normalized.len())
                .into_par_iter()
                .map(|i| {
                    let own = &normalized[i];
                    top_k(
                        normalized
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(j, v)| (j, dot(own, v))),
                        k,
                    )
                })
                .collect();
            let corpus = Corpus::Cosine {
                normalized,
                ann: None,
            };
            (lists, corpus)
        } else {
            let index = Arc::new(AnnIndex::build_normalized(
                normalized,
                &AnnConfig::default(),
            ));
            let lists = index
                .knn_graph_with_sims(k)
                .into_iter()
                .map(|row| row.into_iter().filter(|&(_, s)| s > 0.0).collect())
                .collect();
            let corpus = Corpus::Cosine {
                normalized: Vec::new(),
                ann: Some(index),
            };
            (lists, corpus)
        };
        Ok(Self::from_graph(corpus, lists, keys, k, dim, radial))
    }

    /// Fit on the active-axis graph: each item links to the `k` items
    /// whose sets of active axes (`|v_i| > active_threshold`) overlap its
    /// own the most, by Jaccard similarity.
    ///
    /// For sparse embeddings. The threshold has to sit above the noise
    /// floor and below the signal: too low and every axis is active, so
    /// all items look alike; too high and none is.
    pub fn fit_with_params(
        embeddings: &[Embedding],
        k: usize,
        active_threshold: f64,
        radial: RadialStrategy,
    ) -> Result<Self, ProjectionError> {
        let (dim, k) = validate(embeddings, k)?;
        let sets: Vec<Vec<usize>> = embeddings
            .iter()
            .map(|e| active_axes(&e.values, active_threshold))
            .collect();
        let keys: Vec<(u64, u64)> = embeddings.iter().map(|e| hashes(&e.values)).collect();
        let lists = (0..sets.len())
            .into_par_iter()
            .map(|i| {
                let own = &sets[i];
                top_k(
                    sets.iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(j, other)| (j, jaccard_sorted(own, other))),
                    k,
                )
            })
            .collect();
        let corpus = Corpus::ActiveAxes {
            threshold: active_threshold,
            sets,
        };
        Ok(Self::from_graph(corpus, lists, keys, k, dim, radial))
    }

    /// Shared tail of the fits: symmetrize the neighbour lists, solve for
    /// the eigenvectors, index the fitted items.
    fn from_graph(
        corpus: Corpus,
        lists: Vec<Vec<(usize, f64)>>,
        keys: Vec<(u64, u64)>,
        k: usize,
        dim: usize,
        radial: RadialStrategy,
    ) -> Self {
        let graph = Graph::from_neighbor_lists(&lists);
        let (vectors, eigenvalues) = graph.top_nontrivial_eigenvectors();
        let mut vectors = vectors.into_iter();
        let eigenvectors: [Vec<f64>; 3] =
            std::array::from_fn(|_| vectors.next().expect("solver returns three vectors"));

        let directions: Vec<[f64; 3]> = (0..graph.len())
            .map(|i| unit([eigenvectors[0][i], eigenvectors[1][i], eigenvectors[2][i]]))
            .collect();
        let neighbors: Vec<Vec<usize>> = lists
            .iter()
            .map(|row| row.iter().map(|&(j, _)| j).collect())
            .collect();
        let neighbor_recall = crate::umap::knn_recall_score(&directions, &neighbors);

        let mut exact: HashMap<u64, Vec<u32>> = HashMap::new();
        for (i, &(key, _)) in keys.iter().enumerate() {
            exact.entry(key).or_default().push(i as u32);
        }

        Self {
            corpus,
            k,
            radial,
            dim,
            degrees: graph.degrees,
            eigenvectors,
            eigenvalues,
            exact,
            fingerprints: keys.into_iter().map(|(_, check)| check).collect(),
            neighbor_recall,
        }
    }

    /// Which similarity the neighbour graph was built from.
    pub fn similarity(&self) -> LaplacianSimilarity {
        match self.corpus {
            Corpus::Cosine { .. } => LaplacianSimilarity::Cosine,
            Corpus::ActiveAxes { .. } => LaplacianSimilarity::ActiveAxes,
        }
    }

    /// The three kept eigenvalues of the normalized affinity, largest
    /// first. Values close to 1.0 mean the graph nearly splits into
    /// separate groups along that direction.
    pub fn eigenvalues(&self) -> [f64; 3] {
        self.eigenvalues
    }

    /// Mean of `|μ_k|` over the three kept eigenvalues, in [0, 1]. High
    /// when the graph has strong group structure.
    ///
    /// This describes the graph, not how well the layout preserves it: a
    /// graph of well-separated groups scores near 1 however the items
    /// inside each group are arranged. For that, see
    /// [`Self::explained_variance_ratio`].
    pub fn connectivity_ratio(&self) -> f64 {
        self.eigenvalues.iter().map(|m| m.abs()).sum::<f64>() / 3.0
    }

    /// Neighbour recall, under the name the other projections use for
    /// their quality score: the mean share of an item's `k` graph
    /// neighbours that are also among its `k` nearest on the sphere. 1.0
    /// means every neighbourhood survived; random placement scores about
    /// `k/n`. Same definition as
    /// [`UmapSphereProjection::explained_variance_ratio`](crate::umap::UmapSphereProjection::explained_variance_ratio),
    /// so the two compare directly. Not a variance ratio.
    pub fn explained_variance_ratio(&self) -> f64 {
        self.neighbor_recall
    }

    /// Row of the kept eigenvectors for an embedding: its own row if it
    /// was fitted, otherwise the Nyström estimate from its `k` nearest
    /// fitted items.
    fn project_to_3d(&self, embedding: &Embedding) -> (f64, f64, f64) {
        let neighbors = match &self.corpus {
            Corpus::Cosine { normalized, ann } => {
                let unit = embedding.normalized();
                if let Some(i) = self.fitted_index(hashes(&unit)) {
                    return self.fitted_row(i);
                }
                match ann {
                    Some(index) => index.query(&unit, self.k),
                    None => top_k(
                        normalized
                            .iter()
                            .enumerate()
                            .map(|(j, v)| (j, dot(&unit, v))),
                        self.k,
                    ),
                }
            }
            Corpus::ActiveAxes { threshold, sets } => {
                if let Some(i) = self.fitted_index(hashes(&embedding.values)) {
                    return self.fitted_row(i);
                }
                let own = active_axes(&embedding.values, *threshold);
                top_k(
                    sets.iter()
                        .enumerate()
                        .map(|(j, other)| (j, jaccard_sorted(&own, other))),
                    self.k,
                )
            }
        };

        // The query joins the graph as a node linked to those neighbours.
        let degree: f64 = neighbors
            .iter()
            .filter(|&&(_, s)| s > 0.0)
            .map(|&(_, s)| s)
            .sum::<f64>()
            .max(DEGREE_REGULARIZATION);
        let mut coords = [0.0f64; 3];
        for (c, (vector, &mu)) in coords
            .iter_mut()
            .zip(self.eigenvectors.iter().zip(&self.eigenvalues))
        {
            if mu.abs() < 1e-10 {
                continue;
            }
            let sum: f64 = neighbors
                .iter()
                .filter(|&&(_, s)| s > 0.0)
                .map(|&(j, s)| s / (degree * self.degrees[j]).sqrt() * vector[j])
                .sum();
            *c = sum / mu;
        }
        (coords[0], coords[1], coords[2])
    }

    fn fitted_index(&self, (key, check): (u64, u64)) -> Option<usize> {
        self.exact
            .get(&key)?
            .iter()
            .map(|&i| i as usize)
            .find(|&i| self.fingerprints[i] == check)
    }

    fn fitted_row(&self, i: usize) -> (f64, f64, f64) {
        (
            self.eigenvectors[0][i],
            self.eigenvectors[1][i],
            self.eigenvectors[2][i],
        )
    }
}

impl Projection for LaplacianEigenmapProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        self.project_rich(embedding).position
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
        let (x, y, z) = self.project_to_3d(embedding);
        let proj_mag = (x * x + y * y + z * z).sqrt();
        // A zero-magnitude row means the query shares nothing with the
        // corpus and its placement is arbitrary, so certainty is low.
        let certainty = proj_mag.tanh();
        let intensity = embedding.magnitude();
        let r = self.radial.compute_rich(&crate::types::RadialContext::full(
            intensity, proj_mag, certainty,
        ));
        let position = project_xyz_to_spherical(x, y, z, r);
        ProjectedPoint {
            position,
            certainty,
            intensity,
            projection_magnitude: proj_mag,
        }
    }

    fn dimensionality(&self) -> usize {
        self.dim
    }
}

// ── Graph and solver ───────────────────────────────────────────────────

/// The symmetrized neighbour graph, as the normalized affinity
/// `S = D^(-1/2) W D^(-1/2)` in compressed rows.
struct Graph {
    degrees: Vec<f64>,
    /// `row_start[i]..row_start[i + 1]` indexes row `i` in the two below.
    row_start: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}

impl Graph {
    /// Union of the per-item lists: `i — j` is an edge if either lists the
    /// other. Similarity is symmetric, so both directions carry the same
    /// weight. Rows are sorted by column, which fixes the summation order
    /// and so the result, bit for bit.
    fn from_neighbor_lists(lists: &[Vec<(usize, f64)>]) -> Self {
        let n = lists.len();
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(2 * n);
        for (i, row) in lists.iter().enumerate() {
            for &(j, s) in row {
                if s > 0.0 && i != j {
                    edges.push((i, j, s));
                    edges.push((j, i, s));
                }
            }
        }
        edges.sort_unstable_by_key(|e| (e.0, e.1));
        edges.dedup_by_key(|e| (e.0, e.1));

        let mut degrees = vec![0.0f64; n];
        for &(i, _, s) in &edges {
            degrees[i] += s;
        }
        for d in &mut degrees {
            *d = d.max(DEGREE_REGULARIZATION);
        }

        let mut row_start = vec![0usize; n + 1];
        for &(i, _, _) in &edges {
            row_start[i + 1] += 1;
        }
        for i in 0..n {
            row_start[i + 1] += row_start[i];
        }
        let columns = edges.iter().map(|e| e.1).collect();
        let values = edges
            .iter()
            .map(|&(i, j, s)| s / (degrees[i] * degrees[j]).sqrt())
            .collect();
        Self {
            degrees,
            row_start,
            columns,
            values,
        }
    }

    fn len(&self) -> usize {
        self.degrees.len()
    }

    /// `dst = S · src`.
    fn multiply(&self, dst: &mut [f64], src: &[f64]) {
        for (i, out) in dst.iter_mut().enumerate() {
            let row = self.row_start[i]..self.row_start[i + 1];
            *out = self.columns[row.clone()]
                .iter()
                .zip(&self.values[row])
                .map(|(&j, &w)| w * src[j])
                .sum();
        }
    }

    /// The three eigenvectors of `S` with the largest eigenvalues after
    /// the trivial one, and those eigenvalues, largest first.
    ///
    /// Block power iteration on `(S + I)/2`, whose eigenvalues are those
    /// of `S` moved into [0, 1] in the same order. Iterating on `S` itself
    /// converges to the largest *magnitudes*, and a near-bipartite graph
    /// has eigenvalues near −1 that would then be kept in place of the
    /// connectivity directions. A Rayleigh–Ritz step on the block
    /// separates and orders the vectors.
    fn top_nontrivial_eigenvectors(&self) -> (Vec<Vec<f64>>, [f64; 3]) {
        let n = self.len();
        let total: f64 = self.degrees.iter().sum();
        let trivial: Vec<f64> = self.degrees.iter().map(|d| (d / total).sqrt()).collect();

        // n − 1 directions are orthogonal to the trivial one.
        let block = (KEPT + GUARD).min(n - 1);
        let mut rng = SplitMix64::new(RNG_SEED);
        let mut q: Vec<Vec<f64>> = (0..block)
            .map(|_| (0..n).map(|_| rng.normal()).collect())
            .collect();
        orthonormalize(&mut q, &trivial, &mut rng);

        let mut scratch = vec![0.0f64; n];
        let mut ritz = vec![0.0f64; block];
        for iteration in 1..=MAX_ITERS {
            for v in &mut q {
                self.multiply(&mut scratch, v);
                for (x, &sx) in v.iter_mut().zip(&scratch) {
                    *x = 0.5 * (*x + sx);
                }
            }
            orthonormalize(&mut q, &trivial, &mut rng);

            if iteration % CHECK_EVERY == 0 || iteration == MAX_ITERS {
                let residual;
                (ritz, residual) = self.rayleigh_ritz(&mut q);
                if residual < RESIDUAL_TOL {
                    break;
                }
            }
        }

        let mut vectors = q;
        vectors.resize(KEPT, vec![0.0; n]);
        vectors.truncate(KEPT);
        let eigenvalues = std::array::from_fn(|c| ritz.get(c).copied().unwrap_or(0.0));
        (vectors, eigenvalues)
    }

    /// Replace the orthonormal block `q` by the eigenvectors of `S`
    /// restricted to its span, largest eigenvalue first. Returns those
    /// eigenvalues and the largest `‖S·v − μ·v‖` among the kept vectors.
    fn rayleigh_ritz(&self, q: &mut [Vec<f64>]) -> (Vec<f64>, f64) {
        let b = q.len();
        let n = self.len();
        let mut sq = vec![vec![0.0f64; n]; b];
        for (out, v) in sq.iter_mut().zip(q.iter()) {
            self.multiply(out, v);
        }
        let mut h = vec![vec![0.0f64; b]; b];
        for i in 0..b {
            for j in i..b {
                let value = dot(&q[i], &sq[j]);
                h[i][j] = value;
                h[j][i] = value;
            }
        }
        let (values, rotation) = symmetric_eigen(h);
        let mut order: Vec<usize> = (0..b).collect();
        order.sort_by(|&a, &c| values[c].total_cmp(&values[a]).then(a.cmp(&c)));

        let combine = |basis: &[Vec<f64>], col: usize| {
            let mut v = vec![0.0f64; n];
            for (row, b) in basis.iter().enumerate() {
                let weight = rotation[row][col];
                for (x, &y) in v.iter_mut().zip(b) {
                    *x += weight * y;
                }
            }
            v
        };
        let mut residual = 0.0f64;
        let rotated: Vec<Vec<f64>> = order
            .iter()
            .enumerate()
            .map(|(rank, &col)| {
                let mut v = combine(q, col);
                if rank < KEPT {
                    let sv = combine(&sq, col);
                    let miss: f64 = sv
                        .iter()
                        .zip(&v)
                        .map(|(a, b)| (a - values[col] * b).powi(2))
                        .sum();
                    residual = residual.max(miss.sqrt());
                }
                // Fix the sign, which the eigenproblem leaves free: the
                // entry of largest magnitude is positive.
                let pivot = v
                    .iter()
                    .copied()
                    .fold(0.0f64, |m, x| if x.abs() > m.abs() { x } else { m });
                if pivot < 0.0 {
                    for x in &mut v {
                        *x = -*x;
                    }
                }
                v
            })
            .collect();
        for (dst, src) in q.iter_mut().zip(rotated) {
            *dst = src;
        }
        (order.iter().map(|&i| values[i]).collect(), residual)
    }
}

/// Make `q` orthonormal and orthogonal to `exclude` (modified
/// Gram–Schmidt, two passes). A vector that vanishes is redrawn.
fn orthonormalize(q: &mut [Vec<f64>], exclude: &[f64], rng: &mut SplitMix64) {
    for c in 0..q.len() {
        for attempt in 0..3 {
            let (done, rest) = q.split_at_mut(c);
            let v = &mut rest[0];
            for _ in 0..2 {
                for prev in std::iter::once(exclude).chain(done.iter().map(|p| p.as_slice())) {
                    let along = dot(v, prev);
                    for (x, &p) in v.iter_mut().zip(prev) {
                        *x -= along * p;
                    }
                }
            }
            if normalize_vec(v) > 1e-12 {
                break;
            }
            if attempt == 2 {
                v.fill(0.0);
            } else {
                for x in v.iter_mut() {
                    *x = rng.normal();
                }
            }
        }
    }
}

/// Eigenvalues and eigenvectors (as columns) of a small symmetric matrix,
/// by cyclic Jacobi rotations.
fn symmetric_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                let (row_p, row_q) = (a[p].clone(), a[q].clone());
                for i in 0..n {
                    a[p][i] = c * row_p[i] - s * row_q[i];
                    a[q][i] = s * row_p[i] + c * row_q[i];
                }
                for row in v.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Shared input checks. Returns the embedding dimension and `k` clamped
/// to the corpus.
fn validate(embeddings: &[Embedding], k: usize) -> Result<(usize, usize), ProjectionError> {
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
    for (index, e) in embeddings.iter().enumerate() {
        if e.dimension() != dim {
            return Err(ProjectionError::InconsistentDimension {
                index,
                expected: dim,
                got: e.dimension(),
            });
        }
    }
    Ok((dim, k.min(n - 1).max(1)))
}

/// The `k` candidates with the highest positive similarity, best first.
/// Ties go to the higher index.
fn top_k(candidates: impl Iterator<Item = (usize, f64)>, k: usize) -> Vec<(usize, f64)> {
    let mut kept: Vec<(usize, f64)> = candidates.filter(|&(_, s)| s > 0.0).collect();
    kept.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
    kept.truncate(k);
    kept
}

/// Sorted indices of the axes whose absolute weight exceeds `threshold`.
fn active_axes(values: &[f64], threshold: f64) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (v.abs() > threshold).then_some(i))
        .collect()
}

/// Two independent 64-bit hashes of a vector's bit patterns: a lookup key
/// and a check against collisions on it.
fn hashes(values: &[f64]) -> (u64, u64) {
    let mut key = 0xcbf2_9ce4_8422_2325u64;
    let mut check = 0x9e37_79b9_7f4a_7c15u64;
    for &x in values {
        let bits = x.to_bits();
        key = (key ^ bits).wrapping_mul(0x0000_0100_0000_01b3);
        check = (check.rotate_left(27) ^ bits).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    (key, check)
}

fn unit(p: [f64; 3]) -> [f64; 3] {
    let m = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if m < f64::EPSILON {
        [1.0, 0.0, 0.0]
    } else {
        [p[0] / m, p[1] / m, p[2] / m]
    }
}

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
        let [m1, m2, m3] = lap.eigenvalues();
        assert!((r - (m1.abs() + m2.abs() + m3.abs()) / 3.0).abs() < 1e-12);
    }

    /// `count` unit vectors evenly spaced on a circle. With `k = 2` every
    /// item links to its two neighbours, so the graph is a ring.
    fn ring(count: usize) -> Vec<Embedding> {
        (0..count)
            .map(|i| {
                let angle = std::f64::consts::TAU * i as f64 / count as f64;
                emb(&[angle.cos(), angle.sin()])
            })
            .collect()
    }

    #[test]
    fn keeps_the_largest_eigenvalues_not_the_largest_magnitudes() {
        // A 12-ring's normalized affinity has eigenvalues cos(2πj/12):
        // 1, then 0.866 twice, 0.5 twice, ... down to −1. The −1 has the
        // largest magnitude after the trivial 1, and plain power iteration
        // would keep it; it says which half of a two-coloring an item is
        // in, not where it sits on the ring.
        let lap = LaplacianEigenmapProjection::fit_cosine(&ring(12), 2, RadialStrategy::Fixed(1.0))
            .unwrap();
        let [m1, m2, m3] = lap.eigenvalues();
        let cos30 = (std::f64::consts::PI / 6.0).cos();
        assert!((m1 - cos30).abs() < 1e-8, "μ_1 = {m1}");
        assert!((m2 - cos30).abs() < 1e-8, "μ_2 = {m2}");
        assert!((m3 - 0.5).abs() < 1e-8, "μ_3 = {m3}");
    }

    #[test]
    fn kept_vectors_are_eigenvectors() {
        let corpus = three_cluster_corpus();
        for lap in [
            LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap(),
            LaplacianEigenmapProjection::fit_with_params(
                &corpus,
                4,
                DEFAULT_ACTIVE_THRESHOLD,
                RadialStrategy::Fixed(1.0),
            )
            .unwrap(),
        ] {
            // Rebuild S from the projection's own neighbour search.
            let lists: Vec<Vec<(usize, f64)>> = match &lap.corpus {
                Corpus::Cosine { normalized, .. } => (0..corpus.len())
                    .map(|i| {
                        top_k(
                            (0..corpus.len())
                                .filter(|&j| j != i)
                                .map(|j| (j, dot(&normalized[i], &normalized[j]))),
                            lap.k,
                        )
                    })
                    .collect(),
                Corpus::ActiveAxes { sets, .. } => (0..corpus.len())
                    .map(|i| {
                        top_k(
                            (0..corpus.len())
                                .filter(|&j| j != i)
                                .map(|j| (j, jaccard_sorted(&sets[i], &sets[j]))),
                            lap.k,
                        )
                    })
                    .collect(),
            };
            let graph = Graph::from_neighbor_lists(&lists);
            let trivial: Vec<f64> = graph.degrees.iter().map(|d| d.sqrt()).collect();
            let mut image = vec![0.0; corpus.len()];
            for (vector, mu) in lap.eigenvectors.iter().zip(lap.eigenvalues) {
                graph.multiply(&mut image, vector);
                let residual: f64 = image
                    .iter()
                    .zip(vector)
                    .map(|(sv, v)| (sv - mu * v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(residual < 1e-5, "‖Sv − μv‖ = {residual} at μ = {mu}");
                assert!((dot(vector, vector) - 1.0).abs() < 1e-9);
                assert!(dot(vector, &trivial).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn fitted_items_keep_their_own_rows() {
        let corpus = three_cluster_corpus();
        for lap in [
            LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap(),
            LaplacianEigenmapProjection::fit_with_params(
                &corpus,
                4,
                DEFAULT_ACTIVE_THRESHOLD,
                RadialStrategy::Fixed(1.0),
            )
            .unwrap(),
        ] {
            for (i, e) in corpus.iter().enumerate() {
                let (x, y, z) = lap.project_to_3d(e);
                assert_eq!(
                    [x, y, z],
                    [
                        lap.eigenvectors[0][i],
                        lap.eigenvectors[1][i],
                        lap.eigenvectors[2][i]
                    ],
                    "item {i}"
                );
            }
        }
    }

    #[test]
    fn a_query_is_placed_from_its_nearest_fitted_items() {
        // A query next to ring item 3. Its two nearest fitted items are 3
        // and one of 3's neighbours, so it lands between them, nearer to 3
        // than any item two or more steps round the ring. Weighing every
        // fitted item instead pulls it toward the middle of the ring.
        let corpus = ring(24);
        let lap = LaplacianEigenmapProjection::fit_cosine(&corpus, 2, RadialStrategy::Fixed(1.0))
            .unwrap();
        let angle = std::f64::consts::TAU * 3.1 / 24.0;
        let query = lap.project(&emb(&[angle.cos(), angle.sin()]));
        let to = |i: usize| angular_distance(&query, &lap.project(&corpus[i]));
        for far in (0..24).filter(|i| !(2..=5).contains(i)) {
            assert!(to(3) < to(far), "3: {}, {far}: {}", to(3), to(far));
        }
    }

    #[test]
    fn dense_embeddings_keep_their_groups_on_the_cosine_graph() {
        // Five groups in 20 dimensions with every axis well above the
        // active threshold for every item. All active sets are equal, so
        // the active-axis graph sees no difference between items; the
        // cosine graph does.
        let mut corpus = Vec::new();
        let mut rng = SplitMix64::new(7);
        for group in 0..5 {
            for _ in 0..8 {
                let v: Vec<f64> = (0..20)
                    .map(|axis| {
                        let center = if axis % 5 == group { 2.0 } else { 0.5 };
                        center + 0.05 * rng.normal()
                    })
                    .collect();
                corpus.push(emb(&v));
            }
        }
        let lap = LaplacianEigenmapProjection::fit_cosine(&corpus, 5, RadialStrategy::Fixed(1.0))
            .unwrap();
        assert_eq!(lap.similarity(), LaplacianSimilarity::Cosine);
        let at: Vec<SphericalPoint> = corpus.iter().map(|e| lap.project(e)).collect();
        for i in 0..40 {
            for j in 0..40 {
                for l in 0..40 {
                    if i != j && i / 8 == j / 8 && i / 8 != l / 8 {
                        assert!(
                            angular_distance(&at[i], &at[j]) < angular_distance(&at[i], &at[l]),
                            "{i} is nearer to {l} than to {j}"
                        );
                    }
                }
            }
        }

        let sets: Vec<Vec<usize>> = corpus
            .iter()
            .map(|e| active_axes(&e.values, DEFAULT_ACTIVE_THRESHOLD))
            .collect();
        assert!(sets.iter().all(|s| s.len() == 20));
    }

    #[test]
    fn two_fits_are_identical() {
        let corpus = three_cluster_corpus();
        let a = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        let b = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        assert_eq!(a.eigenvalues, b.eigenvalues);
        assert_eq!(a.eigenvectors, b.eigenvectors);
        let query = emb(&[0.9, 0.7, 0.8, 0.1, 0.0, 0.05, 0.0, 0.02]);
        assert_eq!(a.project(&query), b.project(&query));
    }

    #[test]
    fn repeated_and_zero_vectors_fit() {
        let mut corpus = three_cluster_corpus();
        corpus.push(corpus[0].clone());
        corpus.push(emb(&[0.0; 8]));
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        for e in &corpus {
            let sp = lap.project(e);
            assert!(sp.theta.is_finite() && sp.phi.is_finite(), "{sp:?}");
        }
    }

    #[test]
    fn fit_rejects_mixed_dimensions() {
        let mut corpus = three_cluster_corpus();
        corpus.push(emb(&[1.0, 0.0, 0.0]));
        assert!(matches!(
            LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)),
            Err(ProjectionError::InconsistentDimension { index: 15, .. })
        ));
    }

    #[test]
    fn symmetric_eigen_diagonalizes() {
        let a = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let (mut values, vectors) = symmetric_eigen(a.clone());
        for col in 0..3 {
            for row in 0..3 {
                let av: f64 = (0..3).map(|j| a[row][j] * vectors[j][col]).sum();
                assert!((av - values[col] * vectors[row][col]).abs() < 1e-12);
            }
        }
        values.sort_by(f64::total_cmp);
        let root2 = 2.0f64.sqrt();
        for (got, want) in values.iter().zip([2.0 - root2, 2.0, 2.0 + root2]) {
            assert!((got - want).abs() < 1e-12);
        }
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
    fn projecting_a_fitted_item_twice_agrees() {
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
