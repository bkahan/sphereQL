use std::collections::{BinaryHeap, HashMap};

use sphereql_core::*;
use sphereql_index::*;

use crate::projection::Projection;
use crate::types::Embedding;

#[derive(Debug, Clone)]
pub struct EmbeddingItem {
    pub id: String,
    pub position: SphericalPoint,
    pub original_magnitude: f64,
}

impl SpatialItem for EmbeddingItem {
    type Id = String;
    fn id(&self) -> &String {
        &self.id
    }
    fn position(&self) -> &SphericalPoint {
        &self.position
    }
}

pub struct EmbeddingIndexBuilder<P> {
    projection: P,
    inner: SpatialIndexBuilder,
}

impl<P: Projection> EmbeddingIndexBuilder<P> {
    pub fn new(projection: P) -> Self {
        Self {
            projection,
            inner: SpatialIndexBuilder::new(),
        }
    }

    pub fn shell_boundary(mut self, r: f64) -> Self {
        self.inner = self.inner.shell_boundary(r);
        self
    }

    pub fn uniform_shells(mut self, count: usize, max_r: f64) -> Self {
        self.inner = self.inner.uniform_shells(count, max_r);
        self
    }

    pub fn theta_divisions(mut self, n: usize) -> Self {
        self.inner = self.inner.theta_divisions(n);
        self
    }

    pub fn phi_divisions(mut self, n: usize) -> Self {
        self.inner = self.inner.phi_divisions(n);
        self
    }

    pub fn build(self) -> EmbeddingIndex<P> {
        EmbeddingIndex {
            projection: self.projection,
            index: self.inner.build(),
        }
    }
}

pub struct EmbeddingIndex<P> {
    projection: P,
    index: SpatialIndex<EmbeddingItem>,
}

impl<P: Projection> EmbeddingIndex<P> {
    pub fn builder(projection: P) -> EmbeddingIndexBuilder<P> {
        EmbeddingIndexBuilder::new(projection)
    }

    pub fn insert(&mut self, id: impl Into<String>, embedding: &Embedding) {
        let position = self.projection.project(embedding);
        self.index.insert(EmbeddingItem {
            id: id.into(),
            position,
            original_magnitude: embedding.magnitude(),
        });
    }

    /// Insert with an explicit radial value, overriding the projection's RadialStrategy.
    /// The angular coordinates (theta, phi) are still determined by the projection.
    /// Use this for metadata-driven radius: recency scores, importance weights, etc.
    pub fn insert_with_radius(&mut self, id: impl Into<String>, embedding: &Embedding, r: f64) {
        let projected = self.projection.project(embedding);
        self.index.insert(EmbeddingItem {
            id: id.into(),
            position: SphericalPoint::new_unchecked(r, projected.theta, projected.phi),
            original_magnitude: embedding.magnitude(),
        });
    }

    /// Find the k embeddings whose projected directions are closest to the query.
    pub fn search_nearest(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Vec<NearestResult<EmbeddingItem>> {
        let projected = self.projection.project(query);
        self.index.nearest(&projected, k)
    }

    /// Find all embeddings whose projected cosine similarity to the query
    /// is at least `min_cosine_similarity`.
    ///
    /// Internally maps cos(sim) → angular distance and uses `within_distance`.
    pub fn search_similar(
        &self,
        query: &Embedding,
        min_cosine_similarity: f64,
    ) -> SpatialQueryResult<EmbeddingItem> {
        let projected = self.projection.project(query);
        let max_angle = min_cosine_similarity.clamp(-1.0, 1.0).acos();
        self.index.within_distance(&projected, max_angle)
    }

    pub fn search_region(&self, region: &Region) -> SpatialQueryResult<EmbeddingItem> {
        self.index.query_region(region)
    }

    pub fn remove(&mut self, id: &str) -> Option<EmbeddingItem> {
        self.index.remove(&id.to_string())
    }

    pub fn get(&self, id: &str) -> Option<&EmbeddingItem> {
        self.index.get(&id.to_string())
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn projection(&self) -> &P {
        &self.projection
    }

    pub fn all_items(&self) -> Vec<&EmbeddingItem> {
        self.index.all_items()
    }

    /// Find the shortest semantic path between two items through a k-NN graph.
    ///
    /// Builds a k-nearest-neighbor graph over all indexed embeddings, then
    /// runs Dijkstra's algorithm weighted by angular distance. The resulting
    /// path traces the chain of closest intermediate concepts connecting
    /// the source to the target.
    pub fn concept_path(
        &self,
        source_id: &str,
        target_id: &str,
        k: usize,
    ) -> Option<ConceptPath> {
        let items = self.index.all_items();
        let n = items.len();
        if n < 2 {
            return None;
        }

        let id_to_idx: HashMap<&str, usize> = items
            .iter()
            .enumerate()
            .map(|(i, item)| (item.id.as_str(), i))
            .collect();

        let source_idx = *id_to_idx.get(source_id)?;
        let target_idx = *id_to_idx.get(target_id)?;

        // Build k-NN adjacency list (undirected)
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, item) in items.iter().enumerate() {
            let nearest = self.index.nearest(item.position(), k + 1);
            for result in &nearest {
                if let Some(&j) = id_to_idx.get(result.item.id.as_str())
                    && i != j
                {
                    adj[i].push((j, result.distance));
                }
            }
        }
        // Symmetrize
        let snapshot: Vec<Vec<(usize, f64)>> = adj.clone();
        for (i, edges) in snapshot.iter().enumerate() {
            for &(j, d) in edges {
                if !adj[j].iter().any(|&(k, _)| k == i) {
                    adj[j].push((i, d));
                }
            }
        }

        // Dijkstra (min-heap via reversed Ord)
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut heap = BinaryHeap::new();

        dist[source_idx] = 0.0;
        heap.push(DijkstraEntry { dist: 0.0, node: source_idx });

        while let Some(entry) = heap.pop() {
            let u = entry.node;
            if entry.dist > dist[u] {
                continue;
            }
            if u == target_idx {
                break;
            }
            for &(v, w) in &adj[u] {
                let nd = dist[u] + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    prev[v] = Some(u);
                    heap.push(DijkstraEntry { dist: nd, node: v });
                }
            }
        }

        if dist[target_idx].is_infinite() {
            return None;
        }

        // Reconstruct
        let mut path = Vec::new();
        let mut cur = target_idx;
        loop {
            path.push(PathStep {
                id: items[cur].id.clone(),
                cumulative_distance: dist[cur],
            });
            match prev[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path.reverse();

        Some(ConceptPath {
            total_distance: dist[target_idx],
            steps: path,
        })
    }
}

// --- Concept path types ---

#[derive(Debug, Clone)]
pub struct ConceptPath {
    pub steps: Vec<PathStep>,
    pub total_distance: f64,
}

#[derive(Debug, Clone)]
pub struct PathStep {
    pub id: String,
    pub cumulative_distance: f64,
}

#[derive(PartialEq)]
struct DijkstraEntry {
    dist: f64,
    node: usize,
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed: BinaryHeap is a max-heap, so smaller dist = higher priority
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// --- Slicing manifold ---

/// A 2D plane fitted through the 3D projected point cloud that captures
/// the maximum variance. Found by PCA on the Cartesian coordinates of
/// the projected embeddings.
///
/// The plane is defined by:
/// - `centroid`: the mean of all 3D points
/// - `basis_u`, `basis_v`: orthonormal vectors spanning the plane (directions of max variance)
/// - `normal`: vector perpendicular to the plane (direction of minimum variance)
#[derive(Debug, Clone)]
pub struct SlicingManifold {
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub basis_u: [f64; 3],
    pub basis_v: [f64; 3],
    pub variance_ratio: f64,
}

impl SlicingManifold {
    /// Fit the optimal slicing plane to a set of 3D points.
    /// Each point is (x, y, z) in Cartesian coordinates.
    pub fn fit(points: &[[f64; 3]]) -> Self {
        let n = points.len() as f64;
        assert!(n >= 3.0, "need at least 3 points to fit a plane");

        // Centroid
        let mut c = [0.0; 3];
        for p in points {
            for i in 0..3 {
                c[i] += p[i];
            }
        }
        for ci in &mut c {
            *ci /= n;
        }

        // 3×3 covariance matrix (symmetric)
        let mut cov = [[0.0f64; 3]; 3];
        for p in points {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
            }
        }
        for row in &mut cov {
            for v in row.iter_mut() {
                *v /= n;
            }
        }

        // Eigendecomposition of 3×3 symmetric matrix via Jacobi iteration
        let (eigenvalues, eigenvectors) = eigen_symmetric_3x3(&cov);

        // eigenvalues are sorted descending: λ₀ ≥ λ₁ ≥ λ₂
        // basis_u = eigenvector of λ₀, basis_v = eigenvector of λ₁, normal = eigenvector of λ₂
        let total_var = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
        let variance_ratio = if total_var > 0.0 {
            (eigenvalues[0] + eigenvalues[1]) / total_var
        } else {
            1.0
        };

        Self {
            centroid: c,
            normal: eigenvectors[2],
            basis_u: eigenvectors[0],
            basis_v: eigenvectors[1],
            variance_ratio,
        }
    }

    /// Project a 3D point onto the plane, returning (u, v) coordinates.
    pub fn project_2d(&self, point: &[f64; 3]) -> (f64, f64) {
        let d = [
            point[0] - self.centroid[0],
            point[1] - self.centroid[1],
            point[2] - self.centroid[2],
        ];
        let u = d[0] * self.basis_u[0] + d[1] * self.basis_u[1] + d[2] * self.basis_u[2];
        let v = d[0] * self.basis_v[0] + d[1] * self.basis_v[1] + d[2] * self.basis_v[2];
        (u, v)
    }

    /// Signed distance from the plane (positive = same side as normal).
    pub fn distance(&self, point: &[f64; 3]) -> f64 {
        let d = [
            point[0] - self.centroid[0],
            point[1] - self.centroid[1],
            point[2] - self.centroid[2],
        ];
        d[0] * self.normal[0] + d[1] * self.normal[1] + d[2] * self.normal[2]
    }
}

/// Eigendecomposition of a 3×3 symmetric matrix via Jacobi rotations.
/// Returns (eigenvalues_desc, eigenvectors_desc) sorted by decreasing eigenvalue.
fn eigen_symmetric_3x3(m: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = *m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // eigenvector matrix

    #[allow(clippy::needless_range_loop)]
    for _ in 0..50 {
        // Find largest off-diagonal element
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-15 {
            break;
        }

        // Jacobi rotation to zero out a[p][q]
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Rotate a ← GᵀaG
        let mut new_a = a;
        for i in 0..3 {
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
        }
        let snapshot = new_a;
        for j in 0..3 {
            new_a[p][j] = c * snapshot[p][j] + s * snapshot[q][j];
            new_a[q][j] = -s * snapshot[p][j] + c * snapshot[q][j];
        }
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Rotate eigenvectors: V ← VG
        let mut new_v = v;
        for i in 0..3 {
            new_v[i][p] = c * v[i][p] + s * v[i][q];
            new_v[i][q] = -s * v[i][p] + c * v[i][q];
        }
        v = new_v;
    }

    let eigenvalues = [a[0][0], a[1][1], a[2][2]];

    // Sort by descending eigenvalue
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_vals = [
        eigenvalues[order[0]],
        eigenvalues[order[1]],
        eigenvalues[order[2]],
    ];
    // Eigenvectors are columns of v
    let sorted_vecs = [
        [v[0][order[0]], v[1][order[0]], v[2][order[0]]],
        [v[0][order[1]], v[1][order[1]], v[2][order[1]]],
        [v[0][order[2]], v[1][order[2]], v[2][order[2]]],
    ];

    (sorted_vals, sorted_vecs)
}

// --- Concept Globs (spherical k-means + silhouette auto-k) ---

/// A cluster of semantically related embeddings in the projected 3D space.
#[derive(Debug, Clone)]
pub struct ConceptGlob {
    pub id: usize,
    pub centroid: [f64; 3],
    pub member_ids: Vec<String>,
    pub member_distances: Vec<f64>,
    pub radius: f64,
}

/// Result of glob detection: the set of all globs plus quality metrics.
#[derive(Debug, Clone)]
pub struct GlobResult {
    pub globs: Vec<ConceptGlob>,
    pub k: usize,
    pub silhouette: f64,
}

impl GlobResult {
    /// Detect concept globs from 3D projected points.
    ///
    /// If `k` is `Some`, uses that many clusters.
    /// If `None`, auto-selects k ∈ [2, max_k] by maximizing the silhouette score.
    pub fn detect(
        points: &[[f64; 3]],
        ids: &[String],
        k: Option<usize>,
        max_k: usize,
    ) -> Self {
        let n = points.len();
        assert_eq!(n, ids.len());
        assert!(n >= 2, "need at least 2 points for clustering");

        let max_k = max_k.min(n);

        if let Some(k) = k {
            let k = k.clamp(2, max_k);
            let (assignments, silhouette) = kmeans_3d(points, k);
            let globs = build_globs(points, ids, &assignments, k);
            return Self {
                globs,
                k,
                silhouette,
            };
        }

        // Auto-detect: try k = 2..=max_k, pick best silhouette
        let mut best_k = 2;
        let mut best_sil = f64::NEG_INFINITY;
        let mut best_assignments = vec![0usize; n];

        for trial_k in 2..=max_k {
            let (assignments, sil) = kmeans_3d(points, trial_k);
            if sil > best_sil {
                best_sil = sil;
                best_k = trial_k;
                best_assignments = assignments;
            }
        }

        let globs = build_globs(points, ids, &best_assignments, best_k);
        Self {
            globs,
            k: best_k,
            silhouette: best_sil,
        }
    }
}

fn kmeans_3d(points: &[[f64; 3]], k: usize) -> (Vec<usize>, f64) {
    let n = points.len();
    let max_iter = 50;

    // Init: spread initial centers evenly across the point set
    let mut centers: Vec<[f64; 3]> = (0..k)
        .map(|i| points[i * n / k])
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;

        // Assign
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (j, c) in centers.iter().enumerate() {
                let d = dist3(p, c);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centers
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            for d in 0..3 {
                sums[a][d] += points[i][d];
            }
            counts[a] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..3 {
                    centers[j][d] = sums[j][d] / counts[j] as f64;
                }
            }
        }
    }

    let sil = silhouette_score(points, &assignments, k);
    (assignments, sil)
}

fn silhouette_score(points: &[[f64; 3]], assignments: &[usize], k: usize) -> f64 {
    let n = points.len();
    if n <= 1 || k <= 1 {
        return 0.0;
    }

    let mut total = 0.0;
    for i in 0..n {
        let ci = assignments[i];

        // a(i): mean dist to same-cluster members
        let mut a_sum = 0.0;
        let mut a_cnt = 0;
        for j in 0..n {
            if j != i && assignments[j] == ci {
                a_sum += dist3(&points[i], &points[j]);
                a_cnt += 1;
            }
        }
        let a = if a_cnt > 0 { a_sum / a_cnt as f64 } else { 0.0 };

        // b(i): min mean dist to any other cluster
        let mut b = f64::MAX;
        for ck in 0..k {
            if ck == ci {
                continue;
            }
            let mut b_sum = 0.0;
            let mut b_cnt = 0;
            for j in 0..n {
                if assignments[j] == ck {
                    b_sum += dist3(&points[i], &points[j]);
                    b_cnt += 1;
                }
            }
            if b_cnt > 0 {
                b = b.min(b_sum / b_cnt as f64);
            }
        }
        if b == f64::MAX {
            b = 0.0;
        }

        let denom = a.max(b);
        total += if denom > 0.0 { (b - a) / denom } else { 0.0 };
    }

    total / n as f64
}

fn build_globs(
    points: &[[f64; 3]],
    ids: &[String],
    assignments: &[usize],
    k: usize,
) -> Vec<ConceptGlob> {
    let mut globs = Vec::with_capacity(k);

    for cluster_id in 0..k {
        let member_indices: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a == cluster_id)
            .map(|(i, _)| i)
            .collect();

        if member_indices.is_empty() {
            continue;
        }

        // Centroid
        let mut centroid = [0.0; 3];
        for &i in &member_indices {
            for d in 0..3 {
                centroid[d] += points[i][d];
            }
        }
        let n = member_indices.len() as f64;
        for c in &mut centroid {
            *c /= n;
        }

        // Member distances from centroid
        let member_distances: Vec<f64> = member_indices
            .iter()
            .map(|&i| dist3(&points[i], &centroid))
            .collect();

        let radius = member_distances
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        let member_ids: Vec<String> = member_indices
            .iter()
            .map(|&i| ids[i].clone())
            .collect();

        globs.push(ConceptGlob {
            id: cluster_id,
            centroid,
            member_ids,
            member_distances,
            radius,
        });
    }

    globs
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Builds SphereQL [`Region`]s from semantic constraints on embeddings.
pub struct SemanticQuery;

impl SemanticQuery {
    /// Spherical cap: all points within `max_angular_distance` radians of the query.
    pub fn within_angle<P: Projection>(
        query: &Embedding,
        projection: &P,
        max_angular_distance: f64,
    ) -> Region {
        let point = projection.project(query);
        let half_angle = max_angular_distance.clamp(1e-10, std::f64::consts::PI);
        Region::Cap(
            Cap::new(
                SphericalPoint::new_unchecked(1.0, point.theta, point.phi),
                half_angle,
            )
            .unwrap(),
        )
    }

    /// Spherical cap from a cosine similarity threshold.
    /// cos_sim >= threshold ↔ angular_distance <= arccos(threshold).
    pub fn above_similarity<P: Projection>(
        query: &Embedding,
        projection: &P,
        min_similarity: f64,
    ) -> Region {
        let half_angle = min_similarity.clamp(-1.0, 1.0).acos();
        Self::within_angle(query, projection, half_angle)
    }

    /// Radial shell: embeddings whose projected radius falls in [inner, outer].
    pub fn in_shell(inner: f64, outer: f64) -> Region {
        Region::Shell(Shell::new(inner, outer).expect("invalid shell bounds"))
    }

    /// Intersection of a similarity cap with a radial shell.
    /// "Semantically similar AND within a magnitude/metadata range."
    pub fn similar_in_shell<P: Projection>(
        query: &Embedding,
        projection: &P,
        min_similarity: f64,
        shell_inner: f64,
        shell_outer: f64,
    ) -> Region {
        Region::intersection(vec![
            Self::above_similarity(query, projection, min_similarity),
            Self::in_shell(shell_inner, shell_outer),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{PcaProjection, RandomProjection};
    use crate::types::RadialStrategy;
    use sphereql_core::angular_distance;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    fn test_corpus() -> Vec<Embedding> {
        vec![
            emb(&[1.0, 0.0, 0.0, 0.1, 0.0]),
            emb(&[0.0, 1.0, 0.0, 0.0, 0.1]),
            emb(&[0.0, 0.0, 1.0, 0.1, 0.0]),
            emb(&[1.0, 1.0, 0.0, 0.05, 0.05]),
            emb(&[-1.0, 0.0, 0.0, -0.1, 0.0]),
            emb(&[0.0, -1.0, 0.0, 0.0, -0.1]),
        ]
    }

    // --- EmbeddingIndex ---

    #[test]
    fn insert_and_get() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        idx.insert("a", &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]));
        idx.insert("b", &emb(&[0.0, 1.0, 0.0, 0.0, 0.0]));

        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
        assert!(idx.get("a").is_some());
        assert!(idx.get("b").is_some());
        assert!(idx.get("c").is_none());
    }

    #[test]
    fn remove() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();

        idx.insert("a", &emb(&[1.0; 5]));
        assert_eq!(idx.len(), 1);

        let removed = idx.remove("a");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "a");
        assert_eq!(idx.len(), 0);
        assert!(idx.get("a").is_none());
    }

    #[test]
    fn remove_nonexistent() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();
        assert!(idx.remove("nope").is_none());
    }

    #[test]
    fn search_nearest_returns_sorted() {
        let corpus = test_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let mut idx = EmbeddingIndex::builder(pca)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        for (i, e) in corpus.iter().enumerate() {
            idx.insert(format!("item-{i}"), e);
        }

        let query = emb(&[0.95, 0.1, 0.0, 0.05, 0.0]);
        let results = idx.search_nearest(&query, 3);

        assert_eq!(results.len(), 3);
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }

    #[test]
    fn search_similar_respects_threshold() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        idx.insert("close_a", &emb(&[1.0, 0.1, 0.0, 0.0, 0.0]));
        idx.insert("close_b", &emb(&[0.9, 0.2, 0.0, 0.0, 0.0]));
        idx.insert("far", &emb(&[-1.0, 0.0, 0.0, 0.0, 0.0]));

        let query = emb(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let projected_query = idx.projection().project(&query);
        let result = idx.search_similar(&query, 0.5);

        let max_angle = 0.5_f64.acos();
        for item in &result.items {
            let d = angular_distance(&projected_query, item.position());
            assert!(d <= max_angle + 1e-10, "item {} too far: {d}", item.id);
        }
    }

    #[test]
    fn insert_with_radius_overrides() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();

        idx.insert_with_radius("custom", &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]), 42.0);
        let item = idx.get("custom").unwrap();
        assert!((item.position.r - 42.0).abs() < 1e-12);
    }

    #[test]
    fn original_magnitude_stored() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();

        let e = emb(&[3.0, 4.0, 0.0, 0.0, 0.0]);
        idx.insert("vec", &e);
        let item = idx.get("vec").unwrap();
        assert!((item.original_magnitude - 5.0).abs() < 1e-10);
    }

    #[test]
    fn magnitude_radial_with_shell_query() {
        let corpus = test_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude);
        let mut idx = EmbeddingIndex::builder(pca)
            .uniform_shells(5, 10.0)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        idx.insert("small", &emb(&[0.1, 0.0, 0.0, 0.0, 0.0]));
        idx.insert("medium", &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]));
        idx.insert("large", &emb(&[5.0, 0.0, 0.0, 0.0, 0.0]));

        let shell = Shell::new(0.5, 2.0).unwrap();
        let result = idx.search_region(&Region::Shell(shell));

        let ids: Vec<&str> = result.items.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"medium"), "medium (mag=1.0) should be in [0.5, 2.0]");
        assert!(!ids.contains(&"large"), "large (mag=5.0) should not be in [0.5, 2.0]");
    }

    #[test]
    fn empty_index() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let idx = EmbeddingIndex::builder(rp).build();

        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.get("x").is_none());

        let results = idx.search_nearest(&emb(&[1.0; 5]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn projection_accessor() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let idx = EmbeddingIndex::builder(rp).build();
        assert_eq!(idx.projection().dimensionality(), 5);
    }

    // --- SemanticQuery ---

    #[test]
    fn above_similarity_creates_cap() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let region = SemanticQuery::above_similarity(&emb(&[1.0; 5]), &rp, 0.8);
        assert!(matches!(region, Region::Cap(_)));
    }

    #[test]
    fn within_angle_creates_cap() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let region = SemanticQuery::within_angle(&emb(&[1.0; 5]), &rp, 0.5);
        assert!(matches!(region, Region::Cap(_)));
    }

    #[test]
    fn in_shell_creates_shell() {
        let region = SemanticQuery::in_shell(1.0, 5.0);
        assert!(matches!(region, Region::Shell(_)));
    }

    #[test]
    fn similar_in_shell_creates_intersection() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let region = SemanticQuery::similar_in_shell(&emb(&[1.0; 5]), &rp, 0.7, 1.0, 5.0);

        match region {
            Region::Intersection(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(parts[0], Region::Cap(_)));
                assert!(matches!(parts[1], Region::Shell(_)));
            }
            other => panic!("expected Intersection, got {other:?}"),
        }
    }

    #[test]
    fn semantic_query_region_used_in_index() {
        let corpus = test_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let projection_clone = pca.clone();
        let mut idx = EmbeddingIndex::builder(pca)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        for (i, e) in corpus.iter().enumerate() {
            idx.insert(format!("item-{i}"), e);
        }

        let query = emb(&[1.0, 0.0, 0.0, 0.05, 0.0]);
        let region = SemanticQuery::above_similarity(&query, &projection_clone, 0.5);
        let result = idx.search_region(&region);

        for item in &result.items {
            assert!(region.contains(item.position()));
        }
    }
}
