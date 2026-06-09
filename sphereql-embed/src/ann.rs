//! Approximate nearest neighbors via random projection forest.
//!
//! Builds `n_trees` binary trees by recursively splitting the data at
//! random hyperplanes. At query time, each tree nominates a candidate
//! leaf; the union of candidates is re-scored exactly to produce the
//! final k-NN list.
//!
//! Deterministic for a given seed (uses [`SplitMix64`]).
//!
//! Complexity:
//! - Build: O(N · d · trees · log N)
//! - Query: O(trees · log N · d + |candidates| · d)
//!
//! Designed for cosine similarity (all vectors are L2-normalized
//! internally). Reusable across UMAP graph construction,
//! `GraphModularity` scoring, and downstream consumers (globetrot
//! similarity lookup).

use crate::projection::{SplitMix64, dot, normalize_vec};

/// Default seed for the RP-forest. Deterministic by design.
const DEFAULT_ANN_SEED: u64 = 0xA00F_0E57;

/// Configuration for the RP-forest index.
#[derive(Debug, Clone)]
pub struct AnnConfig {
    /// Number of random projection trees. More trees = better recall,
    /// slower build + query. 8 is a good default for N < 1M.
    pub n_trees: usize,
    /// Maximum leaf size. Nodes with fewer than this many items are not
    /// split further. Smaller = deeper trees = more precise but slower.
    pub max_leaf_size: usize,
    /// PRNG seed for reproducibility.
    pub seed: u64,
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self {
            n_trees: 8,
            max_leaf_size: 40,
            seed: DEFAULT_ANN_SEED,
        }
    }
}

/// A built RP-forest index over L2-normalized vectors.
pub struct AnnIndex {
    trees: Vec<RpTree>,
    /// L2-normalized copies of the input vectors. Stored so that exact
    /// re-scoring at query time uses the same normalization as the
    /// tree splits.
    normalized: Vec<Vec<f64>>,
    dim: usize,
}

/// One node in an RP-tree. Either a split (hyperplane + children) or a
/// leaf (list of item indices).
enum RpNode {
    Split {
        /// Unit normal of the splitting hyperplane.
        normal: Vec<f64>,
        /// Items with `dot(x, normal) >= offset` go right.
        offset: f64,
        left: Box<RpNode>,
        right: Box<RpNode>,
    },
    Leaf {
        indices: Vec<usize>,
    },
}

struct RpTree {
    root: RpNode,
}

impl AnnIndex {
    /// Build the index from raw (un-normalized) vectors. Each vector is
    /// L2-normalized internally. All vectors must have the same
    /// dimensionality. Panics if `data` is empty or dimensions
    /// disagree.
    pub fn build(data: &[Vec<f64>], config: &AnnConfig) -> Self {
        assert!(
            !data.is_empty(),
            "AnnIndex::build requires at least one vector"
        );
        let dim = data[0].len();
        for (i, v) in data.iter().enumerate() {
            assert_eq!(
                v.len(),
                dim,
                "AnnIndex::build: vector {i} has dim {}, expected {dim}",
                v.len()
            );
        }

        let normalized: Vec<Vec<f64>> = data
            .iter()
            .map(|v| {
                let mut n = v.clone();
                normalize_vec(&mut n);
                n
            })
            .collect();

        Self::build_from_normalized(normalized, dim, config)
    }

    /// Build from pre-normalized vectors (avoids a redundant
    /// normalization pass when the caller already has unit vectors).
    /// Enforces the same contract as [`Self::build`]: panics if
    /// `normalized` is empty or dimensions disagree.
    pub fn build_normalized(normalized: Vec<Vec<f64>>, config: &AnnConfig) -> Self {
        assert!(
            !normalized.is_empty(),
            "AnnIndex::build_normalized requires at least one vector"
        );
        let dim = normalized[0].len();
        for (i, v) in normalized.iter().enumerate() {
            assert_eq!(
                v.len(),
                dim,
                "AnnIndex::build_normalized: vector {i} has dim {}, expected {dim}",
                v.len()
            );
        }
        Self::build_from_normalized(normalized, dim, config)
    }

    fn build_from_normalized(normalized: Vec<Vec<f64>>, dim: usize, config: &AnnConfig) -> Self {
        let all_indices: Vec<usize> = (0..normalized.len()).collect();
        let mut rng = SplitMix64::new(config.seed);

        let trees: Vec<RpTree> = (0..config.n_trees)
            .map(|_| {
                let root = build_tree(
                    &normalized,
                    &all_indices,
                    dim,
                    config.max_leaf_size,
                    &mut rng,
                );
                RpTree { root }
            })
            .collect();

        Self {
            trees,
            normalized,
            dim,
        }
    }

    /// Find the `k` approximate nearest neighbors of `query` by cosine
    /// similarity. Returns `(index, similarity)` pairs sorted by
    /// descending similarity. `query` is L2-normalized internally.
    pub fn query(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        assert_eq!(query.len(), self.dim);
        let mut q = query.to_vec();
        normalize_vec(&mut q);

        let mut candidates = Vec::new();
        for tree in &self.trees {
            collect_leaf(&tree.root, &q, &mut candidates);
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&i| (i, dot(&q, &self.normalized[i])))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    /// Find the `k` approximate nearest neighbors of the item at
    /// `index` (excludes self from results).
    pub fn query_by_index(&self, index: usize, k: usize) -> Vec<(usize, f64)> {
        let q = &self.normalized[index];
        let mut candidates = Vec::new();
        for tree in &self.trees {
            collect_leaf(&tree.root, q, &mut candidates);
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .filter(|&&i| i != index)
            .map(|&i| (i, dot(q, &self.normalized[i])))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    /// Build a full k-NN adjacency list for all items. Returns
    /// `knn[i]` = indices of the k nearest neighbors of item `i`
    /// (excluding self), sorted by descending similarity.
    pub fn knn_graph(&self, k: usize) -> Vec<Vec<usize>> {
        (0..self.normalized.len())
            .map(|i| {
                self.query_by_index(i, k)
                    .into_iter()
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect()
    }

    /// Number of indexed items.
    pub fn len(&self) -> usize {
        self.normalized.len()
    }

    /// True if the index contains no items.
    pub fn is_empty(&self) -> bool {
        self.normalized.is_empty()
    }
}

// ── Tree construction ──────────────────────────────────────────────────

fn build_tree(
    data: &[Vec<f64>],
    indices: &[usize],
    dim: usize,
    max_leaf: usize,
    rng: &mut SplitMix64,
) -> RpNode {
    if indices.len() <= max_leaf {
        return RpNode::Leaf {
            indices: indices.to_vec(),
        };
    }

    // Annoy-style RP split: pick two items, take their difference as the
    // hyperplane normal. Falls back to a random Gaussian normal if the
    // two items collide.
    let a = indices[(rng.next_u64() as usize) % indices.len()];
    let mut b = indices[(rng.next_u64() as usize) % indices.len()];
    let mut attempts = 0;
    while b == a && attempts < 10 {
        b = indices[(rng.next_u64() as usize) % indices.len()];
        attempts += 1;
    }

    let mut normal: Vec<f64> = data[a]
        .iter()
        .zip(data[b].iter())
        .map(|(&ai, &bi)| ai - bi)
        .collect();
    let mag = normalize_vec(&mut normal);
    if mag < f64::EPSILON {
        normal = (0..dim).map(|_| rng.normal()).collect();
        normalize_vec(&mut normal);
    }

    // Median projection gives a balanced split.
    let mut projections: Vec<f64> = indices.iter().map(|&i| dot(&data[i], &normal)).collect();
    projections.sort_unstable_by(|a, b| a.total_cmp(b));
    let offset = projections[projections.len() / 2];

    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for &i in indices {
        if dot(&data[i], &normal) < offset {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    // Guard against degenerate splits where every item lands on one side.
    if left_idx.is_empty() || right_idx.is_empty() {
        let mid = indices.len() / 2;
        left_idx = indices[..mid].to_vec();
        right_idx = indices[mid..].to_vec();
    }

    let left = build_tree(data, &left_idx, dim, max_leaf, rng);
    let right = build_tree(data, &right_idx, dim, max_leaf, rng);

    RpNode::Split {
        normal,
        offset,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn collect_leaf(node: &RpNode, query: &[f64], out: &mut Vec<usize>) {
    match node {
        RpNode::Leaf { indices } => {
            out.extend_from_slice(indices);
        }
        RpNode::Split {
            normal,
            offset,
            left,
            right,
        } => {
            if dot(query, normal) < *offset {
                collect_leaf(left, query, out);
            } else {
                collect_leaf(right, query, out);
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.normal()).collect())
            .collect()
    }

    #[test]
    fn build_and_query_smoke() {
        let data = random_vectors(200, 32, 42);
        let index = AnnIndex::build(&data, &AnnConfig::default());
        assert_eq!(index.len(), 200);
        assert!(!index.is_empty());

        let results = index.query(&data[0], 5);
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn query_by_index_excludes_self() {
        let data = random_vectors(100, 16, 7);
        let index = AnnIndex::build(&data, &AnnConfig::default());
        let results = index.query_by_index(0, 5);
        assert!(results.iter().all(|(i, _)| *i != 0));
    }

    #[test]
    fn knn_graph_shape() {
        let data = random_vectors(50, 16, 99);
        let index = AnnIndex::build(&data, &AnnConfig::default());
        let knn = index.knn_graph(5);
        assert_eq!(knn.len(), 50);
        for neighbors in &knn {
            assert_eq!(neighbors.len(), 5);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let data = random_vectors(100, 16, 42);
        let cfg = AnnConfig {
            seed: 0xBEEF,
            ..AnnConfig::default()
        };
        let index1 = AnnIndex::build(&data, &cfg);
        let index2 = AnnIndex::build(&data, &cfg);
        let r1 = index1.query(&data[5], 10);
        let r2 = index2.query(&data[5], 10);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-12);
        }
    }

    #[test]
    fn finds_true_nearest_in_top_results() {
        // Two tight clusters; query a member of cluster A. True nearest
        // neighbors should all be from cluster A.
        let mut data = Vec::new();
        let mut rng = SplitMix64::new(42);
        for _ in 0..50 {
            let mut v = vec![0.0; 16];
            v[0] = 1.0 + rng.normal() * 0.05;
            v[1] = 0.0 + rng.normal() * 0.05;
            data.push(v);
        }
        for _ in 0..50 {
            let mut v = vec![0.0; 16];
            v[0] = 0.0 + rng.normal() * 0.05;
            v[1] = 1.0 + rng.normal() * 0.05;
            data.push(v);
        }

        let index = AnnIndex::build(&data, &AnnConfig::default());
        let results = index.query_by_index(0, 10);
        for (idx, _) in &results {
            assert!(*idx < 50, "expected cluster A member, got index {idx}");
        }
    }

    #[test]
    fn empty_panics() {
        let result = std::panic::catch_unwind(|| {
            AnnIndex::build(&[], &AnnConfig::default());
        });
        assert!(result.is_err());
    }

    #[test]
    fn build_normalized_ragged_input_panics() {
        let result = std::panic::catch_unwind(|| {
            AnnIndex::build_normalized(
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0]],
                &AnnConfig::default(),
            );
        });
        assert!(result.is_err(), "ragged input must be rejected");
    }
}
