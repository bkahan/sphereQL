use std::collections::HashMap;

use sphereql_core::{SphericalPoint, angular_distance};

use crate::kernel_pca::KernelPcaProjection;
use crate::projection::{PcaProjection, Projection};
use crate::types::{Embedding, RadialStrategy};

// ── Thresholds ─────────────────────────────────────────────────────────

/// Minimum category size to consider fitting an inner sphere.
const MIN_INNER_SPHERE_SIZE: usize = 20;

/// Minimum EVR improvement (inner − global_subset) to justify an inner sphere.
const MIN_EVR_IMPROVEMENT: f64 = 0.10;

/// Minimum category size to consider kernel PCA for the inner sphere.
const KERNEL_PCA_MIN_SIZE: usize = 80;

/// Minimum EVR improvement of kernel PCA over linear PCA to choose it.
const MIN_KERNEL_IMPROVEMENT: f64 = 0.05;

// ── Category summary ───────────────────────────────────────────────────

/// Aggregate statistics for a single category on the outer sphere.
///
/// Computed from the projected positions of all items in that category.
/// Every category gets a summary regardless of size — this is the
/// foundation of the Category Enrichment Layer.
#[derive(Debug, Clone)]
pub struct CategorySummary {
    /// Category name (as provided by the user).
    pub name: String,
    /// Indices of member items in the pipeline's item list.
    pub member_indices: Vec<usize>,
    /// Mean embedding in high-dimensional space (pre-projection).
    /// Length = embedding dimensionality.
    pub centroid_embedding: Vec<f64>,
    /// The centroid projected onto the outer sphere.
    pub centroid_position: SphericalPoint,
    /// Mean angular distance (radians) of members from the centroid
    /// on the projected sphere. Measures how "spread out" the category is.
    pub angular_spread: f64,
    /// 1.0 / (1.0 + angular_spread). Higher = tighter cluster.
    /// Normalized to (0, 1].
    pub cohesion: f64,
    /// Number of member items.
    pub member_count: usize,
}

// ── Bridge items ───────────────────────────────────────────────────────

/// An item that semantically spans two categories.
///
/// Bridge items are closer to a foreign category's centroid than to the
/// median distance within their own category. They are the conceptual
/// connectors that make cross-domain paths meaningful.
#[derive(Debug, Clone)]
pub struct BridgeItem {
    /// Index of this item in the pipeline's item list.
    pub item_index: usize,
    /// The item's own category index.
    pub source_category: usize,
    /// The foreign category this item bridges toward.
    pub target_category: usize,
    /// Cosine similarity to own category centroid (in high-D space).
    pub affinity_to_source: f64,
    /// Cosine similarity to foreign category centroid (in high-D space).
    pub affinity_to_target: f64,
    /// Bridge strength: harmonic mean of the two affinities.
    /// Higher = equally strong connection to both domains.
    pub bridge_strength: f64,
}

// ── Category graph ─────────────────────────────────────────────────────

/// Edge in the category adjacency graph.
#[derive(Debug, Clone)]
pub struct CategoryEdge {
    /// Index of the neighbor category.
    pub target: usize,
    /// Angular distance between category centroids on the sphere.
    pub centroid_distance: f64,
    /// Number of bridge items connecting these two categories.
    pub bridge_count: usize,
    /// Strongest bridge connecting these two categories (0.0 if no bridges).
    pub max_bridge_strength: f64,
    /// Mean bridge strength across all bridges (0.0 if no bridges).
    pub mean_bridge_strength: f64,
    /// Combined edge weight (lower = more connected).
    /// Computed as `centroid_distance / (1 + bridge_count * mean_bridge_strength)`.
    /// This prefers fewer strong bridges over many weak ones.
    pub weight: f64,
}

/// The full category adjacency graph.
#[derive(Debug, Clone)]
pub struct CategoryGraph {
    /// Adjacency list: `adjacency[i]` contains edges from category i.
    pub adjacency: Vec<Vec<CategoryEdge>>,
    /// Bridge items keyed by (source_category, target_category).
    /// Sorted by descending bridge_strength within each pair.
    pub bridges: HashMap<(usize, usize), Vec<BridgeItem>>,
}

// ── Category-level concept path ────────────────────────────────────────

/// A step in a category-level concept path.
#[derive(Debug, Clone)]
pub struct CategoryPathStep {
    /// Category index.
    pub category_index: usize,
    /// Category name.
    pub category_name: String,
    /// Cumulative distance from the start.
    pub cumulative_distance: f64,
    /// Bridge items connecting this step to the next (empty for the last step).
    pub bridges_to_next: Vec<BridgeItem>,
}

/// Result of a category-level concept path query.
#[derive(Debug, Clone)]
pub struct CategoryPath {
    /// Ordered steps from source to target category.
    pub steps: Vec<CategoryPathStep>,
    /// Total path distance.
    pub total_distance: f64,
}

// ── Inner sphere (Phase 2) ─────────────────────────────────────────────

/// The projection type used for a category's inner sphere.
///
/// Wraps either a linear PCA or kernel PCA projection, chosen
/// automatically based on the category's size and measured EVR
/// improvement over the global projection.
#[derive(Clone)]
pub enum InnerProjection {
    /// Standard linear PCA — used for categories with 20–79 members,
    /// or when kernel PCA doesn't improve over linear.
    LinearPca(PcaProjection),
    /// Gaussian kernel PCA — used for categories with ≥80 members
    /// where kernel PCA measurably outperforms linear PCA.
    KernelPca(KernelPcaProjection),
}

impl std::fmt::Debug for InnerProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LinearPca(_) => write!(f, "LinearPca"),
            Self::KernelPca(_) => write!(f, "KernelPca"),
        }
    }
}

impl Projection for InnerProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        match self {
            Self::LinearPca(p) => p.project(embedding),
            Self::KernelPca(p) => p.project(embedding),
        }
    }
    fn project_rich(&self, embedding: &Embedding) -> crate::types::ProjectedPoint {
        match self {
            Self::LinearPca(p) => p.project_rich(embedding),
            Self::KernelPca(p) => p.project_rich(embedding),
        }
    }
    fn dimensionality(&self) -> usize {
        match self {
            Self::LinearPca(p) => p.dimensionality(),
            Self::KernelPca(p) => p.dimensionality(),
        }
    }
}

/// A category-specific inner sphere with its own optimized projection.
///
/// Only created for categories that meet all of:
/// 1. At least `MIN_INNER_SPHERE_SIZE` members
/// 2. Inner EVR improves over global subset EVR by ≥ `MIN_EVR_IMPROVEMENT`
///
/// The inner sphere gives higher-resolution angular discrimination
/// within the category than the global outer projection can provide.
#[derive(Debug, Clone)]
pub struct InnerSphere {
    /// The category-specific projection (linear PCA or kernel PCA).
    pub projection: InnerProjection,
    /// Positions of member items in the inner sphere's coordinate system.
    /// `inner_positions[i]` corresponds to `member_indices[i]`.
    pub inner_positions: Vec<SphericalPoint>,
    /// Global item indices of the members (same order as inner_positions).
    pub member_indices: Vec<usize>,
    /// Explained variance ratio of the inner projection.
    pub explained_variance_ratio: f64,
    /// Mean certainty of these items under the global (outer) projection.
    /// Baseline for measuring improvement.
    pub global_subset_evr: f64,
    /// `explained_variance_ratio - global_subset_evr`.
    pub evr_improvement: f64,
}

/// A single item from a [`drill_down`](CategoryLayer::drill_down) query.
#[derive(Debug, Clone)]
pub struct DrillDownResult {
    /// Index of the item in the pipeline's global item list.
    pub item_index: usize,
    /// Angular distance to the query in the relevant coordinate system.
    pub distance: f64,
    /// Whether the inner sphere's projection was used (true) or the
    /// outer sphere was used as fallback (false).
    pub used_inner_sphere: bool,
}

/// Stats for a single inner sphere, returned by
/// [`inner_sphere_stats`](CategoryLayer::inner_sphere_stats).
#[derive(Debug, Clone)]
pub struct InnerSphereReport {
    /// Category name.
    pub category_name: String,
    /// Category index.
    pub category_index: usize,
    /// Number of members in the inner sphere.
    pub member_count: usize,
    /// `"LinearPca"` or `"KernelPca"`.
    pub projection_type: &'static str,
    /// Explained variance ratio of the inner projection.
    pub inner_evr: f64,
    /// Mean certainty of members under the global projection.
    pub global_subset_evr: f64,
    /// EVR improvement over global.
    pub evr_improvement: f64,
}

// ── The enrichment layer ───────────────────────────────────────────────

/// Category Enrichment Layer: aggregate statistics, inter-category graph,
/// bridge item detection, and automatic inner spheres over a projected
/// SphereQL corpus.
///
/// This is a read-only structure computed from an existing pipeline's
/// data. It adds category-level reasoning without modifying the
/// underlying projection or spatial index.
#[derive(Debug, Clone)]
pub struct CategoryLayer {
    /// One summary per unique category, in insertion order.
    pub summaries: Vec<CategorySummary>,
    /// Map from category name to index in `summaries`.
    pub name_to_index: HashMap<String, usize>,
    /// The inter-category adjacency graph.
    pub graph: CategoryGraph,
    /// Outer-sphere positions for all items (same indexing as embeddings).
    outer_positions: Vec<SphericalPoint>,
    /// Inner spheres keyed by category index. Only present for categories
    /// that meet the size and EVR-improvement thresholds.
    pub inner_spheres: HashMap<usize, InnerSphere>,
}

impl CategoryLayer {
    /// Build the category enrichment layer from pipeline data.
    ///
    /// - `categories[i]` is the category name for item i.
    /// - `embeddings[i]` is the raw embedding for item i.
    /// - `projected_positions[i]` is the spherical position on the outer sphere.
    /// - `projection` is used to project category centroids and measure
    ///   per-point certainty for inner sphere threshold decisions.
    ///
    /// Inner spheres are automatically constructed for categories that:
    /// 1. Have ≥ 20 members
    /// 2. Show ≥ 0.10 EVR improvement over the global projection
    ///
    /// Categories with ≥ 80 members additionally try kernel PCA and
    /// select it if it improves EVR by ≥ 0.05 over linear PCA.
    ///
    /// O(N·C + C²) for the base layer, plus O(n_c²·d) per inner sphere.
    pub fn build<P: Projection>(
        categories: &[String],
        embeddings: &[Embedding],
        projected_positions: &[SphericalPoint],
        projection: &P,
    ) -> Self {
        let n = categories.len();
        assert_eq!(n, embeddings.len());
        assert_eq!(n, projected_positions.len());

        // 1. Discover unique categories and group member indices
        let mut name_to_index: HashMap<String, usize> = HashMap::new();
        let mut cat_names: Vec<String> = Vec::new();
        let mut cat_members: Vec<Vec<usize>> = Vec::new();

        for (i, cat) in categories.iter().enumerate() {
            let idx = if let Some(&idx) = name_to_index.get(cat) {
                idx
            } else {
                let idx = cat_names.len();
                name_to_index.insert(cat.clone(), idx);
                cat_names.push(cat.clone());
                cat_members.push(Vec::new());
                idx
            };
            cat_members[idx].push(i);
        }

        let num_cats = cat_names.len();
        let dim = if n > 0 { embeddings[0].dimension() } else { 0 };

        // 2. Compute category summaries
        let mut summaries: Vec<CategorySummary> = Vec::with_capacity(num_cats);

        for (ci, name) in cat_names.iter().enumerate() {
            let members = &cat_members[ci];
            let count = members.len();

            // Centroid in high-D space
            let mut centroid_emb = vec![0.0; dim];
            for &mi in members {
                for (j, &v) in embeddings[mi].values.iter().enumerate() {
                    centroid_emb[j] += v;
                }
            }
            if count > 0 {
                for v in &mut centroid_emb {
                    *v /= count as f64;
                }
            }

            // Project the centroid
            let centroid_embedding_obj = Embedding::new(centroid_emb.clone());
            let centroid_position = projection.project(&centroid_embedding_obj);

            // Angular spread: mean angular distance of members from centroid
            let angular_spread = if count > 1 {
                let total: f64 = members
                    .iter()
                    .map(|&mi| angular_distance(&projected_positions[mi], &centroid_position))
                    .sum();
                total / count as f64
            } else {
                0.0
            };

            let cohesion = 1.0 / (1.0 + angular_spread);

            summaries.push(CategorySummary {
                name: name.clone(),
                member_indices: members.clone(),
                centroid_embedding: centroid_emb,
                centroid_position,
                angular_spread,
                cohesion,
                member_count: count,
            });
        }

        // 3. Build category graph + detect bridges
        let graph = Self::build_graph(&summaries, embeddings, num_cats);

        // 4. Build inner spheres for qualifying categories (Phase 2)
        let inner_spheres = Self::build_inner_spheres(&summaries, embeddings, projection);

        CategoryLayer {
            summaries,
            name_to_index,
            graph,
            outer_positions: projected_positions.to_vec(),
            inner_spheres,
        }
    }

    /// Build the inter-category adjacency graph and detect bridge items.
    fn build_graph(
        summaries: &[CategorySummary],
        embeddings: &[Embedding],
        num_cats: usize,
    ) -> CategoryGraph {
        // Precompute centroid pairwise distances
        let mut centroid_dists = vec![vec![0.0; num_cats]; num_cats];
        for i in 0..num_cats {
            for j in (i + 1)..num_cats {
                let d = angular_distance(
                    &summaries[i].centroid_position,
                    &summaries[j].centroid_position,
                );
                centroid_dists[i][j] = d;
                centroid_dists[j][i] = d;
            }
        }

        // Detect bridge items
        let mut bridges: HashMap<(usize, usize), Vec<BridgeItem>> = HashMap::new();

        for (ci, summary) in summaries.iter().enumerate() {
            let centroid_a = &summary.centroid_embedding;

            for &mi in &summary.member_indices {
                let item_emb = &embeddings[mi];
                let sim_to_own = cosine_similarity(&item_emb.values, centroid_a);

                for (cj, other_summary) in summaries.iter().enumerate() {
                    if ci == cj {
                        continue;
                    }

                    let sim_to_other =
                        cosine_similarity(&item_emb.values, &other_summary.centroid_embedding);

                    if sim_to_other > 0.0 && sim_to_other > sim_to_own * 0.5 {
                        let bridge_strength = if sim_to_own + sim_to_other > f64::EPSILON {
                            2.0 * sim_to_own * sim_to_other / (sim_to_own + sim_to_other)
                        } else {
                            0.0
                        };

                        bridges.entry((ci, cj)).or_default().push(BridgeItem {
                            item_index: mi,
                            source_category: ci,
                            target_category: cj,
                            affinity_to_source: sim_to_own,
                            affinity_to_target: sim_to_other,
                            bridge_strength,
                        });
                    }
                }
            }
        }

        for list in bridges.values_mut() {
            list.sort_by(|a, b| {
                b.bridge_strength
                    .partial_cmp(&a.bridge_strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut adjacency: Vec<Vec<CategoryEdge>> = vec![Vec::new(); num_cats];
        for i in 0..num_cats {
            for (j, &cd) in centroid_dists[i].iter().enumerate() {
                if i == j {
                    continue;
                }
                let bridge_list = bridges.get(&(i, j));
                let bridge_count = bridge_list.map_or(0, |b| b.len());
                let max_bridge_strength = bridge_list
                    .and_then(|b| b.first().map(|item| item.bridge_strength))
                    .unwrap_or(0.0);
                let mean_bridge_strength = bridge_list
                    .map(|b| {
                        let sum: f64 = b.iter().map(|item| item.bridge_strength).sum();
                        sum / b.len() as f64
                    })
                    .unwrap_or(0.0);

                let weight = cd / (1.0 + bridge_count as f64 * mean_bridge_strength);

                adjacency[i].push(CategoryEdge {
                    target: j,
                    centroid_distance: cd,
                    bridge_count,
                    max_bridge_strength,
                    mean_bridge_strength,
                    weight,
                });
            }
            adjacency[i].sort_by(|a, b| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        CategoryGraph { adjacency, bridges }
    }

    /// Evaluate each category and build inner spheres where they help.
    fn build_inner_spheres<P: Projection>(
        summaries: &[CategorySummary],
        embeddings: &[Embedding],
        projection: &P,
    ) -> HashMap<usize, InnerSphere> {
        let mut result = HashMap::new();

        for (ci, summary) in summaries.iter().enumerate() {
            if summary.member_count < MIN_INNER_SPHERE_SIZE {
                continue;
            }

            let member_embs: Vec<Embedding> = summary
                .member_indices
                .iter()
                .map(|&i| embeddings[i].clone())
                .collect();

            // Global subset EVR: mean certainty under global projection
            let global_subset_evr: f64 = member_embs
                .iter()
                .map(|e| projection.project_rich(e).certainty)
                .sum::<f64>()
                / member_embs.len() as f64;

            // Fit inner linear PCA
            let inner_pca = PcaProjection::fit(&member_embs, RadialStrategy::Fixed(1.0));
            let inner_linear_evr = inner_pca.explained_variance_ratio();

            if inner_linear_evr - global_subset_evr < MIN_EVR_IMPROVEMENT {
                continue;
            }

            let (inner_proj, inner_evr) = if summary.member_count >= KERNEL_PCA_MIN_SIZE {
                let inner_kpca = KernelPcaProjection::fit(&member_embs, RadialStrategy::Fixed(1.0));
                let kernel_evr = inner_kpca.explained_variance_ratio();

                if kernel_evr > inner_linear_evr + MIN_KERNEL_IMPROVEMENT {
                    (InnerProjection::KernelPca(inner_kpca), kernel_evr)
                } else {
                    (InnerProjection::LinearPca(inner_pca), inner_linear_evr)
                }
            } else {
                (InnerProjection::LinearPca(inner_pca), inner_linear_evr)
            };

            let inner_positions: Vec<SphericalPoint> =
                member_embs.iter().map(|e| inner_proj.project(e)).collect();

            result.insert(
                ci,
                InnerSphere {
                    projection: inner_proj,
                    inner_positions,
                    member_indices: summary.member_indices.clone(),
                    explained_variance_ratio: inner_evr,
                    global_subset_evr,
                    evr_improvement: inner_evr - global_subset_evr,
                },
            );
        }

        result
    }

    // ── Phase 1 query methods (unchanged) ──────────────────────────────

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.summaries.len()
    }

    /// Look up a category by name.
    pub fn get_category(&self, name: &str) -> Option<&CategorySummary> {
        self.name_to_index
            .get(name)
            .map(|&idx| &self.summaries[idx])
    }

    /// Get the k nearest neighbor categories to the given category.
    pub fn category_neighbors(&self, category_name: &str, k: usize) -> Vec<&CategorySummary> {
        let Some(&ci) = self.name_to_index.get(category_name) else {
            return Vec::new();
        };
        self.graph.adjacency[ci]
            .iter()
            .take(k)
            .map(|edge| &self.summaries[edge.target])
            .collect()
    }

    /// Get bridge items between two categories.
    pub fn bridge_items(
        &self,
        source_category: &str,
        target_category: &str,
        max_bridges: usize,
    ) -> Vec<&BridgeItem> {
        let Some(&si) = self.name_to_index.get(source_category) else {
            return Vec::new();
        };
        let Some(&ti) = self.name_to_index.get(target_category) else {
            return Vec::new();
        };
        self.graph
            .bridges
            .get(&(si, ti))
            .map(|list| list.iter().take(max_bridges).collect())
            .unwrap_or_default()
    }

    /// Find the shortest path between two categories through the category graph.
    pub fn category_path(
        &self,
        source_category: &str,
        target_category: &str,
    ) -> Option<CategoryPath> {
        let &si = self.name_to_index.get(source_category)?;
        let &ti = self.name_to_index.get(target_category)?;
        if si == ti {
            return Some(CategoryPath {
                steps: vec![CategoryPathStep {
                    category_index: si,
                    category_name: self.summaries[si].name.clone(),
                    cumulative_distance: 0.0,
                    bridges_to_next: Vec::new(),
                }],
                total_distance: 0.0,
            });
        }

        let n = self.summaries.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];

        dist[si] = 0.0;

        for _ in 0..n {
            let mut u = None;
            let mut best = f64::INFINITY;
            for (i, (&d, &v)) in dist.iter().zip(visited.iter()).enumerate() {
                if !v && d < best {
                    best = d;
                    u = Some(i);
                }
            }
            let Some(u) = u else { break };
            if u == ti {
                break;
            }
            visited[u] = true;

            for edge in &self.graph.adjacency[u] {
                let nd = dist[u] + edge.weight;
                if nd < dist[edge.target] {
                    dist[edge.target] = nd;
                    prev[edge.target] = Some(u);
                }
            }
        }

        if dist[ti].is_infinite() {
            return None;
        }

        let mut path_indices = Vec::new();
        let mut cur = ti;
        loop {
            path_indices.push(cur);
            match prev[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path_indices.reverse();

        let mut steps = Vec::with_capacity(path_indices.len());
        for (step_idx, &ci) in path_indices.iter().enumerate() {
            let bridges_to_next = if step_idx + 1 < path_indices.len() {
                let next_ci = path_indices[step_idx + 1];
                self.graph
                    .bridges
                    .get(&(ci, next_ci))
                    .map(|list| list.iter().take(3).cloned().collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            steps.push(CategoryPathStep {
                category_index: ci,
                category_name: self.summaries[ci].name.clone(),
                cumulative_distance: dist[ci],
                bridges_to_next,
            });
        }

        Some(CategoryPath {
            total_distance: dist[ti],
            steps,
        })
    }

    /// Find all categories whose centroid is within `max_angle` radians
    /// of the given embedding's projected position.
    pub fn categories_near_embedding<P: Projection>(
        &self,
        embedding: &Embedding,
        projection: &P,
        max_angle: f64,
    ) -> Vec<(usize, f64)> {
        let pos = projection.project(embedding);
        let mut results: Vec<(usize, f64)> = self
            .summaries
            .iter()
            .enumerate()
            .map(|(i, s)| (i, angular_distance(&pos, &s.centroid_position)))
            .filter(|&(_, d)| d <= max_angle)
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // ── Phase 2 query methods (inner spheres) ──────────────────────────

    /// Whether a given category has an inner sphere.
    pub fn has_inner_sphere(&self, category_name: &str) -> bool {
        self.name_to_index
            .get(category_name)
            .is_some_and(|&ci| self.inner_spheres.contains_key(&ci))
    }

    /// Get the inner sphere for a category, if one exists.
    pub fn get_inner_sphere(&self, category_name: &str) -> Option<&InnerSphere> {
        self.name_to_index
            .get(category_name)
            .and_then(|&ci| self.inner_spheres.get(&ci))
    }

    /// Number of categories that have inner spheres.
    pub fn num_inner_spheres(&self) -> usize {
        self.inner_spheres.len()
    }

    /// Drill down into a category: find the k nearest members to a query
    /// embedding, using the inner sphere's projection if available.
    ///
    /// Falls back to angular distance from the category centroid on the
    /// outer sphere when no inner sphere exists.
    pub fn drill_down(
        &self,
        category_name: &str,
        embedding: &Embedding,
        k: usize,
    ) -> Vec<DrillDownResult> {
        let Some(&ci) = self.name_to_index.get(category_name) else {
            return Vec::new();
        };
        let summary = &self.summaries[ci];

        if let Some(inner) = self.inner_spheres.get(&ci) {
            let query_pos = inner.projection.project(embedding);
            let mut results: Vec<DrillDownResult> = inner
                .inner_positions
                .iter()
                .enumerate()
                .map(|(local_idx, pos)| DrillDownResult {
                    item_index: inner.member_indices[local_idx],
                    distance: angular_distance(&query_pos, pos),
                    used_inner_sphere: true,
                })
                .collect();
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
            results
        } else {
            // Fallback: rank by distance from category centroid on outer sphere
            let centroid = &summary.centroid_position;
            let mut results: Vec<DrillDownResult> = summary
                .member_indices
                .iter()
                .map(|&mi| DrillDownResult {
                    item_index: mi,
                    distance: angular_distance(&self.outer_positions[mi], centroid),
                    used_inner_sphere: false,
                })
                .collect();
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
            results
        }
    }

    /// Drill down with an explicit outer projection for the fallback case.
    ///
    /// When no inner sphere exists, the query is projected using the
    /// provided projection and compared against stored outer positions.
    pub fn drill_down_with_projection<P: Projection>(
        &self,
        category_name: &str,
        embedding: &Embedding,
        projection: &P,
        k: usize,
    ) -> Vec<DrillDownResult> {
        let Some(&ci) = self.name_to_index.get(category_name) else {
            return Vec::new();
        };
        let summary = &self.summaries[ci];

        if let Some(inner) = self.inner_spheres.get(&ci) {
            let query_pos = inner.projection.project(embedding);
            let mut results: Vec<DrillDownResult> = inner
                .inner_positions
                .iter()
                .enumerate()
                .map(|(local_idx, pos)| DrillDownResult {
                    item_index: inner.member_indices[local_idx],
                    distance: angular_distance(&query_pos, pos),
                    used_inner_sphere: true,
                })
                .collect();
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
            results
        } else {
            let query_pos = projection.project(embedding);
            let mut results: Vec<DrillDownResult> = summary
                .member_indices
                .iter()
                .map(|&mi| DrillDownResult {
                    item_index: mi,
                    distance: angular_distance(&self.outer_positions[mi], &query_pos),
                    used_inner_sphere: false,
                })
                .collect();
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
            results
        }
    }

    /// Report which categories have inner spheres, their projection type,
    /// and EVR metrics.
    pub fn inner_sphere_stats(&self) -> Vec<InnerSphereReport> {
        let mut reports: Vec<InnerSphereReport> = self
            .inner_spheres
            .iter()
            .map(|(&ci, inner)| {
                let proj_type = match &inner.projection {
                    InnerProjection::LinearPca(_) => "LinearPca",
                    InnerProjection::KernelPca(_) => "KernelPca",
                };
                InnerSphereReport {
                    category_name: self.summaries[ci].name.clone(),
                    category_index: ci,
                    member_count: inner.member_indices.len(),
                    projection_type: proj_type,
                    inner_evr: inner.explained_variance_ratio,
                    global_subset_evr: inner.global_subset_evr,
                    evr_improvement: inner.evr_improvement,
                }
            })
            .collect();
        reports.sort_by_key(|r| r.category_index);
        reports
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let mag_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = mag_a * mag_b;
    if denom < f64::EPSILON {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    // --- Phase 1 test helpers ---

    fn test_corpus() -> (Vec<String>, Vec<Embedding>) {
        let categories = vec![
            "science".into(),
            "science".into(),
            "science".into(),
            "science".into(),
            "cooking".into(),
            "cooking".into(),
            "cooking".into(),
            "cooking".into(),
            "music".into(),
            "music".into(),
            "music".into(),
            "music".into(),
        ];
        let embeddings = vec![
            emb(&[1.0, 0.1, 0.0, 0.05, 0.02]),
            emb(&[0.9, 0.15, 0.05, 0.03, 0.01]),
            emb(&[0.95, 0.05, 0.1, 0.04, 0.03]),
            emb(&[0.85, 0.2, 0.0, 0.06, 0.01]),
            emb(&[0.1, 1.0, 0.0, 0.02, 0.05]),
            emb(&[0.15, 0.9, 0.05, 0.03, 0.04]),
            emb(&[0.05, 0.95, 0.1, 0.01, 0.06]),
            emb(&[0.2, 0.85, 0.0, 0.04, 0.03]),
            emb(&[0.0, 0.1, 1.0, 0.05, 0.02]),
            emb(&[0.05, 0.15, 0.9, 0.03, 0.01]),
            emb(&[0.1, 0.05, 0.95, 0.04, 0.03]),
            emb(&[0.0, 0.2, 0.85, 0.06, 0.01]),
        ];
        (categories, embeddings)
    }

    fn build_test_layer() -> (CategoryLayer, Vec<Embedding>, PcaProjection) {
        let (categories, embeddings) = test_corpus();
        let pca = PcaProjection::fit(&embeddings, RadialStrategy::Fixed(1.0));
        let projected: Vec<SphericalPoint> = embeddings.iter().map(|e| pca.project(e)).collect();
        let layer = CategoryLayer::build(&categories, &embeddings, &projected, &pca);
        (layer, embeddings, pca)
    }

    // --- Phase 2 test helpers ---

    fn large_category_corpus() -> (Vec<String>, Vec<Embedding>) {
        let mut categories = Vec::new();
        let mut embeddings = Vec::new();

        for i in 0..25 {
            categories.push("big".into());
            let t = i as f64 / 25.0;
            let mut v = vec![0.0; 10];
            v[0] = 1.0 + 0.3 * (t * std::f64::consts::TAU).sin();
            v[1] = 0.5 + 0.3 * (t * std::f64::consts::TAU).cos();
            v[2] = 0.2 * t;
            for (d, slot) in v.iter_mut().enumerate().take(10).skip(3) {
                *slot = 0.01 * ((i * 7 + d) as f64 % 1.0);
            }
            embeddings.push(emb(&v));
        }

        for i in 0..4 {
            categories.push("small_a".into());
            let mut v = vec![0.0; 10];
            v[5] = 1.0 + 0.1 * i as f64;
            v[6] = 0.05;
            embeddings.push(emb(&v));
        }

        for i in 0..4 {
            categories.push("small_b".into());
            let mut v = vec![0.0; 10];
            v[8] = 1.0 + 0.1 * i as f64;
            v[9] = 0.05;
            embeddings.push(emb(&v));
        }

        (categories, embeddings)
    }

    fn build_large_test_layer() -> (CategoryLayer, Vec<Embedding>, PcaProjection) {
        let (categories, embeddings) = large_category_corpus();
        let pca = PcaProjection::fit(&embeddings, RadialStrategy::Fixed(1.0));
        let projected: Vec<SphericalPoint> = embeddings.iter().map(|e| pca.project(e)).collect();
        let layer = CategoryLayer::build(&categories, &embeddings, &projected, &pca);
        (layer, embeddings, pca)
    }

    // ======== Phase 1 tests (unchanged) ========

    #[test]
    fn builds_correct_number_of_categories() {
        let (layer, _, _) = build_test_layer();
        assert_eq!(layer.num_categories(), 3);
    }

    #[test]
    fn category_names_correct() {
        let (layer, _, _) = build_test_layer();
        let names: Vec<&str> = layer.summaries.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"science"));
        assert!(names.contains(&"cooking"));
        assert!(names.contains(&"music"));
    }

    #[test]
    fn member_counts_correct() {
        let (layer, _, _) = build_test_layer();
        for summary in &layer.summaries {
            assert_eq!(summary.member_count, 4);
            assert_eq!(summary.member_indices.len(), 4);
        }
    }

    #[test]
    fn centroid_embedding_is_mean() {
        let (layer, embeddings, _) = build_test_layer();
        let science = layer.get_category("science").unwrap();
        let mut expected = vec![0.0; 5];
        for emb in embeddings.iter().take(4) {
            for (j, &v) in emb.values.iter().enumerate() {
                expected[j] += v;
            }
        }
        for v in &mut expected {
            *v /= 4.0;
        }
        for (j, (&actual, &exp)) in science
            .centroid_embedding
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (actual - exp).abs() < 1e-10,
                "centroid dim {j}: {actual} != {exp}"
            );
        }
    }

    #[test]
    fn angular_spread_is_nonnegative() {
        let (layer, _, _) = build_test_layer();
        for s in &layer.summaries {
            assert!(s.angular_spread >= 0.0);
        }
    }

    #[test]
    fn cohesion_in_range() {
        let (layer, _, _) = build_test_layer();
        for s in &layer.summaries {
            assert!(s.cohesion > 0.0 && s.cohesion <= 1.0);
        }
    }

    #[test]
    fn graph_has_edges_for_all_pairs() {
        let (layer, _, _) = build_test_layer();
        for (i, edges) in layer.graph.adjacency.iter().enumerate() {
            assert_eq!(edges.len(), layer.num_categories() - 1, "cat {i}");
        }
    }

    #[test]
    fn edge_weights_positive() {
        let (layer, _, _) = build_test_layer();
        for edges in &layer.graph.adjacency {
            for e in edges {
                assert!(e.weight > 0.0);
                assert!(e.centroid_distance > 0.0);
            }
        }
    }

    #[test]
    fn edges_sorted_by_weight() {
        let (layer, _, _) = build_test_layer();
        for edges in &layer.graph.adjacency {
            for w in edges.windows(2) {
                assert!(w[0].weight <= w[1].weight);
            }
        }
    }

    #[test]
    fn edge_bridge_strength_fields_populated() {
        let (layer, _, _) = build_test_layer();
        for edges in &layer.graph.adjacency {
            for e in edges {
                assert!(e.max_bridge_strength >= 0.0 && e.max_bridge_strength <= 1.0);
                assert!(e.mean_bridge_strength >= 0.0 && e.mean_bridge_strength <= 1.0);
                assert!(e.mean_bridge_strength <= e.max_bridge_strength + 1e-10);
                if e.bridge_count > 0 {
                    assert!(e.max_bridge_strength > 0.0);
                    assert!(e.mean_bridge_strength > 0.0);
                } else {
                    assert!(e.max_bridge_strength == 0.0);
                    assert!(e.mean_bridge_strength == 0.0);
                }
            }
        }
    }

    #[test]
    fn edge_weight_incorporates_bridge_strength() {
        let (layer, _, _) = build_test_layer();
        for edges in &layer.graph.adjacency {
            for e in edges {
                let expected =
                    e.centroid_distance / (1.0 + e.bridge_count as f64 * e.mean_bridge_strength);
                assert!(
                    (e.weight - expected).abs() < 1e-10,
                    "weight {:.6} != expected {:.6}",
                    e.weight,
                    expected
                );
            }
        }
    }

    #[test]
    fn get_category_by_name() {
        let (layer, _, _) = build_test_layer();
        assert!(layer.get_category("science").is_some());
        assert!(layer.get_category("astrology").is_none());
    }

    #[test]
    fn category_neighbors_returns_sorted() {
        let (layer, _, _) = build_test_layer();
        assert_eq!(layer.category_neighbors("science", 2).len(), 2);
    }

    #[test]
    fn category_neighbors_k_larger_than_available() {
        let (layer, _, _) = build_test_layer();
        assert_eq!(layer.category_neighbors("science", 100).len(), 2);
    }

    #[test]
    fn category_neighbors_unknown_returns_empty() {
        let (layer, _, _) = build_test_layer();
        assert!(layer.category_neighbors("nonexistent", 5).is_empty());
    }

    #[test]
    fn bridge_items_detected() {
        let (layer, _, _) = build_test_layer();
        let _ = layer.bridge_items("science", "cooking", 10);
    }

    #[test]
    fn bridge_items_unknown_category_returns_empty() {
        let (layer, _, _) = build_test_layer();
        assert!(layer.bridge_items("science", "nonexistent", 10).is_empty());
    }

    #[test]
    fn bridge_strength_in_valid_range() {
        let (layer, _, _) = build_test_layer();
        for list in layer.graph.bridges.values() {
            for b in list {
                assert!(b.bridge_strength >= 0.0 && b.bridge_strength <= 1.0);
            }
        }
    }

    #[test]
    fn bridges_sorted_by_strength() {
        let (layer, _, _) = build_test_layer();
        for list in layer.graph.bridges.values() {
            for w in list.windows(2) {
                assert!(w[0].bridge_strength >= w[1].bridge_strength);
            }
        }
    }

    #[test]
    fn category_path_same_category() {
        let (layer, _, _) = build_test_layer();
        let path = layer.category_path("science", "science").unwrap();
        assert_eq!(path.steps.len(), 1);
        assert!(path.total_distance.abs() < 1e-12);
    }

    #[test]
    fn category_path_adjacent() {
        let (layer, _, _) = build_test_layer();
        let path = layer.category_path("science", "cooking").unwrap();
        assert!(path.steps.len() >= 2);
        assert_eq!(path.steps.first().unwrap().category_name, "science");
        assert_eq!(path.steps.last().unwrap().category_name, "cooking");
        assert!(path.total_distance > 0.0);
    }

    #[test]
    fn category_path_unknown_returns_none() {
        let (layer, _, _) = build_test_layer();
        assert!(layer.category_path("science", "nonexistent").is_none());
    }

    #[test]
    fn category_path_distances_monotonic() {
        let (layer, _, _) = build_test_layer();
        let path = layer.category_path("science", "music").unwrap();
        for w in path.steps.windows(2) {
            assert!(w[1].cumulative_distance >= w[0].cumulative_distance);
        }
    }

    #[test]
    fn categories_near_embedding_finds_correct() {
        let (layer, _, pca) = build_test_layer();
        let near = layer.categories_near_embedding(
            &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]),
            &pca,
            std::f64::consts::PI,
        );
        assert!(!near.is_empty());
        assert_eq!(layer.summaries[near[0].0].name, "science");
    }

    #[test]
    fn categories_near_embedding_sorted_by_distance() {
        let (layer, _, pca) = build_test_layer();
        let near = layer.categories_near_embedding(
            &emb(&[0.5, 0.5, 0.5, 0.0, 0.0]),
            &pca,
            std::f64::consts::PI,
        );
        for w in near.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn categories_near_embedding_respects_threshold() {
        let (layer, _, pca) = build_test_layer();
        let near = layer.categories_near_embedding(&emb(&[1.0, 0.0, 0.0, 0.0, 0.0]), &pca, 0.01);
        for &(_, d) in &near {
            assert!(d <= 0.01);
        }
    }

    #[test]
    fn cosine_similarity_identical() {
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        assert!(cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_opposite() {
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        assert!(cosine_similarity(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).abs() < 1e-12);
    }

    // ======== Phase 2 tests (inner spheres) ========

    #[test]
    fn small_categories_get_no_inner_sphere() {
        let (layer, _, _) = build_test_layer();
        assert_eq!(layer.num_inner_spheres(), 0);
        assert!(!layer.has_inner_sphere("science"));
    }

    #[test]
    fn large_category_may_get_inner_sphere() {
        let (layer, _, _) = build_large_test_layer();
        assert!(!layer.has_inner_sphere("small_a"));
        assert!(!layer.has_inner_sphere("small_b"));
        let _ = layer.has_inner_sphere("big");
    }

    #[test]
    fn inner_sphere_stats_count_matches() {
        let (layer, _, _) = build_large_test_layer();
        assert_eq!(layer.inner_sphere_stats().len(), layer.num_inner_spheres());
    }

    #[test]
    fn inner_sphere_stats_sorted_by_index() {
        let (layer, _, _) = build_large_test_layer();
        let stats = layer.inner_sphere_stats();
        for w in stats.windows(2) {
            assert!(w[0].category_index <= w[1].category_index);
        }
    }

    #[test]
    fn inner_sphere_evr_improvement_positive() {
        let (layer, _, _) = build_large_test_layer();
        for inner in layer.inner_spheres.values() {
            assert!(inner.evr_improvement >= MIN_EVR_IMPROVEMENT);
        }
    }

    #[test]
    fn inner_sphere_positions_match_member_count() {
        let (layer, _, _) = build_large_test_layer();
        for (&ci, inner) in &layer.inner_spheres {
            assert_eq!(inner.inner_positions.len(), inner.member_indices.len());
            assert_eq!(inner.member_indices.len(), layer.summaries[ci].member_count);
        }
    }

    #[test]
    fn inner_sphere_member_indices_valid() {
        let (layer, _, _) = build_large_test_layer();
        let total = layer.outer_positions.len();
        for inner in layer.inner_spheres.values() {
            for &mi in &inner.member_indices {
                assert!(mi < total);
            }
        }
    }

    #[test]
    fn inner_sphere_report_projection_type_valid() {
        let (layer, _, _) = build_large_test_layer();
        for r in layer.inner_sphere_stats() {
            assert!(r.projection_type == "LinearPca" || r.projection_type == "KernelPca");
        }
    }

    #[test]
    fn inner_sphere_evr_in_range() {
        let (layer, _, _) = build_large_test_layer();
        for inner in layer.inner_spheres.values() {
            assert!(inner.explained_variance_ratio >= 0.0 && inner.explained_variance_ratio <= 1.0);
            assert!(inner.global_subset_evr >= 0.0 && inner.global_subset_evr <= 1.0);
        }
    }

    #[test]
    fn has_inner_sphere_unknown_category() {
        let (layer, _, _) = build_test_layer();
        assert!(!layer.has_inner_sphere("nonexistent"));
    }

    #[test]
    fn get_inner_sphere_returns_none_for_small() {
        let (layer, _, _) = build_test_layer();
        assert!(layer.get_inner_sphere("science").is_none());
    }

    #[test]
    fn drill_down_returns_results() {
        let (layer, _, pca) = build_large_test_layer();
        let q = emb(&[1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let results = layer.drill_down_with_projection("big", &q, &pca, 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn drill_down_sorted_by_distance() {
        let (layer, _, pca) = build_large_test_layer();
        let q = emb(&[1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let results = layer.drill_down_with_projection("big", &q, &pca, 10);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn drill_down_unknown_category_empty() {
        let (layer, _, pca) = build_large_test_layer();
        assert!(
            layer
                .drill_down_with_projection("nonexistent", &emb(&[1.0; 10]), &pca, 5)
                .is_empty()
        );
    }

    #[test]
    fn drill_down_item_indices_valid() {
        let (layer, _, pca) = build_large_test_layer();
        let q = emb(&[1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let total = layer.outer_positions.len();
        for r in layer.drill_down_with_projection("big", &q, &pca, 25) {
            assert!(r.item_index < total);
        }
    }

    #[test]
    fn drill_down_small_category_uses_outer() {
        let (layer, _, pca) = build_large_test_layer();
        let q = emb(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        for r in layer.drill_down_with_projection("small_a", &q, &pca, 4) {
            assert!(!r.used_inner_sphere);
        }
    }

    #[test]
    fn drill_down_distances_nonnegative() {
        let (layer, _, pca) = build_large_test_layer();
        for r in layer.drill_down_with_projection("big", &emb(&[1.0; 10]), &pca, 10) {
            assert!(r.distance >= 0.0);
        }
    }

    #[test]
    fn drill_down_without_projection_works() {
        let (layer, _, _) = build_large_test_layer();
        let q = emb(&[1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(layer.drill_down("big", &q, 5).len() <= 5);
    }

    #[test]
    fn inner_projection_enum_debug() {
        let corpus: Vec<Embedding> = (0..5)
            .map(|i| emb(&[i as f64, 0.0, 0.0, 0.0, 0.0]))
            .collect();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        assert_eq!(
            format!("{:?}", InnerProjection::LinearPca(pca)),
            "LinearPca"
        );
    }

    #[test]
    fn inner_projection_projects_correctly() {
        let corpus: Vec<Embedding> = (0..5)
            .map(|i| emb(&[i as f64, 0.0, 0.0, 0.0, 0.0]))
            .collect();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let proj = InnerProjection::LinearPca(pca.clone());
        let e = emb(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let sp_enum = proj.project(&e);
        let sp_direct = pca.project(&e);
        assert!((sp_enum.theta - sp_direct.theta).abs() < 1e-12);
        assert!((sp_enum.phi - sp_direct.phi).abs() < 1e-12);
    }

    #[test]
    fn inner_projection_dimensionality() {
        let corpus: Vec<Embedding> = (0..5)
            .map(|i| emb(&[i as f64, 0.0, 0.0, 0.0, 0.0]))
            .collect();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        assert_eq!(InnerProjection::LinearPca(pca).dimensionality(), 5);
    }
}
