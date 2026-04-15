use std::collections::HashMap;

use sphereql_core::{angular_distance, SphericalPoint};

use crate::projection::Projection;
use crate::types::Embedding;

// ── Category summary ───────────────────────────────────────────────

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

// ── Bridge items ─────────────────────────────────────────────────

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

// ── Category graph ───────────────────────────────────────────────

/// Edge in the category adjacency graph.
#[derive(Debug, Clone)]
pub struct CategoryEdge {
    /// Index of the neighbor category.
    pub target: usize,
    /// Angular distance between category centroids on the sphere.
    pub centroid_distance: f64,
    /// Number of bridge items connecting these two categories.
    pub bridge_count: usize,
    /// Combined edge weight (lower = more connected).
    /// Computed as centroid_distance / (1 + bridge_count).
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

// ── Category-level concept path ────────────────────────────────────

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

// ── The enrichment layer ───────────────────────────────────────────

/// Category Enrichment Layer: aggregate statistics, inter-category graph,
/// and bridge item detection over a projected SphereQL corpus.
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
}

impl CategoryLayer {
    /// Build the category enrichment layer from pipeline data.
    ///
    /// - `categories[i]` is the category name for item i.
    /// - `embeddings[i]` is the raw embedding for item i.
    /// - `projected_positions[i]` is the spherical position on the outer sphere.
    /// - `projection` is used to project category centroids.
    ///
    /// O(N·C + C²) where N = total items, C = number of unique categories.
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

        CategoryLayer {
            summaries,
            name_to_index,
            graph,
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

        // Detect bridge items:
        // An item in category A is a bridge to category B if its cosine
        // similarity to B's centroid exceeds some threshold relative to
        // its similarity to A's centroid.
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

                    // Bridge criterion: similarity to foreign category is at least
                    // 50% of similarity to own category, AND positive.
                    if sim_to_other > 0.0 && sim_to_other > sim_to_own * 0.5 {
                        let bridge_strength = if sim_to_own + sim_to_other > f64::EPSILON {
                            2.0 * sim_to_own * sim_to_other / (sim_to_own + sim_to_other)
                        } else {
                            0.0
                        };

                        let bridge = BridgeItem {
                            item_index: mi,
                            source_category: ci,
                            target_category: cj,
                            affinity_to_source: sim_to_own,
                            affinity_to_target: sim_to_other,
                            bridge_strength,
                        };

                        bridges.entry((ci, cj)).or_default().push(bridge);
                    }
                }
            }
        }

        // Sort bridges by descending strength within each pair
        for list in bridges.values_mut() {
            list.sort_by(|a, b| {
                b.bridge_strength
                    .partial_cmp(&a.bridge_strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Build adjacency list
        let mut adjacency: Vec<Vec<CategoryEdge>> = vec![Vec::new(); num_cats];
        for i in 0..num_cats {
            for j in 0..num_cats {
                if i == j {
                    continue;
                }
                let bridge_count = bridges.get(&(i, j)).map_or(0, |b| b.len());
                let cd = centroid_dists[i][j];
                let weight = cd / (1.0 + bridge_count as f64);

                adjacency[i].push(CategoryEdge {
                    target: j,
                    centroid_distance: cd,
                    bridge_count,
                    weight,
                });
            }
            // Sort edges by weight (lowest = most connected)
            adjacency[i].sort_by(|a, b| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        CategoryGraph { adjacency, bridges }
    }

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
    /// Returns up to `max_bridges` items, sorted by descending bridge strength.
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
    ///
    /// Uses Dijkstra on the category adjacency graph weighted by the combined
    /// metric of centroid distance and bridge count.
    ///
    /// The result includes bridge items at each transition point.
    pub fn category_path(
        &self,
        source_category: &str,
        target_category: &str,
    ) -> Option<CategoryPath> {
        let Some(&si) = self.name_to_index.get(source_category) else {
            return None;
        };
        let Some(&ti) = self.name_to_index.get(target_category) else {
            return None;
        };
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

        // Simple Dijkstra — category count is small enough for O(C²)
        for _ in 0..n {
            // Find unvisited node with smallest distance
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

        // Reconstruct path
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

        // Build steps with bridge items at each transition
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
}

// ── Helpers ──────────────────────────────────────────────────────

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

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::PcaProjection;
    use crate::types::RadialStrategy;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    /// Build a test corpus with 3 categories, 5D embeddings.
    /// - "science":  strong in dim 0
    /// - "cooking":  strong in dim 1
    /// - "music":    strong in dim 2
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
            // science: strong dim 0
            emb(&[1.0, 0.1, 0.0, 0.05, 0.02]),
            emb(&[0.9, 0.15, 0.05, 0.03, 0.01]),
            emb(&[0.95, 0.05, 0.1, 0.04, 0.03]),
            emb(&[0.85, 0.2, 0.0, 0.06, 0.01]),
            // cooking: strong dim 1
            emb(&[0.1, 1.0, 0.0, 0.02, 0.05]),
            emb(&[0.15, 0.9, 0.05, 0.03, 0.04]),
            emb(&[0.05, 0.95, 0.1, 0.01, 0.06]),
            emb(&[0.2, 0.85, 0.0, 0.04, 0.03]),
            // music: strong dim 2
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

    // --- Construction ---

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

        // Manually compute mean of science embeddings (indices 0..4)
        let mut expected = vec![0.0; 5];
        for i in 0..4 {
            for (j, &v) in embeddings[i].values.iter().enumerate() {
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
        for summary in &layer.summaries {
            assert!(
                summary.angular_spread >= 0.0,
                "{}: spread = {}",
                summary.name,
                summary.angular_spread
            );
        }
    }

    #[test]
    fn cohesion_in_range() {
        let (layer, _, _) = build_test_layer();
        for summary in &layer.summaries {
            assert!(
                summary.cohesion > 0.0 && summary.cohesion <= 1.0,
                "{}: cohesion = {}",
                summary.name,
                summary.cohesion
            );
        }
    }

    // --- Category graph ---

    #[test]
    fn graph_has_edges_for_all_pairs() {
        let (layer, _, _) = build_test_layer();
        for (i, edges) in layer.graph.adjacency.iter().enumerate() {
            assert_eq!(
                edges.len(),
                layer.num_categories() - 1,
                "category {i} has wrong edge count"
            );
        }
    }

    #[test]
    fn edge_weights_positive() {
        let (layer, _, _) = build_test_layer();
        for edges in &layer.graph.adjacency {
            for edge in edges {
                assert!(edge.weight > 0.0, "edge weight must be positive");
                assert!(
                    edge.centroid_distance > 0.0,
                    "centroid distance must be positive"
                );
            }
        }
    }

    #[test]
    fn edges_sorted_by_weight() {
        let (layer, _, _) = build_test_layer();
        for edges in &layer.graph.adjacency {
            for w in edges.windows(2) {
                assert!(
                    w[0].weight <= w[1].weight,
                    "edges not sorted by weight: {} > {}",
                    w[0].weight,
                    w[1].weight
                );
            }
        }
    }

    // --- Category lookup ---

    #[test]
    fn get_category_by_name() {
        let (layer, _, _) = build_test_layer();
        let science = layer.get_category("science");
        assert!(science.is_some());
        assert_eq!(science.unwrap().name, "science");

        let nonexistent = layer.get_category("astrology");
        assert!(nonexistent.is_none());
    }

    #[test]
    fn category_neighbors_returns_sorted() {
        let (layer, _, _) = build_test_layer();
        let neighbors = layer.category_neighbors("science", 2);
        assert_eq!(neighbors.len(), 2);
        let names: Vec<&str> = neighbors.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"cooking") || names.contains(&"music"));
    }

    #[test]
    fn category_neighbors_k_larger_than_available() {
        let (layer, _, _) = build_test_layer();
        let neighbors = layer.category_neighbors("science", 100);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn category_neighbors_unknown_returns_empty() {
        let (layer, _, _) = build_test_layer();
        let neighbors = layer.category_neighbors("nonexistent", 5);
        assert!(neighbors.is_empty());
    }

    // --- Bridge items ---

    #[test]
    fn bridge_items_detected() {
        let (layer, _, _) = build_test_layer();
        let _bridges = layer.bridge_items("science", "cooking", 10);
    }

    #[test]
    fn bridge_items_unknown_category_returns_empty() {
        let (layer, _, _) = build_test_layer();
        let bridges = layer.bridge_items("science", "nonexistent", 10);
        assert!(bridges.is_empty());
    }

    #[test]
    fn bridge_strength_in_valid_range() {
        let (layer, _, _) = build_test_layer();
        for list in layer.graph.bridges.values() {
            for bridge in list {
                assert!(
                    bridge.bridge_strength >= 0.0 && bridge.bridge_strength <= 1.0,
                    "bridge_strength out of range: {}",
                    bridge.bridge_strength
                );
            }
        }
    }

    #[test]
    fn bridges_sorted_by_strength() {
        let (layer, _, _) = build_test_layer();
        for list in layer.graph.bridges.values() {
            for w in list.windows(2) {
                assert!(
                    w[0].bridge_strength >= w[1].bridge_strength,
                    "bridges not sorted by strength: {} < {}",
                    w[0].bridge_strength,
                    w[1].bridge_strength
                );
            }
        }
    }

    // --- Category paths ---

    #[test]
    fn category_path_same_category() {
        let (layer, _, _) = build_test_layer();
        let path = layer.category_path("science", "science");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.steps.len(), 1);
        assert!((path.total_distance - 0.0).abs() < 1e-12);
    }

    #[test]
    fn category_path_adjacent() {
        let (layer, _, _) = build_test_layer();
        let path = layer.category_path("science", "cooking");
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.steps.len() >= 2);
        assert_eq!(path.steps.first().unwrap().category_name, "science");
        assert_eq!(path.steps.last().unwrap().category_name, "cooking");
        assert!(path.total_distance > 0.0);
    }

    #[test]
    fn category_path_unknown_returns_none() {
        let (layer, _, _) = build_test_layer();
        assert!(layer.category_path("science", "nonexistent").is_none());
        assert!(layer.category_path("nonexistent", "science").is_none());
    }

    #[test]
    fn category_path_distances_monotonic() {
        let (layer, _, _) = build_test_layer();
        let path = layer.category_path("science", "music").unwrap();
        for w in path.steps.windows(2) {
            assert!(
                w[1].cumulative_distance >= w[0].cumulative_distance,
                "cumulative distances not monotonic"
            );
        }
    }

    // --- Categories near embedding ---

    #[test]
    fn categories_near_embedding_finds_correct() {
        let (layer, _, pca) = build_test_layer();
        let science_emb = emb(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let near = layer.categories_near_embedding(&science_emb, &pca, std::f64::consts::PI);
        assert!(!near.is_empty());
        let nearest_name = &layer.summaries[near[0].0].name;
        assert_eq!(nearest_name, "science");
    }

    #[test]
    fn categories_near_embedding_sorted_by_distance() {
        let (layer, _, pca) = build_test_layer();
        let query = emb(&[0.5, 0.5, 0.5, 0.0, 0.0]);
        let near = layer.categories_near_embedding(&query, &pca, std::f64::consts::PI);
        for w in near.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted by distance");
        }
    }

    #[test]
    fn categories_near_embedding_respects_threshold() {
        let (layer, _, pca) = build_test_layer();
        let query = emb(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let near = layer.categories_near_embedding(&query, &pca, 0.01);
        for &(_, d) in &near {
            assert!(d <= 0.01, "result exceeds max angle");
        }
    }

    // --- Cosine similarity helper ---

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-12);
    }
}
