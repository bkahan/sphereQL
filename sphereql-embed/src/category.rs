use std::collections::HashMap;

use sphereql_core::{SphericalPoint, angular_distance, cosine_similarity};

use crate::config::PipelineConfig;
use crate::kernel_pca::KernelPcaProjection;
use crate::projection::{PcaProjection, Projection};
use crate::spatial_quality::SpatialQuality;
use crate::types::{Embedding, RadialStrategy};

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
    /// Solid angle of this category's cap on S² (steradians).
    pub cap_area: f64,
    /// Fraction of this category's cap not overlapped by any other. [0, 1].
    pub exclusivity: f64,
    /// Voronoi cell area on S² (steradians).
    pub voronoi_area: f64,
    /// Items per steradian of Voronoi cell.
    pub territorial_efficiency: f64,
    /// Mean territorial-adjusted bridge strength across all outgoing edges.
    /// 0.0 if this category has no neighbors. Populated after the graph
    /// is built in [`CategoryLayer::build`].
    pub bridge_quality: f64,
}

// ── Bridge items ───────────────────────────────────────────────────────

/// Quality classification for a bridge item.
///
/// Assigned after all bridges are collected, comparing each bridge's
/// strength against the corpus-wide median and the pair's territorial
/// separation on S².
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeClassification {
    /// Strength at or above the median and the source/target caps are
    /// spatially distinct — a real cross-domain connector.
    Genuine,
    /// The two categories overlap heavily on S² (low territorial factor).
    /// This bridge is more shared-territory noise than a genuine connector.
    OverlapArtifact,
    /// Strength below median in a territorially clean pair — a real
    /// connection but not a strong one.
    Weak,
}

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
    /// Quality label assigned after the full bridge set is observed.
    pub classification: BridgeClassification,
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
    /// Spatial confidence of this hop: bridge_strength × territorial_factor.
    /// Higher = more trustworthy transition. 0.0 for the last step.
    pub hop_confidence: f64,
}

/// Result of a category-level concept path query.
#[derive(Debug, Clone)]
pub struct CategoryPath {
    /// Ordered steps from source to target category.
    pub steps: Vec<CategoryPathStep>,
    /// Total path distance.
    pub total_distance: f64,
    /// Product of all hop confidences along the path.
    /// Low values indicate the path routes through shaky connections.
    pub path_confidence: f64,
}

// ── Inner sphere (Phase 2) ─────────────────────────────────────────────

/// The projection type used for a category's inner sphere.
///
/// Wraps either a linear PCA or kernel PCA projection, chosen
/// automatically based on the category's size and measured EVR
/// improvement over the global projection.
#[derive(Clone)]
pub enum InnerProjection {
    /// Standard linear PCA — chosen for categories meeting
    /// [`InnerSphereConfig::min_size`](crate::config::InnerSphereConfig::min_size)
    /// but below
    /// [`kernel_pca_min_size`](crate::config::InnerSphereConfig::kernel_pca_min_size),
    /// or when kernel PCA fails to improve over linear by
    /// [`min_kernel_improvement`](crate::config::InnerSphereConfig::min_kernel_improvement).
    LinearPca(PcaProjection),
    /// Gaussian kernel PCA — chosen for categories meeting
    /// [`kernel_pca_min_size`](crate::config::InnerSphereConfig::kernel_pca_min_size)
    /// where it measurably outperforms linear PCA.
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
/// 1. At least [`InnerSphereConfig::min_size`](crate::config::InnerSphereConfig::min_size) members
/// 2. Inner EVR improves over global subset EVR by ≥
///    [`InnerSphereConfig::min_evr_improvement`](crate::config::InnerSphereConfig::min_evr_improvement)
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
    /// Pre-computed spatial properties of the category layout on S².
    /// Used for bridge quality scoring, confidence signals, and routing.
    pub spatial_quality: SpatialQuality,
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
    ///
    /// Uses [`PipelineConfig::default`] for all tunables. Call
    /// [`Self::build_with_config`] to override.
    pub fn build<P: Projection>(
        categories: &[String],
        embeddings: &[Embedding],
        projected_positions: &[SphericalPoint],
        projection: &P,
        evr: f64,
    ) -> Self {
        Self::build_with_config(
            categories,
            embeddings,
            projected_positions,
            projection,
            evr,
            &PipelineConfig::default(),
        )
    }

    /// Configurable variant of [`Self::build`]. Threads a [`PipelineConfig`]
    /// through spatial quality, bridge classification, and inner-sphere
    /// gating.
    pub fn build_with_config<P: Projection>(
        categories: &[String],
        embeddings: &[Embedding],
        projected_positions: &[SphericalPoint],
        projection: &P,
        evr: f64,
        config: &PipelineConfig,
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
                // Spatial fields filled after SpatialQuality is computed
                cap_area: 0.0,
                exclusivity: 0.0,
                voronoi_area: 0.0,
                territorial_efficiency: 0.0,
                // Filled after build_graph() classifies bridges
                bridge_quality: 0.0,
            });
        }

        // 3. Compute spatial quality from category geometry
        let centroids: Vec<SphericalPoint> =
            summaries.iter().map(|s| s.centroid_position).collect();
        let half_angles: Vec<f64> = summaries.iter().map(|s| s.angular_spread).collect();
        let mut spatial_quality =
            SpatialQuality::compute_with_config(&centroids, &half_angles, evr, config);

        // Backfill spatial fields on summaries
        for (i, summary) in summaries.iter_mut().enumerate() {
            summary.cap_area = spatial_quality.cap_areas[i];
            summary.exclusivity = spatial_quality.exclusivities[i];
            summary.voronoi_area = spatial_quality.voronoi_area(i);
            summary.territorial_efficiency =
                spatial_quality.territorial_efficiency(i, summary.member_count);
        }

        // 4. Build category graph + detect bridges (spatially informed)
        let graph = Self::build_graph(&summaries, embeddings, num_cats, &spatial_quality, config);

        // Backfill bridge_quality on summaries using the just-built graph.
        // bridge_quality = mean of (edge.mean_bridge_strength × territorial_factor)
        // over all outgoing edges; 0 for isolated categories.
        for (i, summary) in summaries.iter_mut().enumerate() {
            let edges = &graph.adjacency[i];
            if edges.is_empty() {
                summary.bridge_quality = 0.0;
            } else {
                let total: f64 = edges
                    .iter()
                    .map(|e| {
                        e.mean_bridge_strength * spatial_quality.territorial_factor(i, e.target)
                    })
                    .sum();
                summary.bridge_quality = total / edges.len() as f64;
            }
        }

        // Populate C×C spatial bridge quality matrix on SpatialQuality.
        spatial_quality.set_bridge_quality_matrix(&graph);

        // 5. Build inner spheres for qualifying categories
        let inner_spheres = Self::build_inner_spheres(&summaries, embeddings, projection, config);

        CategoryLayer {
            summaries,
            name_to_index,
            graph,
            outer_positions: projected_positions.to_vec(),
            inner_spheres,
            spatial_quality,
        }
    }

    /// Build the inter-category adjacency graph and detect bridge items.
    ///
    /// Bridge detection uses the spatial quality's EVR-adaptive threshold
    /// and exclusivity-weighted strength to prevent bridge inflation at
    /// low EVR and discount bridges between overlapping categories.
    fn build_graph(
        summaries: &[CategorySummary],
        embeddings: &[Embedding],
        num_cats: usize,
        spatial: &SpatialQuality,
        config: &PipelineConfig,
    ) -> CategoryGraph {
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

        let bridge_threshold = spatial.bridge_threshold;

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

                    // EVR-adaptive threshold: stricter when projection is lossy
                    if sim_to_other > 0.0 && sim_to_other > sim_to_own * bridge_threshold {
                        let raw_strength = if sim_to_own + sim_to_other > f64::EPSILON {
                            2.0 * sim_to_own * sim_to_other / (sim_to_own + sim_to_other)
                        } else {
                            0.0
                        };

                        // Discount bridges between heavily overlapping categories.
                        // Two categories that can't be distinguished on S² produce
                        // "bridges" that are really just shared territory.
                        let territorial = spatial.territorial_factor(ci, cj);
                        let bridge_strength = raw_strength * territorial;

                        bridges.entry((ci, cj)).or_default().push(BridgeItem {
                            item_index: mi,
                            source_category: ci,
                            target_category: cj,
                            affinity_to_source: sim_to_own,
                            affinity_to_target: sim_to_other,
                            bridge_strength,
                            // Populated in the classification pass below.
                            classification: BridgeClassification::Weak,
                        });
                    }
                }
            }
        }

        // Classification pass: compare each bridge against the corpus-wide
        // median strength and the pair's territorial separation on S².
        let mut all_strengths: Vec<f64> = bridges
            .values()
            .flat_map(|list| list.iter().map(|b| b.bridge_strength))
            .collect();
        let median_strength = if all_strengths.is_empty() {
            0.0
        } else {
            all_strengths.sort_by(|a, b| a.total_cmp(b));
            all_strengths[all_strengths.len() / 2]
        };

        let overlap_threshold = config.bridges.overlap_artifact_territorial;
        for list in bridges.values_mut() {
            for b in list.iter_mut() {
                let tf = spatial.territorial_factor(b.source_category, b.target_category);
                b.classification = if tf < overlap_threshold {
                    BridgeClassification::OverlapArtifact
                } else if b.bridge_strength >= median_strength {
                    BridgeClassification::Genuine
                } else {
                    BridgeClassification::Weak
                };
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

                // Voronoi neighbors get a routing bonus — they're geometrically
                // adjacent on S², so traversal between them is natural even
                // without strong bridges.
                let voronoi_factor = if spatial.are_voronoi_neighbors(i, j) {
                    0.8
                } else {
                    1.0
                };

                let weight =
                    cd * voronoi_factor / (1.0 + bridge_count as f64 * mean_bridge_strength);

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
        config: &PipelineConfig,
    ) -> HashMap<usize, InnerSphere> {
        let mut result = HashMap::new();
        let cfg = &config.inner_sphere;

        for (ci, summary) in summaries.iter().enumerate() {
            if summary.member_count < cfg.min_size {
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

            // Fit inner linear PCA. On failure (too few members, dim
            // too low, etc.) skip this category's inner sphere
            // silently — the outer sphere still covers queries.
            let Ok(inner_pca) = PcaProjection::fit(&member_embs, RadialStrategy::Fixed(1.0)) else {
                continue;
            };
            let inner_linear_evr = inner_pca.explained_variance_ratio();

            if inner_linear_evr - global_subset_evr < cfg.min_evr_improvement {
                continue;
            }

            let (inner_proj, inner_evr) = if summary.member_count >= cfg.kernel_pca_min_size {
                // Kernel PCA can fail on degenerate subsets too; fall
                // back to the already-successful linear fit.
                match KernelPcaProjection::fit(&member_embs, RadialStrategy::Fixed(1.0)) {
                    Ok(inner_kpca) => {
                        let kernel_evr = inner_kpca.explained_variance_ratio();
                        if kernel_evr > inner_linear_evr + cfg.min_kernel_improvement {
                            (InnerProjection::KernelPca(inner_kpca), kernel_evr)
                        } else {
                            (InnerProjection::LinearPca(inner_pca), inner_linear_evr)
                        }
                    }
                    Err(_) => (InnerProjection::LinearPca(inner_pca), inner_linear_evr),
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
                    hop_confidence: 0.0,
                }],
                total_distance: 0.0,
                path_confidence: 1.0,
            });
        }

        // Dijkstra via binary-heap. Previously a linear scan over
        // `dist` per pop gave O(C²) — fine for tens of categories,
        // sloppy under the larger corpora the tuner now exercises.
        // Match the pattern already used in query.rs::concept_path.
        let n = self.summaries.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut heap = std::collections::BinaryHeap::new();

        dist[si] = 0.0;
        heap.push(HeapEntry {
            dist: 0.0,
            node: si,
        });

        while let Some(HeapEntry { dist: d, node: u }) = heap.pop() {
            if u == ti {
                break;
            }
            // Stale entry — we already relaxed this node to something
            // shorter. Skip without touching neighbors.
            if d > dist[u] {
                continue;
            }
            for edge in &self.graph.adjacency[u] {
                let nd = d + edge.weight;
                if nd < dist[edge.target] {
                    dist[edge.target] = nd;
                    prev[edge.target] = Some(u);
                    heap.push(HeapEntry {
                        dist: nd,
                        node: edge.target,
                    });
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

            let hop_confidence = if step_idx + 1 < path_indices.len() {
                let next_ci = path_indices[step_idx + 1];
                let edge_strength = self.graph.adjacency[ci]
                    .iter()
                    .find(|e| e.target == next_ci)
                    .map_or(0.0, |e| e.max_bridge_strength);
                let territorial = self.spatial_quality.territorial_factor(ci, next_ci);
                let voronoi_bonus = if self.spatial_quality.are_voronoi_neighbors(ci, next_ci) {
                    1.2
                } else {
                    1.0
                };
                (edge_strength * territorial * voronoi_bonus).min(1.0)
            } else {
                0.0
            };

            steps.push(CategoryPathStep {
                category_index: ci,
                category_name: self.summaries[ci].name.clone(),
                cumulative_distance: dist[ci],
                bridges_to_next,
                hop_confidence,
            });
        }

        let path_confidence = steps
            .iter()
            .take(steps.len().saturating_sub(1))
            .map(|s| s.hop_confidence)
            .fold(1.0, |acc, c| acc * c.max(0.01));

        Some(CategoryPath {
            total_distance: dist[ti],
            steps,
            path_confidence,
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
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Certainty-weighted category routing.
    ///
    /// Like [`Self::categories_near_embedding`] but penalizes routes through
    /// low-certainty projection regions. The effective distance is scaled
    /// by `1 / sqrt(certainty)`, so poorly-projected queries don't get
    /// routed to whatever random centroid happens to be angularly close
    /// in the distorted projection.
    ///
    /// Returns `(category_index, raw_angular_distance, effective_distance, certainty)`.
    pub fn categories_near_embedding_weighted<P: Projection>(
        &self,
        embedding: &Embedding,
        projection: &P,
        max_angle: f64,
    ) -> Vec<(usize, f64, f64, f64)> {
        let rich = projection.project_rich(embedding);
        let pos = rich.position;
        let certainty = rich.certainty.max(0.001);

        let mut results: Vec<(usize, f64, f64, f64)> = self
            .summaries
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let raw_dist = angular_distance(&pos, &s.centroid_position);
                // Low certainty inflates distance → avoids noisy routing
                let effective = raw_dist / certainty.sqrt();
                (i, raw_dist, effective, certainty)
            })
            .filter(|&(_, raw, _, _)| raw <= max_angle)
            .collect();
        results.sort_by(|a, b| a.2.total_cmp(&b.2));
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

/// Min-heap entry keyed on `dist` for [`CategoryLayer::category_path`]'s
/// Dijkstra. `BinaryHeap` is a max-heap, so `Ord` is reversed.
/// `total_cmp` is NaN-safe (NaN sorts to the end).
#[derive(PartialEq)]
struct HeapEntry {
    dist: f64,
    node: usize,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.total_cmp(&self.dist)
    }
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
        let pca = PcaProjection::fit(&embeddings, RadialStrategy::Fixed(1.0)).unwrap();
        let projected: Vec<SphericalPoint> = embeddings.iter().map(|e| pca.project(e)).collect();
        let evr = pca.explained_variance_ratio();
        let layer = CategoryLayer::build(&categories, &embeddings, &projected, &pca, evr);
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
        let pca = PcaProjection::fit(&embeddings, RadialStrategy::Fixed(1.0)).unwrap();
        let projected: Vec<SphericalPoint> = embeddings.iter().map(|e| pca.project(e)).collect();
        let evr = pca.explained_variance_ratio();
        let layer = CategoryLayer::build(&categories, &embeddings, &projected, &pca, evr);
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
                // The Voronoi bonus can only reduce weight, so weight ≤ base formula.
                // The territorial factor can also reduce bridge strength.
                let base_no_bonus =
                    e.centroid_distance / (1.0 + e.bridge_count as f64 * e.mean_bridge_strength);
                assert!(
                    e.weight <= base_no_bonus + 1e-10,
                    "weight {:.6} should be ≤ base {:.6} (Voronoi bonus reduces it)",
                    e.weight,
                    base_no_bonus,
                );
                assert!(e.weight > 0.0, "weight must be positive");
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
    fn bridge_classification_populated() {
        let (layer, _, _) = build_test_layer();
        for bridges in layer.graph.bridges.values() {
            for b in bridges {
                // Every bridge should have one of the three classifications.
                assert!(
                    b.classification == BridgeClassification::Genuine
                        || b.classification == BridgeClassification::OverlapArtifact
                        || b.classification == BridgeClassification::Weak
                );
            }
        }
    }

    #[test]
    fn bridge_quality_nonnegative() {
        let (layer, _, _) = build_test_layer();
        for s in &layer.summaries {
            assert!(
                s.bridge_quality >= 0.0,
                "{} has negative bridge_quality",
                s.name
            );
        }
    }

    #[test]
    fn bridge_quality_matrix_symmetric_ish() {
        let (layer, _, _) = build_test_layer();
        let m = &layer.spatial_quality.bridge_quality_matrix;
        let n = m.len();
        assert_eq!(n, layer.num_categories());
        for (i, row) in m.iter().enumerate() {
            assert_eq!(row.len(), n);
            assert_eq!(row[i], 0.0, "diagonal should be zero");
        }
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
        let min_improvement = PipelineConfig::default().inner_sphere.min_evr_improvement;
        for inner in layer.inner_spheres.values() {
            assert!(inner.evr_improvement >= min_improvement);
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
    fn inner_projection_enum_debug() {
        let corpus: Vec<Embedding> = (0..5)
            .map(|i| emb(&[i as f64, 0.0, 0.0, 0.0, 0.0]))
            .collect();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
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
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
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
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).unwrap();
        assert_eq!(InnerProjection::LinearPca(pca).dimensionality(), 5);
    }
}
