//! Pluggable quality metrics for auto-tuning a SphereQL pipeline.
//!
//! Auto-tuning needs a scalar objective to optimize. EVR alone is a bad
//! target — a high-EVR projection can still produce a geometry where every
//! bridge is an `OverlapArtifact`, so the "quality" the user experiences
//! is low. This module defines a [`QualityMetric`] trait and four concrete
//! metrics that each measure a distinct slice of pipeline quality.
//!
//! Compose them via [`CompositeMetric`] for a weighted objective.
//!
//! All metrics produce a scalar in `[0, 1]` where higher is better, so they
//! can be mixed linearly without per-metric rescaling.

use std::collections::HashSet;

use sphereql_core::{SphericalPoint, angular_distance};

use crate::category::BridgeClassification;
use crate::pipeline::SphereQLPipeline;

/// A scalar pipeline-quality score.
///
/// Implementers evaluate some slice of pipeline health and return a value
/// in `[0, 1]`. Higher = better. Metrics must be deterministic given a
/// pipeline (no internal RNG), so auto-tuners can compare scores across
/// configurations without noise.
pub trait QualityMetric {
    /// Short identifier for logs and tuner reports.
    fn name(&self) -> &str;
    /// Evaluate the pipeline. Must return a value in `[0, 1]`.
    fn score(&self, pipeline: &SphereQLPipeline) -> f64;
}

// ── Territorial health ─────────────────────────────────────────────────

/// Mean pairwise `territorial_factor` across every category pair.
///
/// High = categories own distinct regions of S² (sharp boundaries).
/// Low = cap overlap dominates, and most cross-category signal degenerates
/// into shared-territory noise.
pub struct TerritorialHealth;

impl QualityMetric for TerritorialHealth {
    fn name(&self) -> &str {
        "territorial_health"
    }

    fn score(&self, pipeline: &SphereQLPipeline) -> f64 {
        let layer = pipeline.category_layer();
        let n = layer.num_categories();
        if n < 2 {
            return 1.0;
        }
        let sq = &layer.spatial_quality;
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                sum += sq.territorial_factor(i, j);
                count += 1;
            }
        }
        if count == 0 {
            1.0
        } else {
            (sum / count as f64).clamp(0.0, 1.0)
        }
    }
}

// ── Bridge coherence ───────────────────────────────────────────────────

/// Fraction of bridges classified as [`BridgeClassification::Genuine`].
///
/// High = the projection surfaces meaningful cross-domain connectors.
/// Low = most "bridges" are `OverlapArtifact` (shared cap territory) or
/// `Weak` (below median strength in a clean-territory pair). Returns `1.0`
/// if there are no bridges at all (nothing to be incoherent about).
pub struct BridgeCoherence;

impl QualityMetric for BridgeCoherence {
    fn name(&self) -> &str {
        "bridge_coherence"
    }

    fn score(&self, pipeline: &SphereQLPipeline) -> f64 {
        let layer = pipeline.category_layer();
        let mut genuine = 0usize;
        let mut total = 0usize;
        for bridges in layer.graph.bridges.values() {
            for b in bridges {
                total += 1;
                if b.classification == BridgeClassification::Genuine {
                    genuine += 1;
                }
            }
        }
        if total == 0 {
            1.0
        } else {
            genuine as f64 / total as f64
        }
    }
}

// ── Cluster silhouette ─────────────────────────────────────────────────

/// Mean silhouette score of the category assignment on the projected
/// sphere, remapped to `[0, 1]`.
///
/// For each item: `s_i = (b_i − a_i) / max(a_i, b_i)` where `a_i` is the
/// mean angular distance to other members of its own category and `b_i`
/// is the minimum mean angular distance to any other category. Native
/// silhouette lives in `[-1, 1]`; we return `(mean_s + 1) / 2`.
///
/// High = categories form tight, well-separated clusters on S².
/// Low = categories blur into each other on the projected surface.
pub struct ClusterSilhouette;

impl QualityMetric for ClusterSilhouette {
    fn name(&self) -> &str {
        "cluster_silhouette"
    }

    fn score(&self, pipeline: &SphereQLPipeline) -> f64 {
        let layer = pipeline.category_layer();
        let n_cats = layer.num_categories();
        if n_cats < 2 {
            return 1.0;
        }

        // Gather all projected positions by category using exported points
        // so we don't rely on private pipeline state.
        let exported = pipeline.exported_points();
        if exported.len() < 2 {
            return 1.0;
        }

        // Map each item to its category index (by matching name to summary).
        let mut positions_by_cat: Vec<Vec<sphereql_core::SphericalPoint>> =
            vec![Vec::new(); n_cats];
        for ep in &exported {
            if let Some(ci) = layer.name_to_index.get(&ep.category).copied() {
                let sp = sphereql_core::SphericalPoint::new_unchecked(ep.r, ep.theta, ep.phi);
                positions_by_cat[ci].push(sp);
            }
        }

        // Skip categories that somehow have zero positions (shouldn't happen
        // on a built pipeline, but guard anyway).
        let total_points: usize = positions_by_cat.iter().map(|v| v.len()).sum();
        if total_points < 2 {
            return 1.0;
        }

        let mut silhouette_sum = 0.0;
        let mut scored_points = 0usize;

        for (ci, own) in positions_by_cat.iter().enumerate() {
            if own.len() < 2 {
                // Single-member clusters have undefined silhouette — skip.
                continue;
            }
            for (idx, p) in own.iter().enumerate() {
                // a(i): mean distance to same-category peers.
                let mut a_sum = 0.0;
                for (j, q) in own.iter().enumerate() {
                    if j != idx {
                        a_sum += angular_distance(p, q);
                    }
                }
                let a = a_sum / (own.len() - 1) as f64;

                // b(i): min over other categories of mean distance.
                let mut b = f64::INFINITY;
                for (other_ci, other) in positions_by_cat.iter().enumerate() {
                    if other_ci == ci || other.is_empty() {
                        continue;
                    }
                    let sum: f64 = other.iter().map(|q| angular_distance(p, q)).sum();
                    let mean = sum / other.len() as f64;
                    if mean < b {
                        b = mean;
                    }
                }
                if b.is_infinite() {
                    continue;
                }

                let s = (b - a) / a.max(b).max(f64::EPSILON);
                silhouette_sum += s;
                scored_points += 1;
            }
        }

        if scored_points == 0 {
            return 1.0;
        }
        let mean_s = silhouette_sum / scored_points as f64;
        ((mean_s + 1.0) / 2.0).clamp(0.0, 1.0)
    }
}

// ── Graph modularity ───────────────────────────────────────────────────

/// Modularity of the category assignment on the k-NN graph of projected
/// positions on S².
///
/// Modularity measures how well a partition (here: categories) aligns
/// with the community structure of a graph. Formally for a graph with
/// `m` edges, each node `i` with degree `kᵢ`, and community assignment
/// `cᵢ`:
///
/// ```text
/// Q = Σ_c [ (L_c / m) − (D_c / 2m)² ]
/// ```
///
/// where `L_c` is the number of edges inside community `c` and `D_c` is
/// the total degree of its members. Raw `Q` lies in `[-0.5, 1.0]`. We
/// clamp to `[0, 1]` since negative modularity (anti-structured
/// partition) is strictly worse than random assignment and shouldn't be
/// rewarded differentially by the tuner.
///
/// This metric is **connectivity-native** — it evaluates "are same-category
/// points close *in the neighbor graph*" rather than "are same-category
/// points close in raw angular distance". That makes it an honest
/// objective for connectivity-preserving projections like
/// [`LaplacianEigenmapProjection`](crate::laplacian::LaplacianEigenmapProjection),
/// which otherwise get discounted by variance-centric metrics
/// ([`ClusterSilhouette`]) that reward PCA's spread.
pub struct GraphModularity {
    /// Number of nearest neighbors per node in the k-NN graph.
    ///
    /// Larger `k` smooths local structure and blurs category boundaries;
    /// smaller `k` is more sensitive to noise but resolves tighter
    /// communities. Default 15.
    pub k: usize,
}

impl Default for GraphModularity {
    fn default() -> Self {
        Self { k: 15 }
    }
}

impl GraphModularity {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl QualityMetric for GraphModularity {
    fn name(&self) -> &str {
        "graph_modularity"
    }

    fn score(&self, pipeline: &SphereQLPipeline) -> f64 {
        let layer = pipeline.category_layer();
        let n_cats = layer.num_categories();
        if n_cats < 2 {
            return 1.0;
        }

        let exported = pipeline.exported_points();
        let n = exported.len();
        if n < 2 {
            return 1.0;
        }

        // Positions + per-item category index.
        let positions: Vec<SphericalPoint> = exported
            .iter()
            .map(|p| SphericalPoint::new_unchecked(p.r, p.theta, p.phi))
            .collect();
        let item_cats: Vec<Option<usize>> = exported
            .iter()
            .map(|p| layer.name_to_index.get(&p.category).copied())
            .collect();

        let k = self.k.min(n - 1).max(1);

        // Symmetric k-NN graph: edge {i, j} exists if j ∈ top-k(i) OR
        // i ∈ top-k(j). Keyed by (min(i,j), max(i,j)) in a HashSet to
        // dedupe the union.
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, angular_distance(&positions[i], &positions[j])))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(j, _) in dists.iter().take(k) {
                let e = if i < j { (i, j) } else { (j, i) };
                edges.insert(e);
            }
        }

        let m = edges.len() as f64;
        if m < 1.0 {
            return 0.0;
        }

        // Degrees.
        let mut degree = vec![0usize; n];
        for &(a, b) in &edges {
            degree[a] += 1;
            degree[b] += 1;
        }

        // Per-category totals.
        let mut intra_edges = vec![0.0f64; n_cats];
        let mut degree_sum = vec![0.0f64; n_cats];
        for (i, &d) in degree.iter().enumerate() {
            if let Some(c) = item_cats[i] {
                degree_sum[c] += d as f64;
            }
        }
        for &(a, b) in &edges {
            if let (Some(ca), Some(cb)) = (item_cats[a], item_cats[b])
                && ca == cb
            {
                intra_edges[ca] += 1.0;
            }
        }

        let two_m = 2.0 * m;
        let q: f64 = (0..n_cats)
            .map(|c| (intra_edges[c] / m) - (degree_sum[c] / two_m).powi(2))
            .sum();

        // Clamp raw [-0.5, 1.0] range to [0, 1] — negative modularity means
        // the partition anti-correlates with the graph, which is strictly
        // worse than a random partition (Q ≈ 0) for tuner purposes.
        q.clamp(0.0, 1.0)
    }
}

// ── Composite ──────────────────────────────────────────────────────────

/// Weighted linear combination of multiple [`QualityMetric`]s.
///
/// Weights are renormalized to sum to 1.0 at construction, so the resulting
/// score stays in `[0, 1]` when every sub-metric also does.
pub struct CompositeMetric {
    label: String,
    components: Vec<(Box<dyn QualityMetric>, f64)>,
}

impl CompositeMetric {
    /// Build a composite from (metric, weight) pairs. Weights with
    /// non-positive values are dropped; remaining weights are renormalized
    /// to sum to 1.
    pub fn new(label: impl Into<String>, components: Vec<(Box<dyn QualityMetric>, f64)>) -> Self {
        let filtered: Vec<(Box<dyn QualityMetric>, f64)> =
            components.into_iter().filter(|(_, w)| *w > 0.0).collect();
        let sum: f64 = filtered.iter().map(|(_, w)| *w).sum();
        let components: Vec<(Box<dyn QualityMetric>, f64)> = if sum > 0.0 {
            filtered.into_iter().map(|(m, w)| (m, w / sum)).collect()
        } else {
            Vec::new()
        };
        Self {
            label: label.into(),
            components,
        }
    }

    /// Default composite: 40% bridge coherence + 35% territorial health +
    /// 25% cluster silhouette. Emphasizes what PCA-on-sparse-data tends to
    /// sacrifice first.
    pub fn default_composite() -> Self {
        Self::new(
            "default_composite",
            vec![
                (Box::new(BridgeCoherence) as Box<dyn QualityMetric>, 0.40),
                (Box::new(TerritorialHealth) as Box<dyn QualityMetric>, 0.35),
                (Box::new(ClusterSilhouette) as Box<dyn QualityMetric>, 0.25),
            ],
        )
    }

    /// Connectivity-native composite: 50% graph modularity + 30% bridge
    /// coherence + 20% territorial health. Designed as a counter-hypothesis
    /// to [`Self::default_composite`] — which weights the variance-centric
    /// [`ClusterSilhouette`] and systematically rewards PCA-style spread.
    /// This composite instead rewards projections that preserve
    /// same-category adjacency in the k-NN graph regardless of how much
    /// raw angular variance the projection produces.
    pub fn connectivity_composite() -> Self {
        Self::new(
            "connectivity_composite",
            vec![
                (
                    Box::new(GraphModularity::default()) as Box<dyn QualityMetric>,
                    0.50,
                ),
                (Box::new(BridgeCoherence) as Box<dyn QualityMetric>, 0.30),
                (Box::new(TerritorialHealth) as Box<dyn QualityMetric>, 0.20),
            ],
        )
    }

    /// Score each component separately. Useful for diagnostic reports where
    /// the user wants to see which sub-metric dominates the composite.
    pub fn score_components(&self, pipeline: &SphereQLPipeline) -> Vec<(String, f64, f64)> {
        self.components
            .iter()
            .map(|(m, w)| (m.name().to_string(), *w, m.score(pipeline)))
            .collect()
    }
}

impl QualityMetric for CompositeMetric {
    fn name(&self) -> &str {
        &self.label
    }

    fn score(&self, pipeline: &SphereQLPipeline) -> f64 {
        if self.components.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .components
            .iter()
            .map(|(m, w)| w * m.score(pipeline))
            .sum();
        total.clamp(0.0, 1.0)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{PipelineInput, SphereQLPipeline};

    fn make_pipeline() -> SphereQLPipeline {
        let n = 30;
        let dim = 10;
        let mut embeddings = Vec::new();
        let mut categories = Vec::new();
        for i in 0..n {
            let mut v = vec![0.0; dim];
            if i < n / 2 {
                v[0] = 1.0 + (i as f64 * 0.01);
                v[1] = 0.1;
                categories.push("alpha".to_string());
            } else {
                v[0] = 0.1;
                v[1] = 1.0 + (i as f64 * 0.01);
                categories.push("beta".to_string());
            }
            v[2] = 0.05 * i as f64;
            embeddings.push(v);
        }
        SphereQLPipeline::new(PipelineInput {
            categories,
            embeddings,
        })
        .unwrap()
    }

    #[test]
    fn territorial_health_in_range() {
        let p = make_pipeline();
        let s = TerritorialHealth.score(&p);
        assert!((0.0..=1.0).contains(&s), "got {s}");
    }

    #[test]
    fn bridge_coherence_in_range() {
        let p = make_pipeline();
        let s = BridgeCoherence.score(&p);
        assert!((0.0..=1.0).contains(&s), "got {s}");
    }

    #[test]
    fn cluster_silhouette_in_range() {
        let p = make_pipeline();
        let s = ClusterSilhouette.score(&p);
        assert!((0.0..=1.0).contains(&s), "got {s}");
    }

    #[test]
    fn composite_in_range() {
        let p = make_pipeline();
        let m = CompositeMetric::default_composite();
        let s = m.score(&p);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn composite_with_zero_weight_drops_component() {
        let m = CompositeMetric::new(
            "test",
            vec![
                (Box::new(TerritorialHealth) as Box<dyn QualityMetric>, 1.0),
                (Box::new(BridgeCoherence) as Box<dyn QualityMetric>, 0.0),
            ],
        );
        let p = make_pipeline();
        let combined = m.score(&p);
        let solo = TerritorialHealth.score(&p);
        // With BridgeCoherence dropped, composite should equal solo.
        assert!((combined - solo).abs() < 1e-12);
    }

    #[test]
    fn composite_renormalizes_weights() {
        // Absolute weights 2.0 and 3.0 should normalize to 0.4 and 0.6.
        let m = CompositeMetric::new(
            "test",
            vec![
                (Box::new(TerritorialHealth) as Box<dyn QualityMetric>, 2.0),
                (Box::new(BridgeCoherence) as Box<dyn QualityMetric>, 3.0),
            ],
        );
        let p = make_pipeline();
        let combined = m.score(&p);
        let expected = 0.4 * TerritorialHealth.score(&p) + 0.6 * BridgeCoherence.score(&p);
        assert!((combined - expected).abs() < 1e-12);
    }

    #[test]
    fn composite_component_breakdown_sums_to_score() {
        let p = make_pipeline();
        let m = CompositeMetric::default_composite();
        let breakdown = m.score_components(&p);
        let weighted: f64 = breakdown.iter().map(|(_, w, s)| w * s).sum();
        let total = m.score(&p);
        assert!((weighted - total).abs() < 1e-12);
    }

    #[test]
    fn metric_names_stable() {
        assert_eq!(TerritorialHealth.name(), "territorial_health");
        assert_eq!(BridgeCoherence.name(), "bridge_coherence");
        assert_eq!(ClusterSilhouette.name(), "cluster_silhouette");
        assert_eq!(GraphModularity::default().name(), "graph_modularity");
        assert_eq!(
            CompositeMetric::default_composite().name(),
            "default_composite"
        );
        assert_eq!(
            CompositeMetric::connectivity_composite().name(),
            "connectivity_composite"
        );
    }

    #[test]
    fn silhouette_is_deterministic() {
        let p = make_pipeline();
        let a = ClusterSilhouette.score(&p);
        let b = ClusterSilhouette.score(&p);
        assert_eq!(a, b);
    }

    #[test]
    fn graph_modularity_in_range() {
        let p = make_pipeline();
        let s = GraphModularity::default().score(&p);
        assert!((0.0..=1.0).contains(&s), "got {s}");
    }

    #[test]
    fn graph_modularity_is_deterministic() {
        let p = make_pipeline();
        let m = GraphModularity::default();
        let a = m.score(&p);
        let b = m.score(&p);
        assert_eq!(a, b);
    }

    #[test]
    fn graph_modularity_detects_real_structure() {
        // With two well-separated clusters assigned to distinct categories,
        // modularity should be materially positive. This is the property
        // that makes the metric useful: a partition aligned with graph
        // adjacency scores higher than random.
        let p = make_pipeline();
        let s = GraphModularity::default().score(&p);
        // The test corpus is deliberately well-structured (two disjoint
        // clusters, distinct categories). Modularity should comfortably
        // exceed the "random partition" baseline near 0.
        assert!(
            s > 0.1,
            "well-separated category assignment should have Q > 0.1, got {s}"
        );
    }

    #[test]
    fn graph_modularity_k_parameter_respected() {
        let p = make_pipeline();
        let s_small = GraphModularity::new(3).score(&p);
        let s_large = GraphModularity::new(15).score(&p);
        // Both should be valid scores; the exact relationship between k
        // and the score is corpus-dependent, so we only check validity.
        assert!((0.0..=1.0).contains(&s_small));
        assert!((0.0..=1.0).contains(&s_large));
    }

    #[test]
    fn connectivity_composite_weights_modularity_most() {
        // The connectivity composite should score higher than the default
        // composite when graph modularity is high — sanity check that the
        // weighted combination behaves as advertised.
        let p = make_pipeline();
        let cc = CompositeMetric::connectivity_composite();
        let breakdown = cc.score_components(&p);
        // graph_modularity should be the heaviest-weighted component.
        let heaviest = breakdown
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(heaviest.0, "graph_modularity");
        assert!((heaviest.1 - 0.50).abs() < 1e-12);
    }
}
