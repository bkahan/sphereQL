//! Pluggable quality metrics for auto-tuning a SphereQL pipeline.
//!
//! Auto-tuning needs a scalar objective to optimize. EVR alone is a bad
//! target — a high-EVR projection can still produce a geometry where every
//! bridge is an `OverlapArtifact`, so the "quality" the user experiences
//! is low. This module defines a [`QualityMetric`] trait and three concrete
//! metrics that each measure a distinct slice of pipeline quality.
//!
//! Compose them via [`CompositeMetric`] for a weighted objective.
//!
//! All metrics produce a scalar in `[0, 1]` where higher is better, so they
//! can be mixed linearly without per-metric rescaling.

use sphereql_core::angular_distance;

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
                let sp =
                    sphereql_core::SphericalPoint::new_unchecked(ep.r, ep.theta, ep.phi);
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
    pub fn new(
        label: impl Into<String>,
        components: Vec<(Box<dyn QualityMetric>, f64)>,
    ) -> Self {
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
        assert!(s >= 0.0 && s <= 1.0, "got {s}");
    }

    #[test]
    fn bridge_coherence_in_range() {
        let p = make_pipeline();
        let s = BridgeCoherence.score(&p);
        assert!(s >= 0.0 && s <= 1.0, "got {s}");
    }

    #[test]
    fn cluster_silhouette_in_range() {
        let p = make_pipeline();
        let s = ClusterSilhouette.score(&p);
        assert!(s >= 0.0 && s <= 1.0, "got {s}");
    }

    #[test]
    fn composite_in_range() {
        let p = make_pipeline();
        let m = CompositeMetric::default_composite();
        let s = m.score(&p);
        assert!(s >= 0.0 && s <= 1.0);
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
        assert_eq!(
            CompositeMetric::default_composite().name(),
            "default_composite"
        );
    }

    #[test]
    fn silhouette_is_deterministic() {
        let p = make_pipeline();
        let a = ClusterSilhouette.score(&p);
        let b = ClusterSilhouette.score(&p);
        assert_eq!(a, b);
    }
}
