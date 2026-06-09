//! Composite corpus-quality metric.
//!
//! Combines four sub-scores into one tuner objective, each in `[0, 1]`:
//!
//! 1. **EVR** — variance explained by the projection. Pulled from
//!    [`SphereQLPipeline::explained_variance_ratio`]. Already in `[0, 1]`.
//!
//! 2. **Bridge coherence** — delegates to
//!    [`crate::quality_metric::BridgeCoherence`], so the sub-score is
//!    bit-identical to the standalone metric, including its
//!    neutral-when-no-Genuine floor
//!    ([`BRIDGE_COHERENCE_NEUTRAL`](crate::quality_metric::BRIDGE_COHERENCE_NEUTRAL)).
//!    The floor matters here: under
//!    `BridgeConfig::min_evr_for_classification`, low-EVR corpora have
//!    zero `Genuine` bridges, and a raw `genuine/total` would pin this
//!    0.30-weighted term at 0 — freezing the self-tune objective on
//!    exactly the bulk corpora it exists for.
//!
//! 3. **Curvature health** — corpus mean of `1 - clamp(|mean_excess_z|,
//!    0, 1)` across the per-category curvature signatures returned by
//!    [`curvature_analysis`]. Categories whose centroids sit close to
//!    the corpus-wide spherical-excess regime score near 1; outliers
//!    drag the score toward 0.
//!
//! 4. **Category balance** — Shannon entropy of category sizes,
//!    normalized to `[0, 1]` against `log2(n_categories)`. Tracks how
//!    evenly concepts are distributed across categories.
//!
//! Default weights (sum = 1):
//!
//! ```text
//! quality = 0.30 * EVR
//!         + 0.30 * bridge_coherence
//!         + 0.20 * curvature_health
//!         + 0.20 * category_balance
//! ```
//!
//! Weights are configurable via [`CorpusQualityWeights`]; the metric
//! normalizes by their sum, so they do not need to total 1. The metric
//! is deterministic for a given pipeline.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::navigator::curvature_analysis;
use crate::pipeline::SphereQLPipeline;
use crate::quality_metric::{BridgeCoherence, QualityMetric};

/// Weights for the four sub-scores. Must be finite, non-negative, and
/// not all zero. They do NOT need to sum to 1 — [`CorpusQuality`]
/// normalizes by their sum at score time.
#[derive(Debug, Clone, Copy)]
pub struct CorpusQualityWeights {
    pub w_evr: f64,
    pub w_bridge: f64,
    pub w_curvature: f64,
    pub w_balance: f64,
}

impl Default for CorpusQualityWeights {
    fn default() -> Self {
        Self {
            w_evr: 0.30,
            w_bridge: 0.30,
            w_curvature: 0.20,
            w_balance: 0.20,
        }
    }
}

impl CorpusQualityWeights {
    /// Returns the sum of the weights if they are valid, otherwise a
    /// human-readable error. Used at construction time and at every
    /// `score()` call to guard against in-place mutation via FFI.
    pub fn validate(&self) -> Result<f64, String> {
        let w = [self.w_evr, self.w_bridge, self.w_curvature, self.w_balance];
        for v in w {
            if !v.is_finite() {
                return Err(format!("non-finite weight: {v}"));
            }
            if v < 0.0 {
                return Err(format!("negative weight: {v}"));
            }
        }
        let total: f64 = w.iter().sum();
        if total <= 0.0 {
            return Err("all weights are zero".into());
        }
        Ok(total)
    }
}

/// Per-axis sub-scores for one [`CorpusQuality::score`] call. Returned
/// via [`CorpusQuality::last_breakdown`] so tuner reports and dashboards
/// can attribute the composite to its components.
#[derive(Debug, Clone, Copy)]
pub struct CorpusQualityBreakdown {
    pub evr: f64,
    pub bridge_coherence: f64,
    pub curvature_health: f64,
    pub category_balance: f64,
    pub composite: f64,
}

/// Composite metric: a single tuner-friendly score that fuses EVR,
/// bridge coherence, curvature health, and category balance.
///
/// Construct via [`CorpusQuality::new`] or [`CorpusQuality::default`].
/// The most recent sub-score breakdown is cached in a [`Mutex`] (the
/// `QualityMetric` trait requires `Send + Sync`) and is readable via
/// [`Self::last_breakdown`] after each call to [`Self::score`].
#[derive(Debug)]
pub struct CorpusQuality {
    weights: CorpusQualityWeights,
    last_breakdown: Mutex<Option<CorpusQualityBreakdown>>,
}

impl Default for CorpusQuality {
    fn default() -> Self {
        Self::new(CorpusQualityWeights::default())
    }
}

impl Clone for CorpusQuality {
    fn clone(&self) -> Self {
        let snap = self.last_breakdown.lock().ok().and_then(|g| *g);
        Self {
            weights: self.weights,
            last_breakdown: Mutex::new(snap),
        }
    }
}

impl CorpusQuality {
    /// Construct with explicit weights. Panics if weights are invalid
    /// (non-finite, negative, or all zero) — invalid weights are a
    /// programmer error, not a runtime condition the tuner should
    /// silently mask.
    pub fn new(weights: CorpusQualityWeights) -> Self {
        weights
            .validate()
            .expect("CorpusQualityWeights::validate failed");
        Self {
            weights,
            last_breakdown: Mutex::new(None),
        }
    }

    pub fn weights(&self) -> CorpusQualityWeights {
        self.weights
    }

    /// Snapshot of the sub-scores from the most recent `score()` call.
    /// Returns `None` before the first call.
    pub fn last_breakdown(&self) -> Option<CorpusQualityBreakdown> {
        self.last_breakdown.lock().ok().and_then(|g| *g)
    }
}

impl QualityMetric for CorpusQuality {
    fn name(&self) -> &str {
        "corpus_quality"
    }

    fn score(&self, pipeline: &SphereQLPipeline) -> f64 {
        let evr = pipeline.explained_variance_ratio().clamp(0.0, 1.0);
        let bridge_coherence = compute_bridge_coherence(pipeline);
        let curvature_health = compute_curvature_health(pipeline);
        let category_balance = compute_category_balance(pipeline.categories());

        let total = self
            .weights
            .validate()
            .expect("weights re-validated at score time");
        let composite = (self.weights.w_evr * evr
            + self.weights.w_bridge * bridge_coherence
            + self.weights.w_curvature * curvature_health
            + self.weights.w_balance * category_balance)
            / total;
        let composite = composite.clamp(0.0, 1.0);

        if let Ok(mut guard) = self.last_breakdown.lock() {
            *guard = Some(CorpusQualityBreakdown {
                evr,
                bridge_coherence,
                curvature_health,
                category_balance,
                composite,
            });
        }
        composite
    }
}

// ── Sub-score computations ─────────────────────────────────────────────

/// Delegates to the canonical [`BridgeCoherence`] metric — one
/// implementation, one set of edge-case rules. This used to be a local
/// copy of the `genuine/total` loop that predated the
/// neutral-when-no-Genuine floor; under the EVR classification gate
/// that copy pinned the sub-score at 0 on every low-EVR corpus.
fn compute_bridge_coherence(pipeline: &SphereQLPipeline) -> f64 {
    BridgeCoherence.score(pipeline)
}

fn compute_curvature_health(pipeline: &SphereQLPipeline) -> f64 {
    let layer = pipeline.category_layer();
    if layer.num_categories() < 3 {
        // Spherical excess needs three centroids; fewer than three
        // categories has nothing to bow, so treat the corpus as
        // maximally healthy.
        return 1.0;
    }
    let report = curvature_analysis(layer, 0);
    if report.signatures.is_empty() {
        return 1.0;
    }
    let mean_abs_z: f64 = report
        .signatures
        .iter()
        .map(|s| s.mean_excess_z.abs().min(1.0))
        .sum::<f64>()
        / report.signatures.len() as f64;
    (1.0 - mean_abs_z).clamp(0.0, 1.0)
}

fn compute_category_balance(categories: &[String]) -> f64 {
    if categories.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for c in categories {
        *counts.entry(c.as_str()).or_insert(0) += 1;
    }
    if counts.len() <= 1 {
        return 0.0;
    }
    let total = categories.len() as f64;
    let mut entropy = 0.0;
    for &n in counts.values() {
        let p = n as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    let max_entropy = (counts.len() as f64).log2();
    if max_entropy == 0.0 {
        0.0
    } else {
        (entropy / max_entropy).clamp(0.0, 1.0)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineInput;
    use crate::quality_metric::QualityMetric;

    #[test]
    fn weights_validate_rejects_negative() {
        let w = CorpusQualityWeights {
            w_evr: -0.1,
            w_bridge: 1.0,
            w_curvature: 1.0,
            w_balance: 1.0,
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn weights_validate_rejects_all_zero() {
        let w = CorpusQualityWeights {
            w_evr: 0.0,
            w_bridge: 0.0,
            w_curvature: 0.0,
            w_balance: 0.0,
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn category_balance_uniform_is_one() {
        let cats: Vec<String> = (0..30)
            .flat_map(|i| std::iter::repeat_n(format!("cat_{i}"), 10))
            .collect();
        let s = compute_category_balance(&cats);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn category_balance_collapses_when_one_category_dominates() {
        let mut cats: Vec<String> = std::iter::repeat_n("a".to_string(), 95).collect();
        cats.extend(std::iter::repeat_n("b".to_string(), 5));
        let s = compute_category_balance(&cats);
        assert!(s < 0.4, "expected balance < 0.4 for skewed corpus, got {s}");
    }

    #[test]
    fn default_metric_has_expected_name() {
        let m = CorpusQuality::default();
        assert_eq!(m.name(), "corpus_quality");
    }

    fn synthetic_pipeline() -> SphereQLPipeline {
        let n_per = 12usize;
        let n_cats = 8usize;
        let dim = 16usize;
        let mut categories = Vec::with_capacity(n_per * n_cats);
        let mut embeddings = Vec::with_capacity(n_per * n_cats);
        let mut rng_state: u64 = 0xDEADBEEF;
        for c in 0..n_cats {
            for _ in 0..n_per {
                categories.push(format!("cat_{c}"));
                let mut v = vec![0.0_f64; dim];
                v[c % dim] = 1.0;
                for x in v.iter_mut() {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    *x += (u - 0.5) * 0.02;
                }
                embeddings.push(v);
            }
        }
        SphereQLPipeline::new(PipelineInput {
            categories,
            embeddings,
        })
        .expect("build pipeline")
    }

    /// End-to-end smoke: build a tiny pipeline, score it, check the
    /// breakdown is populated and composite is in `[0, 1]`.
    #[test]
    fn smoke_score_on_synthetic_input() {
        let pipeline = synthetic_pipeline();
        let m = CorpusQuality::default();
        let s = m.score(&pipeline);
        assert!((0.0..=1.0).contains(&s), "composite out of range: {s}");
        let bd = m.last_breakdown().expect("breakdown populated");
        assert!((0.0..=1.0).contains(&bd.evr));
        assert!((0.0..=1.0).contains(&bd.bridge_coherence));
        assert!((0.0..=1.0).contains(&bd.curvature_health));
        assert!((0.0..=1.0).contains(&bd.category_balance));
        assert!((bd.composite - s).abs() < 1e-12);
    }

    #[test]
    fn bridge_subscore_matches_canonical_bridge_coherence() {
        // The sub-score must be bit-identical to the standalone metric
        // (including its neutral floor) — there is exactly one
        // implementation now.
        let pipeline = synthetic_pipeline();
        let m = CorpusQuality::default();
        let _ = m.score(&pipeline);
        let bd = m.last_breakdown().unwrap();
        let standalone = BridgeCoherence.score(&pipeline);
        assert_eq!(bd.bridge_coherence, standalone);
    }

    #[test]
    fn custom_weights_change_composite() {
        let n_per = 10usize;
        let n_cats = 6usize;
        let dim = 12usize;
        let mut categories = Vec::with_capacity(n_per * n_cats);
        let mut embeddings = Vec::with_capacity(n_per * n_cats);
        for c in 0..n_cats {
            for r in 0..n_per {
                categories.push(format!("cat_{c}"));
                let mut v = vec![0.0_f64; dim];
                v[c % dim] = 1.0 + (r as f64) * 0.001;
                embeddings.push(v);
            }
        }
        let input = PipelineInput {
            categories,
            embeddings,
        };
        let pipeline = SphereQLPipeline::new(input).expect("build pipeline");

        let balanced = CorpusQuality::default();
        let evr_only = CorpusQuality::new(CorpusQualityWeights {
            w_evr: 1.0,
            w_bridge: 0.0,
            w_curvature: 0.0,
            w_balance: 0.0,
        });
        let s_default = balanced.score(&pipeline);
        let s_evr = evr_only.score(&pipeline);
        // Both stay in range; with all the weight on EVR, the composite
        // collapses to clamp(EVR).
        assert!((0.0..=1.0).contains(&s_default));
        assert!((0.0..=1.0).contains(&s_evr));
        let bd = evr_only.last_breakdown().unwrap();
        assert!((s_evr - bd.evr).abs() < 1e-12);
    }
}
