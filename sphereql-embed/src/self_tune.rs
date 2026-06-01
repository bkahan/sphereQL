//! Post-hoc corpus self-tuning algorithm.
//!
//! The pieces from Phases 2–5 plug together to enable a corpus to
//! "improve itself" without new source fetches. This module implements
//! the inner loop:
//!
//! 1. Build a `SphereQLPipeline` from `(categories, embeddings)`
//!    derived from the corpus.
//! 2. Score it with [`CorpusQuality`] (Phase 5) and read the
//!    per-axis breakdown.
//! 3. Reweight each concept's `quality` field by a small set of
//!    geometry-aware multipliers (bridge classification, curvature
//!    outlier penalty, home-affinity smoothing, source-confidence
//!    smoothing).
//! 4. Optionally prune concepts below a quality floor, but never
//!    below the configured per-category minimum.
//! 5. Repeat until the composite score plateaus or the iteration cap
//!    is hit.
//!
//! The caller owns the corpus snapshot and the embed function (which
//! turns sparse `(axis, weight)` features into a dense embedding vector
//! — Phase 6's binary uses `sphereql_corpus::embed`). The loop is
//! deterministic given a fixed corpus + embed seed + pipeline config.

use std::collections::HashMap;

use crate::category::BridgeClassification;
use crate::config::PipelineConfig;
use crate::corpus_quality::{CorpusQuality, CorpusQualityBreakdown};
use crate::navigator::curvature_analysis;
use crate::pipeline::{PipelineInput, SphereQLPipeline};
use crate::quality_metric::QualityMetric;

/// Concept view that the self-tuner mutates. The corpus crate's
/// [`Concept`](sphereql_corpus::Concept) uses `&'static str` for label
/// and category — fine for read-only consumers, but the self-tuner
/// owns the strings for the lifetime of the tune run and re-emits them
/// to Parquet, so we copy into owned `String`s here.
#[derive(Debug, Clone)]
pub struct TunableConcept {
    pub label: String,
    pub category: String,
    pub features: Vec<(usize, f64)>,
    pub quality: f64,
    pub axis_coherence: f64,
    pub bridge_degree: u8,
    pub source_confidence: f64,
    pub home_affinity: f64,
    pub source: Option<String>,
    pub openalex_id: Option<String>,
}

/// Per-iteration outcome.
#[derive(Debug, Clone)]
pub struct SelfTuneIteration {
    pub iteration: usize,
    pub n_concepts: usize,
    pub composite_score: f64,
    pub breakdown: CorpusQualityBreakdown,
    pub n_pruned: usize,
    pub mean_quality: f64,
    pub mean_quality_delta: f64,
}

/// Why the loop stopped.
#[derive(Debug, Clone, Copy)]
pub enum StopReason {
    /// Two consecutive composite scores were within `plateau_epsilon`.
    Plateau,
    /// Hit `max_iterations` with no plateau detected.
    MaxIterations,
    /// Pruning emptied the corpus (typically because the
    /// per-category floor and the global quality floor are
    /// incompatible with the input).
    PruneFloorHit,
}

/// Full run report.
#[derive(Debug, Clone)]
pub struct SelfTuneReport {
    pub iterations: Vec<SelfTuneIteration>,
    pub stopped_reason: StopReason,
}

/// Configuration for one self-tune run.
#[derive(Debug, Clone)]
pub struct SelfTuneConfig {
    pub max_iterations: usize,
    pub plateau_epsilon: f64,
    pub min_quality_to_keep: f64,
    pub min_concepts_per_category: usize,
    pub bridge_genuine_boost: f64,
    pub bridge_artifact_penalty: f64,
    pub curvature_outlier_penalty: f64,
    pub curvature_z_threshold: f64,
    pub home_affinity_smoothing: f64,
    pub source_confidence_smoothing: f64,
}

impl Default for SelfTuneConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            plateau_epsilon: 0.001,
            min_quality_to_keep: 0.3,
            min_concepts_per_category: 50,
            bridge_genuine_boost: 1.05,
            bridge_artifact_penalty: 0.85,
            curvature_outlier_penalty: 0.9,
            curvature_z_threshold: 1.5,
            home_affinity_smoothing: 0.5,
            source_confidence_smoothing: 0.8,
        }
    }
}

/// Run one full self-tune loop.
///
/// Returns the (possibly pruned) corpus and a per-iteration report.
/// The corpus is consumed by value and the mutated copy is returned —
/// the caller is responsible for persisting it (e.g. via
/// [`sphereql_corpus::parquet_writer::write_concepts`]).
///
/// `embed_fn` turns sparse features into the dense embedding vector
/// the pipeline expects. Phase 6's binary passes
/// `|f| sphereql_corpus::embed(f, seed)` so the same noise seed is
/// applied each iteration (the loop is deterministic).
pub fn run_self_tune<F>(
    mut corpus: Vec<TunableConcept>,
    embed_fn: F,
    base_pipeline_config: PipelineConfig,
    quality: &CorpusQuality,
    cfg: &SelfTuneConfig,
) -> (Vec<TunableConcept>, SelfTuneReport)
where
    F: Fn(&[(usize, f64)]) -> Vec<f64>,
{
    let mut iterations: Vec<SelfTuneIteration> = Vec::new();
    let mut stopped = StopReason::MaxIterations;

    for iter in 0..cfg.max_iterations {
        if corpus.is_empty() {
            stopped = StopReason::PruneFloorHit;
            break;
        }

        let pipeline = match build_pipeline(&corpus, &embed_fn, &base_pipeline_config) {
            Some(p) => p,
            None => {
                // Too few items left for the pipeline to fit. Treat
                // this as a prune-floor outcome rather than panicking;
                // the caller's report records the previous iteration's
                // state.
                stopped = StopReason::PruneFloorHit;
                break;
            }
        };

        let composite = quality.score(&pipeline);
        let breakdown = quality
            .last_breakdown()
            .expect("CorpusQuality::score populates last_breakdown");

        let n_before = corpus.len();
        let pre_mean_q: f64 = if n_before == 0 {
            0.0
        } else {
            corpus.iter().map(|c| c.quality).sum::<f64>() / n_before as f64
        };

        reweight_in_place(&mut corpus, &pipeline, cfg);
        let n_pruned = prune_below_floor(&mut corpus, cfg);

        let n_after = corpus.len().max(1) as f64;
        let post_mean_q: f64 = corpus.iter().map(|c| c.quality).sum::<f64>() / n_after;

        iterations.push(SelfTuneIteration {
            iteration: iter,
            n_concepts: n_before,
            composite_score: composite,
            breakdown,
            n_pruned,
            mean_quality: post_mean_q,
            mean_quality_delta: post_mean_q - pre_mean_q,
        });

        if iter >= 1 {
            let prev = iterations[iter - 1].composite_score;
            if (composite - prev).abs() < cfg.plateau_epsilon {
                stopped = StopReason::Plateau;
                break;
            }
        }
    }

    (
        corpus,
        SelfTuneReport {
            iterations,
            stopped_reason: stopped,
        },
    )
}

// ── Internals ──────────────────────────────────────────────────────────

fn build_pipeline<F>(
    corpus: &[TunableConcept],
    embed_fn: &F,
    config: &PipelineConfig,
) -> Option<SphereQLPipeline>
where
    F: Fn(&[(usize, f64)]) -> Vec<f64>,
{
    if corpus.len() < 3 {
        return None;
    }
    let categories: Vec<String> = corpus.iter().map(|c| c.category.clone()).collect();
    let embeddings: Vec<Vec<f64>> = corpus.iter().map(|c| embed_fn(&c.features)).collect();
    let input = PipelineInput {
        categories,
        embeddings,
    };
    SphereQLPipeline::new_with_config(input, config.clone()).ok()
}

/// Apply all four reweight multipliers in place. Public for the
/// binary, which replays the final iteration's reweight on the
/// returned corpus before writing to Parquet.
pub fn reweight_in_place(
    corpus: &mut [TunableConcept],
    pipeline: &SphereQLPipeline,
    cfg: &SelfTuneConfig,
) {
    let bridge_map = build_bridge_map(pipeline);
    let curvature_map = build_curvature_map(pipeline);

    for (i, concept) in corpus.iter_mut().enumerate() {
        let mut q = concept.quality;

        // 1. Bridge classification contribution.
        if let Some(cls) = bridge_map.get(&i) {
            match cls {
                BridgeClassification::Genuine => q *= cfg.bridge_genuine_boost,
                BridgeClassification::OverlapArtifact | BridgeClassification::Weak => {
                    q *= cfg.bridge_artifact_penalty;
                }
            }
        }

        // 2. Curvature outlier penalty.
        if let Some(z) = curvature_map.get(concept.category.as_str())
            && z.abs() > cfg.curvature_z_threshold
        {
            q *= cfg.curvature_outlier_penalty;
        }

        // 3. Home-affinity smoothing: q *= h + (1-h) * home_affinity.
        q *= cfg.home_affinity_smoothing
            + (1.0 - cfg.home_affinity_smoothing) * concept.home_affinity;

        // 4. Source-confidence smoothing: q *= s + (1-s) * source_conf.
        q *= cfg.source_confidence_smoothing
            + (1.0 - cfg.source_confidence_smoothing) * concept.source_confidence;

        concept.quality = q.clamp(0.0, 1.0);
    }
}

fn build_bridge_map(pipeline: &SphereQLPipeline) -> HashMap<usize, BridgeClassification> {
    let layer = pipeline.category_layer();
    let mut out = HashMap::new();
    for bridges in layer.graph.bridges.values() {
        for b in bridges {
            out.insert(b.item_index, b.classification);
        }
    }
    out
}

fn build_curvature_map(pipeline: &SphereQLPipeline) -> HashMap<String, f64> {
    let layer = pipeline.category_layer();
    if layer.num_categories() < 3 {
        return HashMap::new();
    }
    let report = curvature_analysis(layer, 0);
    report
        .signatures
        .into_iter()
        .map(|s| (s.category_name, s.mean_excess_z))
        .collect()
}

/// Prune concepts whose `quality < cfg.min_quality_to_keep`. Pruning
/// is greedy from the lowest quality up and respects
/// `min_concepts_per_category`. Returns the number of pruned concepts.
pub fn prune_below_floor(corpus: &mut Vec<TunableConcept>, cfg: &SelfTuneConfig) -> usize {
    if corpus.is_empty() {
        return 0;
    }
    let mut indices: Vec<usize> = (0..corpus.len()).collect();
    indices.sort_by(|a, b| {
        corpus[*a]
            .quality
            .partial_cmp(&corpus[*b].quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut counts: HashMap<String, usize> = HashMap::new();
    for c in corpus.iter() {
        *counts.entry(c.category.clone()).or_insert(0) += 1;
    }

    let mut to_remove: Vec<bool> = vec![false; corpus.len()];
    let mut removed = 0usize;
    for i in indices {
        let c = &corpus[i];
        if c.quality >= cfg.min_quality_to_keep {
            break;
        }
        let count = *counts.get(c.category.as_str()).unwrap_or(&0);
        if count <= cfg.min_concepts_per_category {
            continue;
        }
        to_remove[i] = true;
        counts.insert(c.category.clone(), count - 1);
        removed += 1;
    }

    if removed == 0 {
        return 0;
    }
    let kept: Vec<TunableConcept> = corpus
        .drain(..)
        .zip(to_remove)
        .filter_map(|(c, rm)| if rm { None } else { Some(c) })
        .collect();
    *corpus = kept;
    removed
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_concept(
        label: &str,
        category: &str,
        quality: f64,
        home_affinity: f64,
        source_confidence: f64,
    ) -> TunableConcept {
        TunableConcept {
            label: label.into(),
            category: category.into(),
            features: vec![(0, 1.0), (1, 0.5)],
            quality,
            axis_coherence: 0.7,
            bridge_degree: 1,
            source_confidence,
            home_affinity,
            source: Some("synthetic".into()),
            openalex_id: None,
        }
    }

    #[test]
    fn prune_respects_category_floor() {
        // 60 low-quality "x" rows + 60 high-quality "y" rows, with the
        // floor at 50 per category. Only 10 of the low-quality rows
        // should be pruned before "x" hits the floor.
        let mut corpus: Vec<TunableConcept> = (0..60)
            .map(|i| synthetic_concept(&format!("a{i}"), "x", 0.1, 0.5, 0.5))
            .collect();
        corpus.extend((0..60).map(|i| synthetic_concept(&format!("b{i}"), "y", 0.9, 0.9, 0.9)));
        let cfg = SelfTuneConfig {
            min_quality_to_keep: 0.5,
            min_concepts_per_category: 50,
            ..Default::default()
        };
        let pruned = prune_below_floor(&mut corpus, &cfg);
        let counts: HashMap<String, usize> = corpus.iter().fold(HashMap::new(), |mut acc, c| {
            *acc.entry(c.category.clone()).or_insert(0) += 1;
            acc
        });
        assert_eq!(counts["x"], 50);
        assert_eq!(counts["y"], 60);
        assert_eq!(pruned, 10);
    }

    #[test]
    fn prune_skips_when_quality_above_floor() {
        let mut corpus: Vec<TunableConcept> = (0..100)
            .map(|i| synthetic_concept(&format!("a{i}"), "x", 0.9, 0.9, 0.9))
            .collect();
        let cfg = SelfTuneConfig::default();
        let pruned = prune_below_floor(&mut corpus, &cfg);
        assert_eq!(pruned, 0);
        assert_eq!(corpus.len(), 100);
    }

    #[test]
    fn home_affinity_zero_halves_quality() {
        // Verify the algebra of multiplier 3 in isolation.
        let cfg = SelfTuneConfig::default();
        let pre = 1.0_f64;
        let post =
            pre * (cfg.home_affinity_smoothing + (1.0 - cfg.home_affinity_smoothing) * 0.0_f64);
        assert!((post - cfg.home_affinity_smoothing).abs() < 1e-12);
    }

    #[test]
    fn source_confidence_zero_attenuates_to_smoothing() {
        let cfg = SelfTuneConfig::default();
        let pre = 1.0_f64;
        let post = pre
            * (cfg.source_confidence_smoothing + (1.0 - cfg.source_confidence_smoothing) * 0.0_f64);
        assert!((post - cfg.source_confidence_smoothing).abs() < 1e-12);
    }

    #[test]
    fn run_self_tune_returns_mutated_corpus_and_report() {
        // Build a tiny synthetic corpus that the pipeline can actually
        // fit: 6 categories × 8 concepts, 16-dim embeddings.
        let n_per = 8usize;
        let n_cats = 6usize;
        let dim = 16usize;
        let mut corpus = Vec::with_capacity(n_per * n_cats);
        for c in 0..n_cats {
            for r in 0..n_per {
                corpus.push(TunableConcept {
                    label: format!("c{c}_r{r}"),
                    category: format!("cat_{c}"),
                    features: vec![(c % dim, 1.0)],
                    quality: 0.8,
                    axis_coherence: 0.7,
                    bridge_degree: 1,
                    source_confidence: 0.6,
                    home_affinity: 0.8,
                    source: Some("synthetic".into()),
                    openalex_id: None,
                });
            }
        }
        let cfg = SelfTuneConfig {
            max_iterations: 3,
            // Keep all the synthetic concepts — the floor logic is
            // covered by the dedicated prune tests.
            min_quality_to_keep: 0.0,
            min_concepts_per_category: 1,
            ..Default::default()
        };
        let metric = CorpusQuality::default();
        let embed_fn = |feats: &[(usize, f64)]| -> Vec<f64> {
            let mut v = vec![0.0_f64; dim];
            for &(axis, w) in feats {
                if axis < dim {
                    v[axis] = w;
                }
            }
            v
        };

        let (out, report) =
            run_self_tune(corpus, embed_fn, PipelineConfig::default(), &metric, &cfg);

        assert!(!report.iterations.is_empty());
        assert_eq!(out.len(), n_per * n_cats);
        for it in &report.iterations {
            assert!((0.0..=1.0).contains(&it.composite_score));
            assert!((0.0..=1.0).contains(&it.mean_quality));
            assert!((0.0..=1.0).contains(&it.breakdown.evr));
        }
    }
}
