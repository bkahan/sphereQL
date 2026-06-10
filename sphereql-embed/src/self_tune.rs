//! Post-hoc corpus self-tuning algorithm.
//!
//! The pieces from Phases 2–5 plug together to enable a corpus to
//! "improve itself" without new source fetches. This module implements
//! the inner loop:
//!
//! 1. Build a `SphereQLPipeline` from `(categories, embeddings)`
//!    derived from the corpus. For PCA configs the fit is
//!    **quality-weighted**: each concept contributes covariance mass
//!    proportional to its current `quality` (floored at
//!    [`QUALITY_WEIGHT_FLOOR`], combined with the same
//!    `1/√|category|` imbalance rebalancing the pipeline's PCA arm
//!    uses). This is what closes the loop — reweighting in step 3
//!    changes the geometry the next iteration scores, instead of
//!    mutating a field the pipeline never reads.
//! 2. Score it with [`CorpusQuality`] (Phase 5) and read the
//!    per-axis breakdown.
//! 3. Recompute each concept's `quality` from its **base quality**
//!    (the value it entered the run with) times a small set of
//!    geometry-aware multipliers (bridge classification, curvature
//!    outlier penalty, home-affinity smoothing, source-confidence
//!    smoothing). Computing from base makes the reweight idempotent:
//!    the static-attribute multipliers (home affinity, source
//!    confidence) apply once per run, not once per iteration, so
//!    quality no longer decays geometrically on attributes that never
//!    change.
//! 4. Optionally prune concepts below a quality floor, but never
//!    below the configured per-category minimum.
//! 5. Repeat until the composite score plateaus or the iteration cap
//!    is hit, then score the final (post-mutation) corpus once more —
//!    the per-iteration composites are *entry* scores, so without this
//!    the run's last reweight/prune would never be measured
//!    ([`SelfTuneReport::final_composite`]).
//!
//! The caller owns the corpus snapshot and the embed function (which
//! turns sparse `(axis, weight)` features into a dense embedding vector
//! — Phase 6's binary uses `sphereql_corpus::embed`). The loop is
//! deterministic given a fixed corpus + embed seed + pipeline config.

use std::collections::HashMap;

use crate::category::BridgeClassification;
use crate::config::{PipelineConfig, ProjectionKind};
use crate::configured_projection::ConfiguredProjection;
use crate::corpus_quality::{CorpusQuality, CorpusQualityBreakdown};
use crate::navigator::curvature_analysis;
use crate::pipeline::{PipelineInput, SphereQLPipeline};
use crate::projection::PcaProjection;
use crate::quality_metric::QualityMetric;
use crate::types::{Embedding, RadialStrategy};

/// Floor applied to a concept's `quality` when it is used as a PCA
/// covariance weight. A zero-quality concept must still be projectable
/// (items are only ever removed by pruning, never by the fit), so its
/// weight is clamped up to this value instead of vanishing from the
/// covariance entirely.
const QUALITY_WEIGHT_FLOOR: f64 = 0.05;

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
///
/// On the plateau-detecting iteration the loop stops **before**
/// reweighting or pruning, so that record carries `n_pruned = 0` and
/// `mean_quality_delta = 0.0` — the corpus was not touched.
#[derive(Debug, Clone)]
pub struct SelfTuneIteration {
    pub iteration: usize,
    pub n_concepts: usize,
    /// Composite score of the corpus **entering** this iteration
    /// (before this iteration's reweight + prune). The score of the
    /// final post-mutation corpus is
    /// [`SelfTuneReport::final_composite`].
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
    /// Composite score of the corpus as returned to the caller — i.e.
    /// **after** the final iteration's reweight and prune. The
    /// per-iteration `composite_score`s are entry scores, so this is
    /// the only measurement of the state that actually gets persisted.
    /// `None` when the final corpus is too small to build a pipeline.
    pub final_composite: Option<f64>,
}

/// Configuration for one self-tune run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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

impl SelfTuneConfig {
    /// Check that every field is in its meaningful range. Run by
    /// [`run_self_tune`] before touching the corpus.
    pub fn validate(&self) -> Result<(), String> {
        fn unit(name: &str, v: f64) -> Result<(), String> {
            if (0.0..=1.0).contains(&v) {
                Ok(())
            } else {
                Err(format!("{name} must be in [0, 1], got {v}"))
            }
        }
        unit("home_affinity_smoothing", self.home_affinity_smoothing)?;
        unit(
            "source_confidence_smoothing",
            self.source_confidence_smoothing,
        )?;
        unit("bridge_artifact_penalty", self.bridge_artifact_penalty)?;
        unit("curvature_outlier_penalty", self.curvature_outlier_penalty)?;
        if !self.bridge_genuine_boost.is_finite() || self.bridge_genuine_boost < 1.0 {
            return Err(format!(
                "bridge_genuine_boost must be >= 1.0, got {}",
                self.bridge_genuine_boost
            ));
        }
        if !self.plateau_epsilon.is_finite() || self.plateau_epsilon < 0.0 {
            return Err(format!(
                "plateau_epsilon must be finite and >= 0.0, got {}",
                self.plateau_epsilon
            ));
        }
        if self.max_iterations < 1 {
            return Err("max_iterations must be >= 1".into());
        }
        Ok(())
    }
}

/// Run one full self-tune loop.
///
/// Returns the (possibly pruned) corpus and a per-iteration report, or
/// an error if `cfg` fails [`SelfTuneConfig::validate`]. The corpus is
/// consumed by value and the mutated copy is returned — the caller is
/// responsible for persisting it (e.g. via
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
) -> Result<(Vec<TunableConcept>, SelfTuneReport), String>
where
    F: Fn(&[(usize, f64)]) -> Vec<f64>,
{
    cfg.validate()?;

    // Base qualities: the values each concept entered the run with.
    // Every iteration's reweight starts from these, so multipliers
    // never compound across iterations. Kept index-parallel to
    // `corpus`; pruning removes entries from both.
    let mut bases: Vec<f64> = corpus.iter().map(|c| c.quality).collect();

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

        // Plateau check happens before this iteration's reweight +
        // prune: detecting a plateau means the corpus already
        // converged, so mutating it once more would persist a state
        // the loop never measured.
        if iter >= 1 {
            let prev = iterations[iter - 1].composite_score;
            if (composite - prev).abs() < cfg.plateau_epsilon {
                iterations.push(SelfTuneIteration {
                    iteration: iter,
                    n_concepts: n_before,
                    composite_score: composite,
                    breakdown,
                    n_pruned: 0,
                    mean_quality: pre_mean_q,
                    mean_quality_delta: 0.0,
                });
                stopped = StopReason::Plateau;
                break;
            }
        }

        reweight_from_base(&mut corpus, &bases, &pipeline, cfg);
        let n_pruned = prune_below_floor_synced(&mut corpus, &mut bases, cfg);

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
    }

    // Exit measurement: every per-iteration composite is an entry
    // score, so the final reweight + prune would otherwise go
    // unmeasured. This is the score of the corpus the caller persists.
    let final_composite =
        build_pipeline(&corpus, &embed_fn, &base_pipeline_config).map(|p| quality.score(&p));

    Ok((
        corpus,
        SelfTuneReport {
            iterations,
            stopped_reason: stopped,
            final_composite,
        },
    ))
}

// ── Internals ────────────────────────────────────────────────────────

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
    let embeddings: Vec<Embedding> = corpus
        .iter()
        .map(|c| Embedding::new(embed_fn(&c.features)))
        .collect();

    if config.projection_kind == ProjectionKind::Pca {
        // Quality enters the geometry here: each concept's covariance
        // weight is its (floored) quality divided by √|category| — the
        // same imbalance rebalancing fit_projection_for_config's PCA
        // arm applies, scaled by how much the previous iteration
        // trusts the concept. Without this, the reweight pass mutates
        // a field the pipeline never reads and the composite can only
        // move via pruning.
        let mut cat_counts: HashMap<&str, usize> = HashMap::new();
        for c in corpus {
            *cat_counts.entry(c.category.as_str()).or_default() += 1;
        }
        let weights: Vec<f64> = corpus
            .iter()
            .map(|c| {
                c.quality.max(QUALITY_WEIGHT_FLOOR)
                    / (cat_counts[c.category.as_str()] as f64).sqrt()
            })
            .collect();
        let pca = PcaProjection::fit_weighted(&embeddings, &weights, RadialStrategy::Magnitude)
            .ok()?
            .with_volumetric(true);
        SphereQLPipeline::with_configured_projection_and_config(
            categories,
            embeddings,
            ConfiguredProjection::Pca(pca),
            config.clone(),
        )
        .ok()
    } else {
        // Non-PCA projections have no per-sample weight hook yet;
        // fall back to the standard constructor.
        let raw: Vec<Vec<f64>> = embeddings.into_iter().map(|e| e.values).collect();
        SphereQLPipeline::new_with_config(
            PipelineInput {
                categories,
                embeddings: raw,
            },
            config.clone(),
        )
        .ok()
    }
}

/// Apply all four reweight multipliers, treating each concept's
/// *current* quality as the base. Because the base is snapshotted from
/// the current qualities on every call, calling this more than once
/// compounds the multipliers. For idempotent reweighting, hold an
/// invariant bases vector and call [`reweight_from_base`] — that is
/// what the run loop does with the run-entry qualities.
pub fn reweight_in_place(
    corpus: &mut [TunableConcept],
    pipeline: &SphereQLPipeline,
    cfg: &SelfTuneConfig,
) {
    let bases: Vec<f64> = corpus.iter().map(|c| c.quality).collect();
    reweight_from_base(corpus, &bases, pipeline, cfg);
}

/// Recompute every concept's quality as `base × multipliers`, where
/// `bases` is index-parallel to `corpus`. Idempotent for a fixed
/// `(bases, pipeline)` pair: applying it twice produces the same
/// qualities as applying it once.
fn reweight_from_base(
    corpus: &mut [TunableConcept],
    bases: &[f64],
    pipeline: &SphereQLPipeline,
    cfg: &SelfTuneConfig,
) {
    debug_assert_eq!(
        corpus.len(),
        bases.len(),
        "bases must stay index-parallel to corpus"
    );
    let bridge_map = build_bridge_map(pipeline);
    let curvature_map = build_curvature_map(pipeline);

    for (i, concept) in corpus.iter_mut().enumerate() {
        let mut q = bases[i];

        // 1. Bridge classification contribution.
        if let Some(cls) = bridge_map.get(&i) {
            match cls {
                BridgeClassification::Genuine => q *= cfg.bridge_genuine_boost,
                BridgeClassification::OverlapArtifact | BridgeClassification::Weak => {
                    q *= cfg.bridge_artifact_penalty;
                }
            }
        }

        // 2. Curvature outlier penalty. Category-granular: this is a
        // trust signal about the concept's whole category, not a
        // per-concept fitness measure.
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

/// Decide which concepts the floor logic would remove. Greedy from the
/// lowest quality up, respecting `min_concepts_per_category`. Returns
/// the removal mask (index-parallel to `corpus`) and the count.
fn prune_mask(corpus: &[TunableConcept], cfg: &SelfTuneConfig) -> (Vec<bool>, usize) {
    let mut indices: Vec<usize> = (0..corpus.len()).collect();
    indices.sort_by(|a, b| corpus[*a].quality.total_cmp(&corpus[*b].quality));

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
    (to_remove, removed)
}

/// Drop the entries of `v` whose mask position is `true`.
fn apply_mask<T>(v: &mut Vec<T>, mask: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        let rm = mask[i];
        i += 1;
        !rm
    });
}

/// Prune concepts whose `quality < cfg.min_quality_to_keep`. Pruning
/// is greedy from the lowest quality up and respects
/// `min_concepts_per_category`. Returns the number of pruned concepts.
pub fn prune_below_floor(corpus: &mut Vec<TunableConcept>, cfg: &SelfTuneConfig) -> usize {
    if corpus.is_empty() {
        return 0;
    }
    let (mask, removed) = prune_mask(corpus, cfg);
    if removed == 0 {
        return 0;
    }
    apply_mask(corpus, &mask);
    removed
}

/// Prune like [`prune_below_floor`] while keeping the parallel `bases`
/// vector index-aligned with the surviving concepts.
fn prune_below_floor_synced(
    corpus: &mut Vec<TunableConcept>,
    bases: &mut Vec<f64>,
    cfg: &SelfTuneConfig,
) -> usize {
    if corpus.is_empty() {
        return 0;
    }
    let (mask, removed) = prune_mask(corpus, cfg);
    if removed == 0 {
        return 0;
    }
    apply_mask(corpus, &mask);
    apply_mask(bases, &mask);
    removed
}

// ── Tests ──────────────────────────────────────────────────────────

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

    /// 6 categories × 8 concepts, 16-dim embeddings — the smallest
    /// synthetic corpus the full pipeline fits comfortably.
    fn synthetic_corpus(n_cats: usize, n_per: usize, dim: usize) -> Vec<TunableConcept> {
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
        corpus
    }

    fn dense_embed(dim: usize) -> impl Fn(&[(usize, f64)]) -> Vec<f64> {
        move |feats: &[(usize, f64)]| -> Vec<f64> {
            let mut v = vec![0.0_f64; dim];
            for &(axis, w) in feats {
                if axis < dim {
                    v[axis] = w;
                }
            }
            v
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
    fn prune_synced_keeps_bases_aligned() {
        let mut corpus: Vec<TunableConcept> = (0..60)
            .map(|i| synthetic_concept(&format!("a{i}"), "x", 0.1, 0.5, 0.5))
            .collect();
        corpus.extend((0..60).map(|i| synthetic_concept(&format!("b{i}"), "y", 0.9, 0.9, 0.9)));
        // Tag bases with a recognizable per-item value: index as f64.
        let mut bases: Vec<f64> = (0..corpus.len()).map(|i| i as f64).collect();
        let cfg = SelfTuneConfig {
            min_quality_to_keep: 0.5,
            min_concepts_per_category: 50,
            ..Default::default()
        };
        let pruned = prune_below_floor_synced(&mut corpus, &mut bases, &cfg);
        assert_eq!(pruned, 10);
        assert_eq!(corpus.len(), bases.len());
        // Every surviving "y" concept (original indices 60..120) must
        // still carry its original base tag.
        for (c, &b) in corpus.iter().zip(bases.iter()) {
            if c.category == "y" {
                assert!((60.0..120.0).contains(&b), "base {b} misaligned");
            }
        }
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
    fn reweight_from_base_is_idempotent() {
        // Applying the reweight twice from the same bases must produce
        // exactly the same qualities as applying it once — the static
        // multipliers (home affinity, source confidence) must not
        // compound. This was the bug that made long runs decay quality
        // geometrically regardless of geometry.
        let dim = 16usize;
        let mut corpus = synthetic_corpus(6, 8, dim);
        let embed_fn = dense_embed(dim);
        let pipeline = build_pipeline(&corpus, &embed_fn, &PipelineConfig::default())
            .expect("pipeline should build");
        let cfg = SelfTuneConfig::default();
        let bases: Vec<f64> = corpus.iter().map(|c| c.quality).collect();

        reweight_from_base(&mut corpus, &bases, &pipeline, &cfg);
        let after_once: Vec<f64> = corpus.iter().map(|c| c.quality).collect();
        reweight_from_base(&mut corpus, &bases, &pipeline, &cfg);
        let after_twice: Vec<f64> = corpus.iter().map(|c| c.quality).collect();

        assert_eq!(after_once, after_twice, "reweight must be idempotent");
        // And it must actually have done something relative to base
        // (home_affinity = 0.8 < 1.0 attenuates at least once).
        assert!(after_once.iter().zip(bases.iter()).any(|(a, b)| a < b));
    }

    #[test]
    fn build_pipeline_handles_zero_quality_floor() {
        // Zero-quality concepts still get covariance mass via
        // QUALITY_WEIGHT_FLOOR — the fit must not degenerate.
        let dim = 16usize;
        let mut corpus = synthetic_corpus(6, 8, dim);
        for c in corpus.iter_mut() {
            c.quality = 0.0;
        }
        let embed_fn = dense_embed(dim);
        let pipeline = build_pipeline(&corpus, &embed_fn, &PipelineConfig::default());
        assert!(pipeline.is_some(), "floored weights must keep fit viable");
    }

    #[test]
    fn run_self_tune_returns_mutated_corpus_and_report() {
        let dim = 16usize;
        let corpus = synthetic_corpus(6, 8, dim);
        let n_total = corpus.len();
        let cfg = SelfTuneConfig {
            max_iterations: 3,
            // Keep all the synthetic concepts — the floor logic is
            // covered by the dedicated prune tests.
            min_quality_to_keep: 0.0,
            min_concepts_per_category: 1,
            ..Default::default()
        };
        let metric = CorpusQuality::default();
        let embed_fn = dense_embed(dim);

        let (out, report) =
            run_self_tune(corpus, embed_fn, PipelineConfig::default(), &metric, &cfg)
                .expect("default-derived config is valid");

        assert!(!report.iterations.is_empty());
        assert_eq!(out.len(), n_total);
        for it in &report.iterations {
            assert!((0.0..=1.0).contains(&it.composite_score));
            assert!((0.0..=1.0).contains(&it.mean_quality));
            assert!((0.0..=1.0).contains(&it.breakdown.evr));
        }
        // The exit state is measured: final_composite reflects the
        // corpus after the last reweight/prune.
        let final_score = report.final_composite.expect("final corpus is buildable");
        assert!((0.0..=1.0).contains(&final_score));
    }

    #[test]
    fn run_self_tune_quality_does_not_collapse_across_iterations() {
        // With idempotent reweighting, mean quality after iteration k
        // must not keep shrinking geometrically: iterations beyond the
        // first see |Δmean_q| near zero unless pruning or geometry
        // changes drive it. (Pre-fix, every iteration multiplied the
        // static attenuation in again: Δ ≈ -10% per iteration on this
        // corpus.)
        let dim = 16usize;
        let corpus = synthetic_corpus(6, 8, dim);
        let cfg = SelfTuneConfig {
            max_iterations: 4,
            min_quality_to_keep: 0.0,
            min_concepts_per_category: 1,
            // Disable the plateau stop so we observe several iterations.
            plateau_epsilon: 0.0,
            ..Default::default()
        };
        let metric = CorpusQuality::default();
        let embed_fn = dense_embed(dim);

        let (_, report) = run_self_tune(corpus, embed_fn, PipelineConfig::default(), &metric, &cfg)
            .expect("default-derived config is valid");

        // Iteration 0 applies the attenuation once (large delta is
        // expected). Every later iteration recomputes from base, so
        // deltas must be small — well under the pre-fix ~0.08/iter.
        for it in report.iterations.iter().skip(1) {
            assert!(
                it.mean_quality_delta.abs() < 0.02,
                "iteration {} mean_quality_delta {} suggests compounding",
                it.iteration,
                it.mean_quality_delta
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_range_smoothing() {
        let low = SelfTuneConfig {
            home_affinity_smoothing: -0.1,
            ..Default::default()
        };
        assert!(low.validate().is_err());

        let high = SelfTuneConfig {
            home_affinity_smoothing: 1.5,
            ..Default::default()
        };
        assert!(high.validate().is_err());

        assert!(SelfTuneConfig::default().validate().is_ok());
    }

    #[test]
    fn run_self_tune_surfaces_invalid_config() {
        let dim = 16usize;
        let corpus = synthetic_corpus(6, 8, dim);
        let cfg = SelfTuneConfig {
            home_affinity_smoothing: 1.5,
            ..Default::default()
        };
        let metric = CorpusQuality::default();
        let err = run_self_tune(
            corpus,
            dense_embed(dim),
            PipelineConfig::default(),
            &metric,
            &cfg,
        )
        .expect_err("out-of-range smoothing must be rejected");
        assert!(err.contains("home_affinity_smoothing"));
    }

    #[test]
    fn plateau_iteration_does_not_mutate_corpus() {
        // With a huge plateau_epsilon, iteration 1 always detects a
        // plateau. It must stop before reweighting/pruning, so the
        // returned corpus is exactly the result of iteration 0's
        // single reweight pass.
        let dim = 16usize;
        let corpus = synthetic_corpus(6, 8, dim);
        let embed_fn = dense_embed(dim);
        let cfg = SelfTuneConfig {
            max_iterations: 5,
            min_quality_to_keep: 0.0,
            min_concepts_per_category: 1,
            plateau_epsilon: 1.0,
            ..Default::default()
        };
        let metric = CorpusQuality::default();

        let mut expected = corpus.clone();
        let pipeline = build_pipeline(&expected, &embed_fn, &PipelineConfig::default())
            .expect("pipeline should build");
        let bases: Vec<f64> = expected.iter().map(|c| c.quality).collect();
        reweight_from_base(&mut expected, &bases, &pipeline, &cfg);

        let (out, report) =
            run_self_tune(corpus, embed_fn, PipelineConfig::default(), &metric, &cfg)
                .expect("config is valid");

        assert!(matches!(report.stopped_reason, StopReason::Plateau));
        assert_eq!(report.iterations.len(), 2);
        let plateau_it = report.iterations.last().unwrap();
        assert_eq!(plateau_it.n_pruned, 0);
        assert_eq!(plateau_it.mean_quality_delta, 0.0);

        let got: Vec<f64> = out.iter().map(|c| c.quality).collect();
        let want: Vec<f64> = expected.iter().map(|c| c.quality).collect();
        assert_eq!(
            got, want,
            "plateau iteration must leave qualities untouched"
        );
    }
}
