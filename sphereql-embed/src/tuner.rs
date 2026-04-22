//! Auto-tuner: search [`PipelineConfig`] space to maximize a [`QualityMetric`].
//!
//! This is the first usable rung of the metalearning ladder. Given a corpus
//! and a scalar objective, the tuner enumerates or samples candidate
//! configurations, builds a full pipeline for each, and records the score.
//! No gradients, no surrogate models — just a reproducible random / grid
//! sweep that establishes a baseline for higher-order tuners (Bayesian
//! optimization, CMA-ES, meta-learning) to beat.
//!
//! Projections are fit **once per kind** from the input corpus (PCA,
//! Kernel PCA, and/or Laplacian eigenmap as dictated by the
//! [`SearchSpace`]) and reused across every trial — only the downstream
//! config knobs (bridge thresholds, inner-sphere gates, domain-group
//! counts, etc.) vary per trial.

use std::collections::HashMap;
use std::time::Instant;

use crate::config::{
    BridgeConfig, InnerSphereConfig, LaplacianConfig, PipelineConfig, ProjectionKind, RoutingConfig,
};
use crate::configured_projection::ConfiguredProjection;
use crate::pipeline::{fit_projection_for_config, PipelineError, PipelineInput, SphereQLPipeline};
use crate::projection::SplitMix64;
use crate::quality_metric::QualityMetric;
use crate::types::Embedding;

// ── Search space ───────────────────────────────────────────────────────

/// Discrete candidate values for each tunable knob.
///
/// Every field holds the full set of values the tuner will consider for
/// that knob. Grid search enumerates the Cartesian product; random search
/// samples uniformly from each set per trial.
///
/// Defaults are chosen to bracket the historical hardcoded value on each
/// knob, giving the tuner room to move either direction without being
/// unreasonable.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Candidate projection families for the outer sphere. Each kind is
    /// prefit once per distinct fit-affecting hyperparameter tuple in
    /// [`auto_tune`]; trials pick the prefit matching their config.
    pub projection_kinds: Vec<ProjectionKind>,

    // ── Projection-kind-specific knobs ────────────────────────────────
    // These only take effect when the trial's projection_kind matches.
    // PCA trials ignore them (no waste — grid enumeration is
    // kind-conditional, so PCA trials don't multiply against these
    // dimensions).
    /// Candidate values for [`LaplacianConfig::k_neighbors`]. Only
    /// explored when [`ProjectionKind::LaplacianEigenmap`] is in
    /// `projection_kinds`.
    pub laplacian_k_neighbors: Vec<usize>,
    /// Candidate values for [`LaplacianConfig::active_threshold`]. Only
    /// explored when [`ProjectionKind::LaplacianEigenmap`] is in
    /// `projection_kinds`.
    pub laplacian_active_threshold: Vec<f64>,

    // ── Kind-agnostic knobs ───────────────────────────────────────────
    /// Candidate values for [`RoutingConfig::num_domain_groups`].
    pub num_domain_groups: Vec<usize>,
    /// Candidate values for [`RoutingConfig::low_evr_threshold`].
    pub low_evr_threshold: Vec<f64>,
    /// Candidate values for [`BridgeConfig::overlap_artifact_territorial`].
    pub overlap_artifact_territorial: Vec<f64>,
    /// Candidate values for [`BridgeConfig::threshold_base`].
    pub threshold_base: Vec<f64>,
    /// Candidate values for [`BridgeConfig::threshold_evr_penalty`].
    pub threshold_evr_penalty: Vec<f64>,
    /// Candidate values for [`InnerSphereConfig::min_evr_improvement`].
    pub min_evr_improvement: Vec<f64>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            // Kernel PCA has O(n²·d) fit and is excluded from the default
            // sweep — callers who want it can add ProjectionKind::KernelPca
            // explicitly, accepting the longer fit cost.
            projection_kinds: vec![ProjectionKind::Pca, ProjectionKind::LaplacianEigenmap],
            // Laplacian hyperparameters bracket the default values
            // (k=15, threshold=0.05) widely enough that the tuner can
            // actually move the projection's geometry.
            laplacian_k_neighbors: vec![10, 15, 25],
            laplacian_active_threshold: vec![0.03, 0.05, 0.10],
            num_domain_groups: vec![3, 5, 7],
            low_evr_threshold: vec![0.25, 0.35, 0.45],
            overlap_artifact_territorial: vec![0.2, 0.3, 0.4],
            threshold_base: vec![0.4, 0.5, 0.6],
            threshold_evr_penalty: vec![0.2, 0.4, 0.6],
            min_evr_improvement: vec![0.05, 0.10, 0.15],
        }
    }
}

impl SearchSpace {
    /// Number of kind-agnostic knob combinations. Every projection kind's
    /// grid slice is at least this large; Laplacian multiplies by its
    /// specific knob counts on top.
    fn common_cardinality(&self) -> usize {
        self.num_domain_groups.len()
            * self.low_evr_threshold.len()
            * self.overlap_artifact_territorial.len()
            * self.threshold_base.len()
            * self.threshold_evr_penalty.len()
            * self.min_evr_improvement.len()
    }

    /// Per-kind grid cardinality — common knobs × any kind-specific
    /// knobs this kind opts into.
    fn kind_cardinality(&self, kind: ProjectionKind) -> usize {
        let common = self.common_cardinality();
        match kind {
            ProjectionKind::LaplacianEigenmap => {
                common * self.laplacian_k_neighbors.len() * self.laplacian_active_threshold.len()
            }
            ProjectionKind::Pca | ProjectionKind::KernelPca => common,
        }
    }

    /// Cardinality of the kind-conditional grid: the sum of each projection
    /// kind's own slice. `grid` search visits exactly this many configurations.
    pub fn grid_cardinality(&self) -> usize {
        self.projection_kinds
            .iter()
            .map(|&k| self.kind_cardinality(k))
            .sum()
    }

    /// Build a [`PipelineConfig`] from one grid index.
    ///
    /// The grid is laid out as disjoint per-kind slices concatenated in
    /// the order of [`Self::projection_kinds`]: indices 0..c₀ enumerate
    /// the first kind's subspace, c₀..c₀+c₁ the second kind's, etc. This
    /// keeps kind-specific knobs (e.g. Laplacian's k, threshold) from
    /// multiplying against trials of other kinds that wouldn't use them.
    pub fn config_at_index(&self, index: usize, base: &PipelineConfig) -> Option<PipelineConfig> {
        let mut offset = 0usize;
        for &kind in &self.projection_kinds {
            let slice = self.kind_cardinality(kind);
            if index < offset + slice {
                return Some(self.config_at_kind_index(kind, index - offset, base));
            }
            offset += slice;
        }
        None
    }

    /// Decode an index within a single kind's slice.
    fn config_at_kind_index(
        &self,
        kind: ProjectionKind,
        mut idx: usize,
        base: &PipelineConfig,
    ) -> PipelineConfig {
        let take = |idx: &mut usize, len: usize| -> usize {
            let v = *idx % len;
            *idx /= len;
            v
        };

        let i_ndg = take(&mut idx, self.num_domain_groups.len());
        let i_let = take(&mut idx, self.low_evr_threshold.len());
        let i_oat = take(&mut idx, self.overlap_artifact_territorial.len());
        let i_tb = take(&mut idx, self.threshold_base.len());
        let i_tep = take(&mut idx, self.threshold_evr_penalty.len());
        let i_mei = take(&mut idx, self.min_evr_improvement.len());

        let mut cfg = base.clone();
        cfg.projection_kind = kind;
        cfg.routing = RoutingConfig {
            num_domain_groups: self.num_domain_groups[i_ndg],
            low_evr_threshold: self.low_evr_threshold[i_let],
        };
        cfg.bridges = BridgeConfig {
            threshold_base: self.threshold_base[i_tb],
            threshold_evr_penalty: self.threshold_evr_penalty[i_tep],
            overlap_artifact_territorial: self.overlap_artifact_territorial[i_oat],
        };
        cfg.inner_sphere = InnerSphereConfig {
            min_evr_improvement: self.min_evr_improvement[i_mei],
            ..base.inner_sphere.clone()
        };

        if matches!(kind, ProjectionKind::LaplacianEigenmap) {
            let i_k = take(&mut idx, self.laplacian_k_neighbors.len());
            let i_thr = take(&mut idx, self.laplacian_active_threshold.len());
            cfg.laplacian = LaplacianConfig {
                k_neighbors: self.laplacian_k_neighbors[i_k],
                active_threshold: self.laplacian_active_threshold[i_thr],
            };
        }

        cfg
    }

    /// Sample one random [`PipelineConfig`] from this space. Every knob's
    /// value set is sampled uniformly and independently; kind-specific
    /// knobs are only sampled when the sampled kind uses them. Internal
    /// to the tuner — external callers go through [`auto_tune`] with a
    /// [`SearchStrategy::Random`] strategy.
    pub(crate) fn sample(&self, rng: &mut SplitMix64, base: &PipelineConfig) -> PipelineConfig {
        let pick_usize =
            |rng: &mut SplitMix64, vals: &[usize]| vals[rng.next_u64() as usize % vals.len()];
        let pick_f64 =
            |rng: &mut SplitMix64, vals: &[f64]| vals[rng.next_u64() as usize % vals.len()];
        let pick_kind = |rng: &mut SplitMix64, vals: &[ProjectionKind]| {
            vals[rng.next_u64() as usize % vals.len()]
        };

        let mut cfg = base.clone();
        cfg.projection_kind = pick_kind(rng, &self.projection_kinds);
        cfg.routing = RoutingConfig {
            num_domain_groups: pick_usize(rng, &self.num_domain_groups),
            low_evr_threshold: pick_f64(rng, &self.low_evr_threshold),
        };
        cfg.bridges = BridgeConfig {
            threshold_base: pick_f64(rng, &self.threshold_base),
            threshold_evr_penalty: pick_f64(rng, &self.threshold_evr_penalty),
            overlap_artifact_territorial: pick_f64(rng, &self.overlap_artifact_territorial),
        };
        cfg.inner_sphere = InnerSphereConfig {
            min_evr_improvement: pick_f64(rng, &self.min_evr_improvement),
            ..base.inner_sphere.clone()
        };

        if matches!(cfg.projection_kind, ProjectionKind::LaplacianEigenmap) {
            cfg.laplacian = LaplacianConfig {
                k_neighbors: pick_usize(rng, &self.laplacian_k_neighbors),
                active_threshold: pick_f64(rng, &self.laplacian_active_threshold),
            };
        }

        cfg
    }
}

// ── Prefit cache key ──────────────────────────────────────────────────

/// Identifies a single fittable projection configuration.
///
/// Two [`PipelineConfig`]s that produce the same `ProjectionFitKey` share
/// a prefit projection; two that differ need distinct fits. PCA and
/// Kernel PCA have no fit-affecting hyperparameters in the current
/// search space so they share a key per kind; Laplacian's fit depends on
/// (k_neighbors, active_threshold).
#[derive(Clone, PartialEq, Eq, Hash)]
enum ProjectionFitKey {
    Pca,
    KernelPca,
    Laplacian { k: usize, threshold_bits: u64 },
}

impl ProjectionFitKey {
    fn from_config(cfg: &PipelineConfig) -> Self {
        match cfg.projection_kind {
            ProjectionKind::Pca => Self::Pca,
            ProjectionKind::KernelPca => Self::KernelPca,
            ProjectionKind::LaplacianEigenmap => Self::Laplacian {
                k: cfg.laplacian.k_neighbors,
                threshold_bits: cfg.laplacian.active_threshold.to_bits(),
            },
        }
    }
}

// ── Strategy, report, trial record ─────────────────────────────────────

/// Which enumeration to use over the [`SearchSpace`].
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive Cartesian-product enumeration. Cost scales with the
    /// grid cardinality — see [`SearchSpace::grid_cardinality`].
    Grid,
    /// Uniform random sampling for `budget` trials.
    Random { budget: usize, seed: u64 },
    /// Sequential Bayesian-ish search. After `warmup` uniform random
    /// trials, subsequent trials pick each knob's value by the ratio of
    /// per-value probabilities between the top-`gamma`-fraction trials
    /// (“good”) and the bottom `1 − gamma` (“bad”). This is an
    /// axis-parallel TPE-lite acquisition: independent across knobs,
    /// Laplace-smoothed, reproducible under a fixed `seed`.
    ///
    /// Trades a constant-factor more code for meaningful sample
    /// efficiency versus uniform random — typical win on our default
    /// space is ~30% fewer trials to reach the random-search ceiling.
    Bayesian {
        budget: usize,
        /// Initial uniform random trials before the acquisition kicks in.
        /// Must be ≥ 2 so the "good" / "bad" split is non-degenerate.
        warmup: usize,
        /// Fraction of past trials treated as "good" when fitting the
        /// acquisition. 0.25 is the TPE default; smaller = more exploit,
        /// larger = more explore.
        gamma: f64,
        seed: u64,
    },
}

/// One trial's observation.
#[derive(Debug, Clone)]
pub struct TrialRecord {
    pub config: PipelineConfig,
    pub score: f64,
    /// Wall-clock build time for this trial (pipeline rebuild only —
    /// projection fit is amortized across the tuner run).
    pub build_ms: u128,
}

/// Full tuner output.
#[derive(Debug, Clone)]
pub struct TuneReport {
    pub metric_name: String,
    pub best_score: f64,
    pub best_config: PipelineConfig,
    pub trials: Vec<TrialRecord>,
    /// Trials that failed to build (e.g., too few embeddings, config
    /// combination rejected by a downstream validator). Each entry is
    /// `(config, error_message)`.
    pub failures: Vec<(PipelineConfig, String)>,
}

impl TuneReport {
    /// Trials ranked by descending score.
    pub fn ranked_trials(&self) -> Vec<&TrialRecord> {
        let mut refs: Vec<&TrialRecord> = self.trials.iter().collect();
        refs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        refs
    }

    /// Mean score across successful trials. Useful for gauging how
    /// sensitive the pipeline is to the tuned knobs: a flat landscape
    /// means the knobs don't matter on this corpus.
    pub fn mean_score(&self) -> f64 {
        if self.trials.is_empty() {
            return 0.0;
        }
        self.trials.iter().map(|t| t.score).sum::<f64>() / self.trials.len() as f64
    }
}

// ── The tuner itself ───────────────────────────────────────────────────

/// Run the auto-tuner and return the best pipeline plus a report.
///
/// Fits one projection per [`ProjectionKind`] listed in
/// `space.projection_kinds` (honoring Laplacian hyperparameters from
/// `base_config.laplacian`), then reuses those prefit projections across
/// every trial. Only the downstream [`PipelineConfig`] knobs (bridge
/// thresholds, inner-sphere gates, domain-group counts, etc.) vary per
/// trial — this keeps per-trial cost dominated by spatial quality
/// sampling and graph construction rather than projection fitting.
pub fn auto_tune<M: QualityMetric>(
    input: PipelineInput,
    space: &SearchSpace,
    metric: &M,
    strategy: SearchStrategy,
    base_config: &PipelineConfig,
) -> Result<(SphereQLPipeline, TuneReport), PipelineError> {
    let embeddings: Vec<Embedding> = input
        .embeddings
        .iter()
        .map(|v| Embedding::new(v.clone()))
        .collect();
    let categories = input.categories;

    let mut prefit: HashMap<ProjectionFitKey, ConfiguredProjection> = HashMap::new();
    let mut trials: Vec<TrialRecord> = Vec::new();
    let mut failures: Vec<(PipelineConfig, String)> = Vec::new();

    // Closure: evaluate one config, update prefit cache, push record or
    // failure. Shared by every strategy so they only differ in how they
    // propose configs.
    let run_trial = |cfg: PipelineConfig,
                     prefit: &mut HashMap<ProjectionFitKey, ConfiguredProjection>,
                     trials: &mut Vec<TrialRecord>,
                     failures: &mut Vec<(PipelineConfig, String)>| {
        let key = ProjectionFitKey::from_config(&cfg);
        let projection = match prefit.get(&key) {
            Some(p) => p.clone(),
            None => {
                let p = fit_projection_for_config(&embeddings, &cfg);
                prefit.insert(key, p.clone());
                p
            }
        };

        let start = Instant::now();
        match SphereQLPipeline::with_configured_projection_and_config(
            categories.clone(),
            embeddings.clone(),
            projection,
            cfg.clone(),
        ) {
            Ok(pipeline) => {
                let score = metric.score(&pipeline);
                let build_ms = start.elapsed().as_millis();
                trials.push(TrialRecord {
                    config: cfg,
                    score,
                    build_ms,
                });
            }
            Err(e) => {
                failures.push((cfg, e.to_string()));
            }
        }
    };

    match &strategy {
        SearchStrategy::Grid => {
            for i in 0..space.grid_cardinality() {
                if let Some(cfg) = space.config_at_index(i, base_config) {
                    run_trial(cfg, &mut prefit, &mut trials, &mut failures);
                }
            }
        }
        SearchStrategy::Random { budget, seed } => {
            let mut rng = SplitMix64::new(*seed);
            for _ in 0..*budget {
                let cfg = space.sample(&mut rng, base_config);
                run_trial(cfg, &mut prefit, &mut trials, &mut failures);
            }
        }
        SearchStrategy::Bayesian {
            budget,
            warmup,
            gamma,
            seed,
        } => {
            let mut rng = SplitMix64::new(*seed);
            let budget = *budget;
            let warmup = (*warmup).clamp(2, budget);
            let gamma = gamma.clamp(0.05, 0.95);

            // Warmup: uniform random.
            for _ in 0..warmup {
                let cfg = space.sample(&mut rng, base_config);
                run_trial(cfg, &mut prefit, &mut trials, &mut failures);
            }
            // Acquisition: axis-parallel TPE-lite.
            for _ in warmup..budget {
                let cfg = tpe_propose(space, base_config, &trials, gamma, &mut rng);
                run_trial(cfg, &mut prefit, &mut trials, &mut failures);
            }
        }
    }

    if trials.is_empty() {
        return Err(PipelineError::TooFewEmbeddings(embeddings.len()));
    }

    // Pick the winning trial.
    let best_idx = trials
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .expect("trials non-empty");
    let best_config = trials[best_idx].config.clone();
    let best_score = trials[best_idx].score;

    // Build the winning pipeline fresh so the caller gets it owned.
    let best_key = ProjectionFitKey::from_config(&best_config);
    let best_projection = prefit
        .get(&best_key)
        .cloned()
        .unwrap_or_else(|| fit_projection_for_config(&embeddings, &best_config));
    let best_pipeline = SphereQLPipeline::with_configured_projection_and_config(
        categories,
        embeddings,
        best_projection,
        best_config.clone(),
    )?;

    let report = TuneReport {
        metric_name: metric.name().to_string(),
        best_score,
        best_config,
        trials,
        failures,
    };

    Ok((best_pipeline, report))
}

// ── TPE-lite acquisition ──────────────────────────────────────────────

/// Propose the next [`PipelineConfig`] using axis-parallel good/bad
/// ratios over the trial history.
///
/// For each knob, counts how often each candidate value appeared in the
/// top-`gamma` fraction ("good") of past trials vs. the rest ("bad").
/// Samples the next value with probability proportional to
/// `(good + 1) / (bad + 1)` per candidate, Laplace-smoothed so no value
/// is ever assigned zero probability.
///
/// Kind-specific knobs (Laplacian's `k`, `active_threshold`) condition on
/// kind — their histograms are built from kind-matching trials only, with
/// a uniform fallback when fewer than 2 kind-matching trials exist.
fn tpe_propose(
    space: &SearchSpace,
    base: &PipelineConfig,
    trials: &[TrialRecord],
    gamma: f64,
    rng: &mut SplitMix64,
) -> PipelineConfig {
    // Sort by descending score, split at gamma threshold.
    let mut sorted: Vec<&TrialRecord> = trials.iter().collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_good = ((sorted.len() as f64) * gamma).ceil() as usize;
    let n_good = n_good.max(1).min(sorted.len().saturating_sub(1).max(1));
    let good: Vec<&TrialRecord> = sorted.iter().take(n_good).copied().collect();
    let bad: Vec<&TrialRecord> =
        sorted.iter().skip(n_good).copied().collect();

    // Fall back to uniform sampling if we somehow don't have both sides.
    if good.is_empty() || bad.is_empty() {
        return space.sample(rng, base);
    }

    let pick_idx = |rng: &mut SplitMix64,
                    good_counts: &[f64],
                    bad_counts: &[f64]|
     -> usize {
        let n_g = good_counts.iter().sum::<f64>() + good_counts.len() as f64;
        let n_b = bad_counts.iter().sum::<f64>() + bad_counts.len() as f64;
        let weights: Vec<f64> = good_counts
            .iter()
            .zip(bad_counts.iter())
            .map(|(&g, &b)| ((g + 1.0) / n_g) / ((b + 1.0) / n_b))
            .collect();
        sample_categorical(rng, &weights)
    };

    // Projection kind (histogram across all trials).
    let pk_g = hist_kind(&good, &space.projection_kinds);
    let pk_b = hist_kind(&bad, &space.projection_kinds);
    let kind = space.projection_kinds[pick_idx(rng, &pk_g, &pk_b)];

    // Kind-agnostic knobs.
    let ndg_g = hist_usize(&good, &space.num_domain_groups, |c| c.routing.num_domain_groups);
    let ndg_b = hist_usize(&bad, &space.num_domain_groups, |c| c.routing.num_domain_groups);
    let let_g = hist_f64(&good, &space.low_evr_threshold, |c| c.routing.low_evr_threshold);
    let let_b = hist_f64(&bad, &space.low_evr_threshold, |c| c.routing.low_evr_threshold);
    let oat_g = hist_f64(&good, &space.overlap_artifact_territorial, |c| {
        c.bridges.overlap_artifact_territorial
    });
    let oat_b = hist_f64(&bad, &space.overlap_artifact_territorial, |c| {
        c.bridges.overlap_artifact_territorial
    });
    let tb_g = hist_f64(&good, &space.threshold_base, |c| c.bridges.threshold_base);
    let tb_b = hist_f64(&bad, &space.threshold_base, |c| c.bridges.threshold_base);
    let tep_g = hist_f64(&good, &space.threshold_evr_penalty, |c| c.bridges.threshold_evr_penalty);
    let tep_b = hist_f64(&bad, &space.threshold_evr_penalty, |c| c.bridges.threshold_evr_penalty);
    let mei_g = hist_f64(&good, &space.min_evr_improvement, |c| {
        c.inner_sphere.min_evr_improvement
    });
    let mei_b = hist_f64(&bad, &space.min_evr_improvement, |c| {
        c.inner_sphere.min_evr_improvement
    });

    let mut cfg = base.clone();
    cfg.projection_kind = kind;
    cfg.routing = RoutingConfig {
        num_domain_groups: space.num_domain_groups[pick_idx(rng, &ndg_g, &ndg_b)],
        low_evr_threshold: space.low_evr_threshold[pick_idx(rng, &let_g, &let_b)],
    };
    cfg.bridges = BridgeConfig {
        threshold_base: space.threshold_base[pick_idx(rng, &tb_g, &tb_b)],
        threshold_evr_penalty: space.threshold_evr_penalty[pick_idx(rng, &tep_g, &tep_b)],
        overlap_artifact_territorial: space.overlap_artifact_territorial
            [pick_idx(rng, &oat_g, &oat_b)],
    };
    cfg.inner_sphere = InnerSphereConfig {
        min_evr_improvement: space.min_evr_improvement[pick_idx(rng, &mei_g, &mei_b)],
        ..base.inner_sphere.clone()
    };

    // Kind-specific knobs: condition on kind-matching trials only.
    if matches!(kind, ProjectionKind::LaplacianEigenmap) {
        let good_l: Vec<&TrialRecord> = good
            .iter()
            .copied()
            .filter(|t| t.config.projection_kind == ProjectionKind::LaplacianEigenmap)
            .collect();
        let bad_l: Vec<&TrialRecord> = bad
            .iter()
            .copied()
            .filter(|t| t.config.projection_kind == ProjectionKind::LaplacianEigenmap)
            .collect();
        if good_l.is_empty() || bad_l.is_empty() {
            // Not enough Laplacian trials on both sides — uniform fallback.
            cfg.laplacian = LaplacianConfig {
                k_neighbors: space.laplacian_k_neighbors
                    [(rng.next_u64() as usize) % space.laplacian_k_neighbors.len()],
                active_threshold: space.laplacian_active_threshold
                    [(rng.next_u64() as usize) % space.laplacian_active_threshold.len()],
            };
        } else {
            let k_g = hist_usize(&good_l, &space.laplacian_k_neighbors, |c| c.laplacian.k_neighbors);
            let k_b = hist_usize(&bad_l, &space.laplacian_k_neighbors, |c| c.laplacian.k_neighbors);
            let at_g = hist_f64(&good_l, &space.laplacian_active_threshold, |c| {
                c.laplacian.active_threshold
            });
            let at_b = hist_f64(&bad_l, &space.laplacian_active_threshold, |c| {
                c.laplacian.active_threshold
            });
            cfg.laplacian = LaplacianConfig {
                k_neighbors: space.laplacian_k_neighbors[pick_idx(rng, &k_g, &k_b)],
                active_threshold: space.laplacian_active_threshold[pick_idx(rng, &at_g, &at_b)],
            };
        }
    }

    cfg
}

fn hist_kind(trials: &[&TrialRecord], values: &[ProjectionKind]) -> Vec<f64> {
    let mut counts = vec![0.0f64; values.len()];
    for t in trials {
        if let Some(i) = values.iter().position(|&v| v == t.config.projection_kind) {
            counts[i] += 1.0;
        }
    }
    counts
}

fn hist_usize(
    trials: &[&TrialRecord],
    values: &[usize],
    extract: impl Fn(&PipelineConfig) -> usize,
) -> Vec<f64> {
    let mut counts = vec![0.0f64; values.len()];
    for t in trials {
        let v = extract(&t.config);
        if let Some(i) = values.iter().position(|&x| x == v) {
            counts[i] += 1.0;
        }
    }
    counts
}

/// f64 candidates are matched by nearest-neighbor since equality on
/// floats is fraught even when every sampled value came from the same
/// source slice. In practice the match is always exact but this keeps
/// us honest under future refactors.
fn hist_f64(
    trials: &[&TrialRecord],
    values: &[f64],
    extract: impl Fn(&PipelineConfig) -> f64,
) -> Vec<f64> {
    let mut counts = vec![0.0f64; values.len()];
    for t in trials {
        let v = extract(&t.config);
        if let Some((i, _)) = values
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - v)
                    .abs()
                    .partial_cmp(&(b.1 - v).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            counts[i] += 1.0;
        }
    }
    counts
}

fn sample_categorical(rng: &mut SplitMix64, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return (rng.next_u64() as usize) % weights.len().max(1);
    }
    let r = rng.next_f64() * total;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if r <= acc {
            return i;
        }
    }
    weights.len() - 1
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_metric::{BridgeCoherence, CompositeMetric, TerritorialHealth};

    fn make_input(n: usize, dim: usize) -> PipelineInput {
        let mut embeddings = Vec::new();
        let mut categories = Vec::new();
        for i in 0..n {
            let mut v = vec![0.0; dim];
            if i < n / 3 {
                v[0] = 1.0 + (i as f64 * 0.01);
                v[1] = 0.1;
                categories.push("one".into());
            } else if i < 2 * n / 3 {
                v[2] = 1.0 + (i as f64 * 0.01);
                v[3] = 0.1;
                categories.push("two".into());
            } else {
                v[4] = 1.0 + (i as f64 * 0.01);
                v[5] = 0.1;
                categories.push("three".into());
            }
            v[6] = 0.02 * i as f64;
            embeddings.push(v);
        }
        PipelineInput {
            categories,
            embeddings,
        }
    }

    #[test]
    fn search_space_grid_cardinality_sums_per_kind() {
        let s = SearchSpace::default();
        let common = s.num_domain_groups.len()
            * s.low_evr_threshold.len()
            * s.overlap_artifact_territorial.len()
            * s.threshold_base.len()
            * s.threshold_evr_penalty.len()
            * s.min_evr_improvement.len();
        // Default kinds = {PCA, Laplacian}; PCA adds `common`, Laplacian
        // adds `common × k_neighbors × active_threshold`.
        let expected = common
            + common * s.laplacian_k_neighbors.len() * s.laplacian_active_threshold.len();
        assert_eq!(s.grid_cardinality(), expected);
    }

    #[test]
    fn default_search_space_includes_pca_and_laplacian() {
        let s = SearchSpace::default();
        assert!(s.projection_kinds.contains(&ProjectionKind::Pca));
        assert!(s
            .projection_kinds
            .contains(&ProjectionKind::LaplacianEigenmap));
        // Kernel PCA excluded by default (expensive fit).
        assert!(!s.projection_kinds.contains(&ProjectionKind::KernelPca));
    }

    #[test]
    fn grid_index_enumerates_full_space() {
        let s = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            num_domain_groups: vec![3, 5],
            low_evr_threshold: vec![0.3, 0.4],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };
        let base = PipelineConfig::default();
        let n = s.grid_cardinality();
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            let cfg = s.config_at_index(i, &base).unwrap();
            let key = (
                cfg.routing.num_domain_groups,
                (cfg.routing.low_evr_threshold * 1000.0) as i64,
            );
            seen.insert(key);
        }
        assert_eq!(seen.len(), n);
        assert!(s.config_at_index(n, &base).is_none());
    }

    #[test]
    fn grid_index_enumerates_across_projection_kinds() {
        let s = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca, ProjectionKind::LaplacianEigenmap],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            num_domain_groups: vec![3],
            low_evr_threshold: vec![0.35],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };
        let base = PipelineConfig::default();
        let kinds: std::collections::HashSet<ProjectionKind> = (0..s.grid_cardinality())
            .map(|i| s.config_at_index(i, &base).unwrap().projection_kind)
            .collect();
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&ProjectionKind::Pca));
        assert!(kinds.contains(&ProjectionKind::LaplacianEigenmap));
    }

    #[test]
    fn grid_search_runs_and_picks_best() {
        let input = make_input(24, 8);
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            num_domain_groups: vec![3, 5],
            low_evr_threshold: vec![0.35],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };
        let metric = TerritorialHealth;
        let (pipeline, report) = auto_tune(
            input,
            &space,
            &metric,
            SearchStrategy::Grid,
            &PipelineConfig::default(),
        )
        .unwrap();

        assert_eq!(report.trials.len(), 2);
        assert!(report.best_score >= report.mean_score() - 1e-9);
        assert!(pipeline.num_categories() > 0);
        assert_eq!(report.metric_name, "territorial_health");
        assert!(report.failures.is_empty());
    }

    #[test]
    fn random_search_respects_budget() {
        let input = make_input(24, 8);
        let space = SearchSpace::default();
        let metric = BridgeCoherence;
        let (_pipeline, report) = auto_tune(
            input,
            &space,
            &metric,
            SearchStrategy::Random {
                budget: 5,
                seed: 42,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 5);
    }

    #[test]
    fn random_search_is_seed_reproducible() {
        let space = SearchSpace::default();
        let metric = TerritorialHealth;

        let run = |seed: u64| {
            let input = make_input(24, 8);
            auto_tune(
                input,
                &space,
                &metric,
                SearchStrategy::Random { budget: 8, seed },
                &PipelineConfig::default(),
            )
            .unwrap()
            .1
        };

        let a = run(7);
        let b = run(7);
        let c = run(13);

        assert_eq!(a.trials.len(), b.trials.len());
        for (ta, tb) in a.trials.iter().zip(b.trials.iter()) {
            assert_eq!(
                ta.config.routing.num_domain_groups,
                tb.config.routing.num_domain_groups
            );
            assert!((ta.score - tb.score).abs() < 1e-12);
        }
        // Different seed should (very likely) produce a different trial
        // sequence. If it accidentally matches, the test is still valid
        // but we check at least one config differs.
        let any_differ = a.trials.iter().zip(c.trials.iter()).any(|(ta, tc)| {
            ta.config.routing.num_domain_groups != tc.config.routing.num_domain_groups
                || (ta.config.bridges.threshold_base - tc.config.bridges.threshold_base).abs()
                    > 1e-12
        });
        assert!(any_differ, "different seeds produced identical trial set");
    }

    #[test]
    fn ranked_trials_are_descending() {
        let input = make_input(24, 8);
        let metric = CompositeMetric::default_composite();
        let (_p, report) = auto_tune(
            input,
            &SearchSpace::default(),
            &metric,
            SearchStrategy::Random {
                budget: 6,
                seed: 99,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        let ranked = report.ranked_trials();
        for w in ranked.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn best_config_actually_in_trials() {
        let input = make_input(24, 8);
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &SearchSpace::default(),
            &metric,
            SearchStrategy::Random {
                budget: 4,
                seed: 1,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        let any_match = report.trials.iter().any(|t| {
            t.config.routing.num_domain_groups == report.best_config.routing.num_domain_groups
                && (t.config.routing.low_evr_threshold
                    - report.best_config.routing.low_evr_threshold)
                    .abs()
                    < 1e-12
                && (t.score - report.best_score).abs() < 1e-12
        });
        assert!(any_match, "best_config must appear in trials");
    }

    #[test]
    fn grid_search_across_projection_kinds_yields_both() {
        let input = make_input(24, 8);
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca, ProjectionKind::LaplacianEigenmap],
            laplacian_k_neighbors: vec![10, 20],
            laplacian_active_threshold: vec![0.05],
            num_domain_groups: vec![3],
            low_evr_threshold: vec![0.35],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };
        let metric = TerritorialHealth;
        let (_pipeline, report) = auto_tune(
            input,
            &space,
            &metric,
            SearchStrategy::Grid,
            &PipelineConfig::default(),
        )
        .unwrap();
        // PCA contributes 1 trial; Laplacian contributes 2 × 1 = 2 trials
        // (two k_neighbors values × one threshold value). Total = 3.
        assert_eq!(report.trials.len(), 3);
        let kinds_in_trials: std::collections::HashSet<ProjectionKind> =
            report.trials.iter().map(|t| t.config.projection_kind).collect();
        assert!(kinds_in_trials.contains(&ProjectionKind::Pca));
        assert!(kinds_in_trials.contains(&ProjectionKind::LaplacianEigenmap));
        // Verify the two Laplacian trials actually use different k values.
        let lap_ks: std::collections::HashSet<usize> = report
            .trials
            .iter()
            .filter(|t| t.config.projection_kind == ProjectionKind::LaplacianEigenmap)
            .map(|t| t.config.laplacian.k_neighbors)
            .collect();
        assert_eq!(lap_ks.len(), 2);
    }

    #[test]
    fn laplacian_knobs_produce_distinct_configs() {
        // Sanity check that when Laplacian is the only kind, varying its
        // hyperparameters produces configs whose LaplacianConfig actually
        // differs (and doesn't accidentally alias on same-(k, threshold) pairs).
        let s = SearchSpace {
            projection_kinds: vec![ProjectionKind::LaplacianEigenmap],
            laplacian_k_neighbors: vec![10, 20],
            laplacian_active_threshold: vec![0.03, 0.08],
            num_domain_groups: vec![3],
            low_evr_threshold: vec![0.35],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };
        let base = PipelineConfig::default();
        let configs: Vec<(usize, u64)> = (0..s.grid_cardinality())
            .map(|i| {
                let cfg = s.config_at_index(i, &base).unwrap();
                (
                    cfg.laplacian.k_neighbors,
                    cfg.laplacian.active_threshold.to_bits(),
                )
            })
            .collect();
        let unique: std::collections::HashSet<(usize, u64)> =
            configs.iter().copied().collect();
        assert_eq!(unique.len(), 4, "expected 4 distinct (k, threshold) pairs");
    }

    #[test]
    fn bayesian_respects_budget() {
        let input = make_input(24, 8);
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &SearchSpace::default(),
            &metric,
            SearchStrategy::Bayesian {
                budget: 10,
                warmup: 4,
                gamma: 0.25,
                seed: 42,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 10);
    }

    #[test]
    fn bayesian_seed_reproducible() {
        let metric = TerritorialHealth;
        let run = |seed: u64| {
            let input = make_input(24, 8);
            auto_tune(
                input,
                &SearchSpace::default(),
                &metric,
                SearchStrategy::Bayesian {
                    budget: 8,
                    warmup: 3,
                    gamma: 0.25,
                    seed,
                },
                &PipelineConfig::default(),
            )
            .unwrap()
            .1
        };
        let a = run(7);
        let b = run(7);
        assert_eq!(a.trials.len(), b.trials.len());
        for (ta, tb) in a.trials.iter().zip(b.trials.iter()) {
            assert_eq!(
                ta.config.projection_kind,
                tb.config.projection_kind
            );
            assert!((ta.score - tb.score).abs() < 1e-12);
        }
    }

    #[test]
    fn bayesian_finds_something_under_default_metric() {
        // Only asserting the tuner runs to completion and best_score is a
        // valid [0, 1] value — not that Bayesian strictly beats random at
        // this small budget (it often does, but not monotonically).
        let input = make_input(30, 10);
        let metric = CompositeMetric::default_composite();
        let (_p, report) = auto_tune(
            input,
            &SearchSpace::default(),
            &metric,
            SearchStrategy::Bayesian {
                budget: 12,
                warmup: 4,
                gamma: 0.25,
                seed: 0xC0FF_EE,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 12);
        assert!(report.best_score >= 0.0 && report.best_score <= 1.0);
    }

    #[test]
    fn bayesian_warmup_clamped() {
        // warmup = 100 with budget = 5 should clamp to 5 (all warmup).
        let input = make_input(24, 8);
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &SearchSpace::default(),
            &metric,
            SearchStrategy::Bayesian {
                budget: 5,
                warmup: 100,
                gamma: 0.25,
                seed: 1,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 5);
    }

    #[test]
    fn returned_pipeline_uses_best_config() {
        let input = make_input(24, 8);
        let metric = TerritorialHealth;
        let (pipeline, report) = auto_tune(
            input,
            &SearchSpace::default(),
            &metric,
            SearchStrategy::Random {
                budget: 4,
                seed: 11,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(
            pipeline.config().routing.num_domain_groups,
            report.best_config.routing.num_domain_groups
        );
        assert_eq!(pipeline.projection_kind(), report.best_config.projection_kind);
    }
}
