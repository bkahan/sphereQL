//! Auto-tuner: search [`PipelineConfig`] space to maximize a [`QualityMetric`].
//!
//! This is the first usable rung of the metalearning ladder. Given a corpus
//! and a scalar objective, the tuner enumerates or samples candidate
//! configurations, builds a full pipeline for each, and records the score.
//! Three strategies ship: exhaustive [`SearchStrategy::Grid`], uniform
//! [`SearchStrategy::Random`], and the axis-parallel TPE-lite
//! [`SearchStrategy::Bayesian`] acquisition — all reproducible under a
//! fixed seed, establishing baselines for higher-order tuners (CMA-ES,
//! meta-learning) to beat.
//!
//! Projections are fit **once per distinct fit-affecting hyperparameter
//! tuple** from the input corpus and reused across every trial: PCA and
//! Kernel PCA key per kind, Laplacian per `(k_neighbors,
//! active_threshold)`, and UMAP per `(n_neighbors, n_epochs,
//! category_weight)` — with UMAP's kNN graph additionally cached per
//! `n_neighbors` (see [`TuneReport::umap_graph_builds`]). Only the
//! downstream config knobs (bridge thresholds, inner-sphere gates,
//! domain-group counts, etc.) vary per trial.

use std::collections::HashMap;
use std::time::Instant;

use crate::config::{
    BridgeConfig, InnerSphereConfig, LaplacianConfig, PipelineConfig, ProjectionKind,
    RoutingConfig, UmapConfig,
};
use crate::configured_projection::ConfiguredProjection;
use crate::pipeline::{
    PipelineError, PipelineInput, SphereQLPipeline, fit_projection_for_config, fit_umap_from_graph,
};
use crate::projection::SplitMix64;
use crate::quality_metric::QualityMetric;
use crate::types::Embedding;

// ── Search space ─────────────────────────────────────────────────────

/// Discrete candidate values for each tunable knob.
///
/// Every field holds the full set of values the tuner will consider for
/// that knob. Grid search enumerates the Cartesian product; random search
/// samples uniformly from each set per trial.
///
/// Defaults are chosen to bracket the historical hardcoded value on each
/// knob, giving the tuner room to move either direction without being
/// unreasonable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchSpace {
    /// Candidate projection families for the outer sphere. Each kind is
    /// prefit once per distinct fit-affecting hyperparameter tuple in
    /// [`auto_tune`]; trials pick the prefit matching their config.
    pub projection_kinds: Vec<ProjectionKind>,

    // ── Projection-kind-specific knobs ────────────────────────────
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

    /// Candidate values for [`UmapConfig::n_neighbors`]. Only explored
    /// when [`ProjectionKind::UmapSphere`] is in `projection_kinds`.
    pub umap_n_neighbors: Vec<usize>,
    /// Candidate values for [`UmapConfig::n_epochs`]. Only explored
    /// when [`ProjectionKind::UmapSphere`] is in `projection_kinds`.
    pub umap_n_epochs: Vec<usize>,
    /// Candidate values for [`UmapConfig::category_weight`]. Only
    /// explored when [`ProjectionKind::UmapSphere`] is in
    /// `projection_kinds`.
    pub umap_category_weight: Vec<f64>,

    // ── Kind-agnostic knobs ───────────────────────────────────────
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

impl SearchSpace {
    /// Search space optimized for large corpora (> 5000 items).
    ///
    /// Includes PCA and UMAP (but not Laplacian eigenmap, which is O(N²)
    /// on the affinity matrix). UMAP uses the ANN-backed kNN graph,
    /// making it O(N log N) for graph construction.
    pub fn large_corpus() -> Self {
        Self {
            projection_kinds: vec![ProjectionKind::Pca, ProjectionKind::UmapSphere],
            // Laplacian is excluded — kept as singletons so the axes
            // exist if a caller swaps the kind in later, but they cost
            // nothing at grid time because Laplacian isn't enumerated.
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![10, 15, 30],
            umap_n_epochs: vec![150, 300],
            umap_category_weight: vec![0.0, 1.5, 3.0],
            num_domain_groups: vec![3, 5, 7],
            low_evr_threshold: vec![0.25, 0.35],
            overlap_artifact_territorial: vec![0.2, 0.3],
            threshold_base: vec![0.4, 0.5],
            threshold_evr_penalty: vec![0.3, 0.5],
            min_evr_improvement: vec![0.05, 0.10],
        }
    }
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
            umap_n_neighbors: vec![10, 15, 30],
            umap_n_epochs: vec![150, 250],
            umap_category_weight: vec![0.0, 1.5, 3.0],
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
    /// Validate this space against a [`SearchStrategy`] and return a
    /// structured [`PipelineError::InvalidSearchSpace`] if anything is
    /// off — empty axis, missing kind-specific knobs, or budget too
    /// small for the strategy to make progress. Called upfront from
    /// [`auto_tune`] so every strategy fails at the same boundary
    /// instead of panicking mid-trial (random/Bayesian) or silently
    /// rolling up into `AllTrialsFailed { failures: [] }` (Grid).
    pub fn validate(&self, strategy: &SearchStrategy) -> Result<(), PipelineError> {
        if self.projection_kinds.is_empty() {
            return Err(PipelineError::InvalidSearchSpace(
                "axis `projection_kinds` is empty".into(),
            ));
        }
        for &kind in &self.projection_kinds {
            self.check_axes_non_empty(kind)?;
        }
        match strategy {
            SearchStrategy::Grid => {}
            SearchStrategy::Random { budget, .. } => {
                if *budget == 0 {
                    return Err(PipelineError::InvalidSearchSpace(
                        "Random search requires budget >= 1".into(),
                    ));
                }
            }
            SearchStrategy::Bayesian {
                budget,
                warmup,
                gamma,
                ..
            } => {
                if *budget < 2 {
                    return Err(PipelineError::InvalidSearchSpace(format!(
                        "Bayesian search requires budget >= 2 (got {budget})"
                    )));
                }
                if *warmup < 2 {
                    return Err(PipelineError::InvalidSearchSpace(format!(
                        "Bayesian search requires warmup >= 2 (got {warmup})"
                    )));
                }
                if !gamma.is_finite() || *gamma <= 0.0 || *gamma >= 1.0 {
                    return Err(PipelineError::InvalidSearchSpace(format!(
                        "Bayesian gamma must be in (0, 1), got {gamma}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn check_axes_non_empty(&self, kind: ProjectionKind) -> Result<(), PipelineError> {
        let common = [
            ("num_domain_groups", self.num_domain_groups.len()),
            ("low_evr_threshold", self.low_evr_threshold.len()),
            (
                "overlap_artifact_territorial",
                self.overlap_artifact_territorial.len(),
            ),
            ("threshold_base", self.threshold_base.len()),
            ("threshold_evr_penalty", self.threshold_evr_penalty.len()),
            ("min_evr_improvement", self.min_evr_improvement.len()),
        ];
        for (name, len) in common {
            if len == 0 {
                return Err(PipelineError::InvalidSearchSpace(format!(
                    "axis `{name}` is empty"
                )));
            }
        }
        if matches!(kind, ProjectionKind::LaplacianEigenmap) {
            if self.laplacian_k_neighbors.is_empty() {
                return Err(PipelineError::InvalidSearchSpace(
                    "axis `laplacian_k_neighbors` is empty".into(),
                ));
            }
            if self.laplacian_active_threshold.is_empty() {
                return Err(PipelineError::InvalidSearchSpace(
                    "axis `laplacian_active_threshold` is empty".into(),
                ));
            }
        }
        if matches!(kind, ProjectionKind::UmapSphere) {
            if self.umap_n_neighbors.is_empty() {
                return Err(PipelineError::InvalidSearchSpace(
                    "axis `umap_n_neighbors` is empty".into(),
                ));
            }
            if self.umap_n_epochs.is_empty() {
                return Err(PipelineError::InvalidSearchSpace(
                    "axis `umap_n_epochs` is empty".into(),
                ));
            }
            if self.umap_category_weight.is_empty() {
                return Err(PipelineError::InvalidSearchSpace(
                    "axis `umap_category_weight` is empty".into(),
                ));
            }
        }
        Ok(())
    }

    /// Defensive guard for the public [`Self::config_at_index`] path.
    /// `auto_tune` always validates upfront via [`Self::validate`], but
    /// external callers that decode grid indices on a hand-built
    /// [`SearchSpace`] still need a clear panic instead of a
    /// mod-by-zero deeper in the decoder.
    fn assert_axes_non_empty(&self, kind: ProjectionKind) {
        if let Err(e) = self.check_axes_non_empty(kind) {
            panic!("{e}");
        }
    }

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
            ProjectionKind::UmapSphere => {
                common
                    * self.umap_n_neighbors.len()
                    * self.umap_n_epochs.len()
                    * self.umap_category_weight.len()
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
            self.assert_axes_non_empty(kind);
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
            ..base.routing.clone()
        };
        cfg.bridges = BridgeConfig {
            threshold_base: self.threshold_base[i_tb],
            threshold_evr_penalty: self.threshold_evr_penalty[i_tep],
            overlap_artifact_territorial: self.overlap_artifact_territorial[i_oat],
            ..base.bridges.clone()
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

        if matches!(kind, ProjectionKind::UmapSphere) {
            let i_nn = take(&mut idx, self.umap_n_neighbors.len());
            let i_ne = take(&mut idx, self.umap_n_epochs.len());
            let i_cw = take(&mut idx, self.umap_category_weight.len());
            cfg.umap = UmapConfig {
                n_neighbors: self.umap_n_neighbors[i_nn],
                n_epochs: self.umap_n_epochs[i_ne],
                category_weight: self.umap_category_weight[i_cw],
                ..base.umap.clone()
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
        // `auto_tune` calls [`Self::validate`] upfront, so by the time
        // we reach here projection_kinds and every per-kind axis are
        // guaranteed non-empty. `debug_assert!` keeps the invariant
        // visible during development without re-introducing a runtime
        // panic on the hot path.
        debug_assert!(
            !self.projection_kinds.is_empty(),
            "SearchSpace::sample called without prior validate()"
        );
        let mut cfg = base.clone();
        cfg.projection_kind = pick_uniform(rng, &self.projection_kinds);
        debug_assert!(
            self.check_axes_non_empty(cfg.projection_kind).is_ok(),
            "SearchSpace::sample called without prior validate()"
        );
        cfg.routing = RoutingConfig {
            num_domain_groups: pick_uniform(rng, &self.num_domain_groups),
            low_evr_threshold: pick_uniform(rng, &self.low_evr_threshold),
            ..base.routing.clone()
        };
        cfg.bridges = BridgeConfig {
            threshold_base: pick_uniform(rng, &self.threshold_base),
            threshold_evr_penalty: pick_uniform(rng, &self.threshold_evr_penalty),
            overlap_artifact_territorial: pick_uniform(rng, &self.overlap_artifact_territorial),
            ..base.bridges.clone()
        };
        cfg.inner_sphere = InnerSphereConfig {
            min_evr_improvement: pick_uniform(rng, &self.min_evr_improvement),
            ..base.inner_sphere.clone()
        };

        if matches!(cfg.projection_kind, ProjectionKind::LaplacianEigenmap) {
            cfg.laplacian = LaplacianConfig {
                k_neighbors: pick_uniform(rng, &self.laplacian_k_neighbors),
                active_threshold: pick_uniform(rng, &self.laplacian_active_threshold),
            };
        }

        if matches!(cfg.projection_kind, ProjectionKind::UmapSphere) {
            cfg.umap = UmapConfig {
                n_neighbors: pick_uniform(rng, &self.umap_n_neighbors),
                n_epochs: pick_uniform(rng, &self.umap_n_epochs),
                category_weight: pick_uniform(rng, &self.umap_category_weight),
                ..base.umap.clone()
            };
        }

        cfg
    }
}

// ── Prefit cache key ─────────────────────────────────────────────────

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
    Laplacian {
        k: usize,
        threshold_bits: u64,
    },
    UmapSphere {
        n_neighbors: usize,
        n_epochs: usize,
        category_weight_bits: u64,
    },
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
            ProjectionKind::UmapSphere => Self::UmapSphere {
                n_neighbors: cfg.umap.n_neighbors,
                n_epochs: cfg.umap.n_epochs,
                category_weight_bits: cfg.umap.category_weight.to_bits(),
            },
        }
    }
}

// ── Strategy, report, trial record ───────────────────────────────────────

/// Which enumeration to use over the [`SearchSpace`].
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive Cartesian-product enumeration. Cost scales with the
    /// grid cardinality — see [`SearchSpace::grid_cardinality`].
    Grid,
    /// Uniform random sampling for `budget` trials.
    Random {
        budget: usize,
        seed: u64,
        /// Optional wall-time cap in seconds. When set, the tuner stops
        /// proposing new trials once cumulative elapsed time exceeds
        /// this limit. Already-running trials are not interrupted.
        /// `None` = unlimited (legacy behavior).
        max_wall_secs: Option<u64>,
    },
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
        /// Optional wall-time cap in seconds. Same semantics as
        /// [`Self::Random::max_wall_secs`].
        max_wall_secs: Option<u64>,
    },
}

impl SearchStrategy {
    /// Extract the wall-time cap, if one was set.
    fn max_wall_secs(&self) -> Option<u64> {
        match self {
            Self::Random { max_wall_secs, .. } => *max_wall_secs,
            Self::Bayesian { max_wall_secs, .. } => *max_wall_secs,
            Self::Grid => None,
        }
    }
}

/// One trial's observation.
#[derive(Debug, Clone)]
pub struct TrialRecord {
    pub config: PipelineConfig,
    pub score: f64,
    /// Wall-clock build time for this trial (pipeline rebuild only —
    /// projection fit is amortized across the tuner run).
    pub build_ms: u128,
    /// Per-component metric breakdown as `(name, weight, score)`.
    /// Populated when the metric is a composite (see
    /// [`QualityMetric::score_with_components`]); empty for leaf
    /// metrics. The fastest way to diagnose a flat tuner landscape:
    /// a component whose score barely varies across trials carries no
    /// signal for the knobs being swept.
    pub components: Vec<(String, f64, f64)>,
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
    /// Number of distinct UMAP kNN graphs built during the run. The
    /// tuner caches graphs by `n_neighbors`, so this equals the number
    /// of unique `n_neighbors` values tried across UMAP trials. Lower
    /// than the count of UMAP trials means the cache hit — a metric
    /// for verifying the reuse path is actually firing.
    pub umap_graph_builds: usize,
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

// ── The tuner itself ─────────────────────────────────────────────────

/// Run the auto-tuner and return the best pipeline plus a report.
///
/// Fits one projection per [`ProjectionKind`] listed in
/// `space.projection_kinds` (honoring Laplacian hyperparameters from
/// `base_config.laplacian`), then reuses those prefit projections across
/// every trial. Only the downstream [`PipelineConfig`] knobs (bridge
/// thresholds, inner-sphere gates, domain-group counts, etc.) vary per
/// trial — this keeps per-trial cost dominated by spatial quality
/// sampling and graph construction rather than projection fitting.
///
/// Under [`SearchStrategy::Random`] and [`SearchStrategy::Bayesian`],
/// `base_config` itself is evaluated as trial 0 (counted against the
/// budget) so a warm-start prediction competes directly with sampled
/// candidates. [`SearchStrategy::Grid`] is excluded: its trial set is
/// defined as the exact Cartesian enumeration of the space, and callers
/// assert on [`SearchSpace::grid_cardinality`] matching the trial count.
pub fn auto_tune<M: QualityMetric + ?Sized>(
    input: PipelineInput,
    space: &SearchSpace,
    metric: &M,
    strategy: SearchStrategy,
    base_config: &PipelineConfig,
) -> Result<(SphereQLPipeline, TuneReport), PipelineError> {
    // Validate space + strategy upfront so every search mode fails at
    // the same boundary with a structured PipelineError instead of
    // panicking mid-trial (random/Bayesian) or silently rolling up
    // into AllTrialsFailed { failures: [] } (Grid).
    space.validate(&strategy)?;

    // `PipelineInput` is owned — move the Vec<f64>s straight into the
    // Embedding wrappers instead of cloning each row.
    let categories = input.categories;
    let embeddings: Vec<Embedding> = input.embeddings.into_iter().map(Embedding::new).collect();

    let mut prefit: HashMap<ProjectionFitKey, ConfiguredProjection> = HashMap::new();
    // UMAP kNN graphs are reusable across configs that share `n_neighbors`
    // but differ in `n_epochs` / `category_weight`. Building the graph
    // dominates UMAP fit cost (O(N log N) for the ANN-backed graph plus
    // PCA warm-start), so caching it collapses the per-config sweep onto
    // a handful of graph builds.
    let mut umap_graph_cache: HashMap<usize, crate::umap::UmapGraph> = HashMap::new();
    let mut umap_graph_builds: usize = 0;
    let mut trials: Vec<TrialRecord> = Vec::new();
    let mut failures: Vec<(PipelineConfig, String)> = Vec::new();
    // Only the current best trial's pipeline stays alive — keeping every
    // trial's pipeline would multiply peak memory by the trial count at
    // 500k scale. Replaced (and the old one dropped) whenever a trial
    // scores at least as high, matching the old post-loop `max_by`
    // selection (last max wins) without rebuilding the winner.
    let mut best: Option<(f64, SphereQLPipeline)> = None;

    // Closure: evaluate one config, update prefit cache, push record or
    // failure. Shared by every strategy so they only differ in how they
    // propose configs.
    let run_trial = |cfg: PipelineConfig,
                     prefit: &mut HashMap<ProjectionFitKey, ConfiguredProjection>,
                     umap_graph_cache: &mut HashMap<usize, crate::umap::UmapGraph>,
                     umap_graph_builds: &mut usize,
                     trials: &mut Vec<TrialRecord>,
                     failures: &mut Vec<(PipelineConfig, String)>,
                     best: &mut Option<(f64, SphereQLPipeline)>| {
        let key = ProjectionFitKey::from_config(&cfg);
        let projection = if cfg.projection_kind == ProjectionKind::UmapSphere {
            // UMAP fast path: build the kNN graph once per `n_neighbors`
            // and reuse it across `(n_epochs, category_weight)` variations.
            // The fully-realized projection still goes into `prefit` so
            // the final pipeline rebuild and any exact-config repeats are
            // free.
            match prefit.get(&key) {
                Some(p) => p.clone(),
                None => {
                    let k = cfg.umap.n_neighbors;
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        umap_graph_cache.entry(k)
                    {
                        match crate::umap::UmapGraph::build(&embeddings, k) {
                            Ok(g) => {
                                entry.insert(g);
                                *umap_graph_builds += 1;
                            }
                            Err(err) => {
                                failures.push((cfg, err.to_string()));
                                return;
                            }
                        }
                    }
                    let graph = &umap_graph_cache[&k];
                    match fit_umap_from_graph(graph, &categories, &cfg) {
                        Ok(p) => {
                            prefit.insert(key, p.clone());
                            p
                        }
                        Err(e) => {
                            failures.push((cfg, e.to_string()));
                            return;
                        }
                    }
                }
            }
        } else {
            match prefit.get(&key) {
                Some(p) => p.clone(),
                None => match fit_projection_for_config(&embeddings, &categories, &cfg) {
                    Ok(p) => {
                        prefit.insert(key, p.clone());
                        p
                    }
                    Err(e) => {
                        failures.push((cfg, e.to_string()));
                        return;
                    }
                },
            }
        };

        let start = Instant::now();
        // Embeddings are borrowed (the pipeline doesn't retain them), so
        // only `categories` — which it does own — is cloned per trial.
        // TODO: an Arc<[String]> categories field would drop that clone
        // too (~tens of MB at 500k), but it touches the pipeline's
        // retained-field accessors and downstream crates.
        match SphereQLPipeline::with_projection_parts(
            categories.clone(),
            &embeddings,
            projection,
            cfg.clone(),
        ) {
            Ok(pipeline) => {
                let (score, components) = metric.score_with_components(&pipeline);
                let build_ms = start.elapsed().as_millis();
                trials.push(TrialRecord {
                    config: cfg,
                    score,
                    build_ms,
                    components,
                });
                let replace = match best {
                    Some((best_score, _)) => {
                        !matches!(score.partial_cmp(best_score), Some(std::cmp::Ordering::Less))
                    }
                    None => true,
                };
                if replace {
                    *best = Some((score, pipeline));
                }
            }
            Err(e) => {
                failures.push((cfg, e.to_string()));
            }
        }
    };

    let wall_start = Instant::now();
    let max_wall = strategy.max_wall_secs();
    let wall_exceeded = || match max_wall {
        Some(max_secs) => wall_start.elapsed().as_secs() >= max_secs,
        None => false,
    };

    match &strategy {
        SearchStrategy::Grid => {
            // Grid deliberately skips the base-config seed trial: its
            // contract is "trial set == the exact Cartesian enumeration"
            // and callers assert on grid_cardinality matching the count.
            for i in 0..space.grid_cardinality() {
                if let Some(cfg) = space.config_at_index(i, base_config) {
                    run_trial(
                        cfg,
                        &mut prefit,
                        &mut umap_graph_cache,
                        &mut umap_graph_builds,
                        &mut trials,
                        &mut failures,
                        &mut best,
                    );
                }
            }
        }
        SearchStrategy::Random { budget, seed, .. } => {
            let mut rng = SplitMix64::new(*seed);
            // Trial 0: warm-start seed. base_config competes directly
            // with the sampled candidates and counts against the budget.
            run_trial(
                base_config.clone(),
                &mut prefit,
                &mut umap_graph_cache,
                &mut umap_graph_builds,
                &mut trials,
                &mut failures,
                &mut best,
            );
            if !wall_exceeded() {
                for _ in 1..*budget {
                    let cfg = space.sample(&mut rng, base_config);
                    run_trial(
                        cfg,
                        &mut prefit,
                        &mut umap_graph_cache,
                        &mut umap_graph_builds,
                        &mut trials,
                        &mut failures,
                        &mut best,
                    );
                    if wall_exceeded() {
                        break;
                    }
                }
            }
        }
        SearchStrategy::Bayesian {
            budget,
            warmup,
            gamma,
            seed,
            ..
        } => {
            // budget/warmup/gamma already validated above by space.validate(&strategy).
            let budget = *budget;
            let mut rng = SplitMix64::new(*seed);
            let warmup = (*warmup).clamp(2, budget);
            let gamma = gamma.clamp(0.05, 0.95);

            // Trial 0: warm-start seed (counts as the first warmup trial).
            run_trial(
                base_config.clone(),
                &mut prefit,
                &mut umap_graph_cache,
                &mut umap_graph_builds,
                &mut trials,
                &mut failures,
                &mut best,
            );
            // Remaining warmup: uniform random.
            if !wall_exceeded() {
                for _ in 1..warmup {
                    let cfg = space.sample(&mut rng, base_config);
                    run_trial(
                        cfg,
                        &mut prefit,
                        &mut umap_graph_cache,
                        &mut umap_graph_builds,
                        &mut trials,
                        &mut failures,
                        &mut best,
                    );
                    if wall_exceeded() {
                        break;
                    }
                }
            }
            // Acquisition: axis-parallel TPE-lite.
            if !wall_exceeded() {
                for _ in warmup..budget {
                    let cfg = tpe_propose(space, base_config, &trials, gamma, &mut rng);
                    run_trial(
                        cfg,
                        &mut prefit,
                        &mut umap_graph_cache,
                        &mut umap_graph_builds,
                        &mut trials,
                        &mut failures,
                        &mut best,
                    );
                    if wall_exceeded() {
                        break;
                    }
                }
            }
        }
    }

    if trials.is_empty() {
        // Every candidate config was rejected downstream. Surface the
        // real failure list instead of the misleading `TooFewEmbeddings`
        // roll-up we used to return here.
        return Err(PipelineError::AllTrialsFailed { failures });
    }

    // Every successful trial offered its pipeline to `best`, so a
    // non-empty `trials` guarantees one was kept. Returning it directly
    // saves rebuilding the winner from scratch (a second O(N·d)
    // projection pass + category-layer build).
    let (best_score, best_pipeline) = best.expect("non-empty trials imply a kept best pipeline");
    let best_config = best_pipeline.config().clone();

    let report = TuneReport {
        metric_name: metric.name().to_string(),
        best_score,
        best_config,
        trials,
        failures,
        umap_graph_builds,
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
    let bad: Vec<&TrialRecord> = sorted.iter().skip(n_good).copied().collect();

    // Fall back to uniform sampling if we somehow don't have both sides.
    if good.is_empty() || bad.is_empty() {
        return space.sample(rng, base);
    }

    let pick_idx = |rng: &mut SplitMix64, good_counts: &[f64], bad_counts: &[f64]| -> usize {
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

    // Kind-agnostic knobs: histograms deliberately pool ALL trials
    // regardless of projection kind. Conditioning each knob on the
    // sampled kind would shrink the histograms to near-uselessness at
    // typical budgets — accepting cross-kind aliasing is the
    // axis-parallel TPE simplification.
    let ndg_g = hist_usize(&good, &space.num_domain_groups, |c| {
        c.routing.num_domain_groups
    });
    let ndg_b = hist_usize(&bad, &space.num_domain_groups, |c| {
        c.routing.num_domain_groups
    });
    let let_g = hist_f64(&good, &space.low_evr_threshold, |c| {
        c.routing.low_evr_threshold
    });
    let let_b = hist_f64(&bad, &space.low_evr_threshold, |c| {
        c.routing.low_evr_threshold
    });
    let oat_g = hist_f64(&good, &space.overlap_artifact_territorial, |c| {
        c.bridges.overlap_artifact_territorial
    });
    let oat_b = hist_f64(&bad, &space.overlap_artifact_territorial, |c| {
        c.bridges.overlap_artifact_territorial
    });
    let tb_g = hist_f64(&good, &space.threshold_base, |c| c.bridges.threshold_base);
    let tb_b = hist_f64(&bad, &space.threshold_base, |c| c.bridges.threshold_base);
    let tep_g = hist_f64(&good, &space.threshold_evr_penalty, |c| {
        c.bridges.threshold_evr_penalty
    });
    let tep_b = hist_f64(&bad, &space.threshold_evr_penalty, |c| {
        c.bridges.threshold_evr_penalty
    });
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
        ..base.routing.clone()
    };
    cfg.bridges = BridgeConfig {
        threshold_base: space.threshold_base[pick_idx(rng, &tb_g, &tb_b)],
        threshold_evr_penalty: space.threshold_evr_penalty[pick_idx(rng, &tep_g, &tep_b)],
        overlap_artifact_territorial: space.overlap_artifact_territorial
            [pick_idx(rng, &oat_g, &oat_b)],
        ..base.bridges.clone()
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
                k_neighbors: pick_uniform(rng, &space.laplacian_k_neighbors),
                active_threshold: pick_uniform(rng, &space.laplacian_active_threshold),
            };
        } else {
            let k_g = hist_usize(&good_l, &space.laplacian_k_neighbors, |c| {
                c.laplacian.k_neighbors
            });
            let k_b = hist_usize(&bad_l, &space.laplacian_k_neighbors, |c| {
                c.laplacian.k_neighbors
            });
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

    if matches!(kind, ProjectionKind::UmapSphere) {
        let good_u: Vec<&TrialRecord> = good
            .iter()
            .copied()
            .filter(|t| t.config.projection_kind == ProjectionKind::UmapSphere)
            .collect();
        let bad_u: Vec<&TrialRecord> = bad
            .iter()
            .copied()
            .filter(|t| t.config.projection_kind == ProjectionKind::UmapSphere)
            .collect();
        if good_u.is_empty() || bad_u.is_empty() {
            cfg.umap = UmapConfig {
                n_neighbors: pick_uniform(rng, &space.umap_n_neighbors),
                n_epochs: pick_uniform(rng, &space.umap_n_epochs),
                category_weight: pick_uniform(rng, &space.umap_category_weight),
                ..base.umap.clone()
            };
        } else {
            let nn_g = hist_usize(&good_u, &space.umap_n_neighbors, |c| c.umap.n_neighbors);
            let nn_b = hist_usize(&bad_u, &space.umap_n_neighbors, |c| c.umap.n_neighbors);
            let ne_g = hist_usize(&good_u, &space.umap_n_epochs, |c| c.umap.n_epochs);
            let ne_b = hist_usize(&bad_u, &space.umap_n_epochs, |c| c.umap.n_epochs);
            let cw_g = hist_f64(&good_u, &space.umap_category_weight, |c| {
                c.umap.category_weight
            });
            let cw_b = hist_f64(&bad_u, &space.umap_category_weight, |c| {
                c.umap.category_weight
            });
            cfg.umap = UmapConfig {
                n_neighbors: space.umap_n_neighbors[pick_idx(rng, &nn_g, &nn_b)],
                n_epochs: space.umap_n_epochs[pick_idx(rng, &ne_g, &ne_b)],
                category_weight: space.umap_category_weight[pick_idx(rng, &cw_g, &cw_b)],
                ..base.umap.clone()
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
        if let Some((i, _)) = values.iter().enumerate().min_by(|a, b| {
            (a.1 - v)
                .abs()
                .partial_cmp(&(b.1 - v).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            counts[i] += 1.0;
        }
    }
    counts
}

/// Pick one element of `vals` uniformly at random. Panics if `vals` is
/// empty — callers always pass non-empty `SearchSpace` axes, so the
/// empty case would be a programmer error rather than a recoverable
/// input.
fn pick_uniform<T: Copy>(rng: &mut SplitMix64, vals: &[T]) -> T {
    // next_f64 instead of next_u64 % len: the modulo form is biased
    // toward low indices whenever len doesn't divide 2^64. The min
    // guards the next_f64 == 1.0 edge.
    vals[((rng.next_f64() * vals.len() as f64) as usize).min(vals.len() - 1)]
}

fn sample_categorical(rng: &mut SplitMix64, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        let n = weights.len().max(1);
        return ((rng.next_f64() * n as f64) as usize).min(n - 1);
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

// ── Tests ──────────────────────────────────────────────────────────

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

    fn full_search_space() -> SearchSpace {
        SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
            num_domain_groups: vec![3],
            low_evr_threshold: vec![0.3],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        }
    }

    #[test]
    fn validate_rejects_empty_projection_kinds_for_every_strategy() {
        let mut s = full_search_space();
        s.projection_kinds.clear();
        for strategy in [
            SearchStrategy::Grid,
            SearchStrategy::Random {
                budget: 4,
                seed: 1,
                max_wall_secs: None,
            },
            SearchStrategy::Bayesian {
                budget: 4,
                warmup: 2,
                gamma: 0.25,
                seed: 1,
                max_wall_secs: None,
            },
        ] {
            match s.validate(&strategy) {
                Err(PipelineError::InvalidSearchSpace(msg)) => {
                    assert!(msg.contains("projection_kinds"), "msg = {msg:?}");
                }
                other => panic!("expected InvalidSearchSpace, got {other:?}"),
            }
        }
    }

    #[test]
    fn validate_rejects_empty_axis() {
        let mut s = full_search_space();
        s.threshold_base.clear();
        match s.validate(&SearchStrategy::Grid) {
            Err(PipelineError::InvalidSearchSpace(msg)) => {
                assert!(msg.contains("threshold_base"), "msg = {msg:?}");
            }
            other => panic!("expected InvalidSearchSpace, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_empty_laplacian_axis_only_when_kind_present() {
        let mut s = full_search_space();
        s.laplacian_k_neighbors.clear();
        // PCA-only space: missing laplacian axis is fine because the
        // kind isn't in `projection_kinds`.
        assert!(s.validate(&SearchStrategy::Grid).is_ok());
        s.projection_kinds.push(ProjectionKind::LaplacianEigenmap);
        match s.validate(&SearchStrategy::Grid) {
            Err(PipelineError::InvalidSearchSpace(msg)) => {
                assert!(msg.contains("laplacian_k_neighbors"), "msg = {msg:?}");
            }
            other => panic!("expected InvalidSearchSpace, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_bad_bayesian_params() {
        let s = full_search_space();
        let cases: &[(SearchStrategy, &str)] = &[
            (
                SearchStrategy::Bayesian {
                    budget: 1,
                    warmup: 2,
                    gamma: 0.25,
                    seed: 1,
                    max_wall_secs: None,
                },
                "budget",
            ),
            (
                SearchStrategy::Bayesian {
                    budget: 5,
                    warmup: 1,
                    gamma: 0.25,
                    seed: 1,
                    max_wall_secs: None,
                },
                "warmup",
            ),
            (
                SearchStrategy::Bayesian {
                    budget: 5,
                    warmup: 2,
                    gamma: 0.0,
                    seed: 1,
                    max_wall_secs: None,
                },
                "gamma",
            ),
            (
                SearchStrategy::Bayesian {
                    budget: 5,
                    warmup: 2,
                    gamma: f64::NAN,
                    seed: 1,
                    max_wall_secs: None,
                },
                "gamma",
            ),
        ];
        for (strategy, needle) in cases {
            match s.validate(strategy) {
                Err(PipelineError::InvalidSearchSpace(msg)) => {
                    assert!(msg.contains(needle), "msg={msg:?} needle={needle:?}");
                }
                other => panic!("expected InvalidSearchSpace for {needle:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn auto_tune_propagates_invalid_search_space_for_grid() {
        let s = SearchSpace {
            projection_kinds: vec![],
            ..full_search_space()
        };
        let metric = BridgeCoherence;
        let base = PipelineConfig::default();
        match auto_tune(make_input(30, 10), &s, &metric, SearchStrategy::Grid, &base) {
            Err(PipelineError::InvalidSearchSpace(_)) => {}
            Err(other) => panic!("expected InvalidSearchSpace, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
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
        let expected =
            common + common * s.laplacian_k_neighbors.len() * s.laplacian_active_threshold.len();
        assert_eq!(s.grid_cardinality(), expected);
    }

    #[test]
    fn default_search_space_includes_pca_and_laplacian() {
        let s = SearchSpace::default();
        assert!(s.projection_kinds.contains(&ProjectionKind::Pca));
        assert!(
            s.projection_kinds
                .contains(&ProjectionKind::LaplacianEigenmap)
        );
        // Kernel PCA excluded by default (expensive fit).
        assert!(!s.projection_kinds.contains(&ProjectionKind::KernelPca));
    }

    #[test]
    fn grid_index_enumerates_full_space() {
        let s = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
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
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
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
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
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
    fn trial_records_carry_component_breakdown_for_composites() {
        let input = make_input(24, 8);
        let metric = CompositeMetric::default_composite();
        let (_p, report) = auto_tune(
            input,
            &full_search_space(),
            &metric,
            SearchStrategy::Grid,
            &PipelineConfig::default(),
        )
        .unwrap();
        assert!(!report.trials.is_empty());
        for t in &report.trials {
            assert_eq!(
                t.components.len(),
                4,
                "composite trials must record the 4-component breakdown"
            );
            let recomposed: f64 = t.components.iter().map(|(_, w, s)| w * s).sum();
            assert!(
                (t.score - recomposed).abs() < 1e-12,
                "breakdown must recompose to the recorded score"
            );
        }
    }

    #[test]
    fn trial_records_have_empty_components_for_leaf_metrics() {
        let input = make_input(24, 8);
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &full_search_space(),
            &metric,
            SearchStrategy::Grid,
            &PipelineConfig::default(),
        )
        .unwrap();
        assert!(!report.trials.is_empty());
        for t in &report.trials {
            assert!(t.components.is_empty());
        }
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
                max_wall_secs: None,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 5);
    }

    #[test]
    fn random_search_respects_wall_time_cap() {
        let input = make_input(24, 8);
        let space = SearchSpace::default();
        let metric = TerritorialHealth;
        let (_pipeline, report) = auto_tune(
            input,
            &space,
            &metric,
            SearchStrategy::Random {
                budget: 1000,
                seed: 42,
                max_wall_secs: Some(1),
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert!(
            report.trials.len() < 1000,
            "wall time cap should have stopped early, got {} trials",
            report.trials.len()
        );
        assert!(
            !report.trials.is_empty(),
            "should complete at least one trial before checking wall time"
        );
    }

    #[test]
    fn none_wall_time_is_unlimited() {
        let input = make_input(24, 8);
        let space = full_search_space();
        let metric = TerritorialHealth;
        let (_pipeline, report) = auto_tune(
            input,
            &space,
            &metric,
            SearchStrategy::Random {
                budget: 3,
                seed: 1,
                max_wall_secs: None,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 3);
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
                SearchStrategy::Random {
                    budget: 8,
                    seed,
                    max_wall_secs: None,
                },
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
                max_wall_secs: None,
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
                max_wall_secs: None,
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
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
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
        let kinds_in_trials: std::collections::HashSet<ProjectionKind> = report
            .trials
            .iter()
            .map(|t| t.config.projection_kind)
            .collect();
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
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
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
        let unique: std::collections::HashSet<(usize, u64)> = configs.iter().copied().collect();
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
                max_wall_secs: None,
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
                    max_wall_secs: None,
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
            assert_eq!(ta.config.projection_kind, tb.config.projection_kind);
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
                seed: 0xC0FFEE,
                max_wall_secs: None,
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
                max_wall_secs: None,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(report.trials.len(), 5);
    }

    #[test]
    fn umap_search_space_cardinality() {
        let s = SearchSpace::large_corpus();
        let common = s.num_domain_groups.len()
            * s.low_evr_threshold.len()
            * s.overlap_artifact_territorial.len()
            * s.threshold_base.len()
            * s.threshold_evr_penalty.len()
            * s.min_evr_improvement.len();
        let umap_specific =
            s.umap_n_neighbors.len() * s.umap_n_epochs.len() * s.umap_category_weight.len();
        // PCA contributes `common`, UMAP contributes `common * umap_specific`.
        let expected = common + common * umap_specific;
        assert_eq!(s.grid_cardinality(), expected);
    }

    #[test]
    fn umap_trials_produce_umap_configs() {
        let input = make_input(24, 8);
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::UmapSphere],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![10, 20],
            umap_n_epochs: vec![50],
            umap_category_weight: vec![1.0],
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

        assert_eq!(report.trials.len(), 2);
        for t in &report.trials {
            assert_eq!(t.config.projection_kind, ProjectionKind::UmapSphere);
        }
        let nn_values: std::collections::HashSet<usize> = report
            .trials
            .iter()
            .map(|t| t.config.umap.n_neighbors)
            .collect();
        assert_eq!(nn_values.len(), 2);
    }

    #[test]
    fn umap_graph_cache_reuses_across_trials_sharing_n_neighbors() {
        // Six UMAP configs all share `n_neighbors = 10` and differ only in
        // `n_epochs` × `category_weight`. The kNN graph + PCA warm-start
        // should be built once, then reused — `umap_graph_builds` must
        // equal the number of distinct `n_neighbors` values (= 1), not
        // the number of trials.
        let input = make_input(24, 8);
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::UmapSphere],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![10],
            umap_n_epochs: vec![30, 60],
            umap_category_weight: vec![0.0, 1.0, 2.0],
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

        assert_eq!(report.trials.len(), 6, "6 UMAP configs in the grid");
        assert_eq!(
            report.umap_graph_builds, 1,
            "all 6 configs share n_neighbors=10, so the cache should build the graph exactly once"
        );
    }

    #[test]
    fn umap_graph_cache_builds_one_per_unique_n_neighbors() {
        // Two distinct `n_neighbors` values × two `n_epochs` = 4 UMAP
        // trials. The cache builds the graph once per unique
        // `n_neighbors`, so `umap_graph_builds` should equal 2.
        let input = make_input(24, 8);
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::UmapSphere],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![10, 20],
            umap_n_epochs: vec![30, 60],
            umap_category_weight: vec![0.0],
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

        assert_eq!(report.trials.len(), 4);
        assert_eq!(
            report.umap_graph_builds, 2,
            "n_neighbors ∈ {{10, 20}} should produce exactly 2 graph builds"
        );
    }

    #[test]
    fn umap_graph_cache_zero_when_no_umap_trials() {
        // PCA-only search space — no UMAP trials, no graph builds.
        let input = make_input(24, 8);
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![10],
            umap_n_epochs: vec![30],
            umap_category_weight: vec![0.0],
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

        assert_eq!(report.umap_graph_builds, 0);
    }

    #[test]
    fn validate_rejects_empty_umap_axis_only_when_kind_present() {
        let mut s = full_search_space();
        s.umap_n_neighbors.clear();
        // PCA-only space: missing UMAP axis is fine.
        assert!(s.validate(&SearchStrategy::Grid).is_ok());
        s.projection_kinds.push(ProjectionKind::UmapSphere);
        match s.validate(&SearchStrategy::Grid) {
            Err(PipelineError::InvalidSearchSpace(msg)) => {
                assert!(msg.contains("umap_n_neighbors"), "msg = {msg:?}");
            }
            other => panic!("expected InvalidSearchSpace, got {other:?}"),
        }
    }

    #[test]
    fn tpe_proposes_dominating_value_more_often_than_uniform() {
        // Hand-crafted history: every top-gamma trial used
        // num_domain_groups = 7, every bad trial used 3 or 5. The
        // acquisition should propose 7 far more often than the uniform
        // 1/3 baseline.
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
            num_domain_groups: vec![3, 5, 7],
            low_evr_threshold: vec![0.3],
            overlap_artifact_territorial: vec![0.3],
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };
        let base = PipelineConfig::default();

        let trial = |ndg: usize, score: f64| -> TrialRecord {
            let mut config = base.clone();
            config.projection_kind = ProjectionKind::Pca;
            config.routing.num_domain_groups = ndg;
            TrialRecord {
                config,
                score,
                build_ms: 0,
                components: Vec::new(),
            }
        };

        let mut trials = Vec::new();
        for i in 0..4 {
            trials.push(trial(7, 0.9 + i as f64 * 0.01));
        }
        for i in 0..6 {
            trials.push(trial(3, 0.1 + i as f64 * 0.01));
            trials.push(trial(5, 0.1 + i as f64 * 0.005));
        }

        let mut rng = SplitMix64::new(42);
        let n_proposals = 300;
        let mut count_7 = 0usize;
        for _ in 0..n_proposals {
            let cfg = tpe_propose(&space, &base, &trials, 0.25, &mut rng);
            if cfg.routing.num_domain_groups == 7 {
                count_7 += 1;
            }
        }

        // Uniform would land near 100/300. The good/bad ratio for 7 puts
        // its sampling probability above 0.9, so 180 is a comfortable
        // margin that still fails if the acquisition stops conditioning
        // on the split.
        assert!(
            count_7 > 180,
            "dominating value proposed only {count_7}/{n_proposals} times (uniform ≈ {})",
            n_proposals / 3
        );
    }

    #[test]
    fn random_seeds_base_config_as_trial_zero() {
        let input = make_input(24, 8);
        let mut base = PipelineConfig::default();
        base.bridges.overlap_artifact_territorial = 0.123; // off-axis
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &full_search_space(),
            &metric,
            SearchStrategy::Random {
                budget: 4,
                seed: 9,
                max_wall_secs: None,
            },
            &base,
        )
        .unwrap();

        assert_eq!(report.trials.len(), 4, "seed trial counts against budget");
        assert!(
            (report.trials[0].config.bridges.overlap_artifact_territorial - 0.123).abs() < 1e-12,
            "trial 0 must be base_config itself"
        );
        for t in &report.trials[1..] {
            assert!(
                (t.config.bridges.overlap_artifact_territorial - 0.3).abs() < 1e-12,
                "sampled trials must come from the space's axes"
            );
        }
    }

    #[test]
    fn bayesian_seeds_base_config_as_trial_zero() {
        let input = make_input(24, 8);
        let mut base = PipelineConfig::default();
        base.bridges.overlap_artifact_territorial = 0.123;
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &full_search_space(),
            &metric,
            SearchStrategy::Bayesian {
                budget: 5,
                warmup: 2,
                gamma: 0.25,
                seed: 9,
                max_wall_secs: None,
            },
            &base,
        )
        .unwrap();

        assert_eq!(report.trials.len(), 5);
        assert!(
            (report.trials[0].config.bridges.overlap_artifact_territorial - 0.123).abs() < 1e-12
        );
    }

    #[test]
    fn grid_does_not_seed_base_config() {
        let input = make_input(24, 8);
        let mut base = PipelineConfig::default();
        base.bridges.overlap_artifact_territorial = 0.123;
        let metric = TerritorialHealth;
        let (_p, report) = auto_tune(
            input,
            &full_search_space(),
            &metric,
            SearchStrategy::Grid,
            &base,
        )
        .unwrap();

        assert_eq!(
            report.trials.len(),
            full_search_space().grid_cardinality(),
            "grid trial count must stay the exact enumeration"
        );
        for t in &report.trials {
            assert!((t.config.bridges.overlap_artifact_territorial - 0.3).abs() < 1e-12);
        }
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
                max_wall_secs: None,
            },
            &PipelineConfig::default(),
        )
        .unwrap();
        assert_eq!(
            pipeline.config().routing.num_domain_groups,
            report.best_config.routing.num_domain_groups
        );
        assert_eq!(
            pipeline.projection_kind(),
            report.best_config.projection_kind
        );
    }
}
