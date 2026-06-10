//! Meta-learning across corpora: predict a [`PipelineConfig`] for a new
//! corpus by consulting past tuner runs on similar corpora.
//!
//! This is Level 2 of SphereQL's self-optimization hierarchy (per the
//! metalearning-direction memory):
//!
//! - **L1** (`tuner::auto_tune`): per-corpus search. Produces a best config.
//! - **L2** (this module): cross-corpus generalization. Takes the (corpus
//!   features, best config) pairs produced by L1 and learns a function
//!   `CorpusFeatures → PipelineConfig` so new corpora can skip search or
//!   warm-start it.
//! - **L3**: online adaptation from query feedback. Deferred.
//!
//! Today's meta-model is a simple z-score-normalized nearest neighbor
//! over [`CorpusFeatures::to_vec`], with two model-space adjustments:
//! scale-type features (item/category/dim counts) are `ln(1+x)`
//! compressed before normalization so a single 500k corpus can't
//! dominate the statistics, and training sets mixing multiple
//! `metric_name`s are stratified to the dominant metric at fit time
//! (scores under different objectives are not comparable). It works
//! with any `N ≥ 1` training records, is deterministic, and has no
//! free hyperparameters. When you've accumulated ≥ 10 diverse corpora
//! you can swap in something fancier (gradient-boosted trees, small
//! MLP) against the same [`MetaModel`] trait — the storage format
//! ([`MetaTrainingRecord`]) stays stable.
//!
//! # Storage
//!
//! Records are serialized as a flat JSON array:
//!
//! ```json
//! [
//!   { "corpus_id": "built_in_775", "features": {...}, "best_config": {...}, ... },
//!   ...
//! ]
//! ```
//!
//! [`MetaTrainingRecord::save_list`] and [`MetaTrainingRecord::load_list`]
//! are convenience wrappers; the format is plain enough to edit by hand
//! or process with `jq`.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::{PipelineConfig, ProjectionKind};
use crate::corpus_features::{CORPUS_FEATURE_COUNT, CorpusFeatures};
use crate::feedback::FeedbackSummary;
use crate::tuner::TuneReport;
use crate::util::{default_timestamp, migrate_legacy_array_to_jsonl, sphereql_home_dir};

/// One observation for the meta-learner: "on this corpus profile, this
/// config was found to be best under this metric."
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetaTrainingRecord {
    /// User-supplied corpus identifier. Not used by the model — just for
    /// human-readable provenance in logs and training-set audits.
    pub corpus_id: String,
    /// Low-dim profile of the corpus. Input to the meta-model.
    pub features: CorpusFeatures,
    /// The config that won the tuner run. Target of the meta-model.
    pub best_config: PipelineConfig,
    /// The score achieved by `best_config` under `metric_name`.
    pub best_score: f64,
    /// Normalized improvement of `best_score` over the run's mean trial
    /// score, as a fraction of the available headroom:
    /// `(best − mean) / (1 − mean)`, clamped to `[0, 1]`.
    ///
    /// Unlike `best_score`, this is comparable across corpora of
    /// different intrinsic difficulty — a 0.9 on an easy corpus and a
    /// 0.6 on a hard one can represent the same "the tuner found real
    /// signal" evidence. A lift near 0 means the landscape was flat
    /// and the winning config is weak evidence.
    /// [`DistanceWeightedMetaModel`] prefers this over `best_score`
    /// when present. `None` for records created before this field
    /// existed or from runs with fewer than 2 trials.
    #[serde(default)]
    pub score_lift: Option<f64>,
    /// Which quality metric was being optimized. Records with different
    /// metrics aren't directly comparable; both shipped models
    /// stratify to the dominant metric at fit time.
    pub metric_name: String,
    /// Short description of the search strategy, e.g.
    /// `"random{budget=24,seed=...}"`. Free-form — for auditing only.
    pub strategy: String,
    /// RFC 3339 timestamp (or any string). Free-form.
    pub timestamp: String,
}

impl MetaTrainingRecord {
    /// Build a record from the ingredients of one tuner run.
    ///
    /// `corpus_id` and `strategy_label` are free-form strings the caller
    /// provides for provenance — the tuner doesn't know either on its
    /// own. `timestamp` defaults to seconds-since-Unix-epoch (sortable,
    /// unambiguous, dependency-free); swap in your own format via
    /// [`Self::with_timestamp`] if you want human-readable.
    pub fn from_tune_result(
        corpus_id: impl Into<String>,
        features: CorpusFeatures,
        report: &TuneReport,
        strategy_label: impl Into<String>,
    ) -> Self {
        Self {
            corpus_id: corpus_id.into(),
            features,
            best_config: report.best_config.clone(),
            best_score: report.best_score,
            score_lift: score_lift_from_report(report),
            metric_name: report.metric_name.clone(),
            strategy: strategy_label.into(),
            timestamp: default_timestamp(),
        }
    }

    /// Replace the timestamp. Useful when the caller has a preferred
    /// format (e.g. an RFC 3339 string from `chrono`).
    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = ts.into();
        self
    }

    /// Save a list of records as a JSON array to disk. Creates parent
    /// directories as needed.
    ///
    /// Kept for callers who want a pretty-printed snapshot (backups,
    /// audits, diffs). The default on-disk store uses JSONL under
    /// [`Self::append_to_default_store`] for O(1) appends — read it
    /// back via [`Self::load_default_store`], which auto-detects
    /// legacy array files as well.
    pub fn save_list(records: &[Self], path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(records)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load a list of records from disk.
    ///
    /// Accepts both a JSON array (legacy format and what `save_list`
    /// writes) and JSON Lines (one record per line, the new append
    /// format). Detection is first-character based: `[` ⇒ array,
    /// anything else ⇒ JSONL. Returns an empty vec if the file
    /// doesn't exist.
    pub fn load_list(path: impl AsRef<Path>) -> io::Result<Vec<Self>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw = fs::read_to_string(path)?;
        let trimmed = raw.trim_start();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        if trimmed.starts_with('[') {
            // Legacy JSON array.
            return serde_json::from_str(trimmed)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e));
        }
        // JSONL: one record per non-empty line.
        trimmed
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str::<Self>(l)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            })
            .collect()
    }

    /// Default on-disk training-store path: `~/.sphereql/meta_records.json`.
    pub fn default_store_path() -> io::Result<PathBuf> {
        Ok(sphereql_home_dir()?.join("meta_records.json"))
    }

    /// Append this record to the user's default training store.
    ///
    /// Constant-time per call: opens the file in append mode and
    /// writes one JSON-encoded line. Previously this loaded every
    /// record, pushed the new one, and rewrote the entire file —
    /// O(N) per append, which dominated at N → 10k.
    ///
    /// Existing stores written in legacy array format keep working;
    /// on the first append we re-emit the file as JSONL (one-time
    /// O(N) migration), then subsequent appends are O(1).
    pub fn append_to_default_store(&self) -> io::Result<PathBuf> {
        let path = Self::default_store_path()?;
        self.append_to(&path)?;
        Ok(path)
    }

    /// Append this record to an arbitrary JSONL file. Creates the
    /// file and any missing parent directories on first call.
    pub fn append_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        use std::io::Write;

        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        // Migrate a legacy array file to JSONL on the first append —
        // one-time cost that converts N records, after which appends
        // are O(1). New files skip this path entirely.
        migrate_legacy_array_to_jsonl(path, |head| {
            let records: Vec<Self> = serde_json::from_str(head.trim_start())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let mut migrated = String::with_capacity(head.len());
            for r in &records {
                serde_json::to_string(r)
                    .map(|line| {
                        migrated.push_str(&line);
                        migrated.push('\n');
                    })
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            }
            Ok(migrated)
        })?;

        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let line = serde_json::to_string(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        writeln!(f, "{line}")
    }

    /// Load all records from the user's default training store. Returns
    /// an empty vec if the store doesn't exist yet.
    pub fn load_default_store() -> io::Result<Vec<Self>> {
        Self::load_list(Self::default_store_path()?)
    }

    /// Blend this record's automated `best_score` with a feedback
    /// summary's `mean_score` into a single adjusted score.
    ///
    /// `alpha` ∈ `[0, 1]` controls how much weight to give feedback:
    ///   - `0.0` returns `best_score` unchanged (ignore feedback).
    ///   - `1.0` returns the feedback mean (trust feedback entirely).
    ///   - `0.5` weights them equally.
    ///
    /// `alpha` is clamped to `[0, 1]`. When `summary` belongs to a
    /// different corpus than `self` the function still computes the
    /// blend — verifying corpus_id alignment is the caller's
    /// responsibility; this keeps the API composable under custom
    /// lookup schemes.
    ///
    /// Note: this blends `best_score`, not [`Self::score_lift`]. The
    /// result is on the raw-score scale, which is not comparable
    /// across corpora of different difficulty — don't substitute it
    /// for `score_lift` in cross-corpus comparisons.
    pub fn adjust_score_with_feedback(&self, summary: &FeedbackSummary, alpha: f64) -> f64 {
        let a = alpha.clamp(0.0, 1.0);
        (1.0 - a) * self.best_score + a * summary.mean_score
    }
}

/// Compute [`MetaTrainingRecord::score_lift`] from a tuner report:
/// `(best − mean) / (1 − mean)` — the fraction of the run's available
/// headroom the winning config captured. `None` with fewer than 2
/// trials (no distribution to compare against); `Some(0.0)` when every
/// trial already scored ~1.0 (the config demonstrably didn't matter).
fn score_lift_from_report(report: &TuneReport) -> Option<f64> {
    if report.trials.len() < 2 {
        return None;
    }
    let mean = report.mean_score();
    let headroom = 1.0 - mean;
    if headroom < 1e-9 {
        return Some(0.0);
    }
    Some(((report.best_score - mean) / headroom).clamp(0.0, 1.0))
}

// ── Shared helpers ─────────────────────────────────────────────────

/// Indices of the scale-type features in [`CorpusFeatures::to_vec`]
/// (`n_items`, `n_categories`, `dim`, `mean_members_per_category`)
/// that get `ln(1+x)` compression before z-scoring. Raw counts span
/// 775 → 500,000 across real corpora; without the log transform a
/// single large corpus either dominates the normalized distance or
/// stretches the z-scale until every other corpus collapses together.
const LOG_SCALED_FEATURES: [usize; 4] = [0, 1, 2, 3];

/// Map a raw feature vector into model space: `ln(1+x)` on the scale
/// features, everything else unchanged. Applied consistently to both
/// training records and queries before normalization.
fn to_model_space(raw: &[f64; CORPUS_FEATURE_COUNT]) -> [f64; CORPUS_FEATURE_COUNT] {
    let mut out = *raw;
    for &i in &LOG_SCALED_FEATURES {
        out[i] = out[i].max(0.0).ln_1p();
    }
    out
}

/// Retain only the records sharing the most common `metric_name`.
/// Scores produced under different objectives aren't comparable, so
/// mixing them corrupts both the z-score statistics and the
/// distance-weighted selection. Ties break toward the
/// lexicographically largest name for determinism. Returns the input
/// unchanged when it is empty or already single-metric.
fn filter_dominant_metric(records: &[MetaTrainingRecord]) -> Vec<MetaTrainingRecord> {
    if records.is_empty() {
        return Vec::new();
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for r in records {
        *counts.entry(r.metric_name.as_str()).or_default() += 1;
    }
    if counts.len() <= 1 {
        return records.to_vec();
    }
    let dominant = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)))
        .map(|(k, _)| (*k).to_string())
        .expect("counts non-empty");
    records
        .iter()
        .filter(|r| r.metric_name == dominant)
        .cloned()
        .collect()
}

/// Per-feature mean + std computed across a training set (in model
/// space — see [`to_model_space`]), used for z-score normalization by
/// both meta-model implementations.
///
/// Returns `(means, stds)`. Features with near-zero variance get a
/// stored std of `0.0` rather than the true tiny value, so
/// [`normalize_features`] can detect the degenerate case and zero the
/// feature out instead of dividing by something that blows up.
fn compute_feature_stats(
    records: &[MetaTrainingRecord],
) -> ([f64; CORPUS_FEATURE_COUNT], [f64; CORPUS_FEATURE_COUNT]) {
    let mut means = [0.0; CORPUS_FEATURE_COUNT];
    let mut stds = [0.0; CORPUS_FEATURE_COUNT];
    let n = records.len();
    if n == 0 {
        return (means, [1.0; CORPUS_FEATURE_COUNT]);
    }
    let vecs: Vec<[f64; CORPUS_FEATURE_COUNT]> = records
        .iter()
        .map(|r| to_model_space(&r.features.to_vec()))
        .collect();

    for i in 0..CORPUS_FEATURE_COUNT {
        let mean: f64 = vecs.iter().map(|v| v[i]).sum::<f64>() / n as f64;
        means[i] = mean;
        let var: f64 =
            vecs.iter().map(|v| (v[i] - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        let sd = var.sqrt();
        stds[i] = if sd > f64::EPSILON { sd } else { 0.0 };
    }
    (means, stds)
}

/// Z-score normalize a model-space feature vector against precomputed
/// `means`/`stds`. Features whose stored std is below `f64::EPSILON`
/// (zero-variance in the training set) map to `0.0` rather than
/// dividing by a near-zero number.
fn normalize_features(
    model_space: &[f64; CORPUS_FEATURE_COUNT],
    means: &[f64; CORPUS_FEATURE_COUNT],
    stds: &[f64; CORPUS_FEATURE_COUNT],
) -> [f64; CORPUS_FEATURE_COUNT] {
    let mut out = [0.0; CORPUS_FEATURE_COUNT];
    for i in 0..CORPUS_FEATURE_COUNT {
        let sd = stds[i];
        out[i] = if sd > f64::EPSILON {
            (model_space[i] - means[i]) / sd
        } else {
            0.0
        };
    }
    out
}

/// Euclidean distance between two z-score-normalized feature vectors.
fn normalized_euclidean(a: &[f64; CORPUS_FEATURE_COUNT], b: &[f64; CORPUS_FEATURE_COUNT]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Upper median of a non-empty value sequence (sorted by `total_cmp`).
fn median_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    assert!(!v.is_empty(), "median of empty sequence");
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Upper median of a non-empty integer sequence.
fn median_usize(values: impl Iterator<Item = usize>) -> usize {
    let mut v: Vec<usize> = values.collect();
    assert!(!v.is_empty(), "median of empty sequence");
    v.sort_unstable();
    v[v.len() / 2]
}

// ── Trait ──────────────────────────────────────────────────────────

/// Predicts a [`PipelineConfig`] from a [`CorpusFeatures`] profile.
///
/// Implementers fit on a training set of [`MetaTrainingRecord`]s (pairs
/// of (features, best_config) observed from past tuner runs) and predict
/// a config for a new corpus.
pub trait MetaModel {
    /// Fit on a training set. Replacing any prior state.
    fn fit(&mut self, records: &[MetaTrainingRecord]);

    /// True once `fit` has been called with at least one usable record.
    /// [`Self::predict`] panics when this is false, so Result-returning
    /// boundaries (e.g. `SphereQLPipeline::new_from_metamodel`) check
    /// this first. No default impl on purpose: every model must answer
    /// for its own notion of "fitted" rather than inherit a guess.
    fn is_fitted(&self) -> bool;

    /// Predict the config that should work best on a corpus with the
    /// given profile. Panics if `fit` has not been called with at least
    /// one record — callers should treat `MetaModel` as a trained object
    /// and front-load `fit`, or check [`Self::is_fitted`] when the
    /// training state isn't statically known.
    fn predict(&self, features: &CorpusFeatures) -> PipelineConfig;

    /// Short name for logs and model comparison.
    fn name(&self) -> &str;
}

// ── Nearest-neighbor baseline ─────────────────────────────────────────

/// The simplest useful meta-model: given a new corpus, return the
/// best_config of the training record whose corpus-feature vector is
/// closest in z-score-normalized Euclidean distance (scale features
/// log-compressed first — see [`to_model_space`]).
///
/// - Works with `N ≥ 1` records.
/// - Deterministic; no hyperparameters.
/// - Degenerate features (zero variance across training records) are
///   dropped from the distance computation at fit time so they don't
///   divide by zero or dominate via raw-scale inflation.
/// - Training sets mixing multiple `metric_name`s are stratified to
///   the dominant metric at fit time — see [`Self::records`] for what
///   was actually retained.
#[derive(Debug, Clone)]
pub struct NearestNeighborMetaModel {
    records: Vec<MetaTrainingRecord>,
    feature_means: [f64; CORPUS_FEATURE_COUNT],
    feature_stds: [f64; CORPUS_FEATURE_COUNT],
}

impl Default for NearestNeighborMetaModel {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            feature_means: [0.0; CORPUS_FEATURE_COUNT],
            feature_stds: [1.0; CORPUS_FEATURE_COUNT],
        }
    }
}

impl NearestNeighborMetaModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the training records — useful for introspecting what the
    /// model considers the nearest-neighbor candidate pool (after the
    /// dominant-metric stratification applied by `fit`).
    pub fn records(&self) -> &[MetaTrainingRecord] {
        &self.records
    }

    /// Distance from a given feature vector to every stored record,
    /// sorted ascending. Returned as `(record_index, distance)` pairs.
    pub fn rank_candidates(&self, features: &CorpusFeatures) -> Vec<(usize, f64)> {
        let q = normalize_features(
            &to_model_space(&features.to_vec()),
            &self.feature_means,
            &self.feature_stds,
        );
        let mut ranked: Vec<(usize, f64)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let v = normalize_features(
                    &to_model_space(&r.features.to_vec()),
                    &self.feature_means,
                    &self.feature_stds,
                );
                (i, normalized_euclidean(&q, &v))
            })
            .collect();
        // `total_cmp` sorts NaN to the end — which is what we want
        // under a "nearest first" policy: any record whose distance
        // is non-finite sinks to the bottom instead of silently
        // equating with finite candidates.
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        ranked
    }

    /// k-NN prediction with per-knob aggregation, instead of copying
    /// the single nearest record's config wholesale.
    ///
    /// Takes the `k` nearest records (clamped to the training-set
    /// size), picks the majority `projection_kind` (ties break toward
    /// the nearest record), and sets each tuned knob to the median of
    /// the top-k values — kind-specific knobs aggregate over
    /// kind-matching neighbors only.
    ///
    /// Blended knobs: `routing.num_domain_groups`,
    /// `routing.low_evr_threshold`, `bridges.threshold_base`,
    /// `bridges.threshold_evr_penalty`,
    /// `bridges.overlap_artifact_territorial`,
    /// `inner_sphere.min_evr_improvement`, plus the kind-specific
    /// `laplacian.*` / `umap.*` knobs. Everything else
    /// (`bridges.balanced_affinity_quantile`,
    /// `bridges.min_evr_for_classification`,
    /// `routing.group_routing_alpha`, the remaining `inner_sphere.*`
    /// knobs, `spatial.*`, `min_category_size`) is inherited
    /// nearest-neighbor from the closest record.
    ///
    /// With `k = 1` this is exactly [`MetaModel::predict`]. Larger `k`
    /// trades sharpness for robustness against a single poorly-tuned
    /// or mislabeled record. Panics like `predict` if `fit` has not
    /// been called with at least one record.
    pub fn predict_blended(&self, features: &CorpusFeatures, k: usize) -> PipelineConfig {
        assert!(
            !self.records.is_empty(),
            "NearestNeighborMetaModel::predict_blended called before fit(); \
             call .fit(records) with at least one record first"
        );
        let ranked = self.rank_candidates(features);
        let k = k.clamp(1, ranked.len());
        let top: Vec<&MetaTrainingRecord> =
            ranked[..k].iter().map(|&(i, _)| &self.records[i]).collect();

        // Majority projection kind; `top` is nearest-first so `find`
        // breaks ties toward the closest record.
        let mut kind_counts: HashMap<ProjectionKind, usize> = HashMap::new();
        for r in &top {
            *kind_counts
                .entry(r.best_config.projection_kind)
                .or_default() += 1;
        }
        let max_count = kind_counts.values().copied().max().unwrap_or(0);
        let kind = top
            .iter()
            .map(|r| r.best_config.projection_kind)
            .find(|kk| kind_counts[kk] == max_count)
            .unwrap_or(top[0].best_config.projection_kind);

        // Fields not blended below — bridges.balanced_affinity_quantile,
        // bridges.min_evr_for_classification, routing.group_routing_alpha,
        // the remaining inner_sphere.* knobs, spatial.*, and
        // min_category_size — are intentionally inherited from the
        // nearest record: they're outside the tuner's SearchSpace, so
        // every record carries the same defaults and a median across
        // neighbors would add nothing.
        let mut cfg = top[0].best_config.clone();
        cfg.projection_kind = kind;

        // Kind-agnostic knobs: median over the full top-k.
        cfg.routing.num_domain_groups =
            median_usize(top.iter().map(|r| r.best_config.routing.num_domain_groups));
        cfg.routing.low_evr_threshold =
            median_f64(top.iter().map(|r| r.best_config.routing.low_evr_threshold));
        cfg.bridges.threshold_base =
            median_f64(top.iter().map(|r| r.best_config.bridges.threshold_base));
        cfg.bridges.threshold_evr_penalty = median_f64(
            top.iter()
                .map(|r| r.best_config.bridges.threshold_evr_penalty),
        );
        cfg.bridges.overlap_artifact_territorial = median_f64(
            top.iter()
                .map(|r| r.best_config.bridges.overlap_artifact_territorial),
        );
        cfg.inner_sphere.min_evr_improvement = median_f64(
            top.iter()
                .map(|r| r.best_config.inner_sphere.min_evr_improvement),
        );

        // Kind-specific knobs: aggregate over kind-matching neighbors
        // only. Non-empty by construction — `kind` won the majority
        // vote over `top`, so at least one member matches it.
        let kind_matching: Vec<&&MetaTrainingRecord> = top
            .iter()
            .filter(|r| r.best_config.projection_kind == kind)
            .collect();
        if !kind_matching.is_empty() {
            match kind {
                ProjectionKind::LaplacianEigenmap => {
                    cfg.laplacian.k_neighbors = median_usize(
                        kind_matching
                            .iter()
                            .map(|r| r.best_config.laplacian.k_neighbors),
                    );
                    cfg.laplacian.active_threshold = median_f64(
                        kind_matching
                            .iter()
                            .map(|r| r.best_config.laplacian.active_threshold),
                    );
                }
                ProjectionKind::UmapSphere => {
                    cfg.umap.n_neighbors =
                        median_usize(kind_matching.iter().map(|r| r.best_config.umap.n_neighbors));
                    cfg.umap.n_epochs =
                        median_usize(kind_matching.iter().map(|r| r.best_config.umap.n_epochs));
                    cfg.umap.category_weight = median_f64(
                        kind_matching
                            .iter()
                            .map(|r| r.best_config.umap.category_weight),
                    );
                }
                ProjectionKind::Pca | ProjectionKind::KernelPca => {}
            }
        }
        cfg
    }
}

impl MetaModel for NearestNeighborMetaModel {
    fn fit(&mut self, records: &[MetaTrainingRecord]) {
        self.records = filter_dominant_metric(records);
        let (means, stds) = compute_feature_stats(&self.records);
        self.feature_means = means;
        self.feature_stds = if self.records.is_empty() {
            [1.0; CORPUS_FEATURE_COUNT]
        } else {
            stds
        };
    }

    fn is_fitted(&self) -> bool {
        !self.records.is_empty()
    }

    fn predict(&self, features: &CorpusFeatures) -> PipelineConfig {
        // Invariant: callers are expected to call fit() before predict().
        // The trait contract documents this requirement, and the panic is
        // intentional — a silent wrong prediction (returning Default) would
        // be much harder to diagnose than a clear failure at the call site.
        assert!(
            !self.records.is_empty(),
            "NearestNeighborMetaModel::predict called before fit(); \
             call .fit(records) with at least one record first"
        );
        let ranked = self.rank_candidates(features);
        let best_idx = ranked[0].0;
        self.records[best_idx].best_config.clone()
    }

    fn name(&self) -> &str {
        "nearest_neighbor"
    }
}

// ── Distance-weighted ─────────────────────────────────────────────────

/// Picks the training record that maximizes `evidence × w(distance)`,
/// where `w(d) = 1 / (d + epsilon)` over z-score-normalized Euclidean
/// distance and `evidence` is [`MetaTrainingRecord::score_lift`] when
/// present, falling back to `best_score` for legacy records.
///
/// The distinction from [`NearestNeighborMetaModel`]: NN picks the
/// closest record regardless of how well that record performed, so a
/// single poorly-tuned outlier can pull predictions off. Distance-weighted
/// folds the record's evidence into the selection — a record is "good" if
/// it's both similar to the query AND demonstrated real tuner signal.
/// Using lift instead of the raw score avoids the easy-corpus bias:
/// `best_score = 0.9` on an easy corpus is weaker evidence than
/// `best_score = 0.6` that beat its run's mean by a wide margin. At
/// N = 1 this degenerates to NN (same record either way).
///
/// `epsilon` is a smoothing floor on the distance term; at `d ≈ 0` it
/// prevents the weight from exploding and over-committing to a single
/// near-duplicate record. Default `0.1`.
#[derive(Debug, Clone)]
pub struct DistanceWeightedMetaModel {
    records: Vec<MetaTrainingRecord>,
    feature_means: [f64; CORPUS_FEATURE_COUNT],
    feature_stds: [f64; CORPUS_FEATURE_COUNT],
    epsilon: f64,
}

impl Default for DistanceWeightedMetaModel {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            feature_means: [0.0; CORPUS_FEATURE_COUNT],
            feature_stds: [1.0; CORPUS_FEATURE_COUNT],
            epsilon: 0.1,
        }
    }
}

impl DistanceWeightedMetaModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the smoothing constant added to distance before
    /// inversion. Larger `epsilon` makes predictions smoother; smaller
    /// sharpens the preference for near-duplicate records. Must be
    /// strictly positive (silently clamped to `1e-12` if a zero or
    /// negative value is passed).
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon.max(1e-12);
        self
    }

    pub fn records(&self) -> &[MetaTrainingRecord] {
        &self.records
    }

    /// Per-record (weighted_score, distance) pairs for the given query
    /// features, sorted by descending weighted score. Useful for
    /// introspecting why a particular prediction was made.
    pub fn score_candidates(&self, features: &CorpusFeatures) -> Vec<(usize, f64, f64)> {
        let q = normalize_features(
            &to_model_space(&features.to_vec()),
            &self.feature_means,
            &self.feature_stds,
        );
        let mut out: Vec<(usize, f64, f64)> = self
            .records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                // Cross-corpus-comparable evidence when available;
                // raw best_score for legacy records without lift.
                let evidence = r.score_lift.unwrap_or(r.best_score);
                // Filter non-finite evidence at score time. NaN would
                // otherwise propagate into `weighted`, hit the `total_cmp`
                // below as "greatest" (NaN sorts to the end of a total
                // order, but the *top* under a "descending" sort would
                // put NaN first), and silently become the prediction.
                if !evidence.is_finite() {
                    return None;
                }
                let v = normalize_features(
                    &to_model_space(&r.features.to_vec()),
                    &self.feature_means,
                    &self.feature_stds,
                );
                let d = normalized_euclidean(&q, &v);
                let weighted = evidence / (d + self.epsilon);
                if !weighted.is_finite() {
                    return None;
                }
                Some((i, weighted, d))
            })
            .collect();
        // `total_cmp` is NaN-safe; non-finite scores were already
        // dropped above, so the ordering is total.
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

impl MetaModel for DistanceWeightedMetaModel {
    fn fit(&mut self, records: &[MetaTrainingRecord]) {
        self.records = filter_dominant_metric(records);
        let (means, stds) = compute_feature_stats(&self.records);
        self.feature_means = means;
        self.feature_stds = if self.records.is_empty() {
            [1.0; CORPUS_FEATURE_COUNT]
        } else {
            stds
        };
    }

    fn is_fitted(&self) -> bool {
        !self.records.is_empty()
    }

    fn predict(&self, features: &CorpusFeatures) -> PipelineConfig {
        // Invariant: callers are expected to call fit() before predict().
        // The trait contract documents this requirement, and the panic is
        // intentional — a silent wrong prediction (returning Default) would
        // be much harder to diagnose than a clear failure at the call site.
        assert!(
            !self.records.is_empty(),
            "DistanceWeightedMetaModel::predict called before fit(); \
             call .fit(records) with at least one record first"
        );
        let ranked = self.score_candidates(features);
        // Fall back to record 0 if every record was filtered as
        // non-finite — the records are non-empty (asserted) but none
        // produced a comparable score.
        let best_idx = ranked.first().map_or(0, |&(idx, _, _)| idx);
        self.records[best_idx].best_config.clone()
    }

    fn name(&self) -> &str {
        "distance_weighted"
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProjectionKind;
    use crate::tuner::TrialRecord;

    fn feat(n: usize, c: usize, sparsity: f64, intra: f64) -> CorpusFeatures {
        CorpusFeatures {
            n_items: n,
            n_categories: c,
            dim: 128,
            mean_members_per_category: n as f64 / c as f64,
            category_size_entropy: 1.0,
            mean_sparsity: sparsity,
            axis_utilization_entropy: 0.9,
            noise_estimate: 0.02,
            mean_intra_category_similarity: intra,
            mean_inter_category_similarity: 0.1,
            category_separation_ratio: intra / 0.1,
        }
    }

    fn record(id: &str, f: CorpusFeatures, kind: ProjectionKind, score: f64) -> MetaTrainingRecord {
        MetaTrainingRecord {
            corpus_id: id.to_string(),
            features: f,
            best_config: PipelineConfig {
                projection_kind: kind,
                ..Default::default()
            },
            best_score: score,
            score_lift: None,
            metric_name: "test_metric".to_string(),
            strategy: "test_strategy".to_string(),
            timestamp: "2026-04-22T00:00:00Z".to_string(),
        }
    }

    fn trial(score: f64) -> TrialRecord {
        TrialRecord {
            config: PipelineConfig::default(),
            score,
            build_ms: 0,
            components: Vec::new(),
        }
    }

    #[test]
    fn record_json_roundtrip() {
        let r = record("r1", feat(100, 5, 0.2, 0.6), ProjectionKind::Pca, 0.5);
        let json = serde_json::to_string(&r).unwrap();
        let back: MetaTrainingRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.corpus_id, "r1");
        assert_eq!(back.best_config.projection_kind, ProjectionKind::Pca);
        assert!((back.best_score - 0.5).abs() < 1e-12);
    }

    #[test]
    fn record_without_score_lift_field_still_deserializes() {
        // Legacy stores predate `score_lift`; #[serde(default)] must
        // accept them and produce None.
        let r = record("r1", feat(100, 5, 0.2, 0.6), ProjectionKind::Pca, 0.5);
        let mut json: serde_json::Value = serde_json::to_value(&r).unwrap();
        json.as_object_mut().unwrap().remove("score_lift");
        let back: MetaTrainingRecord = serde_json::from_value(json).unwrap();
        assert!(back.score_lift.is_none());
    }

    #[test]
    fn to_model_space_log_compresses_scale_features_only() {
        let f = feat(500, 20, 0.25, 0.6);
        let raw = f.to_vec();
        let ms = to_model_space(&raw);
        for &i in &LOG_SCALED_FEATURES {
            assert!(
                (ms[i] - raw[i].ln_1p()).abs() < 1e-12,
                "scale feature {i} should be ln(1+x)"
            );
        }
        for i in 0..CORPUS_FEATURE_COUNT {
            if !LOG_SCALED_FEATURES.contains(&i) {
                assert_eq!(ms[i], raw[i], "non-scale feature {i} must pass through");
            }
        }
    }

    #[test]
    fn is_fitted_flips_after_fit() {
        let mut nn = NearestNeighborMetaModel::new();
        let mut dw = DistanceWeightedMetaModel::new();
        assert!(!nn.is_fitted());
        assert!(!dw.is_fitted());

        let r = record("only", feat(500, 20, 0.1, 0.4), ProjectionKind::Pca, 0.7);
        nn.fit(std::slice::from_ref(&r));
        dw.fit(std::slice::from_ref(&r));
        assert!(nn.is_fitted());
        assert!(dw.is_fitted());

        nn.fit(&[]);
        assert!(
            !nn.is_fitted(),
            "refit on empty set must clear fitted state"
        );
    }

    #[test]
    fn nn_predict_single_record_returns_its_config() {
        let r = record(
            "only",
            feat(500, 20, 0.1, 0.4),
            ProjectionKind::LaplacianEigenmap,
            0.7,
        );
        let mut m = NearestNeighborMetaModel::new();
        m.fit(std::slice::from_ref(&r));
        let predicted = m.predict(&feat(1000, 30, 0.05, 0.3));
        assert_eq!(predicted.projection_kind, ProjectionKind::LaplacianEigenmap);
    }

    #[test]
    fn nn_predict_picks_nearest_neighbor() {
        // Two records with very different features. A query close to r_a
        // should get r_a's config.
        let r_a = record(
            "sparse",
            feat(500, 5, 0.05, 0.8),
            ProjectionKind::LaplacianEigenmap,
            0.7,
        );
        let r_b = record("dense", feat(500, 5, 0.50, 0.2), ProjectionKind::Pca, 0.6);
        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r_a.clone(), r_b.clone()]);

        let query_near_a = feat(500, 5, 0.06, 0.78);
        let query_near_b = feat(500, 5, 0.48, 0.22);

        assert_eq!(
            m.predict(&query_near_a).projection_kind,
            ProjectionKind::LaplacianEigenmap,
        );
        assert_eq!(
            m.predict(&query_near_b).projection_kind,
            ProjectionKind::Pca,
        );
    }

    #[test]
    fn nn_rank_candidates_sorted_ascending() {
        let r_a = record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.7);
        let r_b = record("b", feat(500, 5, 0.50, 0.2), ProjectionKind::KernelPca, 0.6);
        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r_a, r_b]);
        let q = feat(500, 5, 0.07, 0.75);
        let ranked = m.rank_candidates(&q);
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].1 <= ranked[1].1);
    }

    #[test]
    fn nn_handles_zero_variance_feature() {
        // Both records have identical n_items/n_categories/dim — those
        // features have zero std and should be ignored in the distance
        // rather than produce NaN.
        let r_a = record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.7);
        let r_b = record(
            "b",
            feat(500, 5, 0.50, 0.2),
            ProjectionKind::LaplacianEigenmap,
            0.6,
        );
        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r_a, r_b]);
        let q = feat(500, 5, 0.1, 0.7);
        let ranked = m.rank_candidates(&q);
        assert!(ranked[0].1.is_finite());
        assert!(ranked[1].1.is_finite());
    }

    #[test]
    fn fit_stratifies_to_dominant_metric() {
        // Two "m1" records + one "m2" record: fit must retain only the
        // m1 pair, so even a query sitting exactly on the m2 record's
        // features predicts an m1 config.
        let r1 = record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.7);
        let r2 = record("b", feat(500, 5, 0.50, 0.2), ProjectionKind::Pca, 0.6);
        let mut alien = record("c", feat(500, 5, 0.30, 0.5), ProjectionKind::KernelPca, 0.9);
        alien.metric_name = "other_metric".to_string();

        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r1, r2, alien.clone()]);
        assert_eq!(m.records().len(), 2, "dominant-metric records retained");
        assert!(m.records().iter().all(|r| r.metric_name == "test_metric"));
        let predicted = m.predict(&alien.features);
        assert_ne!(predicted.projection_kind, ProjectionKind::KernelPca);
    }

    #[test]
    fn filter_dominant_metric_tie_picks_lexicographically_largest() {
        // Three metrics, one record each — an exact 3-way tie. The
        // deterministic tie-break is the lexicographically largest name.
        let mut r1 = record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.7);
        r1.metric_name = "alpha".to_string();
        let mut r2 = record("b", feat(500, 5, 0.50, 0.2), ProjectionKind::Pca, 0.6);
        r2.metric_name = "beta".to_string();
        let mut r3 = record("c", feat(500, 5, 0.30, 0.5), ProjectionKind::Pca, 0.5);
        r3.metric_name = "gamma".to_string();

        let kept = filter_dominant_metric(&[r1, r2, r3]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].metric_name, "gamma");
    }

    #[test]
    fn single_metric_training_set_is_untouched() {
        let records = vec![
            record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.7),
            record("b", feat(500, 5, 0.50, 0.2), ProjectionKind::Pca, 0.6),
        ];
        let mut m = NearestNeighborMetaModel::new();
        m.fit(&records);
        assert_eq!(m.records().len(), 2);
    }

    #[test]
    #[should_panic(expected = "called before fit")]
    fn nn_predict_before_fit_panics() {
        let m = NearestNeighborMetaModel::new();
        let _ = m.predict(&feat(100, 5, 0.1, 0.3));
    }

    #[test]
    fn predict_blended_k1_matches_predict() {
        let r_a = record(
            "a",
            feat(500, 5, 0.05, 0.8),
            ProjectionKind::LaplacianEigenmap,
            0.7,
        );
        let r_b = record("b", feat(500, 5, 0.50, 0.2), ProjectionKind::Pca, 0.6);
        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r_a, r_b]);
        let q = feat(500, 5, 0.06, 0.78);
        let single = m.predict(&q);
        let blended = m.predict_blended(&q, 1);
        assert_eq!(blended.projection_kind, single.projection_kind);
        assert_eq!(
            blended.routing.num_domain_groups,
            single.routing.num_domain_groups
        );
        assert!((blended.bridges.threshold_base - single.bridges.threshold_base).abs() < 1e-12);
    }

    #[test]
    fn predict_blended_takes_median_of_knobs() {
        // Three same-kind records whose num_domain_groups are 3, 5, 9:
        // the k=3 blend must pick the median (5), not any single
        // record's value.
        let mut r1 = record("a", feat(500, 5, 0.10, 0.70), ProjectionKind::Pca, 0.7);
        r1.best_config.routing.num_domain_groups = 3;
        let mut r2 = record("b", feat(500, 5, 0.12, 0.68), ProjectionKind::Pca, 0.6);
        r2.best_config.routing.num_domain_groups = 5;
        let mut r3 = record("c", feat(500, 5, 0.14, 0.66), ProjectionKind::Pca, 0.5);
        r3.best_config.routing.num_domain_groups = 9;

        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r1, r2, r3]);
        let blended = m.predict_blended(&feat(500, 5, 0.12, 0.68), 3);
        assert_eq!(blended.projection_kind, ProjectionKind::Pca);
        assert_eq!(blended.routing.num_domain_groups, 5);
    }

    #[test]
    fn predict_blended_majority_kind_wins() {
        // Two Laplacian records + one PCA record: the blend at k=3 must
        // pick Laplacian, and aggregate laplacian knobs over the two
        // kind-matching neighbors only.
        let mut r1 = record(
            "a",
            feat(500, 5, 0.10, 0.70),
            ProjectionKind::LaplacianEigenmap,
            0.7,
        );
        r1.best_config.laplacian.k_neighbors = 10;
        let mut r2 = record(
            "b",
            feat(500, 5, 0.12, 0.68),
            ProjectionKind::LaplacianEigenmap,
            0.6,
        );
        r2.best_config.laplacian.k_neighbors = 20;
        let r3 = record("c", feat(500, 5, 0.14, 0.66), ProjectionKind::Pca, 0.5);

        let mut m = NearestNeighborMetaModel::new();
        m.fit(&[r1, r2, r3]);
        let blended = m.predict_blended(&feat(500, 5, 0.12, 0.68), 3);
        assert_eq!(blended.projection_kind, ProjectionKind::LaplacianEigenmap);
        // Upper median of {10, 20} = 20.
        assert_eq!(blended.laplacian.k_neighbors, 20);
    }

    #[test]
    fn save_and_load_list_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("sphereql_meta_test.json");
        let _ = fs::remove_file(&path);

        let records = vec![
            record("r1", feat(100, 5, 0.2, 0.5), ProjectionKind::Pca, 0.4),
            record(
                "r2",
                feat(800, 30, 0.05, 0.6),
                ProjectionKind::LaplacianEigenmap,
                0.5,
            ),
        ];
        MetaTrainingRecord::save_list(&records, &path).unwrap();

        let loaded = MetaTrainingRecord::load_list(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].corpus_id, "r1");
        assert_eq!(
            loaded[1].best_config.projection_kind,
            ProjectionKind::LaplacianEigenmap
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let path = std::env::temp_dir().join("sphereql_nonexistent_12345.json");
        let loaded = MetaTrainingRecord::load_list(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn append_to_migrates_legacy_array_file() {
        let dir =
            std::env::temp_dir().join(format!("sphereql_meta_migrate_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("records.json");

        // Seed with a legacy array file (what `save_list` writes).
        let legacy = vec![
            record("r1", feat(100, 5, 0.2, 0.5), ProjectionKind::Pca, 0.4),
            record(
                "r2",
                feat(800, 30, 0.05, 0.6),
                ProjectionKind::LaplacianEigenmap,
                0.5,
            ),
        ];
        MetaTrainingRecord::save_list(&legacy, &path).unwrap();

        // First append migrates the file to JSONL.
        record("r3", feat(200, 8, 0.1, 0.4), ProjectionKind::KernelPca, 0.6)
            .append_to(&path)
            .unwrap();

        let loaded = MetaTrainingRecord::load_list(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].corpus_id, "r1");
        assert_eq!(loaded[1].corpus_id, "r2");
        assert_eq!(loaded[2].corpus_id, "r3");
        assert_eq!(
            loaded[1].best_config.projection_kind,
            ProjectionKind::LaplacianEigenmap
        );

        // Post-migration shape is JSONL (one record per line).
        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.trim_start().starts_with('['));
        assert_eq!(raw.lines().count(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_tune_result_copies_fields() {
        let cfg = PipelineConfig {
            projection_kind: ProjectionKind::LaplacianEigenmap,
            ..Default::default()
        };
        let report = TuneReport {
            metric_name: "connectivity_composite".to_string(),
            best_score: 0.42,
            best_config: cfg.clone(),
            trials: Vec::new(),
            failures: Vec::new(),
            umap_graph_builds: 0,
        };
        let r = MetaTrainingRecord::from_tune_result(
            "test_corpus",
            feat(100, 5, 0.1, 0.5),
            &report,
            "random{budget=24,seed=42}",
        );
        assert_eq!(r.corpus_id, "test_corpus");
        assert_eq!(r.metric_name, "connectivity_composite");
        assert!((r.best_score - 0.42).abs() < 1e-12);
        // Fewer than 2 trials → no lift evidence.
        assert!(r.score_lift.is_none());
        assert_eq!(
            r.best_config.projection_kind,
            ProjectionKind::LaplacianEigenmap
        );
        assert_eq!(r.strategy, "random{budget=24,seed=42}");
        // Timestamp should be epoch-seconds-ish — a non-empty numeric string.
        assert!(!r.timestamp.is_empty());
        assert!(r.timestamp.parse::<u64>().is_ok());
    }

    #[test]
    fn from_tune_result_computes_headroom_lift() {
        // Trials {0.4, 0.6, 0.8}: mean = 0.6, best = 0.8, headroom = 0.4,
        // lift = (0.8 - 0.6) / 0.4 = 0.5.
        let report = TuneReport {
            metric_name: "m".to_string(),
            best_score: 0.8,
            best_config: PipelineConfig::default(),
            trials: vec![trial(0.4), trial(0.6), trial(0.8)],
            failures: Vec::new(),
            umap_graph_builds: 0,
        };
        let r = MetaTrainingRecord::from_tune_result("c", feat(10, 2, 0.1, 0.3), &report, "s");
        let lift = r.score_lift.expect("two or more trials produce lift");
        assert!((lift - 0.5).abs() < 1e-12, "got {lift}");
    }

    #[test]
    fn from_tune_result_single_trial_has_no_lift() {
        // Exactly one trial: no distribution to compare against, so
        // the record carries no lift evidence.
        let report = TuneReport {
            metric_name: "m".to_string(),
            best_score: 0.7,
            best_config: PipelineConfig::default(),
            trials: vec![trial(0.7)],
            failures: Vec::new(),
            umap_graph_builds: 0,
        };
        let r = MetaTrainingRecord::from_tune_result("c", feat(10, 2, 0.1, 0.3), &report, "s");
        assert!(r.score_lift.is_none());
    }

    #[test]
    fn from_tune_result_lift_zero_when_landscape_saturated() {
        // Every trial at ~1.0: no headroom, the config carried no signal.
        let report = TuneReport {
            metric_name: "m".to_string(),
            best_score: 1.0,
            best_config: PipelineConfig::default(),
            trials: vec![trial(1.0), trial(1.0)],
            failures: Vec::new(),
            umap_graph_builds: 0,
        };
        let r = MetaTrainingRecord::from_tune_result("c", feat(10, 2, 0.1, 0.3), &report, "s");
        assert_eq!(r.score_lift, Some(0.0));
    }

    #[test]
    fn with_timestamp_overrides_default() {
        let report = TuneReport {
            metric_name: "m".to_string(),
            best_score: 0.5,
            best_config: PipelineConfig::default(),
            trials: Vec::new(),
            failures: Vec::new(),
            umap_graph_builds: 0,
        };
        let r = MetaTrainingRecord::from_tune_result("c", feat(10, 2, 0.1, 0.3), &report, "s")
            .with_timestamp("2026-04-22T12:00:00Z");
        assert_eq!(r.timestamp, "2026-04-22T12:00:00Z");
    }

    #[test]
    fn save_list_creates_parent_dirs() {
        let dir = std::env::temp_dir().join(format!("sphereql_create_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("records.json");

        let r = record("r1", feat(100, 5, 0.1, 0.5), ProjectionKind::Pca, 0.4);
        MetaTrainingRecord::save_list(&[r], &path).unwrap();
        assert!(path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_store_path_resolves() {
        // Verify the helper returns a path under $HOME or $USERPROFILE.
        // We can't assert the exact path (portability + test isolation),
        // just that it resolves and ends with the expected filename.
        let path = MetaTrainingRecord::default_store_path().unwrap();
        assert!(path.ends_with("meta_records.json"));
        assert!(path.iter().any(|c| c.to_string_lossy() == ".sphereql"));
    }

    #[test]
    fn dw_predict_single_record_returns_its_config() {
        // At N=1 distance-weighted must agree with NN.
        let r = record(
            "only",
            feat(500, 20, 0.1, 0.4),
            ProjectionKind::LaplacianEigenmap,
            0.7,
        );
        let mut m = DistanceWeightedMetaModel::new();
        m.fit(std::slice::from_ref(&r));
        let predicted = m.predict(&feat(1000, 30, 0.05, 0.3));
        assert_eq!(predicted.projection_kind, ProjectionKind::LaplacianEigenmap);
    }

    #[test]
    fn dw_prefers_higher_score_when_equidistant() {
        // Two records at identical features but different best_scores
        // — the high-score one should be picked.
        let shared_feat = feat(500, 5, 0.1, 0.5);
        let lo = record(
            "low",
            shared_feat.clone(),
            ProjectionKind::LaplacianEigenmap,
            0.2,
        );
        let hi = record("high", shared_feat.clone(), ProjectionKind::Pca, 0.9);

        let mut m = DistanceWeightedMetaModel::new();
        m.fit(&[lo, hi]);
        let predicted = m.predict(&shared_feat);
        // Note: at perfectly identical features, distance is 0 and both
        // weights are 1/epsilon; the higher-score record wins.
        assert_eq!(predicted.projection_kind, ProjectionKind::Pca);
    }

    #[test]
    fn dw_prefers_lift_evidence_over_raw_score() {
        // Same features; one record from an "easy" corpus (raw 0.9,
        // but the run was flat — lift 0.0) and one from a "hard"
        // corpus (raw 0.6, but the config beat its run's mean by a
        // wide margin — lift 0.8). The hard-won config is the better
        // evidence and must win.
        let shared_feat = feat(500, 5, 0.1, 0.5);
        let mut easy = record("easy", shared_feat.clone(), ProjectionKind::KernelPca, 0.9);
        easy.score_lift = Some(0.0);
        let mut hard = record(
            "hard",
            shared_feat.clone(),
            ProjectionKind::LaplacianEigenmap,
            0.6,
        );
        hard.score_lift = Some(0.8);

        let mut m = DistanceWeightedMetaModel::new();
        m.fit(&[easy, hard]);
        let predicted = m.predict(&shared_feat);
        assert_eq!(predicted.projection_kind, ProjectionKind::LaplacianEigenmap);
    }

    #[test]
    fn dw_all_records_without_lift_fall_back_to_best_score() {
        // Legacy training sets predate score_lift entirely. Evidence
        // must fall back to best_score and still yield a prediction —
        // the higher-scoring record wins at equal distance.
        let shared_feat = feat(500, 5, 0.1, 0.5);
        let lo = record("lo", shared_feat.clone(), ProjectionKind::Pca, 0.2);
        let hi = record(
            "hi",
            shared_feat.clone(),
            ProjectionKind::LaplacianEigenmap,
            0.9,
        );
        assert!(lo.score_lift.is_none() && hi.score_lift.is_none());

        let mut m = DistanceWeightedMetaModel::new();
        m.fit(&[lo, hi]);
        let ranked = m.score_candidates(&shared_feat);
        assert_eq!(ranked.len(), 2, "no record filtered as non-finite");
        let predicted = m.predict(&shared_feat);
        assert_eq!(predicted.projection_kind, ProjectionKind::LaplacianEigenmap);
    }

    #[test]
    fn dw_prefers_closer_when_similar_score() {
        // Two records with similar best_scores but very different
        // features — the closer one to the query should win.
        let close = record(
            "close",
            feat(500, 5, 0.06, 0.82),
            ProjectionKind::LaplacianEigenmap,
            0.70,
        );
        let far = record(
            "far",
            feat(500, 5, 0.55, 0.15),
            ProjectionKind::Pca,
            0.72, // only slightly better
        );
        let mut m = DistanceWeightedMetaModel::new();
        m.fit(&[close, far]);
        let q = feat(500, 5, 0.05, 0.80); // very close to "close"'s features
        assert_eq!(
            m.predict(&q).projection_kind,
            ProjectionKind::LaplacianEigenmap,
        );
    }

    #[test]
    fn dw_score_candidates_sorted_descending() {
        let ra = record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.6);
        let rb = record("b", feat(500, 5, 0.50, 0.2), ProjectionKind::Pca, 0.9);
        let mut m = DistanceWeightedMetaModel::new();
        m.fit(&[ra, rb]);
        let ranked = m.score_candidates(&feat(500, 5, 0.07, 0.78));
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].1 >= ranked[1].1);
    }

    #[test]
    fn dw_is_deterministic() {
        let records = vec![
            record("a", feat(500, 5, 0.05, 0.8), ProjectionKind::Pca, 0.7),
            record(
                "b",
                feat(500, 5, 0.50, 0.2),
                ProjectionKind::LaplacianEigenmap,
                0.6,
            ),
        ];
        let mut m1 = DistanceWeightedMetaModel::new();
        m1.fit(&records);
        let mut m2 = DistanceWeightedMetaModel::new();
        m2.fit(&records);
        let q = feat(500, 5, 0.10, 0.7);
        assert_eq!(
            m1.predict(&q).projection_kind,
            m2.predict(&q).projection_kind
        );
    }

    #[test]
    fn dw_epsilon_clamps_non_positive() {
        let m = DistanceWeightedMetaModel::new().with_epsilon(-1.0);
        // Internal epsilon shouldn't be negative; we can probe via
        // score_candidates: at d=0 the weight is r.best_score/epsilon;
        // with a non-positive epsilon we'd otherwise divide by zero.
        let r = record("r", feat(100, 5, 0.1, 0.3), ProjectionKind::Pca, 0.5);
        let mut m = m;
        m.fit(std::slice::from_ref(&r));
        let ranked = m.score_candidates(&r.features);
        assert!(ranked[0].1.is_finite());
    }

    #[test]
    #[should_panic(expected = "called before fit")]
    fn dw_predict_before_fit_panics() {
        let m = DistanceWeightedMetaModel::new();
        let _ = m.predict(&feat(100, 5, 0.1, 0.3));
    }

    #[test]
    fn dw_name_stable() {
        let m = DistanceWeightedMetaModel::new();
        assert_eq!(m.name(), "distance_weighted");
    }

    #[test]
    fn adjust_score_with_feedback_blends_at_alpha() {
        let r = record("r", feat(100, 5, 0.1, 0.3), ProjectionKind::Pca, 0.8);
        let summary = FeedbackSummary {
            corpus_id: "r".into(),
            n_events: 10,
            mean_score: 0.4,
            min_score: 0.1,
            max_score: 0.9,
        };
        // alpha = 0 → keep best_score
        assert!((r.adjust_score_with_feedback(&summary, 0.0) - 0.8).abs() < 1e-12);
        // alpha = 1 → replace with feedback
        assert!((r.adjust_score_with_feedback(&summary, 1.0) - 0.4).abs() < 1e-12);
        // alpha = 0.5 → midpoint 0.6
        assert!((r.adjust_score_with_feedback(&summary, 0.5) - 0.6).abs() < 1e-12);
        // alpha clamped: values outside [0,1] are clipped.
        assert!((r.adjust_score_with_feedback(&summary, 2.0) - 0.4).abs() < 1e-12);
        assert!((r.adjust_score_with_feedback(&summary, -1.0) - 0.8).abs() < 1e-12);
    }
}
