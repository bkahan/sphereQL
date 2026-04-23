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
//! over [`CorpusFeatures::to_vec`]. It works with any `N ≥ 1` training
//! records, is deterministic, and has no free hyperparameters. When
//! you've accumulated ≥ 10 diverse corpora you can swap in something
//! fancier (gradient-boosted trees, small MLP) against the same
//! [`MetaModel`] trait — the storage format
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

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::PipelineConfig;
use crate::corpus_features::{CORPUS_FEATURE_COUNT, CorpusFeatures};
use crate::feedback::FeedbackSummary;
use crate::tuner::TuneReport;
use crate::util::{default_timestamp, sphereql_home_dir};

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
    /// Which quality metric was being optimized. Records with different
    /// metrics aren't directly comparable and shouldn't be mixed when
    /// fitting a model unless that metric is a tuner input too.
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

    /// Load a list of records from a JSON array file. Returns an empty
    /// vec if the file does not exist.
    pub fn load_list(path: impl AsRef<Path>) -> io::Result<Vec<Self>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw = fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Default on-disk training-store path: `~/.sphereql/meta_records.json`.
    pub fn default_store_path() -> io::Result<PathBuf> {
        Ok(sphereql_home_dir()?.join("meta_records.json"))
    }

    /// Append this record to the user's default training store.
    /// Creates parent dirs and the file itself if they don't exist yet,
    /// so repeat calls naturally accumulate a dataset across tuner runs.
    pub fn append_to_default_store(&self) -> io::Result<PathBuf> {
        let path = Self::default_store_path()?;
        let mut records = Self::load_list(&path)?;
        records.push(self.clone());
        Self::save_list(&records, &path)?;
        Ok(path)
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
    pub fn adjust_score_with_feedback(&self, summary: &FeedbackSummary, alpha: f64) -> f64 {
        let a = alpha.clamp(0.0, 1.0);
        (1.0 - a) * self.best_score + a * summary.mean_score
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Per-feature mean + std computed across a training set, used for
/// z-score normalization by both meta-model implementations.
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
    let vecs: Vec<[f64; CORPUS_FEATURE_COUNT]> =
        records.iter().map(|r| r.features.to_vec()).collect();

    for i in 0..CORPUS_FEATURE_COUNT {
        let mean: f64 = vecs.iter().map(|v| v[i]).sum::<f64>() / n as f64;
        means[i] = mean;
        let var: f64 = vecs.iter().map(|v| (v[i] - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt();
        stds[i] = if sd > f64::EPSILON { sd } else { 0.0 };
    }
    (means, stds)
}

/// Z-score normalize a raw feature vector against precomputed
/// `means`/`stds`. Features whose stored std is below `f64::EPSILON`
/// (zero-variance in the training set) map to `0.0` rather than
/// dividing by a near-zero number.
fn normalize_features(
    raw: &[f64; CORPUS_FEATURE_COUNT],
    means: &[f64; CORPUS_FEATURE_COUNT],
    stds: &[f64; CORPUS_FEATURE_COUNT],
) -> [f64; CORPUS_FEATURE_COUNT] {
    let mut out = [0.0; CORPUS_FEATURE_COUNT];
    for i in 0..CORPUS_FEATURE_COUNT {
        let sd = stds[i];
        out[i] = if sd > f64::EPSILON {
            (raw[i] - means[i]) / sd
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

// ── Trait ──────────────────────────────────────────────────────────────

/// Predicts a [`PipelineConfig`] from a [`CorpusFeatures`] profile.
///
/// Implementers fit on a training set of [`MetaTrainingRecord`]s (pairs
/// of (features, best_config) observed from past tuner runs) and predict
/// a config for a new corpus.
pub trait MetaModel {
    /// Fit on a training set. Replacing any prior state.
    fn fit(&mut self, records: &[MetaTrainingRecord]);

    /// Predict the config that should work best on a corpus with the
    /// given profile. Panics if `fit` has not been called with at least
    /// one record — callers should treat `MetaModel` as a trained object
    /// and front-load `fit`.
    fn predict(&self, features: &CorpusFeatures) -> PipelineConfig;

    /// Short name for logs and model comparison.
    fn name(&self) -> &str;
}

// ── Nearest-neighbor baseline ─────────────────────────────────────────

/// The simplest useful meta-model: given a new corpus, return the
/// best_config of the training record whose corpus-feature vector is
/// closest in z-score-normalized Euclidean distance.
///
/// - Works with `N ≥ 1` records.
/// - Deterministic; no hyperparameters.
/// - Degenerate features (zero variance across training records) are
///   dropped from the distance computation at fit time so they don't
///   divide by zero or dominate via raw-scale inflation.
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
    /// model considers the nearest-neighbor candidate pool.
    pub fn records(&self) -> &[MetaTrainingRecord] {
        &self.records
    }

    /// Distance from a given feature vector to every stored record,
    /// sorted ascending. Returned as `(record_index, distance)` pairs.
    pub fn rank_candidates(&self, features: &CorpusFeatures) -> Vec<(usize, f64)> {
        let q = normalize_features(&features.to_vec(), &self.feature_means, &self.feature_stds);
        let mut ranked: Vec<(usize, f64)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let v = normalize_features(
                    &r.features.to_vec(),
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
}

impl MetaModel for NearestNeighborMetaModel {
    fn fit(&mut self, records: &[MetaTrainingRecord]) {
        self.records = records.to_vec();
        let (means, stds) = compute_feature_stats(&self.records);
        self.feature_means = means;
        self.feature_stds = if self.records.is_empty() {
            [1.0; CORPUS_FEATURE_COUNT]
        } else {
            stds
        };
    }

    fn predict(&self, features: &CorpusFeatures) -> PipelineConfig {
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

/// Picks the training record that maximizes `best_score × w(distance)`,
/// where `w(d) = 1 / (d + epsilon)` over z-score-normalized Euclidean
/// distance.
///
/// The distinction from [`NearestNeighborMetaModel`]: NN picks the
/// closest record regardless of how well that record performed, so a
/// single poorly-tuned outlier can pull predictions off. Distance-weighted
/// folds the record's score into the selection — a record is "good" if
/// it's both similar to the query AND had a high score. At N = 1 this
/// degenerates to NN (same record either way). At N ≥ 3 the two models
/// start diverging in predictable, useful ways.
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
        let q = normalize_features(&features.to_vec(), &self.feature_means, &self.feature_stds);
        let mut out: Vec<(usize, f64, f64)> = self
            .records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                // Filter non-finite `best_score` at score time. NaN would
                // otherwise propagate into `weighted`, hit the `total_cmp`
                // below as "greatest" (NaN sorts to the end of a total
                // order, but the *top* under a "descending" sort would
                // put NaN first), and silently become the prediction.
                if !r.best_score.is_finite() {
                    return None;
                }
                let v = normalize_features(
                    &r.features.to_vec(),
                    &self.feature_means,
                    &self.feature_stds,
                );
                let d = normalized_euclidean(&q, &v);
                let weighted = r.best_score / (d + self.epsilon);
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
        self.records = records.to_vec();
        let (means, stds) = compute_feature_stats(&self.records);
        self.feature_means = means;
        self.feature_stds = if self.records.is_empty() {
            [1.0; CORPUS_FEATURE_COUNT]
        } else {
            stds
        };
    }

    fn predict(&self, features: &CorpusFeatures) -> PipelineConfig {
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

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProjectionKind;

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
            metric_name: "test_metric".to_string(),
            strategy: "test_strategy".to_string(),
            timestamp: "2026-04-22T00:00:00Z".to_string(),
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
    #[should_panic(expected = "called before fit")]
    fn nn_predict_before_fit_panics() {
        let m = NearestNeighborMetaModel::new();
        let _ = m.predict(&feat(100, 5, 0.1, 0.3));
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
    fn with_timestamp_overrides_default() {
        let report = TuneReport {
            metric_name: "m".to_string(),
            best_score: 0.5,
            best_config: PipelineConfig::default(),
            trials: Vec::new(),
            failures: Vec::new(),
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
