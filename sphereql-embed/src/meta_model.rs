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
use std::path::Path;

use crate::config::PipelineConfig;
use crate::corpus_features::{CorpusFeatures, CORPUS_FEATURE_COUNT};

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
    /// Save a list of records as a JSON array to disk.
    pub fn save_list(records: &[Self], path: impl AsRef<Path>) -> io::Result<()> {
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

    fn normalized(&self, raw: &[f64; CORPUS_FEATURE_COUNT]) -> [f64; CORPUS_FEATURE_COUNT] {
        let mut out = [0.0; CORPUS_FEATURE_COUNT];
        for i in 0..CORPUS_FEATURE_COUNT {
            let sd = self.feature_stds[i];
            out[i] = if sd > f64::EPSILON {
                (raw[i] - self.feature_means[i]) / sd
            } else {
                0.0
            };
        }
        out
    }

    /// Distance from a given feature vector to every stored record,
    /// sorted ascending. Returned as `(record_index, distance)` pairs.
    pub fn rank_candidates(&self, features: &CorpusFeatures) -> Vec<(usize, f64)> {
        let q = self.normalized(&features.to_vec());
        let mut ranked: Vec<(usize, f64)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let v = self.normalized(&r.features.to_vec());
                let d = q
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (i, d)
            })
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

impl MetaModel for NearestNeighborMetaModel {
    fn fit(&mut self, records: &[MetaTrainingRecord]) {
        self.records = records.to_vec();
        let n = self.records.len();
        if n == 0 {
            self.feature_means = [0.0; CORPUS_FEATURE_COUNT];
            self.feature_stds = [1.0; CORPUS_FEATURE_COUNT];
            return;
        }

        // Compute per-feature mean and std for z-score normalization.
        for i in 0..CORPUS_FEATURE_COUNT {
            let mean: f64 = self
                .records
                .iter()
                .map(|r| r.features.to_vec()[i])
                .sum::<f64>()
                / n as f64;
            self.feature_means[i] = mean;

            let var: f64 = self
                .records
                .iter()
                .map(|r| (r.features.to_vec()[i] - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let sd = var.sqrt();
            // Keep sd of 0 or near-zero untouched so normalized() can
            // zero out the degenerate feature rather than explode it.
            self.feature_stds[i] = if sd > f64::EPSILON { sd } else { 0.0 };
        }
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

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProjectionKind;

    fn feat(
        n: usize,
        c: usize,
        sparsity: f64,
        intra: f64,
    ) -> CorpusFeatures {
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
        let mut cfg = PipelineConfig::default();
        cfg.projection_kind = kind;
        MetaTrainingRecord {
            corpus_id: id.to_string(),
            features: f,
            best_config: cfg,
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
        m.fit(&[r.clone()]);
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
        let r_b = record(
            "dense",
            feat(500, 5, 0.50, 0.2),
            ProjectionKind::Pca,
            0.6,
        );
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
}
