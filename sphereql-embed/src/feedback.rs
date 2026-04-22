//! User-supplied feedback signals that refine stored meta-learning records.
//!
//! Automated quality metrics ([`QualityMetric`](crate::quality_metric::QualityMetric))
//! only see the geometry of the built pipeline. They can't tell whether
//! actual users found query results useful. This module defines a minimal
//! feedback primitive — one scalar signal per query — plus an aggregator
//! that summarizes signals per `corpus_id` so a
//! [`MetaTrainingRecord`](crate::meta_model::MetaTrainingRecord)'s
//! `best_score` can be blended with observed user satisfaction.
//!
//! Intended flow (L3 of the metalearning ladder):
//!
//! 1. Deploy a tuned pipeline to users.
//! 2. On each query result, collect a satisfaction signal (thumbs, rating,
//!    click-through, …). Map it to `[0, 1]` and emit a [`FeedbackEvent`].
//! 3. Aggregate events into a [`FeedbackAggregator`], persisted under
//!    [`FeedbackAggregator::default_store_path`].
//! 4. When selecting a stored record for a new corpus, blend the record's
//!    automated `best_score` with the corpus's feedback summary via
//!    [`MetaTrainingRecord::adjust_score_with_feedback`](crate::meta_model::MetaTrainingRecord::adjust_score_with_feedback).
//!
//! The meta-model is deliberately *not* retrained here — that's a v2
//! concern. This module supplies the primitives; composition is up to
//! the caller.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::util::{default_timestamp, sphereql_home_dir};

/// One user-supplied satisfaction signal attached to a specific query.
///
/// `score` is a normalized scalar in `[0, 1]`:
/// - `1.0` = perfect, user got exactly what they wanted.
/// - `0.5` = neutral / ambiguous.
/// - `0.0` = wrong, unhelpful, or actively misleading.
///
/// Upstream mapping (stars to `[0, 1]`, CTR to `[0, 1]`, etc.) is the
/// caller's responsibility — the aggregator just computes statistics on
/// whatever scalar you supply.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeedbackEvent {
    /// Must match the `corpus_id` of the [`MetaTrainingRecord`](crate::meta_model::MetaTrainingRecord)
    /// the pipeline was built from.
    pub corpus_id: String,
    /// Caller-supplied query identifier. Free-form; used for deduping
    /// and auditing. An empty string is allowed.
    pub query_id: String,
    /// Satisfaction signal in `[0, 1]`. Clamped at read time by
    /// [`FeedbackAggregator::summarize`]; store raw values if you want.
    pub score: f64,
    /// Free-form timestamp string. Seconds-since-epoch by default from
    /// [`FeedbackEvent::now`]; swap in your own format as needed.
    pub timestamp: String,
}

impl FeedbackEvent {
    /// Construct with a default timestamp (epoch seconds).
    pub fn now(corpus_id: impl Into<String>, query_id: impl Into<String>, score: f64) -> Self {
        Self {
            corpus_id: corpus_id.into(),
            query_id: query_id.into(),
            score,
            timestamp: default_timestamp(),
        }
    }

    /// Append this event to the user's default feedback store
    /// (`~/.sphereql/feedback_events.json`). Loads the existing log,
    /// appends, and rewrites — idempotent across sessions. Returns the
    /// resolved store path.
    ///
    /// Mirrors [`MetaTrainingRecord::append_to_default_store`](crate::meta_model::MetaTrainingRecord::append_to_default_store)
    /// — both are instance methods on the data they persist, not
    /// statics on their aggregator, so the two APIs feel identical
    /// in downstream code.
    pub fn append_to_default_store(&self) -> io::Result<PathBuf> {
        let path = FeedbackAggregator::default_store_path()?;
        let mut agg = FeedbackAggregator::load(&path)?;
        agg.record(self.clone());
        agg.save(&path)?;
        Ok(path)
    }
}

/// Summary statistics for the feedback observed on a single corpus.
///
/// All scalar fields are computed over the subset of events whose
/// `corpus_id` matches the summarized corpus. `mean_score` is
/// clamp-averaged to `[0, 1]` so downstream blending stays bounded even
/// when raw event scores are dirty.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeedbackSummary {
    pub corpus_id: String,
    pub n_events: usize,
    pub mean_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

// ── Aggregator ─────────────────────────────────────────────────────────

/// Accumulates [`FeedbackEvent`]s across sessions and summarizes them by
/// `corpus_id`.
///
/// Serializable as a flat JSON array of events — same pattern as
/// [`MetaTrainingRecord::save_list`](crate::meta_model::MetaTrainingRecord::save_list).
/// Append is O(1) amortized; [`Self::summarize`] is O(N) per call, which
/// is fine for the scale feedback naturally reaches (hundreds to
/// thousands of events per corpus).
///
/// `#[serde(transparent)]` keeps the derive-based serializer
/// (`serde_json::to_string(&agg)`) and the hand-rolled
/// [`Self::save`] / [`Self::load`] path on the same JSON shape — a flat
/// array of events. Without it, the derive would emit `{"events": [...]}`
/// which `load` rejects.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct FeedbackAggregator {
    events: Vec<FeedbackEvent>,
}

impl FeedbackAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of accumulated events (across all corpus_ids).
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Append one event.
    pub fn record(&mut self, event: FeedbackEvent) {
        self.events.push(event);
    }

    /// Read-only borrow of the raw event log.
    pub fn events(&self) -> &[FeedbackEvent] {
        &self.events
    }

    /// Distinct `corpus_id`s that have any feedback attached.
    pub fn corpus_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .events
            .iter()
            .map(|e| e.corpus_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort();
        ids
    }

    /// Summarize feedback for a specific corpus. Returns `None` if the
    /// corpus has no events yet.
    pub fn summarize(&self, corpus_id: &str) -> Option<FeedbackSummary> {
        let mut count = 0usize;
        let mut sum = 0.0f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for e in &self.events {
            if e.corpus_id != corpus_id {
                continue;
            }
            let s = e.score.clamp(0.0, 1.0);
            count += 1;
            sum += s;
            if s < min {
                min = s;
            }
            if s > max {
                max = s;
            }
        }
        if count == 0 {
            return None;
        }
        Some(FeedbackSummary {
            corpus_id: corpus_id.to_string(),
            n_events: count,
            mean_score: sum / count as f64,
            min_score: min,
            max_score: max,
        })
    }

    /// Summarize every corpus that has events. Sorted by `corpus_id`.
    pub fn summarize_all(&self) -> Vec<FeedbackSummary> {
        let mut per_corpus: HashMap<String, (usize, f64, f64, f64)> = HashMap::new();
        for e in &self.events {
            let s = e.score.clamp(0.0, 1.0);
            let entry = per_corpus.entry(e.corpus_id.clone()).or_insert((
                0,
                0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ));
            entry.0 += 1;
            entry.1 += s;
            if s < entry.2 {
                entry.2 = s;
            }
            if s > entry.3 {
                entry.3 = s;
            }
        }
        let mut out: Vec<FeedbackSummary> = per_corpus
            .into_iter()
            .map(|(corpus_id, (count, sum, min, max))| FeedbackSummary {
                corpus_id,
                n_events: count,
                mean_score: sum / count as f64,
                min_score: min,
                max_score: max,
            })
            .collect();
        out.sort_by(|a, b| a.corpus_id.cmp(&b.corpus_id));
        out
    }

    /// Save this aggregator (event list) to a JSON file. Creates parent
    /// directories as needed.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.events)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load an aggregator from a JSON event-log file. Returns an empty
    /// aggregator if the file does not exist.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }
        let raw = fs::read_to_string(path)?;
        let events: Vec<FeedbackEvent> = serde_json::from_str(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self { events })
    }

    /// Default on-disk feedback log: `~/.sphereql/feedback_events.json`.
    /// Parallel convention to
    /// [`MetaTrainingRecord::default_store_path`](crate::meta_model::MetaTrainingRecord::default_store_path).
    pub fn default_store_path() -> io::Result<PathBuf> {
        Ok(sphereql_home_dir()?.join("feedback_events.json"))
    }

    /// Load the default on-disk feedback store. Empty aggregator if the
    /// store does not exist yet.
    pub fn load_default_store() -> io::Result<Self> {
        Self::load(Self::default_store_path()?)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(corpus: &str, query: &str, score: f64) -> FeedbackEvent {
        FeedbackEvent {
            corpus_id: corpus.into(),
            query_id: query.into(),
            score,
            timestamp: "0".into(),
        }
    }

    #[test]
    fn empty_aggregator_has_no_summary() {
        let a = FeedbackAggregator::new();
        assert!(a.is_empty());
        assert!(a.summarize("anything").is_none());
        assert!(a.summarize_all().is_empty());
        assert!(a.corpus_ids().is_empty());
    }

    #[test]
    fn summarize_single_corpus() {
        let mut a = FeedbackAggregator::new();
        a.record(ev("c1", "q1", 0.8));
        a.record(ev("c1", "q2", 0.6));
        a.record(ev("c1", "q3", 1.0));
        let s = a.summarize("c1").unwrap();
        assert_eq!(s.n_events, 3);
        assert!((s.mean_score - 0.8).abs() < 1e-12);
        assert!((s.min_score - 0.6).abs() < 1e-12);
        assert!((s.max_score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn summarize_clamps_out_of_range_scores() {
        let mut a = FeedbackAggregator::new();
        a.record(ev("c", "q1", -0.5));
        a.record(ev("c", "q2", 1.5));
        let s = a.summarize("c").unwrap();
        // -0.5 → 0, 1.5 → 1 → mean = 0.5
        assert!((s.mean_score - 0.5).abs() < 1e-12);
        assert_eq!(s.min_score, 0.0);
        assert_eq!(s.max_score, 1.0);
    }

    #[test]
    fn summarize_isolates_corpus_ids() {
        let mut a = FeedbackAggregator::new();
        a.record(ev("alpha", "q", 0.2));
        a.record(ev("beta", "q", 0.9));
        assert!((a.summarize("alpha").unwrap().mean_score - 0.2).abs() < 1e-12);
        assert!((a.summarize("beta").unwrap().mean_score - 0.9).abs() < 1e-12);
        assert!(a.summarize("gamma").is_none());
    }

    #[test]
    fn summarize_all_is_sorted_by_corpus_id() {
        let mut a = FeedbackAggregator::new();
        a.record(ev("zebra", "q", 0.5));
        a.record(ev("ant", "q", 0.5));
        a.record(ev("mule", "q", 0.5));
        let sums = a.summarize_all();
        assert_eq!(sums.len(), 3);
        assert_eq!(sums[0].corpus_id, "ant");
        assert_eq!(sums[1].corpus_id, "mule");
        assert_eq!(sums[2].corpus_id, "zebra");
    }

    #[test]
    fn corpus_ids_distinct_and_sorted() {
        let mut a = FeedbackAggregator::new();
        a.record(ev("b", "q", 0.5));
        a.record(ev("a", "q", 0.5));
        a.record(ev("b", "q2", 0.5));
        let ids = a.corpus_ids();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn now_constructor_produces_parseable_timestamp() {
        let e = FeedbackEvent::now("c", "q", 0.5);
        assert_eq!(e.corpus_id, "c");
        assert!(e.timestamp.parse::<u64>().is_ok());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("sphereql_fb_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("events.json");

        let mut a = FeedbackAggregator::new();
        a.record(ev("c1", "q", 0.7));
        a.record(ev("c2", "q", 0.3));
        a.save(&path).unwrap();

        let loaded = FeedbackAggregator::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.events()[0].corpus_id, "c1");
        assert_eq!(loaded.events()[1].corpus_id, "c2");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let path = std::env::temp_dir().join("sphereql_fb_nonexistent_xyz.json");
        let a = FeedbackAggregator::load(&path).unwrap();
        assert!(a.is_empty());
    }

    #[test]
    fn default_store_path_ends_with_expected_filename() {
        let p = FeedbackAggregator::default_store_path().unwrap();
        assert!(p.ends_with("feedback_events.json"));
        assert!(p.iter().any(|c| c.to_string_lossy() == ".sphereql"));
    }
}
