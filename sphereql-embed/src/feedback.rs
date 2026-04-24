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
    /// (`~/.sphereql/feedback_events.json`).
    ///
    /// O(1) per call: opens the file in append mode and writes one
    /// JSON-encoded line. Previous implementation loaded the full
    /// aggregator, pushed the event, and rewrote the file — O(N)
    /// per append, which mattered on a production firehose.
    /// Legacy array-format stores are migrated to JSONL on the first
    /// append (one-time O(N) cost) and append at O(1) afterward.
    ///
    /// Mirrors [`MetaTrainingRecord::append_to_default_store`](crate::meta_model::MetaTrainingRecord::append_to_default_store)
    /// — both are instance methods on the data they persist.
    pub fn append_to_default_store(&self) -> io::Result<PathBuf> {
        let path = FeedbackAggregator::default_store_path()?;
        self.append_to(&path)?;
        Ok(path)
    }

    /// Append this event to an arbitrary JSONL file. Creates the file
    /// and any missing parent directories on first call.
    pub fn append_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        use std::io::Write;

        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        // One-time migration: if the store is a legacy JSON array
        // (what FeedbackAggregator::save used to write), rewrite it
        // as JSONL so subsequent appends stay O(1).
        if path.exists() {
            let head = fs::read_to_string(path)?;
            if head.trim_start().starts_with('[') {
                let events: Vec<Self> = serde_json::from_str(head.trim_start())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let mut migrated = String::with_capacity(head.len());
                for e in &events {
                    serde_json::to_string(e)
                        .map(|line| {
                            migrated.push_str(&line);
                            migrated.push('\n');
                        })
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                }
                fs::write(path, migrated)?;
            }
        }

        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let line = serde_json::to_string(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        writeln!(f, "{line}")
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
    // Optional ring-buffer semantics. When `max_events` is set to some
    // `N`, `record` drops the oldest event whenever the log would
    // exceed `N`. Serialized transparently as a flat array; the cap
    // itself is a runtime-only knob and is not persisted.
    #[serde(skip)]
    max_events: Option<usize>,
    events: Vec<FeedbackEvent>,
}

impl FeedbackAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a bounded event window. Once the log reaches
    /// `max_events`, every new `record` call drops the oldest event so
    /// memory stays capped — appropriate for long-running services
    /// that only need recent feedback for per-corpus summaries.
    ///
    /// Without this cap the event log grows indefinitely; on a 1
    /// event/sec firehose that reaches 100 MB of JSON within a week.
    pub fn with_window(max_events: usize) -> Self {
        Self {
            max_events: Some(max_events),
            events: Vec::with_capacity(max_events.min(1024)),
        }
    }

    /// Attach (or drop) the event-count cap after construction.
    pub fn set_max_events(&mut self, max_events: Option<usize>) {
        self.max_events = max_events;
        if let Some(cap) = max_events
            && self.events.len() > cap
        {
            let excess = self.events.len() - cap;
            self.events.drain(0..excess);
        }
    }

    /// Total number of accumulated events (across all corpus_ids).
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Append one event. When a [`Self::with_window`] cap is set and
    /// the log is already at capacity, the oldest event is evicted
    /// first — a FIFO ring over the underlying `Vec`.
    pub fn record(&mut self, event: FeedbackEvent) {
        if let Some(cap) = self.max_events
            && self.events.len() >= cap
        {
            // `remove(0)` is O(n) but `record` on a capped aggregator
            // is paired with O(n) event summarization anyway, and the
            // cap is expected to be in the hundreds, not millions.
            let excess = self.events.len() + 1 - cap;
            self.events.drain(0..excess);
        }
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
    ///
    /// Accepts both a JSON array (legacy, what `save` writes for
    /// backward compat) and JSON Lines (new format written by
    /// [`FeedbackEvent::append_to_default_store`] and sibling append
    /// paths). Detection is first-character based.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }
        let raw = fs::read_to_string(path)?;
        let trimmed = raw.trim_start();
        if trimmed.is_empty() {
            return Ok(Self::new());
        }
        let events: Vec<FeedbackEvent> = if trimmed.starts_with('[') {
            serde_json::from_str(trimmed)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        } else {
            trimmed
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    serde_json::from_str::<FeedbackEvent>(l)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                })
                .collect::<io::Result<Vec<_>>>()?
        };
        Ok(Self {
            max_events: None,
            events,
        })
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
    fn with_window_evicts_oldest() {
        let mut a = FeedbackAggregator::with_window(3);
        for i in 0..5 {
            a.record(ev("c", &format!("q{i}"), i as f64 / 10.0));
        }
        // Only the last 3 events survive (q2, q3, q4).
        assert_eq!(a.len(), 3);
        let ids: Vec<&str> = a.events().iter().map(|e| e.query_id.as_str()).collect();
        assert_eq!(ids, vec!["q2", "q3", "q4"]);
    }

    #[test]
    fn set_max_events_trims_existing_log() {
        let mut a = FeedbackAggregator::new();
        for i in 0..10 {
            a.record(ev("c", &format!("q{i}"), 0.5));
        }
        a.set_max_events(Some(4));
        assert_eq!(a.len(), 4);
        let ids: Vec<&str> = a.events().iter().map(|e| e.query_id.as_str()).collect();
        assert_eq!(ids, vec!["q6", "q7", "q8", "q9"]);
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
    fn append_to_creates_jsonl_then_load_roundtrips() {
        let dir = std::env::temp_dir().join(format!("sphereql_fb_jsonl_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("events.json");

        // Append three events one by one — each call is O(1).
        ev("c1", "q1", 0.8).append_to(&path).unwrap();
        ev("c1", "q2", 0.6).append_to(&path).unwrap();
        ev("c2", "q3", 0.4).append_to(&path).unwrap();

        let loaded = FeedbackAggregator::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.events()[0].query_id, "q1");
        assert_eq!(loaded.events()[1].query_id, "q2");
        assert_eq!(loaded.events()[2].query_id, "q3");

        // Raw file really is JSONL (one record per line).
        let raw = fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().count(), 3);
        assert!(!raw.trim_start().starts_with('['));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn append_to_migrates_legacy_array_file() {
        let dir = std::env::temp_dir().join(format!("sphereql_fb_migrate_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("events.json");

        // Seed with a legacy array file (what `save` used to write).
        let mut legacy = FeedbackAggregator::new();
        legacy.record(ev("c1", "q1", 0.9));
        legacy.record(ev("c2", "q2", 0.5));
        legacy.save(&path).unwrap();

        // First append migrates the file to JSONL.
        ev("c3", "q3", 0.7).append_to(&path).unwrap();

        let loaded = FeedbackAggregator::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.events()[0].query_id, "q1");
        assert_eq!(loaded.events()[2].query_id, "q3");

        // Post-migration shape is JSONL.
        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.trim_start().starts_with('['));
        assert_eq!(raw.lines().count(), 3);

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
