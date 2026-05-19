//! Loader for the extended corpus.
//!
//! Phase 3: the primary read path is Parquet
//! ([`crate::parquet_loader`]). JSON is retained as a fallback for
//! environments where the Parquet artifact is missing — gated by the
//! default `json-fallback` feature.
//!
//! The eager [`load_extended_corpus`] returns `Vec<Concept>`.
//! [`stream_extended_corpus`] returns a row-by-row iterator suitable
//! for the 500K+ scale targets.
//!
//! ## Phase 2 backwards-compatibility (JSON fallback only)
//!
//! Old JSON (pre-Phase 2) had only `label`, `category`, `features`.
//! New JSON adds five signal fields. Missing fields fall back to
//! [`Concept`]'s `NEUTRAL_*` constants via `#[serde(default = "...")]`,
//! so legacy corpora load unchanged.
//!
//! ## Memory warning
//!
//! Labels and category names are leaked to `&'static str` via
//! `Box::leak` so they fit [`Concept`]'s signature. Call at most once
//! per process. The ownership migration is tracked for a future phase.

use std::path::PathBuf;

use crate::concept::Concept;
use crate::parquet_loader::{self, ParquetLoadError};

#[cfg(feature = "json-fallback")]
use serde::Deserialize;

#[cfg(feature = "json-fallback")]
const EXTENDED_JSON: &str = include_str!("../data/extended_corpus.json");

#[cfg(feature = "json-fallback")]
#[derive(Deserialize)]
struct RawConcept {
    label: String,
    category: String,
    features: Vec<[f64; 2]>,
    #[serde(default = "default_quality")]
    quality: f64,
    #[serde(default = "default_axis_coherence")]
    axis_coherence: f64,
    #[serde(default = "default_bridge_degree")]
    bridge_degree: u8,
    #[serde(default = "default_source_confidence")]
    source_confidence: f64,
    #[serde(default = "default_home_affinity")]
    home_affinity: f64,
    #[serde(default)]
    #[allow(dead_code)]
    source: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    openalex_id: Option<String>,
}

#[cfg(feature = "json-fallback")]
fn default_quality() -> f64 {
    Concept::NEUTRAL_QUALITY
}
#[cfg(feature = "json-fallback")]
fn default_axis_coherence() -> f64 {
    Concept::NEUTRAL_AXIS_COHERENCE
}
#[cfg(feature = "json-fallback")]
fn default_bridge_degree() -> u8 {
    Concept::NEUTRAL_BRIDGE_DEGREE
}
#[cfg(feature = "json-fallback")]
fn default_source_confidence() -> f64 {
    Concept::NEUTRAL_SOURCE_CONFIDENCE
}
#[cfg(feature = "json-fallback")]
fn default_home_affinity() -> f64 {
    Concept::NEUTRAL_HOME_AFFINITY
}

#[cfg(feature = "json-fallback")]
#[derive(Deserialize)]
struct RawCorpus {
    concepts: Vec<RawConcept>,
}

fn extended_parquet_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/extended_corpus.parquet")
}

/// Load the extended corpus.
///
/// Reads `data/extended_corpus.parquet` first; on `NotFound` falls back
/// to the embedded JSON when the `json-fallback` feature is enabled
/// (default). Any other Parquet error is fatal.
///
/// # Panics
///
/// Panics if neither artifact can be loaded. This is a logic bug — the
/// data files are built by `tools/generate_extended.py` and validated
/// before commit; a failure here means the build pipeline is broken.
pub fn load_extended_corpus() -> Vec<Concept> {
    let path = extended_parquet_path();
    match parquet_loader::load_concepts(&path) {
        Ok(concepts) => concepts,
        Err(ParquetLoadError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            #[cfg(feature = "json-fallback")]
            {
                load_from_json()
            }
            #[cfg(not(feature = "json-fallback"))]
            panic!(
                "extended corpus Parquet not found at {} and json-fallback feature is disabled",
                path.display()
            );
        }
        Err(e) => panic!(
            "failed to load extended corpus from {}: {e}",
            path.display()
        ),
    }
}

/// Streaming loader for callers that can iterate row-by-row.
///
/// Yields concepts in row-group order without materializing the full
/// corpus. Suitable for the 500K+ scale target where holding the full
/// `Vec<Concept>` would be expensive.
///
/// Returns the Parquet error directly (including `NotFound`) — there is
/// no JSON streaming fallback since the JSON loader is a one-shot DOM
/// parse with no scaling benefit.
pub fn stream_extended_corpus()
-> Result<Box<dyn Iterator<Item = Result<Concept, ParquetLoadError>> + Send>, ParquetLoadError> {
    parquet_loader::stream_concepts(extended_parquet_path())
}

#[cfg(feature = "json-fallback")]
pub(crate) fn load_from_json() -> Vec<Concept> {
    let raw: RawCorpus = serde_json::from_str(EXTENDED_JSON)
        .expect("extended_corpus.json is malformed — regenerate via tools/generate_extended.py");

    raw.concepts
        .into_iter()
        .map(|rc| Concept {
            label: leak_str(rc.label),
            category: leak_str(rc.category),
            features: rc
                .features
                .into_iter()
                .map(|pair| (pair[0] as usize, pair[1]))
                .collect(),
            quality: rc.quality,
            axis_coherence: rc.axis_coherence,
            bridge_degree: rc.bridge_degree,
            source_confidence: rc.source_confidence,
            home_affinity: rc.home_affinity,
        })
        .collect()
}

#[cfg(feature = "json-fallback")]
fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_extended_corpus_smoke() {
        let eager = load_extended_corpus().len();
        let streamed: usize = stream_extended_corpus()
            .expect("stream open")
            .filter_map(Result::ok)
            .count();
        assert_eq!(eager, streamed, "stream count must match eager count");
    }
}
