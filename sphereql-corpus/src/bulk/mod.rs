//! Streaming, scalable corpus ingest (Phase 7).
//!
//! The earlier corpus generators (Phase 1 Python, Phase 4 Wikidata
//! adapter) loaded the whole result set into memory and emitted a
//! single ~5 K-concept JSON. That worked at thousands of rows; it
//! falls over at 500 K, and is a non-starter at 5 M – 500 M.
//!
//! This module replaces that path with a streaming three-stage
//! pipeline:
//!
//! ```text
//!   ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
//!   │ BulkSource (iter)  │ -> │  AxisExtractor     │ -> │  ParquetSink       │
//!   │ - WikidataSparql   │    │  hashed claim axes │    │  10 K-row batches  │
//!   │ - OpenAlexShard    │    │  one-pass, no map  │    │  resume from ckpt  │
//!   │ - WikidataDump     │    │                    │    │                    │
//!   └────────────────────┘    └────────────────────┘    └────────────────────┘
//! ```
//!
//! Memory stays bounded — one batch (10 K rows) at a time — so the
//! same code path handles 5 K (local smoke), 500 K (default target),
//! and 500 M (run on a bigger box, swap source). Categories are not
//! committed to a fixed 31-cat whitelist; ingest writes the source's
//! best guess and the post-ingest `cluster_bulk.py` step overwrites
//! with k-means-emergent labels.

use std::fmt;

pub mod axis;
pub mod openalex_shard;
pub mod sink;
pub mod wikidata_sparql;
#[cfg(feature = "bulk-dump")]
pub mod wikidata_dump;

pub use axis::HashedClaimAxisExtractor;
pub use openalex_shard::{OpenAlexShardConfig, OpenAlexShardSource};
pub use sink::{ParquetSink, SinkCheckpoint};
pub use wikidata_sparql::{SparqlConfig, WikidataSparqlSource};
#[cfg(feature = "bulk-dump")]
pub use wikidata_dump::{WikidataDumpConfig, WikidataDumpSource};

/// One raw item streamed from a `BulkSource` — source-agnostic.
///
/// `claims` is the predicate→object structure both Wikidata and
/// OpenAlex naturally expose. For Wikidata: `(P31, Q5)`-style pairs.
/// For OpenAlex Works: `(topic, T11999)`, `(concept, C99165)`, etc.
/// The hashed axis extractor consumes this verbatim — no source-
/// specific code in the embedding stage.
#[derive(Debug, Clone)]
pub struct BulkItem {
    pub external_id: String,
    pub label: String,
    pub description: String,
    pub claims: Vec<Claim>,
    pub source_name: String,
    pub source_confidence: f64,
    pub category_hint: Option<String>,
    pub quality_hint: f64,
}

/// One predicate-object pair. `weight` is `1.0` for plain Wikidata
/// claims and the topic/concept `score` (∈ `[0,1]`) for OpenAlex.
#[derive(Debug, Clone)]
pub struct Claim {
    pub predicate: String,
    pub object: String,
    pub weight: f64,
}

impl Claim {
    pub fn new(predicate: impl Into<String>, object: impl Into<String>, weight: f64) -> Self {
        Self {
            predicate: predicate.into(),
            object: object.into(),
            weight,
        }
    }
}

/// All the failure modes a streaming source can surface to the
/// orchestrator. The binary turns these into a soft-error counter +
/// log line; one bad item shouldn't kill an 8-hour run.
#[derive(Debug)]
pub enum BulkSourceError {
    Io(std::io::Error),
    Decode(String),
    Network(String),
    Parse(String),
    EndOfStream,
}

impl fmt::Display for BulkSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BulkSourceError::Io(e) => write!(f, "io: {e}"),
            BulkSourceError::Decode(s) => write!(f, "decode: {s}"),
            BulkSourceError::Network(s) => write!(f, "network: {s}"),
            BulkSourceError::Parse(s) => write!(f, "parse: {s}"),
            BulkSourceError::EndOfStream => write!(f, "end of stream"),
        }
    }
}

impl std::error::Error for BulkSourceError {}

impl From<std::io::Error> for BulkSourceError {
    fn from(e: std::io::Error) -> Self {
        BulkSourceError::Io(e)
    }
}

/// Pluggable corpus source. Implementors are `Iterator`s yielding
/// `Result<BulkItem, BulkSourceError>` — one bad row doesn't stop the
/// stream. `source_name` is what the writer stamps into the row's
/// `source` column, so consumers can audit provenance after the fact.
///
/// Implementors should support resuming from a positive `start_offset`
/// (constructed via their own `new_with_offset(...)`) so an
/// interrupted run can pick up where the checkpoint left off.
pub trait BulkSource: Iterator<Item = Result<BulkItem, BulkSourceError>> {
    fn source_name(&self) -> &str;
}
