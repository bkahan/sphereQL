//! Out-of-core query server for the sphereQL streaming viewer.
//!
//! The offline `sphereql-vis` emitter inlines an entire `Scene` into one HTML
//! file — perfect for a portable demo, but it tops out around a few hundred
//! thousand points (one JSON blob, parsed once). For millions of points the
//! viewer instead talks to *this* server: the corpus, its projection, and the
//! ANN + spatial indexes live here in memory, and the browser holds only the
//! visible working set.
//!
//! The contract is the pure [`sphereql_vis`] crate's [`Manifest`] +
//! [`TilePoint`] tiles ([`sphereql_vis::tile`]):
//!
//! - `GET /manifest` — the bounded scene descriptor (stats, overlays, palette,
//!   bounds, LOD scheme). Fetched once; size is independent of N.
//! - `GET /tiles` — a binary [`SQT1`](sphereql_vis::tile) tile of the points in
//!   a viewport cone, stratified down to an LOD budget.
//! - `POST /points` — lazy per-point metadata (label, category, quality, raw
//!   vector) by row, for the inspector.
//! - `POST /nearest` — ANN neighbors of a row or query vector, for trace.
//! - `GET /category_stats` — the palette (name → color → count).
//!
//! [`build_router`] wires these onto an [`AppState`] built by
//! [`AppState::from_corpus`]; [`crate::state`] owns the corpus → pipeline →
//! index build, and [`crate::tiles`] owns the cone-query + LOD decimation.
//!
//! [`Manifest`]: sphereql_vis::Manifest
//! [`TilePoint`]: sphereql_vis::TilePoint

use std::path::PathBuf;

use sphereql_corpus::CorpusId;

pub mod routes;
pub mod state;
pub mod tiles;

pub use routes::build_router;
pub use state::{AppState, BuildError, PointItem, StoredPoint, gate_projection};

/// Resolve a `--corpus` argument into a [`CorpusId`].
///
/// Matches the registry's short names (`hand_crafted`, `extended`, `full`,
/// `stress`, `dbpedia_50k`, …); anything else is treated as a path to a Parquet
/// file. Names are matched case-insensitively and `-`/`_` are interchangeable
/// so `--corpus hand-crafted` and `--corpus HAND_CRAFTED` both work.
pub fn parse_corpus(s: &str) -> CorpusId {
    let key = s.trim().to_ascii_lowercase().replace('-', "_");
    match key.as_str() {
        "hand_crafted" | "handcrafted" | "demo" => CorpusId::HandCrafted,
        "extended" => CorpusId::Extended,
        "full" => CorpusId::Full,
        "stress" => CorpusId::Stress,
        "dbpedia_50k" => CorpusId::DBpedia50k,
        "dbpedia_50k_clustered" => CorpusId::DBpedia50kClustered,
        "dbpedia_500k" => CorpusId::DBpedia500k,
        "dbpedia_500k_clustered" => CorpusId::DBpedia500kClustered,
        "wikidata_50k" => CorpusId::Wikidata50k,
        _ => CorpusId::Parquet(PathBuf::from(s)),
    }
}
