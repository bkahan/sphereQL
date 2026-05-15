//! Shared test corpora for SphereQL examples.
//!
//! Ships three public corpora, all using the same 128-dim embedding
//! format:
//!
//! - [`build_corpus`] — 775 concepts across 31 academic domains with
//!   hand-crafted sparse embeddings. Every semantic axis receives
//!   meaningful mass; bridge concepts deliberately straddle category
//!   boundaries. Default noise amplitude `0.04`.
//!
//! - [`build_extended_corpus`] — ~5,000+ concepts across the same 31
//!   domains, sourced from the OpenAlex Topics taxonomy with
//!   keyword-derived axis mappings. A 6×+ expansion of the hand-crafted
//!   corpus for benchmarking and stress-testing at scale.
//!
//! - [`build_stress_corpus`] — 300 concepts across 10 synthetic
//!   categories with exactly 2 authored signal axes per concept and
//!   `0.2` noise amplitude (5× the default). A controlled A/B probe
//!   where variance-maximizing projections (PCA) degrade and
//!   connectivity-preserving projections (Laplacian eigenmap) recover
//!   the authored signature. See [`stress_corpus`] for details.
//!
//! All corpora are embedded via [`embed`](fn@embed) (default noise) or
//! [`embed_with_noise`] (explicit amplitude).
//!
//! [`build_full_corpus`] returns the union of the hand-crafted and
//! extended corpora (~5,775+ concepts total).

pub mod axes;
pub mod concept;
pub mod corpus;
pub mod derived;
pub mod embed;
pub mod extended;
pub mod loader;
pub mod parquet_loader;
pub mod parquet_writer;
pub mod stress_corpus;

pub use axes::*;
pub use concept::Concept;
pub use corpus::build_corpus;
pub use embed::{DEFAULT_NOISE_AMPLITUDE, DIM, embed, embed_with_noise};
pub use extended::{build_extended_corpus, build_full_corpus};
pub use loader::{load_extended_corpus, stream_extended_corpus};
pub use parquet_loader::{ConceptMetadata, ParquetLoadError, load_concepts_with_metadata};
pub use parquet_writer::{ConceptRow, write_concepts};
pub use stress_corpus::{
    STRESS_CATEGORIES, STRESS_CONCEPTS_PER_CATEGORY, STRESS_NOISE_AMPLITUDE, build_stress_corpus,
};
