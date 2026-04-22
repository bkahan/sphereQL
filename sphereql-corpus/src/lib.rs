//! Shared test corpora for SphereQL examples.
//!
//! Ships two public corpora, both using the same 128-dim embedding
//! format:
//!
//! - [`build_corpus`] — 775 concepts across 31 academic domains with
//!   hand-crafted sparse embeddings. Every semantic axis receives
//!   meaningful mass; bridge concepts deliberately straddle category
//!   boundaries. Default noise amplitude `0.04`.
//!
//! - [`build_stress_corpus`] — 300 concepts across 10 synthetic
//!   categories with exactly 2 authored signal axes per concept and
//!   `0.2` noise amplitude (5× the default). A controlled A/B probe
//!   where variance-maximizing projections (PCA) degrade and
//!   connectivity-preserving projections (Laplacian eigenmap) recover
//!   the authored signature. See [`stress_corpus`] for details.
//!
//! Both corpora are embedded via [`embed`] (default noise) or
//! [`embed_with_noise`] (explicit amplitude).

pub mod axes;
pub mod concept;
pub mod corpus;
pub mod embed;
pub mod stress_corpus;

pub use axes::*;
pub use concept::Concept;
pub use corpus::build_corpus;
pub use embed::{DEFAULT_NOISE_AMPLITUDE, DIM, embed, embed_with_noise};
pub use stress_corpus::{
    STRESS_CATEGORIES, STRESS_CONCEPTS_PER_CATEGORY, STRESS_NOISE_AMPLITUDE, build_stress_corpus,
};
