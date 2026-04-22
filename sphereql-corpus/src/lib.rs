//! Shared test corpus for SphereQL examples.
//!
//! Provides 775 concepts across 31 academic domains with 128-dimensional
//! hand-crafted embeddings. Designed to stress-test spherical projection:
//! every semantic axis receives meaningful mass, and bridge concepts
//! deliberately straddle category boundaries.

pub mod axes;
pub mod concept;
pub mod corpus;
pub mod embed;
pub mod stress_corpus;

pub use axes::*;
pub use concept::Concept;
pub use corpus::build_corpus;
pub use embed::{embed, embed_with_noise, DEFAULT_NOISE_AMPLITUDE, DIM};
pub use stress_corpus::{
    build_stress_corpus, STRESS_CATEGORIES, STRESS_CONCEPTS_PER_CATEGORY, STRESS_NOISE_AMPLITUDE,
};
