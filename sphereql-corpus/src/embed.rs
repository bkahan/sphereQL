//! Re-export shim for the deterministic synthetic embedder.
//!
//! The embedder itself now lives in [`sphereql_core::synthetic`] so the
//! self-tune entry point in `sphereql-embed` can use it without pulling
//! in this crate's native-only arrow/parquet dependencies. The corpus
//! crate keeps `sphereql_corpus::embed` (and the internal `crate::embed`
//! callers) working through these re-exports.

pub use sphereql_core::synthetic::{DEFAULT_NOISE_AMPLITUDE, DIM, embed, embed_with_noise};
