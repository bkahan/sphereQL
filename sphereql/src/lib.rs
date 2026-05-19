//! # sphereQL
//!
//! Project high-dimensional embeddings onto a 3D sphere for fast semantic
//! search, spatial queries, category-aware exploration, and interactive
//! visualization.
//!
//! This umbrella crate re-exports each workspace sub-crate behind a named
//! feature flag. Enable only what you need:
//!
//! ```toml
//! [dependencies]
//! sphereql = { version = "0.2.0-alpha", features = ["full"] }
//! ```
//!
//! ## Feature flags
//!
//! | Feature | What it unlocks |
//! |---|---|
//! | `core` (default) | [`core`] — spherical math, coordinates, distance metrics, regions |
//! | `index` | [`index`] — spatial indexing with shell + sector partitioning |
//! | `layout` | [`layout`] — layout engines (Fibonacci, k-means, force-directed) |
//! | `embed` | [`embed`] — projections, query pipeline, category enrichment, auto-tuner, meta-model |
//! | `graphql` | [`graphql`] — `async-graphql` schema with spatial + category queries |
//! | `vectordb` | [`vectordb`] — Qdrant / Pinecone / InMemory vector store bridge |
//! | `pinecone` | Adds Pinecone backend (pulls `reqwest`; excluded from `full`) |
//! | `retain-embeddings` | Keep original high-d embeddings in the pipeline for cosine re-ranking |
//! | `full` | All of the above except `pinecone` |
//!
//! ## Minimal example
//!
//! ```rust,ignore
//! use sphereql::embed::*;
//!
//! let input = PipelineInput {
//!     categories: vec!["science".into(), "cooking".into()],
//!     embeddings: vec![vec![0.1, 0.9, 0.3], vec![0.9, 0.1, 0.0]],
//! };
//! let pipeline = SphereQLPipeline::new(input).unwrap();
//! let query = PipelineQuery { embedding: vec![0.15, 0.85, 0.35] };
//! let results = pipeline.query(SphereQLQuery::Nearest { k: 1 }, &query).unwrap();
//! ```
//!
//! See the [repository README](https://github.com/bkahan/sphereQL) and
//! `docs/` for quickstarts in Rust, Python, and WASM.

#[cfg(feature = "core")]
pub mod core {
    pub use sphereql_core::*;
}

#[cfg(feature = "index")]
pub mod index {
    pub use sphereql_index::*;
}

#[cfg(feature = "layout")]
pub mod layout {
    pub use sphereql_layout::*;
}

#[cfg(feature = "embed")]
pub mod embed {
    pub use sphereql_embed::*;
}

#[cfg(feature = "graphql")]
pub mod graphql {
    pub use sphereql_graphql::*;
}

#[cfg(feature = "vectordb")]
pub mod vectordb {
    pub use sphereql_vectordb::*;
}
