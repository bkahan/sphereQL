//! Vector database integration layer.
//!
//! Bridges external vector stores (in-memory, Pinecone, Qdrant) with the
//! sphereQL embedding pipeline. Handles sync, projection, and hybrid search
//! with cosine similarity re-ranking in the original embedding space.

pub mod bridge;
pub mod error;
pub mod memory;
pub mod store;
pub mod types;

#[cfg(feature = "pinecone")]
pub mod pinecone;
#[cfg(feature = "qdrant")]
pub mod qdrant;

pub use bridge::*;
pub use error::*;
pub use memory::*;
pub use store::*;
pub use types::*;
