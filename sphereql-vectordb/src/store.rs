use async_trait::async_trait;

use crate::error::VectorStoreError;
use crate::types::{PayloadUpdate, SearchResult, VectorPage, VectorRecord};

/// Async interface to a vector database backend.
///
/// Implementations exist for:
/// - [`InMemoryStore`](crate::InMemoryStore) \u2014 always available, for testing and small datasets
/// - [`QdrantStore`](crate::qdrant::QdrantStore) \u2014 behind the `qdrant` feature flag
///
/// All methods accept `&self` \u2014 implementations must handle interior
/// mutability (typically via `RwLock` or the backend's own concurrency).
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert or update records. Existing records with the same ID are replaced.
    async fn upsert(&self, records: &[VectorRecord]) -> Result<(), VectorStoreError>;

    /// Retrieve records by ID. Missing IDs are silently skipped.
    async fn get(&self, ids: &[String]) -> Result<Vec<VectorRecord>, VectorStoreError>;

    /// Delete records by ID. Missing IDs are silently skipped.
    async fn delete(&self, ids: &[String]) -> Result<(), VectorStoreError>;

    /// Approximate nearest-neighbor search. Returns up to `k` results
    /// ordered by descending similarity (highest score first).
    async fn search(&self, vector: &[f64], k: usize)
    -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Scroll through all records with cursor-based pagination.
    /// Pass `None` for the first page; use the returned `next_offset` for subsequent pages.
    async fn list(
        &self,
        limit: usize,
        offset: Option<&str>,
    ) -> Result<VectorPage, VectorStoreError>;

    /// Total number of records in the collection.
    async fn count(&self) -> Result<usize, VectorStoreError>;

    /// Update only the metadata of existing records without re-uploading vectors.
    /// Fields are merged into existing metadata (not replaced).
    async fn set_payload(&self, updates: &[PayloadUpdate]) -> Result<(), VectorStoreError>;

    /// The vector dimensionality this store expects.
    fn dimension(&self) -> usize;

    /// The collection/index name.
    fn collection_name(&self) -> &str;
}
