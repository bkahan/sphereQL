use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::VectorStoreError;
use crate::store::VectorStore;
use crate::types::{PayloadUpdate, SearchResult, VectorPage, VectorRecord};

/// In-memory vector store for testing and small datasets.
///
/// Uses brute-force cosine similarity for search \u2014 O(n) per query.
/// Thread-safe via `tokio::sync::RwLock`.
pub struct InMemoryStore {
    collection: String,
    dimension: usize,
    records: RwLock<HashMap<String, VectorRecord>>,
    /// Insertion-order index for deterministic scroll pagination.
    order: RwLock<Vec<String>>,
}

impl InMemoryStore {
    pub fn new(collection: impl Into<String>, dimension: usize) -> Self {
        Self {
            collection: collection.into(),
            dimension,
            records: RwLock::new(HashMap::new()),
            order: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryStore {
    async fn upsert(&self, records: &[VectorRecord]) -> Result<(), VectorStoreError> {
        let mut store = self.records.write().await;
        let mut order = self.order.write().await;

        for record in records {
            if record.vector.len() != self.dimension {
                return Err(VectorStoreError::DimensionMismatch {
                    expected: self.dimension,
                    got: record.vector.len(),
                });
            }
            if !store.contains_key(&record.id) {
                order.push(record.id.clone());
            }
            store.insert(record.id.clone(), record.clone());
        }
        Ok(())
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<VectorRecord>, VectorStoreError> {
        let store = self.records.read().await;
        Ok(ids.iter().filter_map(|id| store.get(id).cloned()).collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<(), VectorStoreError> {
        let mut store = self.records.write().await;
        let mut order = self.order.write().await;

        for id in ids {
            store.remove(id);
            order.retain(|o| o != id);
        }
        Ok(())
    }

    async fn search(
        &self,
        vector: &[f64],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if vector.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let store = self.records.read().await;
        let query_mag = magnitude(vector);

        let mut scored: Vec<SearchResult> = store
            .values()
            .map(|record| {
                let score = cosine_similarity(vector, &record.vector, query_mag);
                SearchResult {
                    id: record.id.clone(),
                    score,
                    vector: Some(record.vector.clone()),
                    metadata: record.metadata.clone(),
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn list(
        &self,
        limit: usize,
        offset: Option<&str>,
    ) -> Result<VectorPage, VectorStoreError> {
        let store = self.records.read().await;
        let order = self.order.read().await;

        let start = match offset {
            Some(off) => off.parse::<usize>().unwrap_or(0),
            None => 0,
        };

        let records: Vec<VectorRecord> = order
            .iter()
            .skip(start)
            .take(limit)
            .filter_map(|id| store.get(id).cloned())
            .collect();

        let next_start = start + records.len();
        let next_offset = if next_start < order.len() {
            Some(next_start.to_string())
        } else {
            None
        };

        Ok(VectorPage {
            records,
            next_offset,
        })
    }

    async fn count(&self) -> Result<usize, VectorStoreError> {
        Ok(self.records.read().await.len())
    }

    async fn set_payload(&self, updates: &[PayloadUpdate]) -> Result<(), VectorStoreError> {
        let mut store = self.records.write().await;

        for update in updates {
            if let Some(record) = store.get_mut(&update.id) {
                for (key, value) in &update.metadata {
                    record.metadata.insert(key.clone(), value.clone());
                }
            }
        }
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn collection_name(&self) -> &str {
        &self.collection
    }
}

fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64], a_mag: f64) -> f64 {
    let b_mag = magnitude(b);
    if a_mag < f64::EPSILON || b_mag < f64::EPSILON {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (a_mag * b_mag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, vector: Vec<f64>) -> VectorRecord {
        VectorRecord::new(id, vector)
    }

    fn record_with_meta(id: &str, vector: Vec<f64>, key: &str, val: &str) -> VectorRecord {
        VectorRecord::new(id, vector).with_metadata(key, serde_json::json!(val))
    }

    #[tokio::test]
    async fn upsert_and_get() {
        let store = InMemoryStore::new("test", 3);
        let records = vec![
            record("a", vec![1.0, 0.0, 0.0]),
            record("b", vec![0.0, 1.0, 0.0]),
        ];
        store.upsert(&records).await.unwrap();

        let fetched = store.get(&["a".into(), "b".into()]).await.unwrap();
        assert_eq!(fetched.len(), 2);
        assert_eq!(fetched[0].id, "a");
        assert_eq!(fetched[1].id, "b");
    }

    #[tokio::test]
    async fn upsert_replaces_existing() {
        let store = InMemoryStore::new("test", 3);
        store
            .upsert(&[record("a", vec![1.0, 0.0, 0.0])])
            .await
            .unwrap();
        store
            .upsert(&[record("a", vec![0.0, 1.0, 0.0])])
            .await
            .unwrap();

        let fetched = store.get(&["a".into()]).await.unwrap();
        assert_eq!(fetched[0].vector, vec![0.0, 1.0, 0.0]);
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn dimension_mismatch_rejected() {
        let store = InMemoryStore::new("test", 3);
        let result = store
            .upsert(&[record("a", vec![1.0, 0.0])])
            .await;
        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { expected: 3, got: 2 })
        ));
    }

    #[tokio::test]
    async fn delete_removes_records() {
        let store = InMemoryStore::new("test", 3);
        store
            .upsert(&[
                record("a", vec![1.0, 0.0, 0.0]),
                record("b", vec![0.0, 1.0, 0.0]),
            ])
            .await
            .unwrap();

        store.delete(&["a".into()]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
        assert!(store.get(&["a".into()]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_nonexistent_is_silent() {
        let store = InMemoryStore::new("test", 3);
        store.delete(&["nope".into()]).await.unwrap();
    }

    #[tokio::test]
    async fn search_returns_sorted_by_similarity() {
        let store = InMemoryStore::new("test", 3);
        store
            .upsert(&[
                record("exact", vec![1.0, 0.0, 0.0]),
                record("close", vec![0.9, 0.1, 0.0]),
                record("far", vec![0.0, 0.0, 1.0]),
            ])
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "exact");
        assert!((results[0].score - 1.0).abs() < 1e-10);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[tokio::test]
    async fn search_respects_k() {
        let store = InMemoryStore::new("test", 3);
        store
            .upsert(&[
                record("a", vec![1.0, 0.0, 0.0]),
                record("b", vec![0.9, 0.1, 0.0]),
                record("c", vec![0.8, 0.2, 0.0]),
            ])
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn list_paginates() {
        let store = InMemoryStore::new("test", 3);
        store
            .upsert(&[
                record("a", vec![1.0, 0.0, 0.0]),
                record("b", vec![0.0, 1.0, 0.0]),
                record("c", vec![0.0, 0.0, 1.0]),
            ])
            .await
            .unwrap();

        let page1 = store.list(2, None).await.unwrap();
        assert_eq!(page1.records.len(), 2);
        assert!(page1.next_offset.is_some());
        assert_eq!(page1.records[0].id, "a");
        assert_eq!(page1.records[1].id, "b");

        let page2 = store
            .list(2, page1.next_offset.as_deref())
            .await
            .unwrap();
        assert_eq!(page2.records.len(), 1);
        assert!(page2.next_offset.is_none());
        assert_eq!(page2.records[0].id, "c");
    }

    #[tokio::test]
    async fn set_payload_merges() {
        let store = InMemoryStore::new("test", 3);
        store
            .upsert(&[record_with_meta("a", vec![1.0, 0.0, 0.0], "color", "red")])
            .await
            .unwrap();

        store
            .set_payload(&[PayloadUpdate {
                id: "a".into(),
                metadata: [("size".into(), serde_json::json!(42))]
                    .into_iter()
                    .collect(),
            }])
            .await
            .unwrap();

        let fetched = store.get(&["a".into()]).await.unwrap();
        assert_eq!(fetched[0].metadata["color"], "red");
        assert_eq!(fetched[0].metadata["size"], 42);
    }

    #[tokio::test]
    async fn empty_store() {
        let store = InMemoryStore::new("test", 3);
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(store.get(&["a".into()]).await.unwrap().is_empty());
        assert!(store.search(&[1.0, 0.0, 0.0], 5).await.unwrap().is_empty());
        assert_eq!(store.collection_name(), "test");
        assert_eq!(store.dimension(), 3);
    }

    #[tokio::test]
    async fn cosine_similarity_basic() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b, magnitude(&a));
        assert!((sim - 1.0).abs() < 1e-12);

        let c = [0.0, 1.0, 0.0];
        let sim2 = cosine_similarity(&a, &c, magnitude(&a));
        assert!(sim2.abs() < 1e-12);

        let d = [-1.0, 0.0, 0.0];
        let sim3 = cosine_similarity(&a, &d, magnitude(&a));
        assert!((sim3 + 1.0).abs() < 1e-12);
    }
}
