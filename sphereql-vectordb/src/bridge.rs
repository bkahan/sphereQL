use std::collections::HashMap;

use sphereql_embed::{
    Embedding, PcaProjection, PipelineInput, PipelineQuery, Projection, RadialStrategy,
    SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};

use crate::error::VectorStoreError;
use crate::store::VectorStore;
use crate::types::{
    PayloadUpdate, SPHEREQL_PHI_KEY, SPHEREQL_R_KEY, SPHEREQL_THETA_KEY, SearchResult, VectorRecord,
};

/// Configuration for the bridge between a vector store and sphereQL.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Maximum records per batch for upsert/sync operations.
    pub batch_size: usize,
    /// Upper bound on records to pull from the store. Guards against
    /// accidentally loading millions of vectors into memory.
    pub max_records: usize,
    /// The radial strategy used when projecting embeddings to spherical coords.
    pub radial_strategy: RadialStrategy,
    /// Enable volumetric PCA mode: r comes from PCA projection magnitude
    /// instead of embedding magnitude. Distributes points through 3D volume
    /// rather than clustering on the sphere surface.
    pub volumetric: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_records: 500_000,
            radial_strategy: RadialStrategy::Magnitude,
            volumetric: true,
        }
    }
}

/// Bidirectional bridge between a [`VectorStore`] and sphereQL's pipeline.
///
/// # Lifecycle
///
/// 1. Create with [`new`](Self::new) \u2014 connects to the store.
/// 2. Call [`build_pipeline`](Self::build_pipeline) \u2014 pulls vectors from the store,
///    fits a PCA projection, and builds the `SphereQLPipeline`.
/// 3. Use [`query`](Self::query) or [`hybrid_search`](Self::hybrid_search) to search.
/// 4. Call [`sync_projections`](Self::sync_projections) to push spherical coordinates
///    (r, \u03b8, \u03c6) back to the store as payload metadata.
pub struct VectorStoreBridge<S: VectorStore> {
    store: S,
    config: BridgeConfig,
    pipeline: Option<SphereQLPipeline>,
    projection: Option<PcaProjection>,
    /// Original store IDs, parallel to the pipeline's internal index order.
    store_ids: Vec<String>,
    /// Raw embeddings, kept for sync_projections.
    embeddings: Vec<Vec<f64>>,
}

impl<S: VectorStore> VectorStoreBridge<S> {
    pub fn new(store: S, config: BridgeConfig) -> Self {
        Self {
            store,
            config,
            pipeline: None,
            projection: None,
            store_ids: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    /// Pull all vectors from the store and build the sphereQL pipeline.
    ///
    /// `category_fn` extracts a category string from each record's metadata.
    /// Categories drive glob detection and are attached to query results.
    ///
    /// ```ignore
    /// bridge.build_pipeline(|record| {
    ///     record.metadata.get("topic")
    ///         .and_then(|v| v.as_str())
    ///         .unwrap_or("unknown")
    ///         .to_string()
    /// }).await?;
    /// ```
    pub async fn build_pipeline(
        &mut self,
        category_fn: impl Fn(&VectorRecord) -> String,
    ) -> Result<&SphereQLPipeline, VectorStoreError> {
        let records = self.fetch_all().await?;
        let n = records.len();

        if n < 3 {
            return Err(VectorStoreError::InsufficientData(format!(
                "need at least 3 records, got {n}"
            )));
        }

        let dim = records[0].vector.len();
        for (i, r) in records.iter().enumerate() {
            if r.vector.len() != dim {
                return Err(VectorStoreError::DimensionMismatch {
                    expected: dim,
                    got: r.vector.len(),
                });
            }
            if r.vector.iter().all(|&v| v.abs() < f64::EPSILON) {
                return Err(VectorStoreError::InvalidConfig(format!(
                    "record '{}' (index {i}) is a zero vector",
                    r.id
                )));
            }
        }

        let categories: Vec<String> = records.iter().map(&category_fn).collect();
        let embeddings: Vec<Vec<f64>> = records.iter().map(|r| r.vector.clone()).collect();
        let store_ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();

        let embs: Vec<Embedding> = embeddings
            .iter()
            .map(|v| Embedding::from(v.as_slice()))
            .collect();

        let projection = PcaProjection::fit(&embs, self.config.radial_strategy.clone())
            .with_volumetric(self.config.volumetric);

        let pipeline = SphereQLPipeline::new(PipelineInput {
            categories,
            embeddings: embeddings.clone(),
        });

        self.pipeline = Some(pipeline);
        self.projection = Some(projection);
        self.store_ids = store_ids;
        self.embeddings = embeddings;

        Ok(self.pipeline.as_ref().unwrap())
    }

    /// Push sphereQL spherical coordinates back to the store as payload metadata.
    ///
    /// For each record, writes:
    /// - `_sphereql_r`: radial distance
    /// - `_sphereql_theta`: azimuthal angle [0, 2\u03c0)
    /// - `_sphereql_phi`: polar angle [0, \u03c0]
    ///
    /// Uses `set_payload` so existing metadata is preserved (merged, not replaced).
    /// Returns the number of records updated.
    pub async fn sync_projections(&self) -> Result<usize, VectorStoreError> {
        let projection = self
            .projection
            .as_ref()
            .ok_or(VectorStoreError::PipelineNotBuilt)?;

        let mut updates = Vec::with_capacity(self.store_ids.len());

        for (id, emb) in self.store_ids.iter().zip(self.embeddings.iter()) {
            let embedding = Embedding::from(emb.as_slice());
            let sp = projection.project(&embedding);

            let mut metadata = HashMap::new();
            metadata.insert(SPHEREQL_R_KEY.into(), serde_json::json!(sp.r));
            metadata.insert(SPHEREQL_THETA_KEY.into(), serde_json::json!(sp.theta));
            metadata.insert(SPHEREQL_PHI_KEY.into(), serde_json::json!(sp.phi));

            updates.push(PayloadUpdate {
                id: id.clone(),
                metadata,
            });
        }

        let count = updates.len();

        for chunk in updates.chunks(self.config.batch_size) {
            self.store.set_payload(chunk).await?;
        }

        Ok(count)
    }

    /// Execute a typed sphereQL query against the pipeline.
    ///
    /// The `query_embedding` is projected through the same PCA that was
    /// fitted during `build_pipeline`, so the coordinate system is consistent.
    pub fn query(
        &self,
        q: SphereQLQuery<'_>,
        query_embedding: &[f64],
    ) -> Result<SphereQLOutput, VectorStoreError> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or(VectorStoreError::PipelineNotBuilt)?;

        let pq = PipelineQuery {
            embedding: query_embedding.to_vec(),
        };
        Ok(pipeline.query(q, &pq))
    }

    /// Hybrid search: use the vector store's ANN for initial recall,
    /// then re-rank through sphereQL's angular distance.
    ///
    /// 1. Queries the store for `recall_k` nearest neighbors (fast ANN).
    /// 2. Projects all candidates through the PCA.
    /// 3. Re-ranks by angular distance in sphereQL's spherical coordinate space.
    /// 4. Returns the top `final_k` results.
    ///
    /// This combines the store's scalable ANN index with sphereQL's
    /// geometry-aware re-ranking for higher precision.
    pub async fn hybrid_search(
        &self,
        query_embedding: &[f64],
        final_k: usize,
        recall_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let projection = self
            .projection
            .as_ref()
            .ok_or(VectorStoreError::PipelineNotBuilt)?;

        let candidates = self.store.search(query_embedding, recall_k).await?;

        let query_emb = Embedding::from(query_embedding);
        let query_sp = projection.project(&query_emb);

        let mut scored: Vec<(SearchResult, f64)> = candidates
            .into_iter()
            .filter_map(|mut result| {
                let vec = result.vector.as_ref()?;
                let emb = Embedding::from(vec.as_slice());
                let sp = projection.project(&emb);
                let angular_dist = sphereql_core::angular_distance(&query_sp, &sp);
                result.score = 1.0 - (angular_dist / std::f64::consts::PI);
                Some((result, angular_dist))
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(final_k);

        Ok(scored.into_iter().map(|(r, _)| r).collect())
    }

    /// Access the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Access the pipeline (if built).
    pub fn pipeline(&self) -> Option<&SphereQLPipeline> {
        self.pipeline.as_ref()
    }

    /// Access the fitted PCA projection (if built).
    pub fn projection(&self) -> Option<&PcaProjection> {
        self.projection.as_ref()
    }

    /// Number of records loaded into the pipeline.
    pub fn len(&self) -> usize {
        self.store_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store_ids.is_empty()
    }

    /// Scroll through the entire store, respecting `max_records`.
    async fn fetch_all(&self) -> Result<Vec<VectorRecord>, VectorStoreError> {
        let mut all = Vec::new();
        let mut offset: Option<String> = None;

        loop {
            let page = self
                .store
                .list(self.config.batch_size, offset.as_deref())
                .await?;

            let done = page.records.is_empty() || page.next_offset.is_none();
            all.extend(page.records);

            if all.len() >= self.config.max_records {
                all.truncate(self.config.max_records);
                break;
            }
            if done {
                break;
            }
            offset = page.next_offset;
        }

        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryStore;

    fn make_records(n: usize, dim: usize) -> Vec<VectorRecord> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0; dim];
                if i < n / 2 {
                    v[0] = 1.0 + i as f64 * 0.01;
                    v[1] = 0.1;
                } else {
                    v[0] = 0.1;
                    v[1] = 1.0 + i as f64 * 0.01;
                }
                v[2] = 0.05 * i as f64;

                let cat = if i < n / 2 { "group_a" } else { "group_b" };
                VectorRecord::new(format!("rec-{i}"), v)
                    .with_metadata("category", serde_json::json!(cat))
            })
            .collect()
    }

    fn category_extractor(r: &VectorRecord) -> String {
        r.metadata
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    }

    #[tokio::test]
    async fn build_pipeline_from_store() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        let pipeline = bridge.build_pipeline(category_extractor).await.unwrap();

        assert_eq!(pipeline.num_items(), 20);
        assert_eq!(bridge.len(), 20);
    }

    #[tokio::test]
    async fn build_pipeline_rejects_too_few_records() {
        let store = InMemoryStore::new("test", 5);
        store.upsert(&make_records(2, 5)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        let err = bridge.build_pipeline(category_extractor).await;
        assert!(matches!(err, Err(VectorStoreError::InsufficientData(_))));
    }

    #[tokio::test]
    async fn query_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.9; 10];
        let result = bridge
            .query(SphereQLQuery::Nearest { k: 5 }, &query_vec)
            .unwrap();

        match result {
            SphereQLOutput::Nearest(items) => {
                assert_eq!(items.len(), 5);
                assert!(items[0].distance <= items[1].distance);
            }
            _ => panic!("expected Nearest"),
        }
    }

    #[tokio::test]
    async fn query_before_build_returns_error() {
        let store = InMemoryStore::new("test", 5);
        let bridge = VectorStoreBridge::new(store, BridgeConfig::default());

        let result = bridge.query(SphereQLQuery::Nearest { k: 5 }, &[1.0; 5]);
        assert!(matches!(result, Err(VectorStoreError::PipelineNotBuilt)));
    }

    #[tokio::test]
    async fn sync_projections_writes_coordinates() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let count = bridge.sync_projections().await.unwrap();
        assert_eq!(count, 20);

        let records = bridge.store().get(&["rec-0".into()]).await.unwrap();

        let meta = &records[0].metadata;
        assert!(meta.contains_key(SPHEREQL_R_KEY));
        assert!(meta.contains_key(SPHEREQL_THETA_KEY));
        assert!(meta.contains_key(SPHEREQL_PHI_KEY));

        let r = meta[SPHEREQL_R_KEY].as_f64().unwrap();
        let theta = meta[SPHEREQL_THETA_KEY].as_f64().unwrap();
        let phi = meta[SPHEREQL_PHI_KEY].as_f64().unwrap();
        assert!(r >= 0.0);
        assert!((0.0..std::f64::consts::TAU).contains(&theta));
        assert!((0.0..=std::f64::consts::PI).contains(&phi));
    }

    #[tokio::test]
    async fn sync_before_build_returns_error() {
        let store = InMemoryStore::new("test", 5);
        let bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        let result = bridge.sync_projections().await;
        assert!(matches!(result, Err(VectorStoreError::PipelineNotBuilt)));
    }

    #[tokio::test]
    async fn hybrid_search_reranks() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(30, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.9; 10];
        let results = bridge.hybrid_search(&query_vec, 5, 15).await.unwrap();

        assert_eq!(results.len(), 5);
        for pair in results.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[tokio::test]
    async fn pipeline_glob_detection_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(30, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.9; 10];
        let result = bridge
            .query(
                SphereQLQuery::DetectGlobs {
                    k: Some(2),
                    max_k: 5,
                },
                &query_vec,
            )
            .unwrap();

        match result {
            SphereQLOutput::Globs(globs) => {
                assert_eq!(globs.len(), 2);
                let total: usize = globs.iter().map(|g| g.member_count).sum();
                assert_eq!(total, 30);
            }
            _ => panic!("expected Globs"),
        }
    }

    #[tokio::test]
    async fn fetch_all_respects_max_records() {
        let store = InMemoryStore::new("test", 3);
        let records: Vec<VectorRecord> = (0..50)
            .map(|i| VectorRecord::new(format!("r-{i}"), vec![1.0, i as f64 * 0.01, 0.0]))
            .collect();
        store.upsert(&records).await.unwrap();

        let config = BridgeConfig {
            max_records: 10,
            batch_size: 5,
            ..Default::default()
        };
        let bridge = VectorStoreBridge::new(store, config);
        let fetched = bridge.fetch_all().await.unwrap();
        assert_eq!(fetched.len(), 10);
    }

    #[tokio::test]
    async fn concept_path_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.9; 10];
        let result = bridge
            .query(
                SphereQLQuery::ConceptPath {
                    source_id: "s-0000",
                    target_id: "s-0015",
                    graph_k: 10,
                },
                &query_vec,
            )
            .unwrap();

        match result {
            SphereQLOutput::ConceptPath(Some(path)) => {
                assert!(path.steps.len() >= 2);
                assert_eq!(path.steps.first().unwrap().id, "s-0000");
                assert_eq!(path.steps.last().unwrap().id, "s-0015");
            }
            _ => panic!("expected ConceptPath(Some)"),
        }
    }
}
