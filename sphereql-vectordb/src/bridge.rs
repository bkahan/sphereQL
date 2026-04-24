use std::collections::HashMap;

use sphereql_embed::{
    ConfiguredProjection, Embedding, PcaProjection, PipelineConfig, PipelineInput, PipelineQuery,
    Projection, ProjectionKind, RadialStrategy, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
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
    projection: Option<ConfiguredProjection>,
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

    /// Pull all vectors from the store and build the sphereQL pipeline
    /// using a PCA projection with the bridge's `radial_strategy` and
    /// `volumetric` settings.
    ///
    /// `category_fn` extracts a category string from each record's metadata.
    /// Categories drive glob detection and are attached to query results.
    ///
    /// For non-PCA projections (kernel PCA, Laplacian eigenmap) or to
    /// override bridge / inner-sphere thresholds, use
    /// [`build_pipeline_with_config`](Self::build_pipeline_with_config).
    pub async fn build_pipeline(
        &mut self,
        category_fn: impl Fn(&VectorRecord) -> String,
    ) -> Result<&SphereQLPipeline, VectorStoreError> {
        let (store_ids, categories, embeddings) = self.prepare_records(category_fn).await?;

        // `PcaProjection::fit` + `with_projection` take `&[Embedding]` and
        // `Vec<Embedding>` respectively, so this path needs one
        // `Vec<Embedding>`. `From<&[f64]>` clones the slice into the
        // `Embedding`; this is one allocation per row.
        let embs: Vec<Embedding> = embeddings
            .iter()
            .map(|v| Embedding::from(v.as_slice()))
            .collect();

        let pca = PcaProjection::fit(&embs, self.config.radial_strategy.clone())
            .map_err(|e| VectorStoreError::InvalidConfig(e.to_string()))?
            .with_volumetric(self.config.volumetric);

        let pipeline = SphereQLPipeline::with_projection(categories, embs, pca.clone())
            .map_err(|e| VectorStoreError::InvalidConfig(e.to_string()))?;

        self.pipeline = Some(pipeline);
        self.projection = Some(ConfiguredProjection::Pca(pca));
        self.store_ids = store_ids;
        self.embeddings = embeddings;

        Ok(self.pipeline.as_ref().unwrap())
    }

    /// Pull all vectors from the store and build the sphereQL pipeline
    /// with an explicit [`PipelineConfig`].
    ///
    /// Selects the projection family from `config.projection_kind` and
    /// uses the matching sub-config (e.g. `config.laplacian` for Laplacian
    /// eigenmap). `BridgeConfig::radial_strategy` and
    /// `BridgeConfig::volumetric` are ignored on this path — the configured
    /// fitter owns those choices.
    pub async fn build_pipeline_with_config(
        &mut self,
        category_fn: impl Fn(&VectorRecord) -> String,
        config: PipelineConfig,
    ) -> Result<&SphereQLPipeline, VectorStoreError> {
        let (store_ids, categories, embeddings) = self.prepare_records(category_fn).await?;

        // `SphereQLPipeline::new_with_config` takes ownership of a
        // `PipelineInput` and re-wraps its `Vec<Vec<f64>>` into
        // `Embedding`s via `From<Vec<f64>>` (a move). We still keep a
        // cached copy of the embeddings for `sync_projections` to
        // replay later, so we clone once here rather than letting the
        // pipeline builder construct and throw away temporaries.
        let pipeline = SphereQLPipeline::new_with_config(
            PipelineInput {
                categories,
                embeddings: embeddings.clone(),
            },
            config,
        )
        .map_err(|e| VectorStoreError::InvalidConfig(e.to_string()))?;

        self.projection = Some(pipeline.projection().clone());
        self.pipeline = Some(pipeline);
        self.store_ids = store_ids;
        self.embeddings = embeddings;

        Ok(self.pipeline.as_ref().unwrap())
    }

    /// Fetch, validate, and split records into the tuple both build paths
    /// need. Guarantees: `n >= 3`, all vectors same dim, no near-zero
    /// magnitudes.
    ///
    /// Previously also eagerly built a `Vec<Embedding>` that
    /// `build_pipeline_with_config` then discarded. Each build path now
    /// constructs its own `Embedding`s only when it actually needs them.
    async fn prepare_records(
        &self,
        category_fn: impl Fn(&VectorRecord) -> String,
    ) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>), VectorStoreError> {
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
            let mag: f64 = r.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
            if mag < 1e-9 {
                return Err(VectorStoreError::InvalidConfig(format!(
                    "record '{}' (index {i}) is a near-zero vector (magnitude {mag:.2e})",
                    r.id
                )));
            }
        }

        let categories: Vec<String> = records.iter().map(&category_fn).collect();
        let embeddings: Vec<Vec<f64>> = records.iter().map(|r| r.vector.clone()).collect();
        let store_ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();

        Ok((store_ids, categories, embeddings))
    }

    /// Push sphereQL spherical coordinates back to the store as payload metadata.
    ///
    /// For each record, writes `_sphereql_r`, `_sphereql_theta`, `_sphereql_phi`.
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
        pipeline
            .query(q, &pq)
            .map_err(|e| VectorStoreError::InvalidConfig(e.to_string()))
    }

    /// Hybrid search: use the vector store's ANN for initial recall,
    /// then re-rank through **original cosine similarity** for precision.
    ///
    /// 1. Queries the store for `recall_k` nearest neighbors (fast ANN).
    /// 2. Re-ranks candidates by cosine similarity in the original
    ///    high-dimensional space (not the lossy 3D projection).
    /// 3. Returns the top `final_k` results.
    ///
    /// The spherical projection is NOT used for re-ranking because its
    /// low explained variance ratio makes angular distance a poor proxy
    /// for true semantic similarity beyond k=1.
    pub async fn hybrid_search(
        &self,
        query_embedding: &[f64],
        final_k: usize,
        recall_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if self.projection.is_none() {
            return Err(VectorStoreError::PipelineNotBuilt);
        }

        let candidates = self.store.search(query_embedding, recall_k).await?;
        let recalled = candidates.len();

        // Re-rank by original cosine similarity — the true semantic
        // distance. Candidates without a `vector` (stores can omit
        // values on cost/bandwidth grounds) can't be re-scored, so
        // they're dropped here. The `dropped_no_vector` counter below
        // makes that visible to callers via the tracing event.
        let mut dropped_no_vector = 0usize;
        let mut scored: Vec<(SearchResult, f64)> = candidates
            .into_iter()
            .filter_map(|mut result| {
                let Some(vec) = result.vector.as_ref() else {
                    dropped_no_vector += 1;
                    return None;
                };
                let cosine_sim = sphereql_core::cosine_similarity(query_embedding, vec);
                result.score = cosine_sim;
                Some((result, cosine_sim))
            })
            .collect();

        // Sort descending by cosine similarity (higher = more similar)
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(final_k);

        // Observability hooks. A caller that asked for `final_k` but
        // got fewer results has two likely causes: the store returned
        // fewer than `recall_k` candidates, or the store returned
        // candidates without stored vectors. Both are recoverable at
        // the caller level (retry with a larger recall, enable value
        // storage on the backend) — surface them without failing the
        // query.
        if scored.len() < final_k {
            tracing::warn!(
                recalled,
                dropped_no_vector,
                final_k,
                recall_k,
                "hybrid_search returned fewer results than requested; \
                 consider raising recall_k or enabling vector storage \
                 on the backend"
            );
        } else if dropped_no_vector > 0 {
            tracing::debug!(
                dropped_no_vector,
                "hybrid_search dropped candidates missing `vector`; \
                 re-rank couldn't score them"
            );
        }

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

    /// Access the fitted projection (if built). The returned
    /// [`ConfiguredProjection`] may be PCA, kernel PCA, or Laplacian
    /// eigenmap depending on which `build_pipeline*` entry point was used.
    pub fn projection(&self) -> Option<&ConfiguredProjection> {
        self.projection.as_ref()
    }

    /// Projection family name ("pca", "kernel_pca", "laplacian_eigenmap"),
    /// or None if the pipeline has not been built.
    pub fn projection_kind(&self) -> Option<ProjectionKind> {
        self.projection.as_ref().map(|p| match p {
            ConfiguredProjection::Pca(_) => ProjectionKind::Pca,
            ConfiguredProjection::KernelPca(_) => ProjectionKind::KernelPca,
            ConfiguredProjection::Laplacian(_) => ProjectionKind::LaplacianEigenmap,
        })
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
    async fn build_pipeline_with_config_uses_configured_projection() {
        use sphereql_embed::{PipelineConfig, ProjectionKind};

        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        let config = PipelineConfig {
            projection_kind: ProjectionKind::KernelPca,
            ..Default::default()
        };
        bridge
            .build_pipeline_with_config(category_extractor, config)
            .await
            .unwrap();

        assert_eq!(bridge.projection_kind(), Some(ProjectionKind::KernelPca));
        assert!(matches!(
            bridge.projection(),
            Some(ConfiguredProjection::KernelPca(_))
        ));
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

    // ── Category enrichment through bridge ────────────────────────────

    #[tokio::test]
    async fn category_concept_path_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.0; 10];
        let result = bridge
            .query(
                SphereQLQuery::CategoryConceptPath {
                    source_category: "group_a",
                    target_category: "group_b",
                },
                &query_vec,
            )
            .unwrap();

        match result {
            SphereQLOutput::CategoryConceptPath(Some(path)) => {
                assert!(path.steps.len() >= 2);
                assert_eq!(path.steps.first().unwrap().category_name, "group_a");
                assert_eq!(path.steps.last().unwrap().category_name, "group_b");
                assert!(path.total_distance > 0.0);
            }
            SphereQLOutput::CategoryConceptPath(None) => {
                panic!("expected a path between group_a and group_b")
            }
            _ => panic!("expected CategoryConceptPath"),
        }
    }

    #[tokio::test]
    async fn category_neighbors_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.0; 10];
        let result = bridge
            .query(
                SphereQLQuery::CategoryNeighbors {
                    category: "group_a",
                    k: 5,
                },
                &query_vec,
            )
            .unwrap();

        match result {
            SphereQLOutput::CategoryNeighbors(neighbors) => {
                assert!(!neighbors.is_empty());
                assert_eq!(neighbors[0].name, "group_b");
            }
            _ => panic!("expected CategoryNeighbors"),
        }
    }

    #[tokio::test]
    async fn drill_down_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.9; 10];
        let result = bridge
            .query(
                SphereQLQuery::DrillDown {
                    category: "group_a",
                    k: 5,
                },
                &query_vec,
            )
            .unwrap();

        match result {
            SphereQLOutput::DrillDown(results) => {
                assert!(!results.is_empty());
                assert!(results.len() <= 5);
                for w in results.windows(2) {
                    assert!(w[0].distance <= w[1].distance);
                }
            }
            _ => panic!("expected DrillDown"),
        }
    }

    #[tokio::test]
    async fn category_stats_through_bridge() {
        let store = InMemoryStore::new("test", 10);
        store.upsert(&make_records(20, 10)).await.unwrap();

        let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
        bridge.build_pipeline(category_extractor).await.unwrap();

        let query_vec = vec![0.0; 10];
        let result = bridge
            .query(SphereQLQuery::CategoryStats, &query_vec)
            .unwrap();

        match result {
            SphereQLOutput::CategoryStats {
                summaries,
                inner_sphere_reports,
            } => {
                assert_eq!(summaries.len(), 2);
                let names: Vec<&str> = summaries.iter().map(|s| s.name.as_str()).collect();
                assert!(names.contains(&"group_a"));
                assert!(names.contains(&"group_b"));
                for s in &summaries {
                    assert!(s.member_count > 0);
                    assert!(s.cohesion > 0.0 && s.cohesion <= 1.0);
                }
                // 10 items per group is below the inner sphere threshold (20)
                assert!(inner_sphere_reports.is_empty());
            }
            _ => panic!("expected CategoryStats"),
        }
    }
}
