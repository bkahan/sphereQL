use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::Runtime;

use sphereql_embed::config::PipelineConfig;
use sphereql_embed::pipeline::{SphereQLOutput, SphereQLQuery};
use sphereql_embed::projection::Projection;
use sphereql_embed::types::Embedding;
use sphereql_vectordb::store::VectorStore;
use sphereql_vectordb::types::{PayloadUpdate, SearchResult, VectorPage};
use sphereql_vectordb::{
    BridgeConfig, InMemoryStore, VectorRecord, VectorStoreBridge, VectorStoreError,
};

use crate::types::{
    Glob, Manifold, Nearest, Path, PyCategoryPath, PyCategorySummary, PyDomainGroup, PyDrillDown,
    PyInnerSphereReport, PyProjectionWarning,
};

// ── SharedStore ──────────────────────────────────────────────────────────
// Newtype so we can impl VectorStore for Arc<InMemoryStore> without
// modifying the vectordb crate.

struct SharedStore(Arc<InMemoryStore>);

#[async_trait]
impl VectorStore for SharedStore {
    async fn upsert(&self, records: &[VectorRecord]) -> Result<(), VectorStoreError> {
        self.0.upsert(records).await
    }
    async fn get(&self, ids: &[String]) -> Result<Vec<VectorRecord>, VectorStoreError> {
        self.0.get(ids).await
    }
    async fn delete(&self, ids: &[String]) -> Result<(), VectorStoreError> {
        self.0.delete(ids).await
    }
    async fn search(
        &self,
        vector: &[f64],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.0.search(vector, k).await
    }
    async fn list(
        &self,
        limit: usize,
        offset: Option<&str>,
    ) -> Result<VectorPage, VectorStoreError> {
        self.0.list(limit, offset).await
    }
    async fn count(&self) -> Result<usize, VectorStoreError> {
        self.0.count().await
    }
    async fn set_payload(&self, updates: &[PayloadUpdate]) -> Result<(), VectorStoreError> {
        self.0.set_payload(updates).await
    }
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn collection_name(&self) -> &str {
        self.0.collection_name()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn make_runtime() -> PyResult<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create tokio runtime: {e}")))
}

fn vstore_err(e: VectorStoreError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn parse_record(dict: &Bound<'_, PyDict>) -> PyResult<VectorRecord> {
    let id: String = dict
        .get_item("id")?
        .ok_or_else(|| PyValueError::new_err("record missing 'id'"))?
        .extract()?;

    let vector: Vec<f64> = dict
        .get_item("vector")?
        .ok_or_else(|| PyValueError::new_err("record missing 'vector'"))?
        .extract()?;

    let mut record = VectorRecord::new(id, vector);

    if let Some(meta_obj) = dict.get_item("metadata")? {
        let meta: &Bound<'_, PyDict> = meta_obj
            .cast()
            .map_err(|_| PyValueError::new_err("'metadata' must be a dict"))?;
        for (k, v) in meta.iter() {
            let key: String = k.extract()?;
            let val: serde_json::Value = pythonize::depythonize(&v)
                .map_err(|e| PyValueError::new_err(format!("metadata value error: {e}")))?;
            record.metadata.insert(key, val);
        }
    }

    Ok(record)
}

fn category_extractor(key: &str) -> impl Fn(&VectorRecord) -> String + '_ {
    move |r: &VectorRecord| {
        r.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    }
}

fn dummy_query_vec(bridge_dim: Option<usize>) -> Vec<f64> {
    vec![0.0; bridge_dim.unwrap_or(0)]
}

// ── Shared bridge method macro ──────────────────────────────────────────
// Every VectorStoreBridge wrapper (InMemory, Qdrant, Pinecone) exposes an
// identical query surface. The macro below stamps that surface onto each
// wrapper type so the methods stay in lock-step without 3× copy-paste.
//
// Requires `pyo3`'s `multiple-pymethods` feature — each wrapper keeps a
// separate `#[pymethods]` block for its type-specific constructor.

macro_rules! impl_vector_bridge_methods {
    ($type:ident, $label:expr) => {
        #[pymethods]
        impl $type {
            /// Pull all vectors from the store and build the sphereQL
            /// pipeline using PCA with the bridge's default radial
            /// strategy and volumetric settings.
            #[pyo3(signature = (*, category_key="category"))]
            fn build_pipeline(&mut self, category_key: &str) -> PyResult<()> {
                let extractor = category_extractor(category_key);
                self.rt
                    .block_on(self.bridge.build_pipeline(extractor))
                    .map_err(vstore_err)?;
                Ok(())
            }

            /// Pull all vectors and build the pipeline with an explicit
            /// PipelineConfig (non-PCA projections, custom bridge /
            /// inner-sphere / routing thresholds).
            #[pyo3(signature = (config, *, category_key="category"))]
            fn build_pipeline_with_config(
                &mut self,
                config: &Bound<'_, PyAny>,
                category_key: &str,
            ) -> PyResult<()> {
                let cfg: PipelineConfig = pythonize::depythonize(config)
                    .map_err(|e| PyValueError::new_err(format!("invalid config dict: {e}")))?;
                let extractor = category_extractor(category_key);
                self.rt
                    .block_on(self.bridge.build_pipeline_with_config(extractor, cfg))
                    .map_err(vstore_err)?;
                Ok(())
            }

            #[pyo3(signature = (embedding, *, k=5))]
            fn query_nearest(&self, embedding: Vec<f64>, k: usize) -> PyResult<Vec<Nearest>> {
                match self
                    .bridge
                    .query(SphereQLQuery::Nearest { k }, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::Nearest(items) => {
                        Ok(items.iter().map(Nearest::from).collect())
                    }
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            #[pyo3(signature = (embedding, *, min_cosine=0.8))]
            fn query_similar(
                &self,
                embedding: Vec<f64>,
                min_cosine: f64,
            ) -> PyResult<Vec<Nearest>> {
                match self
                    .bridge
                    .query(SphereQLQuery::SimilarAbove { min_cosine }, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::KNearest(items) => {
                        Ok(items.iter().map(Nearest::from).collect())
                    }
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            #[pyo3(signature = (source_id, target_id, *, graph_k=10, embedding))]
            fn query_concept_path(
                &self,
                source_id: &str,
                target_id: &str,
                graph_k: usize,
                embedding: Vec<f64>,
            ) -> PyResult<Option<Path>> {
                match self
                    .bridge
                    .query(
                        SphereQLQuery::ConceptPath {
                            source_id,
                            target_id,
                            graph_k,
                        },
                        &embedding,
                    )
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::ConceptPath(path) => Ok(path.as_ref().map(Path::from)),
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            #[pyo3(signature = (embedding, *, k=None, max_k=10))]
            fn query_detect_globs(
                &self,
                embedding: Vec<f64>,
                k: Option<usize>,
                max_k: usize,
            ) -> PyResult<Vec<Glob>> {
                match self
                    .bridge
                    .query(SphereQLQuery::DetectGlobs { k, max_k }, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::Globs(globs) => Ok(globs.iter().map(Glob::from).collect()),
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            /// Fit a local manifold around the query point in the
            /// pipeline's projected space.
            #[pyo3(signature = (embedding, *, neighborhood_k=10))]
            fn query_local_manifold(
                &self,
                embedding: Vec<f64>,
                neighborhood_k: usize,
            ) -> PyResult<Manifold> {
                match self
                    .bridge
                    .query(SphereQLQuery::LocalManifold { neighborhood_k }, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::LocalManifold(m) => Ok(Manifold::from(&m)),
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            // ── Category Enrichment Layer ───────────────────────────

            /// Find the shortest path between two categories through
            /// the category graph.
            fn category_concept_path(
                &self,
                source_category: &str,
                target_category: &str,
            ) -> PyResult<Option<PyCategoryPath>> {
                let dim = self.projection_dim();
                let embedding = dummy_query_vec(dim);
                match self
                    .bridge
                    .query(
                        SphereQLQuery::CategoryConceptPath {
                            source_category,
                            target_category,
                        },
                        &embedding,
                    )
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::CategoryConceptPath(path) => {
                        Ok(path.as_ref().map(PyCategoryPath::from))
                    }
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            /// k-NN over category centroids starting from `category`.
            #[pyo3(signature = (category, *, k=5))]
            fn category_neighbors(
                &self,
                category: &str,
                k: usize,
            ) -> PyResult<Vec<PyCategorySummary>> {
                let dim = self.projection_dim();
                let embedding = dummy_query_vec(dim);
                match self
                    .bridge
                    .query(SphereQLQuery::CategoryNeighbors { category, k }, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::CategoryNeighbors(summaries) => {
                        Ok(summaries.iter().map(PyCategorySummary::from).collect())
                    }
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            /// Drill into one category: k-NN within that category using
            /// its inner sphere's projection when available.
            #[pyo3(signature = (category, embedding, *, k=10))]
            fn drill_down(
                &self,
                category: &str,
                embedding: Vec<f64>,
                k: usize,
            ) -> PyResult<Vec<PyDrillDown>> {
                match self
                    .bridge
                    .query(SphereQLQuery::DrillDown { category, k }, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::DrillDown(results) => {
                        Ok(results.iter().map(PyDrillDown::from).collect())
                    }
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            /// Per-category summaries plus inner-sphere reports.
            fn category_stats(
                &self,
            ) -> PyResult<(Vec<PyCategorySummary>, Vec<PyInnerSphereReport>)> {
                let dim = self.projection_dim();
                let embedding = dummy_query_vec(dim);
                match self
                    .bridge
                    .query(SphereQLQuery::CategoryStats, &embedding)
                    .map_err(vstore_err)?
                {
                    SphereQLOutput::CategoryStats {
                        summaries,
                        inner_sphere_reports,
                    } => Ok((
                        summaries.iter().map(PyCategorySummary::from).collect(),
                        inner_sphere_reports
                            .iter()
                            .map(PyInnerSphereReport::from)
                            .collect(),
                    )),
                    _ => Err(PyRuntimeError::new_err("unexpected output type")),
                }
            }

            // ── Hierarchical routing ────────────────────────────────

            #[pyo3(signature = (embedding, *, k=5))]
            fn hierarchical_nearest(
                &self,
                embedding: Vec<f64>,
                k: usize,
            ) -> PyResult<Vec<Nearest>> {
                let pipeline = self
                    .bridge
                    .pipeline()
                    .ok_or_else(|| PyRuntimeError::new_err("pipeline not built"))?;
                let emb = Embedding::new(embedding);
                Ok(pipeline
                    .hierarchical_nearest(&emb, k)
                    .iter()
                    .map(Nearest::from)
                    .collect())
            }

            /// Coarse domain groups detected from category geometry.
            fn domain_groups(&self) -> PyResult<Vec<PyDomainGroup>> {
                let pipeline = self
                    .bridge
                    .pipeline()
                    .ok_or_else(|| PyRuntimeError::new_err("pipeline not built"))?;
                Ok(pipeline
                    .domain_groups()
                    .iter()
                    .map(PyDomainGroup::from)
                    .collect())
            }

            /// Projection-quality warnings. Empty when EVR is healthy.
            fn projection_warnings(&self) -> PyResult<Vec<PyProjectionWarning>> {
                let pipeline = self
                    .bridge
                    .pipeline()
                    .ok_or_else(|| PyRuntimeError::new_err("pipeline not built"))?;
                Ok(pipeline
                    .projection_warnings()
                    .iter()
                    .map(PyProjectionWarning::from)
                    .collect())
            }

            // ── Hybrid / sync / introspection ────────────────────────

            #[pyo3(signature = (embedding, *, final_k=5, recall_k=20))]
            fn hybrid_search<'py>(
                &self,
                py: Python<'py>,
                embedding: Vec<f64>,
                final_k: usize,
                recall_k: usize,
            ) -> PyResult<Vec<Bound<'py, PyDict>>> {
                let results = self
                    .rt
                    .block_on(self.bridge.hybrid_search(&embedding, final_k, recall_k))
                    .map_err(vstore_err)?;
                results
                    .iter()
                    .map(|r| search_result_to_dict(py, r))
                    .collect()
            }

            fn sync_projections(&self) -> PyResult<usize> {
                self.rt
                    .block_on(self.bridge.sync_projections())
                    .map_err(vstore_err)
            }

            /// Short name of the projection family, or None before the
            /// pipeline is built. One of "pca", "kernel_pca",
            /// "laplacian_eigenmap".
            #[getter]
            fn projection_kind(&self) -> Option<&'static str> {
                self.bridge.projection_kind().map(|k| k.name())
            }

            /// PipelineConfig of the built pipeline as a dict, or None.
            fn config<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
                match self.bridge.pipeline() {
                    Some(p) => Ok(Some(pythonize::pythonize(py, p.config()).map_err(|e| {
                        PyValueError::new_err(format!("failed to serialize config: {e}"))
                    })?)),
                    None => Ok(None),
                }
            }

            fn __len__(&self) -> usize {
                self.bridge.len()
            }

            fn __repr__(&self) -> String {
                format!("{}(records={})", $label, self.bridge.len())
            }
        }

        impl $type {
            /// Dimensionality of the fitted projection, if the pipeline
            /// has been built. Used by category queries that don't
            /// take a user-supplied embedding.
            fn projection_dim(&self) -> Option<usize> {
                self.bridge
                    .pipeline()
                    .map(|p| p.projection().dimensionality())
            }
        }
    };
}

// ── PyInMemoryStore ──────────────────────────────────────────────────────

/// In-memory vector store for testing and small datasets.
///
/// Uses brute-force cosine similarity — O(n) per query.
/// Pass this to VectorStoreBridge to build a sphereQL pipeline.
#[pyclass(name = "InMemoryStore")]
pub struct PyInMemoryStore {
    inner: Arc<InMemoryStore>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyInMemoryStore {
    /// Create an in-memory vector store.
    ///
    /// Args:
    ///     collection: Name for this collection.
    ///     dimension: Dimensionality of stored vectors.
    #[new]
    fn new(collection: String, dimension: usize) -> PyResult<Self> {
        let rt = Arc::new(make_runtime()?);
        Ok(Self {
            inner: Arc::new(InMemoryStore::new(collection, dimension)),
            rt,
        })
    }

    /// Insert or update records.
    ///
    /// Args:
    ///     records: List of dicts with keys 'id' (str), 'vector' (list[float]),
    ///         and optionally 'metadata' (dict).
    fn upsert(&self, records: Vec<Bound<'_, PyDict>>) -> PyResult<()> {
        let recs: Vec<VectorRecord> = records.iter().map(parse_record).collect::<PyResult<_>>()?;
        self.rt
            .block_on(self.inner.upsert(&recs))
            .map_err(vstore_err)
    }

    /// Return the number of records in the store.
    fn count(&self) -> PyResult<usize> {
        self.rt.block_on(self.inner.count()).map_err(vstore_err)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.count()
    }

    fn __repr__(&self) -> String {
        format!(
            "InMemoryStore(collection={:?}, dimension={})",
            self.inner.collection_name(),
            self.inner.dimension()
        )
    }
}

// ── PyVectorStoreBridge (InMemory) ───────────────────────────────────────

/// Bridge between an InMemoryStore and the sphereQL pipeline.
///
/// Pulls vectors from the store, fits a projection, and exposes the
/// full sphereQL query surface (nearest, concept path, globs, category
/// enrichment, hierarchical routing, hybrid search).
#[pyclass(name = "VectorStoreBridge")]
pub struct PyVectorStoreBridge {
    bridge: VectorStoreBridge<SharedStore>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyVectorStoreBridge {
    /// Create a bridge from an InMemoryStore.
    ///
    /// Args:
    ///     store: The InMemoryStore to pull vectors from.
    ///     batch_size: Records per batch for sync operations. Default 100.
    ///     max_records: Upper bound on records to load. Default 500,000.
    #[new]
    #[pyo3(signature = (store, *, batch_size=None, max_records=None))]
    fn new(
        store: &PyInMemoryStore,
        batch_size: Option<usize>,
        max_records: Option<usize>,
    ) -> PyResult<Self> {
        let config = BridgeConfig {
            batch_size: batch_size.unwrap_or(100),
            max_records: max_records.unwrap_or(500_000),
            ..Default::default()
        };
        let shared = SharedStore(Arc::clone(&store.inner));
        let rt = Arc::clone(&store.rt);
        Ok(Self {
            bridge: VectorStoreBridge::new(shared, config),
            rt,
        })
    }
}

impl_vector_bridge_methods!(PyVectorStoreBridge, "VectorStoreBridge");

// ── PyQdrantBridge ───────────────────────────────────────────────────────

#[cfg(feature = "qdrant")]
mod qdrant_bridge {
    use super::*;
    use sphereql_vectordb::qdrant::{QdrantConfig, QdrantStore};

    /// Bridge between a Qdrant collection and the sphereQL pipeline.
    ///
    /// Connects to a Qdrant instance, pulls vectors, fits a projection,
    /// and exposes the full sphereQL query surface plus hybrid search.
    #[pyclass(name = "QdrantBridge")]
    pub struct PyQdrantBridge {
        pub(super) bridge: VectorStoreBridge<QdrantStore>,
        pub(super) rt: Arc<Runtime>,
    }

    #[pymethods]
    impl PyQdrantBridge {
        /// Connect to Qdrant and create a bridge.
        #[new]
        #[pyo3(signature = (url, collection, dimension, *, api_key=None, batch_size=None, max_records=None))]
        fn new(
            url: String,
            collection: String,
            dimension: usize,
            api_key: Option<String>,
            batch_size: Option<usize>,
            max_records: Option<usize>,
        ) -> PyResult<Self> {
            let rt = Arc::new(make_runtime()?);
            let mut config = QdrantConfig::new(url, collection, dimension);
            if let Some(key) = api_key {
                config = config.with_api_key(key);
            }
            let store = rt
                .block_on(QdrantStore::connect(config))
                .map_err(vstore_err)?;

            let bridge_config = BridgeConfig {
                batch_size: batch_size.unwrap_or(100),
                max_records: max_records.unwrap_or(500_000),
                ..Default::default()
            };

            Ok(Self {
                bridge: VectorStoreBridge::new(store, bridge_config),
                rt,
            })
        }
    }

    impl_vector_bridge_methods!(PyQdrantBridge, "QdrantBridge");
}

#[cfg(feature = "qdrant")]
pub use qdrant_bridge::PyQdrantBridge;

// ── PyPineconeBridge ────────────────────────────────────────────────────

#[cfg(feature = "pinecone")]
mod pinecone_bridge {
    use super::*;
    use sphereql_vectordb::pinecone::{PineconeConfig, PineconeStore};

    /// Bridge between a Pinecone index and the sphereQL pipeline.
    ///
    /// Connects to Pinecone via REST API, pulls vectors, fits a
    /// projection, and exposes the full sphereQL query surface.
    #[pyclass(name = "PineconeBridge")]
    pub struct PyPineconeBridge {
        pub(super) bridge: VectorStoreBridge<PineconeStore>,
        pub(super) rt: Arc<Runtime>,
    }

    #[pymethods]
    impl PyPineconeBridge {
        /// Connect to a Pinecone index and create a bridge.
        #[new]
        #[pyo3(signature = (api_key, host, dimension, *, namespace=None, batch_size=None, max_records=None))]
        fn new(
            api_key: String,
            host: String,
            dimension: usize,
            namespace: Option<String>,
            batch_size: Option<usize>,
            max_records: Option<usize>,
        ) -> PyResult<Self> {
            let rt = Arc::new(make_runtime()?);
            let mut config = PineconeConfig::new(api_key, host, dimension);
            if let Some(ns) = namespace {
                config = config.with_namespace(ns);
            }
            let store = PineconeStore::new(config).map_err(vstore_err)?;

            let bridge_config = BridgeConfig {
                batch_size: batch_size.unwrap_or(100),
                max_records: max_records.unwrap_or(500_000),
                ..Default::default()
            };

            Ok(Self {
                bridge: VectorStoreBridge::new(store, bridge_config),
                rt,
            })
        }
    }

    impl_vector_bridge_methods!(PyPineconeBridge, "PineconeBridge");
}

#[cfg(feature = "pinecone")]
pub use pinecone_bridge::PyPineconeBridge;

// ── Dict conversion ──────────────────────────────────────────────────────

fn search_result_to_dict<'py>(py: Python<'py>, r: &SearchResult) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &r.id)?;
    dict.set_item("score", r.score)?;
    let meta = PyDict::new(py);
    for (k, v) in &r.metadata {
        let py_val = json_to_py(py, v)?;
        meta.set_item(k, py_val)?;
    }
    dict.set_item("metadata", meta)?;
    Ok(dict)
}

fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => pyo3::types::PyBool::new(py, *b)
            .to_owned()
            .into_any()
            .unbind(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py).unwrap().into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap_or(0.0)
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        serde_json::Value::Array(arr) => {
            let items: Vec<Py<PyAny>> = arr
                .iter()
                .map(|v| json_to_py(py, v))
                .collect::<PyResult<_>>()?;
            pyo3::types::PyList::new(py, items)?.into_any().unbind()
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into_any().unbind()
        }
    })
}
