//! Python bindings for the metalearning surface: corpus features,
//! auto-tuner, meta-model. Kept separate from `pipeline.rs` because
//! these live at module level rather than on the [`Pipeline`] class.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use sphereql_embed::config::PipelineConfig;
use sphereql_embed::corpus_features::CorpusFeatures;
use sphereql_embed::feedback::{FeedbackAggregator, FeedbackEvent};
use sphereql_embed::meta_model::{
    DistanceWeightedMetaModel, MetaModel, MetaTrainingRecord, NearestNeighborMetaModel,
};
use sphereql_embed::pipeline::PipelineInput;
use sphereql_embed::quality_metric::{
    BridgeCoherence, ClusterSilhouette, CompositeMetric, GraphModularity, QualityMetric,
    TerritorialHealth,
};
use sphereql_embed::tuner::{SearchSpace, SearchStrategy, auto_tune as rust_auto_tune};

use crate::pipeline::Pipeline;
use crate::projection::extract_embeddings_2d;

/// Turn a user-supplied metric name into a boxed `QualityMetric`.
///
/// One allocation per tuner run — negligible next to the actual search.
/// The `auto_tune<M: QualityMetric + ?Sized>` bound lets us hand a
/// `&dyn QualityMetric` straight in, so we skip the six-arm match that
/// used to duplicate the `rust_auto_tune(…)` call site.
///
/// The `Send + Sync` markers on the returned trait object are required
/// so the box can cross a `py.detach()` closure boundary — every
/// concrete `QualityMetric` in this codebase is thread-safe, so the
/// bound is free.
fn resolve_metric(name: &str) -> PyResult<Box<dyn QualityMetric + Send + Sync>> {
    match name {
        "territorial_health" => Ok(Box::new(TerritorialHealth)),
        "bridge_coherence" => Ok(Box::new(BridgeCoherence)),
        "cluster_silhouette" => Ok(Box::new(ClusterSilhouette)),
        "graph_modularity" => Ok(Box::new(GraphModularity::default())),
        "default_composite" => Ok(Box::new(CompositeMetric::default_composite())),
        "connectivity_composite" => Ok(Box::new(CompositeMetric::connectivity_composite())),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?}; expected one of: \
             territorial_health, bridge_coherence, cluster_silhouette, \
             graph_modularity, default_composite, connectivity_composite"
        ))),
    }
}

fn resolve_strategy(
    kind: &str,
    budget: usize,
    seed: u64,
    warmup: usize,
    gamma: f64,
) -> PyResult<SearchStrategy> {
    match kind {
        "grid" => Ok(SearchStrategy::Grid),
        "random" => Ok(SearchStrategy::Random { budget, seed }),
        "bayesian" => Ok(SearchStrategy::Bayesian {
            budget,
            warmup,
            gamma,
            seed,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown strategy {other:?}; expected one of: grid, random, bayesian"
        ))),
    }
}

/// Run the auto-tuner over a default [`SearchSpace`] and return the
/// best-scoring pipeline alongside a summary report.
///
/// Args:
///     categories: Category label per embedding.
///     embeddings: List of embedding vectors.
///     metric: Quality metric name. One of `territorial_health`,
///         `bridge_coherence`, `cluster_silhouette`, `graph_modularity`,
///         `default_composite`, `connectivity_composite`.
///         Default: `default_composite`.
///     strategy: One of `grid`, `random`, `bayesian`. Default: `random`.
///     budget: Number of trials for `random` / `bayesian`. Default: 24.
///     seed: RNG seed for `random` / `bayesian`. Default: 0.
///     warmup: Warmup trials before acquisition kicks in (Bayesian only).
///         Default: 4.
///     gamma: Fraction of trials treated as "good" (Bayesian only).
///         Default: 0.25.
///     base_config: Optional dict of PipelineConfig. Non-tuned knobs
///         stay at these values; tuned knobs explore the search space
///         starting from them. Default: PipelineConfig.default().
///     search_space: Optional dict of SearchSpace. Any field may be
///         omitted — missing fields fall back to SearchSpace.default().
///         Lists override the default candidate values for that knob.
///
/// Returns:
///     Tuple `(Pipeline, report)` where `report` is a dict with
///     `metric_name`, `best_score`, `best_config`, `trials_count`,
///     `failures_count`, `mean_score`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    categories,
    embeddings,
    *,
    metric = "default_composite",
    strategy = "random",
    budget = 24,
    seed = 0,
    warmup = 4,
    gamma = 0.25,
    base_config = None,
    search_space = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn auto_tune<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    embeddings: &Bound<'_, PyAny>,
    metric: &str,
    strategy: &str,
    budget: usize,
    seed: u64,
    warmup: usize,
    gamma: f64,
    base_config: Option<&Bound<'_, PyAny>>,
    search_space: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Pipeline, Bound<'py, PyAny>)> {
    let embs = extract_embeddings_2d(embeddings)?;
    if categories.len() != embs.len() {
        return Err(PyValueError::new_err(format!(
            "categories length ({}) != embeddings length ({})",
            categories.len(),
            embs.len()
        )));
    }
    let dim = embs.first().map(|e| e.dimension()).unwrap_or(0);
    let raw: Vec<Vec<f64>> = embs.iter().map(|e| e.values.clone()).collect();

    let base = match base_config {
        Some(obj) => pythonize::depythonize::<PipelineConfig>(obj)
            .map_err(|e| PyValueError::new_err(format!("invalid base_config dict: {e}")))?,
        None => PipelineConfig::default(),
    };

    let metric_obj = resolve_metric(metric)?;
    let search_strategy = resolve_strategy(strategy, budget, seed, warmup, gamma)?;
    let space = match search_space {
        Some(obj) => pythonize::depythonize::<SearchSpace>(obj)
            .map_err(|e| PyValueError::new_err(format!("invalid search_space dict: {e}")))?,
        None => SearchSpace::default(),
    };

    let (pipeline, report) = py
        .detach(move || {
            rust_auto_tune(
                PipelineInput {
                    categories,
                    embeddings: raw,
                },
                &space,
                metric_obj.as_ref(),
                search_strategy,
                &base,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let report_dict = PyDict::new(py);
    report_dict.set_item("metric_name", &report.metric_name)?;
    report_dict.set_item("best_score", report.best_score)?;
    report_dict.set_item(
        "best_config",
        pythonize::pythonize(py, &report.best_config)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize best_config: {e}")))?,
    )?;
    report_dict.set_item("trials_count", report.trials.len())?;
    report_dict.set_item("failures_count", report.failures.len())?;
    report_dict.set_item("mean_score", report.mean_score())?;

    Ok((Pipeline::from_inner(pipeline, dim), report_dict.into_any()))
}

/// Extract a corpus feature profile.
///
/// Returns a dict matching [`CorpusFeatures`]'s serde representation:
/// `n_items`, `n_categories`, `dim`, `mean_members_per_category`,
/// `category_size_entropy`, `mean_sparsity`, `axis_utilization_entropy`,
/// `noise_estimate`, `mean_intra_category_similarity`,
/// `mean_inter_category_similarity`, `category_separation_ratio`.
///
/// This is the input any [`MetaModel`] uses to predict a `PipelineConfig`
/// for a previously-unseen corpus.
///
/// Args:
///     categories: Category label per embedding.
///     embeddings: List of embedding vectors (list[list[float]] or np.ndarray).
///
/// Returns:
///     dict of corpus features.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (categories, embeddings))]
pub fn corpus_features<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    embeddings: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let embs = extract_embeddings_2d(embeddings)?;
    if categories.len() != embs.len() {
        return Err(PyValueError::new_err(format!(
            "categories length ({}) != embeddings length ({})",
            categories.len(),
            embs.len()
        )));
    }
    let raw: Vec<Vec<f64>> = embs.iter().map(|e| e.values.clone()).collect();
    let features = py.detach(|| CorpusFeatures::extract(&categories, &raw));
    pythonize::pythonize(py, &features)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize CorpusFeatures: {e}")))
}

// ── MetaModel ────────────────────────────────────────────────────────

fn depythonize_records(records: &Bound<'_, PyAny>) -> PyResult<Vec<MetaTrainingRecord>> {
    pythonize::depythonize(records)
        .map_err(|e| PyValueError::new_err(format!("invalid training records: {e}")))
}

fn depythonize_features(features: &Bound<'_, PyAny>) -> PyResult<CorpusFeatures> {
    pythonize::depythonize(features)
        .map_err(|e| PyValueError::new_err(format!("invalid CorpusFeatures dict: {e}")))
}

/// Nearest-neighbor meta-model: picks the training record whose corpus
/// feature vector is closest to the query (z-score normalized Euclidean).
#[gen_stub_pyclass]
#[pyclass(name = "NearestNeighborMetaModel")]
#[derive(Default)]
pub struct PyNearestNeighborMetaModel {
    inner: NearestNeighborMetaModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNearestNeighborMetaModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: NearestNeighborMetaModel::new(),
        }
    }

    /// Fit on a list of training records (dicts with `corpus_id`,
    /// `features`, `best_config`, `best_score`, `metric_name`,
    /// `strategy`, `timestamp`).
    fn fit(&mut self, records: &Bound<'_, PyAny>) -> PyResult<()> {
        let recs = depythonize_records(records)?;
        self.inner.fit(&recs);
        Ok(())
    }

    /// Predict the PipelineConfig for a new corpus profile. Returns a
    /// dict ready to pass to `Pipeline(categories, embeddings, config=...)`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let feats = depythonize_features(features)?;
        let cfg = self.inner.predict(&feats);
        pythonize::pythonize(py, &cfg)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize config: {e}")))
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "NearestNeighborMetaModel(records={})",
            self.inner.records().len()
        )
    }
}

/// Distance-weighted meta-model: picks the training record that
/// maximizes `best_score × 1/(distance + epsilon)`, balancing similarity
/// and observed quality.
#[gen_stub_pyclass]
#[pyclass(name = "DistanceWeightedMetaModel")]
#[derive(Default)]
pub struct PyDistanceWeightedMetaModel {
    inner: DistanceWeightedMetaModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDistanceWeightedMetaModel {
    #[new]
    #[pyo3(signature = (*, epsilon = 0.1))]
    fn new(epsilon: f64) -> Self {
        Self {
            inner: DistanceWeightedMetaModel::new().with_epsilon(epsilon),
        }
    }

    fn fit(&mut self, records: &Bound<'_, PyAny>) -> PyResult<()> {
        let recs = depythonize_records(records)?;
        self.inner.fit(&recs);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let feats = depythonize_features(features)?;
        let cfg = self.inner.predict(&feats);
        pythonize::pythonize(py, &cfg)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize config: {e}")))
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "DistanceWeightedMetaModel(records={})",
            self.inner.records().len()
        )
    }
}

// ── Record store ──────────────────────────────────────────────────────

/// Load all training records from the default store at
/// `~/.sphereql/meta_records.json`. Returns an empty list if the
/// store does not exist yet.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn load_default_store<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let records = MetaTrainingRecord::load_default_store()
        .map_err(|e| PyValueError::new_err(format!("failed to load default store: {e}")))?;
    pythonize::pythonize(py, &records)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize records: {e}")))
}

/// Append one training record to the default store. Returns the absolute
/// path of the store file.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn append_to_default_store(record: &Bound<'_, PyAny>) -> PyResult<String> {
    let rec: MetaTrainingRecord = pythonize::depythonize(record)
        .map_err(|e| PyValueError::new_err(format!("invalid record dict: {e}")))?;
    let path = rec
        .append_to_default_store()
        .map_err(|e| PyValueError::new_err(format!("failed to append record: {e}")))?;
    Ok(path.to_string_lossy().to_string())
}

// ── Feedback primitives (L3 metalearning) ────────────────────────────

/// One user-supplied satisfaction signal attached to a specific query.
///
/// `score` is a scalar in `[0, 1]`; the aggregator clamps it at
/// summarize time, so stored raw values are preserved.
#[gen_stub_pyclass]
#[pyclass(name = "FeedbackEvent", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFeedbackEvent {
    pub(crate) inner: FeedbackEvent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFeedbackEvent {
    #[new]
    #[pyo3(signature = (corpus_id, query_id, score, timestamp=None))]
    fn new(corpus_id: String, query_id: String, score: f64, timestamp: Option<String>) -> Self {
        let event = match timestamp {
            Some(ts) => FeedbackEvent {
                corpus_id,
                query_id,
                score,
                timestamp: ts,
            },
            None => FeedbackEvent::now(corpus_id, query_id, score),
        };
        Self { inner: event }
    }

    #[getter]
    fn corpus_id(&self) -> &str {
        &self.inner.corpus_id
    }

    #[getter]
    fn query_id(&self) -> &str {
        &self.inner.query_id
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    fn timestamp(&self) -> &str {
        &self.inner.timestamp
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize::pythonize(py, &self.inner)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize event: {e}")))
    }

    /// Append this event to `~/.sphereql/feedback_events.json`.
    /// Returns the absolute path of the store file.
    fn append_to_default_store(&self) -> PyResult<String> {
        let path = self
            .inner
            .append_to_default_store()
            .map_err(|e| PyValueError::new_err(format!("failed to append event: {e}")))?;
        Ok(path.to_string_lossy().to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "FeedbackEvent(corpus_id={:?}, query_id={:?}, score={:.4})",
            self.inner.corpus_id, self.inner.query_id, self.inner.score
        )
    }
}

/// Accumulates [`FeedbackEvent`]s and summarizes them by `corpus_id`.
#[gen_stub_pyclass]
#[pyclass(name = "FeedbackAggregator")]
#[derive(Default)]
pub struct PyFeedbackAggregator {
    inner: FeedbackAggregator,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFeedbackAggregator {
    #[new]
    fn new() -> Self {
        Self {
            inner: FeedbackAggregator::new(),
        }
    }

    /// Load an aggregator from a JSON file, or return an empty one if
    /// the file does not exist.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = FeedbackAggregator::load(path)
            .map_err(|e| PyValueError::new_err(format!("failed to load: {e}")))?;
        Ok(Self { inner })
    }

    /// Load the default store at `~/.sphereql/feedback_events.json`.
    #[staticmethod]
    fn load_default() -> PyResult<Self> {
        let inner = FeedbackAggregator::load_default_store()
            .map_err(|e| PyValueError::new_err(format!("failed to load default store: {e}")))?;
        Ok(Self { inner })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyValueError::new_err(format!("failed to save: {e}")))
    }

    fn record(&mut self, event: &PyFeedbackEvent) {
        self.inner.record(event.inner.clone());
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }

    fn corpus_ids(&self) -> Vec<String> {
        self.inner.corpus_ids()
    }

    /// Summarize feedback for one corpus. Returns None if no events.
    fn summarize<'py>(
        &self,
        py: Python<'py>,
        corpus_id: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.inner.summarize(corpus_id) {
            Some(s) => {
                let d = pythonize::pythonize(py, &s).map_err(|e| {
                    PyValueError::new_err(format!("failed to serialize summary: {e}"))
                })?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }

    /// Summarize every corpus that has events.
    fn summarize_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let all = self.inner.summarize_all();
        pythonize::pythonize(py, &all)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize summaries: {e}")))
    }

    fn __repr__(&self) -> String {
        format!("FeedbackAggregator(events={})", self.inner.len())
    }
}
