use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::projection::{PyPcaProjection, extract_embedding, extract_embeddings_2d};
use crate::types::{
    Glob, Manifold, Nearest, Path, PyCategoryPath, PyCategorySummary, PyDomainGroup, PyDrillDown,
    PyInnerSphereReport, PyProjectionWarning,
};
use sphereql_embed::config::PipelineConfig;
use sphereql_embed::pipeline::{
    PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};
use sphereql_embed::types::Embedding;

fn extract_config(config: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PipelineConfig>> {
    match config {
        None => Ok(None),
        Some(obj) => {
            let cfg: PipelineConfig = pythonize::depythonize(obj).map_err(|e| {
                PyValueError::new_err(format!("invalid PipelineConfig dict: {e}"))
            })?;
            Ok(Some(cfg))
        }
    }
}

#[pyclass]
pub struct Pipeline {
    pub(crate) inner: SphereQLPipeline,
    pub(crate) dim: usize,
}

impl Pipeline {
    /// Construct a Pipeline wrapper around an existing SphereQLPipeline.
    /// Used by the auto-tuner and meta-model entry points.
    pub(crate) fn from_inner(inner: SphereQLPipeline, dim: usize) -> Self {
        Self { inner, dim }
    }

    /// Reject queries whose dimensionality doesn't match the corpus.
    ///
    /// Every `PcaProjection`/`KernelPcaProjection`/`LaplacianEigenmap`
    /// `project` method calls `assert_eq!` on the input length, so a
    /// wrong-dim query turns into a `PanicException` surfaced to
    /// Python — terrible UX. This helper catches it at the binding
    /// boundary and returns a clean `PyValueError` instead.
    fn check_query_dim(&self, values: &[f64]) -> PyResult<()> {
        if values.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dim,
                values.len()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Pipeline {
    #[new]
    #[pyo3(signature = (categories, embeddings, *, projection=None, config=None))]
    fn new(
        py: Python<'_>,
        categories: Vec<String>,
        embeddings: &Bound<'_, PyAny>,
        projection: Option<&PyPcaProjection>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let embs = extract_embeddings_2d(embeddings)?;

        if categories.len() != embs.len() {
            return Err(PyValueError::new_err(format!(
                "categories length ({}) != embeddings length ({})",
                categories.len(),
                embs.len()
            )));
        }

        let dim = embs.first().map(|e| e.dimension()).unwrap_or(0);
        let cfg = extract_config(config)?;

        let inner = match (projection, cfg) {
            (Some(pca), Some(config)) => {
                let pca_clone = pca.inner().clone();
                py.detach(move || {
                    SphereQLPipeline::with_projection_and_config(
                        categories, embs, pca_clone, config,
                    )
                })
            }
            (Some(pca), None) => {
                let pca_clone = pca.inner().clone();
                py.detach(move || SphereQLPipeline::with_projection(categories, embs, pca_clone))
            }
            (None, Some(config)) => {
                let raw: Vec<Vec<f64>> = embs.into_iter().map(|e| e.values.clone()).collect();
                py.detach(move || {
                    SphereQLPipeline::new_with_config(
                        PipelineInput {
                            categories,
                            embeddings: raw,
                        },
                        config,
                    )
                })
            }
            (None, None) => {
                let raw: Vec<Vec<f64>> = embs.into_iter().map(|e| e.values.clone()).collect();
                py.detach(move || {
                    SphereQLPipeline::new(PipelineInput {
                        categories,
                        embeddings: raw,
                    })
                })
            }
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner, dim })
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let categories: Vec<String> = parsed["categories"]
            .as_array()
            .ok_or_else(|| PyValueError::new_err("missing 'categories' array"))?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_str()
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("category at index {i} must be a string"))
                    })
                    .map(|s| s.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let embeddings: Vec<Vec<f64>> = parsed["embeddings"]
            .as_array()
            .ok_or_else(|| PyValueError::new_err("missing 'embeddings' array"))?
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let arr = row.as_array().ok_or_else(|| {
                    PyValueError::new_err(format!("embedding at index {i} must be an array"))
                })?;
                arr.iter()
                    .enumerate()
                    .map(|(j, v)| {
                        v.as_f64().ok_or_else(|| {
                            PyValueError::new_err(format!("embedding[{i}][{j}] must be a number"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
        let inner = SphereQLPipeline::new(PipelineInput {
            categories,
            embeddings,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner, dim })
    }

    #[getter]
    fn num_items(&self) -> usize {
        self.inner.num_items()
    }

    #[getter]
    fn categories(&self) -> Vec<String> {
        self.inner.categories().to_vec()
    }

    /// Return the [`PipelineConfig`] this pipeline was built with, as a dict.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize::pythonize(py, self.inner.config())
            .map_err(|e| PyValueError::new_err(format!("failed to serialize config: {e}")))
    }

    /// Short stable name of the projection family — "pca", "kernel_pca",
    /// or "laplacian_eigenmap".
    #[getter]
    fn projection_kind(&self) -> &'static str {
        self.inner.projection_kind().name()
    }

    #[pyo3(signature = (query, k=5))]
    fn nearest(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<Vec<Nearest>> {
        let emb = extract_embedding(query)?;
        self.check_query_dim(&emb.values)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        let result = py.detach(|| self.inner.query(SphereQLQuery::Nearest { k }, &pq));
        match result {
            SphereQLOutput::Nearest(items) => Ok(items.iter().map(Nearest::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (query, k=5))]
    fn nearest_json(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<String> {
        let results = self.nearest(py, query, k)?;
        let out: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id, "category": r.category,
                    "distance": r.distance, "certainty": r.certainty,
                    "intensity": r.intensity,
                })
            })
            .collect();
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (query, min_cosine=0.8))]
    fn similar_above(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        min_cosine: f64,
    ) -> PyResult<Vec<Nearest>> {
        let emb = extract_embedding(query)?;
        self.check_query_dim(&emb.values)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        let result = py.detach(|| {
            self.inner
                .query(SphereQLQuery::SimilarAbove { min_cosine }, &pq)
        });
        match result {
            SphereQLOutput::KNearest(items) => Ok(items.iter().map(Nearest::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (query, min_cosine=0.8))]
    fn similar_above_json(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        min_cosine: f64,
    ) -> PyResult<String> {
        let results = self.similar_above(py, query, min_cosine)?;
        let out: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id, "category": r.category,
                    "distance": r.distance, "certainty": r.certainty,
                    "intensity": r.intensity,
                })
            })
            .collect();
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (source_id, target_id, *, graph_k=10, query=None))]
    fn concept_path(
        &self,
        py: Python<'_>,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Path>> {
        let embedding = match query {
            Some(q) => {
                let e = extract_embedding(q)?.values;
                self.check_query_dim(&e)?;
                e
            }
            None => vec![0.0; self.dim],
        };
        let pq = PipelineQuery { embedding };
        let result = py.detach(|| {
            self.inner.query(
                SphereQLQuery::ConceptPath {
                    source_id,
                    target_id,
                    graph_k,
                },
                &pq,
            )
        });
        match result {
            SphereQLOutput::ConceptPath(path) => Ok(path.as_ref().map(Path::from)),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (source_id, target_id, *, graph_k=10, query=None))]
    fn concept_path_json(
        &self,
        py: Python<'_>,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let result = self.concept_path(py, source_id, target_id, graph_k, query)?;
        match result {
            Some(p) => {
                let steps: Vec<_> = p
                    .steps
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "id": s.id,
                            "category": s.category,
                            "cumulative_distance": s.cumulative_distance,
                        })
                    })
                    .collect();
                serde_json::to_string(&serde_json::json!({
                    "total_distance": p.total_distance,
                    "steps": steps,
                }))
                .map_err(|e| PyValueError::new_err(e.to_string()))
            }
            // `null` literal goes through serde to avoid a magic string if
            // the JSON shape ever changes.
            None => serde_json::to_string(&serde_json::Value::Null)
                .map_err(|e| PyValueError::new_err(e.to_string())),
        }
    }

    #[pyo3(signature = (*, k=None, max_k=10, query=None))]
    fn detect_globs(
        &self,
        py: Python<'_>,
        k: Option<usize>,
        max_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Glob>> {
        let embedding = match query {
            Some(q) => {
                let e = extract_embedding(q)?.values;
                self.check_query_dim(&e)?;
                e
            }
            None => vec![0.0; self.dim],
        };
        let pq = PipelineQuery { embedding };
        let result = py.detach(|| {
            self.inner
                .query(SphereQLQuery::DetectGlobs { k, max_k }, &pq)
        });
        match result {
            SphereQLOutput::Globs(globs) => Ok(globs.iter().map(Glob::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (*, k=None, max_k=10, query=None))]
    fn detect_globs_json(
        &self,
        py: Python<'_>,
        k: Option<usize>,
        max_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let results = self.detect_globs(py, k, max_k, query)?;
        let out: Vec<_> = results
            .iter()
            .map(|g| {
                serde_json::json!({
                    "id": g.id, "centroid": g.centroid,
                    "member_count": g.member_count, "radius": g.radius,
                    "top_categories": g.top_categories,
                })
            })
            .collect();
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (query, *, neighborhood_k=10))]
    fn local_manifold(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        neighborhood_k: usize,
    ) -> PyResult<Manifold> {
        let emb = extract_embedding(query)?;
        self.check_query_dim(&emb.values)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        let result = py.detach(|| {
            self.inner
                .query(SphereQLQuery::LocalManifold { neighborhood_k }, &pq)
        });
        match result {
            SphereQLOutput::LocalManifold(m) => Ok(Manifold::from(&m)),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (query, *, neighborhood_k=10))]
    fn local_manifold_json(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        neighborhood_k: usize,
    ) -> PyResult<String> {
        let m = self.local_manifold(py, query, neighborhood_k)?;
        serde_json::to_string(&serde_json::json!({
            "centroid": m.centroid,
            "normal": m.normal,
            "variance_ratio": m.variance_ratio,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn exported_points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let points = self.inner.exported_points();
        let list = pyo3::types::PyList::empty(py);
        for p in &points {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", &p.id)?;
            dict.set_item("category", &p.category)?;
            dict.set_item("r", p.r)?;
            dict.set_item("theta", p.theta)?;
            dict.set_item("phi", p.phi)?;
            dict.set_item("x", p.x)?;
            dict.set_item("y", p.y)?;
            dict.set_item("z", p.z)?;
            dict.set_item("certainty", p.certainty)?;
            dict.set_item("intensity", p.intensity)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    #[getter]
    fn explained_variance_ratio(&self) -> f64 {
        self.inner.explained_variance_ratio()
    }

    fn unique_categories(&self) -> Vec<String> {
        self.inner.unique_categories()
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    fn to_csv(&self) -> String {
        self.inner.to_csv()
    }

    fn __len__(&self) -> usize {
        self.inner.num_items()
    }

    fn __bool__(&self) -> bool {
        self.inner.num_items() > 0
    }

    fn __repr__(&self) -> String {
        format!("Pipeline(items={})", self.inner.num_items())
    }

    // ── Category Enrichment Layer ──────────────────────────────────────

    #[pyo3(signature = (source_category, target_category))]
    fn category_concept_path(
        &self,
        py: Python<'_>,
        source_category: &str,
        target_category: &str,
    ) -> PyResult<Option<PyCategoryPath>> {
        let pq = PipelineQuery {
            embedding: vec![0.0; self.dim],
        };
        let result = py.detach(|| {
            self.inner.query(
                SphereQLQuery::CategoryConceptPath {
                    source_category,
                    target_category,
                },
                &pq,
            )
        });
        match result {
            SphereQLOutput::CategoryConceptPath(path) => {
                Ok(path.as_ref().map(PyCategoryPath::from))
            }
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (category, k=5))]
    fn category_neighbors(
        &self,
        py: Python<'_>,
        category: &str,
        k: usize,
    ) -> PyResult<Vec<PyCategorySummary>> {
        let pq = PipelineQuery {
            embedding: vec![0.0; self.dim],
        };
        let result = py.detach(|| {
            self.inner
                .query(SphereQLQuery::CategoryNeighbors { category, k }, &pq)
        });
        match result {
            SphereQLOutput::CategoryNeighbors(summaries) => {
                Ok(summaries.iter().map(PyCategorySummary::from).collect())
            }
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (category, query, k=10))]
    fn drill_down(
        &self,
        py: Python<'_>,
        category: &str,
        query: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<Vec<PyDrillDown>> {
        let emb = extract_embedding(query)?;
        self.check_query_dim(&emb.values)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        let result = py.detach(|| {
            self.inner
                .query(SphereQLQuery::DrillDown { category, k }, &pq)
        });
        match result {
            SphereQLOutput::DrillDown(results) => {
                Ok(results.iter().map(PyDrillDown::from).collect())
            }
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    // ── Hierarchical routing (Phase 5) ─────────────────────────────────

    /// Hierarchical nearest-neighbor search. Falls back to plain
    /// [`nearest`](Self::nearest) when EVR is above
    /// `config.routing.low_evr_threshold`; otherwise routes the query
    /// to a domain group and drills into its member categories.
    #[pyo3(signature = (query, k=5))]
    fn hierarchical_nearest(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<Vec<Nearest>> {
        let emb = extract_embedding(query)?;
        self.check_query_dim(&emb.values)?;
        let embedding = Embedding::new(emb.values);
        let results = py.detach(|| self.inner.hierarchical_nearest(&embedding, k));
        Ok(results.iter().map(Nearest::from).collect())
    }

    /// Coarse domain groups detected from category geometry.
    fn domain_groups(&self) -> Vec<PyDomainGroup> {
        self.inner
            .domain_groups()
            .iter()
            .map(PyDomainGroup::from)
            .collect()
    }

    /// Structured warnings when projection quality (EVR) is below the
    /// configured threshold. Empty when the projection is healthy.
    fn projection_warnings(&self) -> Vec<PyProjectionWarning> {
        self.inner
            .projection_warnings()
            .iter()
            .map(PyProjectionWarning::from)
            .collect()
    }

    fn category_stats(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Vec<PyCategorySummary>, Vec<PyInnerSphereReport>)> {
        let pq = PipelineQuery {
            embedding: vec![0.0; self.dim],
        };
        let result = py.detach(|| self.inner.query(SphereQLQuery::CategoryStats, &pq));
        match result {
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
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }
}
