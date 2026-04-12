use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::projection::{PyPcaProjection, extract_embedding, extract_embeddings_2d};
use crate::types::{Glob, Manifold, Nearest, Path};
use sphereql_embed::pipeline::{
    PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};

#[pyclass]
pub struct Pipeline {
    pub(crate) inner: SphereQLPipeline,
    dim: usize,
}

#[pymethods]
impl Pipeline {
    #[new]
    #[pyo3(signature = (categories, embeddings, *, projection=None))]
    fn new(
        py: Python<'_>,
        categories: Vec<String>,
        embeddings: &Bound<'_, PyAny>,
        projection: Option<&PyPcaProjection>,
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

        let inner = match projection {
            Some(pca) => {
                let pca_clone = pca.inner().clone();
                py.detach(move || SphereQLPipeline::with_projection(categories, embs, pca_clone))
            }
            None => {
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
            .map(|v| v.as_str().unwrap_or("unknown").to_string())
            .collect();

        let embeddings: Vec<Vec<f64>> = parsed["embeddings"]
            .as_array()
            .ok_or_else(|| PyValueError::new_err("missing 'embeddings' array"))?
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect()
            })
            .collect();

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

    #[pyo3(signature = (query, k=5))]
    fn nearest(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<Nearest>> {
        let emb = extract_embedding(query)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        match self.inner.query(SphereQLQuery::Nearest { k }, &pq) {
            SphereQLOutput::Nearest(items) => Ok(items.iter().map(Nearest::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (query, k=5))]
    fn nearest_json(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<String> {
        let results = self.nearest(query, k)?;
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
    fn similar_above(&self, query: &Bound<'_, PyAny>, min_cosine: f64) -> PyResult<Vec<Nearest>> {
        let emb = extract_embedding(query)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        match self
            .inner
            .query(SphereQLQuery::SimilarAbove { min_cosine }, &pq)
        {
            SphereQLOutput::KNearest(items) => Ok(items.iter().map(Nearest::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (query, min_cosine=0.8))]
    fn similar_above_json(&self, query: &Bound<'_, PyAny>, min_cosine: f64) -> PyResult<String> {
        let results = self.similar_above(query, min_cosine)?;
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
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Path>> {
        let embedding = match query {
            Some(q) => extract_embedding(q)?.values,
            None => vec![0.0; self.dim],
        };
        let pq = PipelineQuery { embedding };
        match self.inner.query(
            SphereQLQuery::ConceptPath {
                source_id,
                target_id,
                graph_k,
            },
            &pq,
        ) {
            SphereQLOutput::ConceptPath(path) => Ok(path.as_ref().map(Path::from)),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (source_id, target_id, *, graph_k=10, query=None))]
    fn concept_path_json(
        &self,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let result = self.concept_path(source_id, target_id, graph_k, query)?;
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
            None => Ok("null".to_string()),
        }
    }

    #[pyo3(signature = (*, k=None, max_k=10, query=None))]
    fn detect_globs(
        &self,
        k: Option<usize>,
        max_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Glob>> {
        let embedding = match query {
            Some(q) => extract_embedding(q)?.values,
            None => vec![0.0; self.dim],
        };
        let pq = PipelineQuery { embedding };
        match self
            .inner
            .query(SphereQLQuery::DetectGlobs { k, max_k }, &pq)
        {
            SphereQLOutput::Globs(globs) => Ok(globs.iter().map(Glob::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (*, k=None, max_k=10, query=None))]
    fn detect_globs_json(
        &self,
        k: Option<usize>,
        max_k: usize,
        query: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let results = self.detect_globs(k, max_k, query)?;
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
        query: &Bound<'_, PyAny>,
        neighborhood_k: usize,
    ) -> PyResult<Manifold> {
        let emb = extract_embedding(query)?;
        let pq = PipelineQuery {
            embedding: emb.values,
        };
        match self
            .inner
            .query(SphereQLQuery::LocalManifold { neighborhood_k }, &pq)
        {
            SphereQLOutput::LocalManifold(m) => Ok(Manifold::from(&m)),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    #[pyo3(signature = (query, *, neighborhood_k=10))]
    fn local_manifold_json(
        &self,
        query: &Bound<'_, PyAny>,
        neighborhood_k: usize,
    ) -> PyResult<String> {
        let m = self.local_manifold(query, neighborhood_k)?;
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
}
