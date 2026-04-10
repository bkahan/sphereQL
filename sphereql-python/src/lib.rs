use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use sphereql_embed::pipeline::{
    GlobSummary, ManifoldResult, NearestResult, PathResult, PipelineInput, PipelineQuery,
    SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};

// ── Python result types ────────────────────────────────────────────────

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
struct Nearest {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    category: String,
    #[pyo3(get)]
    distance: f64,
    #[pyo3(get)]
    certainty: f64,
    #[pyo3(get)]
    intensity: f64,
}

#[pymethods]
impl Nearest {
    fn __repr__(&self) -> String {
        format!(
            "Nearest(id={:?}, category={:?}, distance={:.4})",
            self.id, self.category, self.distance
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "id": self.id,
            "category": self.category,
            "distance": self.distance,
            "certainty": self.certainty,
            "intensity": self.intensity,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl From<&NearestResult> for Nearest {
    fn from(r: &NearestResult) -> Self {
        Self {
            id: r.id.clone(),
            category: r.category.clone(),
            distance: r.distance,
            certainty: r.certainty,
            intensity: r.intensity,
        }
    }
}

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
struct PathStep {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    category: String,
    #[pyo3(get)]
    cumulative_distance: f64,
}

#[pymethods]
impl PathStep {
    fn __repr__(&self) -> String {
        format!(
            "PathStep(id={:?}, category={:?}, cumulative_distance={:.4})",
            self.id, self.category, self.cumulative_distance
        )
    }
}

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
struct Path {
    #[pyo3(get)]
    total_distance: f64,
    #[pyo3(get)]
    steps: Vec<PathStep>,
}

#[pymethods]
impl Path {
    fn __repr__(&self) -> String {
        format!(
            "Path(steps={}, total_distance={:.4})",
            self.steps.len(),
            self.total_distance
        )
    }

    fn to_json(&self) -> PyResult<String> {
        let steps: Vec<_> = self
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
            "total_distance": self.total_distance,
            "steps": steps,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl From<&PathResult> for Path {
    fn from(p: &PathResult) -> Self {
        Self {
            total_distance: p.total_distance,
            steps: p
                .steps
                .iter()
                .map(|s| PathStep {
                    id: s.id.clone(),
                    category: s.category.clone(),
                    cumulative_distance: s.cumulative_distance,
                })
                .collect(),
        }
    }
}

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
struct Glob {
    #[pyo3(get)]
    id: usize,
    #[pyo3(get)]
    centroid: [f64; 3],
    #[pyo3(get)]
    member_count: usize,
    #[pyo3(get)]
    radius: f64,
    #[pyo3(get)]
    top_categories: Vec<(String, usize)>,
}

#[pymethods]
impl Glob {
    fn __repr__(&self) -> String {
        format!(
            "Glob(id={}, members={}, radius={:.4})",
            self.id, self.member_count, self.radius
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "id": self.id,
            "centroid": self.centroid,
            "member_count": self.member_count,
            "radius": self.radius,
            "top_categories": self.top_categories,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl From<&GlobSummary> for Glob {
    fn from(g: &GlobSummary) -> Self {
        Self {
            id: g.id,
            centroid: g.centroid,
            member_count: g.member_count,
            radius: g.radius,
            top_categories: g.top_categories.clone(),
        }
    }
}

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
struct Manifold {
    #[pyo3(get)]
    centroid: [f64; 3],
    #[pyo3(get)]
    normal: [f64; 3],
    #[pyo3(get)]
    variance_ratio: f64,
}

#[pymethods]
impl Manifold {
    fn __repr__(&self) -> String {
        format!("Manifold(variance_ratio={:.4})", self.variance_ratio)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "centroid": self.centroid,
            "normal": self.normal,
            "variance_ratio": self.variance_ratio,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl From<&ManifoldResult> for Manifold {
    fn from(m: &ManifoldResult) -> Self {
        Self {
            centroid: m.centroid,
            normal: m.normal,
            variance_ratio: m.variance_ratio,
        }
    }
}

// ── Pipeline ───────────────────────────────────────────────────────────

#[pyclass]
struct Pipeline {
    inner: SphereQLPipeline,
}

#[pymethods]
impl Pipeline {
    #[new]
    fn new(categories: Vec<String>, embeddings: Vec<Vec<f64>>) -> PyResult<Self> {
        if categories.len() != embeddings.len() {
            return Err(PyValueError::new_err(format!(
                "categories length ({}) != embeddings length ({})",
                categories.len(),
                embeddings.len()
            )));
        }
        Ok(Self {
            inner: SphereQLPipeline::new(PipelineInput {
                categories,
                embeddings,
            }),
        })
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

        Self::new(categories, embeddings)
    }

    fn nearest(&self, query: Vec<f64>, k: usize) -> PyResult<Vec<Nearest>> {
        let pq = PipelineQuery { embedding: query };
        match self.inner.query(SphereQLQuery::Nearest { k }, &pq) {
            SphereQLOutput::Nearest(items) => Ok(items.iter().map(Nearest::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    fn nearest_json(&self, query: Vec<f64>, k: usize) -> PyResult<String> {
        let results = self.nearest(query, k)?;
        let out: Vec<_> = results.iter().map(|r| {
            serde_json::json!({
                "id": r.id, "category": r.category,
                "distance": r.distance, "certainty": r.certainty,
                "intensity": r.intensity,
            })
        }).collect();
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn similar_above(&self, query: Vec<f64>, min_cosine: f64) -> PyResult<Vec<Nearest>> {
        let pq = PipelineQuery { embedding: query };
        match self.inner.query(SphereQLQuery::SimilarAbove { min_cosine }, &pq) {
            SphereQLOutput::KNearest(items) => Ok(items.iter().map(Nearest::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    fn similar_above_json(&self, query: Vec<f64>, min_cosine: f64) -> PyResult<String> {
        let results = self.similar_above(query, min_cosine)?;
        let out: Vec<_> = results.iter().map(|r| {
            serde_json::json!({
                "id": r.id, "category": r.category,
                "distance": r.distance, "certainty": r.certainty,
                "intensity": r.intensity,
            })
        }).collect();
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn concept_path(
        &self,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query: Vec<f64>,
    ) -> PyResult<Option<Path>> {
        let pq = PipelineQuery { embedding: query };
        match self.inner.query(
            SphereQLQuery::ConceptPath { source_id, target_id, graph_k },
            &pq,
        ) {
            SphereQLOutput::ConceptPath(path) => Ok(path.as_ref().map(Path::from)),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    fn concept_path_json(
        &self,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query: Vec<f64>,
    ) -> PyResult<String> {
        let result = self.concept_path(source_id, target_id, graph_k, query)?;
        match result {
            Some(p) => p.to_json(),
            None => Ok("null".to_string()),
        }
    }

    fn detect_globs(
        &self,
        query: Vec<f64>,
        k: Option<usize>,
        max_k: usize,
    ) -> PyResult<Vec<Glob>> {
        let pq = PipelineQuery { embedding: query };
        match self.inner.query(SphereQLQuery::DetectGlobs { k, max_k }, &pq) {
            SphereQLOutput::Globs(globs) => Ok(globs.iter().map(Glob::from).collect()),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    fn detect_globs_json(
        &self,
        query: Vec<f64>,
        k: Option<usize>,
        max_k: usize,
    ) -> PyResult<String> {
        let results = self.detect_globs(query, k, max_k)?;
        let out: Vec<_> = results.iter().map(|g| {
            serde_json::json!({
                "id": g.id, "centroid": g.centroid,
                "member_count": g.member_count, "radius": g.radius,
                "top_categories": g.top_categories,
            })
        }).collect();
        serde_json::to_string(&out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn local_manifold(&self, query: Vec<f64>, neighborhood_k: usize) -> PyResult<Manifold> {
        let pq = PipelineQuery { embedding: query };
        match self.inner.query(SphereQLQuery::LocalManifold { neighborhood_k }, &pq) {
            SphereQLOutput::LocalManifold(m) => Ok(Manifold::from(&m)),
            _ => Err(PyValueError::new_err("unexpected output type")),
        }
    }

    fn local_manifold_json(&self, query: Vec<f64>, neighborhood_k: usize) -> PyResult<String> {
        self.local_manifold(query, neighborhood_k)?.to_json()
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

// ── Module ─────────────────────────────────────────────────────────────

#[pymodule]
fn sphereql(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    m.add_class::<Nearest>()?;
    m.add_class::<Path>()?;
    m.add_class::<PathStep>()?;
    m.add_class::<Glob>()?;
    m.add_class::<Manifold>()?;
    Ok(())
}
