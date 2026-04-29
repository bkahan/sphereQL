use crate::core_types::PySphericalPoint;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "LinguaPipeline")]
pub struct PyLinguaPipeline {
    inner: sphereql_lingua::LinguaPipeline,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLinguaPipeline {
    #[new]
    fn new() -> Self {
        Self {
            inner: sphereql_lingua::LinguaPipeline::new(),
        }
    }

    fn process(&self, text: &str) -> PyConceptGraph {
        let g = self.inner.process(text);
        PyConceptGraph::from_inner(g, &self.inner)
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "ConceptGraph", frozen)]
pub struct PyConceptGraph {
    inner: sphereql_lingua::ConceptGraph,
    sphereql_cache: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConceptGraph {
    #[getter]
    fn num_concepts(&self) -> usize {
        self.inner.concepts.len()
    }
    #[getter]
    fn num_relations(&self) -> usize {
        self.inner.relations.len()
    }

    fn concepts(&self) -> Vec<PyConcept> {
        self.inner
            .concepts
            .iter()
            .map(PyConcept::from_concept)
            .collect()
    }
    fn centroid(&self) -> Option<PySphericalPoint> {
        self.inner.centroid().map(PySphericalPoint::from_inner)
    }
    fn to_sphereql(&self) -> String {
        self.sphereql_cache.clone()
    }
    fn get_concept(&self, name: &str) -> Option<PyConcept> {
        self.inner.get_concept(name).map(PyConcept::from_concept)
    }
    fn __repr__(&self) -> String {
        format!(
            "ConceptGraph(concepts={}, relations={})",
            self.inner.concepts.len(),
            self.inner.relations.len()
        )
    }
    fn __len__(&self) -> usize {
        self.inner.concepts.len()
    }
}

impl PyConceptGraph {
    pub(crate) fn from_inner(
        inner: sphereql_lingua::ConceptGraph,
        pipeline: &sphereql_lingua::LinguaPipeline,
    ) -> Self {
        let sphereql_cache = inner.to_sphereql(pipeline.taxonomy());
        Self {
            inner,
            sphereql_cache,
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "LinguaConcept", frozen)]
#[derive(Clone)]
pub struct PyConcept {
    #[pyo3(get)]
    pub normalized: String,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub frequency: u32,
    #[pyo3(get)]
    pub salience: f64,
    point: Option<sphereql_core::SphericalPoint>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConcept {
    #[getter]
    fn point(&self) -> Option<PySphericalPoint> {
        self.point.map(PySphericalPoint::from_inner)
    }
    #[getter]
    fn theta(&self) -> Option<f64> {
        self.point.map(|p| p.theta)
    }
    #[getter]
    fn phi(&self) -> Option<f64> {
        self.point.map(|p| p.phi)
    }
    #[getter]
    fn r(&self) -> Option<f64> {
        self.point.map(|p| p.r)
    }
    fn __repr__(&self) -> String {
        if let Some(p) = self.point {
            format!(
                "LinguaConcept('{}', t={:.3}, p={:.3}, r={:.3})",
                self.normalized, p.theta, p.phi, p.r
            )
        } else {
            format!("LinguaConcept('{}', unresolved)", self.normalized)
        }
    }
}

impl PyConcept {
    fn from_concept(c: &sphereql_lingua::Concept) -> Self {
        Self {
            normalized: c.normalized.clone(),
            text: c.text.clone(),
            frequency: c.frequency,
            salience: c.salience_score,
            point: c.point,
        }
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "lingua_process")]
pub fn py_lingua_process(text: &str) -> PyConceptGraph {
    let pipeline = sphereql_lingua::LinguaPipeline::new();
    let g = pipeline.process(text);
    PyConceptGraph::from_inner(g, &pipeline)
}
