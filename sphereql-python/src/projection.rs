use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use sphereql_embed::projection::{PcaProjection, Projection, RandomProjection};
use sphereql_embed::types::{Embedding, RadialStrategy};

use crate::core_types::{PyProjectedPoint, PySphericalPoint};

// ── Helpers ────────────────────────────────────────────────────────────

fn parse_radial(radial: &Bound<'_, PyAny>) -> PyResult<RadialStrategy> {
    if let Ok(s) = radial.extract::<String>() {
        match s.as_str() {
            "magnitude" => Ok(RadialStrategy::Magnitude),
            other => Err(PyValueError::new_err(format!(
                "unknown radial strategy '{other}': expected 'magnitude' or a float"
            ))),
        }
    } else if let Ok(v) = radial.extract::<f64>() {
        Ok(RadialStrategy::Fixed(v))
    } else {
        Err(PyValueError::new_err(
            "radial must be 'magnitude' or a float value",
        ))
    }
}

pub(crate) fn extract_embedding(obj: &Bound<'_, PyAny>) -> PyResult<Embedding> {
    // Try numpy 1D array first
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        let readonly = arr.try_readonly()?;
        return Ok(Embedding::new(readonly.as_slice()?.to_vec()));
    }
    // Try numpy 1D array of f32 (upcast)
    if let Ok(arr) = obj.cast::<PyArray1<f32>>() {
        let readonly = arr.try_readonly()?;
        return Ok(Embedding::new(
            readonly.as_slice()?.iter().map(|&v| v as f64).collect(),
        ));
    }
    // Fall back to list[float]
    let vec: Vec<f64> = obj.extract()?;
    Ok(Embedding::new(vec))
}

fn extract_embeddings_2d(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Embedding>> {
    // Try numpy 2D array (f64)
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let slice = arr.as_slice()?;
        return Ok((0..rows)
            .map(|i| Embedding::new(slice[i * cols..(i + 1) * cols].to_vec()))
            .collect());
    }
    // Try numpy 2D array (f32 upcast)
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f32>>() {
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let slice = arr.as_slice()?;
        return Ok((0..rows)
            .map(|i| {
                Embedding::new(
                    slice[i * cols..(i + 1) * cols]
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                )
            })
            .collect());
    }
    // Fall back to list[list[float]]
    let vecs: Vec<Vec<f64>> = obj.extract()?;
    Ok(vecs.into_iter().map(Embedding::new).collect())
}

// ── PcaProjection ──────────────────────────────────────────────────────

#[pyclass(name = "PcaProjection")]
pub struct PyPcaProjection {
    inner: PcaProjection,
}

#[pymethods]
impl PyPcaProjection {
    #[classmethod]
    #[pyo3(signature = (embeddings, *, radial=None, volumetric=false))]
    fn fit(
        _cls: &Bound<'_, PyType>,
        embeddings: &Bound<'_, PyAny>,
        radial: Option<&Bound<'_, PyAny>>,
        volumetric: bool,
    ) -> PyResult<Self> {
        let embs = extract_embeddings_2d(embeddings)?;
        let strategy = match radial {
            Some(r) => parse_radial(r)?,
            None => RadialStrategy::Magnitude,
        };
        let pca = PcaProjection::fit(&embs, strategy);
        let pca = if volumetric {
            pca.with_volumetric(true)
        } else {
            pca
        };
        Ok(Self { inner: pca })
    }

    #[getter]
    fn dimensionality(&self) -> usize {
        self.inner.dimensionality()
    }

    #[getter]
    fn explained_variance_ratio(&self) -> f64 {
        self.inner.explained_variance_ratio()
    }

    fn project(&self, embedding: &Bound<'_, PyAny>) -> PyResult<PySphericalPoint> {
        let emb = extract_embedding(embedding)?;
        Ok(PySphericalPoint::from_inner(self.inner.project(&emb)))
    }

    fn project_rich(&self, embedding: &Bound<'_, PyAny>) -> PyResult<PyProjectedPoint> {
        let emb = extract_embedding(embedding)?;
        Ok(PyProjectedPoint::from_inner(self.inner.project_rich(&emb)))
    }

    fn project_batch<'py>(
        &self,
        py: Python<'py>,
        embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<PySphericalPoint>> {
        let embs = extract_embeddings_2d(embeddings)?;
        let results = py.detach(|| {
            embs.iter()
                .map(|e| self.inner.project(e))
                .collect::<Vec<_>>()
        });
        Ok(results.into_iter().map(PySphericalPoint::from_inner).collect())
    }

    fn project_rich_batch<'py>(
        &self,
        py: Python<'py>,
        embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<PyProjectedPoint>> {
        let embs = extract_embeddings_2d(embeddings)?;
        let results = py.detach(|| {
            embs.iter()
                .map(|e| self.inner.project_rich(e))
                .collect::<Vec<_>>()
        });
        Ok(results
            .into_iter()
            .map(PyProjectedPoint::from_inner)
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "PcaProjection(dim={}, explained_variance={:.4})",
            self.inner.dimensionality(),
            self.inner.explained_variance_ratio()
        )
    }
}

impl PyPcaProjection {
    #[allow(dead_code)] // used by pipeline module (prompt 04)
    pub(crate) fn inner(&self) -> &PcaProjection {
        &self.inner
    }
}

// ── RandomProjection ───────────────────────────────────────────────────

#[pyclass(name = "RandomProjection")]
pub struct PyRandomProjection {
    inner: RandomProjection,
}

#[pymethods]
impl PyRandomProjection {
    #[new]
    #[pyo3(signature = (dim, *, radial=None, seed=42))]
    fn new(dim: usize, radial: Option<&Bound<'_, PyAny>>, seed: u64) -> PyResult<Self> {
        let strategy = match radial {
            Some(r) => parse_radial(r)?,
            None => RadialStrategy::Magnitude,
        };
        Ok(Self {
            inner: RandomProjection::new(dim, strategy, seed),
        })
    }

    #[getter]
    fn dimensionality(&self) -> usize {
        self.inner.dimensionality()
    }

    fn project(&self, embedding: &Bound<'_, PyAny>) -> PyResult<PySphericalPoint> {
        let emb = extract_embedding(embedding)?;
        Ok(PySphericalPoint::from_inner(self.inner.project(&emb)))
    }

    fn project_rich(&self, embedding: &Bound<'_, PyAny>) -> PyResult<PyProjectedPoint> {
        let emb = extract_embedding(embedding)?;
        Ok(PyProjectedPoint::from_inner(self.inner.project_rich(&emb)))
    }

    fn project_batch<'py>(
        &self,
        py: Python<'py>,
        embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<PySphericalPoint>> {
        let embs = extract_embeddings_2d(embeddings)?;
        let results = py.detach(|| {
            embs.iter()
                .map(|e| self.inner.project(e))
                .collect::<Vec<_>>()
        });
        Ok(results.into_iter().map(PySphericalPoint::from_inner).collect())
    }

    fn project_rich_batch<'py>(
        &self,
        py: Python<'py>,
        embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<PyProjectedPoint>> {
        let embs = extract_embeddings_2d(embeddings)?;
        let results = py.detach(|| {
            embs.iter()
                .map(|e| self.inner.project_rich(e))
                .collect::<Vec<_>>()
        });
        Ok(results
            .into_iter()
            .map(PyProjectedPoint::from_inner)
            .collect())
    }

    fn __repr__(&self) -> String {
        format!("RandomProjection(dim={})", self.inner.dimensionality())
    }
}
