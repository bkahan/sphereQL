use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use sphereql_embed::kernel_pca::KernelPcaProjection;
use sphereql_embed::laplacian::{
    DEFAULT_ACTIVE_THRESHOLD, DEFAULT_K_NEIGHBORS, LaplacianEigenmapProjection,
};
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

fn validate_finite(values: &[f64]) -> PyResult<()> {
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "embedding values must be finite (no NaN or Inf), found {} at index {i}",
            values[i]
        )));
    }
    Ok(())
}

pub(crate) fn extract_embedding(obj: &Bound<'_, PyAny>) -> PyResult<Embedding> {
    // Try numpy 1D array first
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        let readonly = arr.try_readonly()?;
        let values = readonly.as_slice()?.to_vec();
        validate_finite(&values)?;
        return Ok(Embedding::new(values));
    }
    // Try numpy 1D array of f32 (upcast)
    if let Ok(arr) = obj.cast::<PyArray1<f32>>() {
        let readonly = arr.try_readonly()?;
        let values: Vec<f64> = readonly.as_slice()?.iter().map(|&v| v as f64).collect();
        validate_finite(&values)?;
        return Ok(Embedding::new(values));
    }
    // Fall back to list[float]
    let vec: Vec<f64> = obj.extract()?;
    validate_finite(&vec)?;
    Ok(Embedding::new(vec))
}

pub(crate) fn extract_embeddings_2d(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Embedding>> {
    // Try numpy 2D array (f64)
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let slice = arr.as_slice()?;
        validate_finite(slice)?;
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
        let values: Vec<f64> = slice.iter().map(|&v| v as f64).collect();
        validate_finite(&values)?;
        return Ok((0..rows)
            .map(|i| Embedding::new(values[i * cols..(i + 1) * cols].to_vec()))
            .collect());
    }
    // Fall back to list[list[float]]
    let vecs: Vec<Vec<f64>> = obj.extract()?;
    for (i, v) in vecs.iter().enumerate() {
        if let Some(j) = v.iter().position(|val| !val.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "embedding values must be finite (no NaN or Inf), found {} at [{i}][{j}]",
                v[j]
            )));
        }
    }
    Ok(vecs.into_iter().map(Embedding::new).collect())
}

// ── PcaProjection ──────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "PcaProjection")]
pub struct PyPcaProjection {
    inner: PcaProjection,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPcaProjection {
    #[classmethod]
    #[pyo3(signature = (embeddings, *, radial=None, volumetric=false))]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        embeddings: &Bound<'_, PyAny>,
        radial: Option<&Bound<'_, PyAny>>,
        volumetric: bool,
    ) -> PyResult<Self> {
        let embs = extract_embeddings_2d(embeddings)?;
        let strategy = match radial {
            Some(r) => parse_radial(r)?,
            None => RadialStrategy::Magnitude,
        };
        let pca = py
            .detach(|| PcaProjection::fit(&embs, strategy))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        Ok(results
            .into_iter()
            .map(PySphericalPoint::from_inner)
            .collect())
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

#[gen_stub_pyclass]
#[pyclass(name = "RandomProjection")]
pub struct PyRandomProjection {
    inner: RandomProjection,
}

#[gen_stub_pymethods]
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
        Ok(results
            .into_iter()
            .map(PySphericalPoint::from_inner)
            .collect())
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

// ── KernelPcaProjection ───────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "KernelPcaProjection")]
pub struct PyKernelPcaProjection {
    inner: KernelPcaProjection,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyKernelPcaProjection {
    #[classmethod]
    #[pyo3(signature = (embeddings, *, sigma=None, radial=None, volumetric=false))]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        embeddings: &Bound<'_, PyAny>,
        sigma: Option<f64>,
        radial: Option<&Bound<'_, PyAny>>,
        volumetric: bool,
    ) -> PyResult<Self> {
        let embs = extract_embeddings_2d(embeddings)?;
        let strategy = match radial {
            Some(r) => parse_radial(r)?,
            None => RadialStrategy::Magnitude,
        };
        let kpca = py
            .detach(|| match sigma {
                Some(s) => KernelPcaProjection::fit_with_sigma(&embs, s, strategy),
                None => KernelPcaProjection::fit(&embs, strategy),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let kpca = if volumetric {
            kpca.with_volumetric(true)
        } else {
            kpca
        };
        Ok(Self { inner: kpca })
    }

    #[getter]
    fn dimensionality(&self) -> usize {
        self.inner.dimensionality()
    }

    #[getter]
    fn explained_variance_ratio(&self) -> f64 {
        self.inner.explained_variance_ratio()
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }

    #[getter]
    fn num_training_points(&self) -> usize {
        self.inner.num_training_points()
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
        Ok(results
            .into_iter()
            .map(PySphericalPoint::from_inner)
            .collect())
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
            "KernelPcaProjection(dim={}, sigma={:.4}, explained_variance={:.4})",
            self.inner.dimensionality(),
            self.inner.sigma(),
            self.inner.explained_variance_ratio()
        )
    }
}

// ── LaplacianEigenmapProjection ───────────────────────────────────────
//
// Connectivity-preserving spectral projection. On sparse / noise-heavy
// corpora where variance-maximizing projections (PCA / kernel PCA) pull
// toward noise axes, Laplacian eigenmaps preserve neighbor structure
// instead and typically keep category boundaries cleaner. See
// `sphereql_embed::laplacian` for algorithmic details.

#[gen_stub_pyclass]
#[pyclass(name = "LaplacianEigenmap")]
pub struct PyLaplacianEigenmapProjection {
    inner: LaplacianEigenmapProjection,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLaplacianEigenmapProjection {
    /// Fit a Laplacian-eigenmap projection to a corpus.
    ///
    /// - `embeddings`: 2-D numpy array (f32/f64) or list-of-lists; needs
    ///   at least 4 rows and any positive dimensionality.
    /// - `radial`: `"magnitude"` (default) or a finite float to fix the
    ///   radial coordinate.
    /// - `k_neighbors`: density of the sparsified k-NN graph. Higher =
    ///   smoother embedding, less noise-sensitivity, but more blurred
    ///   category boundaries. Typical range 10–30; defaults to 15.
    /// - `active_threshold`: absolute-weight cutoff for the active-axis
    ///   filter. Axes with `|v| ≤ threshold` are dropped before computing
    ///   Jaccard similarity. Defaults to 0.05.
    #[classmethod]
    #[pyo3(signature = (embeddings, *, radial=None, k_neighbors=None, active_threshold=None))]
    fn fit(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        embeddings: &Bound<'_, PyAny>,
        radial: Option<&Bound<'_, PyAny>>,
        k_neighbors: Option<usize>,
        active_threshold: Option<f64>,
    ) -> PyResult<Self> {
        let embs = extract_embeddings_2d(embeddings)?;
        let strategy = match radial {
            Some(r) => parse_radial(r)?,
            None => RadialStrategy::Magnitude,
        };
        let k = k_neighbors.unwrap_or(DEFAULT_K_NEIGHBORS);
        let thresh = active_threshold.unwrap_or(DEFAULT_ACTIVE_THRESHOLD);
        let proj = py
            .detach(|| LaplacianEigenmapProjection::fit_with_params(&embs, k, thresh, strategy))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: proj })
    }

    #[getter]
    fn dimensionality(&self) -> usize {
        self.inner.dimensionality()
    }

    /// Mean of `|μ_k|` across the three retained eigenvalues, in `[0, 1]`.
    /// Exposed under the same name as PCA's EVR for adaptive-threshold
    /// compatibility — but the two quantities have distinct meaning.
    #[getter]
    fn explained_variance_ratio(&self) -> f64 {
        self.inner.explained_variance_ratio()
    }

    /// Same scalar as `explained_variance_ratio`, but under the
    /// spectrally-accurate name.
    #[getter]
    fn connectivity_ratio(&self) -> f64 {
        self.inner.connectivity_ratio()
    }

    /// The three retained non-trivial eigenvalues, in descending order.
    #[getter]
    fn eigenvalues(&self) -> (f64, f64, f64) {
        let [a, b, c] = self.inner.eigenvalues();
        (a, b, c)
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
        Ok(results
            .into_iter()
            .map(PySphericalPoint::from_inner)
            .collect())
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
        let [a, b, c] = self.inner.eigenvalues();
        format!(
            "LaplacianEigenmap(dim={}, connectivity={:.4}, eigenvalues=[{:.3}, {:.3}, {:.3}])",
            self.inner.dimensionality(),
            self.inner.connectivity_ratio(),
            a,
            b,
            c
        )
    }
}
