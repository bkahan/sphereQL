use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use sphereql_core::{
    CartesianPoint, GeoPoint, SphericalPoint, angular_distance as rust_angular_distance,
    cartesian_to_spherical as rust_cartesian_to_spherical, chord_distance as rust_chord_distance,
    geo_to_spherical as rust_geo_to_spherical, great_circle_distance as rust_great_circle_distance,
    spherical_to_cartesian as rust_spherical_to_cartesian,
    spherical_to_geo as rust_spherical_to_geo,
};

// ── SphericalPoint ─────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "SphericalPoint", frozen, from_py_object)]
#[derive(Clone)]
pub struct PySphericalPoint {
    inner: SphericalPoint,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySphericalPoint {
    #[new]
    fn new(r: f64, theta: f64, phi: f64) -> PyResult<Self> {
        SphericalPoint::new(r, theta, phi)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    #[getter]
    fn phi(&self) -> f64 {
        self.inner.phi
    }

    fn to_cartesian(&self) -> PyCartesianPoint {
        PyCartesianPoint::from_inner(rust_spherical_to_cartesian(&self.inner))
    }

    fn to_geo(&self) -> PyGeoPoint {
        PyGeoPoint::from_inner(rust_spherical_to_geo(&self.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "SphericalPoint(r={}, theta={}, phi={})",
            self.inner.r, self.inner.theta, self.inner.phi
        )
    }

    fn __eq__(&self, other: &PySphericalPoint) -> bool {
        self.inner == other.inner
    }
}

impl PySphericalPoint {
    pub(crate) fn inner(&self) -> &SphericalPoint {
        &self.inner
    }

    pub(crate) fn from_inner(inner: SphericalPoint) -> Self {
        Self { inner }
    }
}

// ── CartesianPoint ─────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "CartesianPoint", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCartesianPoint {
    inner: CartesianPoint,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCartesianPoint {
    #[new]
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: CartesianPoint::new(x, y, z),
        }
    }

    #[getter]
    fn x(&self) -> f64 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.inner.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.inner.z
    }

    fn magnitude(&self) -> f64 {
        self.inner.magnitude()
    }

    fn normalize(&self) -> PyCartesianPoint {
        Self::from_inner(self.inner.normalize())
    }

    fn to_spherical(&self) -> PySphericalPoint {
        PySphericalPoint::from_inner(rust_cartesian_to_spherical(&self.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "CartesianPoint(x={}, y={}, z={})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }

    fn __eq__(&self, other: &PyCartesianPoint) -> bool {
        self.inner == other.inner
    }
}

impl PyCartesianPoint {
    pub(crate) fn from_inner(inner: CartesianPoint) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &CartesianPoint {
        &self.inner
    }
}

// ── GeoPoint ───────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "GeoPoint", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyGeoPoint {
    inner: GeoPoint,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeoPoint {
    #[new]
    fn new(lat: f64, lon: f64, alt: f64) -> PyResult<Self> {
        GeoPoint::new(lat, lon, alt)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn lat(&self) -> f64 {
        self.inner.lat
    }

    #[getter]
    fn lon(&self) -> f64 {
        self.inner.lon
    }

    #[getter]
    fn alt(&self) -> f64 {
        self.inner.alt
    }

    fn to_spherical(&self) -> PySphericalPoint {
        PySphericalPoint::from_inner(rust_geo_to_spherical(&self.inner))
    }

    fn to_cartesian(&self) -> PyCartesianPoint {
        PyCartesianPoint::from_inner(rust_spherical_to_cartesian(&rust_geo_to_spherical(
            &self.inner,
        )))
    }

    fn __repr__(&self) -> String {
        format!(
            "GeoPoint(lat={}, lon={}, alt={})",
            self.inner.lat, self.inner.lon, self.inner.alt
        )
    }

    fn __eq__(&self, other: &PyGeoPoint) -> bool {
        self.inner == other.inner
    }
}

impl PyGeoPoint {
    pub(crate) fn from_inner(inner: GeoPoint) -> Self {
        Self { inner }
    }

    pub(crate) fn inner_geo(&self) -> &GeoPoint {
        &self.inner
    }
}

// ── ProjectedPoint ─────────────────────────────────────────────────────

#[cfg(feature = "embed")]
#[gen_stub_pyclass]
#[pyclass(name = "ProjectedPoint", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyProjectedPoint {
    inner: sphereql_embed::ProjectedPoint,
}

#[cfg(feature = "embed")]
#[gen_stub_pymethods]
#[pymethods]
impl PyProjectedPoint {
    #[getter]
    fn position(&self) -> PySphericalPoint {
        PySphericalPoint::from_inner(self.inner.position)
    }

    #[getter]
    fn certainty(&self) -> f64 {
        self.inner.certainty
    }

    #[getter]
    fn intensity(&self) -> f64 {
        self.inner.intensity
    }

    #[getter]
    fn projection_magnitude(&self) -> f64 {
        self.inner.projection_magnitude
    }

    fn __repr__(&self) -> String {
        format!(
            "ProjectedPoint(r={:.4}, theta={:.4}, phi={:.4}, certainty={:.4}, intensity={:.4})",
            self.inner.position.r,
            self.inner.position.theta,
            self.inner.position.phi,
            self.inner.certainty,
            self.inner.intensity
        )
    }
}

#[cfg(feature = "embed")]
impl PyProjectedPoint {
    #[allow(dead_code)] // used by projection module (prompt 03)
    pub(crate) fn from_inner(inner: sphereql_embed::ProjectedPoint) -> Self {
        Self { inner }
    }
}

// ── Distance functions ─────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "angular_distance")]
pub fn py_angular_distance(a: &PySphericalPoint, b: &PySphericalPoint) -> f64 {
    rust_angular_distance(a.inner(), b.inner())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "great_circle_distance")]
pub fn py_great_circle_distance(a: &PySphericalPoint, b: &PySphericalPoint, radius: f64) -> f64 {
    rust_great_circle_distance(a.inner(), b.inner(), radius)
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "chord_distance")]
pub fn py_chord_distance(a: &PySphericalPoint, b: &PySphericalPoint) -> f64 {
    rust_chord_distance(a.inner(), b.inner())
}

// ── Conversion functions ───────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "spherical_to_cartesian")]
pub fn py_spherical_to_cartesian(p: &PySphericalPoint) -> PyCartesianPoint {
    PyCartesianPoint::from_inner(rust_spherical_to_cartesian(p.inner()))
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "cartesian_to_spherical")]
pub fn py_cartesian_to_spherical(p: &PyCartesianPoint) -> PySphericalPoint {
    PySphericalPoint::from_inner(rust_cartesian_to_spherical(p.inner()))
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "spherical_to_geo")]
pub fn py_spherical_to_geo(p: &PySphericalPoint) -> PyGeoPoint {
    PyGeoPoint::from_inner(rust_spherical_to_geo(p.inner()))
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "geo_to_spherical")]
pub fn py_geo_to_spherical(p: &PyGeoPoint) -> PySphericalPoint {
    PySphericalPoint::from_inner(rust_geo_to_spherical(p.inner_geo()))
}
