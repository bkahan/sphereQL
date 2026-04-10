use pyo3::prelude::*;

#[cfg(feature = "core")]
mod core_types;
#[cfg(feature = "embed")]
mod pipeline;
#[cfg(feature = "embed")]
mod projection;
#[cfg(feature = "embed")]
mod types;

#[pymodule]
fn sphereql(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "core")]
    {
        m.add_class::<core_types::PySphericalPoint>()?;
        m.add_class::<core_types::PyCartesianPoint>()?;
        m.add_class::<core_types::PyGeoPoint>()?;
        m.add_function(wrap_pyfunction!(core_types::py_angular_distance, m)?)?;
        m.add_function(wrap_pyfunction!(core_types::py_great_circle_distance, m)?)?;
        m.add_function(wrap_pyfunction!(core_types::py_chord_distance, m)?)?;
        m.add_function(wrap_pyfunction!(core_types::py_spherical_to_cartesian, m)?)?;
        m.add_function(wrap_pyfunction!(core_types::py_cartesian_to_spherical, m)?)?;
        m.add_function(wrap_pyfunction!(core_types::py_spherical_to_geo, m)?)?;
        m.add_function(wrap_pyfunction!(core_types::py_geo_to_spherical, m)?)?;
    }

    #[cfg(feature = "embed")]
    {
        m.add_class::<core_types::PyProjectedPoint>()?;
        m.add_class::<projection::PyPcaProjection>()?;
        m.add_class::<projection::PyRandomProjection>()?;
        m.add_class::<pipeline::Pipeline>()?;
        m.add_class::<types::Nearest>()?;
        m.add_class::<types::Path>()?;
        m.add_class::<types::PathStep>()?;
        m.add_class::<types::Glob>()?;
        m.add_class::<types::Manifold>()?;
    }

    Ok(())
}
