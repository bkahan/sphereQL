use pyo3::prelude::*;

#[cfg(feature = "core")]
mod core_types;
#[cfg(feature = "embed")]
mod meta;
#[cfg(feature = "embed")]
mod nav;
#[cfg(feature = "embed")]
mod pipeline;
#[cfg(feature = "embed")]
mod projection;
#[cfg(feature = "embed")]
mod types;
#[cfg(feature = "vectordb")]
mod vectordb;
#[cfg(feature = "embed")]
mod viz;

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
        m.add_class::<projection::PyKernelPcaProjection>()?;
        m.add_class::<pipeline::Pipeline>()?;
        m.add_class::<types::Nearest>()?;
        m.add_class::<types::Path>()?;
        m.add_class::<types::PathStep>()?;
        m.add_class::<types::Glob>()?;
        m.add_class::<types::Manifold>()?;
        m.add_class::<types::PyCategorySummary>()?;
        m.add_class::<types::PyBridgeItem>()?;
        m.add_class::<types::PyCategoryPathStep>()?;
        m.add_class::<types::PyCategoryPath>()?;
        m.add_class::<types::PyDrillDown>()?;
        m.add_class::<types::PyInnerSphereReport>()?;
        m.add_class::<types::PyDomainGroup>()?;
        m.add_class::<types::PyProjectionWarning>()?;
        m.add_class::<nav::PyNavigatorConfig>()?;
        m.add_class::<nav::PyNavigatorReport>()?;
        m.add_class::<nav::PyAntipodalReport>()?;
        m.add_class::<nav::PyCoverageReport>()?;
        m.add_class::<nav::PyVoronoiCell>()?;
        m.add_class::<nav::PyCurvatureTriple>()?;
        m.add_class::<nav::PyLuneReport>()?;
        m.add_function(wrap_pyfunction!(nav::run_navigator, m)?)?;
        m.add_function(wrap_pyfunction!(viz::visualize, m)?)?;
        m.add_function(wrap_pyfunction!(viz::visualize_pipeline, m)?)?;
        m.add_function(wrap_pyfunction!(meta::corpus_features, m)?)?;
        m.add_function(wrap_pyfunction!(meta::auto_tune, m)?)?;
    }

    #[cfg(feature = "vectordb")]
    {
        m.add_class::<vectordb::PyInMemoryStore>()?;
        m.add_class::<vectordb::PyVectorStoreBridge>()?;
    }

    #[cfg(feature = "pinecone")]
    {
        m.add_class::<vectordb::PyPineconeBridge>()?;
    }

    #[cfg(feature = "qdrant")]
    {
        m.add_class::<vectordb::PyQdrantBridge>()?;
    }

    Ok(())
}
