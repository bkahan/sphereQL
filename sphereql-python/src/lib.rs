use pyo3::prelude::*;

#[cfg(feature = "embed")]
mod pipeline;
#[cfg(feature = "embed")]
mod types;

#[pymodule]
fn sphereql(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "embed")]
    {
        _m.add_class::<pipeline::Pipeline>()?;
        _m.add_class::<types::Nearest>()?;
        _m.add_class::<types::Path>()?;
        _m.add_class::<types::PathStep>()?;
        _m.add_class::<types::Glob>()?;
        _m.add_class::<types::Manifold>()?;
    }
    Ok(())
}
