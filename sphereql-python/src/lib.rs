use pyo3::prelude::*;

mod pipeline;
mod types;

#[pymodule]
fn sphereql(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<pipeline::Pipeline>()?;
    m.add_class::<types::Nearest>()?;
    m.add_class::<types::Path>()?;
    m.add_class::<types::PathStep>()?;
    m.add_class::<types::Glob>()?;
    m.add_class::<types::Manifold>()?;
    Ok(())
}
