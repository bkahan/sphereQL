//! Generate `.pyi` stub files for the `sphereql` Python package.
//!
//! Run from the `sphereql-python` directory:
//!
//! ```sh
//! cargo run --bin gen-stubs
//! ```
//!
//! This writes `sphereql.pyi` alongside the built extension so IDEs and
//! static type checkers (pyright, mypy) pick it up.
//!
//! The bin must be built without pyo3's `extension-module` feature — that
//! feature tells the linker "Python will supply libpython at load time",
//! which is only correct for the cdylib that maturin packages, not for a
//! standalone native binary. Default cargo builds don't enable it; the
//! feature is opted in via `pyproject.toml` during `maturin develop` /
//! `maturin build`.

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = sphereql::stub_info()?;
    stub.generate()?;
    Ok(())
}
