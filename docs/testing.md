# Testing

## Running tests locally

```bash
# All workspace tests. sphereql-python is excluded under --all-features:
# pyo3's extension-module feature breaks the `cargo test` link step
# (Python provides those symbols at load time). Its tests run via pytest.
cargo test --workspace --all-features --exclude sphereql-python

# Doc-tests
cargo test --doc --workspace --exclude sphereql-python

# Clippy lint pass (CI runs this with RUSTFLAGS=-Dwarnings)
cargo clippy --workspace --all-features --all-targets

# Format check
cargo fmt --all -- --check

# Bindings drift check: every public sphereql-embed/layout API must
# have a Python or WASM binding (or an allowlist entry)
cargo run -p check-drift

# WASM target check (native builds won't catch wasm32 breakage)
cargo build -p sphereql-wasm --target wasm32-unknown-unknown

# Python tests
cd sphereql-python
maturin develop
pytest -v

# Benchmarks
cargo bench -p sphereql-core
cargo bench -p sphereql-index
```

## CI

The [CI pipeline](../.github/workflows/ci.yml) runs on every push and
PR to `main`:

- `cargo test --workspace --all-features --exclude sphereql-python`
  plus doc-tests.
- `cargo clippy` with `-Dwarnings`.
- `cargo doc --workspace --all-features --no-deps`.
- `cargo fmt --all -- --check`.
- Per-feature compilation matrix (`core`, `index`, `layout`, `embed`,
  `graphql`, `vectordb`, `full`, `no-default-features`).
- Python build + `pytest` on Python 3.10–3.13.
- Stub freshness: regenerates `__init__.pyi` via
  `cargo run --bin gen-stubs` and fails if the checked-in stubs drifted.
- WASM build to `wasm32-unknown-unknown`.
- Bindings drift check (`cargo run -p check-drift`), mirrored by the
  separate [bindings-drift workflow](../.github/workflows/bindings-drift.yml).

## Release pipeline

Separate release workflows publish to
[crates.io](../.github/workflows/crates-publish.yml) and
[PyPI](../.github/workflows/python-publish.yml) automatically when a
GitHub Release is created. PyPI wheels are built for Linux
x86_64/aarch64, macOS x86_64/aarch64, and Windows x86_64.
