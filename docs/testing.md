# Testing

## Running tests locally

```bash
# All workspace tests
cargo test --workspace

# All features (including qdrant/pinecone compile checks)
cargo test --workspace --all-features

# Clippy lint pass
cargo clippy --workspace --all-features --all-targets

# Format check
cargo fmt --check

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

- `cargo test --workspace --all-features` + doc-tests.
- `cargo clippy` with `-Dwarnings`.
- `cargo fmt --check`.
- Per-feature compilation matrix (`core`, `index`, `layout`, `embed`,
  `graphql`, `vectordb`, `full`, `no-default-features`).
- Python build + `pytest` on Python 3.12.

## Release pipeline

Separate release workflows publish to
[crates.io](../.github/workflows/crates-publish.yml) and
[PyPI](../.github/workflows/python-publish.yml) automatically when a
GitHub Release is created. PyPI wheels are built for Linux
x86_64/aarch64, macOS x86_64/aarch64, and Windows x86_64.
