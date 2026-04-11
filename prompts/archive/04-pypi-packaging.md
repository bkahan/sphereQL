# Prompt: Prepare sphereql Python Package for PyPI Release

## Context

`sphereql-python/` is a PyO3/maturin crate that builds a native Python
extension. It has a `pyproject.toml` using maturin as the build backend.
The package needs to be ready for `pip install sphereql` to work.

This prompt assumes prompts 01 (Python bridge) and 02 (viz tool) have
already been completed — the Python bindings now include core types,
Pipeline, VectorStoreBridge, and the visualize function.

## Deliverables

### 1. Fix `pyproject.toml` for PyPI

Update `sphereql-python/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "sphereql"
version = "0.1.0"
description = "Spherical coordinate knowledge representation — fast semantic search, visualization, and analysis"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.8"
authors = [
    { name = "Ben Kahan", email = "benkahan1@gmail.com" }
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Database",
]
keywords = ["embeddings", "semantic-search", "vector-database", "spherical", "pca", "visualization"]

[project.optional-dependencies]
qdrant = []  # marker only — triggers Rust feature at build time

[project.urls]
Repository = "https://github.com/bkahan/sphereQL"

[tool.maturin]
features = ["pyo3/extension-module", "core", "embed", "vectordb"]
python-source = "python"
module-name = "sphereql"
```

### 2. Create Python stub package at `sphereql-python/python/sphereql/__init__.py`

This re-exports everything from the native module with proper `__all__`
and adds the module docstring:

```python
"""
SphereQL — Spherical coordinate knowledge representation.

Project high-dimensional embeddings onto a 3D sphere for fast semantic
search, interactive visualization, and knowledge structure analysis.

Quick start:
    >>> import sphereql
    >>> pipeline = sphereql.Pipeline(categories, embeddings)
    >>> results = pipeline.nearest(query_embedding, k=5)
    >>> sphereql.visualize(categories, embeddings)
"""

from sphereql.sphereql import *  # noqa: F401,F403

__version__ = "0.1.0"
```

### 3. Create `sphereql-python/python/sphereql/py.typed`

Empty marker file for PEP 561 typed package support.

### 4. Create type stubs at `sphereql-python/python/sphereql/sphereql.pyi`

Write complete type stubs for every class and function exposed by the
native module. This is critical for IDE autocompletion and type checking.

Include stubs for:
- `SphericalPoint`, `CartesianPoint`, `GeoPoint`, `ProjectedPoint`
- All conversion/distance functions
- `PcaProjection`, `RandomProjection`
- `Pipeline` with all query methods
- `Nearest`, `Path`, `PathStep`, `Glob`, `Manifold` result types
- `InMemoryStore`, `VectorStoreBridge`
- `QdrantBridge` (conditional, but include in stubs)
- `visualize()`, `visualize_pipeline()`

Use proper Python typing: `list[float]`, `dict[str, Any]`,
`Optional[str]`, etc. Include docstrings in the stubs.

### 5. Create `sphereql-python/README.md`

Write a Python-focused README (distinct from the Rust workspace README).
Sections:

- **What is SphereQL** — 2-sentence pitch.
- **Install** — `pip install sphereql` and `pip install sphereql[qdrant]`.
- **Quick Start: Semantic Search** — 15-line example showing Pipeline
  construction and nearest query.
- **Quick Start: Visualization** — 5-line example showing `visualize()`.
- **Quick Start: Vector DB Bridge** — 10-line example showing
  InMemoryStore → VectorStoreBridge → hybrid_search.
- **How It Works** — 3-sentence explanation of PCA projection to sphere.
- **API Reference** — link to type stubs or future docs.
- **License** — MIT.

Keep it under 150 lines. No filler. Every line should help someone decide
to use or understand the library.

### 6. Create GitHub Actions workflow at `.github/workflows/python.yml`

```yaml
name: Python
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: dtolnay/rust-toolchain@stable
      - name: Install maturin
        run: pip install maturin[patchelf] pytest
      - name: Build and install
        run: cd sphereql-python && maturin develop --features "core,embed,vectordb"
      - name: Test
        run: cd sphereql-python && pytest tests/ -v

  # Only publish on tagged releases
  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: publish
          args: --manifest-path sphereql-python/Cargo.toml
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
```

## Constraints

- Do NOT modify Rust source files — this prompt is pure packaging/docs/CI.
- The `python/` source directory layout is required by maturin for pure
  Python files alongside native extensions.
- Type stubs must be accurate to the actual PyO3 signatures. Read every
  `#[pyclass]` and `#[pyfunction]` in `sphereql-python/src/` to verify.
- Run `cd sphereql-python && maturin develop && pytest tests/ -v` to
  verify everything works before considering done.
