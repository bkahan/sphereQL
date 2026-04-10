# Prompt 06: Build validation and CI integration

## Context

The `sphereql-python` crate is fully implemented with tests and type stubs
(prompts 01–05). This prompt validates the build and adds CI configuration.

## Task

### 1. Verify the Rust side compiles

Run these and fix any issues:

```bash
cargo check -p sphereql-python
cargo clippy -p sphereql-python
cargo fmt --check
```

### 2. Verify wasm32 is not broken

The new crate should not be included in wasm32 builds. Verify:

```bash
cargo check --target wasm32-unknown-unknown -p sphereql-wasm
```

This should still work because `sphereql-python` is a separate workspace
member with no reverse dependencies.

### 3. Verify all existing tests still pass

```bash
cargo test
```

All 267+ existing tests must pass with zero failures.

### 4. Build the Python wheel and run tests

```bash
cd sphereql-python
pip install maturin numpy pytest
maturin develop --release
pytest tests/ -v
```

Fix any test failures. Common issues:
- PyO3 type conversion errors (wrong arg types in `#[pymethods]`)
- Numpy array dtype mismatches (ensure float32 upcast works)
- Missing `#[pyclass]` registrations in `lib.rs`

### 5. Add a Makefile target (optional convenience)

Add to the root `Makefile` (create if it doesn't exist), or to
`sphereql-python/Makefile`:

```makefile
.PHONY: python-dev python-test

python-dev:
	cd sphereql-python && maturin develop --release

python-test: python-dev
	cd sphereql-python && pytest tests/ -v
```

### 6. Add a README for the Python package

Create `sphereql-python/README.md`:

```markdown
# sphereql (Python)

Python bindings for the sphereQL spherical coordinate engine.

## Install (development)

    cd sphereql-python
    pip install maturin numpy
    maturin develop --release

## Usage

    import numpy as np
    import sphereql

    # Project embeddings onto a sphere
    embeddings = np.random.randn(100, 384)
    pca = sphereql.PcaProjection.fit(embeddings)

    point = pca.project(embeddings[0])
    print(point.r, point.theta, point.phi)

    # Full pipeline
    categories = ["cat_a"] * 50 + ["cat_b"] * 50
    pipeline = sphereql.Pipeline(categories, embeddings)
    hits = pipeline.nearest(embeddings[0], k=5)
    for h in hits:
        print(f"{h.id} [{h.category}] dist={h.distance:.4f} cert={h.certainty:.2f}")

## Type Stubs

IDE autocompletion is provided via `sphereql.pyi`.
```

## Verification

All of the following must succeed:
1. `cargo check -p sphereql-python` — no errors
2. `cargo clippy -p sphereql-python` — no warnings
3. `cargo test` — all workspace tests pass
4. `cargo check --target wasm32-unknown-unknown -p sphereql-wasm` — still works
5. `cd sphereql-python && maturin develop --release && pytest tests/ -v` — all
   Python tests pass

## Do NOT

- Modify any existing crate code to accommodate PyO3 (the type surface is
  already compatible).
- Add PyO3 feature flags to existing crates — keep the Python binding fully
  contained in `sphereql-python`.
