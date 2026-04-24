# Project status

sphereQL is at **v0.2.0-alpha**. The core API is functional and covered
by 600+ tests, but may change before 1.0. See
[CHANGELOG.md](../CHANGELOG.md) for the v0.1 → v0.2 diff.

## Bindings status

Python and WASM bindings are at full parity with the Rust surface:

- Projection families: PCA, Kernel PCA, **Laplacian eigenmap**, Random.
- Pipeline queries: nearest, similar-above, concept path, glob detection,
  local manifold.
- Category enrichment: concept paths, category neighbors, drill-down,
  hierarchical nearest, domain groups, category stats.
- Metalearning: `auto_tune`, `NearestNeighborMetaModel`,
  `DistanceWeightedMetaModel`, `FeedbackAggregator`,
  `MetaTrainingRecord` default-store helpers.
- Partial-config support: any `PipelineConfig` field can be omitted;
  missing keys fall back to defaults (no more "specify every knob").

**Python**: type stubs (`.pyi`) are auto-generated via `pyo3-stub-gen` —
IDE and `mypy`/`pyright` pick them up automatically.

**WASM**: return values are typed via `tsify` — `wasm-pack build` emits
a `.d.ts` with a named interface for every payload. No `JSON.parse`
step required on the JS side for pipeline, category, or metalearning
methods.

**GraphQL**: full spatial + category enrichment surface via a single
`MergedQueryRoot`. Text queries embed server-side through an
injectable `TextEmbedder` trait (default `NoEmbedder` returns a
descriptive error until a real embedder is plugged in).

## Known limitations

- **Search precision degrades at higher k.** The 384-d to 3-d projection
  is inherently lossy (~2–5% explained variance). Precision at k=1 is
  perfect; at k=5 it drops to ~20%. Mitigations available today:
    - Hybrid search (angular recall + cosine re-ranking) for production
      precision.
    - `pipeline.hierarchical_nearest(embedding, k)` falls back to
      domain-group routing when the fitted EVR is below
      `RoutingConfig::low_evr_threshold`.
    - `ProjectionKind::LaplacianEigenmap` via `auto_tune` often
      preserves more neighbor structure than PCA on sparse/noisy
      corpora (see [empirical findings](empirical-findings.md)).

  See [search-precision-roadmap.md](search-precision-roadmap.md) for
  tracked improvements.

- **Inner spheres require 20+ items per category.** Categories below
  this threshold fall back to the outer sphere for drill-down queries.
  This is by design — small categories don't benefit from a separate
  projection, and the threshold is configurable via
  `InnerSphereConfig::min_size`.

## Drift protection

A CI job (`.github/workflows/bindings-drift.yml`) runs
`cargo run -p check-drift` on every PR that touches `sphereql-embed`,
`sphereql-layout`, or either binding crate. The tool `syn`-parses the
public API surface of embed + layout, compares against
`#[pyfunction]`/`#[pyclass]` exports in Python and `#[wasm_bindgen]` /
Tsify-derived exports in WASM, and fails when a new public item has
neither a binding nor an entry in `.bindings-ignore.toml`.

Adding a new public type to `sphereql-embed`? Either bind it in Python
and/or WASM, or add it to `.bindings-ignore.toml` with a `reason` field
explaining why a 1:1 binding isn't required. The allowlist ships with
~90 intentionally-exempt items (config structs reached via dict,
internal helpers, layout-crate internals, foreign-trait objects).

## Roadmap

- Improve search precision at higher k values.
- HNSW or VP-tree indexing for better recall without brute-force
  fallback.
- Streaming/incremental PCA for large-scale datasets.
- Expose `sphereql-layout`'s managed layouts through the bindings.
