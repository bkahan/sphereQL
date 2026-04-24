# Changelog

All notable user-visible changes. sphereQL follows semver from v1.0
onward; while on `0.x-alpha` expect breaking changes between minor
versions.

## [0.2.0-alpha] — 2026-04-24

### Added — bindings parity

- **Laplacian eigenmap** exposed as a standalone projection class in
  Python (`sphereql.LaplacianEigenmap`) and WASM
  (`LaplacianEigenmapProjection`). Mirrors the Rust API: fit, project,
  project_batch, connectivity_ratio, eigenvalues.
- **Pipeline-level Laplacian** — pass
  `config={"projection_kind": "LaplacianEigenmap"}` to `Pipeline()`
  (Python) or `newWithConfig` (WASM) to build the outer sphere with
  Laplacian. `auto_tune` continues to sweep projection kind as a first-
  class tuner axis.

### Added — GraphQL category layer

- `CategoryQueryRoot` with seven resolvers: `conceptPath`,
  `categoryConceptPath`, `categoryNeighbors`, `drillDown`,
  `hierarchicalNearest`, `categoryStats`, `domainGroups`.
- `MergedQueryRoot` unifies the spatial and category surfaces under a
  single `UnifiedSchema`.
- `build_unified_schema` / `build_unified_schema_from_items` helpers
  wire the schema from a `SphereQLPipeline` + pluggable `TextEmbedder`
  + spatial index + event bus.
- `CategorizedItemInput` / `CategorizedItemOutput` for schema-native
  item I/O, plus `items_to_pipeline_input` for pipeline construction.

### Added — `TextEmbedder` trait (new module in `sphereql-embed`)

- `TextEmbedder` trait with `Send + Sync` and a single fallible
  `embed(&str) -> Result<Embedding, EmbedderError>` method.
- `NoEmbedder` default that errors descriptively — wired into GraphQL
  schemas so text query resolvers fail cleanly when no embedder is
  configured.
- `FnEmbedder` closure wrapper for quick test / example wiring.
- `Arc<T>` and `Box<T>` forwarding impls so embedders can be shared
  across async request handlers.

### Added — type stubs (Python) via `pyo3-stub-gen`

- `cargo run --bin gen-stubs` from `sphereql-python/` emits
  `python/sphereql/__init__.pyi` — 800+ lines covering every exposed
  API. Stubs ship with the wheel so IDEs / mypy / pyright get full
  completion and type-checking without extra setup.
- Dropped the stale hand-written `python/sphereql/sphereql.pyi` and
  top-level `sphereql.pyi.old` leftover.

### Added — typed WASM returns via `tsify`

- Every pipeline / category / metalearning method now returns typed
  values: `NearestOut`, `PathOut`, `PathStepOut`, `GlobOut`,
  `ManifoldOut`, `CategorySummaryOut`, `BridgeItemOut`,
  `CategoryPathOut`, `CategoryPathStepOut`, `DrillDownOut`,
  `InnerSphereReportOut`, `CategoryStatsOut`, `DomainGroupOut`,
  `ProjectionWarningOut`, `TuneReportOut`, `FeedbackSummaryOut`,
  `CorpusFeaturesOut`.
- `wasm-pack build` emits a `.d.ts` with a named interface for every
  payload. JS consumers receive real objects — no `JSON.parse` step.

### Added — bindings drift CI

- New `scripts/check-drift` workspace member: `syn`-parses
  `sphereql-embed` + `sphereql-layout` public APIs and fails when a
  new public `fn` / `struct` / `enum` isn't exposed via Python or
  WASM and isn't in `.bindings-ignore.toml`.
- Name matching tolerates common aliasing patterns (Py/Wasm prefix
  stripping, `Out`/`Info`/`Hit` suffix variants, `Result` / `Summary` /
  `Report` trimming, case-insensitive fallback).
- Initial allowlist covers ~90 intentionally-exempt items (config
  structs reached via dict, internal helpers, layout-crate types,
  foreign-trait objects) each with a `reason` field.
- `.github/workflows/bindings-drift.yml` runs on PRs that touch the
  relevant crates or the allowlist itself.

### Changed

- **`PipelineConfig` tolerates partial JSON**: every sub-config now
  carries `#[serde(default)]`, so `{"projection_kind": "LaplacianEigenmap"}`
  is valid on its own — no more "specify every knob". Benefits the
  WASM `newWithConfig`, Python `config={…}`, and any future REST /
  GraphQL config input.
- Documented the pipeline id-handling caveat: `CategorizedItemInput.id`
  is currently dropped during pipeline construction (the pipeline
  assigns its own `s-NNNN` ids). Callers must use those generated ids
  in follow-on queries; an upstream change will round-trip user ids.

### Docs

- Project-status, quickstart-python, quickstart-wasm, architecture,
  main README, and per-crate READMEs refreshed across the board.
- New GraphQL category-schema section in architecture.md with a
  closure-wired `TextEmbedder` example.

### Internal

- Workspace version bumped `0.1.0-alpha-2` → `0.2.0-alpha`; every
  path-dep `version = "0.1.0-alpha"` spec bumped in lockstep.
- 300+ new tests covering the above (Laplacian bindings, pipeline-
  level config, GraphQL resolvers, tsify round-trip, TextEmbedder,
  drift-check allowlist).

## [0.1.0-alpha] — earlier

Initial release. Core spherical math, spatial index, embedding
pipeline, category enrichment layer, GraphQL spatial queries, Python
(PCA / Kernel PCA) and WASM (PCA / Kernel PCA) bindings, vector DB
bridges (InMemory / Qdrant / Pinecone), auto-tuner + meta-model
framework (Rust-only at the time). See git history for full detail.

[0.2.0-alpha]: https://github.com/bkahan/sphereQL/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/bkahan/sphereQL/releases/tag/v0.1.0-alpha
