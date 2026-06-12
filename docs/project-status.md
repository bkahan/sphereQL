# Project status

sphereQL is at **v0.2.0-alpha**. The core API is functional and covered
by 850+ Rust tests plus 200+ pytest tests on the Python bindings, but
may change before 1.0. See [CHANGELOG.md](../CHANGELOG.md) for the
v0.1 → v0.2 diff and the unreleased UMAP / ML-framework audit waves.

## Bindings status

Python and WASM bindings are at full parity with the Rust surface:

- Projection families: PCA (category-weighted via `PipelineConfig`),
  Kernel PCA, **Laplacian eigenmap**, Random — plus **UMAP-on-sphere**,
  reachable through `PipelineConfig` / `auto_tune` in every binding
  (no standalone UMAP class in the Python/WASM bindings; the
  pipeline-level config is the supported surface).
- Pipeline queries: nearest, similar-above, concept path, glob detection,
  local manifold.
- Category enrichment: concept paths, category neighbors, drill-down,
  hierarchical nearest, domain groups, category stats.
- Metalearning: `auto_tune`, `NearestNeighborMetaModel`,
  `DistanceWeightedMetaModel`, `FeedbackAggregator`,
  `MetaTrainingRecord` default-store helpers. Quality-metric resolvers
  accept `bridge_diversity` alongside the original metric names.
  `auto_tune` warm-starts from the meta-model — the predicted config is
  evaluated as trial 0 (inside the budget) so it competes directly with
  searched configs. `MetaModel::is_fitted` plus error-returning (not
  panicking) constructors guard the unfitted case;
  `NearestNeighborMetaModel::predict_blended(features, k)` aggregates
  over the k nearest records, and records carry `score_lift` for
  cross-corpus-comparable ranking.
- Partial-config support: any `PipelineConfig` field can be omitted;
  missing keys fall back to defaults (no more "specify every knob").
- Hardened boundaries: `SphereQLPipeline::to_json` returns an error on
  non-finite coordinates instead of panicking; Python and WASM surface
  it as an exception.

**Python**: type stubs (`.pyi`) are auto-generated via `pyo3-stub-gen` —
IDE and `mypy`/`pyright` pick them up automatically.

**WASM**: return values are typed via `tsify` — `wasm-pack build` emits
a `.d.ts` with a named interface for every payload. No `JSON.parse`
step required on the JS side for pipeline, category, or metalearning
methods.

**Lingua (text → graph)**: `sphereql-lingua` exposes a six-stage pipeline
that converts free-form text into a `ConceptGraph` with every node placed
at a SphereQL `(r, θ, φ)` position — concept extraction, θ domain
assignment, φ abstraction resolution, r salience weighting, relation
encoding, graph assembly. The default pipeline runs on heuristic
extractors and needs no external dependencies; swap in an LLM-backed
`ConceptExtractor` for production fidelity. The Python `lingua-spherica`
package is a types-and-math skeleton only — the pipeline lives in Rust
and is reached from Python via the `sphereql-python` bindings.

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
      corpora (see [empirical findings](empirical-findings.md)), and
      `ProjectionKind::UmapSphere` (ANN-backed, category-supervised,
      tunable `min_dist` via the `umap_min_dist` tuner axis) is the
      tunable option for the 10k–100k range. Training items keep their
      exact Adam-optimized positions (no kNN re-derivation), UMAP
      quality is scored as kNN recall (trustworthiness-style
      neighborhood preservation) rather than an EVR proxy — old-proxy
      tuner records are not score-comparable — and the opt-in
      `warm_start_anchor` keeps disconnected kNN components anchored
      on sparse corpora.

  See [search-precision-roadmap.md](search-precision-roadmap.md) for
  tracked improvements.

- **Inner spheres require 20+ items per category.** Categories below
  this threshold fall back to the outer sphere for drill-down queries.
  This is by design — small categories don't benefit from a separate
  projection, and the threshold is configurable via
  `InnerSphereConfig::min_size`. For corpora with many singleton
  categories, `PipelineConfig::min_category_size` additionally excludes
  tiny categories from bridge/domain-group analysis while keeping their
  items indexed and queryable.

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

## Large-scale corpus support

The `sphereql-corpus/data/` directory contains bulk-ingested parquet corpora
built by the `tools/bulk_ingest` pipeline:

- **DBpedia 500K** — `dbpedia_500k.parquet` with clustered and pre-tuned
  variants; produced from the DBpedia TTL dump via `DBpediaTtlSource`.
- **Wikidata 50K** — `wikidata_50k.parquet` with checkpoint-resume support.

Each parquet file embeds the auto-tuned `PipelineConfig` in its metadata so
consumers can reconstruct a pre-tuned pipeline without rerunning `auto_tune`.
The bulk pipeline runs: download → parse → embed → cluster → auto-tune →
validate → write. A `run_self_tune` pass gates each corpus on a minimum
`QualityMetric` score before it is committed; it validates its config up
front and returns `Result` instead of silently zeroing quality on
out-of-range smoothings.

Scaling machinery that supports this path:

- **RP-forest ANN index** (`sphereql-embed::ann`) — deterministic
  approximate kNN backing UMAP graph construction and
  `GraphModularity` edge building at ≥ 2000 items.
- **UMAP graph caching** in the tuner — one kNN-graph build per unique
  `n_neighbors`, reused across epoch/weight sweeps
  (`TuneReport::umap_graph_builds` reports cache effectiveness).
- **Borrowed-corpus tuner trials** — trials share the embedding matrix
  instead of cloning it (~3 GB per trial at 500k×768), and the winning
  pipeline is kept from its trial rather than rebuilt.
- **Wall-time budgets** — `SearchStrategy::{Random, Bayesian}` accept
  `max_wall_secs` so large-corpus tuning can be time-boxed instead of
  trial-count-boxed.
- **Category-weighted PCA** and `min_category_size` for
  singleton-heavy bulk corpora.

## Roadmap

- Improve search precision at higher k values.
- Query-time ANN (HNSW or VP-tree) for better recall without
  brute-force fallback — the RP-forest currently covers fit-time kNN
  (UMAP, modularity); the query path is the remaining gap.
- Streaming/incremental PCA for large-scale datasets.
- Expose `sphereql-layout`'s managed layouts through the bindings.
