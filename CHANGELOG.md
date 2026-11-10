# Changelog

All notable user-visible changes. sphereQL follows semver from v1.0
onward; while on `0.x` expect breaking changes between minor
versions.

## [Unreleased]

## [0.4.0] — 2026-11-09

0.4.0 rather than 0.3.1: bridge classification changes for every
projection, `PipelineConfig`, `BridgeConfig` and `RadialStrategy` gain
public members (none are `#[non_exhaustive]`), and seeded tuner runs
pick different configs.

### Breaking

- **Bridge classification uses full-dimensional signals.** A bridge is
  `OverlapArtifact` when its two categories' full-dim centroids are closer
  than the `overlap_separation_quantile` of bridged pairs (new field,
  default 0.10). Otherwise it is `Genuine` when it goes to the item's
  nearest other category and `min(affinity_to_source, affinity_to_target)`
  is at least `genuine_affinity_ratio` (new field, default 0.85) times the
  `balanced_affinity_quantile` of the corpus's home affinities, and `Weak`
  if not. Without the 0.85 the floor is unreachable on tight corpora (no
  bridge on the 5k OpenAlex corpus qualified); in a blind review of 214
  bridges, 83–87% of those labeled `Genuine` were real cross-field links
  on both bundled corpora, against about 40% of the rest. 0.3.0 left every bridge `Weak` when
  the projection's EVR was below `min_evr_for_classification` (0.20), and
  each projection family reports a different quantity as EVR, so whether
  a corpus got classified at all depended on the projection; above the
  gate it used the S² territorial factor and the median bridge strength.
  Labels change for every projection, so anything keyed on
  `Genuine` bridges sees a different set. `min_evr_for_classification`
  and `overlap_artifact_territorial` are no longer read; both stay for
  serde compatibility.
- **New public members:** `PipelineConfig.radial` (`RadialConfig` /
  `RadialMode`), `BridgeConfig.overlap_separation_quantile`,
  `BridgeConfig.genuine_affinity_ratio`, and
  `RadialStrategy::MagnitudeStretch`. Struct literals without
  `..Default::default()` and exhaustive matches on `RadialStrategy` need
  updating. Saved configs still load (the new fields have serde defaults).
- **Seeded tuner runs pick different configs than 0.3.0.** Random
  search and the Bayesian warmup now take the projection kinds in
  turn, and each kind draws its knobs from its own seed-derived
  stream, so a kind's n-th trial gets the same knobs whatever else is
  in the space. Before, one RNG was shared across trials: adding a
  family (UMAP, say) reshuffled every later trial, and how many trials
  each kind got was left to chance. A seed still reproduces within a
  release; runs recorded under 0.3.0 won't replay exactly.

### Added — configurable radial coordinate

- **`PipelineConfig.radial` (`RadialConfig` / `RadialMode`)** — control how
  the radial coordinate `r` is assigned. `Magnitude` (default) is the
  faithful embedding-magnitude pass-through; `Stretch` percentile-maps the
  corpus magnitude band onto a configurable `[lo, hi]` range so `r` no longer
  clusters in a narrow near-maximum band on sparse/uniform corpora; `Fixed`
  pins all points to the shell. New `RadialStrategy::MagnitudeStretch` carries
  the corpus-derived bounds (resolved once at fit time, applied consistently
  to corpus points and queries). PCA's volumetric mode is auto-disabled when a
  non-default mode is chosen so the strategy takes effect. The
  `visualize_corpus` example gains `--radial lo:hi`.

### Added — `sphereql-vis` crate + first-class Rust visualization

- **New `sphereql-vis` crate** — a pure, data-agnostic 3D-visualization
  emitter: a serializable `Scene` (point cloud + `SceneStats` + an
  `#[non_exhaustive]` `Overlay` set — category centroids, classified
  bridges, slerp-interpolated geodesic concept paths, Voronoi territory
  caps, antipodes, coverage maps, domain-group spokes, globs, manifold
  slices) and a hardened HTML emitter. Depends only on `sphereql-core`.
- **Offline, self-contained output** — the emitted HTML inlines the
  Three.js + OrbitControls runtime (`to_html()`), so files open with no
  network; `to_html_cdn()` produces a smaller CDN-backed file. The old
  Python viewer claimed "self-contained" but loaded Three.js from a CDN.
- **Projection-aware stats** — the stats panel reports the pipeline's
  actual projection family and its matching quality metric (PCA variance
  / Kernel EVR / connectivity ratio / UMAP kNN-recall) instead of a
  hardcoded "PCA variance" label.
- **Umbrella `vis` feature** — `sphereql --features vis` re-exports the
  scene API as `sphereql::vis` (also in `full`).
- **`visualize_corpus` example** — loads a corpus, auto-tunes a
  projection, and renders the full overlay set to one offline HTML file
  (`--open` to launch a browser, `--cdn` for a smaller file).
- **Hardening** — centralized `<script>`-breakout escaping, a reduce-based
  min/max that fixes a latent large-N `RangeError`, non-finite-point
  filtering, deterministic per-category decimation for very large clouds,
  and a snapshot/round-trip/XSS/offline test suite.

### Added — out-of-core streaming viewer (`sphereql-vis-server`)

- **New `sphereql-vis-server` crate** — an axum + tokio query server for
  corpora too large to inline. It loads a corpus, embeds + projects it
  (gating O(n²) families to PCA above ~10k points, UMAP above ~100k), and
  holds the projected positions (spatial-indexed), the raw embeddings
  (ANN-indexed), and per-point metadata in memory. Serves
  `/manifest /tiles /points /nearest /category_stats /path /globs
  /drill_down /diagnostics /reproject /health`, plus the WASM studio at
  `GET /` when present (a landing page otherwise). `--corpus / --addr /
  --projection / --emit-html / --open / --allow-origin` CLI; live
  `/reproject` ("tune") atomically swaps server state, one build at a
  time, latest request wins. Same-origin unless `--allow-origin` says
  otherwise. 34 tests. Not published to crates.io.
- **Streaming contract in `sphereql-vis`** — new `manifest` and `tile`
  modules: a bounded `Manifest` (`Bounds` / `CategoryInfo` / `LodScheme` /
  `MANIFEST_VERSION`) and the binary **`SQT1`** point tile (`encode_tile` /
  `decode_tile` / `TilePoint` / `TileError`; 16-byte header + 20-byte
  records, little-endian). Bounded regardless of N.
- **Shared `viewer.js` runtime + instantiable refactor** — `viewer.js` is
  now a `createViewer(rootEl, opts)` factory shared by the offline HTML,
  the WASM studio, and the server front-end, with a strength channel
  (certainty → intensity → size/opacity), view-preserving live
  `updateScene`, and reasoning-chain rendering. The WASM studio can be
  served as the server's front-end at `/`.
- **Streaming client** — `ServerSource` / `TileCache` / `TileStreamer` /
  `connectToServer` stream tiles by viewport (off-thread decode, memory +
  IndexedDB cache keyed to the server and manifest), with a debugger UI
  (inspect, diagnostics, tune, filter) and share links that restore a
  streaming session.
- **Documentation** — see [`docs/visualization.md`](docs/visualization.md)
  for the full runbook, architecture, design choices, and API.

### Added — WASM studio and viewer tools

- **Studio:** a lingua-live mode (paste prose → concept graph → scene, in
  a Web Worker), a semantic query box, side-by-side compare and morph
  between projections, and the 775-point demo corpus preloaded.
- **Viewer tools:** great-circle ruler, PNG export, shareable `#v=` view
  links, density shading, globe pins, drag-drop of any `Scene` JSON, and
  GPU vertex transforms with id-buffer picking.
- **`js-tests/`** — headless tests that boot the real `viewer.js` in Node
  (14 suites), run in CI.

### Changed

- `sphereql-python`'s `visualize` / `visualize_pipeline` now render through
  `sphereql-vis` (API and signatures unchanged; output is offline and
  projection-aware). The `sphereql-examples` `e2e_transformer` example was
  migrated onto the shared crate, removing its duplicated inline template;
  its bespoke per-query/per-path buttons are consolidated into the shared
  viewer's click-to-inspect and overlay toggles.
- `SearchSpace::default()` and `SearchSpace::large_corpus()` search a
  single value of `overlap_artifact_territorial`, since classification no
  longer reads it. The default grid is a third the size, and the
  meta-model no longer learns from a knob with no effect.
- Glob auto-k scores its silhouette on an evenly strided 2,000-point sample
  above 2,000 points (it was O(k·n²) per candidate k, hours at 500k); the
  exact path is O(n²) and unchanged in value.
- `async-trait` 0.1.89 → 0.1.92, and two `chunks_exact` calls became
  `as_chunks`, for Rust 1.99's clippy (`double_must_use`,
  `chunks_exact_to_as_chunks`).

### Fixed

- **A NaN score could win `auto_tune`.** The best-trial check treated
  an unordered comparison as "not worse", so a NaN replaced a finite
  best. Trials scoring NaN or ±inf now go to `TuneReport::failures`
  and never rank; the tuner's comparators use `total_cmp`.
- **`run_self_tune` wasn't deterministic.** An item that bridges
  several categories took the label of whichever bridge a `HashMap`
  visited last, so two runs on the same corpus could reweight and
  prune different concepts. It now takes the label of its strongest
  bridge, ties going to the lower target category.
- `sphereql-vis-server`: `/globs` ran its clustering on a tokio worker,
  so on a large corpus it tied up a runtime thread for the whole run; it
  now runs in
  `spawn_blocking`, one request at a time. Overlapping `/reproject` builds
  could run at once (double the memory) and the slower one silently won;
  they now run one at a time and a superseded waiting request gets `409`.
- Viewer: a tile response that arrived after a re-projection or filter
  change could replace the new tiles with old ones; re-projection left the
  globe and overlays where the old projection put them; the tile-decode
  worker was never stopped.

### Security

- `sphereql-vis-server` sent `Access-Control-Allow-Origin: *`, so any site
  open in the browser could read `/points` (labels and raw vectors) and
  drive `/reproject`. The API is now same-origin unless `--allow-origin`
  names an origin; `--emit-html` adds `null` for its file:// page and says
  so.
- Viewer: the inspector put a server-supplied neighbor `row` into
  `innerHTML` unescaped, reachable from a shared `#v=` / `#server=` link.
  It is coerced to a number, and a link naming a server on another origin
  now asks before connecting.
- The tile cache was keyed by query string alone, so IndexedDB could serve
  tiles from another server, corpus or projection after a reload. Keys now
  carry the server and a manifest fingerprint, and IndexedDB holds one
  scene.
- The emitted HTML escaped only `</` in the scene payload; a label
  containing `<!--<script>` swallowed the rest of the page. `<`, `>` and
  `&` are now JSON-escaped.

## [0.3.0] — 2026-06-15

### Changed — UMAP projection overhaul (sphereQL-fit, 500k-ready)

- **Training items keep their optimized positions** — `project()` on a
  fitted corpus embedding returns the exact Adam-optimized position
  (certainty 1.0) via a bit-pattern fast path instead of re-deriving a
  softmax kNN average. Pipeline builds on UMAP's own corpus drop from
  O(N²·d) to O(N·d), and the tuner now scores the actual embedding
  rather than a kNN-smeared copy. Unseen embeddings transform through
  the RP-forest index the graph build already constructs (retained on
  `UmapGraph` at ≥ 2000 items).
- **EVR proxy replaced with kNN recall** — `explained_variance_ratio`
  now reports trustworthiness-style neighborhood preservation (mean
  overlap between spherical and original-space kNN sets) instead of
  the saturating fraction-below-median-random-distance proxy. UMAP
  records stored under the old proxy are not score-comparable.
- **Fuzzy simplicial edge weights** (canonical local distance scaling)
  — per-point ρ/σ calibration (`Σ exp(−(d−ρ)/σ) = log₂ k`, nearest
  edge weight 1.0), fuzzy-union symmetrization, with both attraction
  and per-edge negative draws scaled by the weight (matching
  `epochs_per_sample` in expectation; scaling attraction alone
  measurably halved recall).
- **Tunable `min_dist`** (default 0.1) — deterministic least-squares
  (a, b) curve fit, canonical generalized gradients (pinned to the old
  forms at a = b = 1), new `umap_min_dist` tuner axis, and a
  `ProjectionFitKey` component so prefit projections never collide
  across min_dist values. The 775-concept benchmark improved under
  the new default.
- **Stratified category term** — one same-category cohesion pair and
  one different-category separation pair per point per epoch; the old
  single uniform draw was ~97% repulsion at realistic category counts.
- **`warm_start_anchor`** (opt-in, default 0.0 = bit-identical no-op)
  — weak pull toward each point's PCA warm-start position so
  disconnected kNN components on sparse corpora keep their global
  arrangement instead of drifting under unopposed repulsion.

### Changed — ML-framework audit (tuner, metalearning, controller, pipeline)

- **`auto_tune` warm-starts from the meta-model** — Random and Bayesian
  strategies evaluate `base_config` as trial 0 (inside the budget; for
  Bayesian it seeds the TPE history), so a predicted config competes
  directly with searched ones. Grid keeps its exact-enumeration
  contract. `new_from_metamodel_tuned` docs now state the real
  semantics: the prediction supplies values only for knobs *not* in the
  `SearchSpace`.
- **Tuner trials stop cloning the corpus** — trials borrow the
  embedding matrix via an internal constructor instead of cloning it
  per trial (~3 GB/trial at 500k×768), and the winning pipeline is kept
  from its trial instead of being rebuilt (one fewer full projection +
  category-layer build per run).
- **Metric scoring stops re-exporting the corpus** — `ClusterSilhouette`
  and `GraphModularity` read a lazily-built, shared positions/category
  cache on the pipeline instead of allocating a fresh
  `Vec<ExportedPoint>` (with per-item `String` clones) on every call.
  `SpatialQuality` stores pairwise cap intersections in a `HashMap`
  (O(1) lookups; bridge detection was doing O(C²) scans per pair).
- **Statistical fixes** — meta-model feature z-scores use Bessel-
  corrected sample variance; tuner uniform sampling drops its modulo
  bias; `ClusterSilhouette` skips items coinciding with their centroid
  (a forced `s = 1.0` per category).
- **`run_self_tune` validates its config and returns `Result`** —
  smoothings/penalties must be in [0,1], boosts ≥ 1, finite
  `plateau_epsilon`; out-of-range smoothing can no longer silently zero
  out corpus quality. Plateau detection now fires *before* the
  iteration's reweight+prune, so a plateaued corpus is no longer
  mutated one extra unmeasured time. `SelfTuneConfig` is serializable.
  `reweight_in_place` docs no longer claim idempotency (repeated calls
  compound; use `reweight_from_base` with an invariant base).
- **Hardened API boundaries** — `MetaModel::is_fitted` +
  `new_from_metamodel{,_tuned}` return `InvalidInput` for unfitted
  models instead of panicking through a `Result`;
  `SphereQLPipeline::to_json` returns `Result` instead of panicking on
  non-finite coordinates (wasm/python bindings updated);
  `nearest_by_embedding` uses a bounded heap (O(N log k));
  `MetaTrainingRecord::append_to`'s legacy-store migration shares
  `feedback`'s locked tempfile+rename path (was an unsynchronized
  read/rewrite). `CorpusQuality` reports per-component breakdowns to
  `TrialRecord`; `SpatialQuality::compute` is deprecated in favor of
  `compute_with_config`.

### Changed — training loop (self-tune controller + metalearning)

- **Self-tune closes the quality→geometry loop** — `run_self_tune`'s
  PCA builds now weight each concept's covariance contribution by its
  current `quality` (floored, combined with the `1/√|category|`
  rebalancing). Previously reweighting mutated a field the pipeline
  never read, so the composite could only move via pruning and the
  plateau stop fired at iteration 2 by construction.
- **Reweighting is idempotent** — quality is recomputed from the
  run-entry base each iteration instead of compounding; the static
  attenuation (home affinity / source confidence) applies once per
  run, not once per iteration. `SelfTuneReport` gains
  `final_composite` — the score of the corpus actually persisted.
- **`CorpusQuality` bridge sub-score delegates to the canonical
  `BridgeCoherence`** (neutral floor included): low-EVR corpora no
  longer pin 30% of the controller objective at zero.
- **`TrialRecord` gains per-component metric breakdowns**
  (`components: Vec<(name, weight, score)>`) recorded in one pass via
  the new `QualityMetric::score_with_components` — flat-landscape
  diagnosis straight from the `TuneReport`.
- **`MetaTrainingRecord` gains `score_lift`** — `(best − mean)/(1 −
  mean)` over the run's trial distribution; cross-corpus-comparable
  evidence that `DistanceWeightedMetaModel` now ranks on (raw
  `best_score` fallback for legacy records, which keep deserializing
  via `#[serde(default)]`).
- **Meta-models**: scale features (`n_items`, `n_categories`, `dim`,
  `mean_members_per_category`) are `ln(1+x)`-compressed before
  z-scoring; mixed-metric training sets are stratified to the dominant
  `metric_name` at fit time; new
  `NearestNeighborMetaModel::predict_blended(features, k)` aggregates
  per-knob medians + majority projection kind over the k nearest
  records (`k = 1` reproduces `predict`).

### Changed — review pass on the 500k-corpus branch

- **`GraphModularity` scales to large corpora** — k-NN edge construction
  switches to the RP-forest ANN index at ≥ 2000 items (the exact
  all-pairs scan remains below that, so small-corpus scores are
  bit-identical). Previously the metric was O(N²) in angular-distance
  evaluations, which made every `default_composite` tuner trial
  infeasible at 100k–500k items.
- **`UmapSphereProjection::fit` deduplicated** — now delegates to
  `UmapGraph::build` + `fit_from_graph` instead of carrying a duplicate
  copy of the validation block and Adam optimizer loop. Numerically
  identical (pinned by `fit_from_graph_matches_full_fit`).
- **`AnnIndex::build_normalized` validates per-row dimensionality** like
  `build`; ragged input now fails fast with a clear panic message
  instead of an index error deep in `dot()`.
- **Weighted-PCA documentation corrected** — `w = 1/sqrt(|category|)`
  gives a category of size m total covariance mass √m (square-root
  softening of imbalance), not the previously documented "equal mass";
  exactly equal mass would require `w = 1/|category|`. Behavior is
  unchanged.
- **`default_nearest` docs match the code again** — the high-EVR
  routing bypass is documented and its threshold is the named constant
  `HIGH_EVR_ROUTING_BYPASS` (0.90) instead of an inline literal.
- Sort comparators in `ann.rs` / `umap.rs` use `total_cmp` (matching
  the wave-3 convention); tuner module docs updated for the Bayesian
  TPE-lite strategy and the UMAP prefit/graph-cache contract.

### Added — corpus and bulk ingestion

- **DBpedia 500K corpus** — `sphereql-corpus/data/dbpedia_500k.parquet` with a
  clustered variant (`dbpedia_500k.clustered.parquet`). Built from the DBpedia
  TTL dump via the new `DBpediaTtlSource` ingestor.
- **Wikidata 50K corpus** — `sphereql-corpus/data/wikidata_50k.parquet` with
  checkpoint resume support (`wikidata_50k.parquet.checkpoint.json`).
- **Bulk ingestion pipeline** (the `bulk_ingest` example in
  `sphereql-examples`) — end-to-end orchestrator
  covering download → parse → embed → cluster → tune → validate. Corpus sources
  implement a shared trait so DBpedia and Wikidata share the same pipeline path.
- **Emergent clustering** — density-based cluster discovery on the projected
  sphere, emitting a `.clusters.json` sidecar that feeds downstream validation.
- **Bulk validator** — automated quality gate that rejects corpora below a
  minimum `QualityMetric` threshold before writing the final parquet.
- **`[bulk]` config table** in `sphereql-corpus/config.toml` for soft self-tune
  defaults (budget, strategy, metric weights) tunable per corpus source.
- **Corpus self-tune pass** (`run_self_tune`) — runs the auto-tuner on an
  ingested corpus and writes the winning `PipelineConfig` into the parquet
  metadata, so consumers can load a pre-tuned pipeline without rerunning the
  tuner.
- **Python / Qdrant end-to-end integration** — `examples/qdrant_e2e.py`
  demonstrates loading a pre-tuned parquet corpus into a Qdrant collection and
  running hybrid spherical + cosine search through the Python bindings.

### Added

- **`SphereQLPipeline::raw_embeddings()`** — returns the original high-dimensional embeddings when the `retain-embeddings` feature is active; `None` otherwise. Aligned with `ids()`, `categories()`, and `projected_points()`.
- **`SphereQLPipeline::embedding_dim()`** — returns the dimensionality of retained embeddings, or 0 if not retained.
- **`SphereQLPipeline::pairwise_similarities()`** — computes the upper-triangle pairwise cosine similarity matrix from retained embeddings. Returns `None` without the feature.
- **`SphereQLPipeline::nearest_by_embedding()`** — finds the k nearest corpus concepts to a query embedding by cosine similarity in the original embedding space. Returns `None` without the feature.
- **`sphereql_core::pairwise_cosine_similarities()`** — batch upper-triangle pairwise cosine similarity with precomputed norms.
- **`sphereql_core::upper_triangle_index()`** — index helper for the flat upper-triangle vector returned by `pairwise_cosine_similarities`.
- **`retain-embeddings`** feature flag in `sphereql-embed` and `sphereql`; included in the `full` feature aggregate.

- **`sphereql-lingua`** — six-stage pipeline crate that turns free-form
  text into a `ConceptGraph` with every node placed at a SphereQL
  `(r, θ, φ)` position (concept extraction → θ domain assignment → φ
  abstraction resolution → r salience scoring → relation encoding →
  graph assembly). Built on `sphereql-core` so coordinate convention
  and distance math are shared with the rest of the workspace.
- **`lingua-spherica`** (Python) — thin skeleton package exposing the
  coordinate types (`SphericalPoint`, `Concept`, `Relation`,
  `ConceptGraph`, `DomainAnchor`) and spherical-math helpers
  (`angular_distance` Vincenty form, `slerp` with antipodal branch,
  weighted spherical centroid, semantic-distance combiner). The full
  text → graph pipeline lives in the Rust `sphereql-lingua` crate;
  this package is intentionally limited to types + math.
- `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1.
- README "How sphereQL compares to vector databases" section
  positioning against FAISS / Qdrant / Milvus / Weaviate / pgvector.
- `docs/performance.md` projection-fit cost disclosure (KPCA at
  n=10k ≈ 85 minutes).

### Changed — Rust API (breaking)

- `sphereql_core::cosine_similarity` now returns
  `Result<f64, SphereQlError>` instead of `f64`; mismatched-dimension
  inputs produce the new `SphereQlError::DimensionMismatch` variant
  rather than silently returning a nonsense scalar. Existing callers
  must handle (or `?`-propagate) the `Result`.
- New `SphereQlError::DimensionMismatch { expected, got }` variant.
  `SphereQlError` is `#[non_exhaustive]`, but call sites that already
  match on it should add this arm.

### Changed — Rust API stability

- `#[non_exhaustive]` on `SphereQlError`, `IndexError`,
  `VectorStoreError`, and both `DistanceMetric` enums (vectordb +
  graphql) so future variants can be added without breaking external
  exhaustive matches.
- `#[must_use]` on the four index builders (`SpatialIndexBuilder`,
  `ShellIndexBuilder`, `CachedIndexBuilder`, `EmbeddingIndexBuilder`)
  so a forgotten `.build()` is now a compile-time warning.
- `Region` and `LuneSide` intentionally left exhaustive — they are
  sum types meant for full caller match.

### Changed — Python (`lingua-spherica`)

- Modernized to PEP 585 lowercase generics and `X | None` instead of
  `Optional[X]`; removed `typing` imports. `zip(..., strict=True)` on
  paired-iterable code paths so length mismatches surface as errors.
- `SphericalPoint(r, theta, phi)` field order matches the canonical
  Rust `sphereql_core::SphericalPoint::new` signature so positional
  construction round-trips between languages.
- `DomainAnchor.theta_range` now wraps endpoints into `[0, 2π)` and
  documents the wrap-around case (`lo > hi` ⇒ band straddles the
  seam, range is `[lo, 2π) ∪ [0, hi]`).

### Performance — Rust core

- Silhouette computation rewritten to a single O(n²) pass (down from
  the prior nested iteration that was effectively O(n³) on large
  inputs).
- `rayon` parallelization for pairwise overlap / distance scans in
  `sphereql-core::spatial`, gated behind a serial-threshold so small
  inputs don't pay the thread-pool cost. Adds `rayon` as a workspace
  dependency.
- Cached norms on hot distance paths (cosine / angular re-rank) to
  avoid recomputing the per-vector L2 norm on every comparison.
- `sphereql-embed` migration-lock map is now O(1) on insert/lookup
  via `IndexMap`-backed LRU (capacity 128).

### Performance — WASM

- Re-enabled `wasm-opt = ["-Oz", ...]` in the release profile (≈30–40%
  bundle size reduction).
- Switched the wasm32 global allocator to `lol_alloc::LeakingPageAllocator`
  (≈50 bytes vs the ~10 KB `dlmalloc` baseline).
- Panic-hook installation moved behind a new `debug` cargo feature so
  release builds don't pay for `console_error_panic_hook`.
- Finished the `tsify` migration: `LaplacianEigenmapProjection.project`
  and `projectBatch` now return typed `SphericalPointOut` /
  `SphericalPointBatchOut` values directly — no JSON parsing in JS.

### Fixed

- `sphereql-embed` migration locks now use a bounded LRU
  (`IndexMap`-backed, capacity 128) keyed by canonicalized path,
  preventing unbounded growth in long-running processes.
- Atomic feedback-file replacement via `tempfile::NamedTempFile::persist`
  with cleanup-on-failure, replacing the prior `fs::write` + `rename`
  pair.
- `sphereql-layout::ManagedLayout` now `debug_assert`s that the layout
  result entry count matches the input item count.
- PyO3 `json_to_py` propagates conversion errors with `?` instead of
  `unwrap()`.

### Docs

- Backfilled detailed READMEs for `sphereql-core`, `sphereql-index`,
  `sphereql-layout`, `sphereql-vectordb` (previously 7-line stubs).
- New READMEs for `sphereql-lingua` and `lingua-spherica`.
- `CONTRIBUTING.md` updated to point at the `gen-stubs` flow that
  produces `__init__.pyi` (the prior `sphereql.pyi` reference was
  stale).
- `docs/architecture.md`, `docs/project-status.md`, top README updated
  to include the lingua text → graph pipeline.

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

[0.4.0]: https://github.com/bkahan/sphereQL/releases/tag/v0.4.0
[0.3.0]: https://github.com/bkahan/sphereQL/releases/tag/v0.3.0
[0.2.0-alpha]: https://github.com/bkahan/sphereQL/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/bkahan/sphereQL/releases/tag/v0.1.0-alpha
