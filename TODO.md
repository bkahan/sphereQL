# TODO — road to v0.3.0

Collected during the 2026-06-12 documentation audit. Items are grouped by
what blocks the release vs. what can ride along later. File:line refs are
as of branch `docs/v0.3.0-audit`.

## Release blockers / decisions

- [ ] **Missing tuned corpus artifacts** — the registry exposes
  `DBpedia50kTuned` / `DBpedia500kTuned` (`sphereql-corpus/src/registry.rs`)
  and CHANGELOG advertises `dbpedia_500k.clustered.tuned.parquet`, but no
  `*.clustered.tuned.parquet` exists in `sphereql-corpus/data/`. `load()` on
  those variants errors. Generate and ship the artifacts, or drop the
  registry variants before release.
- [ ] **Decide the published PyPI wheel's feature set**
  (`sphereql-python/pyproject.toml [tool.maturin]`). Today the wheel ships
  without `lingua` and without `qdrant`/`pinecone`, while older docs implied
  otherwise (READMEs now state the truth). Pick the v0.3.0 surface.
- [ ] **`pyproject.toml` `[qdrant]` extra is misleading** — it installs the
  Python `qdrant-client`, which the Rust `QdrantBridge` never uses, and
  cannot enable the Cargo feature (`sphereql-python/pyproject.toml:28-29`).
  Remove or repurpose.
- [ ] **Regenerate Python stubs with vectordb enabled** — `gen-stubs` has
  `required-features = ["embed"]` only (`sphereql-python/Cargo.toml:21-27`),
  so the shipped `__init__.pyi` omits `InMemoryStore` / `VectorStoreBridge` /
  `QdrantBridge` even though the wheel includes them.
- [ ] **WASM browser demo likely broken** —
  `sphereql-wasm/examples/index.html:124,157-158,190-191,205-206` still call
  `JSON.parse()` on methods that now return typed tsify values (throws
  `SyntaxError` on a JS object). Half-finished migration visible at line 262.
  Needs a browser run-through.
- [ ] **`meta_learn` example may fail its own verification** — it uses
  `SearchSpace::default()` (`sphereql-examples/examples/meta_learn.rs:45`)
  but asserts the stress profile routes to `LaplacianEigenmap` (line 131), a
  kind the default space no longer includes (PCA+UMAP only). Run it.

## Version bump mechanics (at release time)

- [ ] Bump `[workspace.package].version` `0.2.0-alpha` → `0.3.0` and the
  independently pinned `sphereql-python/pyproject.toml` `0.2.0a0` (two
  places to bump — consider a check-drift rule to keep them in sync).
- [ ] Update version pins/status lines in docs: root `README.md` install
  snippet, per-crate README status sections, `docs/architecture.md`.
- [ ] Cut CHANGELOG `[Unreleased]` → `[0.3.0]` with date and a new compare
  link in the footer (CHANGELOG.md:396-397).
- [ ] Publish to crates.io + PyPI (both currently live at 0.2.0-alpha /
  0.2.0a0).

## Stale benchmarks (every published number predates the UMAP overhaul)

- [ ] Re-run `cargo run -p sphereql-examples --example benchmark --release`
  on current code and refresh the tables in `docs/performance.md` and
  `docs/benchmark-analysis.md` — all retrieval numbers predate commit
  `e169f59` (hybrid re-rank fix), the index acceleration, PQ, and the UMAP
  overhaul. **Commit the resulting `benchmark_results.json`** — it has been
  lost twice; give results a durable home (e.g. `docs/benchmarks/<date>.json`).
- [ ] Add UMAP and the PQ re-rank path to `benchmark.rs` — neither is
  measured today.
- [ ] Re-run the PCA vs Laplacian vs UMAP tuner head-to-head on both corpora
  (must add `LaplacianEigenmap` to `projection_kinds` explicitly now) and
  replace the pre-`BridgeDiversity` stress-corpus scores
  (`docs/empirical-findings.md`).
- [ ] Re-measure tuner wall time at n=775, budget 24, with UMAP in the
  default space (`docs/performance.md:70-74`).
- [ ] Re-measure KPCA fit (~85 min) and query (~84 ms) at n=10k — both
  numbers survive only from a deleted results file.

## Small code-doc fixes (one-liners, then regenerate stubs)

- [ ] `projection_kind` docstrings list only pca/kernel_pca/
  laplacian_eigenmap — missing `umap_sphere`:
  `sphereql-python/src/pipeline.rs:190`, `sphereql-python/src/vectordb.rs:460`,
  `sphereql-wasm/src/lib.rs:119-120`, stub `__init__.pyi:570`.
- [ ] `sphereql-embed/src/lib.rs:4` — crate rustdoc says families are
  "(PCA, kernel PCA, Laplacian eigenmap, random)"; should be UMAP, not random.
- [ ] `sphereql-embed/src/pipeline.rs:825-830` — `explained_variance_ratio`
  doc says "all three" families and omits UMAP's kNN-recall semantics.
- [ ] `sphereql-vectordb/src/store.rs:8-10` — trait doc omits `PineconeStore`.
- [ ] `SearchSpace::large_corpus()` doc says "> 5000 items"
  (`sphereql-embed/src/tuner.rs:99`) while its only consumer switches at
  > 10 000 (`full_e2e.rs:358`). Pick one number.
- [ ] `sphereql-corpus/Cargo.toml` — add `readme = "README.md"` for
  consistency with the other publishable crates.

## Tech debt (not release-gating)

- **Binding coverage drift is real now** — `.bindings-ignore.toml` exempts
  the whole Phase-7 self-tune surface (`run_self_tune`, `SelfTuneConfig`,
  `CorpusQuality`, …) plus bridge `RelationType`. Docs no longer claim "full
  surface"; decide what actually gets bound for Python/WASM and shrink the
  allowlist.
- **CI command surface is triplicated** (README Contributing, CONTRIBUTING.md,
  docs/testing.md) and drifts independently — this audit re-synced all
  three; consider making CONTRIBUTING.md canonical and linking from the rest.
- **Hardcoded test counts** in README + docs/project-status.md were already
  divergent (450 vs 600; actual ~857 Rust + ~213 pytest). A generated badge
  or single source would stop the rot.
- **README workspace table vs docs/architecture.md crate table** — same
  facts maintained twice; the hand-drawn ASCII dependency graph already went
  stale once (missing graphql→embed edge). A check-drift docs rule could
  cover both.
- **`sphereql-wasm` has `wasm-bindgen-test` as a dev-dep but zero tests** —
  CI only verifies the crate builds for wasm32. Add browser-run tests or
  drop the dep.
- **`lingua-spherica`** is a pure-Python re-implementation of
  sphereql-core's coordinate math with no automated cross-language parity
  check. Once `lingua` ships in the default wheel, archive it or shrink it
  to types only (check downstream/notebook consumers first).
- **docs/project-status.md "Known limitations" is framed entirely in EVR
  terms** — correct for PCA, but ages badly as UMAP (kNN-recall scoring)
  becomes the default on large corpora. Same for the mixed EVR/recall
  scales in docs/projections.md; a per-family quality-metric table would be
  cleaner.
- **Doc code snippets aren't compile-checked** — the `fit`-returns-`Result`
  drift fixed in this audit would have been caught by a doctest-style
  harness over `docs/*.md`.
