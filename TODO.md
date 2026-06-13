# TODO — road to v0.3.0

Collected during the 2026-06-12 documentation audit. Items are grouped by
what blocks the release vs. what can ride along later. File:line refs are
as of branch `docs/v0.3.0-audit`.

## Resolved in the 2026-06-12 fix pass

Checked off below. Highlights: a new **version-drift checker**
(`scripts/check-versions`, wired into CI) now enforces that every
release-version string across manifests, READMEs, and docs agrees with
the canonical workspace / pyproject versions; `sphereql-corpus` was added
to the crates.io publish pipeline (lean-packaged via an `include`
allowlist so its dev datasets stay out of the tarball); the Python stubs
are regenerated at the `vectordb` feature level; and the CI-command
surface is now single-sourced from `CONTRIBUTING.md`. Remaining open
items are release-time mechanics, fresh benchmarks, and the larger
tech-debt projects.

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
- [x] **`pyproject.toml` `[qdrant]` extra is misleading** — DONE: removed the
  `[project.optional-dependencies] qdrant` block; the two docs that said
  `pip install sphereql[qdrant]` now point at `maturin --features qdrant`.
- [x] **Regenerate Python stubs with vectordb enabled** — DONE: `gen-stubs`
  now has `required-features = ["vectordb"]`, the CI invocations pass
  `--features vectordb`, and `__init__.pyi` was regenerated. The stubs now
  cover `InMemoryStore` / `VectorStoreBridge` (the wheel's surface).
  `QdrantBridge` / `PineconeBridge` stay behind their own features and are
  correctly absent from the default wheel and stubs.
- [x] **WASM browser demo likely broken** — DONE: removed `JSON.parse()` from
  the tsify/object-returning methods in `index.html`, kept it on the two
  genuine JSON-string returners (`config`, `predict`). Verified statically
  against the WASM return types; a browser run-through is still worthwhile.
- [x] **`meta_learn` example** — RESOLVED (premise was stale): the default
  `SearchSpace` already includes `LaplacianEigenmap` (it is `large_corpus()`,
  not `default()`, that is PCA+UMAP). Ran with `--release` and confirmed the
  stress profile routes to `laplacian_eigenmap` and the example exits 0.
  (`verify()` prints OK/MISS and never panics, so it could not hard-fail.)

## Version bump mechanics (at release time)

- [ ] Bump `[workspace.package].version` `0.2.0-alpha` → `0.3.0` and the
  independently pinned `sphereql-python/pyproject.toml` `0.2.0a0` (two
  places to bump). The drift rule now exists: `cargo run -p check-versions`
  fails if any manifest pin, README, or doc disagrees with these two.
- [ ] Update version pins/status lines in docs: root `README.md` install
  snippet, per-crate README status sections, `docs/architecture.md`.
- [ ] Cut CHANGELOG `[Unreleased]` → `[0.3.0]` with date and a new compare
  link in the footer (CHANGELOG.md:396-397).
- [ ] Publish to crates.io + PyPI (both currently live at 0.2.0-alpha /
  0.2.0a0).

## Stale benchmarks (every published number predates the UMAP overhaul)

- [x] Re-run benchmark + refresh tables — DONE: full run committed to
  `docs/benchmarks/2026-06-12.json` (durable home); `docs/performance.md`
  and `docs/benchmark-analysis.md` refreshed.
- [x] Add UMAP and the PQ re-rank path to `benchmark.rs` — DONE; both now
  measured (UMAP ~0.2 ms/query; PQ-rerank P@5≈0.59 at r=k·2).
- [x] PCA vs Laplacian vs UMAP tuner head-to-head on both corpora — DONE:
  `auto_tune.rs` now sweeps all three explicitly; `docs/empirical-findings.md`
  refreshed. **UMAP wins both corpora** (overturns the old PCA/Laplacian
  split). NOTE new follow-up: `meta_learn`'s stress_300 vs `auto_tune`'s
  stress corpus disagree on Laplacian — reconcile the two generators.
- [x] Re-measure tuner wall time at n=775, budget 24 — DONE: ~9.9 s
  (built-in) / ~4.1 s (stress), with UMAP in the space.
- [x] Re-measure KPCA fit + query at n=10k — DONE: fit ~138 s (≈2.3 min,
  **not** the stale ~85 min); query ~4.5 ms (not ~84 ms). In the committed JSON.

## Small code-doc fixes (one-liners, then regenerate stubs)

- [x] `projection_kind` docstrings — DONE: added `umap_sphere` in
  `sphereql-python/src/pipeline.rs`, `sphereql-python/src/vectordb.rs`,
  `sphereql-wasm/src/lib.rs`, the regenerated stub, and
  `sphereql-vectordb/src/bridge.rs` (the delegation target the getters call
  into — caught in a verification pass).
- [x] `sphereql-embed/src/lib.rs:4` — DONE: "random" → "UMAP".
- [x] `sphereql-embed/src/pipeline.rs` — DONE: `explained_variance_ratio` doc
  now covers all four families incl. UMAP's kNN-recall semantics.
- [x] `sphereql-vectordb/src/store.rs` — DONE: trait doc now lists
  `PineconeStore` (behind the `pinecone` feature).
- [x] `SearchSpace::large_corpus()` doc — DONE: "> 5000" → "> 10 000",
  matching the consumer threshold.
- [x] `sphereql-corpus/Cargo.toml` — DONE: added `readme = "README.md"` (plus
  an `include` allowlist so publishing stays under crates.io's size limit).

## Tech debt (not release-gating)

- ~~**Binding coverage drift**~~ — DONE: bound the Phase-7 self-tune surface
  (`run_self_tune` in Python + WASM, `RelationType` as a string field) and
  fixed a `check-drift` bug (path-qualified `tsify::Tsify` derives were never
  detected — WASM bindings 9→32); allowlist shrunk 122→113.
- ~~**CI command surface is triplicated**~~ — DONE: `CONTRIBUTING.md` is now
  the canonical command/CI reference; the README and `docs/testing.md` link
  to it instead of keeping their own copies.
- ~~**Hardcoded test counts**~~ — DONE: `scripts/check-docs` counts the real
  tests (859 Rust / 225 pytest) and fails if a stated floor over/understates;
  the README + project-status floors are now CI-enforced.
- ~~**README workspace table vs docs/architecture.md crate table**~~ — DONE:
  both tables now list every member (enforced by `check-docs`), and the
  hand-drawn ASCII graph was replaced with a Cargo.toml-derived edge list.
- ~~**`sphereql-wasm` ... zero tests**~~ — DONE: added a `#[wasm_bindgen_test]`
  smoke suite (7 tests incl. `runSelfTune`) + a `wasm-test` CI job
  (`wasm-pack test --node`).
- ~~**`lingua-spherica` parity**~~ — DONE (kept, not archived): added
  `tests/test_parity.py` cross-checking its coordinate math against the
  `sphereql` wheel; wired into CI. slerp/centroid/etc. have no
  python-reachable twin yet → documented skips.
- **docs/project-status.md "Known limitations" is framed entirely in EVR
  terms** — correct for PCA, but ages badly as UMAP (kNN-recall scoring)
  becomes the default on large corpora. Same for the mixed EVR/recall
  scales in docs/projections.md; a per-family quality-metric table would be
  cleaner.
- ~~**Doc code snippets aren't compile-checked**~~ — DONE: `scripts/check-doc-snippets`
  compiles the `rust` blocks in `docs/*.md` against `sphereql --features full`
  (5 real snippets checked, 8 illustrative ones tagged `rust,ignore`); CI-wired.
