# sphereql-wasm

WebAssembly bindings for the [sphereQL](https://github.com/bkahan/sphereQL)
project.

Exposes the pipeline, category enrichment layer, and the metalearning
surface (`corpusFeatures`, `autoTune`, `NearestNeighborMetaModel`,
`DistanceWeightedMetaModel`, `FeedbackAggregator`) plus a standalone
`LaplacianEigenmapProjection` class to the browser via `wasm-bindgen`.
Construct a pipeline once with corpus data, then query it repeatedly
from JavaScript. All four projection families — PCA, kernel PCA,
Laplacian eigenmap, and the new UMAP-on-sphere — are reachable through
`Pipeline.newWithConfig` (`projection_kind: "UmapSphere"` etc.).

Pipeline query, category, and report methods return **typed** values
via [`tsify`](https://github.com/madonoharu/tsify) — the `.d.ts`
emitted by `wasm-pack build` has a named interface for every result
payload, and the JS side receives a real object (no `JSON.parse`).
Inputs, `config()`, `export_json()`, and `MetaModel.predict` still
speak JSON strings.

Coverage tracks the Rust crates closely but not 1:1 — the gaps are
listed in the workspace's
[`.bindings-ignore.toml`](https://github.com/bkahan/sphereQL/blob/main/.bindings-ignore.toml)
and enforced by a drift checker. `run_self_tune` (WASM `runSelfTune`) and
bridge relation-type annotations are now bound; the remaining gaps are
config/report sub-types surfaced as JSON and Rust-only traits.

## Example

A self-contained browser demo lives in [`examples/`](./examples/). It
walks through pipeline construction, `newWithConfig` with a Laplacian
projection, `autoTune` over a custom search space, `hierarchicalNearest`
vs plain nearest, `MetaModel.predict`, and `FeedbackAggregator`.

```bash
wasm-pack build --target web
cp -r pkg examples/pkg
cd examples && python3 -m http.server 8000
```

Open <http://localhost:8000/>. See
[`examples/README.md`](./examples/README.md) for detail.

## Status

Part of the sphereQL workspace, currently `0.3.0`. Pre-1.0:
expect breaking changes between minor versions. Not published to npm —
build locally with `wasm-pack`.

See the [main repository](https://github.com/bkahan/sphereQL) for full
documentation, examples, and architecture overview.
