# sphereql-wasm

WebAssembly bindings for the [sphereQL](https://github.com/bkahan/sphereQL)
project.

Exposes the pipeline, category enrichment layer, and the full
metalearning framework (`corpusFeatures`, `autoTune`, `MetaModel`,
`FeedbackAggregator`) to the browser via `wasm-bindgen`. Construct a
pipeline once with corpus data, then query it repeatedly from
JavaScript.

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

See the [main repository](https://github.com/bkahan/sphereQL) for full
documentation, examples, and architecture overview.
