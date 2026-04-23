# sphereql-wasm

WebAssembly bindings for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Exposes the embedding pipeline (including category enrichment) to the browser via `wasm-bindgen`. Construct a pipeline once with corpus data, then query it repeatedly from JavaScript.

The WASM bindings currently expose the PCA-based pipeline only; the Laplacian eigenmap projection, auto-tuner, and meta-model layers added in Rust are not yet surfaced to the browser.

See the [main repository](https://github.com/bkahan/sphereQL) for full documentation, examples, and architecture overview.
