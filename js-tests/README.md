# js-tests — headless tests for the viewer runtime

The sphereql-vis viewer (`sphereql-vis/src/viewer.js`) and the studio worker
(`sphereql-wasm/studio/worker.js`) are browser JavaScript with no Rust around
their logic, so the Rust suite can't cover them. These tests close that gap:
they boot the **real** `viewer.js` in Node against compact Three.js + DOM stubs
(`harness.cjs`) and exercise its pure logic directly.

```sh
node js-tests/run-all.cjs        # all suites (what CI runs)
node js-tests/04-morph.test.cjs  # a single suite
```

`harness.cjs` evaluates `viewer.js` in a `vm` context with thin THREE/DOM
stubs (a `BufferAttribute` stub keeps the real typed array, so geometry loops
run on correctly-sized buffers) and exposes the module's internals for
assertions. `run(viewer, D, {globals})` merges extra globals into the sandbox
(e.g. a `fetch` / `indexedDB` stub for the `ServerSource` / `TileCache` suites).
The worker suite stubs the `wasm_bindgen` no-modules global and drives the
message protocol.

Coverage:

| suite | what it checks |
|---|---|
| `01-parse-scene` | drag-drop `parseScene` normalization, rebuild swap, **surface_radius ⟷ Rust parity** |
| `02-tools` | great-circle ruler (0/90/180° + on-shell antipodal arc), PNG export, shareable hash round-trip |
| `03-query` | semantic query `highlightByIds` (id resolution, geodesic fan, clear) |
| `04-morph` | morph endpoints (t=0→A, t=1→B, geodesic midpoint), unmatched/ rebuild reset |
| `05-embed-sync` | compare camera broadcast + echo guard, scene injection, locks, non-parent rejection |
| `06-density-pins` | density θ×φ histogram + toggle, pins + TOML round-trip |
| `07-worker` | worker protocol: queue-before-ready, lingua/corpus/query/load dispatch, errors |
| `08-tile-decode` | SQT1 binary tile `decodeTile` vs a **cross-language golden** (the exact bytes `tile.rs` `golden_bytes_match` asserts), incl. u16/u32 LE + error paths |
| `09-datasource` | `InlineSource` manifest/tiles/pointMeta/nearest + boot routes through it unchanged |
| `10-server-source` | `ServerSource` URL/method/body shaping vs the server routes, tile decode over a mock fetch, `TileCache` LRU + worker-decoder fallback |
| `11-transform` | GPU transform parity: `curPos` matches the spread/radial/morph formula, `applyTransform` pushes the uniforms the GLSL reads (so CPU mirror + GPU agree), and the pick-id codec round-trips |
| `12-tile-streamer` | streaming orchestration: `TileStreamer` camera→request mapping, base+detail working set, dedup, LRU eviction, load cancellation; `tileMeshSink` per-tile geometry + global-row pick ids; `safeColor` palette-injection guard |

`01` has an optional integration check that round-trips a real emitted page —
set `SPHEREQL_EMIT_HTML` to a file produced by `visualize_corpus` to enable it;
it is skipped (not failed) otherwise so CI needs no Rust artifacts.

These are logic tests, not a browser. A manual WebGL smoke-test of the studio
(`sphereql-wasm/studio/build.sh` → serve → paste/query/morph/compare) is still
the final check before shipping the studio.
