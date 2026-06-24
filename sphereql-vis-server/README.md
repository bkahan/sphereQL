# sphereql-vis-server

Out-of-core query server (axum + tokio) for the sphereQL streaming viewer.

The offline [`sphereql-vis`](../sphereql-vis) emitter inlines an entire `Scene`
into one HTML file — perfect for a portable demo, but it tops out around a few
hundred thousand points (one JSON blob, parsed once). For millions of points the
viewer talks to *this* server instead: the corpus, its projection, and the ANN +
spatial indexes live here in memory, and the browser holds only the visible
working set.

The contract is the pure `sphereql-vis` crate's `Manifest` + binary `SQT1` tiles
(`sphereql_vis::tile`) — bounded up front, streamed by viewport.

> **Status (branch `feat/vis-server`):** the server compiles, runs, and is fully
> tested (29 tests); its HTTP API is validated end-to-end. **The matching browser
> client was removed from `viewer.js` by a recent instantiable refactor and has
> not been re-ported**, so on this branch no in-tree front-end consumes `/tiles`,
> and `--emit-html` / the studio auto-connect inject a now-undefined
> `connectToServer`. The server is a solid foundation; the client is the gap. See
> [`docs/visualization.md`](../docs/visualization.md) for the full picture.

## Run

```sh
# API + auto-detected WASM studio front-end (the intended one-liner):
cargo run -p sphereql-vis-server -- --corpus stress --open

# API only, explicit bind:
cargo run -p sphereql-vis-server -- --corpus stress --addr 127.0.0.1:8080

# Also write a standalone offline viewer pre-wired to this server:
cargo run -p sphereql-vis-server -- --corpus stress --emit-html --open
```

| Flag | Default | Meaning |
|---|---|---|
| `-c, --corpus <name\|path>` | `stress` | Registry name (`hand_crafted`, `extended`, `full`, `stress`, `dbpedia_50k`, `dbpedia_500k`, `wikidata_50k`, …) or a path to a Parquet file. Case-insensitive; `-`/`_` interchangeable. |
| `-a, --addr <host:port>` | `127.0.0.1:8080` | Bind address. `0.0.0.0:`/`:::` are rewritten to a browser-reachable host for `--open`. |
| `-p, --projection <kind>` | `pca` | `pca` \| `umap_sphere` \| `laplacian` \| `kernel_pca`. Size-gated (below). |
| `-e, --emit-html [path]` | off (`sphere_viz.html` if bare) | Write a standalone offline viewer pre-wired to connect to this server. |
| `-o, --open` | off | Open the browser after the socket binds. |

## What it serves

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness (`ok`) |
| `GET /manifest` | bounded scene descriptor (stats, palette, overlays, bounds, LOD). Fetched once; size independent of N. |
| `GET /tiles` | binary `SQT1` tile of points in a viewport cone, stratified to an LOD budget. Query: `theta,phi,half_angle,budget,lod,cats,min_certainty`. |
| `POST /points` | lazy per-point metadata by row (label, category, certainty, intensity, raw vector) — the inspector payload. |
| `POST /nearest` | ANN neighbors of a row or query vector (over **raw** embeddings) — trace. |
| `GET /category_stats` | the palette (name → color → count). |
| `POST /path` | shortest route between two categories through the category graph. |
| `GET /globs` | concept-cluster detection over the projected cloud. |
| `POST /drill_down` | k-NN within one category (inner-sphere projection when available). |
| `GET /diagnostics` | projection-health: EVR, warnings, certainty/intensity histograms, low-certainty outliers. |
| `POST /reproject` | live re-projection ("tune") — rebuilds and atomically swaps server state. |
| `GET /` | the pre-built WASM studio (when found at `sphereql-wasm/studio/dist`), else absent. |

Full request/response shapes with real payloads are in the
[HTTP API reference](../docs/visualization.md#http-api-reference).

## How it works

`AppState::from_corpus(corpus, projection)` loads the corpus, embeds each item
(`embed(&features, 1000 + row)`), projects it through a `SphereQLPipeline`, and
keeps the heavy **by-N** artifacts in memory:

- a `SpatialIndex` (16×8 θ/φ sectors) over **projected** positions — cone/viewport tiling;
- an `AnnIndex` (`sphereql-embed::ann`) over **raw** embeddings — semantic neighbors;
- row-indexed `StoredPoint` metadata — the inspector;
- the full `SphereQLPipeline` — retained for the `/path`, `/globs`, `/drill_down` trace queries.

Only **bounded aggregates** go in the `Manifest`, so the up-front fetch is
independent of N. `AppState` is immutable after build; `/reproject` builds a fresh
one off the runtime (`spawn_blocking`) and atomically swaps the shared
`Arc<RwLock<Arc<AppState>>>`, so tile reads are never blocked on a rebuild.

### Projection gating

O(n²) families gate down to PCA above hard thresholds:

| Requested | Gated to PCA when |
|---|---|
| `LaplacianEigenmap`, `KernelPca` | `n > 10_000` |
| `UmapSphere` | `n > 100_000` |
| `Pca` | never |

### Tile decimation

`GET /tiles` runs: cone query → `cats`/`min_certainty` filter → **deterministic
stratified decimation** (proportional per category, even stride, ≥1 per category,
no RNG) → `encode_tile`. The per-tile ceiling is `MAX_TILE_POINTS = 200_000`.
Each record carries its **global row index**, so the client fetches metadata
lazily via `POST /points`.

## Hardening

Permissive CORS (the viewer is often a different origin), a 4 MiB POST body limit,
and a `CatchPanicLayer` so a handler fault returns 500 instead of dropping the
connection. `/nearest` and `/drill_down` length-check the query vector against the
embedding dim before touching the ANN index (a mismatched length would otherwise
panic `AnnIndex::query`).

## Tests

```sh
cargo test -p sphereql-vis-server   # 29 tests
```

3 state unit tests (gating, build-from-corpus, reproject) + 7 tile unit tests
(budget/cone/filter/determinism) + 19 in-process axum integration tests
(`tower::ServiceExt::oneshot`, no socket) covering every endpoint.

## Versioning

Part of the sphereQL workspace, `0.3.0`. Not published to crates.io. See the
workspace [CHANGELOG](../CHANGELOG.md).
