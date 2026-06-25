# sphereql-vis

Self-contained 3D visualization for the [sphereQL](https://github.com/bkahan/sphereQL)
project — a pure, data-agnostic leaf crate that depends on nothing but
`sphereql-core` (+ `serde`).

It owns three things:

1. **The `Scene` data model** — the serializable point-cloud-on-a-sphere payload
   every producer maps into (`Scene` / `ScenePoint` / `SceneStats`, plus a
   `#[non_exhaustive]` 9-kind `Overlay` set).
2. **The emitter** — turns a `Scene` into one hardened, offline HTML file.
3. **The runtime** — `viewer.js`, the single rendering implementation shared by
   the offline HTML, the WASM studio, and (by design) the streaming server.

The Python bindings, the Rust examples, and the umbrella `sphereql` crate all map
pipeline output into a `Scene` and call `to_html()`.

## What's here

- **`Scene` / `ScenePoint` / `SceneStats`** — the point cloud plus
  projection-quality metadata. The stats panel reports the right metric per
  projection family via `evr_label` (PCA variance / UMAP kNN-recall /
  Connectivity ratio), not a hardcoded string. `ScenePoint` carries both
  Cartesian and spherical forms, an optional stable `id` (for k-NN highlight +
  cross-scene morph), and optional `certainty`/`intensity` quality signals.
- **`Overlay`** — drawable S² structure: category centroids, classified bridges,
  geodesic/reasoning paths, Voronoi caps, antipodes, coverage/void maps,
  domain-group spokes, knowledge globs, and local manifold slices.
  `#[non_exhaustive]` so new kinds are additive.
- **Emitter** — `Scene::to_html()` inlines the runtime for a fully offline,
  emailable file; `Scene::to_html_cdn()` produces a smaller file that loads
  three.js from a CDN. All interpolated data is escaped against `<script>`
  breakout. `viewer.js` is inlined at the `/*__SPHEREQL_VIEWER__*/` placeholder
  so the baked HTML and the studio can never drift.
- **The Scene-JSON wire form** — `Scene::to_json` / `from_json`. The emitted page
  is not a frozen render: it rebuilds itself from a `Scene` at runtime, so the
  same file can load a different scene by drag-and-drop or from a live in-browser
  WASM pipeline. (Rust `from_json` is strict serde; the *permissive* loading —
  bare points array, xyz-or-rθφ derivation — lives in `viewer.js`'s `parseScene`.)
- **The out-of-core streaming contract** — the `manifest` and `tile` modules:
  - `Manifest` (+ `Bounds`, `CategoryInfo`, `LodScheme`, `MANIFEST_VERSION`) — a
    point-free scene descriptor that reuses `SceneStats` + `Overlay`, bounded
    regardless of N.
  - the binary **`SQT1`** point tile (`encode_tile` / `decode_tile` / `TilePoint`
    / `TileError` / `TILE_VERSION`) — a 16-byte header + 20-byte records.

  These are consumed by [`sphereql-vis-server`](../sphereql-vis-server), which
  streams tiles by viewport for million-point corpora.

## Example

```rust,ignore
use sphereql_vis::{Scene, ScenePoint, SceneStats};

let scene = Scene::builder()
    .title("My corpus")
    .points(points) // Vec<ScenePoint>
    .stats(SceneStats::new("umap_sphere", 0.91).with_label("UMAP kNN-recall"))
    .build();
scene.write_html("sphere_viz.html")?;
```

## Running the viewer

Three delivery paths share this crate's `Scene` model + `viewer.js` runtime:

1. **Offline single-file** — `Scene::to_html()` → one portable `.html`. ✅
2. **WASM studio** — live in-browser pipeline → scene → render
   (`sphereql-wasm/studio`). ✅
3. **Streaming server** — `sphereql-vis-server` streams SQT1 tiles. The server is
   complete and tested; the browser client is mid-migration (see below). ⚠️

The full runbook, architecture, design choices, and API reference live in
**[`docs/visualization.md`](../docs/visualization.md)**.

> **Current state:** a recent refactor made `viewer.js` an instantiable
> `createViewer(rootEl, opts)` factory (adding a strength channel, live
> `updateScene`, and reasoning-chain rendering) and, in the same change, removed
> the server/streaming client from the runtime. The offline + studio paths are
> intact and improved; the streaming browser client awaits a re-port against the
> new factory API. Details and the file:line review in `docs/visualization.md`.

## Third-party

The emitted HTML inlines [three.js](https://threejs.org) (r128) and its
`OrbitControls`, both MIT-licensed. The license headers are preserved in
`src/vendor/`. three.js is © its authors; sphereQL is not affiliated with the
three.js project.

## Versioning

Part of the sphereQL workspace, currently `0.3.0`. Public API is stable enough
to ship against, but reserve the right to break on minor bumps before 1.0. See
the workspace
[CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md).
