# Visualization

sphereQL renders an embedding corpus as an **interactive 3D point cloud on a
sphere (S²)** — every projected item is a point, colored by category, with
overlays (centroids, bridges, geodesic paths, …) layered on the same shell.
There is one rendering runtime (`viewer.js`) and one serializable data model
(`Scene`); everything else is a way to *get a scene in front of that runtime*.

This document is the runbook **and** the architecture/design/API reference. It
is written with an eye toward extracting the viewer into a standalone, generic
embedding-visualization tool, so it is explicit throughout about **what is a
generic 3D-embedding concern vs. what is coupled to sphereQL's domain** (see
[Generic vs. sphereQL-coupled](#generic-vs-sphereql-coupled)).

---

## Current status (branch `feat/vis-server`, 2026-06)

The viewer is **mid-migration**. A recent refactor (`ef5d558`, "make viewer.js
instantiable") rewrote `viewer.js` into a clean `createViewer(rootEl, opts)`
factory and, in the same commit, **removed the entire server/streaming client**
(the `ServerSource` / `TileStreamer` / `connectToServer` layer, plus the ruler,
pins, PNG export, config chrome, and session restore). The result:

| Path | What it is | Status |
|---|---|---|
| **1. Offline single-file emit** | `Scene → to_html()` → one self-contained `.html` | ✅ Works, improved (instantiable, strength channel, live `updateScene`, reasoning chains). **Validated.** |
| **2. WASM studio** | Paste prose / corpus JSON → in-browser pipeline → scene | ✅ Works (boot keeps `window.rebuild`/`parseScene` compat globals + the `#embed` compare protocol). Build path has a sharp edge — see [§ Path 2](#path-2--the-wasm-studio). |
| **3. Server-backed streaming** | `sphereql-vis-server` streams binary tiles for million-point corpora | ⚠️ **Server compiles, runs, and is fully tested (29 tests); its HTTP API is validated. But the browser client was removed from `viewer.js`, so there is currently no in-tree front-end that consumes `/tiles`.** See [§ Path 3](#path-3--the-streaming-server) and [Known issues](#known-issues--gotchas). |

If you only need a picture of a corpus, **use Path 1** — it is solid. Path 3's
*server* is production-quality and a good foundation, but its *browser half* is
not wired up on this branch.

---

## Table of contents

- [The data model (`Scene`)](#the-data-model-scene) — the stable API everything maps into
- [Path 1 — Offline single-file emit](#path-1--offline-single-file-emit)
- [Path 2 — The WASM studio](#path-2--the-wasm-studio)
- [Path 3 — The streaming server](#path-3--the-streaming-server)
- [Architecture](#architecture)
- [Wire contracts](#wire-contracts) — Scene JSON, Manifest, SQT1 tiles
- [The viewer runtime (`viewer.js`)](#the-viewer-runtime-viewerjs)
- [HTTP API reference](#http-api-reference)
- [Design choices](#design-choices)
- [Generic vs. sphereQL-coupled](#generic-vs-sphereql-coupled) — the extraction map
- [Testing & validation](#testing--validation)
- [Known issues / gotchas](#known-issues--gotchas)

---

## The data model (`Scene`)

Everything funnels through one type. The pure `sphereql-vis` crate depends on
nothing but `sphereql-core` (+ `serde`); it owns the `Scene` model, the
hardened HTML emitter, and the `viewer.js` runtime.

```rust,ignore
use sphereql_vis::{Scene, ScenePoint, SceneStats};

let scene = Scene::builder()
    .title("My corpus")
    .points(vec![
        ScenePoint::from_spherical("science", "Newtonian mechanics", 1.0, 0.4, 1.2)
            .with_id("doc-1")
            .with_quality(/* certainty */ 0.83, /* intensity */ 1.7),
        ScenePoint::from_cartesian("cooking", "Sourdough", [0.2, 0.9, 0.3]),
    ])
    .stats(SceneStats::new("umap_sphere", 0.91).with_label("UMAP kNN-recall"))
    .max_points(200_000) // optional decimation cap
    .build();

scene.write_html("sphere_viz.html")?; // one offline file
```

### `ScenePoint`

A single projected item. Carries **both** Cartesian and spherical forms so the
viewer never recomputes them.

| Field | Type | Notes |
|---|---|---|
| `id` | `Option<String>` | Stable identity for k-NN highlight + cross-scene morph. Viewer keys on this; **falls back to array index** when absent. |
| `x,y,z` | `f64` | Display Cartesian (the projection's own radius, possibly volumetric). |
| `r,theta,phi` | `f64` | Spherical mirror (physics convention; see [coordinate-system](coordinate-system.md)). |
| `cat` | `String` | Category — drives color + legend. |
| `label` | `String` | Per-point hover/selection text; may be empty. |
| `certainty` | `Option<f64>` | *sphereQL projection-quality signal* — drives point **size**. |
| `intensity` | `Option<f64>` | *sphereQL signal* — drives **opacity**. |

Constructors: `from_spherical(cat, label, r, theta, phi)` (derives x/y/z via
`sphereql_core::spherical_to_cartesian`) and `from_cartesian(cat, label, [x,y,z])`
(derives r/θ/φ). Builder-style: `.with_id(id)`, `.with_quality(certainty, intensity)`.
`is_finite()` is the sanitization predicate the builder filters on.

### `SceneStats`

The stats-panel metadata. `evr` is the headline quality number in `[0,1]`; its
*meaning* is given by `evr_label` (so the panel reads "PCA variance" /
"UMAP kNN-recall" / "Connectivity ratio" per projection family, not a hardcoded
string). `SceneStats::new(kind, evr)` defaults the label to "Explained variance
ratio"; `.with_label(s)` overrides. `sampled_from` / `dropped_nonfinite` are set
automatically by the builder when it decimates or drops non-finite points.

### `Scene` and `SceneBuilder`

`Scene { title, points, overlays, stats, surface_radius, show_axes }`. Build via
`Scene::builder()`. The builder is the supported (sanitizing) path — `build()`:

1. Drops non-finite points (records the count in `stats.dropped_nonfinite`).
2. If `max_points(cap)` is set and exceeded, applies a **deterministic
   per-category stratified sample** (records `stats.sampled_from`). Every
   category keeps ≥1 point; no RNG, so the same input always yields the same
   sample.
3. Computes `surface_radius` (the shell the reference sphere + on-shell overlays
   sit on) as the **median `‖xyz‖`** when not set explicitly.

Render: `to_html()` (three.js inlined, fully offline), `to_html_cdn()` (smaller,
loads three.js from a CDN), `write_html(path)`. Serialize: `to_json()` /
`from_json()` (the canonical wire form — see [Wire contracts](#wire-contracts)).

### `Overlay`

Drawable S² structure layered over the points. `#[non_exhaustive]` (additive),
**9 variants** today:

| Variant | Carries | Generic? |
|---|---|---|
| `Centroid` | `pos, color, label, members` | generic (cluster marker) |
| `Bridge` | `from, to, color, strength, classification` | **sphereQL** (inter-category bridge classification) |
| `GeodesicPath` | `vertices, nodes/hops, edges, color` | semi (a geodesic polyline is generic; the reasoning-chain framing is sphereQL) |
| `VoronoiCap` | `center, half_angle` | generic (spherical cap) |
| `Antipode` | `centroid, antipode, label, color` | generic |
| `CoverageVoid` | `caps[], voids[]` | generic (coverage map) |
| `DomainGroup` | `centroid, members[], label, color` | **sphereQL** (domain-group hub) |
| `Glob` | `center, radius` | generic (cluster blob) |
| `ManifoldSlice` | `center, normal` | **sphereQL** (local manifold fit) |

Helpers: `on_surface(dir, radius)` places a direction on the shell;
`cap_ring` / `CapRing` / `half_angle_from_solid_angle` build cap geometry.

---

## Path 1 — Offline single-file emit

The portable demo: one HTML file, three.js inlined, no server, no network.
Drag-and-drop a different `Scene` JSON onto the page to reload it live.

### Rust

The end-to-end example loads a corpus, lets the auto-tuner pick a projection,
and emits the full overlay set:

```bash
cargo run -p sphereql-examples --example visualize_corpus --release -- \
    --corpus handcrafted --out target/sphere_viz.html
# flags: --corpus <handcrafted|stress>  --out <path>  --cdn  --open  --radial lo:hi
```

Validated run (`--corpus handcrafted`):

```text
Loading corpus: hand_crafted
  775 concepts loaded
Auto-tuning (budget=16) over [Pca, UmapSphere, LaplacianEigenmap, KernelPca]...
  winner: umap_sphere (score 0.3821, EVR 47.6%)
  radius range: [0.955, 2.126]
  scene: 775 points, 1020 overlays
Wrote .../target/sphere_viz.html (1122 KB)
(self-contained — opens offline, no network needed)
```

`build_corpus_scene` (in `sphereql-examples`) is the reference mapping from a
fitted `SphereQLPipeline` to a `Scene` — read it if you are wiring your own
producer. To emit a scene from arbitrary data, build `ScenePoint`s directly (see
[The data model](#the-data-model-scene)) and call `to_html()`.

Embed the result anywhere with an iframe:

```text
<iframe src="sphere_viz.html" style="width:100%;height:640px;border:0"></iframe>
```

### Python

```python
import sphereql
# offline HTML via the same sphereql-vis emitter
sphereql.visualize(categories, embeddings, title="My Embeddings")
```

`visualize()` / `visualize_pipeline()` map into a `Scene` and write one HTML
file, exactly like the Rust path. (The Python binding does not expose the
streaming server.)

---

## Path 2 — The WASM studio

A live, in-browser projection studio: paste prose and watch `sphereql-lingua`
place each concept on the sphere as you type, or switch to **Corpus JSON** mode
to run the full `sphereql-embed` pipeline client-side and pick a projection.
Everything runs via WebAssembly — no server. It is the **same `viewer.js`** as
the offline export (the studio shell is emitted from `Scene::to_html()`), so the
live studio and the baked file can never drift.

### Build & run

> ⚠️ **The build requires `build.sh`, not just the `build_studio` example.**
> `cargo run -p sphereql-wasm --example build_studio` emits only the HTML shells
> (`index.html`, `embed.html`, `demo-corpus.json`). The worker, drivers, and the
> wasm `pkg/` are assembled by `studio/build.sh`. Running only the example
> leaves `dist/` without `studio.js`, which (a) breaks the studio at runtime and
> (b) makes the server's studio auto-detect fail (see [Known issues](#known-issues--gotchas)).

```bash
cd sphereql-wasm/studio
./build.sh                                # → dist/{index,embed,compare}.html + studio.js + worker.js + pkg/
(cd dist && python -m http.server 8080)   # then open http://localhost:8080/
```

`build.sh` prefers `wasm-pack`, else falls back to `cargo build --target
wasm32-unknown-unknown --release` + `wasm-bindgen --target no-modules` (+
`wasm-opt -Oz` if present). The worker is a **classic** worker that
`importScripts` the no-modules glue — hence `--target no-modules`. A worker +
wasm must be served over `http(s)`; opening `dist/index.html` via `file://` will
not work. `dist/` is git-ignored.

> **Windows note:** `build.sh` is bash. Run it from Git Bash or WSL. The
> individual steps (`cargo run … --example build_studio`, `cargo build --target
> wasm32-unknown-unknown`, `wasm-bindgen`) are plain commands you can also run
> by hand in PowerShell if you prefer. *(Build not executed during this writeup
> — transcribed from `studio/build.sh` and `studio/README.md`.)*

### Architecture (studio)

```text
index.html ── inlined viewer.js (window.rebuild/parseScene) + studio chrome
   └─ studio.js (main thread) ──postMessage──▶ worker.js (classic worker)
        ▲                                          └─ importScripts pkg/sphereql_wasm.js
        └────────── Scene JSON ◀── LinguaStudio.process / Pipeline.buildSceneJson
```

`studio.js` debounces input (~320 ms), tags each request with an id, and drops
stale worker responses so the freshest paste wins. The worker returns **Scene
JSON strings**; the main thread runs them through the viewer's own `parseScene()`
before `rebuild()`, so a studio-built scene gets the same validation as a
dropped file.

**Compare & morph** (Corpus JSON mode): a *morph slider* interpolates every
point from its position in projection A to its id-matched position in B along the
shell; *side-by-side* (`compare.html`) runs two viewer iframes
(`embed.html#embed`), one shared worker feeding each a different projection, with
camera moves relayed through the parent so orbiting one orbits both. Both align
points by **stable id**.

---

## Path 3 — The streaming server

For corpora past a few hundred thousand points, inlining one JSON blob stops
scaling. `sphereql-vis-server` (axum + tokio) instead holds the corpus, its
projection, and the indexes **in memory**, serves a bounded `Manifest` up front,
and streams binary **SQT1** point tiles by viewport + LOD. The browser would
hold only the visible working set.

### Run

```bash
# API + auto-detected WASM studio front-end (the intended one-liner):
cargo run -p sphereql-vis-server -- --corpus stress --open

# API only, explicit bind:
cargo run -p sphereql-vis-server -- --corpus stress --addr 127.0.0.1:8080

# Also write a standalone offline viewer pre-wired to this server:
cargo run -p sphereql-vis-server -- --corpus stress --emit-html --open
```

| Flag | Default | Meaning |
|---|---|---|
| `-c, --corpus <name\|path>` | `stress` | Registry name (`hand_crafted`, `extended`, `full`, `stress`, `dbpedia_50k`, `dbpedia_500k`, `wikidata_50k`, …) or a path to a Parquet file. |
| `-a, --addr <host:port>` | `127.0.0.1:8080` | Bind address. `0.0.0.0:` / `:::` are rewritten to a browser-reachable host for `--open`. |
| `-p, --projection <kind>` | `pca` | `pca` \| `umap_sphere` \| `laplacian` \| `kernel_pca`. Gated by corpus size (see below). |
| `-e, --emit-html [path]` | off (`sphere_viz.html` if bare) | Write a standalone offline viewer pre-wired to connect to this server. |
| `-o, --open` | off | Open the browser after bind. Targets `http://<addr>/` (the studio, if found). |

Validated startup (`--corpus hand_crafted`):

```text
loading corpus 'hand_crafted' …
loaded 775 points, projection 'pca' (EVR 19.5%), 31 categories
serving on http://127.0.0.1:8080/ (Ctrl-C to stop)
```

### Projection gating

`assemble()` gates O(n²) families down to PCA above hard thresholds, so a large
corpus can never wedge the server on an infeasible projection:

| Requested | Gated to PCA when | Otherwise |
|---|---|---|
| `LaplacianEigenmap`, `KernelPca` | `n > 10_000` | kept |
| `UmapSphere` | `n > 100_000` | kept |
| `Pca` | never | always |

### ⚠️ The browser half is not wired up on this branch

The server's HTTP API is complete, tested (29 tests), and was exercised
end-to-end for this document (every endpoint below returns real data). **But**
the browser client that consumed it — `ServerSource`, `TileStreamer`,
`connectToServer`, the SQT1 JS `decodeTile`, the streaming inspector/diag/tune/
filter UI — was removed from `viewer.js` by the instantiable refactor and has
not been re-ported. Concretely on this branch:

- `--emit-html` and the studio auto-connect inject `connectToServer(url)`, which
  **no longer exists** → an uncaught `ReferenceError`; the page renders an empty
  (or demo) scene and never streams.
- The control chrome for connect/tune/filter still exists in `template.html` but
  is unwired.

The server is the right foundation; the client is the gap. The last working
client lives at `git show 2512d9b:sphereql-vis/src/viewer.js` and is documented
below under [the viewer runtime](#the-viewer-runtime-viewerjs) for re-porting.

---

## Architecture

```text
                            ┌──────────────────────────────────────────┐
   producers                │  sphereql-vis  (pure, core-only)         │
   (map data → Scene)       │                                          │
   ───────────────          │   Scene / ScenePoint / SceneStats        │
   sphereql-examples ─┐     │   Overlay (9 kinds)                      │
   Python binding ────┼────▶│   ── wire contracts ──                   │
   WASM pipeline ─────┘     │   Scene JSON  (to_json/from_json)        │
                            │   Manifest    (bounded scene descriptor) │
                            │   SQT1 tile   (encode_tile/decode_tile)  │
                            │   ── runtime ──                          │
                            │   viewer.js  (createViewer factory)      │
                            │   template.html + inlined three.js       │
                            └───────────────┬──────────────────────────┘
                                            │ inlined / served three ways
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
   1. offline .html              2. WASM studio                3. vis-server (axum)
   Scene::to_html()              dist/index.html               holds corpus+indexes
   drag-drop reload              live pipeline → rebuild()      streams SQT1 tiles
   (✅)                           (✅)                            (server ✅ / client ⚠️)
```

**Three crates / packages:**

- **`sphereql-vis`** (pure, depends only on `sphereql-core`) — the `Scene` model,
  the `Overlay` set, the HTML emitter, the `viewer.js` runtime, **and** the
  out-of-core contract (`manifest` + `tile` modules). It knows nothing about
  projections, pyo3, wasm, or HTTP.
- **`sphereql-vis-server`** (axum + tokio; depends on `core, index, embed, vis,
  corpus`) — loads a corpus, embeds + projects it, holds by-N artifacts in
  memory, serves the `Manifest` + SQT1 tiles + trace/tune endpoints.
- **`sphereql-wasm/studio`** — the in-browser studio that emits and serves the
  same `viewer.js`.

**Server internals** (`AppState::from_corpus` → `assemble`):

1. `corpus.load()` → concepts; per-concept `embed(&features, 1000 + i)` (the
   row-index seed convention that keeps labels/vectors/ANN/tiles positionally
   aligned).
2. Build a `SphereQLPipeline` with the gated projection; pull `exported_points()`
   (strict input order), EVR, and the category-enrichment layer.
3. Keep **by-N** artifacts server-side:
   - a `SpatialIndex<PointItem>` (16×8 θ/φ sectors) over **projected** positions, for cone/viewport tiling;
   - an `AnnIndex` (`sphereql-embed::ann`) over **raw** embeddings, for semantic neighbors;
   - `Vec<StoredPoint>` row-indexed metadata (label, cat, certainty, intensity, raw f32 vector) for the inspector;
   - the full `SphereQLPipeline`, retained so `/path`, `/globs`, `/drill_down` can run category-graph / glob / drill-down queries.
4. Put only **bounded aggregates** in the `Manifest` (stats, palette, centroid +
   domain-group overlays, bounds, surface_radius, LOD scheme) — its size is
   independent of N.

`AppState` is immutable after build; `/reproject` builds a fresh one off the
async runtime (`spawn_blocking`) and **atomically swaps** the shared
`Arc<RwLock<Arc<AppState>>>`, so concurrent tile reads are never blocked on a
rebuild.

---

## Wire contracts

Three serialization formats define the seam between producers, the server, and
the runtime. These are the real "API" for a generic embedding viewer.

### 1. Scene JSON (`Scene::to_json` / `from_json`)

The canonical payload — what the baked HTML inlines, what drag-drop accepts, and
what the WASM bridge produces.

```json
{
  "title": "My corpus",
  "points": [
    {"id": "doc-1", "x": 0.9, "y": 0.1, "z": 0.4,
     "cat": "science", "label": "…", "certainty": 0.83, "intensity": 1.7}
  ],
  "overlays": [ {"kind": "centroid", "pos": [..], "color": "#aed581", "label": "physics", "members": 25} ],
  "stats": {"projection_kind": "pca", "evr": 0.83, "evr_label": "PCA variance"},
  "surface_radius": 1.0,
  "show_axes": false
}
```

**Rust `from_json` is strict serde.** The *permissive* loading — accepting a bare
`points` array, or points with only `x/y/z` **or** only `r/theta/phi` (deriving
the missing pair via `x=r·sinφ·cosθ, y=r·sinφ·sinθ, z=r·cosφ`;
`θ=atan2(y,x)∈[0,2π)`, `φ=acos(z/r)`) — lives in `viewer.js`'s `parseScene()`,
the JS runtime loader, **not** in Rust. Producers that emit through Rust must
emit the full shape; the drag-drop / studio path is where the leniency applies.

### 2. Manifest (`GET /manifest`)

The point-free scene descriptor — fetched once; bounded regardless of N. Reuses
`SceneStats` + `Overlay`.

```json
{
  "format_version": 1,
  "title": "SphereQL — hand_crafted",
  "total_points": 775,
  "surface_radius": 0.4040,
  "bounds": {"min": [-0.55,-0.56,-0.44], "max": [0.53,0.53,0.55]},
  "stats": {"projection_kind": "pca", "evr": 0.1949, "evr_label": "PCA variance"},
  "overlays": [ {"kind": "centroid", "pos": [..], "color": "#aed581", "label": "physics", "members": 25} ],
  "palette": [ {"name": "anthropology", "color": "#4fc3f7", "count": 25} ],
  "lod": {"levels": 4, "base_budget": 20000}
}
```

### 3. SQT1 binary tile (`GET /tiles`)

A viewport cone of points, decimated to an LOD budget, as a compact little-endian
binary blob. `TILE_VERSION = 1`. Defined in `sphereql-vis/src/tile.rs`
(`encode_tile` / `decode_tile` / `TilePoint` / `TileError`). Verified empirically:
a whole-sphere tile of the 775-point corpus is 15516 bytes = 16-byte header +
775 × 20-byte records.

```text
Header (16 bytes):
  magic    [4]  "SQT1"  (0x53 0x51 0x54 0x31)
  version  u16  = 1
  flags    u16  = 0           (reserved; quantization lives here in a future v)
  count    u32  = number of records
  reserved u32  = 0

Record (20 bytes), repeated `count` times:
  x        f32  offset 0
  y        f32  offset 4
  z        f32  offset 8
  cat      u16  offset 12      (palette index = position in cat_names)
  _pad     u16  offset 14      = 0
  row      u32  offset 16      (GLOBAL row index → /points key, survives decimation)
```

Positions are exact f32 in v1 (no quantization). `decode_tile` validates the
magic (`BadMagic`), version (`UnsupportedVersion`), header length (`TooShort`),
and record-count vs. buffer length (`LengthMismatch`). The format is a **generic**
out-of-core concern; only the `cat` field's meaning (a sphereQL palette index) is
domain-coupled.

---

## The viewer runtime (`viewer.js`)

One file (`sphereql-vis/src/viewer.js`, ~1046 lines), inlined into the baked HTML
and the studio shell at the `/*__SPHEREQL_VIEWER__*/` placeholder by `emit.rs`, so
all delivery vehicles share one implementation.

### `createViewer(rootEl, opts)`

Each call returns an independent viewer with its own GPU/DOM/THREE state (so two
can run side by side for compare mode):

```text
const viewer = createViewer(rootEl, { onSelect(id){…}, preserveCamera });
// returns:
{ rebuild, updateScene, drawChain, highlightByIds,
  setMorphTarget, applyMorph, clearMorph, dispose, camera, applyViewHash }
```

- **`rebuild(scene, {preserveCamera})`** — full teardown + rebuild from a parsed
  scene. Resets view settings; builds the reference globe, points geometry,
  overlays, legend, labels, stats.
- **`updateScene(scene)`** — view-preserving update. Same N + categories →
  rewrites position/color/strength/`catDir` GPU buffers **in place** (keeping
  camera, spread, radial, scale, and the active selection); structural changes
  fall back to a camera-preserving `rebuild`. This is the live-update path the
  WASM studio drives on every keystroke.
- **`drawChain(chain)`** — renders a reasoning chain (`{vertices, color, nodes:
  [{id, pos, rel}]}`) as an animated draw-on gold line with hop markers and
  billboarded `-[rel]->` labels. Returns a `{clear}` handle.
- **`highlightByIds(ids)`** — semantic-query highlight; maps ids → indices via
  `idToIndex`, emphasizes matches, draws great-circle arcs from the top hit.
- **`setMorphTarget` / `applyMorph(t) / clearMorph`** — interpolate every point
  from projection A to its id-matched position in B along the shell (`t ∈ [0,1]`).
- **`dispose()`** — stops the rAF loop, disconnects the `ResizeObserver`, aborts
  all listeners (`AbortController`), disposes geometry/materials/textures and the
  GL context. Verified leak-free (the recent correctness pass also fixed a
  `CanvasTexture` leak in label sprites).

`parseScene(raw)` and `deriveStrength(p)` are module-level. `deriveStrength`
resolves `strength > certainty > intensity > 1.0` (each clamped to `[0,1]`) so the
viewer is robust to raw emitter JSON that has `certainty`/`intensity` but no
`strength`.

### Rendering math (generic S² display)

The spread/radial/morph transform exists in **three synchronized copies** —
verified in lockstep by the diff review:

- GLSL `sphTransform` (the GPU vertex shader, `VERTEX_TRANSFORM`), shared by the
  points and pick materials.
- CPU `curPos(i)` — exact JS mirror (picking, neighbor lines, labels).
- CPU `transformPos(p)` — the spread+radial-only mirror for static overlay coords.

Point **size** is `size·(0.3 + 0.7·strength)·330/(-mv.z)` and **opacity** scales
with strength, so high-certainty points bloom and low-certainty ones shrink/dim.
Picking is GPU id-buffer based (1×1 render target, 24-bit id readback) with a CPU
nearest-within-14px fallback. There is an optional 24×12 (θ×φ) density heatmap.

### The `#embed` compare protocol (generic)

Two viewer iframes sync cameras via `postMessage`: `sphereql-scene` (inject a
scene), `sphereql-cam` (6-float camera+target), `sphereql-lock` (disable
rotate/zoom), `sphereql-embed-ready`. The camera broadcast is **epsilon-gated**
so the OrbitControls damping that follows an *applied* update is not echoed back
into a feedback loop.

### The removed streaming layer (reference, pre-`ef5d558`)

These symbols are **not in the current file** but define the contract a re-ported
client must satisfy (last present at `2512d9b:sphereql-vis/src/viewer.js`):

- **`DataSource` seam** — `manifest()`, `tiles(params)`, `pointMeta(rows)`,
  `nearest(q,k)`; `InlineSource(scene)` (offline, no network) vs.
  `ServerSource(baseUrl)` (hits the axum API). The offline boot was
  `rebuild(new InlineSource(D).scene)` — byte-identical to today's path.
- **`decodeTile`** — the JS SQT1 decoder mirroring `tile.rs` (header + 20-byte
  records → `{positions, cats, rows}`); `makeWorkerDecoder` runs it off-thread
  with an inline fallback.
- **`TileStreamer`** — camera → cone tile requests; one persistent coarse base
  tile + LRU detail tiles, dedup/cancellation; `tileMeshSink` builds per-tile
  `THREE.Points` with global-row pick ids.
- **`connectToServer(url)`** — fetch manifest → `rebuild({...manifest, points:[]})`
  → palette legend → stream by camera; wires the tune/filter/diag UI.
- **`safeColor`** — allow-lists palette colors from an untrusted server manifest
  (CSS-injection guard).

---

## HTTP API reference

Base URL = the server's bind address. All responses are JSON except `/tiles`
(`application/octet-stream`, SQT1) and `/health` (`text/plain`). Permissive CORS;
POST bodies capped at 4 MiB; a panic catcher returns 500 rather than dropping the
connection. **Every example below is a real response from `--corpus hand_crafted`.**

### `GET /health` → `ok`

### `GET /manifest`
The bounded scene descriptor. See [Wire contracts § Manifest](#2-manifest-get-manifest).

### `GET /category_stats`
The palette: `[{ "name": "physics", "color": "#aed581", "count": 25 }, …]`.

### `GET /tiles`
A binary SQT1 tile of the points in a viewport cone, decimated to a budget.

| Query param | Default | Meaning |
|---|---|---|
| `theta` | `0.0` | cone axis azimuth (rad) |
| `phi` | `π/2` | cone axis polar angle (rad) |
| `half_angle` | `π` | cone half-angle; `≥ π` ⇒ whole sphere |
| `budget` | — | explicit point budget (takes precedence over `lod`) |
| `lod` | — | LOD level → `base_budget << lod` |
| `cats` | — | comma-separated palette ids to keep, e.g. `cats=0,3,5` *(sphereQL)* |
| `min_certainty` | — | keep points with certainty ≥ this *(sphereQL)* |

Filtering happens before a **deterministic stratified decimation** (proportional
per-category, even stride, ≥1 per category; hard ceiling `MAX_TILE_POINTS =
200_000`). `row` in each record is the global index for `/points`.

### `POST /points` — lazy per-point metadata
Request `{"rows": [0, 1]}` → (out-of-range rows silently skipped, capped at 4096):

```json
{"points": [
  {"row": 0, "label": "Newtonian mechanics", "cat": 25, "category": "physics",
   "certainty": 0.127, "intensity": 1.79, "x": 0.238, "y": -0.055, "z": 0.246,
   "vector": [0.683, 1.012, 0.005, "…128 dims"]}
]}
```

### `POST /nearest` — semantic neighbors (ANN over raw embeddings)
Request `{"row": 0, "k": 5}` **or** `{"vector": [..dim..], "k": 5}` (a vector of
the wrong length returns empty neighbors, never panics):

```json
{"neighbors": [{"row": 9, "similarity": 0.845}, {"row": 21, "similarity": 0.790}]}
```

### `GET /diagnostics` — projection-health dashboard *(sphereQL)*
`projection_kind`, `evr`, `evr_label`, `total_points`, `warnings[]`
(severity-tagged), 16-bin `certainty` + `intensity` histograms, and the 16
lowest-certainty `outliers` (where the projection is least faithful).

### `POST /path` — category-graph route *(sphereQL)*
`{"source": "physics", "target": "biology"}` → `{ "path": { "steps":
[{category_name, cumulative_distance, hop_confidence}], "total_distance",
"path_confidence" } }` (`path: null` if disconnected; unknown category → 400).

### `GET /globs` — concept-cluster detection *(sphereQL)*
`?k=` (fixed) or `?max_k=` (silhouette auto-select) → globs with `centroid`,
`member_count`, `radius`, `top_categories`.

### `POST /drill_down` — within-category k-NN *(sphereQL)*
`{"category": "physics", "k": 5, "vector": [..dim..]}` → results with `row`,
`label`, `distance`, `used_inner_sphere` (wrong-dim vector → 400).

### `POST /reproject` — live re-projection ("tune") *(sphereQL)*
`{"projection": "umap_sphere"}` → the new `Manifest`, after an atomic state swap.
Subject to the same size gating as `--projection`. Validated: reprojecting the
775-point corpus to `laplacian` returns EVR 0.92 ("Connectivity ratio").

### `GET /` (only when a built studio is present)
Serves `studio/dist/index.html` (auto-connect injected) with `ServeDir` fallback
for `studio.js`, `pkg/*`, `embed.html`, `compare.html`, `demo-corpus.json`.

---

## Design choices

**One `Scene` model, many producers.** The viewer rebuilds itself from a `Scene`
at runtime rather than being a frozen render. Drag-drop, the WASM studio, and (in
principle) the server all feed the same shape. This is the single most important
seam for extraction: a generic viewer needs only "give me a `Scene`."

**Offline self-containment.** `to_html()` inlines three.js + OrbitControls and all
data, so a scene is one emailable file that opens with no network. A test asserts
no external `src=`. `to_html_cdn()` trades that for a smaller file.

**XSS hardening.** Every foreign string (category, label, stats, palette color)
is escaped (`escHtml`/`textContent`) before DOM insertion, and the streaming
client's `safeColor` allow-lists palette colors from an untrusted server. Treat
this as a hard invariant when editing the runtime.

**A separate streaming server, not a bigger HTML file.** Inlining tops out around
a few hundred thousand points (one JSON blob, parsed once). Past that, the corpus
+ indexes live server-side and the browser holds only the visible working set.
The seam is the `DataSource` interface; `InlineSource` keeps the offline path
identical so the two never diverge in render behavior.

**Binary tiles (SQT1), not JSON.** A viewport of points is 20 bytes each, decoded
off-thread, cached in memory + IndexedDB. JSON per-tile would dominate the wire
and the main-thread parse budget.

**Deterministic decimation everywhere.** Both the offline builder
(`stratified_sample`) and the server tiler (`stratified`) use the same
proportional-per-category, even-stride, no-RNG algorithm with a ≥1-per-category
guarantee — so the legend never loses a category and tiles are byte-reproducible
(a test asserts identical bytes across builds).

**Instantiable viewer (`createViewer` factory).** The recent refactor moved from
module-level globals to a factory returning a handle, so multiple viewers coexist
(compare mode) and each owns its lifecycle (`dispose()` releases everything). This
is also the groundwork for shipping the viewer as a standalone embeddable
component.

---

## Generic vs. sphereQL-coupled

The extraction map. To lift `viewer.js` (and the `Scene` contract) into a generic
embedding-visualization tool, keep the left column; replace or strip the right.

| Concern | Generic (portable) | sphereQL-coupled |
|---|---|---|
| **Data model** | `Scene`, `ScenePoint{id,x/y/z,r/θ/φ,cat,label}`, `SceneStats{projection_kind,evr,evr_label}`, builder sanitization + stratified decimation | `certainty` / `intensity` quality fields and their size/opacity coupling; the per-projection-family `evr_label` meanings |
| **Overlays** | `Centroid`, `VoronoiCap`, `Antipode`, `CoverageVoid`, `Glob` (generic spherical geometry) | `Bridge` (+ `classColor` Genuine/Weak/OverlapArtifact), `DomainGroup`, `ManifoldSlice`, reasoning-chain `GeodesicPath` with `-[rel]->` labels |
| **Runtime** | `createViewer` lifecycle, OrbitControls, zoom-to-cursor, resize, dispose; `sphTransform`/`curPos` spread/radial/morph S² math; GPU id-picking; reference globe; density heatmap; `shellArc`; the `#embed` camera-sync protocol; `applyViewHash` camera restore | `deriveStrength` precedence; per-category spread pivots (`catDir`); domain solo/legend; bridge/reasoning-chain rendering |
| **Wire** | Scene JSON shape; `Manifest{bounds,surface_radius,lod}`; SQT1 tile format; the `DataSource` seam; `decodeTile`/`TileCache`/`TileStreamer` | `palette` as a corpus registry; `cat` as a palette index; `min_certainty` filter |
| **Server** | cone-query + LOD budget + stratified decimation; ANN neighbor index; `Arc<RwLock<Arc>>` atomic-swap | `CorpusId` registry + synthetic `embed`; `SphereQLPipeline` + `ProjectionKind` gating; category enrichment (`/path`, `/globs`, `/drill_down`); EVR/certainty diagnostics |

**A minimal generic viewer** would: keep `createViewer` + `parseScene` + the
transform/picking/globe; drop `deriveStrength`'s certainty/intensity precedence (or
rename to a neutral `weight`); keep the generic overlays and make the domain ones
opt-in; keep Scene JSON + SQT1 + the `DataSource` seam as the data API; and pair it
with *any* tile server that speaks `Manifest` + SQT1 (the sphereQL server being one
implementation). The cleanest cut line is the **`DataSource` seam** and the
**Scene/Manifest/SQT1 wire contracts** — they are already generic.

---

## Testing & validation

```bash
cargo test -p sphereql-vis -p sphereql-vis-server   # Rust: green (53 tests this branch)
```

- **`sphereql-vis`** — emit/golden/XSS/offline-self-containment + `Scene::to_json`
  panic-freedom on non-finite coords + tile/manifest round-trips.
- **`sphereql-vis-server`** — **29 tests**: 3 (gating, build-from-corpus, reproject)
  + 7 (tile budget/cone/filter/determinism) + 19 in-process axum integration tests
  (`tower::oneshot`, no socket) covering every endpoint, including the
  wrong-length-vector regression that must return empty (not panic).

**What is validated vs. not:**

- ✅ Rust crates compile and pass; the full server HTTP API returns correct data
  (exercised end-to-end for this doc); the offline emitter produces a valid
  1.1 MB self-contained file.
- ❌ **The `js-tests/` headless harness is currently broken** — all 14 suites fail
  at boot (`rootEl.querySelector is not a function`) because the harness predates
  the instantiable refactor and still targets removed module-globals. It needs a
  rewrite against the `createViewer` API before it can validate the runtime again.
- ⚠️ **WebGL / browser interaction cannot be auto-verified** — the actual GLSL
  render, hover/select, orbit-sync, studio paste→worker→render, and any streaming
  experience need a manual browser smoke-test.

---

## Known issues / gotchas

These are live on `feat/vis-server` as of this writing (see the review summary in
the PR for file:line detail):

1. **Streaming browser client orphaned.** `--emit-html` and the studio
   auto-connect inject `connectToServer(url)`, removed from `viewer.js` →
   uncaught `ReferenceError`, no streaming. The server API itself is fine.
2. **`js-tests` fully broken** — harness predates the refactor; needs a rewrite.
3. **Studio build is two steps.** `cargo run --example build_studio` alone does
   **not** produce `studio.js`; you must run `studio/build.sh`. The server's
   `find_studio_dir` uses `studio.js` as its sentinel, so a half-built `dist/`
   means the studio is silently not served at `/`.
4. **`--open` without a built studio opens a 404** — when no studio is found,
   the router has no `/` route, so `--open` (which always targets `http://<addr>/`)
   lands on a 404. Use the API directly, or build the studio first.
5. **Dead control chrome in `template.html`** — sliders/buttons/tabs for the
   removed tools (ruler, PNG, share, pin, connect, tune, filter) are still in the
   DOM but unwired. They degrade gracefully (every consumer is null-guarded) but
   the "Connect" UI is misleading.
6. **`#v=` share links restore camera only** — the streaming-session restore
   branch was dropped with the client.

For a generic extraction, items 1–2 and 5–6 are the natural cleanup boundary:
either re-port the streaming client against the `createViewer` API, or formally
drop the server path from the client and document the server as a standalone API.

---

## See also

- [Architecture](architecture.md) — the workspace crate graph.
- [Coordinate system](coordinate-system.md) — the `(r, θ, φ)` convention.
- [Projections](projections.md) — PCA / Kernel PCA / Laplacian / UMAP-on-sphere.
- [`sphereql-vis/README.md`](../sphereql-vis/README.md) — the crate.
- [`sphereql-vis-server/README.md`](../sphereql-vis-server/README.md) — the server.
- [`sphereql-wasm/studio/README.md`](../sphereql-wasm/studio/README.md) — the studio.
