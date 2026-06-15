# sphereql-vis

Self-contained 3D visualization for the [sphereQL](https://github.com/bkahan/sphereQL) project.

A pure, data-agnostic emitter: it owns one serializable `Scene` model and one
hardened Three.js template, and depends on nothing but `sphereql-core`. The
Python bindings, the Rust examples, and the umbrella `sphereql` crate all map
pipeline output into a `Scene` and call `to_html()` to get one offline HTML
file.

## What's here

- **`Scene` / `ScenePoint` / `SceneStats`** — the point cloud plus
  projection-quality metadata (the stats panel reports the right metric per
  projection family via `evr_label`, not a hardcoded string).
- **`Overlay`** — drawable S² structure layered over the points: category
  centroids, classified bridges, slerp-interpolated geodesic paths, Voronoi
  territory caps, antipodes, coverage/void maps, domain-group spokes, knowledge
  globs, and local manifold slices. `#[non_exhaustive]` so new kinds are
  additive.
- **Emitter** — `Scene::to_html()` inlines the runtime for a fully offline,
  emailable file; `Scene::to_html_cdn()` produces a smaller file that loads the
  runtime from a CDN. All interpolated data is escaped against `<script>`
  breakout.

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
