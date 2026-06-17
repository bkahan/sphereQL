//! Self-contained 3D visualization for sphereQL spherical layouts.
//!
//! `sphereql-vis` is a pure, data-agnostic emitter: it owns one serializable
//! [`Scene`] model and one hardened Three.js template, and knows nothing about
//! projections, pyo3, or wasm. Producers (the Python bindings, Rust examples,
//! the umbrella crate) map pipeline output into a [`Scene`] — a point cloud
//! plus any number of [`Overlay`]s (centroids, bridges, geodesic paths,
//! Voronoi caps, antipodes, coverage maps, domain groups, globs, manifold
//! slices) — and call [`Scene::to_html`] to get a single offline HTML file.
//!
//! The emitted document inlines the three.js runtime (no network needed) and
//! escapes all interpolated data, so it is safe to write, email, or open from
//! anywhere. Use [`Scene::to_html_cdn`] for a smaller file that loads three.js
//! from a CDN instead.
//!
//! ```
//! use sphereql_vis::{Scene, ScenePoint, SceneStats};
//!
//! let scene = Scene::builder()
//!     .title("demo")
//!     .points(vec![
//!         ScenePoint::from_spherical("science", "p0", 1.0, 0.0, 1.2),
//!         ScenePoint::from_spherical("cooking", "p1", 1.0, 1.5, 0.9),
//!         ScenePoint::from_spherical("science", "p2", 1.0, 3.0, 2.0),
//!     ])
//!     .stats(SceneStats::new("pca", 0.83).with_label("PCA variance"))
//!     .build();
//!
//! let html = scene.to_html();
//! assert!(html.contains("PCA variance"));
//! assert!(!html.contains("src=\"http")); // offline: no external script loads
//! ```
//!
//! # Scene JSON (the runtime contract)
//!
//! The emitted viewer is not a frozen render: it rebuilds itself from a
//! [`Scene`] at runtime, so the same page can load a different scene by
//! drag-and-drop (or, in the WASM studio, from a live in-browser pipeline).
//! [`Scene::to_json`] / [`Scene::from_json`] are the canonical wire form, and
//! the viewer's drag-drop loader accepts that shape:
//!
//! ```jsonc
//! {
//!   "title": "My corpus",
//!   "points": [
//!     // each point needs finite x/y/z OR finite r/theta/phi — the loader
//!     // derives the missing pair (x=r·sinφ·cosθ, y=r·sinφ·sinθ, z=r·cosφ;
//!     // θ=atan2(y,x)∈[0,2π), φ=acos(z/r)). cat/label default to "".
//!     {"id": "doc-1", "x": 0.9, "y": 0.1, "z": 0.4, "cat": "science", "label": "…"}
//!   ],
//!   "overlays": [ /* {"kind": "centroid"|"bridge"|… , …}, see Overlay */ ],
//!   "stats": {"projection_kind": "pca", "evr": 0.83, "evr_label": "PCA variance"},
//!   "surface_radius": 1.0,  // optional; defaults to the median ‖xyz‖
//!   "show_axes": false
//! }
//! ```
//!
//! The loader is permissive on input (a bare `points` array, or minimal points
//! with only coordinates, are accepted and normalized) but always renders the
//! full [`Scene`] shape internally.

mod emit;
mod template;

pub mod overlay;
pub mod scene;

pub use overlay::{CapRing, Overlay, cap_ring, half_angle_from_solid_angle};
pub use scene::{Scene, SceneBuilder, ScenePoint, SceneStats, on_surface};
