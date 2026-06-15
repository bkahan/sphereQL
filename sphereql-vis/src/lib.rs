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

mod emit;
mod template;

pub mod overlay;
pub mod scene;

pub use overlay::{CapRing, Overlay, cap_ring, half_angle_from_solid_angle};
pub use scene::{Scene, SceneBuilder, ScenePoint, SceneStats, on_surface};
