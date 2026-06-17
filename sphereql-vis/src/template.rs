//! Compile-time assets: the HTML template and the vendored three.js runtime.
//!
//! `three.min.js` and `OrbitControls.js` are the r128 (npm 0.128.0) builds,
//! MIT-licensed (license headers preserved in the files). They are inlined
//! into the emitted HTML by [`crate::emit`] when offline self-containment is
//! requested.

pub(crate) const TEMPLATE: &str = include_str!("template.html");
pub(crate) const THREE_JS: &str = include_str!("vendor/three.min.js");
pub(crate) const ORBIT_CONTROLS_JS: &str = include_str!("vendor/OrbitControls.js");
/// The viewer runtime, inlined into the page at `/*__SPHEREQL_VIEWER__*/`.
/// Kept in a separate file so the (future) WASM studio can load the exact
/// same implementation and the two can never drift.
pub(crate) const VIEWER_JS: &str = include_str!("viewer.js");
