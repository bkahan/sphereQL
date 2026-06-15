//! HTML emission: turn a [`Scene`] into a single self-contained document.
//!
//! All escaping and three.js injection lives here so every consumer
//! (Rust, Python, examples) gets identical, audited output. The template's
//! own `escHtml()` handles DOM insertion of category/label strings; this
//! module hardens the JSON payload against `</script>` breakout and injects
//! the runtime either inline (offline) or via CDN tags.

use crate::scene::Scene;
use crate::template;

/// Where the three.js runtime comes from in the emitted HTML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScriptSource {
    /// Inline the vendored runtime — fully offline, larger file.
    Inline,
    /// Load the runtime from a CDN — small file, requires network to view.
    Cdn,
}

const CDN_THREE: &str = "https://unpkg.com/three@0.128.0/build/three.min.js";
const CDN_ORBIT: &str = "https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js";

/// Render `scene` to a complete HTML document.
pub(crate) fn render_html(scene: &Scene, source: ScriptSource) -> String {
    let data_json = serialize_payload(scene);
    let scripts = match source {
        ScriptSource::Inline => inline_scripts(),
        ScriptSource::Cdn => cdn_scripts(),
    };
    // Order matters: substitute the title and the (large) script blob before
    // the data payload, so neither replacement scans the injected JSON.
    template::TEMPLATE
        .replace("__SPHEREQL_TITLE__", &escape_html_text(&scene.title))
        .replace("<!--__SPHEREQL_SCRIPTS__-->", &scripts)
        .replace("/*__SPHEREQL_DATA__*/", &data_json)
}

/// Serialize the scene to JSON, hardened for embedding inside a `<script>`.
///
/// On serialization failure (e.g. a hand-constructed scene carrying a NaN —
/// the builder filters these, but the fields are public) we fall back to a
/// valid empty payload so the page still loads rather than going blank.
fn serialize_payload(scene: &Scene) -> String {
    let json = serde_json::to_string(scene).unwrap_or_else(|err| {
        eprintln!("sphereql-vis: scene serialization failed: {err}");
        r#"{"title":"SphereQL Visualization","points":[],"overlays":[],"stats":{"projection_kind":"none","evr":0.0,"evr_label":"Explained variance ratio"},"surface_radius":1.0,"show_axes":false}"#.to_string()
    });
    // Prevent a crafted category/label from terminating the <script> tag.
    json.replace("</", "<\\/")
}

/// Escape plain text destined for an HTML text node (the `<title>`).
fn escape_html_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Inline `<script>` blocks carrying the vendored runtime (offline).
fn inline_scripts() -> String {
    format!(
        "<script>{}</script>\n<script>{}</script>",
        guard_script(template::THREE_JS),
        guard_script(template::ORBIT_CONTROLS_JS),
    )
}

/// CDN `<script src>` tags for the same pinned runtime.
fn cdn_scripts() -> String {
    format!("<script src=\"{CDN_THREE}\"></script>\n<script src=\"{CDN_ORBIT}\"></script>")
}

/// Neutralize any `</script` sequence inside inlined JS so it can't close the
/// enclosing tag. `<\/script` is equivalent JS (the backslash before `/` is a
/// no-op in the string/regex contexts where `</script` can legally appear).
fn guard_script(js: &str) -> String {
    js.replace("</script", "<\\/script")
}
