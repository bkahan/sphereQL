//! Smoke test for the pipeline-output → visualization coordinate joins in
//! [`sphereql_examples::build_corpus_scene`].
//!
//! Examples are only *built* in CI, never *run*, so the overlay-join logic
//! (bridge endpoints, geodesic centroid joins, Voronoi caps, antipodes) would
//! otherwise never execute. This builds a tiny real pipeline and asserts the
//! emitted HTML embeds valid JSON with the expected overlays.

use sphereql::embed::{PipelineInput, SphereQLPipeline};
use sphereql_examples::build_corpus_scene;

/// A small but non-degenerate corpus: three categories, a handful of items
/// each, in a low-dimensional space so a pipeline builds quickly.
fn tiny_input() -> (Vec<String>, Vec<Vec<f64>>, Vec<&'static str>) {
    let dim = 8;
    let mut categories = Vec::new();
    let mut embeddings = Vec::new();
    let mut labels = Vec::new();
    let specs: [(&str, usize, &[&str]); 3] = [
        (
            "science",
            0,
            &["atom", "force", "energy", "quantum", "field"],
        ),
        ("cooking", 3, &["bread", "spice", "roast", "sauce", "knife"]),
        ("music", 6, &["chord", "rhythm", "tempo", "scale", "melody"]),
    ];
    for (cat, axis, names) in specs {
        for (k, name) in names.iter().enumerate() {
            let mut v = vec![0.0; dim];
            v[axis] = 1.0 + k as f64 * 0.05;
            v[(axis + 1) % dim] = 0.1 * k as f64;
            categories.push(cat.to_string());
            embeddings.push(v);
            labels.push(*name);
        }
    }
    (categories, embeddings, labels)
}

#[test]
fn build_corpus_scene_produces_valid_overlaid_scene() {
    let (categories, embeddings, labels) = tiny_input();
    let pipeline = SphereQLPipeline::new(PipelineInput {
        categories,
        embeddings,
    })
    .expect("pipeline builds");
    let evr = pipeline.explained_variance_ratio();

    let scene = build_corpus_scene("smoke", &pipeline, &labels, evr);

    // Points survived the join and carry the concept labels.
    assert_eq!(scene.points.len(), 15);
    assert!(scene.points.iter().any(|p| p.label == "quantum"));
    assert!(scene.surface_radius.is_finite() && scene.surface_radius > 0.0);

    // The full overlay set was constructed: at minimum one centroid per
    // category, plus the coverage map.
    let kinds: Vec<&str> = scene
        .overlays
        .iter()
        .map(|o| match o {
            sphereql::vis::Overlay::Centroid { .. } => "centroid",
            sphereql::vis::Overlay::Bridge { .. } => "bridge",
            sphereql::vis::Overlay::GeodesicPath { .. } => "geodesic_path",
            sphereql::vis::Overlay::VoronoiCap { .. } => "voronoi_cap",
            sphereql::vis::Overlay::Antipode { .. } => "antipode",
            sphereql::vis::Overlay::CoverageVoid { .. } => "coverage_void",
            sphereql::vis::Overlay::DomainGroup { .. } => "domain_group",
            _ => "other",
        })
        .collect();
    assert!(kinds.iter().filter(|k| **k == "centroid").count() >= 3);
    assert!(kinds.contains(&"coverage_void"));

    // The emitted HTML embeds parseable JSON and is offline.
    let html = scene.to_html();
    assert!(!html.contains("src=\"http"));
    // Anchored on the payload script's opening `<script>\nconst D=` (unique to
    // our data block — the inlined three.js never opens with `const D=`). The
    // payload is a single newline-free line, so read to the next newline.
    let marker = "<script>\nconst D=";
    let di = html.find(marker).expect("app script present") + marker.len();
    let line = &html[di..];
    let end = line.find('\n').expect("data line is terminated");
    let json = line[..end].strip_suffix(';').unwrap();
    let parsed: serde_json::Value = serde_json::from_str(json).expect("valid embedded JSON");
    assert_eq!(parsed["points"].as_array().unwrap().len(), 15);
    assert!(!parsed["overlays"].as_array().unwrap().is_empty());
}
