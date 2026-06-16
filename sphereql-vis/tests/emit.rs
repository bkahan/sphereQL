//! Emission, hardening, and overlay-join tests for `sphereql-vis`.
//!
//! Assertions target *structure* (substrings, JSON validity, vector norms),
//! not full-file byte equality, so cosmetic template tweaks don't churn the
//! suite.

use sphereql_core::SphericalPoint;
use sphereql_vis::{Overlay, Scene, ScenePoint, SceneStats, on_surface};

fn sample_points() -> Vec<ScenePoint> {
    vec![
        ScenePoint::from_spherical("science", "alpha", 1.0, 0.1, 1.2),
        ScenePoint::from_spherical("science", "beta", 1.0, 0.6, 1.0),
        ScenePoint::from_spherical("science", "gamma", 1.0, 1.1, 1.4),
        ScenePoint::from_spherical("cooking", "delta", 1.0, 3.0, 2.0),
        ScenePoint::from_spherical("cooking", "epsilon", 1.0, 3.4, 1.8),
        ScenePoint::from_spherical("cooking", "zeta", 1.0, 2.7, 2.2),
    ]
}

fn sample_scene() -> Scene {
    Scene::builder()
        .title("Golden Scene")
        .points(sample_points())
        .stats(SceneStats::new("pca", 0.83).with_label("PCA variance"))
        .build()
}

#[test]
fn golden_substrings_present() {
    let html = sample_scene().to_html();
    // Six points serialized.
    assert_eq!(html.matches("\"label\":").count(), 6);
    // Projection-aware label, not a hardcoded string.
    assert!(html.contains("PCA variance"));
    assert!(html.contains("\"projection_kind\":\"pca\""));
    // Backwards-compatible `evr` JSON key is preserved.
    assert!(html.contains("\"evr\":"));
    // OrbitControls runtime is wired up.
    assert!(html.contains("THREE.OrbitControls"));
    assert!(html.contains("Golden Scene"));
}

#[test]
fn offline_has_no_external_script_loads() {
    let html = sample_scene().to_html();
    assert!(
        !html.contains("src=\"http"),
        "inline mode must not load external scripts"
    );
    assert!(!html.contains("src='http"));
    // The runtime must actually be present inline (a non-trivial blob).
    assert!(
        html.len() > 500_000,
        "expected inlined three.js, got {} bytes",
        html.len()
    );
}

#[test]
fn cdn_mode_emits_exactly_the_script_tags() {
    let html = sample_scene().to_html_cdn();
    assert!(html.contains("<script src=\"https://unpkg.com/three@0.128.0/build/three.min.js\">"));
    assert!(html.contains("OrbitControls.js\">"));
    // CDN mode must NOT inline the megabyte runtime.
    assert!(
        html.len() < 200_000,
        "cdn mode should be small, got {} bytes",
        html.len()
    );
}

#[test]
fn scene_serde_round_trips() {
    let sr = 1.0;
    let scene = Scene::builder()
        .points(sample_points())
        .overlay(Overlay::centroid(
            &SphericalPoint::new_unchecked(1.0, 0.5, 1.0),
            sr,
            "#4fc3f7",
            "science",
            3,
        ))
        .overlay(Overlay::geodesic_path(
            &[
                SphericalPoint::new_unchecked(1.0, 0.5, 1.0),
                SphericalPoint::new_unchecked(1.0, 3.0, 2.0),
            ],
            sr,
            8,
            "#fff",
            "science → cooking",
        ))
        .stats(SceneStats::new("umap_sphere", 0.91).with_label("UMAP kNN-recall"))
        .surface_radius(sr)
        .build();

    // Structural round-trip. Coordinates compare approximately, not bit-exact:
    // serde_json's default float parser is allowed to be off by 1 ULP (the
    // `float_roundtrip` feature makes it exact), which is irrelevant for a
    // visualization payload.
    let json = serde_json::to_string(&scene).unwrap();
    let back: Scene = serde_json::from_str(&json).unwrap();
    assert_eq!(back.points.len(), scene.points.len());
    assert_eq!(back.overlays.len(), scene.overlays.len());
    assert_eq!(back.stats.projection_kind, "umap_sphere");
    assert_eq!(back.stats.evr_label, "UMAP kNN-recall");
    assert!((back.stats.evr - scene.stats.evr).abs() < 1e-9);
    for (a, b) in scene.points.iter().zip(&back.points) {
        assert_eq!(a.cat, b.cat);
        assert_eq!(a.label, b.label);
        assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9 && (a.z - b.z).abs() < 1e-9);
    }
    // Overlay variants survive the round-trip in order.
    assert!(matches!(back.overlays[0], Overlay::Centroid { .. }));
    assert!(matches!(back.overlays[1], Overlay::GeodesicPath { .. }));
}

/// Extract the embedded `const D=<JSON>;` payload from emitted HTML.
///
/// Anchored on the app-unique `\nconst pts=D.points` line (now in the inlined
/// viewer runtime, no longer adjacent to the data line) so we never latch onto
/// a `const D=` inside the minified three.js blob. The payload is emitted as a
/// single physical line (`serde_json::to_string` is newline-free), so we take
/// the last `\nconst D=` before the app script and read just that one line.
fn embedded_payload(html: &str) -> &str {
    let pts_marker = "\nconst pts=D.points";
    let pi = html.find(pts_marker).expect("app script present");
    let before = &html[..pi];
    let di = before.rfind("\nconst D=").expect("data assignment present") + "\nconst D=".len();
    let line = &before[di..];
    let end = line.find('\n').expect("data line is terminated");
    line[..end]
        .strip_suffix(';')
        .expect("data line ends with ;")
}

#[test]
fn xss_payload_is_escaped_never_raw() {
    let evil = "</script><img src=x onerror=alert(1)>";
    let scene = Scene::builder()
        .points(vec![
            ScenePoint::from_spherical(evil, evil, 1.0, 0.1, 1.0),
            ScenePoint::from_spherical("b", "b", 1.0, 1.0, 1.0),
            ScenePoint::from_spherical("c", "c", 1.0, 2.0, 1.0),
        ])
        .stats(SceneStats::new("pca", 0.5))
        .build();
    let html = scene.to_html();
    // The raw closing tag must never appear from our payload; it is neutralized
    // to `<\/script`.
    assert!(!html.contains("</script><img"));
    assert!(html.contains("<\\/script><img"));
}

#[test]
fn nonfinite_points_are_dropped_and_json_stays_valid() {
    let mut pts = sample_points();
    // Inject a NaN coordinate via a hand-built point.
    pts.push(ScenePoint {
        id: None,
        x: f64::NAN,
        y: 0.0,
        z: 0.0,
        r: f64::NAN,
        theta: 0.0,
        phi: 0.0,
        cat: "broken".into(),
        label: "nan".into(),
        certainty: None,
        intensity: None,
    });
    let scene = Scene::builder()
        .points(pts)
        .stats(SceneStats::new("pca", 0.5))
        .build();
    assert_eq!(scene.points.len(), 6, "the NaN point must be filtered");
    assert_eq!(scene.stats.dropped_nonfinite, Some(1));

    // Payload must be valid JSON (no bare NaN tokens).
    let html = scene.to_html();
    let json = embedded_payload(&html);
    let parsed: serde_json::Value =
        serde_json::from_str(json).expect("embedded scene payload must be valid JSON");
    assert_eq!(parsed["points"].as_array().unwrap().len(), 6);
}

#[test]
fn overlays_land_on_the_surface_radius() {
    let sr = 2.5;
    let dirs = [
        SphericalPoint::new_unchecked(1.0, 0.3, 1.2),
        SphericalPoint::new_unchecked(1.0, 2.0, 0.7),
        SphericalPoint::new_unchecked(1.0, 4.0, 2.4),
    ];
    // Centroid placement.
    if let Overlay::Centroid { pos, .. } = Overlay::centroid(&dirs[0], sr, "#fff", "c", 1) {
        let norm = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!((norm - sr).abs() < 1e-9, "centroid off-shell: {norm}");
    } else {
        panic!("expected Centroid");
    }
    // Every geodesic vertex sits on the shell.
    if let Overlay::GeodesicPath { vertices, .. } =
        Overlay::geodesic_path(&dirs, sr, 6, "#fff", "p")
    {
        assert!(vertices.len() > 6);
        for v in &vertices {
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (norm - sr).abs() < 1e-9,
                "geodesic vertex off-shell: {norm}"
            );
        }
    } else {
        panic!("expected GeodesicPath");
    }
    // The free helper agrees.
    let p = on_surface(&dirs[1], sr);
    assert!(((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - sr).abs() < 1e-9);
}

#[test]
fn empty_scene_emits_valid_page() {
    let scene = Scene::builder().stats(SceneStats::new("none", 0.0)).build();
    assert_eq!(scene.points.len(), 0);
    assert_eq!(scene.surface_radius, 1.0);
    let html = scene.to_html();
    assert!(html.contains("empty corpus"));
}

#[test]
fn voronoi_cap_half_angle_is_finite_for_noisy_area() {
    // A solid angle beyond the full sphere (Monte-Carlo overshoot) must clamp,
    // never NaN.
    let ha = sphereql_vis::half_angle_from_solid_angle(20.0);
    assert!(ha.is_finite());
    let ha0 = sphereql_vis::half_angle_from_solid_angle(-1.0);
    assert!(ha0.is_finite());
}

#[test]
fn decimation_keeps_every_category() {
    let mut pts = Vec::new();
    for i in 0..1000 {
        pts.push(ScenePoint::from_spherical(
            "big",
            format!("b{i}"),
            1.0,
            0.001 * i as f64 % 6.0,
            1.0,
        ));
    }
    pts.push(ScenePoint::from_spherical("rare", "r0", 1.0, 0.2, 0.5));
    pts.push(ScenePoint::from_spherical("rare", "r1", 1.0, 0.3, 0.6));
    let scene = Scene::builder()
        .points(pts)
        .max_points(100)
        .stats(SceneStats::new("pca", 0.5))
        .build();
    assert!(scene.points.len() <= 110, "decimated near the cap");
    assert!(scene.stats.sampled_from.unwrap() == 1002);
    // The rare category survives.
    assert!(scene.points.iter().any(|p| p.cat == "rare"));
    assert!(scene.points.iter().any(|p| p.cat == "big"));
}

#[test]
fn geodesic_path_has_no_duplicate_vertices() {
    let dirs = [
        SphericalPoint::new_unchecked(1.0, 0.0, 1.0),
        SphericalPoint::new_unchecked(1.0, 1.5, 1.2),
        SphericalPoint::new_unchecked(1.0, 3.0, 0.8),
    ];
    // 2 segments × 4 samples + 1 final endpoint = 9 vertices, junction once.
    if let Overlay::GeodesicPath { vertices, .. } =
        Overlay::geodesic_path(&dirs, 1.0, 4, "#fff", "p")
    {
        assert_eq!(vertices.len(), 9);
        for w in vertices.windows(2) {
            let d = ((w[0][0] - w[1][0]).powi(2)
                + (w[0][1] - w[1][1]).powi(2)
                + (w[0][2] - w[1][2]).powi(2))
            .sqrt();
            assert!(d > 1e-9, "consecutive vertices must not coincide");
        }
    } else {
        panic!("expected GeodesicPath");
    }

    // A single centroid yields exactly one vertex (no double-add).
    if let Overlay::GeodesicPath { vertices, .. } =
        Overlay::geodesic_path(&dirs[..1], 1.0, 4, "#fff", "p")
    {
        assert_eq!(vertices.len(), 1);
    } else {
        panic!("expected GeodesicPath");
    }
}
