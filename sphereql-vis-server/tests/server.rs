//! End-to-end tests against the real axum router, driven in-process with
//! `tower::ServiceExt::oneshot` (no socket). Each test builds the router over a
//! freshly-projected stress corpus (300 points, in-memory — no Parquet file
//! needed) and exercises one endpoint through the full extract → handle →
//! serialize path.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use sphereql_corpus::CorpusId;
use sphereql_embed::ProjectionKind;
use sphereql_vis::{Manifest, decode_tile};
use sphereql_vis_server::{AppState, build_router};
use tower::ServiceExt; // for `oneshot`

const BODY_CAP: usize = 64 * 1024 * 1024;

fn router() -> axum::Router {
    let state =
        AppState::from_corpus(CorpusId::Stress, ProjectionKind::Pca).expect("stress builds");
    build_router(Arc::new(state))
}

/// Build the router and also surface the corpus's category names + embedding
/// dim, so the trace tests can address real categories / sized vectors.
fn router_with_meta() -> (Vec<String>, usize, axum::Router) {
    let state =
        AppState::from_corpus(CorpusId::Stress, ProjectionKind::Pca).expect("stress builds");
    let names = state.cat_names.clone();
    let dim = state.dim;
    (names, dim, build_router(Arc::new(state)))
}

async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    to_bytes(resp.into_body(), BODY_CAP)
        .await
        .expect("read body")
        .to_vec()
}

async fn get(app: axum::Router, uri: &str) -> axum::response::Response {
    app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn post_json(app: axum::Router, uri: &str, body: Value) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn health_ok() {
    let resp = get(router(), "/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(body_bytes(resp).await, b"ok");
}

#[tokio::test]
async fn manifest_round_trips_and_describes_the_corpus() {
    let resp = get(router(), "/manifest").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = body_bytes(resp).await;
    let text = String::from_utf8(bytes).unwrap();
    let manifest = Manifest::from_json(&text).expect("manifest parses");
    assert_eq!(manifest.total_points, 300);
    assert_eq!(manifest.format_version, sphereql_vis::MANIFEST_VERSION);
    assert!(!manifest.palette.is_empty());
    assert!(manifest.surface_radius > 0.0);
    let palette_total: usize = manifest.palette.iter().map(|c| c.count).sum();
    assert_eq!(palette_total, 300);
}

#[tokio::test]
async fn tiles_returns_a_decodable_binary_tile() {
    let resp = get(router(), "/tiles").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get(header::CONTENT_TYPE)
            .map(|v| v.to_str().unwrap().to_string()),
        Some("application/octet-stream".to_string())
    );
    let bytes = body_bytes(resp).await;
    let pts = decode_tile(&bytes).expect("valid SQT1 tile");
    assert_eq!(
        pts.len(),
        300,
        "whole-sphere tile under budget = all points"
    );
}

#[tokio::test]
async fn tiles_respects_budget() {
    let resp = get(router(), "/tiles?budget=40").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let pts = decode_tile(&body_bytes(resp).await).expect("valid tile");
    assert!(pts.len() <= 40, "got {} for budget=40", pts.len());
    assert!(!pts.is_empty());
}

#[tokio::test]
async fn points_returns_metadata_with_raw_vectors() {
    let resp = post_json(router(), "/points", json!({ "rows": [0, 5, 17] })).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let pts = v["points"].as_array().unwrap();
    assert_eq!(pts.len(), 3);
    assert_eq!(pts[0]["row"], 0);
    assert!(pts[0]["label"].is_string());
    assert!(pts[0]["category"].is_string());
    // 128-d raw embedding rides along for the inspector.
    assert_eq!(pts[0]["vector"].as_array().unwrap().len(), 128);
}

#[tokio::test]
async fn points_skips_out_of_range_rows() {
    let resp = post_json(router(), "/points", json!({ "rows": [0, 99999] })).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // The 99999 row does not exist (corpus is 300) and is dropped.
    assert_eq!(v["points"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn nearest_by_row_excludes_self() {
    let resp = post_json(router(), "/nearest", json!({ "row": 0, "k": 5 })).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let ns = v["neighbors"].as_array().unwrap();
    assert!(!ns.is_empty() && ns.len() <= 5);
    for n in ns {
        assert_ne!(n["row"], 0, "a point is not its own neighbor");
    }
    // Similarities are sorted descending.
    let sims: Vec<f64> = ns
        .iter()
        .map(|n| n["similarity"].as_f64().unwrap())
        .collect();
    assert!(sims.windows(2).all(|w| w[0] >= w[1]));
}

#[tokio::test]
async fn nearest_by_vector_returns_neighbors() {
    // A correctly-sized (128-d) query vector is accepted by the ANN index.
    let vector: Vec<f64> = (0..128).map(|i| (i as f64) * 0.01).collect();
    let resp = post_json(router(), "/nearest", json!({ "vector": vector, "k": 4 })).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(!v["neighbors"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn nearest_with_wrong_length_vector_does_not_panic() {
    // Regression: a vector whose length != the index dim (128) would panic
    // `AnnIndex::query`'s assert. The handler must reject it gracefully with an
    // empty result, not a 500 / dropped connection.
    let resp = post_json(
        router(),
        "/nearest",
        json!({ "vector": [1.0, 2.0, 3.0], "k": 5 }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["neighbors"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn globs_detects_clusters() {
    let resp = get(router(), "/globs?max_k=8").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let globs = v["globs"].as_array().unwrap();
    assert!(
        !globs.is_empty(),
        "structured stress corpus should yield globs"
    );
    for g in globs {
        assert_eq!(
            g["centroid"].as_array().unwrap().len(),
            3,
            "glob centroid is 3-D"
        );
        assert!(g["member_count"].as_u64().unwrap() >= 1);
        assert!(g["top_categories"].is_array());
    }
}

#[tokio::test]
async fn path_between_two_categories() {
    let (names, _dim, app) = router_with_meta();
    assert!(names.len() >= 2, "stress corpus has multiple categories");
    let resp = post_json(
        app,
        "/path",
        json!({ "source": names[0], "target": names[1] }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v.get("path").is_some(), "response has a `path` field");
    if let Some(steps) = v["path"]["steps"].as_array() {
        assert!(!steps.is_empty(), "a found path has steps");
        assert_eq!(
            steps[0]["category_name"],
            serde_json::json!(names[0]),
            "path starts at source"
        );
    }
}

#[tokio::test]
async fn path_unknown_category_is_a_400() {
    let (names, _dim, app) = router_with_meta();
    let resp = post_json(
        app,
        "/path",
        json!({ "source": "__no_such_category__", "target": names[0] }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn drill_down_within_a_category() {
    let (names, dim, app) = router_with_meta();
    let vector = vec![0.0_f64; dim]; // valid length; projects to a point we drill from
    let resp = post_json(
        app,
        "/drill_down",
        json!({ "category": names[0], "k": 5, "vector": vector }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let results = v["results"].as_array().unwrap();
    assert!(
        !results.is_empty() && results.len() <= 5,
        "≤k results, non-empty (category has members)"
    );
    for r in results {
        assert_eq!(
            r["category"],
            serde_json::json!(names[0]),
            "drill-down stays within the category"
        );
        assert!(
            r["row"].is_u64() && r["distance"].is_number() && r["used_inner_sphere"].is_boolean()
        );
    }
}

#[tokio::test]
async fn drill_down_wrong_dim_vector_is_a_400() {
    let (names, _dim, app) = router_with_meta();
    let resp = post_json(
        app,
        "/drill_down",
        json!({ "category": names[0], "k": 5, "vector": [1.0, 2.0, 3.0] }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn diagnostics_reports_projection_health() {
    let resp = get(router(), "/diagnostics").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["projection_kind"], "pca");
    assert_eq!(v["total_points"], 300);
    assert!(v["warnings"].is_array());
    let bins = v["certainty"]["bins"].as_array().unwrap();
    assert_eq!(bins.len(), 16, "certainty histogram has 16 bins");
    let total: u64 = bins.iter().map(|b| b.as_u64().unwrap()).sum();
    assert_eq!(total, 300, "histogram covers every point");
    let outliers = v["outliers"].as_array().unwrap();
    assert!(!outliers.is_empty() && outliers.len() <= 16);
    let cs: Vec<f64> = outliers
        .iter()
        .map(|o| o["certainty"].as_f64().unwrap())
        .collect();
    assert!(
        cs.windows(2).all(|w| w[0] <= w[1]),
        "outliers ascending by certainty (least faithful first)"
    );
}

#[tokio::test]
async fn tiles_filter_by_category_and_certainty() {
    // cats=0 → only category 0 streamed.
    let pts = decode_tile(&body_bytes(get(router(), "/tiles?cats=0").await).await).unwrap();
    assert!(
        !pts.is_empty() && pts.iter().all(|p| p.cat == 0),
        "cats=0 keeps only category 0"
    );
    // min_certainty above the [0,1] range → empty tile.
    let none =
        decode_tile(&body_bytes(get(router(), "/tiles?min_certainty=2.0").await).await).unwrap();
    assert!(none.is_empty(), "min_certainty>1 filters everything out");
}

#[tokio::test]
async fn category_stats_lists_the_palette() {
    let resp = get(router(), "/category_stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let cats = v.as_array().unwrap();
    assert!(!cats.is_empty());
    let total: u64 = cats.iter().map(|c| c["count"].as_u64().unwrap()).sum();
    assert_eq!(total, 300);
    assert!(cats[0]["color"].as_str().unwrap().starts_with('#'));
}
