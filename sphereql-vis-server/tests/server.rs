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
