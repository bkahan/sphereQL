//! HTTP surface: the axum router and its handlers.
//!
//! Handlers are thin — they parse/validate the request, call into
//! [`crate::state`] / [`crate::tiles`], and serialize the result. The bulk of
//! the work (cone tiling, decimation, neighbor lookup) lives in those modules
//! so it can be unit-tested without a socket. The router is built by
//! [`build_router`] and driven in tests via `tower::ServiceExt::oneshot`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;

use crate::state::AppState;
use crate::tiles::{TileParams, build_tile};

/// Cap on rows in a single `POST /points` batch — bounds the response.
const MAX_POINTS_BATCH: usize = 4096;
/// Cap on `k` for `POST /nearest`.
const MAX_NEAREST_K: usize = 256;
/// Request body limit (applies to the POST endpoints).
const BODY_LIMIT: usize = 4 * 1024 * 1024;

/// Build the router over a shared [`AppState`]. Adds permissive CORS (the
/// viewer is typically served from `file://` or a different origin), a body
/// limit on the POST endpoints, and a panic catcher so a handler fault returns
/// 500 instead of dropping the connection.
pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    Router::new()
        .route("/health", get(health))
        .route("/manifest", get(manifest))
        .route("/tiles", get(tiles))
        .route("/points", post(points))
        .route("/nearest", post(nearest))
        .route("/category_stats", get(category_stats))
        .layer(RequestBodyLimitLayer::new(BODY_LIMIT))
        .layer(cors)
        .layer(CatchPanicLayer::new())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn manifest(State(state): State<Arc<AppState>>) -> Response {
    Json(&state.manifest).into_response()
}

async fn category_stats(State(state): State<Arc<AppState>>) -> Response {
    Json(&state.manifest.palette).into_response()
}

async fn tiles(State(state): State<Arc<AppState>>, Query(params): Query<TileParams>) -> Response {
    let bytes = build_tile(&state, &params);
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/octet-stream"),
        )],
        bytes,
    )
        .into_response()
}

// ── /points: lazy per-point metadata ───────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct PointsRequest {
    /// Rows to fetch (out-of-range rows are silently skipped; capped at
    /// `MAX_POINTS_BATCH`).
    pub rows: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct PointMeta {
    pub row: u32,
    pub label: String,
    pub cat: u16,
    pub category: String,
    pub certainty: f32,
    pub intensity: f32,
    pub vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct PointsResponse {
    pub points: Vec<PointMeta>,
}

/// Gather metadata for the requested rows (the inspector's payload). Pure, so
/// it is unit-tested directly.
pub fn collect_points(state: &AppState, req: &PointsRequest) -> PointsResponse {
    let points = req
        .rows
        .iter()
        .take(MAX_POINTS_BATCH)
        .filter_map(|&row| {
            let p = state.points.get(row as usize)?;
            Some(PointMeta {
                row,
                label: p.label.clone(),
                cat: p.cat,
                category: state
                    .cat_names
                    .get(p.cat as usize)
                    .cloned()
                    .unwrap_or_default(),
                certainty: p.certainty,
                intensity: p.intensity,
                vector: p.vector.clone(),
            })
        })
        .collect();
    PointsResponse { points }
}

async fn points(State(state): State<Arc<AppState>>, Json(req): Json<PointsRequest>) -> Response {
    Json(collect_points(&state, &req)).into_response()
}

// ── /nearest: semantic neighbors (trace) ────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct NearestRequest {
    /// Query by an existing point's row (its own row is excluded from results).
    pub row: Option<u32>,
    /// Or query by a raw embedding vector.
    pub vector: Option<Vec<f64>>,
    /// Number of neighbors to return (clamped to `MAX_NEAREST_K`).
    pub k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct Neighbor {
    pub row: u32,
    pub similarity: f64,
}

#[derive(Debug, Serialize)]
pub struct NearestResponse {
    pub neighbors: Vec<Neighbor>,
}

/// Resolve a nearest-neighbor query against the ANN index. Returns an empty
/// list when the request names neither a valid row nor a usable vector.
///
/// A query vector whose length doesn't match the index dimensionality
/// ([`AppState::dim`]) is rejected here rather than passed through:
/// `AnnIndex::query` asserts on the length, so a mismatched vector would
/// otherwise panic the handler (a remotely-triggerable fault).
pub fn collect_nearest(state: &AppState, req: &NearestRequest) -> NearestResponse {
    let k = req.k.unwrap_or(10).clamp(1, MAX_NEAREST_K);
    let hits = match (req.row, &req.vector) {
        (Some(row), _) if (row as usize) < state.points.len() => {
            state.ann.query_by_index(row as usize, k)
        }
        (_, Some(v)) if v.len() == state.dim => state.ann.query(v, k),
        _ => Vec::new(),
    };
    let neighbors = hits
        .into_iter()
        .map(|(idx, sim)| Neighbor {
            row: idx as u32,
            similarity: sim,
        })
        .collect();
    NearestResponse { neighbors }
}

async fn nearest(State(state): State<Arc<AppState>>, Json(req): Json<NearestRequest>) -> Response {
    Json(collect_nearest(&state, &req)).into_response()
}
