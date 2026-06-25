//! HTTP surface: the axum router and its handlers.
//!
//! Handlers are thin — they parse/validate the request, call into
//! [`crate::state`] / [`crate::tiles`], and serialize the result. The bulk of
//! the work (cone tiling, decimation, neighbor lookup) lives in those modules
//! so it can be unit-tested without a socket. The router is built by
//! [`build_router`] and driven in tests via `tower::ServiceExt::oneshot`.

use std::sync::{Arc, RwLock};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use sphereql_embed::{PipelineQuery, SphereQLOutput, SphereQLQuery, WarningSeverity};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::services::ServeDir;

use crate::state::AppState;
use crate::tiles::{TileParams, build_tile};

/// Shared, hot-swappable server state. Reads (tiles/queries) snapshot the inner
/// `Arc` under a brief read lock; `/reproject` swaps in a freshly-built state
/// under a write lock. Lock-free in practice — the read lock is held only long
/// enough to clone the `Arc`.
pub type Shared = Arc<RwLock<Arc<AppState>>>;

/// Optional WASM studio front-end to serve alongside the API.
///
/// When present, the server serves the pre-built studio files from `dir` as
/// static assets (via `ServeDir`), and returns the auto-connect-injected
/// `index_html` for `GET /`. The studio's `studio.js`, `compare.html`,
/// `embed.html`, `demo-corpus.json`, and `pkg/*` are all served from `dir`.
pub struct StudioAssets {
    /// Modified `index.html` content (auto-connect script already injected).
    pub index_html: String,
    /// The `studio/dist` directory to serve as a static fallback.
    pub dir: std::path::PathBuf,
}

/// Snapshot the current state (clone the inner `Arc`, release the lock).
fn snapshot(shared: &Shared) -> Arc<AppState> {
    shared.read().expect("state lock not poisoned").clone()
}

/// Cap on rows in a single `POST /points` batch — bounds the response.
const MAX_POINTS_BATCH: usize = 4096;
/// Cap on `k` for `POST /nearest`.
const MAX_NEAREST_K: usize = 256;
/// Request body limit (applies to the POST endpoints).
const BODY_LIMIT: usize = 4 * 1024 * 1024;

/// Build the router over a shared [`AppState`].
///
/// Adds permissive CORS (the viewer is typically served from `file://` or a
/// different origin), a body limit on the POST endpoints, and a panic catcher
/// so a handler fault returns 500 instead of dropping the connection.
///
/// When `studio` is `Some`, the router also serves the pre-built WASM studio
/// as a static site: `GET /` returns the auto-connect-injected `index.html`
/// and all other unmatched paths fall through to `ServeDir` over `studio.dir`.
pub fn build_router(state: Shared, studio: Option<StudioAssets>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    let api = Router::new()
        .route("/health", get(health))
        .route("/manifest", get(manifest))
        .route("/tiles", get(tiles))
        .route("/points", post(points))
        .route("/nearest", post(nearest))
        .route("/category_stats", get(category_stats))
        .route("/path", post(path))
        .route("/globs", get(globs))
        .route("/drill_down", post(drill_down))
        .route("/diagnostics", get(diagnostics))
        .route("/reproject", post(reproject))
        .layer(RequestBodyLimitLayer::new(BODY_LIMIT))
        .layer(cors)
        .layer(CatchPanicLayer::new())
        .with_state(state);

    if let Some(assets) = studio {
        let idx_html = Arc::new(assets.index_html);
        Router::new()
            .route(
                "/",
                get(move || {
                    let html = Arc::clone(&idx_html);
                    async move { Html((*html).clone()) }
                }),
            )
            .merge(api)
            .fallback_service(ServeDir::new(assets.dir))
    } else {
        // No studio to serve — still answer `/` with a minimal landing page so
        // `--open` never lands on a 404 (S6). The page is a static `const`, so
        // it carries no untrusted data and needs no escaping / auto-connect
        // injection. `/` is registered ONLY here: the `Some` branch above owns
        // `/` for the studio index, and axum panics at startup on a duplicate
        // route, so `api` itself must never define `/`.
        api.route("/", get(landing))
    }
}

async fn health() -> &'static str {
    "ok"
}

/// Minimal static landing page served at `GET /` when no WASM studio is built.
/// Lists the JSON endpoints and points at the studio build script. Static — no
/// untrusted data, no escaping, no auto-connect injection.
async fn landing() -> Html<&'static str> {
    Html(LANDING)
}

const LANDING: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SphereQL vis-server</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; max-width: 46rem; margin: 3rem auto; padding: 0 1rem; color: #1a1a1a; }
  code { background: #f0f0f3; padding: .1em .35em; border-radius: 4px; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 1.6rem; }
  ul { padding-left: 1.2rem; } li { margin: .25rem 0; }
  .hint { background: #fffbe6; border: 1px solid #f0e2a0; border-radius: 6px; padding: .8rem 1rem; margin-top: 1.4rem; }
</style>
</head>
<body>
<h1>SphereQL vis-server</h1>
<p>The query API is running. No WASM studio front-end was found at
<code>sphereql-wasm/studio/dist</code>, so this page is served at <code>/</code> instead.</p>

<h2>JSON / binary endpoints</h2>
<ul>
  <li><code>GET /health</code> — liveness check</li>
  <li><code>GET /manifest</code> — bounded scene descriptor (stats, palette, bounds, LOD)</li>
  <li><code>GET /tiles</code> — binary SQT1 tile of points in a viewport cone</li>
  <li><code>POST /points</code> — per-point metadata by row</li>
  <li><code>POST /nearest</code> — ANN neighbors of a row or query vector</li>
  <li><code>GET /category_stats</code> — palette (name → color → count)</li>
  <li><code>POST /path</code> — shortest path between two categories</li>
  <li><code>GET /globs</code> — concept-cluster detection</li>
  <li><code>POST /drill_down</code> — k-NN within one category</li>
  <li><code>GET /diagnostics</code> — projection-health dashboard data</li>
  <li><code>POST /reproject</code> — live re-projection</li>
</ul>

<div class="hint">
  <strong>Want the interactive studio here at <code>/</code>?</strong>
  Build it once with <code>sphereql-wasm/studio/build.sh</code> (emits
  <code>studio/dist</code> with the wasm + worker), then restart the server.
  Or generate a standalone offline viewer with
  <code>--emit-html</code>.
</div>
</body>
</html>
"#;

async fn manifest(State(shared): State<Shared>) -> Response {
    Json(&snapshot(&shared).manifest).into_response()
}

async fn category_stats(State(shared): State<Shared>) -> Response {
    Json(&snapshot(&shared).manifest.palette).into_response()
}

async fn tiles(State(shared): State<Shared>, Query(params): Query<TileParams>) -> Response {
    let state = snapshot(&shared);
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
    /// Display position on the shell (the inspector derives r/θ/φ from this, so
    /// coords show for any inspected row — clicked, neighbor, or outlier).
    pub x: f32,
    pub y: f32,
    pub z: f32,
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
                x: p.xyz[0],
                y: p.xyz[1],
                z: p.xyz[2],
                vector: p.vector.clone(),
            })
        })
        .collect();
    PointsResponse { points }
}

async fn points(State(shared): State<Shared>, Json(req): Json<PointsRequest>) -> Response {
    Json(collect_points(&snapshot(&shared), &req)).into_response()
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

async fn nearest(State(shared): State<Shared>, Json(req): Json<NearestRequest>) -> Response {
    Json(collect_nearest(&snapshot(&shared), &req)).into_response()
}

// ── Trace endpoints (category graph / globs / drill-down) ───────────────────
//
// These run pipeline queries against the retained projection + category layer.
// The category/glob queries ignore the query embedding, but `pipeline.query`
// builds one unconditionally, so we hand it a zero vector of the right length.

fn zero_query(state: &AppState) -> PipelineQuery {
    PipelineQuery {
        embedding: vec![0.0; state.dim.max(1)],
    }
}

/// Map a handler-level failure to a 400 with the message (most failures here
/// are unknown category / dim mismatch — caller errors, not server faults).
fn bad_request(msg: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_REQUEST, msg.to_string()).into_response()
}

// /path — shortest path between two categories through the category graph.

#[derive(Debug, Deserialize)]
pub struct PathRequest {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Serialize)]
pub struct PathStepDto {
    pub category_index: usize,
    pub category_name: String,
    pub cumulative_distance: f64,
    pub hop_confidence: f64,
}

#[derive(Debug, Serialize)]
pub struct PathDto {
    pub steps: Vec<PathStepDto>,
    pub total_distance: f64,
    pub path_confidence: f64,
}

#[derive(Debug, Serialize)]
pub struct PathResponse {
    /// `None` when the two categories are not connected in the graph.
    pub path: Option<PathDto>,
}

async fn path(State(shared): State<Shared>, Json(req): Json<PathRequest>) -> Response {
    let state = snapshot(&shared);
    let q = SphereQLQuery::CategoryConceptPath {
        source_category: &req.source,
        target_category: &req.target,
    };
    match state.pipeline.query(q, &zero_query(&state)) {
        Ok(SphereQLOutput::CategoryConceptPath(opt)) => Json(PathResponse {
            path: opt.map(|p| PathDto {
                total_distance: p.total_distance,
                path_confidence: p.path_confidence,
                steps: p
                    .steps
                    .into_iter()
                    .map(|s| PathStepDto {
                        category_index: s.category_index,
                        category_name: s.category_name,
                        cumulative_distance: s.cumulative_distance,
                        hop_confidence: s.hop_confidence,
                    })
                    .collect(),
            }),
        })
        .into_response(),
        Ok(_) => bad_request("unexpected query output"),
        Err(e) => bad_request(e),
    }
}

// /globs — concept-cluster detection over the whole projected cloud.

#[derive(Debug, Deserialize)]
pub struct GlobParams {
    /// Fixed cluster count; omit for silhouette-based auto-selection.
    pub k: Option<usize>,
    /// Upper bound on the auto-selected k.
    pub max_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct GlobDto {
    pub id: usize,
    pub centroid: [f64; 3],
    pub member_count: usize,
    pub radius: f64,
    pub top_categories: Vec<(String, usize)>,
}

#[derive(Debug, Serialize)]
pub struct GlobsResponse {
    pub globs: Vec<GlobDto>,
}

async fn globs(State(shared): State<Shared>, Query(q): Query<GlobParams>) -> Response {
    let state = snapshot(&shared);
    let max_k = q.max_k.unwrap_or(12).clamp(2, 64);
    let k = q.k.filter(|&k| k >= 2);
    match state
        .pipeline
        .query(SphereQLQuery::DetectGlobs { k, max_k }, &zero_query(&state))
    {
        Ok(SphereQLOutput::Globs(globs)) => Json(GlobsResponse {
            globs: globs
                .into_iter()
                .map(|g| GlobDto {
                    id: g.id,
                    centroid: g.centroid,
                    member_count: g.member_count,
                    radius: g.radius,
                    top_categories: g.top_categories,
                })
                .collect(),
        })
        .into_response(),
        Ok(_) => bad_request("unexpected query output"),
        Err(e) => bad_request(e),
    }
}

// /drill_down — k-NN within one category, relative to a query vector, using the
// category's inner-sphere projection when available.

#[derive(Debug, Deserialize)]
pub struct DrillRequest {
    pub category: String,
    pub k: Option<usize>,
    /// Query embedding (length must equal the corpus embedding dim).
    pub vector: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct DrillResultDto {
    /// Global row index (also the tile `row` and `/points` key).
    pub row: u32,
    pub label: String,
    pub category: String,
    pub distance: f64,
    /// Whether the inner-sphere projection was used (vs the outer fallback).
    pub used_inner_sphere: bool,
}

#[derive(Debug, Serialize)]
pub struct DrillResponse {
    pub results: Vec<DrillResultDto>,
}

async fn drill_down(State(shared): State<Shared>, Json(req): Json<DrillRequest>) -> Response {
    let state = snapshot(&shared);
    let k = req.k.unwrap_or(10).clamp(1, MAX_NEAREST_K);
    if req.vector.len() != state.dim {
        return bad_request(format!(
            "vector length {} != embedding dim {}",
            req.vector.len(),
            state.dim
        ));
    }
    let qe = PipelineQuery {
        embedding: req.vector.clone(),
    };
    match state.pipeline.query(
        SphereQLQuery::DrillDown {
            category: &req.category,
            k,
        },
        &qe,
    ) {
        Ok(SphereQLOutput::DrillDown(rows)) => Json(DrillResponse {
            results: rows
                .into_iter()
                .map(|r| {
                    let p = state.points.get(r.item_index);
                    DrillResultDto {
                        row: r.item_index as u32,
                        label: p.map(|p| p.label.clone()).unwrap_or_default(),
                        category: p
                            .and_then(|p| state.cat_names.get(p.cat as usize).cloned())
                            .unwrap_or_default(),
                        distance: r.distance,
                        used_inner_sphere: r.used_inner_sphere,
                    }
                })
                .collect(),
        })
        .into_response(),
        Ok(_) => bad_request("unexpected query output"),
        Err(e) => bad_request(e),
    }
}

// ── /diagnostics: projection-health dashboard data ──────────────────────────

#[derive(Debug, Serialize)]
pub struct HistogramDto {
    /// Counts per equal-width bin over `[min, max]`.
    pub bins: Vec<usize>,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Serialize)]
pub struct WarningDto {
    pub message: String,
    pub severity: &'static str,
    pub evr: f64,
}

#[derive(Debug, Serialize)]
pub struct OutlierDto {
    pub row: u32,
    pub label: String,
    pub category: String,
    pub certainty: f32,
}

#[derive(Debug, Serialize)]
pub struct DiagnosticsResponse {
    pub projection_kind: String,
    pub evr: f64,
    pub evr_label: String,
    pub total_points: usize,
    /// Projection-fidelity warnings (empty when EVR is healthy).
    pub warnings: Vec<WarningDto>,
    /// Per-point certainty (projection fidelity) distribution, 16 bins.
    pub certainty: HistogramDto,
    /// Per-point intensity (pre-normalization magnitude) distribution, 16 bins.
    pub intensity: HistogramDto,
    /// The lowest-certainty points — where the projection is least trustworthy.
    pub outliers: Vec<OutlierDto>,
}

/// Equal-width histogram of `vals` into `bins` buckets over the observed range.
fn histogram(vals: &[f32], bins: usize) -> HistogramDto {
    let bins = bins.max(1);
    if vals.is_empty() {
        return HistogramDto {
            bins: vec![0; bins],
            min: 0.0,
            max: 0.0,
        };
    }
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in vals {
        let v = v as f64;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let span = (max - min).max(1e-12);
    let mut counts = vec![0usize; bins];
    for &v in vals {
        let t = (((v as f64 - min) / span) * bins as f64).floor() as usize;
        counts[t.min(bins - 1)] += 1;
    }
    HistogramDto {
        bins: counts,
        min,
        max,
    }
}

async fn diagnostics(State(shared): State<Shared>) -> Response {
    let state = snapshot(&shared);
    let certainty: Vec<f32> = state.points.iter().map(|p| p.certainty).collect();
    let intensity: Vec<f32> = state.points.iter().map(|p| p.intensity).collect();
    let warnings = state
        .pipeline
        .projection_warnings()
        .iter()
        .map(|w| WarningDto {
            message: w.message.clone(),
            severity: match w.severity {
                WarningSeverity::Info => "info",
                WarningSeverity::Warning => "warning",
                WarningSeverity::Critical => "critical",
            },
            evr: w.evr,
        })
        .collect();
    // Outliers = the lowest-certainty points (where the projection is least
    // faithful), the top 16 ascending.
    let mut order: Vec<u32> = (0..state.points.len() as u32).collect();
    order.sort_by(|&a, &b| {
        state.points[a as usize]
            .certainty
            .total_cmp(&state.points[b as usize].certainty)
    });
    let outliers = order
        .iter()
        .take(16)
        .map(|&r| {
            let p = &state.points[r as usize];
            OutlierDto {
                row: r,
                label: p.label.clone(),
                category: state
                    .cat_names
                    .get(p.cat as usize)
                    .cloned()
                    .unwrap_or_default(),
                certainty: p.certainty,
            }
        })
        .collect();
    Json(DiagnosticsResponse {
        projection_kind: state.manifest.stats.projection_kind.clone(),
        evr: state.manifest.stats.evr,
        evr_label: state.manifest.stats.evr_label.clone(),
        total_points: state.points.len(),
        warnings,
        certainty: histogram(&certainty, 16),
        intensity: histogram(&intensity, 16),
        outliers,
    })
    .into_response()
}

// ── /reproject: live re-projection ("tune") ────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ReprojectRequest {
    /// `pca` | `umap_sphere` | `laplacian` | `kernel_pca` (gated by corpus size).
    pub projection: String,
}

/// Re-project the in-memory corpus with a different kind and atomically swap it
/// in, returning the new manifest. The heavy re-projection runs off the async
/// runtime via `spawn_blocking`; the write lock is held only for the pointer
/// swap, so concurrent reads (tiles/queries) are never blocked on the rebuild.
async fn reproject(State(shared): State<Shared>, Json(req): Json<ReprojectRequest>) -> Response {
    let Some(kind) = crate::parse_projection(&req.projection) else {
        return bad_request(format!("unknown projection '{}'", req.projection));
    };
    let current = snapshot(&shared);
    match tokio::task::spawn_blocking(move || current.reproject(kind)).await {
        Ok(Ok(next)) => {
            let manifest = next.manifest.clone();
            *shared.write().expect("state lock not poisoned") = Arc::new(next);
            Json(manifest).into_response()
        }
        Ok(Err(e)) => bad_request(e),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "re-projection task failed".to_string(),
        )
            .into_response(),
    }
}
