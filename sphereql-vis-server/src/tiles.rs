//! Viewport tiling: turn a camera cone + LOD budget into a binary
//! [`SQT1`](sphereql_vis::tile) tile of the points in view.
//!
//! The browser never holds all N points. Each frame it asks for the points
//! inside its view cone at a level of detail it can afford; the server narrows
//! to that cone via the [`SpatialIndex`](sphereql_index::SpatialIndex), then
//! stratifies the result down to the budget (proportional per-category
//! allocation + even stride, matching `sphereql-vis`'s `stratified_sample`) so
//! every category stays visible and the decimation is deterministic.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use serde::Deserialize;
use sphereql_core::{Cone, SphericalPoint};
use sphereql_vis::{TilePoint, encode_tile};

use crate::state::AppState;

/// Hard ceiling on points returned in a single tile, regardless of the
/// requested budget — bounds the response and the client decode cost.
const MAX_TILE_POINTS: usize = 200_000;

/// Query parameters for `GET /tiles`.
///
/// All optional. With no parameters you get a global LOD-0 stratified sample of
/// the whole cloud (`half_angle = π`, `budget = lod.base_budget`). A narrower
/// `half_angle` restricts to a viewport cone aimed at `(theta, phi)`; `budget`
/// (or `lod`, which maps to `base_budget << lod`) sets the detail.
#[derive(Debug, Default, Deserialize)]
pub struct TileParams {
    /// Cone axis azimuth in radians.
    pub theta: Option<f64>,
    /// Cone axis polar angle in radians.
    pub phi: Option<f64>,
    /// Cone half-angle in radians; `>= π` (the default) means "whole sphere".
    pub half_angle: Option<f64>,
    /// Explicit point budget for this tile.
    pub budget: Option<usize>,
    /// LOD level; maps to `base_budget << lod` when `budget` is absent.
    pub lod: Option<u8>,
}

impl TileParams {
    /// The effective point budget: explicit `budget`, else `base_budget << lod`,
    /// else `base_budget`. Clamped to at least 1 and at most `MAX_TILE_POINTS`.
    fn budget(&self, base: usize) -> usize {
        let raw = match (self.budget, self.lod) {
            (Some(b), _) => b,
            (None, Some(l)) => base.saturating_mul(1usize << l.min(20)),
            (None, None) => base,
        };
        raw.clamp(1, MAX_TILE_POINTS)
    }
}

/// Rows visible in the requested cone (all rows when the cone spans the sphere
/// or is degenerate).
fn candidate_rows(state: &AppState, params: &TileParams) -> Vec<u32> {
    let half_angle = params.half_angle.unwrap_or(PI);
    let all = || (0..state.points.len() as u32).collect::<Vec<u32>>();
    if !(half_angle.is_finite()) || half_angle >= PI {
        return all();
    }
    let theta = params.theta.unwrap_or(0.0);
    let phi = params.phi.unwrap_or(std::f64::consts::FRAC_PI_2);
    let axis = SphericalPoint::new_unchecked(1.0, theta, phi);
    match Cone::new(SphericalPoint::origin(), axis, half_angle.clamp(1e-3, PI)) {
        Ok(cone) => state
            .spatial
            .query_cone(&cone)
            .items
            .iter()
            .map(|it| it.row)
            .collect(),
        Err(_) => all(),
    }
}

/// Deterministically thin `rows` down to roughly `budget` points, allocating
/// the budget across categories in proportion to their share and taking an
/// even stride within each (so spatial spread is preserved and the result is
/// reproducible). Returns `rows` unchanged when it already fits.
fn stratified(state: &AppState, rows: &[u32], budget: usize) -> Vec<u32> {
    if rows.len() <= budget {
        return rows.to_vec();
    }
    let mut by_cat: BTreeMap<u16, Vec<u32>> = BTreeMap::new();
    for &r in rows {
        by_cat
            .entry(state.points[r as usize].cat)
            .or_default()
            .push(r);
    }
    let total = rows.len() as f64;
    let mut out: Vec<u32> = Vec::with_capacity(budget);
    for group in by_cat.values() {
        // Proportional share, at least one per non-empty category.
        let share = ((group.len() as f64 / total) * budget as f64).round() as usize;
        let take = share.clamp(1, group.len());
        let stride = (group.len() / take).max(1);
        let mut i = 0;
        let mut n = 0;
        while i < group.len() && n < take {
            out.push(group[i]);
            i += stride;
            n += 1;
        }
    }
    // The per-category min-1 can nudge the total just over budget; cap it.
    if out.len() > budget {
        out.truncate(budget);
    }
    out
}

/// Build the binary tile for a `GET /tiles` request: cone-query, LOD-decimate,
/// and encode the survivors as [`TilePoint`]s.
pub fn build_tile(state: &AppState, params: &TileParams) -> Vec<u8> {
    let budget = params.budget(state.manifest.lod.base_budget);
    let rows = candidate_rows(state, params);
    let chosen = stratified(state, &rows, budget);
    let pts: Vec<TilePoint> = chosen
        .iter()
        .map(|&r| {
            let p = &state.points[r as usize];
            TilePoint {
                x: p.xyz[0],
                y: p.xyz[1],
                z: p.xyz[2],
                cat: p.cat,
                row: r,
            }
        })
        .collect();
    encode_tile(&pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sphereql_corpus::CorpusId;
    use sphereql_embed::ProjectionKind;
    use sphereql_vis::decode_tile;

    fn demo() -> AppState {
        AppState::from_corpus(CorpusId::Stress, ProjectionKind::Pca).expect("builds")
    }

    #[test]
    fn budget_resolution() {
        let base = 1000;
        assert_eq!(TileParams::default().budget(base), base);
        assert_eq!(
            TileParams {
                budget: Some(50),
                ..Default::default()
            }
            .budget(base),
            50
        );
        assert_eq!(
            TileParams {
                lod: Some(3),
                ..Default::default()
            }
            .budget(base),
            base * 8
        );
        // Clamped to the hard ceiling and floor.
        assert_eq!(
            TileParams {
                budget: Some(0),
                ..Default::default()
            }
            .budget(base),
            1
        );
        assert_eq!(
            TileParams {
                budget: Some(usize::MAX),
                ..Default::default()
            }
            .budget(base),
            MAX_TILE_POINTS
        );
    }

    #[test]
    fn whole_sphere_tile_returns_all_points_under_budget() {
        let state = demo();
        let bytes = build_tile(&state, &TileParams::default());
        let pts = decode_tile(&bytes).expect("valid tile");
        // 300 points, base budget 20k → no decimation.
        assert_eq!(pts.len(), 300);
        // Every row is in range and every cat indexes the palette.
        for p in &pts {
            assert!((p.row as usize) < state.points.len());
            assert!((p.cat as usize) < state.manifest.palette.len());
        }
    }

    #[test]
    fn budget_decimates() {
        let state = demo();
        let bytes = build_tile(
            &state,
            &TileParams {
                budget: Some(50),
                ..Default::default()
            },
        );
        let pts = decode_tile(&bytes).expect("valid tile");
        assert!(pts.len() <= 50, "got {} points for budget 50", pts.len());
        assert!(!pts.is_empty());
    }

    #[test]
    fn decimation_is_deterministic() {
        let state = demo();
        let params = TileParams {
            budget: Some(80),
            ..Default::default()
        };
        let a = build_tile(&state, &params);
        let b = build_tile(&state, &params);
        assert_eq!(a, b, "same request must yield byte-identical tiles");
    }

    #[test]
    fn narrow_cone_returns_no_more_than_whole_sphere() {
        let state = demo();
        let whole = decode_tile(&build_tile(&state, &TileParams::default()))
            .unwrap()
            .len();
        let cone = decode_tile(&build_tile(
            &state,
            &TileParams {
                theta: Some(0.0),
                phi: Some(std::f64::consts::FRAC_PI_2),
                half_angle: Some(0.3),
                ..Default::default()
            },
        ))
        .unwrap()
        .len();
        assert!(
            cone <= whole,
            "narrow cone ({cone}) returned more than the whole sphere ({whole})"
        );
    }
}
