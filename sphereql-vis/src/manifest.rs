//! The out-of-core scene descriptor: everything the viewer needs to set up a
//! streamed scene *except* the points themselves.
//!
//! In the server-backed (millions-of-points) mode the viewer first fetches a
//! `Manifest` — small and bounded regardless of N — then streams point
//! [`crate::tile::TilePoint`] tiles by viewport/LOD. The manifest carries the
//! cheap-at-any-N aggregates (stats, overlays, the category palette) plus the
//! framing/LOD info the streamer needs. It deliberately reuses [`SceneStats`]
//! and [`Overlay`] so the same panels/legend code drives both the inline
//! `Scene` and the streamed manifest.

use serde::{Deserialize, Serialize};

use crate::overlay::Overlay;
use crate::scene::SceneStats;

/// Current manifest schema version (bump on breaking field changes).
pub const MANIFEST_VERSION: u16 = 1;

/// Axis-aligned bounds of the projected cloud, for camera framing + tiling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        }
    }
}

impl Bounds {
    /// Tight bounds enclosing `xyz`. Returns the unit cube for an empty input
    /// (so framing math never sees a degenerate/inverted box).
    pub fn enclosing<'a, I: IntoIterator<Item = &'a [f64; 3]>>(xyz: I) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for p in xyz {
            any = true;
            for k in 0..3 {
                if p[k] < min[k] {
                    min[k] = p[k];
                }
                if p[k] > max[k] {
                    max[k] = p[k];
                }
            }
        }
        if !any {
            return Bounds::default();
        }
        Bounds { min, max }
    }
}

/// One category in the palette: name → display color (hex) + member count.
/// `cat` ids in tiles index into this vec (by position).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryInfo {
    pub name: String,
    pub color: String,
    pub count: usize,
}

/// Level-of-detail scheme: the streamer requests progressively finer tiles.
/// `levels` is the number of LOD tiers (0 = coarsest, a global stratified
/// sample); `base_budget` is roughly the point count of the coarsest level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LodScheme {
    pub levels: u8,
    pub base_budget: usize,
}

impl Default for LodScheme {
    fn default() -> Self {
        LodScheme {
            levels: 4,
            base_budget: 20_000,
        }
    }
}

/// The streamed-scene descriptor. Small and bounded regardless of N — the
/// points arrive separately as [`crate::tile::TilePoint`] tiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version ([`MANIFEST_VERSION`]).
    pub format_version: u16,
    pub title: String,
    /// Total points in the corpus (the viewer streams a working subset).
    pub total_points: usize,
    /// Display-shell radius (same meaning as `Scene::surface_radius`).
    pub surface_radius: f64,
    pub bounds: Bounds,
    pub stats: SceneStats,
    /// Cheap-at-any-N aggregates (centroids, domain groups, …).
    pub overlays: Vec<Overlay>,
    pub palette: Vec<CategoryInfo>,
    pub lod: LodScheme,
}

impl Manifest {
    /// A minimal manifest with the current version, default bounds/LOD, and no
    /// overlays — fill in the rest field-by-field.
    pub fn new(title: impl Into<String>, total_points: usize, stats: SceneStats) -> Self {
        Manifest {
            format_version: MANIFEST_VERSION,
            title: title.into(),
            total_points,
            surface_radius: 1.0,
            bounds: Bounds::default(),
            stats,
            overlays: Vec::new(),
            palette: Vec::new(),
            lod: LodScheme::default(),
        }
    }

    /// Serialize to JSON (the `GET /manifest` body).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a manifest always serializes")
    }

    /// Parse a manifest from its JSON form.
    pub fn from_json(s: &str) -> serde_json::Result<Manifest> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_encloses_points() {
        let pts = [[1.0, -2.0, 0.0], [-3.0, 5.0, 1.0]];
        let b = Bounds::enclosing(pts.iter());
        assert_eq!(b.min, [-3.0, -2.0, 0.0]);
        assert_eq!(b.max, [1.0, 5.0, 1.0]);
    }

    #[test]
    fn empty_bounds_is_unit_cube() {
        let none: [&[f64; 3]; 0] = [];
        assert_eq!(Bounds::enclosing(none), Bounds::default());
    }

    #[test]
    fn manifest_round_trips() {
        let mut m = Manifest::new("demo", 1_000_000, SceneStats::new("umap_sphere", 0.47));
        m.surface_radius = 1.5;
        m.bounds = Bounds {
            min: [-2.0; 3],
            max: [2.0; 3],
        };
        m.palette = vec![CategoryInfo {
            name: "science".into(),
            color: "#5cc8ff".into(),
            count: 500_000,
        }];
        let back = Manifest::from_json(&m.to_json()).expect("round-trips");
        assert_eq!(back, m);
        assert_eq!(back.format_version, MANIFEST_VERSION);
        assert_eq!(back.total_points, 1_000_000);
    }
}
