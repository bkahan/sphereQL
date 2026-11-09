//! Spatial overlays drawn on top of the point cloud.
//!
//! Each [`Overlay`] variant carries only plain Cartesian geometry (`[f64; 3]`)
//! and display metadata — never engine types — so this crate stays a pure,
//! low-level leaf depending on nothing but `sphereql-core`. Producers build
//! overlays from pipeline output (centroids, bridges, paths, …) using the
//! constructors here, which place on-sphere geometry at the scene's
//! `surface_radius` via [`crate::scene::on_surface`].
//!
//! The enum is `#[non_exhaustive]`: new overlay kinds can be added without a
//! breaking change.

use serde::{Deserialize, Serialize};
use sphereql_core::{SphericalPoint, slerp};

use crate::scene::on_surface;

/// A spherical cap rendered as a wire ring (used by coverage maps).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapRing {
    /// Cap center on the display shell.
    pub center: [f64; 3],
    /// Half-angle of the cap (radians).
    pub half_angle: f64,
    pub label: String,
}

/// A drawable spatial structure layered over the points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Overlay {
    /// A category centroid marker with a billboarded label.
    Centroid {
        pos: [f64; 3],
        color: String,
        label: String,
        members: usize,
    },
    /// A bridge: a line from a member item to the centroid of the domain it
    /// bridges toward, colored by classification.
    Bridge {
        from: [f64; 3],
        to: [f64; 3],
        strength: f64,
        classification: String,
        color: String,
    },
    /// A geodesic concept path: a polyline that bends along the sphere.
    GeodesicPath {
        vertices: Vec<[f64; 3]>,
        label: String,
        color: String,
    },
    /// A Voronoi territory cap (approximate, Monte-Carlo estimated).
    VoronoiCap {
        center: [f64; 3],
        half_angle: f64,
        cell_area: f64,
        color: String,
        label: String,
    },
    /// A category's antipode (semantic opposite), with the axis through center.
    Antipode {
        centroid: [f64; 3],
        antipode: [f64; 3],
        coherence: f64,
        color: String,
        label: String,
    },
    /// Knowledge-coverage map: claimed caps + scattered void samples.
    CoverageVoid {
        caps: Vec<CapRing>,
        voids: Vec<[f64; 3]>,
        coverage_fraction: f64,
    },
    /// A domain group: a hub centroid with spokes to member-category centroids.
    DomainGroup {
        centroid: [f64; 3],
        members: Vec<[f64; 3]>,
        color: String,
        label: String,
    },
    /// A detected knowledge cluster (glob) as a translucent sphere.
    Glob {
        center: [f64; 3],
        radius: f64,
        color: String,
        label: String,
    },
    /// A local manifold slice: an oriented plane + ring around a query.
    ManifoldSlice {
        center: [f64; 3],
        normal: [f64; 3],
        label: String,
    },
}

impl Overlay {
    /// A category centroid marker placed on the display shell.
    pub fn centroid(
        centroid: &SphericalPoint,
        surface_radius: f64,
        color: impl Into<String>,
        label: impl Into<String>,
        members: usize,
    ) -> Self {
        Overlay::Centroid {
            pos: on_surface(centroid, surface_radius),
            color: color.into(),
            label: label.into(),
            members,
        }
    }

    /// A bridge segment from an item's (absolute) position toward the centroid
    /// of the domain it connects to.
    pub fn bridge(
        from_item: [f64; 3],
        toward_centroid: &SphericalPoint,
        surface_radius: f64,
        strength: f64,
        classification: impl Into<String>,
        color: impl Into<String>,
    ) -> Self {
        Overlay::Bridge {
            from: from_item,
            to: on_surface(toward_centroid, surface_radius),
            strength,
            classification: classification.into(),
            color: color.into(),
        }
    }

    /// A geodesic path through `centroids`, slerp-interpolated on the shell.
    ///
    /// `samples_per_segment` controls arc smoothness (>= 1).
    pub fn geodesic_path(
        centroids: &[SphericalPoint],
        surface_radius: f64,
        samples_per_segment: usize,
        color: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        let steps = samples_per_segment.max(1);
        let mut vertices = Vec::new();
        // Each segment contributes its start plus interior samples (t in
        // [0, 1)); the shared junction is the next segment's start, so it is
        // emitted exactly once. The final endpoint (t = 1 of the last
        // segment) is appended after the loop. For a single centroid the
        // loop yields nothing and the trailing push emits it once.
        for seg in centroids.windows(2) {
            let (a, b) = (&seg[0], &seg[1]);
            for s in 0..steps {
                let t = s as f64 / steps as f64;
                vertices.push(on_surface(&slerp(a, b, t), surface_radius));
            }
        }
        if let Some(last) = centroids.last() {
            vertices.push(on_surface(last, surface_radius));
        }
        Overlay::GeodesicPath {
            vertices,
            color: color.into(),
            label: label.into(),
        }
    }

    /// A Voronoi territory cap. `cell_area` is the solid angle (steradians);
    /// the rendered half-angle is derived as `acos(1 - area / 2π)`.
    pub fn voronoi_cap(
        center: &SphericalPoint,
        surface_radius: f64,
        cell_area: f64,
        color: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Overlay::VoronoiCap {
            center: on_surface(center, surface_radius),
            half_angle: half_angle_from_solid_angle(cell_area),
            cell_area,
            color: color.into(),
            label: label.into(),
        }
    }

    /// A category antipode marker + center axis.
    pub fn antipode(
        centroid: &SphericalPoint,
        antipode: &SphericalPoint,
        surface_radius: f64,
        coherence: f64,
        color: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Overlay::Antipode {
            centroid: on_surface(centroid, surface_radius),
            antipode: on_surface(antipode, surface_radius),
            coherence,
            color: color.into(),
            label: label.into(),
        }
    }

    /// A domain-group hub with spokes to its member-category centroids.
    pub fn domain_group(
        group_centroid: &SphericalPoint,
        member_centroids: &[SphericalPoint],
        surface_radius: f64,
        color: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Overlay::DomainGroup {
            centroid: on_surface(group_centroid, surface_radius),
            members: member_centroids
                .iter()
                .map(|c| on_surface(c, surface_radius))
                .collect(),
            color: color.into(),
            label: label.into(),
        }
    }

    /// A glob sphere at an absolute center with an absolute radius.
    pub fn glob(
        center: [f64; 3],
        radius: f64,
        color: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Overlay::Glob {
            center,
            radius,
            color: color.into(),
            label: label.into(),
        }
    }

    /// A local manifold slice plane at an absolute center with a plane normal.
    pub fn manifold_slice(center: [f64; 3], normal: [f64; 3], label: impl Into<String>) -> Self {
        Overlay::ManifoldSlice {
            center,
            normal,
            label: label.into(),
        }
    }
}

/// Build a [`CapRing`] for a coverage cap from a spherical center + half-angle.
pub fn cap_ring(
    center: &SphericalPoint,
    surface_radius: f64,
    half_angle: f64,
    label: impl Into<String>,
) -> CapRing {
    CapRing {
        center: on_surface(center, surface_radius),
        half_angle,
        label: label.into(),
    }
}

/// Half-angle (radians) of a spherical cap subtending `solid_angle` steradians.
///
/// Inverts `Ω = 2π(1 − cos α)`; clamps the argument so out-of-range areas
/// (Monte-Carlo noise) don't produce `NaN`.
pub fn half_angle_from_solid_angle(solid_angle: f64) -> f64 {
    use std::f64::consts::TAU;
    (1.0 - solid_angle / TAU).clamp(-1.0, 1.0).acos()
}
