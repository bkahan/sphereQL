//! The serializable scene data model.
//!
//! A [`Scene`] is the data-agnostic payload that the emitter ([`crate::emit`])
//! turns into a self-contained HTML file. It carries the projected point
//! cloud, any number of [`Overlay`](crate::overlay::Overlay)s, and the
//! quality metadata shown in the stats panel.
//!
//! Coordinate convention: every point's `(x, y, z)` is in *display* Cartesian
//! space (typically the projection's own radius, which may be volumetric).
//! Overlays are expressed in the same absolute space — producers place
//! on-sphere overlays at [`Scene::surface_radius`] using [`on_surface`], so
//! markers, arcs and caps land on the same shell as the bulk of the cloud.

use serde::{Deserialize, Serialize};
use sphereql_core::SphericalPoint;

/// A single projected item rendered as a point in the cloud.
///
/// Carries both Cartesian (`x, y, z`) and spherical (`r, theta, phi`) forms so
/// the viewer can show coordinates without recomputing them. `certainty` and
/// `intensity` are optional projection-quality signals that drive point size
/// and opacity when present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    /// Category label — drives color and the legend.
    pub cat: String,
    /// Per-point display label (shown on hover / selection). May be empty.
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certainty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f64>,
}

impl ScenePoint {
    /// Build a point from spherical coordinates, deriving the Cartesian form
    /// via [`sphereql_core::spherical_to_cartesian`].
    pub fn from_spherical(
        cat: impl Into<String>,
        label: impl Into<String>,
        r: f64,
        theta: f64,
        phi: f64,
    ) -> Self {
        let c =
            sphereql_core::spherical_to_cartesian(&SphericalPoint::new_unchecked(r, theta, phi));
        Self {
            x: c.x,
            y: c.y,
            z: c.z,
            r,
            theta,
            phi,
            cat: cat.into(),
            label: label.into(),
            certainty: None,
            intensity: None,
        }
    }

    /// Build a point from Cartesian coordinates, deriving the spherical form
    /// via [`sphereql_core::cartesian_to_spherical`].
    pub fn from_cartesian(cat: impl Into<String>, label: impl Into<String>, xyz: [f64; 3]) -> Self {
        let sp = sphereql_core::cartesian_to_spherical(&sphereql_core::CartesianPoint::new(
            xyz[0], xyz[1], xyz[2],
        ));
        Self {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
            r: sp.r,
            theta: sp.theta,
            phi: sp.phi,
            cat: cat.into(),
            label: label.into(),
            certainty: None,
            intensity: None,
        }
    }

    /// Attach certainty / intensity quality signals (builder-style).
    pub fn with_quality(mut self, certainty: f64, intensity: f64) -> Self {
        self.certainty = Some(certainty);
        self.intensity = Some(intensity);
        self
    }

    /// True when every numeric field is finite (no `NaN`/`Inf`).
    pub fn is_finite(&self) -> bool {
        [self.x, self.y, self.z, self.r, self.theta, self.phi]
            .iter()
            .all(|v| v.is_finite())
            && self.certainty.is_none_or(f64::is_finite)
            && self.intensity.is_none_or(f64::is_finite)
    }

    fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Projection-quality metadata shown in the stats panel.
///
/// `evr_label` is set by the producer so the panel reports the right metric
/// per projection family (PCA: "PCA variance"; UMAP: "UMAP kNN-recall";
/// Laplacian: "connectivity ratio"; …) rather than a hardcoded string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneStats {
    /// Projection family name (e.g. `"pca"`, `"umap_sphere"`).
    pub projection_kind: String,
    /// The headline quality number in `[0, 1]`. Named `evr` for backwards
    /// compatibility with existing consumers, but its *meaning* is given by
    /// `evr_label`.
    pub evr: f64,
    /// Human-readable label for `evr` (e.g. `"PCA variance"`).
    pub evr_label: String,
    /// When the point cloud was decimated, the original (pre-sample) count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampled_from: Option<usize>,
    /// Count of non-finite points dropped during [`SceneBuilder::build`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropped_nonfinite: Option<usize>,
}

impl SceneStats {
    /// Construct stats with a generic "Explained variance ratio" label.
    pub fn new(projection_kind: impl Into<String>, evr: f64) -> Self {
        Self {
            projection_kind: projection_kind.into(),
            evr,
            evr_label: "Explained variance ratio".to_string(),
            sampled_from: None,
            dropped_nonfinite: None,
        }
    }

    /// Set the metric label (builder-style).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.evr_label = label.into();
        self
    }
}

/// A fully-built visualization scene.
///
/// Construct via [`Scene::builder`]; the builder filters non-finite points,
/// computes [`Scene::surface_radius`] when not given, and applies any
/// decimation cap. Fields are public for convenience but the builder is the
/// supported construction path (it is where sanitization happens).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub title: String,
    pub points: Vec<ScenePoint>,
    pub overlays: Vec<crate::overlay::Overlay>,
    pub stats: SceneStats,
    /// Radius at which the reference sphere and on-sphere overlays are drawn.
    pub surface_radius: f64,
    /// Draw the XYZ reference axis triad.
    pub show_axes: bool,
}

impl Scene {
    /// Start building a scene.
    pub fn builder() -> SceneBuilder {
        SceneBuilder::default()
    }

    /// Median `‖(x, y, z)‖` of `points`; `1.0` for an empty or degenerate
    /// cloud. This is the shell on which on-sphere overlays should be placed.
    pub fn surface_radius_for(points: &[ScenePoint]) -> f64 {
        let mut norms: Vec<f64> = points
            .iter()
            .map(ScenePoint::norm)
            .filter(|n| n.is_finite() && *n > 0.0)
            .collect();
        if norms.is_empty() {
            return 1.0;
        }
        norms.sort_by(f64::total_cmp);
        norms[norms.len() / 2]
    }

    /// Render to a self-contained HTML string (three.js inlined, offline).
    pub fn to_html(&self) -> String {
        crate::emit::render_html(self, crate::emit::ScriptSource::Inline)
    }

    /// Render to an HTML string that loads three.js from a CDN (smaller file,
    /// requires network to view).
    pub fn to_html_cdn(&self) -> String {
        crate::emit::render_html(self, crate::emit::ScriptSource::Cdn)
    }

    /// Render and write to `path`, returning the canonicalized path.
    pub fn write_html(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<std::path::PathBuf> {
        let path = path.as_ref();
        std::fs::write(path, self.to_html())?;
        std::fs::canonicalize(path)
    }
}

/// Place a spherical direction on the display shell of radius `surface_radius`.
///
/// Ignores the input's radial component (uses its angular direction only), so
/// every overlay built with this helper lands exactly on `surface_radius`.
pub fn on_surface(dir: &SphericalPoint, surface_radius: f64) -> [f64; 3] {
    let u = dir.unit_cartesian();
    [
        u[0] * surface_radius,
        u[1] * surface_radius,
        u[2] * surface_radius,
    ]
}

/// Builder for [`Scene`]. See [`Scene::builder`].
#[derive(Debug, Default)]
pub struct SceneBuilder {
    title: Option<String>,
    points: Vec<ScenePoint>,
    overlays: Vec<crate::overlay::Overlay>,
    stats: Option<SceneStats>,
    surface_radius: Option<f64>,
    show_axes: bool,
    max_points: Option<usize>,
}

impl SceneBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn point(mut self, p: ScenePoint) -> Self {
        self.points.push(p);
        self
    }

    pub fn points(mut self, it: impl IntoIterator<Item = ScenePoint>) -> Self {
        self.points.extend(it);
        self
    }

    pub fn overlay(mut self, o: crate::overlay::Overlay) -> Self {
        self.overlays.push(o);
        self
    }

    pub fn overlays(mut self, it: impl IntoIterator<Item = crate::overlay::Overlay>) -> Self {
        self.overlays.extend(it);
        self
    }

    pub fn stats(mut self, s: SceneStats) -> Self {
        self.stats = Some(s);
        self
    }

    /// Force the display shell radius. When unset, it is computed from the
    /// (sanitized) point cloud via [`Scene::surface_radius_for`].
    pub fn surface_radius(mut self, r: f64) -> Self {
        self.surface_radius = Some(r);
        self
    }

    pub fn show_axes(mut self, b: bool) -> Self {
        self.show_axes = b;
        self
    }

    /// Cap the rendered point count. Above the cap, a deterministic
    /// per-category stratified sample keeps every category visible.
    pub fn max_points(mut self, n: usize) -> Self {
        self.max_points = Some(n);
        self
    }

    /// Finalize the scene. Drops non-finite points (recorded in
    /// `stats.dropped_nonfinite`), applies decimation (recorded in
    /// `stats.sampled_from`), and computes `surface_radius` if unset.
    pub fn build(self) -> Scene {
        let original_len = self.points.len();
        let mut points: Vec<ScenePoint> = self
            .points
            .into_iter()
            .filter(ScenePoint::is_finite)
            .collect();
        let dropped = original_len - points.len();

        let mut sampled_from = None;
        if let Some(cap) = self.max_points
            && points.len() > cap
        {
            let before = points.len();
            points = stratified_sample(points, cap);
            sampled_from = Some(before);
        }

        let surface_radius = self
            .surface_radius
            .unwrap_or_else(|| Scene::surface_radius_for(&points));

        let mut stats = self
            .stats
            .unwrap_or_else(|| SceneStats::new("unknown", 0.0));
        if dropped > 0 {
            stats.dropped_nonfinite = Some(dropped);
        }
        if sampled_from.is_some() {
            stats.sampled_from = sampled_from;
        }

        Scene {
            title: self
                .title
                .unwrap_or_else(|| "SphereQL Visualization".to_string()),
            points,
            overlays: self.overlays,
            stats,
            surface_radius,
            show_axes: self.show_axes,
        }
    }
}

/// Deterministic per-category stratified sample down to roughly `cap` points.
///
/// Each category keeps a share of `cap` proportional to its size (at least
/// one point), selected by an even stride so the sample is spatially spread
/// and reproducible (no RNG).
fn stratified_sample(points: Vec<ScenePoint>, cap: usize) -> Vec<ScenePoint> {
    use std::collections::BTreeMap;

    // Group indices by category, preserving insertion order within a group.
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, p) in points.iter().enumerate() {
        groups.entry(p.cat.as_str()).or_default().push(i);
    }
    let total = points.len();

    let mut keep: Vec<usize> = Vec::with_capacity(cap + groups.len());
    for idxs in groups.values() {
        let share = ((idxs.len() * cap) / total).max(1).min(idxs.len());
        // Even stride across the group's members.
        let stride = (idxs.len() as f64 / share as f64).max(1.0);
        let mut acc = 0.0;
        let mut taken = 0;
        while taken < share {
            let pick = (acc as usize).min(idxs.len() - 1);
            keep.push(idxs[pick]);
            acc += stride;
            taken += 1;
        }
    }
    keep.sort_unstable();
    keep.dedup();

    // Rebuild in original order.
    let keep_set: std::collections::HashSet<usize> = keep.into_iter().collect();
    points
        .into_iter()
        .enumerate()
        .filter_map(|(i, p)| keep_set.contains(&i).then_some(p))
        .collect()
}
