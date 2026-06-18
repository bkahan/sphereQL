//! Server state: the in-memory corpus, its projection, the indexes that serve
//! viewport and neighbor queries, and the bounded [`Manifest`] handed to the
//! viewer up front.
//!
//! [`AppState::from_corpus`] runs the same corpus → embed → project → enrich
//! join the offline `build_corpus_scene` example does, but keeps the heavy
//! by-N artifacts server-side: the projected positions (as a spatial index for
//! cone tiling), the raw embeddings (as an ANN index for semantic neighbors),
//! and per-point metadata (for lazy inspection). Only the *bounded* aggregates
//! — stats, category centroids, domain groups, the palette — go into the
//! manifest.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use sphereql_core::SphericalPoint;
use sphereql_corpus::{CorpusId, embed};
use sphereql_embed::ann::{AnnConfig, AnnIndex};
use sphereql_embed::{PipelineConfig, PipelineInput, ProjectionKind, SphereQLPipeline};
use sphereql_index::{SpatialIndex, SpatialItem};
use sphereql_vis::{Bounds, CategoryInfo, LodScheme, Manifest, Overlay, SceneStats};

/// Stable category color palette — mirrors the one baked into the
/// `sphereql-vis` template (and replicated in the examples crate) so a
/// server-streamed point gets the same color as the offline viewer would give
/// it.
pub const PALETTE: &[&str] = &[
    "#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#fff176", "#f06292", "#4dd0e1", "#a1887f",
    "#90a4ae", "#aed581", "#7986cb", "#ffb74d", "#e57373", "#64b5f6", "#dce775", "#9575cd",
];

/// Above this many points, the O(n²) projections (Laplacian eigenmap, kernel
/// PCA) are gated out — mirrors `sphereql-examples::tuning_params`.
const ON2_PROJECTION_LIMIT: usize = 10_000;
/// Above this many points, even the ANN-backed UMAP is too heavy for an
/// interactive build; fall back to linear PCA.
const UMAP_LIMIT: usize = 100_000;

/// One point's lazily-fetched, by-N metadata. Display position + category ride
/// in tiles; everything here is fetched on demand by row via `POST /points`.
#[derive(Debug, Clone)]
pub struct StoredPoint {
    /// Display position on the shell (the same coords that ride in tiles).
    pub xyz: [f32; 3],
    /// Palette index (position in [`AppState::cat_names`]).
    pub cat: u16,
    pub label: String,
    pub certainty: f32,
    pub intensity: f32,
    /// The raw embedding (densified, downcast to f32 for transfer) — the
    /// inspector's vector sparkline.
    pub vector: Vec<f32>,
}

/// A point as the [`SpatialIndex`] sees it: just its angular position and the
/// row index needed to recover the rest. Kept lean because the index clones
/// items into its sectors.
#[derive(Debug, Clone)]
pub struct PointItem {
    pub row: u32,
    pub cat: u16,
    pub pos: SphericalPoint,
}

impl SpatialItem for PointItem {
    type Id = u32;
    fn id(&self) -> &u32 {
        &self.row
    }
    fn position(&self) -> &SphericalPoint {
        &self.pos
    }
}

/// Everything a running server holds. Shared behind an `Arc` across requests;
/// never mutated after build, so no locking is needed.
pub struct AppState {
    /// The bounded scene descriptor served at `GET /manifest`.
    pub manifest: Manifest,
    /// Row-indexed per-point metadata for the inspector.
    pub points: Vec<StoredPoint>,
    /// Projected positions, indexed for cone/viewport tile queries.
    pub spatial: SpatialIndex<PointItem>,
    /// Raw embeddings, indexed for semantic nearest-neighbor (trace) queries.
    pub ann: AnnIndex,
    /// Palette index → category name.
    pub cat_names: Vec<String>,
}

/// Why building [`AppState`] from a corpus failed.
#[derive(Debug)]
pub enum BuildError {
    /// The corpus could not be loaded (missing/corrupt Parquet, etc.).
    Load(String),
    /// The corpus loaded but held no concepts.
    Empty,
    /// The projection pipeline rejected the input.
    Pipeline(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::Load(e) => write!(f, "failed to load corpus: {e}"),
            BuildError::Empty => write!(f, "corpus is empty"),
            BuildError::Pipeline(e) => write!(f, "projection pipeline failed: {e}"),
        }
    }
}

impl Error for BuildError {}

/// Pick a feasible projection for a corpus of `n` points, downgrading O(n²)
/// families (Laplacian eigenmap, kernel PCA) past `ON2_PROJECTION_LIMIT` and
/// UMAP past `UMAP_LIMIT`. Returns the kind actually used.
pub fn gate_projection(requested: ProjectionKind, n: usize) -> ProjectionKind {
    use ProjectionKind::*;
    match requested {
        LaplacianEigenmap | KernelPca if n > ON2_PROJECTION_LIMIT => Pca,
        UmapSphere if n > UMAP_LIMIT => Pca,
        other => other,
    }
}

/// Human-readable label for the headline quality number, per projection family
/// — mirrors `sphereql-examples::evr_label_for`.
fn evr_label_for(kind: &str) -> &'static str {
    match kind {
        "pca" => "PCA variance",
        "kernel_pca" => "Kernel EVR",
        "laplacian_eigenmap" => "Connectivity ratio",
        "umap_sphere" => "UMAP kNN-recall",
        _ => "Explained variance ratio",
    }
}

/// Median ‖xyz‖ of the cloud — the display-shell radius, matching
/// `Scene::surface_radius_for`. Falls back to 1.0 for an empty/degenerate set.
fn surface_radius_for(xyz: &[[f64; 3]]) -> f64 {
    let mut norms: Vec<f64> = xyz
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .filter(|n| n.is_finite() && *n > 0.0)
        .collect();
    if norms.is_empty() {
        return 1.0;
    }
    norms.sort_by(f64::total_cmp);
    norms[norms.len() / 2]
}

impl AppState {
    /// Load a corpus, embed + project it with `requested` (gated to what's
    /// feasible at this size), build the spatial + ANN indexes, and assemble
    /// the bounded manifest. CPU-heavy and synchronous; call before serving.
    pub fn from_corpus(
        corpus: CorpusId,
        requested: ProjectionKind,
    ) -> Result<AppState, BuildError> {
        let concepts = corpus.load().map_err(|e| BuildError::Load(e.to_string()))?;
        let n = concepts.len();
        if n == 0 {
            return Err(BuildError::Empty);
        }

        // ── Corpus → dense embeddings (seed convention matches demo_scene) ──
        let categories: Vec<String> = concepts.iter().map(|c| c.category.to_string()).collect();
        let labels: Vec<String> = concepts.iter().map(|c| c.label.to_string()).collect();
        let embeddings: Vec<Vec<f64>> = concepts
            .iter()
            .enumerate()
            .map(|(i, c)| embed(&c.features, 1000 + i as u64))
            .collect();

        // ANN over the raw embeddings (semantic neighbors), and an f32 copy of
        // each vector for the inspector — both borrow `embeddings` before it is
        // moved into the pipeline.
        let ann = AnnIndex::build(&embeddings, &AnnConfig::default());
        let f32_vecs: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();

        // ── Project ────────────────────────────────────────────────────────
        let kind = gate_projection(requested, n);
        let config = PipelineConfig {
            projection_kind: kind,
            ..PipelineConfig::default()
        };
        let pipeline = SphereQLPipeline::new_with_config(
            PipelineInput {
                categories: categories.clone(),
                embeddings,
            },
            config,
        )
        .map_err(|e| BuildError::Pipeline(format!("{e:?}")))?;

        let exported = pipeline.exported_points();
        let evr = pipeline.explained_variance_ratio();
        let kind_name = pipeline.projection_kind().name();

        // ── Palette: sorted-unique categories → color + count ───────────────
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for c in &categories {
            *counts.entry(c.clone()).or_default() += 1;
        }
        let cat_names: Vec<String> = counts.keys().cloned().collect();
        let cat_id: BTreeMap<&str, u16> = cat_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i as u16))
            .collect();
        let palette: Vec<CategoryInfo> = cat_names
            .iter()
            .enumerate()
            .map(|(i, name)| CategoryInfo {
                name: name.clone(),
                color: PALETTE[i % PALETTE.len()].to_string(),
                count: counts[name],
            })
            .collect();

        // ── Per-point metadata + index items ────────────────────────────────
        let mut points = Vec::with_capacity(exported.len());
        let mut items = Vec::with_capacity(exported.len());
        let mut xyz_all: Vec<[f64; 3]> = Vec::with_capacity(exported.len());
        for (i, p) in exported.iter().enumerate() {
            let cat = cat_id.get(p.category.as_str()).copied().unwrap_or(0);
            xyz_all.push([p.x, p.y, p.z]);
            points.push(StoredPoint {
                xyz: [p.x as f32, p.y as f32, p.z as f32],
                cat,
                label: labels.get(i).cloned().unwrap_or_default(),
                certainty: p.certainty as f32,
                intensity: p.intensity as f32,
                vector: f32_vecs.get(i).cloned().unwrap_or_default(),
            });
            items.push(PointItem {
                row: i as u32,
                cat,
                pos: SphericalPoint::new_unchecked(p.r, p.theta, p.phi),
            });
        }

        let sr = surface_radius_for(&xyz_all);
        let bounds = Bounds::enclosing(xyz_all.iter());

        // ── Spatial index over projected positions ──────────────────────────
        let mut spatial: SpatialIndex<PointItem> = SpatialIndex::<PointItem>::builder()
            .theta_divisions(16)
            .phi_divisions(8)
            .build();
        for item in items {
            spatial.insert(item);
        }

        // ── Bounded overlays: category centroids + domain-group hubs ─────────
        let layer = pipeline.category_layer();
        let color_of = |name: &str| -> &'static str {
            cat_id
                .get(name)
                .map(|&i| PALETTE[i as usize % PALETTE.len()])
                .unwrap_or("#90a4ae")
        };
        let mut overlays: Vec<Overlay> = Vec::new();
        for s in &layer.summaries {
            overlays.push(Overlay::centroid(
                &s.centroid_position,
                sr,
                color_of(&s.name),
                s.name.clone(),
                s.member_count,
            ));
        }
        for (gi, g) in pipeline.domain_groups().iter().enumerate() {
            let members: Vec<SphericalPoint> = g
                .member_categories
                .iter()
                .filter_map(|&i| layer.summaries.get(i).map(|s| s.centroid_position))
                .collect();
            overlays.push(Overlay::domain_group(
                &g.centroid,
                &members,
                sr,
                PALETTE[gi % PALETTE.len()],
                format!("group {gi}: {}", g.category_names.join(", ")),
            ));
        }

        // ── Manifest ─────────────────────────────────────────────────────────
        let stats = SceneStats::new(kind_name, evr).with_label(evr_label_for(kind_name));
        let mut manifest = Manifest::new(format!("SphereQL — {}", corpus.name()), n, stats);
        manifest.surface_radius = sr;
        manifest.bounds = bounds;
        manifest.overlays = overlays;
        manifest.palette = palette;
        manifest.lod = LodScheme::default();

        Ok(AppState {
            manifest,
            points,
            spatial,
            ann,
            cat_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_downgrades_on2_projections_at_scale() {
        // Feasible at small N: kept as requested.
        assert_eq!(
            gate_projection(ProjectionKind::LaplacianEigenmap, 300),
            ProjectionKind::LaplacianEigenmap
        );
        assert_eq!(
            gate_projection(ProjectionKind::KernelPca, 5_000),
            ProjectionKind::KernelPca
        );
        // O(n²) families gated out past the limit.
        assert_eq!(
            gate_projection(ProjectionKind::LaplacianEigenmap, 50_000),
            ProjectionKind::Pca
        );
        assert_eq!(
            gate_projection(ProjectionKind::KernelPca, 50_000),
            ProjectionKind::Pca
        );
        // UMAP survives the mid range but not the millions.
        assert_eq!(
            gate_projection(ProjectionKind::UmapSphere, 50_000),
            ProjectionKind::UmapSphere
        );
        assert_eq!(
            gate_projection(ProjectionKind::UmapSphere, 2_000_000),
            ProjectionKind::Pca
        );
        // PCA is always feasible.
        assert_eq!(
            gate_projection(ProjectionKind::Pca, 2_000_000),
            ProjectionKind::Pca
        );
    }

    #[test]
    fn builds_state_from_stress_corpus() {
        let state = AppState::from_corpus(CorpusId::Stress, ProjectionKind::Pca)
            .expect("stress corpus builds");
        // The stress corpus is 300 concepts across 10 categories.
        assert_eq!(state.manifest.total_points, 300);
        assert_eq!(state.points.len(), 300);
        assert_eq!(
            state.manifest.format_version,
            sphereql_vis::MANIFEST_VERSION
        );
        // Every stored point carries a 128-d raw vector for the inspector.
        assert_eq!(state.points[0].vector.len(), sphereql_corpus::DIM);
        // Palette counts cover the whole corpus.
        let palette_total: usize = state.manifest.palette.iter().map(|c| c.count).sum();
        assert_eq!(palette_total, 300);
        assert!(!state.manifest.palette.is_empty());
        assert_eq!(state.cat_names.len(), state.manifest.palette.len());
        // PCA: stats label is the PCA one.
        assert_eq!(state.manifest.stats.projection_kind, "pca");
        assert_eq!(state.manifest.stats.evr_label, "PCA variance");
        // Bounded overlays present (at least one centroid per category).
        assert!(state.manifest.overlays.len() >= state.cat_names.len());
    }
}
