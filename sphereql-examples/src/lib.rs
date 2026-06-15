//! Shared helpers for the sphereQL examples.
//!
//! These live in a library target (rather than duplicated across example
//! `main`s) so they can be unit-tested in CI — in particular
//! [`build_corpus_scene`], which performs the pipeline-output → visualization
//! coordinate joins that are otherwise only exercised by *building* an
//! example, never *running* it.

use std::collections::BTreeMap;

use sphereql::core::{SphericalPoint, angular_distance};
use sphereql::embed::{
    BridgeClassification, CategoryLayer, NavigatorConfig, ProjectionKind, SearchSpace,
    SphereQLPipeline, run_full_analysis,
};
use sphereql::vis::{Overlay, Scene, ScenePoint, SceneStats, cap_ring};

/// Stable category color palette — mirrors the one baked into the
/// `sphereql-vis` template so overlay colors match the point colors.
pub const PALETTE: &[&str] = &[
    "#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#fff176", "#f06292", "#4dd0e1", "#a1887f",
    "#90a4ae", "#aed581", "#7986cb", "#ffb74d", "#e57373", "#64b5f6", "#dce775", "#9575cd",
];

/// Build a `cat -> color` map using the same sorted-unique ordering the
/// template uses, so a category's centroid/bridge color matches its points.
pub fn category_colors(categories: &[String]) -> BTreeMap<String, &'static str> {
    let mut sorted: Vec<&String> = categories.iter().collect();
    sorted.sort();
    sorted.dedup();
    sorted
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), PALETTE[i % PALETTE.len()]))
        .collect()
}

/// Human-readable label for the headline quality number, per projection family.
pub fn evr_label_for(kind: &str) -> &'static str {
    match kind {
        "pca" => "PCA variance",
        "kernel_pca" => "Kernel EVR",
        "laplacian_eigenmap" => "Connectivity ratio",
        "umap_sphere" => "UMAP kNN-recall",
        _ => "Explained variance ratio",
    }
}

/// Runtime-discovered category samples used across the demos. Sorted
/// descending by member count; "most populous" layers on top of that, and
/// "pairwise most distinct" comes from greedy farthest-point sampling against
/// the projected centroids.
pub struct CategoryPicker {
    by_count: Vec<(String, SphericalPoint)>,
}

impl CategoryPicker {
    pub fn new(layer: &CategoryLayer) -> Self {
        let mut entries: Vec<(String, usize, SphericalPoint)> = layer
            .summaries
            .iter()
            .map(|s| (s.name.clone(), s.member_count, s.centroid_position))
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
        Self {
            by_count: entries.into_iter().map(|(n, _, c)| (n, c)).collect(),
        }
    }

    pub fn most_populous(&self, k: usize) -> Vec<String> {
        self.by_count
            .iter()
            .take(k)
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Greedy farthest-point set on S²: start from the most populous centroid,
    /// then repeatedly add the centroid whose nearest already-picked centroid
    /// is the furthest away. Produces k pairwise-distant category names.
    pub fn distinct_set(&self, k: usize) -> Vec<String> {
        if self.by_count.is_empty() {
            return Vec::new();
        }
        let cap = k.min(self.by_count.len());
        let mut picked: Vec<usize> = Vec::with_capacity(cap);
        picked.push(0);
        while picked.len() < cap {
            let mut best_i = 0;
            let mut best_min = f64::NEG_INFINITY;
            for i in 0..self.by_count.len() {
                if picked.contains(&i) {
                    continue;
                }
                let min_d = picked
                    .iter()
                    .map(|&p| angular_distance(&self.by_count[i].1, &self.by_count[p].1))
                    .fold(f64::INFINITY, f64::min);
                if min_d > best_min {
                    best_min = min_d;
                    best_i = i;
                }
            }
            picked.push(best_i);
        }
        picked
            .into_iter()
            .map(|i| self.by_count[i].0.clone())
            .collect()
    }

    pub fn distinct_pair(&self) -> Option<(String, String)> {
        match self.distinct_set(2).as_slice() {
            [a, b] => Some((a.clone(), b.clone())),
            _ => None,
        }
    }

    /// Build k cross-domain pairs by drawing 2k pairwise-distant categories and
    /// chunking them into disjoint pairs.
    pub fn distinct_pairs(&self, k: usize) -> Vec<(String, String)> {
        let set = self.distinct_set((2 * k).min(self.by_count.len()));
        set.chunks_exact(2)
            .map(|c| (c[0].clone(), c[1].clone()))
            .collect()
    }
}

/// Returns `(budget, search_space)` scaled to corpus size.
///
/// Laplacian eigenmap needs an O(n²) affinity matrix and is filtered out above
/// 10k concepts even if requested. UMAP uses the ANN-backed kNN graph, so it
/// stays affordable into the 10k–100k range. The caller's `selected_kinds`
/// narrows `projection_kinds` to what's feasible at this corpus size.
pub fn tuning_params(n: usize, selected_kinds: &[ProjectionKind]) -> (usize, SearchSpace) {
    let (budget, mut space) = if n <= 10_000 {
        (
            16,
            SearchSpace {
                projection_kinds: vec![
                    ProjectionKind::Pca,
                    ProjectionKind::LaplacianEigenmap,
                    ProjectionKind::UmapSphere,
                    ProjectionKind::KernelPca,
                ],
                ..SearchSpace::default()
            },
        )
    } else if n <= 100_000 {
        (8, SearchSpace::large_corpus())
    } else {
        (
            4,
            SearchSpace {
                projection_kinds: vec![ProjectionKind::Pca, ProjectionKind::UmapSphere],
                ..SearchSpace::default()
            },
        )
    };

    let feasible: Vec<ProjectionKind> = space.projection_kinds.clone();
    space.projection_kinds = selected_kinds
        .iter()
        .copied()
        .filter(|k| feasible.contains(k))
        .collect();

    if space.projection_kinds.is_empty() {
        eprintln!(
            "  Note: none of the selected projections are feasible at n={n}; falling back to PCA."
        );
        space.projection_kinds = vec![ProjectionKind::Pca];
    }

    (budget, space)
}

fn classification_name(c: BridgeClassification) -> &'static str {
    match c {
        BridgeClassification::Genuine => "Genuine",
        BridgeClassification::OverlapArtifact => "OverlapArtifact",
        BridgeClassification::Weak => "Weak",
    }
}

/// Build a rich [`Scene`] from a built pipeline: the projected point cloud
/// plus the full overlay set (centroids, classified bridges, geodesic concept
/// paths, Voronoi territory caps, antipodes, coverage caps, domain-group
/// spokes).
///
/// `labels` supplies a per-item display label aligned to the pipeline's
/// insertion order; missing entries fall back to the item id.
pub fn build_corpus_scene(
    title: &str,
    pipeline: &SphereQLPipeline,
    labels: &[&str],
    evr: f64,
) -> Scene {
    let exported = pipeline.exported_points();
    let layer = pipeline.category_layer();
    let categories: Vec<String> = exported.iter().map(|p| p.category.clone()).collect();
    let colors = category_colors(&categories);
    let color_of = |cat: &str| -> &'static str { colors.get(cat).copied().unwrap_or("#90a4ae") };

    // ── Points ──────────────────────────────────────────────────────────
    let points: Vec<ScenePoint> = exported
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let label = labels.get(i).copied().unwrap_or(p.id.as_str());
            ScenePoint::from_cartesian(p.category.clone(), label, [p.x, p.y, p.z])
                .with_quality(p.certainty, p.intensity)
        })
        .collect();
    let sr = Scene::surface_radius_for(&points);

    let mut overlays: Vec<Overlay> = Vec::new();

    // ── Centroids ───────────────────────────────────────────────────────
    for s in &layer.summaries {
        overlays.push(Overlay::centroid(
            &s.centroid_position,
            sr,
            color_of(&s.name),
            s.name.clone(),
            s.member_count,
        ));
    }

    // ── Bridges: item position → the centroid of the domain it bridges to.
    // `bridges` is keyed by (source_cat_idx, target_cat_idx); cap per pair so
    // the scene stays legible.
    let summaries = &layer.summaries;
    for (&(_ci, cj), items) in &layer.graph.bridges {
        let Some(target) = summaries.get(cj) else {
            continue;
        };
        for item in items.iter().take(2) {
            let Some(p) = exported.get(item.item_index) else {
                continue;
            };
            overlays.push(Overlay::bridge(
                [p.x, p.y, p.z],
                &target.centroid_position,
                sr,
                item.bridge_strength,
                classification_name(item.classification),
                color_of(&target.name),
            ));
        }
    }

    // ── Voronoi territory caps (from each category's cell area) ─────────
    for s in &layer.summaries {
        if s.voronoi_area > 0.0 {
            overlays.push(Overlay::voronoi_cap(
                &s.centroid_position,
                sr,
                s.voronoi_area,
                color_of(&s.name),
                s.name.clone(),
            ));
        }
    }

    // ── Domain-group hubs + spokes ──────────────────────────────────────
    for (gi, g) in pipeline.domain_groups().iter().enumerate() {
        let members: Vec<SphericalPoint> = g
            .member_categories
            .iter()
            .filter_map(|&i| summaries.get(i).map(|s| s.centroid_position))
            .collect();
        overlays.push(Overlay::domain_group(
            &g.centroid,
            &members,
            sr,
            PALETTE[gi % PALETTE.len()],
            format!("group {gi}: {}", g.category_names.join(", ")),
        ));
    }

    // ── Geodesic concept paths between a few distinct category pairs ────
    let picker = CategoryPicker::new(layer);
    for (src, tgt) in picker.distinct_pairs(2) {
        if let Some(path) = pipeline.category_path(&src, &tgt) {
            let centroids: Vec<SphericalPoint> = path
                .steps
                .iter()
                .filter_map(|step| {
                    summaries
                        .get(step.category_index)
                        .map(|s| s.centroid_position)
                })
                .collect();
            if centroids.len() >= 2 {
                overlays.push(Overlay::geodesic_path(
                    &centroids,
                    sr,
                    12,
                    "#ffffff",
                    format!("{src} → {tgt}"),
                ));
            }
        }
    }

    // ── Antipodes + coverage caps (from the full geometric analysis) ────
    let positions: Vec<SphericalPoint> = exported
        .iter()
        .map(|p| SphericalPoint::new_unchecked(p.r, p.theta, p.phi))
        .collect();
    let nav_cfg = NavigatorConfig::default();
    let report = run_full_analysis(layer, &positions, &categories, evr, &nav_cfg);

    for ar in &report.antipodal {
        // Only structured opposites (denser than chance) are worth drawing.
        if ar.antipodal_coherence > 1.0 {
            overlays.push(Overlay::antipode(
                &ar.centroid,
                &ar.antipode_position,
                sr,
                ar.antipodal_coherence,
                color_of(&ar.category_name),
                ar.category_name.clone(),
            ));
        }
    }

    let caps = report
        .coverage
        .category_caps
        .iter()
        .map(|c| cap_ring(&c.centroid, sr, c.half_angle, c.name.clone()))
        .collect::<Vec<_>>();
    overlays.push(Overlay::CoverageVoid {
        caps,
        voids: Vec::new(),
        coverage_fraction: report.coverage.coverage_fraction,
    });

    let kind = pipeline.projection_kind().name();
    Scene::builder()
        .title(title)
        .points(points)
        .overlays(overlays)
        .stats(SceneStats::new(kind, evr).with_label(evr_label_for(kind)))
        .surface_radius(sr)
        .show_axes(true)
        .build()
}
