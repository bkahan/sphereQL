//! AI Knowledge Navigator — semantic spatial queries on S².
//!
//! Wires the pure geometry primitives from `sphereql_core::spatial` to the
//! [`CategoryLayer`] and [`SphereQLPipeline`], giving every geometric query
//! its semantic meaning.
//!
//! Each public struct/function maps to one of the 7 research areas:
//! §1 Antipodal analysis, §2 Coverage & knowledge gaps, §3 Geodesic sweeps,
//! §4 Voronoi tessellation, §5 Overlap & exclusivity, §6 Curvature signatures,
//! §7 Lune decomposition.

use std::collections::HashMap;

use sphereql_core::spatial::*;
use sphereql_core::{SphericalPoint, angular_distance};

use crate::category::CategoryLayer;

// ── §1: Antipodal Analysis ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AntipodalReport {
    pub category_name: String,
    pub centroid: SphericalPoint,
    pub antipode_position: SphericalPoint,
    pub antipodal_items: Vec<AntipodalItem>,
    /// > 1.0 = denser than chance (structured). < 1.0 = sparser (noise).
    pub antipodal_coherence: f64,
    pub dominant_antipodal_category: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AntipodalItem {
    pub item_index: usize,
    pub category: String,
    pub distance_to_antipode: f64,
}

pub fn antipodal_analysis(
    layer: &CategoryLayer,
    all_positions: &[SphericalPoint],
    all_categories: &[String],
    radius: f64,
) -> Vec<AntipodalReport> {
    layer
        .summaries
        .iter()
        .map(|summary| {
            let ap = antipode(&summary.centroid_position);

            let mut items: Vec<AntipodalItem> = all_positions
                .iter()
                .enumerate()
                .filter_map(|(i, pos)| {
                    let d = angular_distance(&ap, pos);
                    if d <= radius {
                        Some(AntipodalItem {
                            item_index: i,
                            category: all_categories[i].clone(),
                            distance_to_antipode: d,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            items.sort_by(|a, b| {
                a.distance_to_antipode
                    .partial_cmp(&b.distance_to_antipode)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let coherence = region_coherence(&ap, radius, all_positions);

            let dominant = if items.is_empty() {
                None
            } else {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for item in &items {
                    *counts.entry(item.category.as_str()).or_default() += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(name, _)| name.to_string())
            };

            AntipodalReport {
                category_name: summary.name.clone(),
                centroid: summary.centroid_position,
                antipode_position: ap,
                antipodal_items: items,
                antipodal_coherence: coherence,
                dominant_antipodal_category: dominant,
            }
        })
        .collect()
}

// ── §2: Coverage & Knowledge Gap Cartography ───────────────────────────

#[derive(Debug, Clone)]
pub struct KnowledgeCoverageReport {
    pub coverage_fraction: f64,
    pub covered_area: f64,
    pub overlap_area: f64,
    pub category_caps: Vec<CategoryCapInfo>,
    pub void_samples: usize,
    pub total_samples: usize,
}

#[derive(Debug, Clone)]
pub struct CategoryCapInfo {
    pub name: String,
    pub centroid: SphericalPoint,
    pub half_angle: f64,
    pub solid_angle: f64,
}

pub fn knowledge_coverage(layer: &CategoryLayer, num_samples: usize) -> KnowledgeCoverageReport {
    let centers: Vec<SphericalPoint> = layer
        .summaries
        .iter()
        .map(|s| s.centroid_position)
        .collect();
    let half_angles: Vec<f64> = layer.summaries.iter().map(|s| s.angular_spread).collect();
    let report = estimate_coverage(&centers, &half_angles, num_samples);

    let category_caps: Vec<CategoryCapInfo> = layer
        .summaries
        .iter()
        .map(|s| CategoryCapInfo {
            name: s.name.clone(),
            centroid: s.centroid_position,
            half_angle: s.angular_spread,
            solid_angle: cap_solid_angle(s.angular_spread),
        })
        .collect();

    KnowledgeCoverageReport {
        coverage_fraction: report.coverage_fraction,
        covered_area: report.covered_area,
        overlap_area: report.overlap_area,
        category_caps,
        void_samples: report.void_count,
        total_samples: report.total_samples,
    }
}

/// Gap-aware confidence: sigmoid falloff based on void_distance.
#[must_use]
pub fn gap_confidence(query: &SphericalPoint, layer: &CategoryLayer, sharpness: f64) -> f64 {
    let centers: Vec<SphericalPoint> = layer
        .summaries
        .iter()
        .map(|s| s.centroid_position)
        .collect();
    let half_angles: Vec<f64> = layer.summaries.iter().map(|s| s.angular_spread).collect();
    let vd = void_distance(query, &centers, &half_angles);
    1.0 / (1.0 + (sharpness * vd).exp())
}

// ── §3: Geodesic Sweep Queries ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GeodesicSweepReport {
    pub start_name: String,
    pub end_name: String,
    pub arc_length: f64,
    pub items: Vec<GeodesicSweepItem>,
    pub density_profile: Vec<usize>,
    pub gap_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct GeodesicSweepItem {
    pub item_index: usize,
    pub category: String,
    pub distance_to_arc: f64,
}

pub fn category_geodesic_sweep(
    layer: &CategoryLayer,
    source_category: &str,
    target_category: &str,
    all_positions: &[SphericalPoint],
    all_categories: &[String],
    epsilon: f64,
    density_bins: usize,
) -> Option<GeodesicSweepReport> {
    let src = layer.get_category(source_category)?;
    let tgt = layer.get_category(target_category)?;

    let hits = geodesic_sweep(
        &src.centroid_position,
        &tgt.centroid_position,
        all_positions,
        epsilon,
    );

    let items: Vec<GeodesicSweepItem> = hits
        .iter()
        .map(|&(idx, dist)| GeodesicSweepItem {
            item_index: idx,
            category: all_categories[idx].clone(),
            distance_to_arc: dist,
        })
        .collect();

    let profile = geodesic_density_profile(
        &src.centroid_position,
        &tgt.centroid_position,
        all_positions,
        epsilon,
        density_bins,
    );

    let gap_fraction = if profile.is_empty() {
        1.0
    } else {
        profile.iter().filter(|&&c| c == 0).count() as f64 / profile.len() as f64
    };

    Some(GeodesicSweepReport {
        start_name: source_category.to_string(),
        end_name: target_category.to_string(),
        arc_length: angular_distance(&src.centroid_position, &tgt.centroid_position),
        items,
        density_profile: profile,
        gap_fraction,
    })
}

#[must_use]
pub fn category_path_deviation(layer: &CategoryLayer, source: &str, target: &str) -> Option<f64> {
    let path = layer.category_path(source, target)?;
    if path.steps.len() < 2 {
        return Some(0.0);
    }
    let waypoints: Vec<SphericalPoint> = path
        .steps
        .iter()
        .map(|step| layer.summaries[step.category_index].centroid_position)
        .collect();
    Some(geodesic_deviation(&waypoints))
}

// ── §4: Voronoi Tessellation ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VoronoiReport {
    pub cells: Vec<VoronoiCellReport>,
    pub total_area: f64,
}

#[derive(Debug, Clone)]
pub struct VoronoiCellReport {
    pub category_name: String,
    pub cell_area: f64,
    pub voronoi_neighbors: Vec<String>,
    pub item_count: usize,
    pub territorial_efficiency: f64,
    pub graph_neighbor_overlap: f64,
}

pub fn voronoi_analysis(layer: &CategoryLayer, num_samples: usize) -> VoronoiReport {
    let centroids: Vec<SphericalPoint> = layer
        .summaries
        .iter()
        .map(|s| s.centroid_position)
        .collect();
    let cells = spherical_voronoi(&centroids, num_samples);

    let cell_reports: Vec<VoronoiCellReport> = cells
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let summary = &layer.summaries[i];
            let voronoi_neighbors: Vec<String> = cell
                .neighbor_indices
                .iter()
                .map(|&j| layer.summaries[j].name.clone())
                .collect();

            let efficiency = if cell.area > 1e-15 {
                summary.member_count as f64 / cell.area
            } else {
                0.0
            };

            let graph_neighbors: Vec<usize> =
                layer.graph.adjacency[i].iter().map(|e| e.target).collect();
            let voronoi_set: std::collections::HashSet<usize> =
                cell.neighbor_indices.iter().copied().collect();
            let graph_set: std::collections::HashSet<usize> =
                graph_neighbors.iter().copied().collect();
            let intersection = voronoi_set.intersection(&graph_set).count();
            let union_count = voronoi_set.union(&graph_set).count();
            let overlap = if union_count > 0 {
                intersection as f64 / union_count as f64
            } else {
                1.0
            };

            VoronoiCellReport {
                category_name: summary.name.clone(),
                cell_area: cell.area,
                voronoi_neighbors,
                item_count: summary.member_count,
                territorial_efficiency: efficiency,
                graph_neighbor_overlap: overlap,
            }
        })
        .collect();

    let total_area: f64 = cell_reports.iter().map(|c| c.cell_area).sum();
    VoronoiReport {
        cells: cell_reports,
        total_area,
    }
}

// ── §5: Overlap & Exclusivity Analysis ───────────────────────────────

#[derive(Debug, Clone)]
pub struct OverlapReport {
    pub pairs: Vec<OverlapPair>,
    pub exclusivities: Vec<CategoryExclusivity>,
}

#[derive(Debug, Clone)]
pub struct OverlapPair {
    pub category_a: String,
    pub category_b: String,
    pub intersection_area: f64,
    pub bridge_count: usize,
    pub overlap_bridge_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct CategoryExclusivity {
    pub category_name: String,
    pub cap_area: f64,
    pub exclusivity: f64,
}

pub fn overlap_analysis(layer: &CategoryLayer, mc_samples_per_cap: usize) -> OverlapReport {
    let centers: Vec<SphericalPoint> = layer
        .summaries
        .iter()
        .map(|s| s.centroid_position)
        .collect();
    let half_angles: Vec<f64> = layer.summaries.iter().map(|s| s.angular_spread).collect();
    let raw_overlaps = pairwise_overlaps(&centers, &half_angles);

    let pairs: Vec<OverlapPair> = raw_overlaps
        .iter()
        .map(|ov| {
            let bridge_count = layer
                .graph
                .bridges
                .get(&(ov.category_a, ov.category_b))
                .map_or(0, |b| b.len())
                + layer
                    .graph
                    .bridges
                    .get(&(ov.category_b, ov.category_a))
                    .map_or(0, |b| b.len());
            let ratio = if bridge_count > 0 {
                ov.intersection_area / bridge_count as f64
            } else if ov.intersection_area > 1e-15 {
                f64::INFINITY
            } else {
                0.0
            };

            OverlapPair {
                category_a: layer.summaries[ov.category_a].name.clone(),
                category_b: layer.summaries[ov.category_b].name.clone(),
                intersection_area: ov.intersection_area,
                bridge_count,
                overlap_bridge_ratio: ratio,
            }
        })
        .collect();

    let exclusivities: Vec<CategoryExclusivity> = (0..layer.summaries.len())
        .map(|i| {
            let exc = cap_exclusivity(i, &centers, &half_angles, mc_samples_per_cap);
            CategoryExclusivity {
                category_name: layer.summaries[i].name.clone(),
                cap_area: cap_solid_angle(half_angles[i]),
                exclusivity: exc,
            }
        })
        .collect();

    OverlapReport {
        pairs,
        exclusivities,
    }
}

// ── §6: Curvature Signatures ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CurvatureReport {
    pub top_triples: Vec<CurvatureTriple>,
    pub signatures: Vec<CategoryCurvatureSignature>,
}

#[derive(Debug, Clone)]
pub struct CurvatureTriple {
    pub categories: [String; 3],
    pub excess: f64,
}

#[derive(Debug, Clone)]
pub struct CategoryCurvatureSignature {
    pub category_name: String,
    pub mean_excess: f64,
    pub max_excess: f64,
    pub min_excess: f64,
}

pub fn curvature_analysis(layer: &CategoryLayer, top_n: usize) -> CurvatureReport {
    let centroids: Vec<SphericalPoint> = layer
        .summaries
        .iter()
        .map(|s| s.centroid_position)
        .collect();
    let n = centroids.len();

    let mut triples: Vec<CurvatureTriple> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let excess = spherical_excess(&centroids[i], &centroids[j], &centroids[k]);
                triples.push(CurvatureTriple {
                    categories: [
                        layer.summaries[i].name.clone(),
                        layer.summaries[j].name.clone(),
                        layer.summaries[k].name.clone(),
                    ],
                    excess,
                });
            }
        }
    }
    triples.sort_by(|a, b| {
        b.excess
            .partial_cmp(&a.excess)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let signatures: Vec<CategoryCurvatureSignature> = (0..n)
        .map(|target| {
            let sig = curvature_signature(target, &centroids);
            let (mean, min, max) = if sig.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let sum: f64 = sig.iter().sum();
                (sum / sig.len() as f64, sig[0], sig[sig.len() - 1])
            };
            CategoryCurvatureSignature {
                category_name: layer.summaries[target].name.clone(),
                mean_excess: mean,
                max_excess: max,
                min_excess: min,
            }
        })
        .collect();

    CurvatureReport {
        top_triples: triples.into_iter().take(top_n).collect(),
        signatures,
    }
}

// ── §7: Lune Decomposition ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LuneReport {
    pub category_a: String,
    pub category_b: String,
    pub a_leaning_count: usize,
    pub b_leaning_count: usize,
    pub on_bisector_count: usize,
    pub asymmetry: f64,
    pub bisector_voronoi_divergence: f64,
}

pub fn lune_analysis(layer: &CategoryLayer, all_positions: &[SphericalPoint]) -> Vec<LuneReport> {
    let n = layer.summaries.len();
    let mut reports = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let bridges_ij = layer.graph.bridges.get(&(i, j));
            let bridges_ji = layer.graph.bridges.get(&(j, i));

            let all_bridge_indices: Vec<usize> = bridges_ij
                .into_iter()
                .chain(bridges_ji)
                .flat_map(|list| list.iter().map(|b| b.item_index))
                .collect();

            if all_bridge_indices.is_empty() {
                continue;
            }

            let ca = &layer.summaries[i].centroid_position;
            let cb = &layer.summaries[j].centroid_position;

            let (mut a_count, mut b_count, mut on_count) = (0usize, 0usize, 0usize);
            for &idx in &all_bridge_indices {
                match lune_classify(ca, cb, &all_positions[idx]) {
                    LuneSide::CloserToA => a_count += 1,
                    LuneSide::CloserToB => b_count += 1,
                    LuneSide::OnBisector => on_count += 1,
                }
            }

            let total = (a_count + b_count + on_count) as f64;
            let asymmetry = if total > 0.0 {
                (a_count as f64 - b_count as f64).abs() / total
            } else {
                0.0
            };

            let mid = sphereql_core::slerp(ca, cb, 0.5);

            let mut min_dist = f64::INFINITY;
            let mut closest_other = None;
            for (k, summary) in layer.summaries.iter().enumerate() {
                if k == i || k == j {
                    continue;
                }
                let d = angular_distance(&mid, &summary.centroid_position);
                if d < min_dist {
                    min_dist = d;
                    closest_other = Some(k);
                }
            }

            let divergence = if let Some(_k) = closest_other {
                let d_i = angular_distance(&mid, ca);
                let d_j = angular_distance(&mid, cb);
                let d_expected = d_i.min(d_j);
                if min_dist < d_expected {
                    (d_expected - min_dist).abs()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            reports.push(LuneReport {
                category_a: layer.summaries[i].name.clone(),
                category_b: layer.summaries[j].name.clone(),
                a_leaning_count: a_count,
                b_leaning_count: b_count,
                on_bisector_count: on_count,
                asymmetry,
                bisector_voronoi_divergence: divergence,
            });
        }
    }
    reports
}

// ── Full Navigator Report ──────────────────────────────────────────

pub struct NavigatorConfig {
    pub antipodal_radius: f64,
    pub coverage_samples: usize,
    pub geodesic_epsilon: f64,
    pub density_bins: usize,
    pub voronoi_samples: usize,
    pub exclusivity_samples: usize,
    pub curvature_top_n: usize,
    pub gap_sharpness: f64,
}

impl Default for NavigatorConfig {
    fn default() -> Self {
        Self {
            antipodal_radius: 0.5,
            coverage_samples: 200_000,
            geodesic_epsilon: 0.3,
            density_bins: 20,
            voronoi_samples: 200_000,
            exclusivity_samples: 50_000,
            curvature_top_n: 20,
            gap_sharpness: 5.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NavigatorReport {
    pub antipodal: Vec<AntipodalReport>,
    pub coverage: KnowledgeCoverageReport,
    pub voronoi: VoronoiReport,
    pub overlap: OverlapReport,
    pub curvature: CurvatureReport,
    pub lunes: Vec<LuneReport>,
    pub num_categories: usize,
    pub num_items: usize,
    pub explained_variance_ratio: f64,
}

pub fn run_full_analysis(
    layer: &CategoryLayer,
    all_positions: &[SphericalPoint],
    all_categories: &[String],
    evr: f64,
    config: &NavigatorConfig,
) -> NavigatorReport {
    let antipodal = antipodal_analysis(
        layer,
        all_positions,
        all_categories,
        config.antipodal_radius,
    );
    let coverage = knowledge_coverage(layer, config.coverage_samples);
    let voronoi = voronoi_analysis(layer, config.voronoi_samples);
    let overlap = overlap_analysis(layer, config.exclusivity_samples);
    let curvature = curvature_analysis(layer, config.curvature_top_n);
    let lunes = lune_analysis(layer, all_positions);

    NavigatorReport {
        antipodal,
        coverage,
        voronoi,
        overlap,
        curvature,
        lunes,
        num_categories: layer.summaries.len(),
        num_items: all_positions.len(),
        explained_variance_ratio: evr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{PipelineInput, SphereQLPipeline};

    fn make_test_pipeline() -> (SphereQLPipeline, Vec<String>) {
        let mut embeddings = Vec::new();
        let mut categories = Vec::new();
        let dim = 10;

        for i in 0..10 {
            let mut v = vec![0.0; dim];
            v[0] = 1.0 + i as f64 * 0.02;
            v[1] = 0.1;
            embeddings.push(v);
            categories.push("alpha".to_string());
        }
        for i in 0..10 {
            let mut v = vec![0.0; dim];
            v[0] = 0.1;
            v[1] = 1.0 + i as f64 * 0.02;
            embeddings.push(v);
            categories.push("beta".to_string());
        }
        for i in 0..10 {
            let mut v = vec![0.0; dim];
            v[2] = 1.0 + i as f64 * 0.02;
            v[3] = 0.5;
            embeddings.push(v);
            categories.push("gamma".to_string());
        }

        let pipeline = SphereQLPipeline::new(PipelineInput {
            categories: categories.clone(),
            embeddings,
        })
        .unwrap();
        (pipeline, categories)
    }

    fn get_positions(pipeline: &SphereQLPipeline) -> Vec<SphericalPoint> {
        pipeline
            .exported_points()
            .iter()
            .map(|p| SphericalPoint::new_unchecked(p.r, p.theta, p.phi))
            .collect()
    }

    #[test]
    fn antipodal_analysis_runs() {
        let (pipeline, categories) = make_test_pipeline();
        let positions = get_positions(&pipeline);
        let reports = antipodal_analysis(pipeline.category_layer(), &positions, &categories, 0.5);
        assert_eq!(reports.len(), 3);
        for r in &reports {
            assert!(!r.category_name.is_empty());
            assert!(r.antipodal_coherence >= 0.0);
        }
    }

    #[test]
    fn coverage_report_valid() {
        let (pipeline, _) = make_test_pipeline();
        let report = knowledge_coverage(pipeline.category_layer(), 50_000);
        assert!(report.coverage_fraction >= 0.0 && report.coverage_fraction <= 1.0);
        assert_eq!(report.category_caps.len(), 3);
    }

    #[test]
    fn gap_confidence_inside_vs_void() {
        let (pipeline, _) = make_test_pipeline();
        let layer = pipeline.category_layer();
        let centroid = layer.summaries[0].centroid_position;
        let ap = antipode(&centroid);
        assert!(gap_confidence(&centroid, layer, 5.0) > gap_confidence(&ap, layer, 5.0));
    }

    #[test]
    fn voronoi_report_valid() {
        let (pipeline, _) = make_test_pipeline();
        let report = voronoi_analysis(pipeline.category_layer(), 50_000);
        assert_eq!(report.cells.len(), 3);
        let total: f64 = report.cells.iter().map(|c| c.cell_area).sum();
        assert!((total - 4.0 * std::f64::consts::PI).abs() < 1.0);
    }

    #[test]
    fn overlap_report_valid() {
        let (pipeline, _) = make_test_pipeline();
        let report = overlap_analysis(pipeline.category_layer(), 20_000);
        assert_eq!(report.exclusivities.len(), 3);
        for e in &report.exclusivities {
            assert!(e.exclusivity >= 0.0 && e.exclusivity <= 1.0);
        }
    }

    #[test]
    fn curvature_report_valid() {
        let (pipeline, _) = make_test_pipeline();
        let report = curvature_analysis(pipeline.category_layer(), 5);
        assert_eq!(report.top_triples.len(), 1);
        assert_eq!(report.signatures.len(), 3);
        for sig in &report.signatures {
            assert!(sig.mean_excess >= 0.0);
        }
    }

    #[test]
    fn lune_analysis_runs() {
        let (pipeline, _) = make_test_pipeline();
        let positions = get_positions(&pipeline);
        let reports = lune_analysis(pipeline.category_layer(), &positions);
        for r in &reports {
            assert!(r.asymmetry >= 0.0 && r.asymmetry <= 1.0);
        }
    }

    #[test]
    fn full_analysis_runs() {
        let (pipeline, categories) = make_test_pipeline();
        let positions = get_positions(&pipeline);
        let evr = pipeline.explained_variance_ratio();
        let report = run_full_analysis(
            pipeline.category_layer(),
            &positions,
            &categories,
            evr,
            &NavigatorConfig::default(),
        );
        assert_eq!(report.num_categories, 3);
        assert_eq!(report.num_items, 30);
        assert!(report.explained_variance_ratio > 0.0);
    }
}
