//! Spatial quality metrics computed from category geometry on S².
//!
//! Bridges the gap between raw spatial primitives (`sphereql_core::spatial`)
//! and the category enrichment layer. Computed once at pipeline build time,
//! then fed into bridge detection, edge weighting, and confidence scoring.

use sphereql_core::SphericalPoint;
use sphereql_core::spatial::{
    CoverageReport, VoronoiCell, cap_exclusivity, cap_intersection_area, cap_solid_angle,
    estimate_coverage, spherical_voronoi,
};

use crate::category::CategoryGraph;
use crate::config::PipelineConfig;

/// Pre-computed spatial properties of the category layout on S².
///
/// Every field here is derived from the category centroids and angular
/// spreads — no embedding-space information, pure sphere geometry.
/// This struct is computed once during `CategoryLayer::build()` and
/// informs bridge detection, edge weights, and confidence scoring.
#[derive(Debug, Clone)]
pub struct SpatialQuality {
    /// Global explained variance ratio of the projection.
    pub evr: f64,

    /// Solid angle of each category's cap (2π(1 − cos α)).
    pub cap_areas: Vec<f64>,

    /// Per-category exclusivity: fraction of cap not overlapped by any other.
    /// 1.0 = isolated, 0.0 = completely overlapped.
    pub exclusivities: Vec<f64>,

    /// Voronoi cell for each category (area + neighbor indices).
    pub voronoi_cells: Vec<VoronoiCell>,

    /// Pairwise cap intersection areas. Keyed by (min(i,j), max(i,j)).
    pub pairwise_intersections: Vec<PairIntersection>,

    /// Coverage report: what fraction of S² is claimed by any category.
    pub coverage: CoverageReport,

    /// EVR-adaptive bridge threshold. Higher EVR → looser threshold.
    /// Formula: 0.5 + (1 − EVR)² × 0.4
    pub bridge_threshold: f64,

    /// C×C matrix of spatially-adjusted bridge quality between category pairs.
    /// `matrix[i][j] = max_bridge_strength(i,j) × territorial_factor(i,j)`.
    /// Empty until [`Self::set_bridge_quality_matrix`] is called with a
    /// built [`CategoryGraph`] (done during `CategoryLayer::build`).
    pub bridge_quality_matrix: Vec<Vec<f64>>,
}

/// Cap intersection area between two categories on S².
///
/// Stored in [`SpatialQuality::pairwise_intersections`]; only pairs with
/// measurable overlap (> 1e-15 sr) are kept to keep the list sparse.
#[derive(Debug, Clone, Copy)]
pub struct PairIntersection {
    /// Lower of the two category indices (`min(i, j)`).
    pub cat_a: usize,
    /// Higher of the two category indices (`max(i, j)`).
    pub cat_b: usize,
    /// Overlap area of the two caps, in steradians.
    pub area: f64,
}

impl SpatialQuality {
    /// Compute spatial quality from category centroids and angular spreads,
    /// using the legacy default Monte Carlo sample counts.
    ///
    /// Prefer [`Self::compute_with_config`] when you need to tune sample
    /// counts or the EVR-adaptive bridge threshold formula.
    pub fn compute(centroids: &[SphericalPoint], half_angles: &[f64], evr: f64) -> Self {
        Self::compute_with_config(centroids, half_angles, evr, &PipelineConfig::default())
    }

    /// Compute spatial quality using configurable sample counts and bridge
    /// threshold parameters.
    ///
    /// Cost at default sample counts: ~100-200ms for 31 categories. This is
    /// a one-time build cost, not per-query.
    pub fn compute_with_config(
        centroids: &[SphericalPoint],
        half_angles: &[f64],
        evr: f64,
        config: &PipelineConfig,
    ) -> Self {
        assert_eq!(
            centroids.len(),
            half_angles.len(),
            "centroids and half_angles must have matching length"
        );
        let n = centroids.len();
        let sc = &config.spatial;

        let cap_areas: Vec<f64> = half_angles.iter().map(|&a| cap_solid_angle(a)).collect();

        let exclusivities: Vec<f64> = (0..n)
            .map(|i| cap_exclusivity(i, centroids, half_angles, sc.exclusivity_samples))
            .collect();

        let voronoi_cells = spherical_voronoi(centroids, sc.voronoi_samples);

        let mut pairwise_intersections =
            Vec::with_capacity(if n >= 2 { n * (n - 1) / 2 } else { 0 });
        for i in 0..n {
            for j in (i + 1)..n {
                let area = cap_intersection_area(
                    &centroids[i],
                    half_angles[i],
                    &centroids[j],
                    half_angles[j],
                );
                if area > 1e-15 {
                    pairwise_intersections.push(PairIntersection {
                        cat_a: i,
                        cat_b: j,
                        area,
                    });
                }
            }
        }

        let coverage = estimate_coverage(centroids, half_angles, sc.coverage_samples);

        // Higher EVR → looser threshold (more of the geometry is trustworthy).
        let bridge_threshold = config.bridges.evr_adaptive_threshold(evr);

        Self {
            evr,
            cap_areas,
            exclusivities,
            voronoi_cells,
            pairwise_intersections,
            coverage,
            bridge_threshold,
            bridge_quality_matrix: vec![vec![0.0; n]; n],
        }
    }

    /// Populate the C×C `bridge_quality_matrix` from a freshly built graph.
    ///
    /// Each cell is `edge.max_bridge_strength × territorial_factor(i, j)`,
    /// left at 0.0 where no edge exists (including the diagonal).
    pub fn set_bridge_quality_matrix(&mut self, graph: &CategoryGraph) {
        let n = self.exclusivities.len();
        self.bridge_quality_matrix = vec![vec![0.0; n]; n];
        for (i, edges) in graph.adjacency.iter().enumerate() {
            for edge in edges {
                let j = edge.target;
                self.bridge_quality_matrix[i][j] =
                    edge.max_bridge_strength * self.territorial_factor(i, j);
            }
        }
    }

    /// Exclusivity-based territorial factor for a category pair.
    ///
    /// Bridges between categories that heavily overlap (low exclusivity)
    /// are discounted — they're shared territory, not genuine connectors.
    /// Returns a value in (0, 1].
    pub fn territorial_factor(&self, cat_a: usize, cat_b: usize) -> f64 {
        let ea = self.exclusivities.get(cat_a).copied().unwrap_or(1.0);
        let eb = self.exclusivities.get(cat_b).copied().unwrap_or(1.0);
        (ea * eb).sqrt().max(0.05)
    }

    /// Whether two categories are Voronoi neighbors (geometrically adjacent on S²).
    pub fn are_voronoi_neighbors(&self, cat_a: usize, cat_b: usize) -> bool {
        self.voronoi_cells
            .get(cat_a)
            .is_some_and(|cell| cell.neighbor_indices.contains(&cat_b))
    }

    /// Voronoi cell area for a category.
    pub fn voronoi_area(&self, cat: usize) -> f64 {
        self.voronoi_cells.get(cat).map_or(0.0, |cell| cell.area)
    }

    /// Territorial efficiency: items per steradian of Voronoi cell.
    pub fn territorial_efficiency(&self, cat: usize, item_count: usize) -> f64 {
        let area = self.voronoi_area(cat);
        if area > 1e-15 {
            item_count as f64 / area
        } else {
            0.0
        }
    }

    /// Cap intersection area between two categories.
    pub fn intersection_area(&self, cat_a: usize, cat_b: usize) -> f64 {
        let (lo, hi) = if cat_a < cat_b {
            (cat_a, cat_b)
        } else {
            (cat_b, cat_a)
        };
        self.pairwise_intersections
            .iter()
            .find(|p| p.cat_a == lo && p.cat_b == hi)
            .map_or(0.0, |p| p.area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn unit(theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(1.0, theta, phi)
    }

    #[test]
    fn spatial_quality_basic() {
        let centroids = vec![
            unit(0.0, FRAC_PI_2),
            unit(PI, FRAC_PI_2),
            unit(FRAC_PI_2, FRAC_PI_2),
        ];
        let half_angles = vec![0.5, 0.5, 0.5];
        let sq = SpatialQuality::compute(&centroids, &half_angles, 0.5);

        assert_eq!(sq.cap_areas.len(), 3);
        assert_eq!(sq.exclusivities.len(), 3);
        assert_eq!(sq.voronoi_cells.len(), 3);
        assert!(sq.coverage.coverage_fraction > 0.0);
        assert!(sq.bridge_threshold > 0.5);
    }

    #[test]
    fn bridge_threshold_scales_with_evr() {
        let centroids = vec![unit(0.0, FRAC_PI_2)];
        let half_angles = vec![0.5];

        let sq_low = SpatialQuality::compute(&centroids, &half_angles, 0.19);
        let sq_high = SpatialQuality::compute(&centroids, &half_angles, 0.80);

        assert!(
            sq_low.bridge_threshold > sq_high.bridge_threshold,
            "low EVR should have stricter threshold: {} vs {}",
            sq_low.bridge_threshold,
            sq_high.bridge_threshold
        );
    }

    #[test]
    fn territorial_factor_range() {
        let centroids = vec![unit(0.0, FRAC_PI_2), unit(PI, FRAC_PI_2)];
        let half_angles = vec![0.5, 0.5];
        let sq = SpatialQuality::compute(&centroids, &half_angles, 0.5);

        let tf = sq.territorial_factor(0, 1);
        assert!(
            tf > 0.0 && tf <= 1.0,
            "territorial factor out of range: {tf}"
        );
    }

    #[test]
    fn voronoi_neighbors_detected() {
        let centroids = vec![
            unit(0.0, FRAC_PI_2),
            unit(0.5, FRAC_PI_2),
            unit(PI, FRAC_PI_2),
        ];
        let half_angles = vec![0.3, 0.3, 0.3];
        let sq = SpatialQuality::compute(&centroids, &half_angles, 0.5);

        assert!(
            sq.are_voronoi_neighbors(0, 1),
            "close centroids should be Voronoi neighbors"
        );
    }

    #[test]
    fn exclusivities_bounded() {
        let centroids = vec![unit(0.0, FRAC_PI_2), unit(PI, FRAC_PI_2)];
        let half_angles = vec![0.3, 0.3];
        let sq = SpatialQuality::compute(&centroids, &half_angles, 0.5);

        for &e in &sq.exclusivities {
            assert!((0.0..=1.0).contains(&e), "exclusivity out of range: {e}");
        }
    }
}
