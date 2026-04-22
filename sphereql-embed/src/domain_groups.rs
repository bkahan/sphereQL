//! Hierarchical domain groups: coarse routing for noisy projections.
//!
//! When EVR is low (< 0.35), routing through N individual categories on a
//! distorted outer sphere is unreliable — random angular noise drowns the
//! signal. Collapsing the N categories into a handful of super-groups
//! (derived from Voronoi adjacency + cap overlap) reduces the routing
//! problem's dimensionality and restores usable coarse structure.

use sphereql_core::{angular_distance, cartesian_to_spherical, CartesianPoint, SphericalPoint};

use crate::category::CategoryLayer;

/// A cluster of related categories detected from sphere geometry.
#[derive(Debug, Clone)]
pub struct DomainGroup {
    /// Indices of member categories in the [`CategoryLayer`] summaries vec.
    pub member_categories: Vec<usize>,
    /// Category names for convenience.
    pub category_names: Vec<String>,
    /// Centroid of the group on S² (mean of member centroids).
    pub centroid: SphericalPoint,
    /// Angular spread of the group (mean distance of members from group centroid).
    pub angular_spread: f64,
    /// Cohesion: `1 / (1 + angular_spread)`.
    pub cohesion: f64,
    /// Total items across all member categories.
    pub total_items: usize,
}

/// Detect up to `target_groups` domain groups from the category layer.
///
/// Greedy agglomerative clustering over a Voronoi-adjacency + cap-overlap
/// similarity matrix. Pairs of spatially adjacent or heavily overlapping
/// categories are merged first; the merge stops when `target_groups`
/// clusters remain (or earlier if fewer categories exist).
pub fn detect_domain_groups(layer: &CategoryLayer, target_groups: usize) -> Vec<DomainGroup> {
    let n = layer.summaries.len();
    if n == 0 {
        return Vec::new();
    }
    let target_groups = target_groups.max(1).min(n);
    let sq = &layer.spatial_quality;

    // 1. Similarity matrix from Voronoi adjacency + normalized cap overlap.
    let mut similarity = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            if sq.are_voronoi_neighbors(i, j) {
                s += 0.5;
            }
            let overlap = sq.intersection_area(i, j);
            let min_cap = sq.cap_areas[i].min(sq.cap_areas[j]);
            if min_cap > 1e-15 {
                s += 0.5 * (overlap / min_cap).min(1.0);
            }
            similarity[i][j] = s;
            similarity[j][i] = s;
        }
    }

    // 2. Greedy agglomerative clustering (average linkage).
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > target_groups {
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let mut total = 0.0;
                let mut count = 0usize;
                for &ci in &clusters[i] {
                    for &cj in &clusters[j] {
                        total += similarity[ci][cj];
                        count += 1;
                    }
                }
                let avg = if count > 0 { total / count as f64 } else { 0.0 };
                if avg > best_sim {
                    best_sim = avg;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let merged = clusters.remove(best_j);
        clusters[best_i].extend(merged);
    }

    // 3. Build DomainGroup records.
    clusters
        .into_iter()
        .map(|members| build_group(layer, members))
        .collect()
}

fn build_group(layer: &CategoryLayer, members: Vec<usize>) -> DomainGroup {
    let category_names: Vec<String> = members
        .iter()
        .map(|&i| layer.summaries[i].name.clone())
        .collect();

    let total_items: usize = members
        .iter()
        .map(|&i| layer.summaries[i].member_count)
        .sum();

    // Group centroid: normalized mean of member unit vectors, then back to spherical.
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    for &i in &members {
        let c = layer.summaries[i].centroid_position.unit_cartesian();
        sx += c[0];
        sy += c[1];
        sz += c[2];
    }
    let mag = (sx * sx + sy * sy + sz * sz).sqrt();
    let centroid = if mag > 1e-15 {
        cartesian_to_spherical(&CartesianPoint::new(sx / mag, sy / mag, sz / mag))
    } else {
        layer.summaries[members[0]].centroid_position
    };

    let angular_spread = if members.len() > 1 {
        members
            .iter()
            .map(|&i| angular_distance(&layer.summaries[i].centroid_position, &centroid))
            .sum::<f64>()
            / members.len() as f64
    } else {
        0.0
    };

    DomainGroup {
        member_categories: members,
        category_names,
        centroid,
        angular_spread,
        cohesion: 1.0 / (1.0 + angular_spread),
        total_items,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{PcaProjection, Projection};
    use crate::types::{Embedding, RadialStrategy};

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    fn build_layer() -> CategoryLayer {
        let categories = vec![
            "science".into(),
            "science".into(),
            "science".into(),
            "cooking".into(),
            "cooking".into(),
            "cooking".into(),
            "music".into(),
            "music".into(),
            "music".into(),
        ];
        let embeddings = vec![
            emb(&[1.0, 0.1, 0.0, 0.05, 0.02]),
            emb(&[0.9, 0.15, 0.05, 0.03, 0.01]),
            emb(&[0.95, 0.05, 0.1, 0.04, 0.03]),
            emb(&[0.1, 1.0, 0.0, 0.02, 0.05]),
            emb(&[0.15, 0.9, 0.05, 0.03, 0.04]),
            emb(&[0.05, 0.95, 0.1, 0.01, 0.06]),
            emb(&[0.0, 0.1, 1.0, 0.05, 0.02]),
            emb(&[0.05, 0.15, 0.9, 0.03, 0.01]),
            emb(&[0.1, 0.05, 0.95, 0.04, 0.03]),
        ];
        let pca = PcaProjection::fit(&embeddings, RadialStrategy::Fixed(1.0));
        let projected: Vec<SphericalPoint> = embeddings.iter().map(|e| pca.project(e)).collect();
        let evr = pca.explained_variance_ratio();
        CategoryLayer::build(&categories, &embeddings, &projected, &pca, evr)
    }

    #[test]
    fn target_clamped_to_category_count() {
        let layer = build_layer();
        let groups = detect_domain_groups(&layer, 99);
        assert_eq!(groups.len(), layer.num_categories());
    }

    #[test]
    fn target_one_merges_everything() {
        let layer = build_layer();
        let groups = detect_domain_groups(&layer, 1);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].member_categories.len(), layer.num_categories());
    }

    #[test]
    fn total_items_preserved() {
        let layer = build_layer();
        let groups = detect_domain_groups(&layer, 2);
        let total_in_groups: usize = groups.iter().map(|g| g.total_items).sum();
        let total_in_layer: usize = layer.summaries.iter().map(|s| s.member_count).sum();
        assert_eq!(total_in_groups, total_in_layer);
    }

    #[test]
    fn every_category_assigned_once() {
        let layer = build_layer();
        let groups = detect_domain_groups(&layer, 2);
        let mut all: Vec<usize> = groups
            .iter()
            .flat_map(|g| g.member_categories.iter().copied())
            .collect();
        all.sort();
        let before_dedup = all.len();
        all.dedup();
        assert_eq!(before_dedup, all.len(), "category assigned to multiple groups");
        assert_eq!(all.len(), layer.num_categories());
    }

    #[test]
    fn cohesion_in_range() {
        let layer = build_layer();
        for g in detect_domain_groups(&layer, 2) {
            assert!(g.cohesion > 0.0 && g.cohesion <= 1.0);
            assert!(g.angular_spread >= 0.0);
        }
    }
}
