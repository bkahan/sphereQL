use std::f64::consts::PI;

use sphereql_core::{
    CartesianPoint, SphericalPoint, angular_distance, cartesian_to_spherical,
    spherical_to_cartesian,
};

use crate::traits::{DimensionMapper, LayoutStrategy};
use crate::types::{LayoutEntry, LayoutQuality, LayoutResult};

const MAX_KMEANS_ITERATIONS: usize = 50;
const OVERLAP_THRESHOLD: f64 = 0.01;

pub struct ClusteredLayout {
    pub num_clusters: usize,
    pub radius: f64,
    pub intra_cluster_spread: f64,
}

impl Default for ClusteredLayout {
    fn default() -> Self {
        Self {
            num_clusters: 4,
            radius: 1.0,
            intra_cluster_spread: 0.3,
        }
    }
}

impl ClusteredLayout {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_clusters(mut self, n: usize) -> Self {
        self.num_clusters = n;
        self
    }

    pub fn with_radius(mut self, r: f64) -> Self {
        self.radius = r;
        self
    }

    pub fn with_spread(mut self, s: f64) -> Self {
        self.intra_cluster_spread = s;
        self
    }
}

fn evenly_spaced_centers(k: usize) -> Vec<CartesianPoint> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..k)
        .map(|i| {
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / k as f64).clamp(-1.0, 1.0).acos();
            let theta = (2.0 * PI * (i as f64) / golden_ratio).rem_euclid(2.0 * PI);
            let sp = SphericalPoint::new_unchecked(1.0, theta, phi);
            spherical_to_cartesian(&sp)
        })
        .collect()
}

fn normalized_mean(points: &[CartesianPoint]) -> CartesianPoint {
    if points.is_empty() {
        return CartesianPoint::new(0.0, 0.0, 1.0);
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for p in points {
        sx += p.x;
        sy += p.y;
        sz += p.z;
    }
    let mean = CartesianPoint::new(sx, sy, sz);
    let n = mean.normalize();
    if n.magnitude() == 0.0 {
        points[0].normalize()
    } else {
        n
    }
}

struct KMeansResult {
    assignments: Vec<usize>,
    centers: Vec<CartesianPoint>,
}

fn kmeans_spherical(
    mapped_cartesian: &[CartesianPoint],
    mapped_spherical: &[SphericalPoint],
    k: usize,
) -> KMeansResult {
    let n = mapped_cartesian.len();

    let mut centers: Vec<CartesianPoint> = if n >= k {
        mapped_cartesian[..k].iter().map(|c| c.normalize()).collect()
    } else {
        evenly_spaced_centers(k)
    };

    let mut assignments = vec![0usize; n];

    for _ in 0..MAX_KMEANS_ITERATIONS {
        let mut changed = false;

        for (i, sp) in mapped_spherical.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f64::MAX;
            for (j, center) in centers.iter().enumerate() {
                let center_sp = cartesian_to_spherical(center);
                let d = angular_distance(sp, &center_sp);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut cluster_points: Vec<Vec<CartesianPoint>> = vec![vec![]; k];
        for (i, &a) in assignments.iter().enumerate() {
            cluster_points[a].push(mapped_cartesian[i]);
        }

        for (j, cp) in cluster_points.iter().enumerate() {
            if cp.is_empty() {
                let mut farthest_idx = 0;
                let mut farthest_dist = 0.0_f64;
                for (i, sp) in mapped_spherical.iter().enumerate() {
                    let center_sp = cartesian_to_spherical(&centers[assignments[i]]);
                    let d = angular_distance(sp, &center_sp);
                    if d > farthest_dist {
                        farthest_dist = d;
                        farthest_idx = i;
                    }
                }
                centers[j] = mapped_cartesian[farthest_idx].normalize();
            } else {
                centers[j] = normalized_mean(cp);
            }
        }
    }

    KMeansResult {
        assignments,
        centers,
    }
}

fn fibonacci_sub_spiral(
    center: &SphericalPoint,
    count: usize,
    spread: f64,
    radius: f64,
) -> Vec<SphericalPoint> {
    if count == 0 {
        return vec![];
    }
    if count == 1 {
        return vec![SphericalPoint::new_unchecked(radius, center.theta, center.phi)];
    }

    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let center_cart = spherical_to_cartesian(&SphericalPoint::new_unchecked(
        1.0,
        center.theta,
        center.phi,
    ));

    let (tangent_u, tangent_v) = local_frame(&center_cart);

    (0..count)
        .map(|i| {
            let frac = i as f64 / count as f64;
            let angular_r = spread * frac.sqrt();
            let angle = golden_angle * i as f64;

            let offset_u = angular_r * angle.cos();
            let offset_v = angular_r * angle.sin();

            let displaced = CartesianPoint::new(
                center_cart.x + offset_u * tangent_u.x + offset_v * tangent_v.x,
                center_cart.y + offset_u * tangent_u.y + offset_v * tangent_v.y,
                center_cart.z + offset_u * tangent_u.z + offset_v * tangent_v.z,
            )
            .normalize();

            let sp = cartesian_to_spherical(&displaced);
            SphericalPoint::new_unchecked(radius, sp.theta, sp.phi)
        })
        .collect()
}

fn local_frame(center: &CartesianPoint) -> (CartesianPoint, CartesianPoint) {
    let up = if center.z.abs() < 0.9 {
        CartesianPoint::new(0.0, 0.0, 1.0)
    } else {
        CartesianPoint::new(1.0, 0.0, 0.0)
    };

    // u = normalize(up x center)
    let ux = up.y * center.z - up.z * center.y;
    let uy = up.z * center.x - up.x * center.z;
    let uz = up.x * center.y - up.y * center.x;
    let u = CartesianPoint::new(ux, uy, uz).normalize();

    // v = center x u
    let vx = center.y * u.z - center.z * u.y;
    let vy = center.z * u.x - center.x * u.z;
    let vz = center.x * u.y - center.y * u.x;
    let v = CartesianPoint::new(vx, vy, vz).normalize();

    (u, v)
}

fn compute_quality(
    positions: &[SphericalPoint],
    assignments: &[usize],
    num_clusters: usize,
) -> LayoutQuality {
    let n = positions.len();

    if n <= 1 {
        return LayoutQuality {
            dispersion_score: if n == 0 { 0.0 } else { 1.0 },
            overlap_score: 0.0,
            silhouette_score: 0.0,
        };
    }

    // Dispersion: average inter-cluster center distance / PI
    let mut cluster_point_sets: Vec<Vec<CartesianPoint>> = vec![vec![]; num_clusters];
    for (i, &a) in assignments.iter().enumerate() {
        cluster_point_sets[a].push(spherical_to_cartesian(&positions[i]));
    }
    let active_centers: Vec<SphericalPoint> = cluster_point_sets
        .iter()
        .filter(|cp| !cp.is_empty())
        .map(|cp| cartesian_to_spherical(&normalized_mean(cp)))
        .collect();

    let dispersion_score = if active_centers.len() >= 2 {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..active_centers.len() {
            for j in (i + 1)..active_centers.len() {
                sum += angular_distance(&active_centers[i], &active_centers[j]);
                count += 1;
            }
        }
        (sum / count as f64 / PI).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Overlap: fraction of pairs within threshold
    let mut overlap_count = 0u64;
    let total_pairs = (n * (n - 1)) / 2;
    for i in 0..n {
        for j in (i + 1)..n {
            if angular_distance(&positions[i], &positions[j]) < OVERLAP_THRESHOLD {
                overlap_count += 1;
            }
        }
    }
    let overlap_score = if total_pairs > 0 {
        overlap_count as f64 / total_pairs as f64
    } else {
        0.0
    };

    // Silhouette coefficient
    let silhouette_score = if num_clusters <= 1 || active_centers.len() <= 1 {
        0.0
    } else {
        let mut sil_sum = 0.0;
        for i in 0..n {
            let ci = assignments[i];

            // a(i) = mean distance to same-cluster members
            let mut a_sum = 0.0;
            let mut a_count = 0;
            for j in 0..n {
                if j != i && assignments[j] == ci {
                    a_sum += angular_distance(&positions[i], &positions[j]);
                    a_count += 1;
                }
            }
            let a = if a_count > 0 {
                a_sum / a_count as f64
            } else {
                0.0
            };

            // b(i) = min over other clusters of mean distance to that cluster
            let mut b = f64::MAX;
            for k in 0..num_clusters {
                if k == ci {
                    continue;
                }
                let mut b_sum = 0.0;
                let mut b_count = 0;
                for j in 0..n {
                    if assignments[j] == k {
                        b_sum += angular_distance(&positions[i], &positions[j]);
                        b_count += 1;
                    }
                }
                if b_count > 0 {
                    let mean_dist = b_sum / b_count as f64;
                    if mean_dist < b {
                        b = mean_dist;
                    }
                }
            }
            if b == f64::MAX {
                b = 0.0;
            }

            let denom = a.max(b);
            let s = if denom > 0.0 { (b - a) / denom } else { 0.0 };
            sil_sum += s;
        }
        sil_sum / n as f64
    };

    LayoutQuality {
        dispersion_score,
        overlap_score,
        silhouette_score,
    }
}

impl<T: Clone + Send + Sync> LayoutStrategy<T> for ClusteredLayout {
    fn layout(&self, items: &[T], mapper: &dyn DimensionMapper<Item = T>) -> LayoutResult<T> {
        if items.is_empty() {
            return LayoutResult {
                entries: vec![],
                quality: LayoutQuality::default(),
            };
        }

        let mapped: Vec<SphericalPoint> = items.iter().map(|item| mapper.map(item)).collect();
        let mapped_cart: Vec<CartesianPoint> =
            mapped.iter().map(spherical_to_cartesian).collect();

        let k = self.num_clusters.min(items.len()).max(1);
        let km = kmeans_spherical(&mapped_cart, &mapped, k);

        let mut cluster_items: Vec<Vec<usize>> = vec![vec![]; k];
        for (i, &a) in km.assignments.iter().enumerate() {
            cluster_items[a].push(i);
        }

        let mut entries: Vec<(usize, LayoutEntry<T>)> = Vec::with_capacity(items.len());
        let mut final_positions: Vec<(usize, SphericalPoint)> = Vec::with_capacity(items.len());
        let mut final_assignments = vec![0usize; items.len()];

        for (cluster_idx, member_indices) in cluster_items.iter().enumerate() {
            let center_sp = cartesian_to_spherical(&km.centers[cluster_idx]);
            let sub_positions = fibonacci_sub_spiral(
                &center_sp,
                member_indices.len(),
                self.intra_cluster_spread,
                self.radius,
            );

            for (sub_idx, &item_idx) in member_indices.iter().enumerate() {
                let pos = sub_positions[sub_idx];
                entries.push((
                    item_idx,
                    LayoutEntry {
                        item: items[item_idx].clone(),
                        position: pos,
                    },
                ));
                final_positions.push((item_idx, pos));
                final_assignments[item_idx] = cluster_idx;
            }
        }

        entries.sort_by_key(|(idx, _)| *idx);
        let entries: Vec<LayoutEntry<T>> = entries.into_iter().map(|(_, e)| e).collect();

        final_positions.sort_by_key(|(idx, _)| *idx);
        let positions: Vec<SphericalPoint> =
            final_positions.into_iter().map(|(_, p)| p).collect();

        let quality = compute_quality(&positions, &final_assignments, k);

        LayoutResult { entries, quality }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedMapper {
        positions: Vec<SphericalPoint>,
    }

    impl DimensionMapper for FixedMapper {
        type Item = usize;
        fn map(&self, item: &usize) -> SphericalPoint {
            self.positions[*item]
        }
    }

    #[test]
    fn empty_items_returns_empty_result() {
        let layout = ClusteredLayout::new();
        let mapper = FixedMapper { positions: vec![] };
        let result = layout.layout(&[], &mapper);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn single_item_gets_placed() {
        let layout = ClusteredLayout::new().with_clusters(1);
        let mapper = FixedMapper {
            positions: vec![SphericalPoint::new_unchecked(1.0, 0.5, 1.0)],
        };
        let result = layout.layout(&[0usize], &mapper);
        assert_eq!(result.entries.len(), 1);
        assert!((result.entries[0].position.r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correct_number_of_entries() {
        let layout = ClusteredLayout::new().with_clusters(3);
        let positions: Vec<SphericalPoint> = (0..20)
            .map(|i| {
                let theta = (i as f64 * 0.3) % (2.0 * PI);
                SphericalPoint::new_unchecked(1.0, theta, 1.0)
            })
            .collect();
        let mapper = FixedMapper { positions };
        let items: Vec<usize> = (0..20).collect();
        let result = layout.layout(&items, &mapper);
        assert_eq!(result.entries.len(), 20);
    }

    #[test]
    fn items_in_same_cluster_are_angularly_close() {
        let mut positions = Vec::new();
        for i in 0..5 {
            positions.push(SphericalPoint::new_unchecked(1.0, 0.01 * i as f64, 0.1));
        }
        for i in 0..5 {
            positions.push(SphericalPoint::new_unchecked(
                1.0,
                0.01 * i as f64,
                PI - 0.1,
            ));
        }
        let mapper = FixedMapper { positions };
        let items: Vec<usize> = (0..10).collect();
        let layout = ClusteredLayout::new().with_clusters(2).with_spread(0.2);
        let result = layout.layout(&items, &mapper);

        let group_a: Vec<&SphericalPoint> =
            result.entries[..5].iter().map(|e| &e.position).collect();
        for i in 0..group_a.len() {
            for j in (i + 1)..group_a.len() {
                let d = angular_distance(group_a[i], group_a[j]);
                assert!(d < 1.0, "Intra-cluster distance too large: {d}");
            }
        }
    }

    #[test]
    fn different_clusters_are_angularly_separated() {
        let mut positions = Vec::new();
        for i in 0..5 {
            positions.push(SphericalPoint::new_unchecked(
                1.0,
                0.01 * i as f64,
                PI / 2.0,
            ));
        }
        for i in 0..5 {
            positions.push(SphericalPoint::new_unchecked(
                1.0,
                PI + 0.01 * i as f64,
                PI / 2.0,
            ));
        }
        let mapper = FixedMapper { positions };
        let items: Vec<usize> = (0..10).collect();
        let layout = ClusteredLayout::new().with_clusters(2).with_spread(0.2);
        let result = layout.layout(&items, &mapper);

        let p_a = &result.entries[0].position;
        let p_b = &result.entries[5].position;
        let d = angular_distance(p_a, p_b);
        assert!(d > 1.0, "Inter-cluster distance too small: {d}");
    }

    #[test]
    fn silhouette_positive_for_well_separated_data() {
        let mut positions = Vec::new();
        for i in 0..10 {
            positions.push(SphericalPoint::new_unchecked(1.0, 0.01 * i as f64, 0.2));
        }
        for i in 0..10 {
            positions.push(SphericalPoint::new_unchecked(
                1.0,
                PI + 0.01 * i as f64,
                PI - 0.2,
            ));
        }
        let mapper = FixedMapper { positions };
        let items: Vec<usize> = (0..20).collect();
        let layout = ClusteredLayout::new().with_clusters(2).with_spread(0.15);
        let result = layout.layout(&items, &mapper);
        assert!(
            result.quality.silhouette_score > 0.0,
            "Silhouette should be positive for well-separated clusters, got {}",
            result.quality.silhouette_score
        );
    }

    #[test]
    fn builder_methods_apply() {
        let layout = ClusteredLayout::new()
            .with_clusters(8)
            .with_radius(2.5)
            .with_spread(0.5);
        assert_eq!(layout.num_clusters, 8);
        assert!((layout.radius - 2.5).abs() < 1e-12);
        assert!((layout.intra_cluster_spread - 0.5).abs() < 1e-12);
    }

    #[test]
    fn output_radius_matches_configured() {
        let layout = ClusteredLayout::new().with_radius(3.0).with_clusters(2);
        let positions = vec![
            SphericalPoint::new_unchecked(1.0, 0.0, 0.5),
            SphericalPoint::new_unchecked(1.0, PI, 2.0),
        ];
        let mapper = FixedMapper { positions };
        let result = layout.layout(&[0usize, 1], &mapper);
        for entry in &result.entries {
            assert!(
                (entry.position.r - 3.0).abs() < 1e-12,
                "Expected radius 3.0, got {}",
                entry.position.r
            );
        }
    }
}
