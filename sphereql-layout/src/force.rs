use std::f64::consts::PI;

use sphereql_core::{
    CartesianPoint, SphericalPoint, angular_distance, cartesian_to_spherical,
    spherical_to_cartesian,
};

use crate::traits::{DimensionMapper, LayoutStrategy};
use crate::types::{LayoutEntry, LayoutQuality, LayoutResult};

const EPSILON: f64 = 1e-6;
const STEP_SIZE_FACTOR: f64 = 0.1;
const OVERLAP_THRESHOLD: f64 = 0.01;

pub struct ForceDirectedLayout {
    pub iterations: usize,
    pub repulsion_strength: f64,
    pub attraction_strength: f64,
    pub cooling_rate: f64,
    pub radius: f64,
}

impl ForceDirectedLayout {
    pub fn new() -> Self {
        Self {
            iterations: 100,
            repulsion_strength: 1.0,
            attraction_strength: 0.1,
            cooling_rate: 0.95,
            radius: 1.0,
        }
    }

    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    pub fn with_repulsion(mut self, f: f64) -> Self {
        self.repulsion_strength = f;
        self
    }

    pub fn with_attraction(mut self, f: f64) -> Self {
        self.attraction_strength = f;
        self
    }

    pub fn with_cooling(mut self, f: f64) -> Self {
        self.cooling_rate = f;
        self
    }

    pub fn with_radius(mut self, r: f64) -> Self {
        self.radius = r;
        self
    }

    fn project_to_unit_sphere(p: &SphericalPoint) -> CartesianPoint {
        let unit = SphericalPoint::new_unchecked(1.0, p.theta, p.phi);
        spherical_to_cartesian(&unit)
    }

    const MAX_QUALITY_N: usize = 5000;

    fn compute_quality(positions: &[SphericalPoint], n: usize) -> LayoutQuality {
        if n <= 1 {
            return LayoutQuality {
                dispersion_score: 1.0,
                overlap_score: 0.0,
                silhouette_score: 0.0,
            };
        }

        let (positions, n) = if n > Self::MAX_QUALITY_N {
            let step = n / Self::MAX_QUALITY_N;
            let sampled: Vec<_> = positions
                .iter()
                .step_by(step)
                .take(Self::MAX_QUALITY_N)
                .copied()
                .collect();
            let len = sampled.len();
            (sampled, len)
        } else {
            (positions.to_vec(), n)
        };

        let ideal_spacing = (4.0 * PI / n as f64).sqrt();
        let total_pairs = (n * (n - 1) / 2) as u64;

        // Parallel pair-scan: each `i`-worker tracks its own (min, count)
        // pair and reduces at the end. f64::min takes NaN as "the other",
        // so we route through `.total_cmp` for predictable results.
        use rayon::prelude::*;
        const SERIAL_THRESHOLD: usize = 128;
        let per_i = |i: usize| -> (f64, u64) {
            let mut min_local = f64::MAX;
            let mut count_local = 0u64;
            for j in (i + 1)..n {
                let d = angular_distance(&positions[i], &positions[j]);
                if d < min_local {
                    min_local = d;
                }
                if d < OVERLAP_THRESHOLD {
                    count_local += 1;
                }
            }
            (min_local, count_local)
        };
        let (min_dist, overlap_count) = if n < SERIAL_THRESHOLD {
            (0..n).map(per_i).fold(
                (f64::MAX, 0u64),
                |(ma, ca), (mb, cb)| (if mb < ma { mb } else { ma }, ca + cb),
            )
        } else {
            (0..n).into_par_iter().map(per_i).reduce(
                || (f64::MAX, 0u64),
                |(ma, ca), (mb, cb)| (if mb < ma { mb } else { ma }, ca + cb),
            )
        };

        let dispersion = (min_dist / ideal_spacing).clamp(0.0, 1.0);
        let overlap = overlap_count as f64 / total_pairs as f64;

        LayoutQuality {
            dispersion_score: dispersion,
            overlap_score: overlap,
            silhouette_score: 0.0,
        }
    }
}

impl Default for ForceDirectedLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> LayoutStrategy<T> for ForceDirectedLayout {
    fn layout(&self, items: &[T], mapper: &dyn DimensionMapper<Item = T>) -> LayoutResult<T> {
        let n = items.len();

        if n == 0 {
            return LayoutResult {
                entries: Vec::new(),
                quality: LayoutQuality::default(),
            };
        }

        let original_positions: Vec<SphericalPoint> =
            items.iter().map(|item| mapper.map(item)).collect();

        let original_cartesian: Vec<CartesianPoint> = original_positions
            .iter()
            .map(Self::project_to_unit_sphere)
            .collect();

        let mut positions: Vec<CartesianPoint> = original_cartesian.clone();

        let mut temperature = 1.0;

        for _ in 0..self.iterations {
            let mut forces: Vec<CartesianPoint> = vec![CartesianPoint::new(0.0, 0.0, 0.0); n];

            for i in 0..n {
                let pi = positions[i];

                // Repulsion from every other point
                for (j, &pj) in positions.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let sp_i = cartesian_to_spherical(&pi);
                    let sp_j = cartesian_to_spherical(&pj);
                    let dist = angular_distance(&sp_i, &sp_j);

                    let dx = pi.x - pj.x;
                    let dy = pi.y - pj.y;
                    let dz = pi.z - pj.z;

                    let cart_dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if cart_dist < EPSILON {
                        continue;
                    }

                    let magnitude = self.repulsion_strength / (dist * dist + EPSILON);

                    forces[i] = CartesianPoint::new(
                        forces[i].x + magnitude * dx / cart_dist,
                        forces[i].y + magnitude * dy / cart_dist,
                        forces[i].z + magnitude * dz / cart_dist,
                    );
                }

                // Attraction toward mapper's original position
                let oi = original_cartesian[i];
                let sp_i = cartesian_to_spherical(&pi);
                let sp_oi = cartesian_to_spherical(&oi);
                let dist_to_original = angular_distance(&sp_i, &sp_oi);

                let dx = oi.x - pi.x;
                let dy = oi.y - pi.y;
                let dz = oi.z - pi.z;
                let cart_dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if cart_dist > EPSILON {
                    let magnitude = self.attraction_strength * dist_to_original;
                    forces[i] = CartesianPoint::new(
                        forces[i].x + magnitude * dx / cart_dist,
                        forces[i].y + magnitude * dy / cart_dist,
                        forces[i].z + magnitude * dz / cart_dist,
                    );
                }
            }

            // Apply forces: project onto tangent plane, then normalize back to sphere
            let step_size = temperature * STEP_SIZE_FACTOR;
            for i in 0..n {
                let p = positions[i];
                let f = forces[i];

                // Project force onto tangent plane at p: f_tangent = f - dot(f, p) * p
                let dot = f.x * p.x + f.y * p.y + f.z * p.z;
                let ft = CartesianPoint::new(f.x - dot * p.x, f.y - dot * p.y, f.z - dot * p.z);

                let new_pos = CartesianPoint::new(
                    p.x + step_size * ft.x,
                    p.y + step_size * ft.y,
                    p.z + step_size * ft.z,
                );

                positions[i] = new_pos.normalize();
            }

            temperature *= self.cooling_rate;
        }

        let final_positions: Vec<SphericalPoint> = positions
            .iter()
            .map(|c| {
                let sp = cartesian_to_spherical(c);
                SphericalPoint::new_unchecked(self.radius, sp.theta, sp.phi)
            })
            .collect();

        let entries: Vec<LayoutEntry<T>> = items
            .iter()
            .zip(final_positions.iter())
            .map(|(item, pos)| LayoutEntry {
                item: item.clone(),
                position: *pos,
            })
            .collect();

        let quality = Self::compute_quality(&final_positions, n);

        LayoutResult { entries, quality }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    struct FixedMapper {
        positions: Vec<SphericalPoint>,
    }

    impl DimensionMapper for FixedMapper {
        type Item = usize;
        fn map(&self, item: &usize) -> SphericalPoint {
            self.positions[*item]
        }
    }

    struct OriginMapper;

    impl DimensionMapper for OriginMapper {
        type Item = usize;
        fn map(&self, _item: &usize) -> SphericalPoint {
            SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2)
        }
    }

    #[test]
    fn empty_items_returns_empty() {
        let layout = ForceDirectedLayout::new();
        let items: Vec<usize> = vec![];
        let result = layout.layout(&items, &OriginMapper);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn single_item_stays_near_mapper_position() {
        let target = SphericalPoint::new_unchecked(1.0, 1.0, 1.0);
        let mapper = FixedMapper {
            positions: vec![target],
        };
        let layout = ForceDirectedLayout::new().with_iterations(50);
        let result = layout.layout(&[0usize], &mapper);

        assert_eq!(result.entries.len(), 1);
        let pos = &result.entries[0].position;
        let dist = angular_distance(pos, &target);
        assert!(
            dist < 0.1,
            "single item should stay near mapper position, but angular distance was {dist}"
        );
    }

    #[test]
    fn two_items_pushed_apart_by_repulsion() {
        let mapper = FixedMapper {
            positions: vec![
                SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.1, FRAC_PI_2),
            ],
        };

        let layout = ForceDirectedLayout::new()
            .with_iterations(200)
            .with_repulsion(2.0)
            .with_attraction(0.01);

        let result = layout.layout(&[0usize, 1], &mapper);
        assert_eq!(result.entries.len(), 2);

        let dist = angular_distance(&result.entries[0].position, &result.entries[1].position);

        assert!(
            dist > PI * 0.5,
            "two items should be pushed far apart by repulsion, but angular distance was {dist}"
        );
    }

    #[test]
    fn all_positions_have_correct_radius() {
        let r = 3.5;
        let mapper = FixedMapper {
            positions: vec![
                SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 1.0, 1.0),
                SphericalPoint::new_unchecked(1.0, 2.0, 0.5),
                SphericalPoint::new_unchecked(1.0, 3.0, 2.5),
            ],
        };
        let layout = ForceDirectedLayout::new().with_radius(r);
        let result = layout.layout(&[0usize, 1, 2, 3], &mapper);

        for (i, entry) in result.entries.iter().enumerate() {
            assert!(
                (entry.position.r - r).abs() < 1e-12,
                "entry {i} has radius {}, expected {r}",
                entry.position.r
            );
        }
    }

    #[test]
    fn more_iterations_produce_better_or_equal_dispersion() {
        let mapper = FixedMapper {
            positions: vec![
                SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.1, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.2, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.3, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.4, FRAC_PI_2),
            ],
        };

        let few = ForceDirectedLayout::new()
            .with_iterations(5)
            .with_repulsion(1.0)
            .with_attraction(0.01);
        let many = ForceDirectedLayout::new()
            .with_iterations(200)
            .with_repulsion(1.0)
            .with_attraction(0.01);

        let items: Vec<usize> = (0..5).collect();
        let result_few = few.layout(&items, &mapper);
        let result_many = many.layout(&items, &mapper);

        assert!(
            result_many.quality.dispersion_score >= result_few.quality.dispersion_score - 1e-6,
            "more iterations ({}) should produce >= dispersion than fewer ({})",
            result_many.quality.dispersion_score,
            result_few.quality.dispersion_score,
        );
    }

    #[test]
    fn cooling_reduces_movement_over_time() {
        let mapper = FixedMapper {
            positions: vec![
                SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.1, FRAC_PI_2),
                SphericalPoint::new_unchecked(1.0, 0.2, FRAC_PI_2),
            ],
        };

        let aggressive_cooling = ForceDirectedLayout::new()
            .with_iterations(100)
            .with_cooling(0.5);

        let no_cooling = ForceDirectedLayout::new()
            .with_iterations(100)
            .with_cooling(1.0);

        let items: Vec<usize> = (0..3).collect();
        let result_cooled = aggressive_cooling.layout(&items, &mapper);
        let result_uncooled = no_cooling.layout(&items, &mapper);

        for entry in &result_cooled.entries {
            assert!(!entry.position.theta.is_nan());
            assert!(!entry.position.phi.is_nan());
        }

        let mut total_dist_cooled = 0.0;
        let mut total_dist_uncooled = 0.0;
        for (i, orig) in mapper.positions.iter().enumerate() {
            total_dist_cooled += angular_distance(&result_cooled.entries[i].position, orig);
            total_dist_uncooled += angular_distance(&result_uncooled.entries[i].position, orig);
        }

        assert!(
            total_dist_uncooled >= total_dist_cooled - 1e-6,
            "uncooled ({total_dist_uncooled}) should move points at least as far as \
             aggressively cooled ({total_dist_cooled})"
        );
    }

    #[test]
    fn default_builder_matches_new() {
        let from_new = ForceDirectedLayout::new();
        let from_default = ForceDirectedLayout::default();
        assert_eq!(from_new.iterations, from_default.iterations);
        assert!((from_new.repulsion_strength - from_default.repulsion_strength).abs() < 1e-15);
        assert!((from_new.attraction_strength - from_default.attraction_strength).abs() < 1e-15);
        assert!((from_new.cooling_rate - from_default.cooling_rate).abs() < 1e-15);
        assert!((from_new.radius - from_default.radius).abs() < 1e-15);
    }
}
