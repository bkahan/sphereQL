use std::f64::consts::{PI, TAU};

use sphereql_core::{SphericalPoint, angular_distance};

use crate::traits::{DimensionMapper, LayoutStrategy};
use crate::types::{LayoutEntry, LayoutQuality, LayoutResult};

const GOLDEN_ANGLE: f64 = PI * (3.0 - 2.236_067_977_499_79); // PI * (3 - sqrt(5)) ≈ 2.3999 rad

pub struct UniformLayout {
    radius: f64,
}

impl UniformLayout {
    pub fn new() -> Self {
        Self { radius: 1.0 }
    }

    pub fn with_radius(radius: f64) -> Self {
        Self { radius }
    }

    fn fibonacci_point(&self, i: usize, n: usize) -> SphericalPoint {
        let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
        let theta = (GOLDEN_ANGLE * i as f64) % TAU;
        // Ensure theta is non-negative after modulo
        let theta = if theta < 0.0 { theta + TAU } else { theta };
        SphericalPoint::new_unchecked(self.radius, theta, phi)
    }

    fn compute_quality(entries: &[LayoutEntry<impl Clone>], n: usize) -> LayoutQuality {
        if n <= 1 {
            return LayoutQuality {
                dispersion_score: 1.0,
                overlap_score: 0.0,
                silhouette_score: 0.0,
            };
        }

        let ideal_spacing = (4.0 * PI / n as f64).sqrt();

        let mut min_dist = f64::MAX;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = angular_distance(&entries[i].position, &entries[j].position);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }

        let dispersion = (min_dist / ideal_spacing).clamp(0.0, 1.0);

        LayoutQuality {
            dispersion_score: dispersion,
            overlap_score: 0.0,
            silhouette_score: 0.0,
        }
    }
}

impl Default for UniformLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> LayoutStrategy<T> for UniformLayout {
    fn layout(&self, items: &[T], mapper: &dyn DimensionMapper<Item = T>) -> LayoutResult<T> {
        let n = items.len();

        if n == 0 {
            return LayoutResult {
                entries: Vec::new(),
                quality: LayoutQuality::default(),
            };
        }

        // UniformLayout places items on a Fibonacci lattice; the mapper's
        // semantic positions are intentionally ignored.
        let _ = mapper;

        let entries: Vec<LayoutEntry<T>> = items
            .iter()
            .enumerate()
            .map(|(i, item)| LayoutEntry {
                item: item.clone(),
                position: self.fibonacci_point(i, n),
            })
            .collect();

        let quality = Self::compute_quality(&entries, n);

        LayoutResult { entries, quality }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct IdentityMapper;

    impl DimensionMapper for IdentityMapper {
        type Item = u32;
        fn map(&self, _item: &u32) -> SphericalPoint {
            SphericalPoint::origin()
        }
    }

    #[test]
    fn empty_items_returns_empty() {
        let layout = UniformLayout::new();
        let items: Vec<u32> = vec![];
        let result = layout.layout(&items, &IdentityMapper);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn single_item_correct_position() {
        let layout = UniformLayout::new();
        let result = layout.layout(&[42u32], &IdentityMapper);
        assert_eq!(result.entries.len(), 1);
        let pos = &result.entries[0].position;
        // For n=1, i=0: phi = acos(1.0 - 2.0*0.5/1.0) = acos(0.0) = PI/2
        assert!((pos.phi - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((pos.r - 1.0).abs() < 1e-10);
        // theta = (golden_angle * 0) % TAU = 0.0
        assert!(pos.theta.abs() < 1e-10);
    }

    #[test]
    fn n_items_all_distinct_positions() {
        let layout = UniformLayout::new();
        let items: Vec<u32> = (0..20).collect();
        let result = layout.layout(&items, &IdentityMapper);
        assert_eq!(result.entries.len(), 20);

        // Verify all positions are distinct by checking no two points are too close
        for i in 0..20 {
            for j in (i + 1)..20 {
                let d = angular_distance(&result.entries[i].position, &result.entries[j].position);
                assert!(d > 1e-10, "points {i} and {j} are not distinct");
            }
        }
    }

    #[test]
    fn all_positions_have_configured_radius() {
        let r = 2.5;
        let layout = UniformLayout::with_radius(r);
        let items: Vec<u32> = (0..50).collect();
        let result = layout.layout(&items, &IdentityMapper);
        for (i, entry) in result.entries.iter().enumerate() {
            assert!(
                (entry.position.r - r).abs() < 1e-12,
                "entry {i} has radius {}, expected {r}",
                entry.position.r
            );
        }
    }

    #[test]
    fn dispersion_score_reasonable_for_100_items() {
        let layout = UniformLayout::new();
        let items: Vec<u32> = (0..100).collect();
        let result = layout.layout(&items, &IdentityMapper);
        assert!(
            result.quality.dispersion_score > 0.5,
            "dispersion score {} should be > 0.5 for 100 items",
            result.quality.dispersion_score,
        );
    }

    #[test]
    fn fibonacci_spiral_good_coverage() {
        let layout = UniformLayout::new();
        let items: Vec<u32> = (0..200).collect();
        let result = layout.layout(&items, &IdentityMapper);

        let ideal_spacing = (4.0 * PI / 200.0).sqrt();
        let mut min_dist = f64::MAX;
        for i in 0..result.entries.len() {
            for j in (i + 1)..result.entries.len() {
                let d = angular_distance(&result.entries[i].position, &result.entries[j].position);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }

        // For a Fibonacci spiral, min distance should be a reasonable fraction of ideal
        assert!(
            min_dist > ideal_spacing * 0.4,
            "min angular distance {min_dist} is too small relative to ideal {ideal_spacing}",
        );
    }

    #[test]
    fn overlap_and_silhouette_are_zero() {
        let layout = UniformLayout::new();
        let items: Vec<u32> = (0..10).collect();
        let result = layout.layout(&items, &IdentityMapper);
        assert!((result.quality.overlap_score).abs() < 1e-12);
        assert!((result.quality.silhouette_score).abs() < 1e-12);
    }

    #[test]
    fn default_radius_is_one() {
        let layout = UniformLayout::default();
        let result = layout.layout(&[1u32], &IdentityMapper);
        assert!((result.entries[0].position.r - 1.0).abs() < 1e-12);
    }
}
