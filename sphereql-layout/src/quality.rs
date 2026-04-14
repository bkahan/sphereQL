use std::f64::consts::PI;

use sphereql_core::{SphericalPoint, angular_distance};

use crate::types::LayoutQuality;

const MAX_QUALITY_N: usize = 5000;

fn sample_positions(positions: &[SphericalPoint]) -> Vec<SphericalPoint> {
    if positions.len() <= MAX_QUALITY_N {
        return positions.to_vec();
    }
    let step = positions.len() / MAX_QUALITY_N;
    positions
        .iter()
        .step_by(step)
        .take(MAX_QUALITY_N)
        .copied()
        .collect()
}

pub fn compute_dispersion(positions: &[SphericalPoint]) -> f64 {
    let positions = sample_positions(positions);
    let n = positions.len();
    if n <= 1 {
        return 1.0;
    }

    let ideal_spacing = (4.0 * PI / n as f64).sqrt();

    let mut min_dist = f64::MAX;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = angular_distance(&positions[i], &positions[j]);
            if d < min_dist {
                min_dist = d;
            }
        }
    }

    (min_dist / ideal_spacing).clamp(0.0, 1.0)
}

pub fn compute_overlap(positions: &[SphericalPoint], threshold: f64) -> f64 {
    let positions = sample_positions(positions);
    let n = positions.len();
    if n < 2 {
        return 0.0;
    }

    let total_pairs = n * (n - 1) / 2;
    let mut overlapping = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            if angular_distance(&positions[i], &positions[j]) < threshold {
                overlapping += 1;
            }
        }
    }

    overlapping as f64 / total_pairs as f64
}

pub fn compute_quality(positions: &[SphericalPoint]) -> LayoutQuality {
    LayoutQuality {
        dispersion_score: compute_dispersion(positions),
        overlap_score: compute_overlap(positions, 0.01),
        silhouette_score: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn empty_and_single_point() {
        assert!((compute_dispersion(&[]) - 1.0).abs() < 1e-12);
        assert!(compute_overlap(&[], 0.01).abs() < 1e-12);

        let single = vec![SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2)];
        assert!((compute_dispersion(&single) - 1.0).abs() < 1e-12);
        assert!(compute_overlap(&single, 0.01).abs() < 1e-12);
    }

    #[test]
    fn two_opposite_points_high_dispersion() {
        let positions = vec![
            SphericalPoint::new_unchecked(1.0, 0.0, 0.0),
            SphericalPoint::new_unchecked(1.0, 0.0, PI),
        ];
        let d = compute_dispersion(&positions);
        assert!(d > 0.7, "dispersion {d} should be high for opposite points");
    }

    #[test]
    fn two_identical_points_full_overlap() {
        let positions = vec![
            SphericalPoint::new_unchecked(1.0, 0.5, 1.0),
            SphericalPoint::new_unchecked(1.0, 0.5, 1.0),
        ];
        let o = compute_overlap(&positions, 0.01);
        assert!(
            (o - 1.0).abs() < 1e-12,
            "overlap {o} should be 1.0 for identical points"
        );
    }

    #[test]
    fn well_spaced_points_decent_dispersion() {
        let positions = vec![
            SphericalPoint::new_unchecked(1.0, 0.0, 0.0),
            SphericalPoint::new_unchecked(1.0, 0.0, PI),
            SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2),
            SphericalPoint::new_unchecked(1.0, FRAC_PI_2, FRAC_PI_2),
            SphericalPoint::new_unchecked(1.0, PI, FRAC_PI_2),
            SphericalPoint::new_unchecked(1.0, 3.0 * FRAC_PI_2, FRAC_PI_2),
        ];
        let d = compute_dispersion(&positions);
        assert!(
            d > 0.5,
            "dispersion {d} should be > 0.5 for well-spaced points"
        );
    }

    #[test]
    fn compute_quality_combines_metrics() {
        let positions = vec![
            SphericalPoint::new_unchecked(1.0, 0.0, 0.0),
            SphericalPoint::new_unchecked(1.0, 0.0, PI),
        ];
        let q = compute_quality(&positions);
        assert!(q.dispersion_score > 0.0);
        assert!(q.overlap_score >= 0.0);
        assert!((q.silhouette_score).abs() < 1e-12);
    }
}
