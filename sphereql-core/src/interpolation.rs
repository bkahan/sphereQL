use crate::conversions::{cartesian_to_spherical, spherical_to_cartesian};
use crate::distance::angular_distance;
use crate::types::{CartesianPoint, SphericalPoint};

/// Spherical linear interpolation between two unit-sphere points.
///
/// The parameter `t` is clamped to [0, 1]. At t=0 returns `a`, at t=1 returns `b`.
///
/// ```
/// use sphereql_core::{SphericalPoint, slerp, angular_distance};
/// use std::f64::consts::FRAC_PI_2;
///
/// let a = SphericalPoint::new_unchecked(1.0, 0.0, FRAC_PI_2);
/// let b = SphericalPoint::new_unchecked(1.0, FRAC_PI_2, FRAC_PI_2);
/// let mid = slerp(&a, &b, 0.5);
/// let da = angular_distance(&mid, &a);
/// let db = angular_distance(&mid, &b);
/// assert!((da - db).abs() < 1e-10);
/// ```
#[must_use]
pub fn slerp(a: &SphericalPoint, b: &SphericalPoint, t: f64) -> SphericalPoint {
    let t = t.clamp(0.0, 1.0);
    let a_unit = SphericalPoint::new_unchecked(1.0, a.theta, a.phi);
    let b_unit = SphericalPoint::new_unchecked(1.0, b.theta, b.phi);

    let omega = angular_distance(&a_unit, &b_unit);

    if omega.abs() < 1e-10 {
        return a_unit;
    }

    let sin_omega = omega.sin();
    let factor_a = ((1.0 - t) * omega).sin() / sin_omega;
    let factor_b = (t * omega).sin() / sin_omega;

    let ac = spherical_to_cartesian(&a_unit);
    let bc = spherical_to_cartesian(&b_unit);

    let result = CartesianPoint::new(
        factor_a * ac.x + factor_b * bc.x,
        factor_a * ac.y + factor_b * bc.y,
        factor_a * ac.z + factor_b * bc.z,
    );

    cartesian_to_spherical(&result)
}

/// Normalized linear interpolation between two unit-sphere points.
///
/// Linearly interpolates in Cartesian space, then normalizes back to the
/// unit sphere. Faster than [`slerp`] but does not produce constant-speed
/// traversal along the great circle arc.
///
/// The parameter `t` is clamped to [0, 1].
#[must_use]
pub fn nlerp(a: &SphericalPoint, b: &SphericalPoint, t: f64) -> SphericalPoint {
    let t = t.clamp(0.0, 1.0);
    let a_unit = SphericalPoint::new_unchecked(1.0, a.theta, a.phi);
    let b_unit = SphericalPoint::new_unchecked(1.0, b.theta, b.phi);

    let ac = spherical_to_cartesian(&a_unit);
    let bc = spherical_to_cartesian(&b_unit);

    let lerped = CartesianPoint::new(
        ac.x + t * (bc.x - ac.x),
        ac.y + t * (bc.y - ac.y),
        ac.z + t * (bc.z - ac.z),
    );

    cartesian_to_spherical(&lerped.normalize())
}

/// Full spherical linear interpolation including radius.
///
/// Interpolates both direction (via [`slerp`]) and radial distance
/// (linearly between `a.r` and `b.r`). The parameter `t` is clamped
/// to [0, 1].
#[must_use]
pub fn full_slerp(a: &SphericalPoint, b: &SphericalPoint, t: f64) -> SphericalPoint {
    let t = t.clamp(0.0, 1.0);
    let direction = slerp(a, b, t);
    let r = a.r + t * (b.r - a.r);
    SphericalPoint::new_unchecked(r, direction.theta, direction.phi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    fn unit_point(theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(1.0, theta, phi)
    }

    #[test]
    fn slerp_at_t0_returns_a() {
        let a = unit_point(0.0, FRAC_PI_2);
        let b = unit_point(FRAC_PI_2, FRAC_PI_2);
        let result = slerp(&a, &b, 0.0);
        assert_relative_eq!(result.theta, a.theta, epsilon = 1e-12);
        assert_relative_eq!(result.phi, a.phi, epsilon = 1e-12);
    }

    #[test]
    fn slerp_at_t1_returns_b() {
        let a = unit_point(0.0, FRAC_PI_2);
        let b = unit_point(FRAC_PI_2, FRAC_PI_2);
        let result = slerp(&a, &b, 1.0);
        assert_relative_eq!(result.theta, b.theta, epsilon = 1e-12);
        assert_relative_eq!(result.phi, b.phi, epsilon = 1e-12);
    }

    #[test]
    fn slerp_midpoint_is_on_great_circle() {
        let a = unit_point(0.0, FRAC_PI_2);
        let b = unit_point(FRAC_PI_2, FRAC_PI_2);
        let mid = slerp(&a, &b, 0.5);

        let dist_a = angular_distance(&unit_point(mid.theta, mid.phi), &a);
        let dist_b = angular_distance(&unit_point(mid.theta, mid.phi), &b);
        assert_relative_eq!(dist_a, dist_b, epsilon = 1e-12);

        let total = angular_distance(&a, &b);
        assert_relative_eq!(dist_a, total / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn slerp_clamps_t() {
        let a = unit_point(0.0, FRAC_PI_2);
        let b = unit_point(FRAC_PI_2, FRAC_PI_2);

        let below = slerp(&a, &b, -0.5);
        let at_zero = slerp(&a, &b, 0.0);
        assert_relative_eq!(below.theta, at_zero.theta, epsilon = 1e-12);
        assert_relative_eq!(below.phi, at_zero.phi, epsilon = 1e-12);

        let above = slerp(&a, &b, 1.5);
        let at_one = slerp(&a, &b, 1.0);
        assert_relative_eq!(above.theta, at_one.theta, epsilon = 1e-12);
        assert_relative_eq!(above.phi, at_one.phi, epsilon = 1e-12);
    }

    #[test]
    fn nlerp_endpoints_match_slerp() {
        let a = unit_point(0.3, 1.0);
        let b = unit_point(1.5, 0.8);

        let s0 = slerp(&a, &b, 0.0);
        let n0 = nlerp(&a, &b, 0.0);
        assert_relative_eq!(s0.theta, n0.theta, epsilon = 1e-10);
        assert_relative_eq!(s0.phi, n0.phi, epsilon = 1e-10);

        let s1 = slerp(&a, &b, 1.0);
        let n1 = nlerp(&a, &b, 1.0);
        assert_relative_eq!(s1.theta, n1.theta, epsilon = 1e-10);
        assert_relative_eq!(s1.phi, n1.phi, epsilon = 1e-10);
    }

    #[test]
    fn full_slerp_interpolates_radius() {
        let a = SphericalPoint::new_unchecked(2.0, 0.0, FRAC_PI_2);
        let b = SphericalPoint::new_unchecked(8.0, FRAC_PI_2, FRAC_PI_2);

        let mid = full_slerp(&a, &b, 0.5);
        assert_relative_eq!(mid.r, 5.0, epsilon = 1e-12);

        let quarter = full_slerp(&a, &b, 0.25);
        assert_relative_eq!(quarter.r, 3.5, epsilon = 1e-12);
    }

    #[test]
    fn full_slerp_endpoints() {
        let a = SphericalPoint::new_unchecked(3.0, 0.5, 1.0);
        let b = SphericalPoint::new_unchecked(7.0, 1.5, 0.8);

        let at_a = full_slerp(&a, &b, 0.0);
        assert_relative_eq!(at_a.r, a.r, epsilon = 1e-12);
        assert_relative_eq!(at_a.theta, a.theta, epsilon = 1e-12);
        assert_relative_eq!(at_a.phi, a.phi, epsilon = 1e-12);

        let at_b = full_slerp(&a, &b, 1.0);
        assert_relative_eq!(at_b.r, b.r, epsilon = 1e-12);
        assert_relative_eq!(at_b.theta, b.theta, epsilon = 1e-12);
        assert_relative_eq!(at_b.phi, b.phi, epsilon = 1e-12);
    }

    #[test]
    fn interpolation_between_identical_points() {
        let p = unit_point(1.0, 0.5);

        let s = slerp(&p, &p, 0.5);
        assert_relative_eq!(s.theta, p.theta, epsilon = 1e-12);
        assert_relative_eq!(s.phi, p.phi, epsilon = 1e-12);

        let n = nlerp(&p, &p, 0.5);
        assert_relative_eq!(n.theta, p.theta, epsilon = 1e-12);
        assert_relative_eq!(n.phi, p.phi, epsilon = 1e-12);

        let f = full_slerp(&p, &p, 0.5);
        assert_relative_eq!(f.r, p.r, epsilon = 1e-12);
        assert_relative_eq!(f.theta, p.theta, epsilon = 1e-12);
        assert_relative_eq!(f.phi, p.phi, epsilon = 1e-12);
    }
}
