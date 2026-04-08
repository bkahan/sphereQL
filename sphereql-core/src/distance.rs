use crate::conversions::spherical_to_cartesian;
use crate::types::{CartesianPoint, SphericalPoint};

pub fn angular_distance(a: &SphericalPoint, b: &SphericalPoint) -> f64 {
    let a_unit = SphericalPoint::new_unchecked(1.0, a.theta, a.phi);
    let b_unit = SphericalPoint::new_unchecked(1.0, b.theta, b.phi);
    let ac = spherical_to_cartesian(&a_unit);
    let bc = spherical_to_cartesian(&b_unit);

    // Vincenty formula: numerically stable for all angular separations
    let cross_x = ac.y * bc.z - ac.z * bc.y;
    let cross_y = ac.z * bc.x - ac.x * bc.z;
    let cross_z = ac.x * bc.y - ac.y * bc.x;
    let cross_mag = (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).sqrt();

    let dot = ac.x * bc.x + ac.y * bc.y + ac.z * bc.z;

    cross_mag.atan2(dot)
}

pub fn great_circle_distance(a: &SphericalPoint, b: &SphericalPoint, radius: f64) -> f64 {
    radius * angular_distance(a, b)
}

pub fn chord_distance(a: &SphericalPoint, b: &SphericalPoint) -> f64 {
    let ac = spherical_to_cartesian(a);
    let bc = spherical_to_cartesian(b);
    euclidean_distance(&ac, &bc)
}

pub fn euclidean_distance(a: &CartesianPoint, b: &CartesianPoint) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn point(theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(1.0, theta, phi)
    }

    #[test]
    fn angular_distance_same_point() {
        let p = point(0.5, 1.0);
        assert_relative_eq!(angular_distance(&p, &p), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn angular_distance_antipodal() {
        let a = point(0.0, FRAC_PI_2);
        let b = point(PI, PI - FRAC_PI_2);
        assert_relative_eq!(angular_distance(&a, &b), PI, epsilon = 1e-12);
    }

    #[test]
    fn angular_distance_90_degrees() {
        let a = point(0.0, 0.0);
        let b = point(0.0, FRAC_PI_2);
        assert_relative_eq!(angular_distance(&a, &b), FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn great_circle_on_unit_sphere() {
        let a = point(0.0, 0.0);
        let b = point(0.0, FRAC_PI_2);
        assert_relative_eq!(great_circle_distance(&a, &b, 1.0), FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn great_circle_with_radius() {
        let a = point(0.0, 0.0);
        let b = point(0.0, FRAC_PI_2);
        let r = 6371.0;
        assert_relative_eq!(
            great_circle_distance(&a, &b, r),
            r * FRAC_PI_2,
            epsilon = 1e-9
        );
    }

    #[test]
    fn chord_distance_same_point() {
        let p = point(1.0, 0.5);
        assert_relative_eq!(chord_distance(&p, &p), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn chord_distance_antipodal_unit_sphere() {
        let a = point(0.0, FRAC_PI_2);
        let b = point(PI, PI - FRAC_PI_2);
        assert_relative_eq!(chord_distance(&a, &b), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn chord_distance_90_degrees_unit_sphere() {
        let a = point(0.0, 0.0);
        let b = point(0.0, FRAC_PI_2);
        assert_relative_eq!(chord_distance(&a, &b), 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn euclidean_distance_basic() {
        let a = CartesianPoint::new(0.0, 0.0, 0.0);
        let b = CartesianPoint::new(1.0, 0.0, 0.0);
        assert_relative_eq!(euclidean_distance(&a, &b), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn euclidean_distance_3d() {
        let a = CartesianPoint::new(1.0, 2.0, 3.0);
        let b = CartesianPoint::new(4.0, 6.0, 3.0);
        assert_relative_eq!(euclidean_distance(&a, &b), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn vincenty_stability_near_zero() {
        let a = point(0.0, FRAC_PI_2);
        let b = point(1e-15, FRAC_PI_2);
        let dist = angular_distance(&a, &b);
        assert!(dist >= 0.0);
        assert!(dist < 1e-10);
    }

    #[test]
    fn vincenty_stability_near_pi() {
        let a = point(0.0, 1e-15);
        let b = point(PI, PI - 1e-15);
        let dist = angular_distance(&a, &b);
        assert_relative_eq!(dist, PI, epsilon = 1e-10);
    }
}
