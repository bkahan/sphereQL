use crate::types::{CartesianPoint, GeoPoint, SphericalPoint};
use std::f64::consts::TAU;

/// Converts a spherical point to Cartesian coordinates.
///
/// ```
/// use sphereql_core::{SphericalPoint, spherical_to_cartesian};
///
/// let p = SphericalPoint::new(1.0, 0.0, std::f64::consts::FRAC_PI_2).unwrap();
/// let c = spherical_to_cartesian(&p);
/// assert!((c.x - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn spherical_to_cartesian(p: &SphericalPoint) -> CartesianPoint {
    let x = p.r * p.phi.sin() * p.theta.cos();
    let y = p.r * p.phi.sin() * p.theta.sin();
    let z = p.r * p.phi.cos();
    CartesianPoint::new(x, y, z)
}

/// Converts a Cartesian point to spherical coordinates.
///
/// ```
/// use sphereql_core::*;
///
/// let original = SphericalPoint::new(1.0, 0.5, 0.7).unwrap();
/// let roundtrip = cartesian_to_spherical(&spherical_to_cartesian(&original));
/// assert!((roundtrip.r - original.r).abs() < 1e-10);
/// ```
#[must_use]
pub fn cartesian_to_spherical(p: &CartesianPoint) -> SphericalPoint {
    let r = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
    if r < f64::EPSILON {
        return SphericalPoint::new_unchecked(0.0, 0.0, 0.0);
    }
    let theta = normalize_theta(p.y.atan2(p.x));
    let phi = (p.z / r).clamp(-1.0, 1.0).acos();
    SphericalPoint::new_unchecked(r, theta, phi)
}

/// Converts a spherical point to geographic coordinates (lat/lon/alt).
///
/// Assumes the unit sphere (r=1.0) represents the surface. Altitude is
/// the excess radius above 1.0, clamped to zero for sub-unit radii.
#[must_use]
pub fn spherical_to_geo(p: &SphericalPoint) -> GeoPoint {
    let lat = 90.0 - p.phi.to_degrees();
    let lon = theta_to_longitude(p.theta);
    let alt = (p.r - 1.0).max(0.0);
    GeoPoint::new_unchecked(lat, lon, alt)
}

/// Converts geographic coordinates to spherical coordinates.
///
/// Latitude maps to polar angle phi, longitude to azimuthal angle theta,
/// and altitude adds to the unit radius.
#[must_use]
pub fn geo_to_spherical(p: &GeoPoint) -> SphericalPoint {
    let phi = (90.0 - p.lat).to_radians();
    let theta = normalize_theta(p.lon.to_radians());
    let r = 1.0 + p.alt;
    SphericalPoint::new_unchecked(r, theta, phi)
}

/// Converts a Cartesian point to geographic coordinates via spherical.
#[must_use]
pub fn cartesian_to_geo(p: &CartesianPoint) -> GeoPoint {
    spherical_to_geo(&cartesian_to_spherical(p))
}

/// Converts geographic coordinates to Cartesian via spherical.
#[must_use]
pub fn geo_to_cartesian(p: &GeoPoint) -> CartesianPoint {
    spherical_to_cartesian(&geo_to_spherical(p))
}

fn normalize_theta(mut theta: f64) -> f64 {
    theta %= TAU;
    if theta < 0.0 {
        theta += TAU;
    }
    theta
}

fn theta_to_longitude(theta: f64) -> f64 {
    let deg = theta.to_degrees();
    if deg > 180.0 { deg - 360.0 } else { deg }
}

impl From<&SphericalPoint> for CartesianPoint {
    fn from(p: &SphericalPoint) -> Self {
        spherical_to_cartesian(p)
    }
}

impl From<&CartesianPoint> for SphericalPoint {
    fn from(p: &CartesianPoint) -> Self {
        cartesian_to_spherical(p)
    }
}

impl From<&SphericalPoint> for GeoPoint {
    fn from(p: &SphericalPoint) -> Self {
        spherical_to_geo(p)
    }
}

impl From<&GeoPoint> for SphericalPoint {
    fn from(p: &GeoPoint) -> Self {
        geo_to_spherical(p)
    }
}

impl From<&CartesianPoint> for GeoPoint {
    fn from(p: &CartesianPoint) -> Self {
        cartesian_to_geo(p)
    }
}

impl From<&GeoPoint> for CartesianPoint {
    fn from(p: &GeoPoint) -> Self {
        geo_to_cartesian(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    fn spherical(r: f64, theta: f64, phi: f64) -> SphericalPoint {
        SphericalPoint::new_unchecked(r, theta, phi)
    }

    fn geo(lat: f64, lon: f64, alt: f64) -> GeoPoint {
        GeoPoint::new_unchecked(lat, lon, alt)
    }

    // --- Roundtrip: spherical -> cartesian -> spherical ---

    #[test]
    fn roundtrip_spherical_cartesian_unit_sphere() {
        let original = spherical(1.0, 1.0, 0.8);
        let cart = spherical_to_cartesian(&original);
        let back = cartesian_to_spherical(&cart);
        assert_relative_eq!(back.r, original.r, epsilon = 1e-12);
        assert_relative_eq!(back.theta, original.theta, epsilon = 1e-12);
        assert_relative_eq!(back.phi, original.phi, epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_spherical_cartesian_large_radius() {
        let original = spherical(42.0, 3.5, 1.2);
        let cart = spherical_to_cartesian(&original);
        let back = cartesian_to_spherical(&cart);
        assert_relative_eq!(back.r, original.r, epsilon = 1e-10);
        assert_relative_eq!(back.theta, original.theta, epsilon = 1e-10);
        assert_relative_eq!(back.phi, original.phi, epsilon = 1e-10);
    }

    #[test]
    fn roundtrip_spherical_cartesian_near_tau() {
        let original = spherical(1.0, TAU - 0.001, FRAC_PI_2);
        let cart = spherical_to_cartesian(&original);
        let back = cartesian_to_spherical(&cart);
        assert_relative_eq!(back.r, original.r, epsilon = 1e-12);
        assert_relative_eq!(back.theta, original.theta, epsilon = 1e-6);
        assert_relative_eq!(back.phi, original.phi, epsilon = 1e-12);
    }

    // --- Roundtrip: spherical -> geo -> spherical ---

    #[test]
    fn roundtrip_spherical_geo_equator() {
        let original = spherical(1.0, 0.5, FRAC_PI_2);
        let g = spherical_to_geo(&original);
        let back = geo_to_spherical(&g);
        assert_relative_eq!(back.r, original.r, epsilon = 1e-12);
        assert_relative_eq!(back.theta, original.theta, epsilon = 1e-12);
        assert_relative_eq!(back.phi, original.phi, epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_spherical_geo_with_altitude() {
        let original = spherical(2.5, 1.0, 0.8);
        let g = spherical_to_geo(&original);
        let back = geo_to_spherical(&g);
        assert_relative_eq!(back.r, original.r, epsilon = 1e-12);
        assert_relative_eq!(back.theta, original.theta, epsilon = 1e-12);
        assert_relative_eq!(back.phi, original.phi, epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_geo_spherical_negative_lon() {
        let original = geo(45.0, -90.0, 0.0);
        let s = geo_to_spherical(&original);
        let back = spherical_to_geo(&s);
        assert_relative_eq!(back.lat, original.lat, epsilon = 1e-12);
        assert_relative_eq!(back.lon, original.lon, epsilon = 1e-12);
        assert_relative_eq!(back.alt, original.alt, epsilon = 1e-12);
    }

    // --- Known values: origin ---

    #[test]
    fn origin_spherical_to_cartesian() {
        let p = spherical(0.0, 0.0, 0.0);
        let c = spherical_to_cartesian(&p);
        assert_relative_eq!(c.x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.y, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn origin_cartesian_to_spherical() {
        let c = CartesianPoint::new(0.0, 0.0, 0.0);
        let s = cartesian_to_spherical(&c);
        assert_relative_eq!(s.r, 0.0, epsilon = 1e-15);
        assert_relative_eq!(s.theta, 0.0, epsilon = 1e-15);
        assert_relative_eq!(s.phi, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn near_zero_cartesian_to_spherical() {
        let c = CartesianPoint::new(1e-20, 1e-20, 1e-20);
        let s = cartesian_to_spherical(&c);
        assert_relative_eq!(s.r, 0.0, epsilon = 1e-15);
    }

    // --- Known values: north pole (phi=0) ---

    #[test]
    fn north_pole_spherical_to_cartesian() {
        let p = spherical(1.0, 0.0, 0.0);
        let c = spherical_to_cartesian(&p);
        assert_relative_eq!(c.x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.y, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.z, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn north_pole_to_geo() {
        let p = spherical(1.0, 0.0, 0.0);
        let g = spherical_to_geo(&p);
        assert_relative_eq!(g.lat, 90.0, epsilon = 1e-12);
        assert_relative_eq!(g.alt, 0.0, epsilon = 1e-12);
    }

    // --- Known values: south pole (phi=π) ---

    #[test]
    fn south_pole_spherical_to_cartesian() {
        let p = spherical(1.0, 0.0, PI);
        let c = spherical_to_cartesian(&p);
        assert_relative_eq!(c.x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.y, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.z, -1.0, epsilon = 1e-15);
    }

    #[test]
    fn south_pole_to_geo() {
        let p = spherical(1.0, 0.0, PI);
        let g = spherical_to_geo(&p);
        assert_relative_eq!(g.lat, -90.0, epsilon = 1e-12);
        assert_relative_eq!(g.alt, 0.0, epsilon = 1e-12);
    }

    // --- Known values: equator points ---

    #[test]
    fn equator_theta_zero() {
        let p = spherical(1.0, 0.0, FRAC_PI_2);
        let c = spherical_to_cartesian(&p);
        assert_relative_eq!(c.x, 1.0, epsilon = 1e-15);
        assert_relative_eq!(c.y, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn equator_theta_half_pi() {
        let p = spherical(1.0, FRAC_PI_2, FRAC_PI_2);
        let c = spherical_to_cartesian(&p);
        assert_relative_eq!(c.x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(c.y, 1.0, epsilon = 1e-15);
        assert_relative_eq!(c.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn equator_theta_pi() {
        let p = spherical(1.0, PI, FRAC_PI_2);
        let c = spherical_to_cartesian(&p);
        assert_relative_eq!(c.x, -1.0, epsilon = 1e-15);
        assert_relative_eq!(c.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(c.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn equator_to_geo_gives_zero_lat() {
        let p = spherical(1.0, 0.5, FRAC_PI_2);
        let g = spherical_to_geo(&p);
        assert_relative_eq!(g.lat, 0.0, epsilon = 1e-12);
    }

    // --- Edge cases: theta boundaries ---

    #[test]
    fn theta_zero_roundtrip() {
        let original = spherical(1.0, 0.0, 1.0);
        let back = cartesian_to_spherical(&spherical_to_cartesian(&original));
        assert_relative_eq!(back.theta, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn theta_near_tau_roundtrip() {
        let original = spherical(1.0, TAU - 1e-10, 1.0);
        let back = cartesian_to_spherical(&spherical_to_cartesian(&original));
        assert_relative_eq!(back.theta, original.theta, epsilon = 1e-6);
    }

    // --- Edge cases: phi boundaries ---

    #[test]
    fn phi_zero_roundtrip() {
        let p = spherical(1.0, 0.0, 0.0);
        let c = spherical_to_cartesian(&p);
        let back = cartesian_to_spherical(&c);
        assert_relative_eq!(back.r, 1.0, epsilon = 1e-12);
        assert_relative_eq!(back.phi, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn phi_pi_roundtrip() {
        let p = spherical(1.0, 0.0, PI);
        let c = spherical_to_cartesian(&p);
        let back = cartesian_to_spherical(&c);
        assert_relative_eq!(back.r, 1.0, epsilon = 1e-12);
        assert_relative_eq!(back.phi, PI, epsilon = 1e-12);
    }

    // --- Geo edge cases ---

    #[test]
    fn geo_longitude_180_boundary() {
        let p = spherical(1.0, PI, FRAC_PI_2);
        let g = spherical_to_geo(&p);
        assert_relative_eq!(g.lon, 180.0, epsilon = 1e-12);
    }

    #[test]
    fn geo_negative_longitude() {
        let p = spherical(1.0, PI + 0.5, FRAC_PI_2);
        let g = spherical_to_geo(&p);
        assert!(g.lon < 0.0);
        let back = geo_to_spherical(&g);
        assert_relative_eq!(back.theta, PI + 0.5, epsilon = 1e-12);
    }

    #[test]
    fn geo_altitude_clamped_to_zero() {
        let p = spherical(0.5, 0.0, FRAC_PI_2);
        let g = spherical_to_geo(&p);
        assert_relative_eq!(g.alt, 0.0, epsilon = 1e-15);
    }

    // --- From trait tests ---

    #[test]
    fn from_trait_spherical_to_cartesian() {
        let s = spherical(1.0, 0.0, FRAC_PI_2);
        let c: CartesianPoint = (&s).into();
        assert_relative_eq!(c.x, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn from_trait_cartesian_to_spherical() {
        let c = CartesianPoint::new(1.0, 0.0, 0.0);
        let s: SphericalPoint = (&c).into();
        assert_relative_eq!(s.r, 1.0, epsilon = 1e-12);
        assert_relative_eq!(s.theta, 0.0, epsilon = 1e-12);
        assert_relative_eq!(s.phi, FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn from_trait_geo_to_cartesian() {
        let g = geo(0.0, 0.0, 0.0);
        let c: CartesianPoint = (&g).into();
        assert_relative_eq!(c.x, 1.0, epsilon = 1e-12);
        assert_relative_eq!(c.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(c.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn from_trait_cartesian_to_geo() {
        let c = CartesianPoint::new(0.0, 0.0, 1.0);
        let g: GeoPoint = (&c).into();
        assert_relative_eq!(g.lat, 90.0, epsilon = 1e-12);
        assert_relative_eq!(g.alt, 0.0, epsilon = 1e-12);
    }
}
