use std::f64::consts::{PI, TAU};

use crate::error::SphereQlError;

/// A point in spherical coordinates (r, theta, phi).
///
/// - `r`: radial distance (must be >= 0)
/// - `theta`: azimuthal angle in [0, 2pi)
/// - `phi`: polar angle in [0, pi]
///
/// ```
/// use sphereql_core::SphericalPoint;
///
/// let p = SphericalPoint::new(1.0, 0.5, 0.7).unwrap();
/// assert_eq!(p.r, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SphericalPoint {
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
}

impl SphericalPoint {
    /// Creates a new `SphericalPoint` with validation.
    ///
    /// Returns an error if `r < 0`, `theta` is outside [0, 2pi), or `phi` is outside [0, pi].
    ///
    /// ```
    /// use sphereql_core::SphericalPoint;
    ///
    /// assert!(SphericalPoint::new(1.0, 0.5, 0.7).is_ok());
    /// assert!(SphericalPoint::new(-1.0, 0.0, 0.0).is_err());
    /// ```
    pub fn new(r: f64, theta: f64, phi: f64) -> Result<Self, SphereQlError> {
        if r < 0.0 {
            return Err(SphereQlError::InvalidRadius(r));
        }
        if !(0.0..TAU).contains(&theta) {
            return Err(SphereQlError::InvalidTheta(theta));
        }
        if !(0.0..=PI).contains(&phi) {
            return Err(SphereQlError::InvalidPhi(phi));
        }
        Ok(Self { r, theta, phi })
    }

    /// Creates a new `SphericalPoint` without validation.
    ///
    /// Caller is responsible for ensuring values are in valid ranges.
    pub fn new_unchecked(r: f64, theta: f64, phi: f64) -> Self {
        Self { r, theta, phi }
    }

    pub fn origin() -> Self {
        Self {
            r: 0.0,
            theta: 0.0,
            phi: 0.0,
        }
    }

    /// Unit Cartesian direction vector (x, y, z) for this point's angular position.
    ///
    /// Ignores the radial component — returns the point on S² at (θ, φ).
    /// Used for fast angular distance approximation via dot product:
    /// `1 - dot(a.unit_cartesian(), b.unit_cartesian())` is monotone with
    /// angular distance, avoiding the full Vincenty formula.
    #[inline]
    pub fn unit_cartesian(&self) -> [f64; 3] {
        let (sin_phi, cos_phi) = self.phi.sin_cos();
        let (sin_theta, cos_theta) = self.theta.sin_cos();
        [sin_phi * cos_theta, sin_phi * sin_theta, cos_phi]
    }
}

impl approx::AbsDiffEq for SphericalPoint {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.r, &other.r, epsilon)
            && f64::abs_diff_eq(&self.theta, &other.theta, epsilon)
            && f64::abs_diff_eq(&self.phi, &other.phi, epsilon)
    }
}

impl approx::RelativeEq for SphericalPoint {
    fn default_max_relative() -> Self::Epsilon {
        f64::EPSILON
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.r, &other.r, epsilon, max_relative)
            && f64::relative_eq(&self.theta, &other.theta, epsilon, max_relative)
            && f64::relative_eq(&self.phi, &other.phi, epsilon, max_relative)
    }
}

/// A point in 3D Cartesian coordinates (x, y, z).
///
/// ```
/// use sphereql_core::CartesianPoint;
///
/// let p = CartesianPoint::new(1.0, 0.0, 0.0);
/// assert_eq!(p.magnitude(), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CartesianPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl CartesianPoint {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < f64::EPSILON {
            return Self::origin();
        }
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }
}

impl approx::AbsDiffEq for CartesianPoint {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.x, &other.x, epsilon)
            && f64::abs_diff_eq(&self.y, &other.y, epsilon)
            && f64::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}

impl approx::RelativeEq for CartesianPoint {
    fn default_max_relative() -> Self::Epsilon {
        f64::EPSILON
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f64::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && f64::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

impl GeoPoint {
    pub fn new(lat: f64, lon: f64, alt: f64) -> Result<Self, SphereQlError> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SphereQlError::InvalidLatitude(lat));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SphereQlError::InvalidLongitude(lon));
        }
        if alt < 0.0 {
            return Err(SphereQlError::InvalidAltitude(alt));
        }
        Ok(Self { lat, lon, alt })
    }

    pub fn new_unchecked(lat: f64, lon: f64, alt: f64) -> Self {
        Self { lat, lon, alt }
    }
}

impl approx::AbsDiffEq for GeoPoint {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.lat, &other.lat, epsilon)
            && f64::abs_diff_eq(&self.lon, &other.lon, epsilon)
            && f64::abs_diff_eq(&self.alt, &other.alt, epsilon)
    }
}

impl approx::RelativeEq for GeoPoint {
    fn default_max_relative() -> Self::Epsilon {
        f64::EPSILON
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.lat, &other.lat, epsilon, max_relative)
            && f64::relative_eq(&self.lon, &other.lon, epsilon, max_relative)
            && f64::relative_eq(&self.alt, &other.alt, epsilon, max_relative)
    }
}
