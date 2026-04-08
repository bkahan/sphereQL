#[derive(Debug, Clone, thiserror::Error)]
pub enum SphereQlError {
    #[error("invalid radius {0}: must be >= 0")]
    InvalidRadius(f64),
    #[error("invalid theta {0}: must be in [0, 2π)")]
    InvalidTheta(f64),
    #[error("invalid phi {0}: must be in [0, π]")]
    InvalidPhi(f64),
    #[error("invalid latitude {0}: must be in [-90, 90]")]
    InvalidLatitude(f64),
    #[error("invalid longitude {0}: must be in [-180, 180]")]
    InvalidLongitude(f64),
    #[error("invalid altitude {0}: must be >= 0")]
    InvalidAltitude(f64),
    #[error("invalid shell bounds: inner {inner} must be < outer {outer}")]
    InvalidShellBounds { inner: f64, outer: f64 },
    #[error("invalid band bounds: phi_min {phi_min} must be < phi_max {phi_max}")]
    InvalidBandBounds { phi_min: f64, phi_max: f64 },
    #[error("invalid cone: half_angle {0} must be in (0, π]")]
    InvalidConeAngle(f64),
    #[error("zero vector cannot be normalized")]
    ZeroVector,
}
