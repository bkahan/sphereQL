/// All errors that `sphereql-core` can surface.
///
/// `#[non_exhaustive]` — new variants can be added in minor releases.
/// Match with a `_ => …` arm when handling exhaustively.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
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
    #[error("invalid cap: half_angle {0} must be in (0, π]")]
    InvalidCapAngle(f64),
    #[error(
        "invalid wedge bounds: theta_min {theta_min} and theta_max {theta_max} must be in [0, 2π)"
    )]
    InvalidWedgeBounds { theta_min: f64, theta_max: f64 },
    #[error("zero vector cannot be normalized")]
    ZeroVector,
    #[error("vector length mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_human_readable() {
        assert!(
            SphereQlError::InvalidRadius(-1.0)
                .to_string()
                .contains("-1")
        );
        assert!(
            SphereQlError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
            .to_string()
            .contains("3")
        );
        assert!(
            SphereQlError::InvalidShellBounds {
                inner: 5.0,
                outer: 1.0
            }
            .to_string()
            .contains("5")
        );
    }

    #[test]
    fn zero_vector_error_formats() {
        let msg = SphereQlError::ZeroVector.to_string();
        assert!(!msg.is_empty());
    }
}
