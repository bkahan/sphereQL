use std::sync::Arc;

use sphereql_core::SphericalPoint;

#[derive(Debug, Clone)]
pub struct Embedding {
    pub values: Vec<f64>,
}

/// A projected point on the sphere with rich attributes from the projection.
///
/// Extends the raw `SphericalPoint` with metadata that captures how much
/// information was preserved (or lost) during dimensionality reduction.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ProjectedPoint {
    /// The spherical position (r, theta, phi).
    pub position: SphericalPoint,
    /// How well the 3D projection captures this point's original direction.
    /// Computed as 1 - (residual / total variance). Range [0, 1]:
    /// - 1.0: perfect reconstruction (all variance explained by 3 PCA components)
    /// - 0.0: the projection lost everything
    pub certainty: f64,
    /// Semantic strength of the original embedding (pre-normalization magnitude).
    /// Higher values indicate more specific/confident embeddings.
    pub intensity: f64,
    /// Magnitude of the 3-component PCA projection before normalization.
    /// Points near the PCA centroid have low projection magnitude and are
    /// ambiguous — they don't strongly align with any principal direction.
    pub projection_magnitude: f64,
}

impl ProjectedPoint {
    pub fn new(
        position: SphericalPoint,
        certainty: f64,
        intensity: f64,
        projection_magnitude: f64,
    ) -> Self {
        Self {
            position,
            certainty,
            intensity,
            projection_magnitude,
        }
    }

    /// Create a basic projected point with no metadata (legacy compat).
    pub fn from_position(position: SphericalPoint, intensity: f64) -> Self {
        Self {
            position,
            certainty: 1.0,
            intensity,
            projection_magnitude: 1.0,
        }
    }
}

impl Embedding {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    pub fn magnitude(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn normalized(&self) -> Vec<f64> {
        let mag = self.magnitude();
        if mag < f64::EPSILON {
            let mut v = vec![0.0; self.values.len()];
            if !v.is_empty() {
                v[0] = 1.0;
            }
            return v;
        }
        self.values.iter().map(|v| v / mag).collect()
    }
}

impl From<Vec<f64>> for Embedding {
    fn from(values: Vec<f64>) -> Self {
        Self { values }
    }
}

impl From<&[f64]> for Embedding {
    fn from(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
        }
    }
}

/// Controls how the radial coordinate r is computed from an embedding.
///
/// The angular coordinates (theta, phi) always encode semantic direction.
/// The radial coordinate is free to encode magnitude, metadata, or a fixed value.
#[derive(Default)]
pub enum RadialStrategy {
    /// Constant radius for all projections.
    Fixed(f64),
    /// r = L2 magnitude of the raw (pre-normalization) embedding.
    /// Encodes embedding "confidence" or specificity.
    #[default]
    Magnitude,
    /// r = f(magnitude). Apply a custom transform to the pre-normalization magnitude.
    /// Useful for log-scaling, clamping, or mapping metadata that correlates with magnitude.
    MagnitudeTransform(Arc<dyn Fn(f64) -> f64 + Send + Sync>),
}

impl Clone for RadialStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::Fixed(r) => Self::Fixed(*r),
            Self::Magnitude => Self::Magnitude,
            Self::MagnitudeTransform(f) => Self::MagnitudeTransform(Arc::clone(f)),
        }
    }
}

impl std::fmt::Debug for RadialStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(r) => write!(f, "Fixed({r})"),
            Self::Magnitude => write!(f, "Magnitude"),
            Self::MagnitudeTransform(_) => write!(f, "MagnitudeTransform(<fn>)"),
        }
    }
}

impl RadialStrategy {
    pub fn compute(&self, magnitude: f64) -> f64 {
        match self {
            Self::Fixed(r) => *r,
            Self::Magnitude => magnitude,
            Self::MagnitudeTransform(f) => f(magnitude),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_magnitude() {
        let e = Embedding::new(vec![3.0, 4.0]);
        assert!((e.magnitude() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn embedding_normalized() {
        let e = Embedding::new(vec![3.0, 4.0]);
        let n = e.normalized();
        assert!((n[0] - 0.6).abs() < 1e-12);
        assert!((n[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn zero_embedding_normalized_fallback() {
        let e = Embedding::new(vec![0.0, 0.0, 0.0]);
        let n = e.normalized();
        assert!((n[0] - 1.0).abs() < 1e-12);
        assert!(n[1].abs() < 1e-12);
        assert!(n[2].abs() < 1e-12);
    }

    #[test]
    fn from_vec() {
        let e: Embedding = vec![1.0, 2.0, 3.0].into();
        assert_eq!(e.dimension(), 3);
    }

    #[test]
    fn from_slice() {
        let data = [1.0, 2.0, 3.0];
        let e: Embedding = data.as_slice().into();
        assert_eq!(e.dimension(), 3);
    }

    #[test]
    fn radial_fixed() {
        let r = RadialStrategy::Fixed(2.5);
        assert!((r.compute(999.0) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn radial_magnitude() {
        let r = RadialStrategy::Magnitude;
        assert!((r.compute(7.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn radial_transform() {
        let r = RadialStrategy::MagnitudeTransform(Arc::new(|m| m.ln_1p()));
        let expected = 5.0_f64.ln_1p();
        assert!((r.compute(5.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn radial_clone() {
        let r = RadialStrategy::MagnitudeTransform(Arc::new(|m| m * 2.0));
        let r2 = r.clone();
        assert!((r.compute(3.0) - r2.compute(3.0)).abs() < 1e-12);
    }

    #[test]
    fn radial_debug() {
        assert_eq!(format!("{:?}", RadialStrategy::Fixed(1.0)), "Fixed(1)");
        assert_eq!(format!("{:?}", RadialStrategy::Magnitude), "Magnitude");
        let t = RadialStrategy::MagnitudeTransform(Arc::new(|m| m));
        assert_eq!(format!("{t:?}"), "MagnitudeTransform(<fn>)");
    }
}
