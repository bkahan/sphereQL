use std::sync::Arc;

use sphereql_core::SphericalPoint;

/// A high-dimensional embedding vector.
///
/// Wraps a `Vec<f64>` with projection-oriented helpers: `magnitude`,
/// `normalized` (L2 unit vector with a [1, 0, …] fallback for the zero
/// vector), and `From<Vec<f64>>` / `From<&[f64]>` for ergonomic construction.
///
/// All projection families normalize embeddings to the unit hypersphere
/// before extracting angular structure, so raw magnitude is preserved
/// separately as the radial coordinate via [`RadialStrategy`].
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

/// Per-point information available when resolving the radial coordinate.
///
/// Modern L2-normalized embeddings make `embedding_magnitude` a constant
/// 1.0, wasting the radial axis. This struct exposes the additional
/// signals the projection naturally produces (post-projection magnitude,
/// per-point certainty) so a [`RadialStrategy`] can encode something
/// useful in `r` instead.
#[derive(Debug, Clone, Copy)]
pub struct RadialContext {
    /// L2 norm of the raw input embedding (pre-normalization).
    pub embedding_magnitude: f64,
    /// L2 norm of the (x, y, z) projected vector before re-scaling.
    /// High values mean the 3 components captured most of the input's
    /// variance; low values mean the input fell mostly into the residual.
    pub projection_magnitude: f64,
    /// Fraction of input variance retained by the projection, in `[0, 1]`.
    /// Source depends on the projection family — PCA uses captured-variance
    /// ratio; KPCA uses Hoffmann's reconstruction-error formula;
    /// Laplacian uses tanh(projection_magnitude); Random reports 1.0.
    pub certainty: f64,
}

impl RadialContext {
    /// Construct from just the embedding magnitude. Other fields default
    /// to neutral values; use [`Self::full`] when projection-side
    /// information is available.
    pub fn from_magnitude(embedding_magnitude: f64) -> Self {
        Self {
            embedding_magnitude,
            projection_magnitude: embedding_magnitude,
            certainty: 1.0,
        }
    }

    /// Construct with all three signals populated.
    pub fn full(embedding_magnitude: f64, projection_magnitude: f64, certainty: f64) -> Self {
        Self {
            embedding_magnitude,
            projection_magnitude,
            certainty,
        }
    }
}

/// Controls how the radial coordinate r is computed from an embedding.
///
/// The angular coordinates (theta, phi) always encode semantic direction.
/// The radial coordinate is free to encode magnitude, fidelity, or a
/// caller-defined function of any per-point signal the projection exposes
/// via [`RadialContext`].
#[derive(Default)]
pub enum RadialStrategy {
    /// Constant radius for all projections.
    Fixed(f64),
    /// r = L2 magnitude of the raw (pre-normalization) embedding.
    /// Encodes embedding "confidence" or specificity. Degenerates to a
    /// constant when inputs are L2-normalized — pick one of the
    /// projection-side variants below in that case.
    #[default]
    Magnitude,
    /// r = f(embedding_magnitude). Apply a custom transform to the
    /// pre-normalization magnitude (e.g. log scaling, clamping).
    MagnitudeTransform(Arc<dyn Fn(f64) -> f64 + Send + Sync>),
    /// r = ‖(x, y, z)‖ — how much of the input variance landed in the
    /// projected 3-vector. Universal across all four projection families.
    /// Recommended starting point for normalized embeddings.
    ProjectionMagnitude,
    /// r = `scale * certainty`, where `certainty ∈ [0, 1]` is the
    /// projection-supplied per-point fidelity score. Higher r ⇒ this
    /// point is well-explained by the 3D projection.
    Certainty { scale: f64 },
    /// r = f(&context). Escape hatch for arbitrary per-point logic over
    /// any combination of the [`RadialContext`] signals.
    Custom(Arc<dyn Fn(&RadialContext) -> f64 + Send + Sync>),
}

impl Clone for RadialStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::Fixed(r) => Self::Fixed(*r),
            Self::Magnitude => Self::Magnitude,
            Self::MagnitudeTransform(f) => Self::MagnitudeTransform(Arc::clone(f)),
            Self::ProjectionMagnitude => Self::ProjectionMagnitude,
            Self::Certainty { scale } => Self::Certainty { scale: *scale },
            Self::Custom(f) => Self::Custom(Arc::clone(f)),
        }
    }
}

impl std::fmt::Debug for RadialStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(r) => write!(f, "Fixed({r})"),
            Self::Magnitude => write!(f, "Magnitude"),
            Self::MagnitudeTransform(_) => write!(f, "MagnitudeTransform(<fn>)"),
            Self::ProjectionMagnitude => write!(f, "ProjectionMagnitude"),
            Self::Certainty { scale } => write!(f, "Certainty {{ scale: {scale} }}"),
            Self::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

impl RadialStrategy {
    /// Resolve `r` against the full per-point context. All four projection
    /// families construct a [`RadialContext`] inline and call this.
    pub fn compute_rich(&self, ctx: &RadialContext) -> f64 {
        match self {
            Self::Fixed(r) => *r,
            Self::Magnitude => ctx.embedding_magnitude,
            Self::MagnitudeTransform(f) => f(ctx.embedding_magnitude),
            Self::ProjectionMagnitude => ctx.projection_magnitude,
            Self::Certainty { scale } => scale * ctx.certainty,
            Self::Custom(f) => f(ctx),
        }
    }

    /// Backward-compatible shim. Use [`Self::compute_rich`] when
    /// projection-side context is available — the new `ProjectionMagnitude`,
    /// `Certainty`, and `Custom` variants only do something useful there.
    pub fn compute(&self, magnitude: f64) -> f64 {
        self.compute_rich(&RadialContext::from_magnitude(magnitude))
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
    fn radial_projection_magnitude_uses_context() {
        let r = RadialStrategy::ProjectionMagnitude;
        let ctx = RadialContext::full(99.0, 0.42, 0.5);
        assert!((r.compute_rich(&ctx) - 0.42).abs() < 1e-12);
    }

    #[test]
    fn radial_certainty_scales() {
        let r = RadialStrategy::Certainty { scale: 2.0 };
        let ctx = RadialContext::full(99.0, 0.42, 0.3);
        assert!((r.compute_rich(&ctx) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn radial_custom_sees_full_context() {
        let r = RadialStrategy::Custom(Arc::new(|c| {
            c.embedding_magnitude + c.projection_magnitude + c.certainty
        }));
        let ctx = RadialContext::full(1.0, 2.0, 0.5);
        assert!((r.compute_rich(&ctx) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn radial_compute_shim_is_backward_compatible() {
        // The deprecated `compute(magnitude)` shim must still produce the
        // same value as before for the original three variants.
        assert!((RadialStrategy::Fixed(7.0).compute(123.0) - 7.0).abs() < 1e-12);
        assert!((RadialStrategy::Magnitude.compute(7.0) - 7.0).abs() < 1e-12);
        let xform = RadialStrategy::MagnitudeTransform(Arc::new(|m| m * m));
        assert!((xform.compute(3.0) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn radial_debug() {
        assert_eq!(format!("{:?}", RadialStrategy::Fixed(1.0)), "Fixed(1)");
        assert_eq!(format!("{:?}", RadialStrategy::Magnitude), "Magnitude");
        let t = RadialStrategy::MagnitudeTransform(Arc::new(|m| m));
        assert_eq!(format!("{t:?}"), "MagnitudeTransform(<fn>)");
    }
}
