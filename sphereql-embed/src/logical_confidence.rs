//! Skeleton for projection-aware logical-confidence scoring.
//!
//! `logical_confidence` is the future answer to "how much does this
//! projection preserve the *reasoning* structure of the corpus?" — the
//! degree to which logical relations between concepts (entailment,
//! taxonomy, contradiction) survive the embedding → sphere mapping.
//!
//! This module is a deliberately empty placeholder so that downstream
//! tests and pipeline code can consume the API today and the underlying
//! metric can be filled in later without churning callers. The current
//! implementation always returns `None`; the projection-comparison test
//! prints `n/a` in the score column until a real metric ships.
//!
//! Replace [`UnimplementedLogicalConfidence`] with a concrete scorer
//! when ready, or wire a different impl in via the [`LogicalConfidence`]
//! trait.

use sphereql_core::SphericalPoint;

/// Returns a scalar confidence in `[0, 1]` describing how well the
/// projected layout preserves the reasoning structure of the corpus, or
/// `None` if the implementation cannot answer for the given inputs (or,
/// in the case of [`UnimplementedLogicalConfidence`], at all).
pub trait LogicalConfidence {
    fn score(&self, projected: &[SphericalPoint], categories: &[String]) -> Option<f64>;
}

/// Default placeholder. Always returns `None`.
///
/// Lets callers reserve the score column / API surface today; swap in a
/// real implementation later without changing call sites.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnimplementedLogicalConfidence;

impl LogicalConfidence for UnimplementedLogicalConfidence {
    fn score(&self, _projected: &[SphericalPoint], _categories: &[String]) -> Option<f64> {
        None
    }
}
