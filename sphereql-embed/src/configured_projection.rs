//! [`ConfiguredProjection`] — a single concrete type over all supported
//! outer-sphere projection families.
//!
//! The pipeline, the spatial index, and the category layer all want one
//! concrete `Projection` type — not a trait object. This enum unifies
//! [`PcaProjection`], [`KernelPcaProjection`], and
//! [`LaplacianEigenmapProjection`] so the pipeline can dispatch uniformly
//! while each trial of the auto-tuner can swap in a different family
//! without touching generics.
//!
//! Adding a new projection family = one variant here + one match arm in
//! the `Projection` impl and the inherent helpers.

use sphereql_core::SphericalPoint;

use crate::config::ProjectionKind;
use crate::kernel_pca::KernelPcaProjection;
use crate::laplacian::LaplacianEigenmapProjection;
use crate::projection::{PcaProjection, Projection};
use crate::types::{Embedding, ProjectedPoint};

/// A projection chosen at pipeline build time.
///
/// Implements [`Projection`] directly so `EmbeddingIndex<ConfiguredProjection>`,
/// `CategoryLayer::build_with_config`, and every other `Projection`-generic
/// API continues to work without changes.
#[derive(Clone)]
pub enum ConfiguredProjection {
    Pca(PcaProjection),
    KernelPca(KernelPcaProjection),
    Laplacian(LaplacianEigenmapProjection),
}

impl std::fmt::Debug for ConfiguredProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pca(_) => write!(f, "ConfiguredProjection::Pca"),
            Self::KernelPca(_) => write!(f, "ConfiguredProjection::KernelPca"),
            Self::Laplacian(_) => write!(f, "ConfiguredProjection::Laplacian"),
        }
    }
}

impl Projection for ConfiguredProjection {
    fn project(&self, embedding: &Embedding) -> SphericalPoint {
        match self {
            Self::Pca(p) => p.project(embedding),
            Self::KernelPca(p) => p.project(embedding),
            Self::Laplacian(p) => p.project(embedding),
        }
    }

    fn project_rich(&self, embedding: &Embedding) -> ProjectedPoint {
        match self {
            Self::Pca(p) => p.project_rich(embedding),
            Self::KernelPca(p) => p.project_rich(embedding),
            Self::Laplacian(p) => p.project_rich(embedding),
        }
    }

    fn dimensionality(&self) -> usize {
        match self {
            Self::Pca(p) => p.dimensionality(),
            Self::KernelPca(p) => p.dimensionality(),
            Self::Laplacian(p) => p.dimensionality(),
        }
    }
}

impl ConfiguredProjection {
    /// Which projection family is active.
    pub fn kind(&self) -> ProjectionKind {
        match self {
            Self::Pca(_) => ProjectionKind::Pca,
            Self::KernelPca(_) => ProjectionKind::KernelPca,
            Self::Laplacian(_) => ProjectionKind::LaplacianEigenmap,
        }
    }

    /// Scalar projection-quality proxy, analogous to PCA's explained
    /// variance ratio. For non-PCA variants this returns the kind's
    /// native quality metric (Kernel PCA EVR, Laplacian connectivity
    /// ratio) — all bounded in `[0, 1]` so downstream EVR-adaptive
    /// thresholds stay well-defined.
    pub fn explained_variance_ratio(&self) -> f64 {
        match self {
            Self::Pca(p) => p.explained_variance_ratio(),
            Self::KernelPca(p) => p.explained_variance_ratio(),
            Self::Laplacian(p) => p.explained_variance_ratio(),
        }
    }

    /// Borrow the inner [`PcaProjection`] if that is the active variant.
    pub fn as_pca(&self) -> Option<&PcaProjection> {
        match self {
            Self::Pca(p) => Some(p),
            _ => None,
        }
    }

    /// Borrow the inner [`KernelPcaProjection`] if that is the active variant.
    pub fn as_kernel_pca(&self) -> Option<&KernelPcaProjection> {
        match self {
            Self::KernelPca(p) => Some(p),
            _ => None,
        }
    }

    /// Borrow the inner [`LaplacianEigenmapProjection`] if that is the
    /// active variant.
    pub fn as_laplacian(&self) -> Option<&LaplacianEigenmapProjection> {
        match self {
            Self::Laplacian(p) => Some(p),
            _ => None,
        }
    }
}

impl From<PcaProjection> for ConfiguredProjection {
    fn from(p: PcaProjection) -> Self {
        Self::Pca(p)
    }
}

impl From<KernelPcaProjection> for ConfiguredProjection {
    fn from(p: KernelPcaProjection) -> Self {
        Self::KernelPca(p)
    }
}

impl From<LaplacianEigenmapProjection> for ConfiguredProjection {
    fn from(p: LaplacianEigenmapProjection) -> Self {
        Self::Laplacian(p)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RadialStrategy;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    fn toy_corpus() -> Vec<Embedding> {
        (0..8)
            .map(|i| {
                let t = i as f64;
                emb(&[1.0 + t * 0.01, 0.5 - t * 0.01, 0.2, 0.05, 0.03])
            })
            .collect()
    }

    #[test]
    fn pca_variant_dispatches() {
        let corpus = toy_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let cp: ConfiguredProjection = pca.into();
        assert_eq!(cp.kind(), ProjectionKind::Pca);
        assert_eq!(cp.dimensionality(), 5);
        let sp = cp.project(&corpus[0]);
        assert!((sp.r - 1.0).abs() < 1e-9);
        assert!(cp.as_pca().is_some());
        assert!(cp.as_kernel_pca().is_none());
        assert!(cp.as_laplacian().is_none());
    }

    #[test]
    fn kernel_pca_variant_dispatches() {
        let corpus = toy_corpus();
        let kpca = KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let cp: ConfiguredProjection = kpca.into();
        assert_eq!(cp.kind(), ProjectionKind::KernelPca);
        assert_eq!(cp.dimensionality(), 5);
        assert!(cp.as_kernel_pca().is_some());
        assert!(cp.as_pca().is_none());
    }

    #[test]
    fn laplacian_variant_dispatches() {
        // Laplacian needs ≥4 embeddings and at least one active axis per
        // point — the toy corpus above qualifies.
        let corpus = toy_corpus();
        let lap = LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let cp: ConfiguredProjection = lap.into();
        assert_eq!(cp.kind(), ProjectionKind::LaplacianEigenmap);
        assert_eq!(cp.dimensionality(), 5);
        assert!(cp.as_laplacian().is_some());
        assert!(cp.as_pca().is_none());
    }

    #[test]
    fn explained_variance_ratio_in_range_for_every_variant() {
        let corpus = toy_corpus();
        let pca: ConfiguredProjection =
            PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).into();
        let kpca: ConfiguredProjection =
            KernelPcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).into();
        let lap: ConfiguredProjection =
            LaplacianEigenmapProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).into();
        for cp in &[pca, kpca, lap] {
            let r = cp.explained_variance_ratio();
            assert!(r >= 0.0 && r <= 1.0, "{:?}: {r}", cp);
        }
    }

    #[test]
    fn debug_formats_kind_not_inner() {
        let corpus = toy_corpus();
        let pca: ConfiguredProjection =
            PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0)).into();
        assert_eq!(format!("{:?}", pca), "ConfiguredProjection::Pca");
    }
}
