//! Configuration surface for the SphereQL pipeline.
//!
//! Every tunable constant that governs projection, bridge detection,
//! inner-sphere gating, domain-group routing, and spatial-quality
//! Monte Carlo sample counts lives here. This is the first-class knob
//! inventory that future auto-tuning and meta-learning passes optimize
//! over.
//!
//! The [`PipelineConfig::default`] values reproduce the historical
//! hardcoded constants; the pipeline accepts any overriding config.

// ── Top-level ──────────────────────────────────────────────────────────

/// All tunable parameters for a SphereQL pipeline build.
///
/// Every field is a sub-config grouped by area. [`Self::default`] returns
/// the values the crate shipped with before the config surface existed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct PipelineConfig {
    /// Outer-sphere projection family.
    pub projection_kind: ProjectionKind,
    /// Inner-sphere gating thresholds.
    pub inner_sphere: InnerSphereConfig,
    /// Bridge detection and classification.
    pub bridges: BridgeConfig,
    /// Hierarchical domain-group routing.
    pub routing: RoutingConfig,
    /// Laplacian eigenmap hyperparameters (only consulted if that
    /// projection is selected).
    pub laplacian: LaplacianConfig,
    /// Spatial quality Monte Carlo sample counts.
    pub spatial: SpatialConfig,
}

// ── Projection kind ────────────────────────────────────────────────────

/// Which projection family the pipeline uses for the outer sphere.
///
/// A first-class tunable axis:
/// [`SearchSpace::projection_kinds`](crate::tuner::SearchSpace::projection_kinds)
/// enumerates the families the auto-tuner sweeps, and
/// [`CorpusFeatures`](crate::corpus_features::CorpusFeatures) →
/// [`PipelineConfig`] meta-models can map corpus profiles onto the
/// kind that works best.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub enum ProjectionKind {
    /// Linear PCA — fast, variance-maximizing. Good default for dense,
    /// low-noise embeddings.
    #[default]
    Pca,
    /// Kernel PCA with a Gaussian (RBF) kernel. Captures nonlinear
    /// manifold structure at O(n²) fit cost.
    KernelPca,
    /// Laplacian eigenmap over a Jaccard-similarity graph of active
    /// axes. Connectivity-preserving; preferred when signal lives in
    /// the co-activation structure of a sparse embedding rather than in
    /// coordinate variance (the typical failure mode of PCA on 128-dim
    /// noise-heavy corpora).
    LaplacianEigenmap,
}

impl ProjectionKind {
    /// Short stable name for logs and tuner reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pca => "pca",
            Self::KernelPca => "kernel_pca",
            Self::LaplacianEigenmap => "laplacian_eigenmap",
        }
    }

    /// All supported kinds, in a stable order.
    pub fn all() -> &'static [ProjectionKind] {
        &[
            ProjectionKind::Pca,
            ProjectionKind::KernelPca,
            ProjectionKind::LaplacianEigenmap,
        ]
    }
}

// ── Inner-sphere ───────────────────────────────────────────────────────

/// Thresholds governing when a category gets its own inner projection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InnerSphereConfig {
    /// Minimum member count for a category to be considered.
    pub min_size: usize,
    /// Minimum EVR improvement (inner − global_subset) to justify building
    /// an inner sphere at all.
    pub min_evr_improvement: f64,
    /// Minimum member count at which kernel PCA is attempted.
    pub kernel_pca_min_size: usize,
    /// Minimum EVR improvement of kernel PCA over linear PCA to prefer it.
    pub min_kernel_improvement: f64,
}

impl Default for InnerSphereConfig {
    fn default() -> Self {
        Self {
            min_size: 20,
            min_evr_improvement: 0.10,
            kernel_pca_min_size: 80,
            min_kernel_improvement: 0.05,
        }
    }
}

// ── Bridges ────────────────────────────────────────────────────────────

/// Parameters controlling bridge detection and classification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BridgeConfig {
    /// Constant term in the EVR-adaptive bridge threshold
    /// `threshold = threshold_base + (1 − evr)² · threshold_evr_penalty`.
    pub threshold_base: f64,
    /// EVR-penalty coefficient in the bridge threshold formula.
    pub threshold_evr_penalty: f64,
    /// Territorial factor below which a bridge is classified as an
    /// `OverlapArtifact` rather than `Genuine` or `Weak`.
    pub overlap_artifact_territorial: f64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            threshold_base: 0.5,
            threshold_evr_penalty: 0.4,
            overlap_artifact_territorial: 0.3,
        }
    }
}

impl BridgeConfig {
    /// EVR-adaptive bridge threshold.
    ///
    /// Higher EVR → looser threshold (projection is more trustworthy).
    /// At EVR=0.19: 0.5 + 0.81² × 0.4 = 0.76 (strict).
    /// At EVR=0.90: 0.5 + 0.01 × 0.4 = 0.50 (essentially unchanged).
    pub fn evr_adaptive_threshold(&self, evr: f64) -> f64 {
        self.threshold_base + (1.0 - evr).powi(2) * self.threshold_evr_penalty
    }
}

// ── Hierarchical routing ───────────────────────────────────────────────

/// Parameters for hierarchical domain-group routing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// Number of domain groups detected at build time by
    /// [`detect_domain_groups`](crate::domain_groups::detect_domain_groups).
    pub num_domain_groups: usize,
    /// EVR below which `hierarchical_nearest` routes through domain
    /// groups and inner spheres instead of the outer sphere.
    pub low_evr_threshold: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            num_domain_groups: 5,
            low_evr_threshold: 0.35,
        }
    }
}

// ── Laplacian eigenmap ─────────────────────────────────────────────────

/// Graph-construction parameters for [`LaplacianEigenmapProjection`](crate::laplacian::LaplacianEigenmapProjection).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LaplacianConfig {
    /// k in the k-NN graph sparsification step.
    pub k_neighbors: usize,
    /// Absolute-weight cutoff below which an axis is treated as noise.
    pub active_threshold: f64,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 15,
            active_threshold: 0.05,
        }
    }
}

// ── Spatial quality ────────────────────────────────────────────────────

/// Monte Carlo sample counts for [`SpatialQuality::compute`](crate::spatial_quality::SpatialQuality::compute).
///
/// These run once at build time. Higher = more precise but slower.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SpatialConfig {
    /// Samples used to estimate what fraction of S² is covered by any
    /// category's cap. Higher = tighter coverage estimate. Default
    /// `100_000` → ~50ms at 31 categories.
    pub coverage_samples: usize,
    /// Samples used per category to estimate its cap exclusivity (the
    /// fraction of its cap not overlapped by any other category).
    /// Runs `n_categories` times so cost scales linearly with C.
    /// Default `30_000` per category.
    pub exclusivity_samples: usize,
    /// Samples used to estimate the spherical Voronoi tessellation over
    /// category centroids. Higher = tighter per-cell area estimates.
    /// Default `100_000` → ~100ms at 31 centroids.
    pub voronoi_samples: usize,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            coverage_samples: 100_000,
            exclusivity_samples: 30_000,
            voronoi_samples: 100_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_legacy_constants() {
        let c = PipelineConfig::default();
        assert_eq!(c.projection_kind, ProjectionKind::Pca);
        assert_eq!(c.inner_sphere.min_size, 20);
        assert_eq!(c.inner_sphere.kernel_pca_min_size, 80);
        assert!((c.inner_sphere.min_evr_improvement - 0.10).abs() < 1e-12);
        assert!((c.inner_sphere.min_kernel_improvement - 0.05).abs() < 1e-12);
        assert!((c.bridges.threshold_base - 0.5).abs() < 1e-12);
        assert!((c.bridges.threshold_evr_penalty - 0.4).abs() < 1e-12);
        assert!((c.bridges.overlap_artifact_territorial - 0.3).abs() < 1e-12);
        assert_eq!(c.routing.num_domain_groups, 5);
        assert!((c.routing.low_evr_threshold - 0.35).abs() < 1e-12);
        assert_eq!(c.laplacian.k_neighbors, 15);
        assert!((c.laplacian.active_threshold - 0.05).abs() < 1e-12);
        assert_eq!(c.spatial.coverage_samples, 100_000);
        assert_eq!(c.spatial.exclusivity_samples, 30_000);
        assert_eq!(c.spatial.voronoi_samples, 100_000);
    }

    #[test]
    fn evr_adaptive_threshold_monotone_in_evr() {
        let b = BridgeConfig::default();
        let low = b.evr_adaptive_threshold(0.15);
        let mid = b.evr_adaptive_threshold(0.50);
        let high = b.evr_adaptive_threshold(0.90);
        // Higher EVR → smaller threshold
        assert!(low > mid);
        assert!(mid > high);
        assert!((high - 0.5).abs() < 0.05);
    }

    #[test]
    fn config_is_clone() {
        let a = PipelineConfig::default();
        let b = a.clone();
        assert_eq!(a.inner_sphere.min_size, b.inner_sphere.min_size);
    }

    #[test]
    fn projection_kind_name_and_all_stable() {
        assert_eq!(ProjectionKind::Pca.name(), "pca");
        assert_eq!(ProjectionKind::KernelPca.name(), "kernel_pca");
        assert_eq!(
            ProjectionKind::LaplacianEigenmap.name(),
            "laplacian_eigenmap"
        );
        assert_eq!(ProjectionKind::all().len(), 3);
    }
}
