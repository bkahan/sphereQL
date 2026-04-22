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
#[derive(Debug, Clone)]
pub struct PipelineConfig {
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

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            inner_sphere: InnerSphereConfig::default(),
            bridges: BridgeConfig::default(),
            routing: RoutingConfig::default(),
            laplacian: LaplacianConfig::default(),
            spatial: SpatialConfig::default(),
        }
    }
}

// ── Inner-sphere ───────────────────────────────────────────────────────

/// Thresholds governing when a category gets its own inner projection.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct SpatialConfig {
    pub coverage_samples: usize,
    pub exclusivity_samples: usize,
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
}
