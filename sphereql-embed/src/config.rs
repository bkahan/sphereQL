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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// UMAP-on-sphere hyperparameters (only consulted if that
    /// projection is selected).
    pub umap: UmapConfig,
    /// Spatial quality Monte Carlo sample counts.
    pub spatial: SpatialConfig,
    /// Minimum number of items a category must have to participate in
    /// category-level analysis (bridges, domain groups, spatial quality,
    /// Voronoi tessellation). Categories below this threshold are excluded
    /// from the enrichment layer but their items remain projected, indexed,
    /// and queryable on the sphere.
    ///
    /// Default 1 (no filtering — every category participates).
    /// Set to 5–10 for corpora with many singleton categories.
    #[serde(default = "default_min_category_size")]
    pub min_category_size: usize,
}

fn default_min_category_size() -> usize {
    1
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            projection_kind: ProjectionKind::default(),
            inner_sphere: InnerSphereConfig::default(),
            bridges: BridgeConfig::default(),
            routing: RoutingConfig::default(),
            laplacian: LaplacianConfig::default(),
            umap: UmapConfig::default(),
            spatial: SpatialConfig::default(),
            min_category_size: default_min_category_size(),
        }
    }
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
    /// UMAP-on-sphere via Adam in the tangent bundle of S². PCA warm
    /// start, kNN attractive + uniform-negative repulsive, optional
    /// supervised category term. Preferred when angular ordering on the
    /// sphere matters more than raw variance preservation, and when a
    /// modest fit cost (O(n²·epochs) for the kNN graph + iterations) is
    /// acceptable.
    UmapSphere,
}

impl ProjectionKind {
    /// Short stable name for logs and tuner reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pca => "pca",
            Self::KernelPca => "kernel_pca",
            Self::LaplacianEigenmap => "laplacian_eigenmap",
            Self::UmapSphere => "umap_sphere",
        }
    }

    /// All supported kinds, in a stable order.
    pub fn all() -> &'static [ProjectionKind] {
        &[
            ProjectionKind::Pca,
            ProjectionKind::KernelPca,
            ProjectionKind::LaplacianEigenmap,
            ProjectionKind::UmapSphere,
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
    /// Percentile of the observed territorial factor distribution below
    /// which a bridge is classified as `OverlapArtifact` rather than
    /// `Genuine` or `Weak`.  0.3 = the bottom 30 % of bridge pairs by
    /// territorial separation are labeled artifacts.  Expressed as a
    /// percentile so that dense corpora (where all exclusivities collapse
    /// toward zero) do not classify every bridge as an artifact.
    pub overlap_artifact_territorial: f64,
    /// Quantile of the home-affinity distribution that sets the
    /// genuine-bridge floor. For each member item, "home affinity" is
    /// the cosine similarity between the item's embedding and its own
    /// category's centroid. A bridge is classified `Genuine` when
    /// `min(affinity_to_source, affinity_to_target)` exceeds the
    /// quantile-q of those home affinities; otherwise `Weak`.
    ///
    /// Why a quantile and not an absolute cosine: home affinity scale
    /// varies with the projection layout. After stratified PCA spreads
    /// imbalanced corpora, home affinities can drop into the 0.3–0.6
    /// band where a fixed 0.5 cosine floor labels almost every cross-
    /// domain item `Weak`. A quantile-based floor adapts to the
    /// corpus's own affinity scale: tight corpora get a strict floor,
    /// spread ones get a permissive one, without per-corpus tuning.
    ///
    /// Smaller q = stricter (only bridges matching the strongest
    /// home affinities qualify). Larger q = more permissive. Default
    /// 0.25: a bridge is `Genuine` if it has at least as much
    /// affinity to both sides as the bottom-25% of items have to
    /// their own home category.
    pub balanced_affinity_quantile: f64,
    /// EVR below which bridge classification is unreliable. When the
    /// outer projection's EVR is below this threshold, all bridges
    /// are labeled `Weak` (honest uncertainty) rather than attempting
    /// territorial-factor-based classification — which collapses to
    /// 100% `OverlapArtifact` when caps overlap everywhere on a
    /// low-EVR projection, flattening the tuner landscape. Default
    /// 0.20.
    pub min_evr_for_classification: f64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            threshold_base: 0.5,
            threshold_evr_penalty: 0.4,
            overlap_artifact_territorial: 0.3,
            balanced_affinity_quantile: 0.25,
            min_evr_for_classification: 0.20,
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
    /// Distance-ratio gate for the default `nearest()` path. A query
    /// drills into the nearest group's inner sphere when
    /// `d_to_nearest / d_to_second_nearest < group_routing_alpha`. A
    /// smaller α is stricter (only routes when one group is clearly
    /// closer). Default `0.8` matches the routing interview decision;
    /// set to `0.0` to disable the default-route behavior entirely
    /// (falls back to outer-sphere k-NN).
    pub group_routing_alpha: f64,
    /// EVR below which `hierarchical_nearest` historically routed
    /// through domain groups instead of the outer sphere.
    ///
    /// Retained for backward-compatibility and debugging — the default
    /// `nearest()` path now uses [`Self::group_routing_alpha`] instead.
    /// `hierarchical_nearest()` still consults this for its EVR-gated
    /// branch.
    pub low_evr_threshold: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            num_domain_groups: 5,
            group_routing_alpha: 0.8,
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

// ── UMAP-on-sphere ─────────────────────────────────────────────────────

/// Hyperparameters for [`UmapSphereProjection`](crate::umap::UmapSphereProjection).
///
/// These are the tunable knobs exposed to the auto-tuner. Non-tunable
/// constants (`learning_rate`, `negative_sample_rate`) stay at their
/// canonical UMAP defaults inside [`fit_projection_for_config`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct UmapConfig {
    /// k in the kNN graph (attractive term). Higher = more global structure.
    pub n_neighbors: usize,
    /// Adam optimization epochs. ~200 for corpora < 10k, ~400 for 50k+.
    pub n_epochs: usize,
    /// Weight on the category supervision term. 0.0 = unsupervised UMAP.
    /// Positive values pull same-category items together and push
    /// different-category items apart. 1.0–3.0 is typical.
    pub category_weight: f64,
    /// PRNG seed for negative sampling and tie-breaking.
    pub seed: u64,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_epochs: 200,
            category_weight: 1.5,
            seed: 0xA1B2_C3D4,
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
        assert!((c.bridges.balanced_affinity_quantile - 0.25).abs() < 1e-12);
        assert!((c.bridges.min_evr_for_classification - 0.20).abs() < 1e-12);
        assert_eq!(c.routing.num_domain_groups, 5);
        assert!((c.routing.low_evr_threshold - 0.35).abs() < 1e-12);
        assert_eq!(c.laplacian.k_neighbors, 15);
        assert!((c.laplacian.active_threshold - 0.05).abs() < 1e-12);
        assert_eq!(c.umap.n_neighbors, 15);
        assert_eq!(c.umap.n_epochs, 200);
        assert!((c.umap.category_weight - 1.5).abs() < 1e-12);
        assert_eq!(c.spatial.coverage_samples, 100_000);
        assert_eq!(c.spatial.exclusivity_samples, 30_000);
        assert_eq!(c.spatial.voronoi_samples, 100_000);
        assert_eq!(c.min_category_size, 1);
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
        assert_eq!(ProjectionKind::UmapSphere.name(), "umap_sphere");
        assert_eq!(ProjectionKind::all().len(), 4);
    }
}
