//! Test-corpus concept type with pre-computed quality signals.
//!
//! Phase 2 added five signal fields (`quality`, `axis_coherence`,
//! `bridge_degree`, `source_confidence`, `home_affinity`) so the
//! sphereql-embed pipeline can weight, filter, and prune without
//! re-deriving them on every run.
//!
//! Old corpora produced before Phase 2 will load with these fields at
//! their neutral defaults (see [`Concept::neutral_signals`]) — pipeline
//! behavior is unchanged.

/// A concept in the test corpus.
///
/// `features` is a sparse vector of `(axis_index, weight)` pairs with
/// `0 <= axis_index < 128` and `0.2 <= weight <= 1.0`.
///
/// The five signal fields are pre-computed by the generator and are
/// purely advisory: they let downstream consumers weight or prune
/// concepts without scanning the full feature vector. They are
/// derivable from `(category, features)` plus generation-time source
/// metadata; see [`derived`](crate::derived).
///
/// ## Memory note
///
/// `label` and `category` are `&'static str`. For the hand-crafted
/// corpus this is natural (string literals). For Parquet-loaded
/// corpora the loader uses `Box::leak` to promote heap-allocated
/// strings to `'static`. This is intentional: the corpus is loaded
/// once per process and lives for the entire program lifetime, so the
/// leak is bounded and the simplicity is worth it. Migrating
/// `Concept` to owned `String` fields is tracked for a future phase.
#[derive(Debug, Clone)]
pub struct Concept {
    pub label: &'static str,
    pub category: &'static str,
    pub features: Vec<(usize, f64)>,
    /// Composite per-concept confidence in `[0, 1]`. Default `1.0` for
    /// pre-Phase-2 corpora (trust everything).
    pub quality: f64,
    /// Concentration of feature mass on category-primary axes, in
    /// `[0, 1]`. `1.0` means all mass on primaries. Default `1.0`.
    pub axis_coherence: f64,
    /// Number of distinct domain ranges activated by `features`.
    /// `0..=30` (domain ranges 0..106). Cross-cutting axes 107..128
    /// are excluded. Default `1` (single-domain).
    pub bridge_degree: u8,
    /// Source-derived trust in `[0, 1]`. For OpenAlex,
    /// `log10(1 + works_count) / 6` clamped. For gap_fill, `0.5`.
    /// Default `1.0` for pre-Phase-2.
    pub source_confidence: f64,
    /// Cosine of feature mass against the category-primary axis mass
    /// vector, in `[0, 1]`. Default `1.0`.
    pub home_affinity: f64,
}

impl Concept {
    /// Neutral defaults that make a pre-Phase-2 corpus behave
    /// identically to its post-Phase-2 form under any weighting scheme
    /// that respects these fields.
    pub const NEUTRAL_QUALITY: f64 = 1.0;
    pub const NEUTRAL_AXIS_COHERENCE: f64 = 1.0;
    pub const NEUTRAL_BRIDGE_DEGREE: u8 = 1;
    pub const NEUTRAL_SOURCE_CONFIDENCE: f64 = 1.0;
    pub const NEUTRAL_HOME_AFFINITY: f64 = 1.0;

    /// Returns `(quality, axis_coherence, bridge_degree, source_confidence, home_affinity)`
    /// at their neutral defaults.
    pub const fn neutral_signals() -> (f64, f64, u8, f64, f64) {
        (
            Self::NEUTRAL_QUALITY,
            Self::NEUTRAL_AXIS_COHERENCE,
            Self::NEUTRAL_BRIDGE_DEGREE,
            Self::NEUTRAL_SOURCE_CONFIDENCE,
            Self::NEUTRAL_HOME_AFFINITY,
        )
    }
}

impl Default for Concept {
    fn default() -> Self {
        Self {
            label: "",
            category: "",
            features: Vec::new(),
            quality: Self::NEUTRAL_QUALITY,
            axis_coherence: Self::NEUTRAL_AXIS_COHERENCE,
            bridge_degree: Self::NEUTRAL_BRIDGE_DEGREE,
            source_confidence: Self::NEUTRAL_SOURCE_CONFIDENCE,
            home_affinity: Self::NEUTRAL_HOME_AFFINITY,
        }
    }
}
