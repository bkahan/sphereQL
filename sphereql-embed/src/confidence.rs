//! Unified confidence scoring for query results.
//!
//! Combines three independent quality signals into a single score:
//! - **EVR**: global projection quality
//! - **Certainty**: per-point projection fidelity
//! - **Gap confidence**: proximity to category caps (void = low confidence)

use sphereql_core::SphericalPoint;

/// Confidence assessment for a single query result or point on S².
#[derive(Debug, Clone, Copy)]
pub struct QualitySignal {
    pub evr: f64,
    pub certainty: f64,
    pub void_distance: f64,
    pub gap_confidence: f64,
    /// evr × certainty × gap_confidence
    pub combined: f64,
    pub level: ConfidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidenceLevel {
    Unreliable,
    Low,
    Moderate,
    High,
}

impl std::fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "HIGH"),
            Self::Moderate => write!(f, "MODERATE"),
            Self::Low => write!(f, "LOW"),
            Self::Unreliable => write!(f, "UNRELIABLE"),
        }
    }
}

impl QualitySignal {
    pub fn compute(evr: f64, certainty: f64, void_dist: f64, sharpness: f64) -> Self {
        let gap_confidence = 1.0 / (1.0 + (sharpness * void_dist).exp());
        let combined = evr * certainty * gap_confidence;
        let level = classify(combined);
        Self {
            evr,
            certainty,
            void_distance: void_dist,
            gap_confidence,
            combined,
            level,
        }
    }

    /// Simplified: no void distance available (e.g., raw k-NN results).
    pub fn from_certainty(evr: f64, certainty: f64) -> Self {
        let gap_confidence = certainty.sqrt().max(0.01);
        let combined = evr * certainty * gap_confidence;
        let level = classify(combined);
        Self {
            evr,
            certainty,
            void_distance: 0.0,
            gap_confidence,
            combined,
            level,
        }
    }

    pub fn passes_threshold(&self, min_combined: f64) -> bool {
        self.combined >= min_combined
    }
}

fn classify(combined: f64) -> ConfidenceLevel {
    if combined > 0.10 {
        ConfidenceLevel::High
    } else if combined > 0.03 {
        ConfidenceLevel::Moderate
    } else if combined > 0.005 {
        ConfidenceLevel::Low
    } else {
        ConfidenceLevel::Unreliable
    }
}

/// Full quality signal using centroids and half-angles for void distance.
pub fn point_quality(
    evr: f64,
    certainty: f64,
    position: &SphericalPoint,
    centroids: &[SphericalPoint],
    half_angles: &[f64],
    sharpness: f64,
) -> QualitySignal {
    let void_dist = sphereql_core::spatial::void_distance(position, centroids, half_angles);
    QualitySignal::compute(evr, certainty, void_dist, sharpness)
}

/// Configuration for quality-based filtering.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Minimum per-point certainty. Default: 0.0 (no filtering).
    pub min_certainty: f64,
    /// Minimum combined confidence. Default: 0.0 (no filtering).
    pub min_combined: f64,
    /// Sigmoid sharpness for gap confidence. Default: 5.0.
    pub gap_sharpness: f64,
    /// EVR threshold for projection warnings. Default: 0.35.
    pub warn_below_evr: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_certainty: 0.0,
            min_combined: 0.0,
            gap_sharpness: 5.0,
            warn_below_evr: 0.35,
        }
    }
}

/// A structured warning about projection quality.
#[derive(Debug, Clone)]
pub struct ProjectionWarning {
    pub message: String,
    pub evr: f64,
    pub severity: WarningSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSeverity {
    Info,
    Warning,
    Critical,
}

impl ProjectionWarning {
    pub fn from_evr(evr: f64, threshold: f64) -> Option<Self> {
        if evr >= threshold {
            return None;
        }
        let (message, severity) = if evr < 0.15 {
            (
                format!(
                    "EVR={:.1}% \u{2014} projection captures very little variance. \
                 Category routing and bridges are unreliable. Use inner spheres.",
                    evr * 100.0
                ),
                WarningSeverity::Critical,
            )
        } else if evr < 0.25 {
            (
                format!(
                    "EVR={:.1}% \u{2014} projection is lossy. Bridge counts may be inflated. \
                 Certainty-weighted results recommended.",
                    evr * 100.0
                ),
                WarningSeverity::Warning,
            )
        } else {
            (
                format!(
                    "EVR={:.1}% \u{2014} below recommended {:.0}%. Results usable with caution.",
                    evr * 100.0,
                    threshold * 100.0
                ),
                WarningSeverity::Info,
            )
        };
        Some(Self {
            message,
            evr,
            severity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_confidence_all_good() {
        let sig = QualitySignal::compute(0.6, 0.8, -0.5, 5.0);
        assert_eq!(sig.level, ConfidenceLevel::High);
        assert!(sig.combined > 0.10);
    }

    #[test]
    fn low_certainty_kills_confidence() {
        let sig = QualitySignal::compute(0.6, 0.007, -0.5, 5.0);
        assert!(sig.combined < 0.01);
    }

    #[test]
    fn void_kills_confidence() {
        let sig = QualitySignal::compute(0.6, 0.8, 1.0, 5.0);
        assert!(sig.gap_confidence < 0.01);
        assert_eq!(sig.level, ConfidenceLevel::Unreliable);
    }

    #[test]
    fn low_evr_reduces_confidence() {
        let good = QualitySignal::compute(0.6, 0.5, -0.3, 5.0);
        let bad = QualitySignal::compute(0.19, 0.5, -0.3, 5.0);
        assert!(good.combined > bad.combined);
    }

    #[test]
    fn from_certainty_fallback() {
        let sig = QualitySignal::from_certainty(0.5, 0.3);
        assert!(sig.combined > 0.0);
        assert_eq!(sig.void_distance, 0.0);
    }

    #[test]
    fn threshold_filtering() {
        let sig = QualitySignal::compute(0.19, 0.26, -0.2, 5.0);
        assert!(sig.passes_threshold(0.0));
        assert!(sig.passes_threshold(0.01));
    }

    #[test]
    fn warning_at_low_evr() {
        let w = ProjectionWarning::from_evr(0.19, 0.35).unwrap();
        assert_eq!(w.severity, WarningSeverity::Warning);
    }

    #[test]
    fn no_warning_at_high_evr() {
        assert!(ProjectionWarning::from_evr(0.60, 0.35).is_none());
    }

    #[test]
    fn critical_at_very_low_evr() {
        let w = ProjectionWarning::from_evr(0.10, 0.35).unwrap();
        assert_eq!(w.severity, WarningSeverity::Critical);
    }

    #[test]
    fn confidence_levels_ordered() {
        assert!(ConfidenceLevel::High > ConfidenceLevel::Moderate);
        assert!(ConfidenceLevel::Moderate > ConfidenceLevel::Low);
        assert!(ConfidenceLevel::Low > ConfidenceLevel::Unreliable);
    }
}
