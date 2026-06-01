//! Pure functions that compute the five `Concept` signal fields.
//!
//! Called by the loader (when backfilling old JSON) and by the
//! generator (in Python via a tested-in-Rust reference impl).
//! Deterministic and side-effect-free.

use std::ops::Range;

/// Disjoint domain-axis ranges, mirroring `tools/mappings.py:589–597`
/// and `extended.rs::DOMAIN_RANGES`. Indices 107..128 are cross-cutting
/// and intentionally excluded from bridge detection.
pub const DOMAIN_RANGES: &[Range<usize>] = &[
    0..7,
    7..12,
    12..16,
    16..19,
    19..23,
    23..26,
    26..30,
    30..34,
    34..38,
    38..41,
    41..45,
    45..49,
    49..52,
    52..55,
    55..59,
    59..63,
    63..68,
    68..71,
    71..73,
    73..76,
    76..79,
    79..82,
    82..85,
    85..89,
    89..92,
    92..96,
    96..100,
    100..102,
    102..104,
    104..107,
];

/// Count distinct domain ranges activated by the feature axes.
/// Returns a value in `0..=DOMAIN_RANGES.len()` (i.e. `0..=30`).
pub fn bridge_degree(features: &[(usize, f64)]) -> u8 {
    let mut hit = [false; 30];
    for &(axis, _) in features {
        for (idx, range) in DOMAIN_RANGES.iter().enumerate() {
            if range.contains(&axis) {
                hit[idx] = true;
                break;
            }
        }
    }
    hit.iter().filter(|h| **h).count() as u8
}

/// Fraction of total feature mass placed on `primary_axes`, in `[0, 1]`.
///
/// `0.0` if `total_mass == 0` (empty feature set).
pub fn axis_coherence(features: &[(usize, f64)], primary_axes: &[usize]) -> f64 {
    let total: f64 = features.iter().map(|(_, w)| w.abs()).sum();
    if total == 0.0 {
        return 0.0;
    }
    let on_primary: f64 = features
        .iter()
        .filter(|(a, _)| primary_axes.contains(a))
        .map(|(_, w)| w.abs())
        .sum();
    (on_primary / total).clamp(0.0, 1.0)
}

/// Cosine of feature mass vector against a uniform mass over
/// `primary_axes`, in `[0, 1]`.
///
/// Used as `home_affinity`: does this concept "point home" to its
/// category's primary axes?
pub fn home_affinity(features: &[(usize, f64)], primary_axes: &[usize]) -> f64 {
    if features.is_empty() || primary_axes.is_empty() {
        return 0.0;
    }
    let mut a = vec![0.0_f64; 128];
    let mut b = vec![0.0_f64; 128];
    for &(axis, w) in features {
        if axis < 128 {
            a[axis] = w;
        }
    }
    for &axis in primary_axes {
        if axis < 128 {
            b[axis] = 1.0;
        }
    }
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(0.0, 1.0)
    }
}

/// Map a raw works-count (OpenAlex) to a confidence in `[0, 1]`:
/// `min(1, log10(1 + works_count) / 6)`. `works_count = 0 → 0.0`.
/// `works_count = 1_000_000 → 1.0`.
pub fn source_confidence_from_works(works_count: u64) -> f64 {
    ((1.0_f64 + works_count as f64).log10() / 6.0).clamp(0.0, 1.0)
}

/// Composite quality in `[0, 1]`. Linear blend, weights total 1.0.
///
/// `quality = 0.4*home_affinity + 0.3*axis_coherence + 0.2*source_confidence + 0.1*bridge_score`
/// where `bridge_score = min(1, bridge_degree / 3.0)`.
pub fn composite_quality(
    home_affinity: f64,
    axis_coherence: f64,
    source_confidence: f64,
    bridge_degree: u8,
) -> f64 {
    let bridge_score = (bridge_degree as f64 / 3.0).min(1.0);
    let q =
        0.4 * home_affinity + 0.3 * axis_coherence + 0.2 * source_confidence + 0.1 * bridge_score;
    q.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_degree_counts_distinct_domain_ranges() {
        let features = vec![(0, 1.0), (7, 0.5), (14, 0.5)];
        assert_eq!(bridge_degree(&features), 3);
        let features = vec![(0, 1.0), (115, 0.5)];
        assert_eq!(bridge_degree(&features), 1);
    }

    #[test]
    fn axis_coherence_full_when_all_on_primary() {
        let features = vec![(0, 1.0), (1, 0.5)];
        let primaries = vec![0, 1];
        assert!((axis_coherence(&features, &primaries) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn axis_coherence_zero_when_none_on_primary() {
        let features = vec![(50, 1.0)];
        let primaries = vec![0, 1];
        assert_eq!(axis_coherence(&features, &primaries), 0.0);
    }

    #[test]
    fn home_affinity_is_unit_when_features_exactly_match_primaries() {
        let features = vec![(0, 1.0), (1, 1.0)];
        let primaries = vec![0, 1];
        assert!((home_affinity(&features, &primaries) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn source_confidence_endpoints() {
        assert_eq!(source_confidence_from_works(0), 0.0);
        assert!(source_confidence_from_works(1_000_000) >= 1.0 - 1e-9);
    }

    #[test]
    fn composite_quality_within_bounds() {
        for ha in [0.0, 0.5, 1.0] {
            for ac in [0.0, 0.5, 1.0] {
                for sc in [0.0, 0.5, 1.0] {
                    for bd in 0..=10_u8 {
                        let q = composite_quality(ha, ac, sc, bd);
                        assert!((0.0..=1.0).contains(&q));
                    }
                }
            }
        }
    }
}
