//! Extended corpus: ~5,000+ concepts sourced from OpenAlex Topics.
//!
//! A 6×+ expansion of the hand-crafted 775-concept [`build_corpus`],
//! covering the same 31 academic categories with the same 128-axis
//! sparse-feature format. Concepts are derived from the public OpenAlex
//! Topics + Subfields taxonomy (CC0) by `tools/generate_extended.py`,
//! supplemented by hand-authored gap-fill entries for thin categories.
//! Every category in the extended corpus has at least 50 concepts, and
//! 95%+ of concepts are bridges (activate axes from two or more domain
//! groups), making it a good benchmark for cross-domain retrieval.
//!
//! Two public entry points:
//!
//! - [`build_extended_corpus`] returns only the OpenAlex-derived concepts.
//! - [`build_full_corpus`] returns the concatenation of [`build_corpus`]
//!   (775 hand-crafted) and [`build_extended_corpus`] in that order.
//!
//! # Memory warning
//!
//! Both functions leak per-concept label + category strings to
//! `&'static str` via `Box::leak`. Call at most once per process. See
//! [`load_extended_corpus`](crate::loader::load_extended_corpus) for
//! the full warning.
//!
//! [`build_corpus`]: crate::build_corpus

use crate::concept::Concept;
use crate::corpus::build_corpus;
use crate::loader::load_extended_corpus;

/// Build the extended corpus from the embedded OpenAlex-derived JSON.
///
/// Returns ~5,000+ concepts across 31 categories. See the module
/// docs for the memory warning — call at most once per process.
pub fn build_extended_corpus() -> Vec<Concept> {
    load_extended_corpus()
}

/// Build the full corpus: hand-crafted 775 concepts followed by the
/// extended OpenAlex-derived concepts.
///
/// Order is stable: the first 775 entries are exactly
/// [`build_corpus`](crate::build_corpus)'s output, then the extended
/// concepts in their JSON file order. Useful when downstream code wants
/// every available concept under one embedding pass.
pub fn build_full_corpus() -> Vec<Concept> {
    let mut full = build_corpus();
    full.extend(build_extended_corpus());
    full
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// Domain-axis ranges mirrored from `tools/mappings.py`. Indices
    /// 107..128 are cross-cutting and intentionally excluded from the
    /// bridge metric.
    const DOMAIN_RANGES: &[std::ops::Range<usize>] = &[
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

    const EXPECTED_CATEGORIES: &[&str] = &[
        "anthropology",
        "architecture",
        "astronomy",
        "biology",
        "chemistry",
        "computer_science",
        "culinary_arts",
        "data_science",
        "earth_science",
        "economics",
        "education",
        "engineering",
        "environmental_science",
        "film_studies",
        "history",
        "law",
        "linguistics",
        "literature",
        "mathematics",
        "medicine",
        "music",
        "nanotechnology",
        "neuroscience",
        "performing_arts",
        "philosophy",
        "physics",
        "political_science",
        "psychology",
        "religion",
        "sociology",
        "visual_arts",
    ];

    #[test]
    fn extended_corpus_loads_successfully() {
        let corpus = build_extended_corpus();
        assert!(
            corpus.len() >= 4000,
            "expected ≥4000 extended concepts, got {}",
            corpus.len()
        );
    }

    #[test]
    fn extended_corpus_covers_all_31_categories() {
        let corpus = build_extended_corpus();
        let cats: HashSet<&str> = corpus.iter().map(|c| c.category).collect();
        assert_eq!(cats.len(), 31, "expected 31 categories, got {}", cats.len());
        for expected in EXPECTED_CATEGORIES {
            assert!(cats.contains(expected), "missing category: {expected}");
        }
    }

    #[test]
    fn every_category_has_at_least_50_concepts() {
        let corpus = build_extended_corpus();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for c in &corpus {
            *counts.entry(c.category).or_insert(0) += 1;
        }
        for (cat, n) in &counts {
            assert!(*n >= 50, "category {cat} has only {n} concepts (<50)");
        }
    }

    #[test]
    fn labels_are_unique() {
        let corpus = build_extended_corpus();
        let labels: HashSet<&str> = corpus.iter().map(|c| c.label).collect();
        assert_eq!(
            labels.len(),
            corpus.len(),
            "found {} duplicate labels",
            corpus.len() - labels.len()
        );
    }

    #[test]
    fn features_are_valid() {
        let corpus = build_extended_corpus();
        for c in &corpus {
            assert!(
                (4..=8).contains(&c.features.len()),
                "concept {} has {} features, want 4..=8",
                c.label,
                c.features.len()
            );

            let mut seen_axes: HashSet<usize> = HashSet::new();
            for (axis, weight) in &c.features {
                assert!(*axis < 128, "concept {} has axis {} ≥ 128", c.label, axis);
                assert!(
                    *weight >= 0.19 && *weight <= 1.01,
                    "concept {} has weight {} outside [0.2, 1.0]",
                    c.label,
                    weight
                );
                assert!(
                    seen_axes.insert(*axis),
                    "concept {} has duplicate axis {}",
                    c.label,
                    axis
                );
            }
        }
    }

    #[test]
    fn bridge_ratio_is_high() {
        let corpus = build_extended_corpus();
        let mut bridges = 0;
        for c in &corpus {
            let mut groups: HashSet<usize> = HashSet::new();
            for (axis, _) in &c.features {
                for (idx, range) in DOMAIN_RANGES.iter().enumerate() {
                    if range.contains(axis) {
                        groups.insert(idx);
                        break;
                    }
                }
            }
            if groups.len() >= 2 {
                bridges += 1;
            }
        }
        let ratio = bridges as f64 / corpus.len() as f64;
        assert!(
            ratio >= 0.75,
            "bridge ratio {:.3} below 0.75 threshold",
            ratio
        );
    }

    #[test]
    fn full_corpus_includes_both() {
        let base = build_corpus();
        let extended = build_extended_corpus();
        let full = build_full_corpus();
        assert_eq!(full.len(), base.len() + extended.len());
        assert_eq!(full[0].label, base[0].label);
    }

    #[test]
    fn mean_features_per_concept_in_range() {
        let corpus = build_extended_corpus();
        let total: usize = corpus.iter().map(|c| c.features.len()).sum();
        let mean = total as f64 / corpus.len() as f64;
        assert!(
            (4.5..=7.0).contains(&mean),
            "mean features/concept {mean:.2} outside [4.5, 7.0]"
        );
    }

    #[test]
    fn all_128_axes_are_used() {
        let corpus = build_extended_corpus();
        let mut used: HashSet<usize> = HashSet::new();
        for c in &corpus {
            for (axis, _) in &c.features {
                used.insert(*axis);
            }
        }
        for axis in 0..128 {
            assert!(used.contains(&axis), "axis {axis} never used");
        }
    }
}

#[cfg(test)]
mod phase2_tests {
    use super::*;
    use crate::derived::bridge_degree;

    /// `>=90%` of concepts must carry a non-neutral `quality` — proves
    /// the generator actually populated Phase 2 fields and the loader
    /// read them through correctly.
    #[test]
    fn signals_populated_in_committed_corpus() {
        let corpus = build_extended_corpus();
        let neutral = corpus
            .iter()
            .filter(|c| (c.quality - Concept::NEUTRAL_QUALITY).abs() < 1e-9)
            .count();
        let pct_neutral = neutral as f64 / corpus.len() as f64;
        assert!(
            pct_neutral < 0.10,
            "expected <10% concepts at neutral default quality, got {:.2}%",
            pct_neutral * 100.0
        );
    }

    /// Stored `bridge_degree` must match recomputed value from `features`
    /// for the first 100 concepts. Catches drift between generator and
    /// `derived.rs` Rust port.
    #[test]
    fn bridge_degree_matches_authored_features() {
        let corpus = build_extended_corpus();
        for c in corpus.iter().take(100) {
            let computed = bridge_degree(&c.features);
            assert_eq!(
                c.bridge_degree, computed,
                "bridge_degree mismatch on {}: stored={}, computed={}",
                c.label, c.bridge_degree, computed
            );
        }
    }

    /// All five Phase 2 signals must be in their documented ranges across
    /// the entire committed corpus.
    #[test]
    fn signals_are_in_documented_ranges() {
        let corpus = build_extended_corpus();
        for c in &corpus {
            assert!(
                (0.0..=1.0).contains(&c.quality),
                "{} quality out of range: {}",
                c.label,
                c.quality
            );
            assert!(
                (0.0..=1.0).contains(&c.axis_coherence),
                "{} axis_coherence out of range: {}",
                c.label,
                c.axis_coherence
            );
            assert!(
                c.bridge_degree <= 30,
                "{} bridge_degree > 30: {}",
                c.label,
                c.bridge_degree
            );
            assert!(
                (0.0..=1.0).contains(&c.source_confidence),
                "{} source_confidence out of range: {}",
                c.label,
                c.source_confidence
            );
            assert!(
                (0.0..=1.0).contains(&c.home_affinity),
                "{} home_affinity out of range: {}",
                c.label,
                c.home_affinity
            );
        }
    }
}
