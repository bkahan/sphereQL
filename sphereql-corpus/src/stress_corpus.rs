//! Stress-test corpus: low signal-to-noise on a small per-category signature.
//!
//! Built to discriminate between projection families on a regime the
//! built-in 775-concept corpus doesn't exercise. Key characteristics:
//!
//! - 10 synthetic categories, 30 concepts each (300 total).
//! - Each concept has exactly **2 authored signal axes** from its
//!   category's 2-axis signature (out of 128 total dimensions). The
//!   signal axes carry amplitude ~1.0 and ~0.6.
//! - Noise amplitude on all inactive axes = `0.2`, 5× the built-in
//!   corpus's `0.04`. At `active_threshold = 0.05` (the default filter)
//!   the noise clears the threshold on most non-signal axes, so the
//!   downstream active-axis count per item is high. The regime this
//!   corpus really tests is **low signal-to-noise in a small authored
//!   signature**, not literal sparsity at the filter level. That's the
//!   setup where variance-maximizing projections (PCA) get pulled
//!   toward the noise floor while connectivity-preserving projections
//!   (Laplacian) can still recover the authored signature through
//!   shared-axis co-activation.
//!
//! Usage:
//!
//! ```no_run
//! use sphereql_corpus::{build_stress_corpus, embed_with_noise};
//!
//! let corpus = build_stress_corpus();
//! let embeddings: Vec<Vec<f64>> = corpus
//!     .iter()
//!     .enumerate()
//!     .map(|(i, c)| embed_with_noise(&c.features, 9000 + i as u64, 0.2))
//!     .collect();
//! ```
//!
//! Intentionally small (300 concepts) to keep tuner runs fast — use
//! this as a controlled A/B probe, not a benchmark proxy.

use crate::concept::Concept;

/// Noise amplitude we recommend for stress-corpus embeddings. Callers
/// pass this to [`embed_with_noise`](crate::embed_with_noise).
pub const STRESS_NOISE_AMPLITUDE: f64 = 0.2;

/// Number of categories in the stress corpus.
pub const STRESS_CATEGORIES: usize = 10;

/// Concepts per category in the stress corpus.
pub const STRESS_CONCEPTS_PER_CATEGORY: usize = 30;

/// Build the stress-test corpus.
///
/// # Memory warning
///
/// Labels and category names are leaked to `&'static str` via
/// `Box::leak` so they fit [`Concept`]'s signature. Each call leaks
/// `STRESS_CATEGORIES * STRESS_CONCEPTS_PER_CATEGORY + STRESS_CATEGORIES`
/// = 310 strings (a few KB total). **Leaks DO accumulate across calls** —
/// each invocation leaks a fresh set. Call at most once per process in
/// production code. Tests and examples are fine: the per-process total
/// is bounded by the number of calls × 310 strings, which stays in the
/// low MB even across a full test suite.
pub fn build_stress_corpus() -> Vec<Concept> {
    let mut out = Vec::with_capacity(STRESS_CATEGORIES * STRESS_CONCEPTS_PER_CATEGORY);

    for c in 0..STRESS_CATEGORIES {
        // Disjoint 2-axis signature per category at (2c, 2c+1).
        let axis_a = 2 * c;
        let axis_b = 2 * c + 1;
        let category_name: &'static str = leak_string(format!("stress_cat_{:02}", c));

        for i in 0..STRESS_CONCEPTS_PER_CATEGORY {
            // Smooth variation around the signature so within-category
            // embeddings aren't identical; adds real local structure the
            // tuner and metrics can latch onto.
            let t = i as f64 / STRESS_CONCEPTS_PER_CATEGORY as f64;
            let theta = t * std::f64::consts::TAU;
            let weight_a = 1.0 + 0.15 * theta.sin();
            let weight_b = 0.6 + 0.15 * theta.cos();

            let label: &'static str = leak_string(format!("stress_cat_{:02}_item_{:02}", c, i));

            out.push(Concept {
                label,
                category: category_name,
                features: vec![(axis_a, weight_a), (axis_b, weight_b)],
            });
        }
    }

    out
}

/// Leak an owned String to `&'static str`. Called once per unique label
/// at corpus-build time; the total number of leaked strings is bounded
/// by `STRESS_CATEGORIES * STRESS_CONCEPTS_PER_CATEGORY + STRESS_CATEGORIES`.
fn leak_string(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn stress_corpus_has_expected_size() {
        let corpus = build_stress_corpus();
        assert_eq!(
            corpus.len(),
            STRESS_CATEGORIES * STRESS_CONCEPTS_PER_CATEGORY
        );
    }

    #[test]
    fn stress_corpus_has_expected_category_count() {
        let corpus = build_stress_corpus();
        let cats: HashSet<&str> = corpus.iter().map(|c| c.category).collect();
        assert_eq!(cats.len(), STRESS_CATEGORIES);
    }

    #[test]
    fn each_concept_has_two_active_axes() {
        let corpus = build_stress_corpus();
        for c in &corpus {
            assert_eq!(c.features.len(), 2);
        }
    }

    #[test]
    fn category_signature_axes_are_disjoint() {
        // Each category should claim its own pair of axes; no two
        // categories should share an active-axis index.
        let corpus = build_stress_corpus();
        let mut per_category: std::collections::HashMap<&str, HashSet<usize>> =
            std::collections::HashMap::new();
        for c in &corpus {
            let entry = per_category.entry(c.category).or_default();
            for (axis, _) in &c.features {
                entry.insert(*axis);
            }
        }
        // Collect all axes per category; pairwise intersections must be empty.
        let cats: Vec<&&str> = per_category.keys().collect();
        for i in 0..cats.len() {
            for j in (i + 1)..cats.len() {
                let a = &per_category[cats[i]];
                let b = &per_category[cats[j]];
                assert!(
                    a.is_disjoint(b),
                    "categories {} and {} share axes",
                    cats[i],
                    cats[j]
                );
            }
        }
    }

    #[test]
    fn labels_are_unique() {
        let corpus = build_stress_corpus();
        let labels: HashSet<&str> = corpus.iter().map(|c| c.label).collect();
        assert_eq!(labels.len(), corpus.len());
    }
}
