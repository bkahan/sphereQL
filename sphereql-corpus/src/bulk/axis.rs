//! Deterministic hashed-claim axis extraction.
//!
//! Replaces the hand-curated `KEYWORD_TO_AXIS` map of the pre-bulk
//! corpus. Every claim `(predicate, object)` pair is hashed (FNV-1a
//! with a configurable seed) into one of `num_axes` buckets. Bucket
//! accumulators sum `log(1 + claim.weight)`, then are renormalized
//! so the maximum bucket lands at `1.0` and buckets below
//! `MIN_WEIGHT` are dropped.
//!
//! Determinism: the hash is FNV-1a, not SipHash or `DefaultHasher`,
//! so the same `(seed, predicate, object)` always produces the same
//! bucket across Rust versions, machines, and runs. That matters
//! because the corpus parquet on disk is a stable artifact — a
//! re-derived axis mapping must match what `bulk_ingest` wrote
//! months earlier.

use crate::bulk::BulkItem;

/// Lowest feature weight kept by [`HashedClaimAxisExtractor::extract`]
/// after per-item renormalization, matching the pre-bulk corpus's
/// `weight_floor`.
pub const MIN_WEIGHT: f64 = 0.2;

#[derive(Debug, Clone)]
pub struct HashedClaimAxisExtractor {
    num_axes: usize,
    seed: u64,
}

impl HashedClaimAxisExtractor {
    pub fn new(num_axes: usize, seed: u64) -> Self {
        assert!(num_axes > 0, "num_axes must be > 0");
        Self { num_axes, seed }
    }

    pub fn num_axes(&self) -> usize {
        self.num_axes
    }

    /// Return the item's sparse feature vector — a sorted-by-axis
    /// list of `(axis_index, weight)` pairs with weights in
    /// `[MIN_WEIGHT, 1.0]`. Items with zero claims (or whose claims
    /// all collide into the same near-zero bucket) yield an empty
    /// vector; callers should skip those rows.
    pub fn extract(&self, item: &BulkItem) -> Vec<(usize, f64)> {
        if item.claims.is_empty() {
            return Vec::new();
        }
        let mut buckets = vec![0.0_f64; self.num_axes];
        for claim in &item.claims {
            let w = claim.weight.max(0.0);
            if w <= 0.0 {
                continue;
            }
            let axis = self.bucket_of(&claim.predicate, &claim.object);
            buckets[axis] += (1.0 + w).ln();
        }
        let max = buckets.iter().cloned().fold(0.0_f64, f64::max);
        if max <= 0.0 {
            return Vec::new();
        }
        let mut out: Vec<(usize, f64)> = buckets
            .into_iter()
            .enumerate()
            .filter_map(|(i, raw)| {
                let w = raw / max;
                if w >= MIN_WEIGHT { Some((i, w)) } else { None }
            })
            .collect();
        // Already in axis order from the enumerate-filter chain, but
        // make the invariant explicit for callers and tests.
        out.sort_by_key(|(i, _)| *i);
        out
    }

    fn bucket_of(&self, predicate: &str, object: &str) -> usize {
        let mut h = self.seed ^ 0xcbf29ce484222325;
        for &b in predicate.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // Tab byte as predicate/object separator — neither field
        // should contain it in practice.
        h ^= 0x09;
        h = h.wrapping_mul(0x100000001b3);
        for &b in object.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % self.num_axes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bulk::Claim;

    fn item_with(claims: Vec<Claim>) -> BulkItem {
        BulkItem {
            external_id: "X1".into(),
            label: "x".into(),
            description: String::new(),
            claims,
            source_name: "test".into(),
            source_confidence: 1.0,
            category_hint: None,
            quality_hint: 0.5,
        }
    }

    #[test]
    fn empty_claims_yield_empty_features() {
        let ex = HashedClaimAxisExtractor::new(128, 0);
        assert!(ex.extract(&item_with(vec![])).is_empty());
    }

    #[test]
    fn hash_is_deterministic_across_calls() {
        let ex = HashedClaimAxisExtractor::new(128, 42);
        let it = item_with(vec![
            Claim::new("P31", "Q5", 1.0),
            Claim::new("P279", "Q35120", 1.0),
        ]);
        let a = ex.extract(&it);
        let b = ex.extract(&it);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn distinct_predicates_hit_distinct_buckets() {
        let ex = HashedClaimAxisExtractor::new(128, 0);
        let a = ex.bucket_of("P31", "Q5");
        let b = ex.bucket_of("P279", "Q5");
        // Almost-always distinct for a 128-bucket FNV; this is an
        // assertion about the chosen hash, not a probabilistic claim.
        assert_ne!(a, b);
    }

    #[test]
    fn output_weights_are_in_range_and_max_one() {
        let ex = HashedClaimAxisExtractor::new(64, 7);
        let it = item_with(vec![
            Claim::new("topic", "T1", 0.9),
            Claim::new("topic", "T2", 0.3),
            Claim::new("topic", "T3", 0.5),
            Claim::new("concept", "C1", 0.8),
        ]);
        let feats = ex.extract(&it);
        assert!(!feats.is_empty());
        let mut sorted = true;
        let mut prev = 0;
        for (i, w) in &feats {
            assert!((MIN_WEIGHT..=1.0).contains(w), "weight {w} out of range");
            if *i < prev {
                sorted = false;
                break;
            }
            prev = *i;
        }
        assert!(sorted, "features must be sorted by axis");
        let max = feats.iter().map(|(_, w)| *w).fold(0.0_f64, f64::max);
        assert!((max - 1.0).abs() < 1e-9, "max bucket should normalize to 1.0");
    }

    #[test]
    fn distribution_uses_many_buckets() {
        // Sanity: 200 distinct predicate-object pairs should land in
        // most of a 128-bucket space. We don't require uniformity, just
        // that we're not collapsing everything to one bucket.
        let ex = HashedClaimAxisExtractor::new(128, 99);
        let mut hits = vec![false; 128];
        for i in 0..200 {
            let b = ex.bucket_of("P31", &format!("Q{i}"));
            hits[b] = true;
        }
        let used = hits.iter().filter(|b| **b).count();
        assert!(used >= 80, "expected wide hash spread, got {used}/128");
    }
}
