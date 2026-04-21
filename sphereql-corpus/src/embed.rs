/// Embedding dimensionality.
pub const DIM: usize = 128;

/// Deterministic pseudo-random embedding from sparse features.
///
/// Fills a 128-dim vector with the given feature weights, then adds
/// low-amplitude noise seeded by `seed` for realism.
pub fn embed(features: &[(usize, f64)], seed: u64) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for &(axis, val) in features {
        v[axis] = val;
    }
    let mut s = seed;
    for x in &mut v {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
    }
    v
}
