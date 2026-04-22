/// Embedding dimensionality.
pub const DIM: usize = 128;

/// Noise amplitude used by [`embed`] — the default regime for the
/// built-in 775-concept corpus.
pub const DEFAULT_NOISE_AMPLITUDE: f64 = 0.04;

/// Deterministic pseudo-random embedding from sparse features.
///
/// Fills a 128-dim vector with the given feature weights, then adds
/// low-amplitude noise (±0.02) seeded by `seed` for realism.
pub fn embed(features: &[(usize, f64)], seed: u64) -> Vec<f64> {
    embed_with_noise(features, seed, DEFAULT_NOISE_AMPLITUDE)
}

/// [`embed`] with a configurable noise amplitude.
///
/// Each dim gets uniform noise in `[-amplitude/2, +amplitude/2]` from a
/// SplitMix64 stream. Use this to synthesize stress-test corpora that
/// bracket the default regime (e.g. `amplitude=0.2` for a signal-to-noise
/// ratio roughly 10× harsher than the built-in corpus).
pub fn embed_with_noise(features: &[(usize, f64)], seed: u64, amplitude: f64) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for &(axis, val) in features {
        v[axis] = val;
    }
    let mut s = seed;
    for x in &mut v {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * amplitude;
    }
    v
}
