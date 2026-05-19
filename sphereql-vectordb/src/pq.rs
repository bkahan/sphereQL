//! Product Quantization sidecar for the full embedding.
//!
//! Spherical projection compresses a `D`-dim embedding to `(r, θ, φ)`,
//! which is great for coarse filtering but loses fidelity. PQ keeps a
//! compressed copy of the *full* embedding alongside the spherical one
//! and uses it as a re-ranker (or as a standalone exact-search escape
//! hatch when EVR is poor).
//!
//! Standard PQ:
//! - Split each embedding into `M` equal-length subvectors of `D/M` dims.
//! - For each subspace, run k-means with `K = 2^bits_per_code` centroids.
//!   With the default `bits_per_code = 8`, `K = 256`, so each item costs
//!   exactly `M` bytes plus the codebook overhead amortized across the
//!   corpus.
//! - At query time, build an `M × K` lookup table from the query's per-
//!   subspace squared-distance to each centroid. The asymmetric distance
//!   to a stored item is then a sum of `M` table lookups — fast and
//!   nearly identical in ranking quality to the exact distance for
//!   typical embedding distributions.
//!
//! This module ships:
//! - [`PqConfig`] — knob bag.
//! - [`PqCodebook`] — trained centroids (one [K, D/M] matrix per subspace).
//! - [`PqIndex`] — corpus codes + standalone search and rerank entry points.
//! - [`PqStore`] — trait for swappable backends. The in-memory impl in
//!   this file is the default; an LMDB-backed impl lives behind the
//!   `pq-lmdb` feature in [`crate::pq_lmdb`].

use std::collections::HashMap;

/// Tunables for product quantization.
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// Number of subvectors. `D` (the embedding dimension) must be
    /// divisible by `m`. Default 8 — gives 96× compression on a 768-dim
    /// embedding when paired with `bits_per_code = 8`.
    pub m: usize,
    /// Bits per code, so each subspace gets `2^bits_per_code` centroids.
    /// Default 8 (256 centroids = one byte per subspace per item).
    /// Values above 8 break the `Vec<u8>` code layout this module uses
    /// and are rejected at training time.
    pub bits_per_code: u8,
    /// k-means iterations per subspace. 25 is plenty for typical
    /// embeddings — convergence is monitored, so this is an upper bound.
    pub kmeans_iters: usize,
    /// Convergence tolerance: stop early when the relative inertia
    /// improvement between two consecutive iterations drops below this.
    pub kmeans_tol: f64,
    /// PRNG seed used for k-means++ centroid initialization.
    pub seed: u64,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            m: 8,
            bits_per_code: 8,
            kmeans_iters: 25,
            kmeans_tol: 1e-4,
            seed: 0x5_u64.wrapping_mul(0x9E37_79B9),
        }
    }
}

impl PqConfig {
    fn k(&self) -> usize {
        1usize << self.bits_per_code as usize
    }
}

/// Errors surfaced by training and search.
#[derive(Debug, thiserror::Error)]
pub enum PqError {
    #[error("PQ training requires at least one embedding")]
    EmptyCorpus,
    #[error("embedding dim {dim} not divisible by m={m}")]
    NonDivisibleDim { dim: usize, m: usize },
    #[error("PqConfig.m must be > 0")]
    InvalidM,
    #[error("vector dim {got} != codebook dim {expected}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("embedding {index} has dim {got}, expected {expected}")]
    InconsistentDim {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("bits_per_code {0} > 8 is unsupported (codes are u8)")]
    BitsTooLarge(u8),
    #[error("not enough training points ({n}) for K={k} centroids")]
    NotEnoughPoints { n: usize, k: usize },
    #[error("PqStore: {0}")]
    Store(String),
}

/// Trained codebook. Vector layout: `centroids[m][k * sub_dim ..]` is
/// the k-th centroid of subspace m, stored as a flat row-major
/// `Vec<f32>` for cache-friendly iteration during LUT build.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    pub m: usize,
    pub k: usize,
    pub sub_dim: usize,
    /// Per-subspace flat centroid arrays. Length `m`, each of size
    /// `k * sub_dim`.
    pub centroids: Vec<Vec<f32>>,
}

impl PqCodebook {
    /// Train on the corpus. Each embedding is `D`-dim; `D` must be
    /// divisible by `config.m`. Internally splits into subvectors and
    /// fits one k-means per subspace using `kmeans_pp_init` followed by
    /// Lloyd iterations.
    pub fn train(corpus: &[Vec<f32>], config: &PqConfig) -> Result<Self, PqError> {
        if config.bits_per_code > 8 {
            return Err(PqError::BitsTooLarge(config.bits_per_code));
        }
        if config.m == 0 {
            return Err(PqError::InvalidM);
        }
        if corpus.is_empty() {
            return Err(PqError::EmptyCorpus);
        }
        let dim = corpus[0].len();
        if dim == 0 || !dim.is_multiple_of(config.m) {
            return Err(PqError::NonDivisibleDim { dim, m: config.m });
        }
        for (i, e) in corpus.iter().enumerate() {
            if e.len() != dim {
                return Err(PqError::InconsistentDim {
                    index: i,
                    expected: dim,
                    got: e.len(),
                });
            }
        }
        let k = config.k();
        if corpus.len() < k {
            return Err(PqError::NotEnoughPoints { n: corpus.len(), k });
        }

        let sub_dim = dim / config.m;
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(config.m);
        for m_idx in 0..config.m {
            let sub: Vec<&[f32]> = corpus
                .iter()
                .map(|e| &e[m_idx * sub_dim..(m_idx + 1) * sub_dim])
                .collect();
            let cents = kmeans_subspace(&sub, sub_dim, k, config, m_idx);
            centroids.push(cents);
        }

        Ok(Self {
            m: config.m,
            k,
            sub_dim,
            centroids,
        })
    }

    /// Encode an embedding to `m` codes (one per subspace). The code is
    /// the index of the nearest centroid in that subspace.
    pub fn encode(&self, embedding: &[f32]) -> Vec<u8> {
        let expected = self.m * self.sub_dim;
        debug_assert_eq!(
            embedding.len(),
            expected,
            "Pq::encode: embedding length {} does not match codebook dimensionality {}",
            embedding.len(),
            expected
        );
        let mut out = vec![0u8; self.m];
        for m_idx in 0..self.m {
            let sub = &embedding[m_idx * self.sub_dim..(m_idx + 1) * self.sub_dim];
            let cents = &self.centroids[m_idx];
            let mut best_d = f32::INFINITY;
            let mut best_k = 0u32;
            for ki in 0..self.k {
                let c = &cents[ki * self.sub_dim..(ki + 1) * self.sub_dim];
                let d = sq_dist_f32(sub, c);
                if d < best_d {
                    best_d = d;
                    best_k = ki as u32;
                }
            }
            out[m_idx] = best_k as u8;
        }
        out
    }

    /// Build the asymmetric query → centroids LUT: `lut[m][k]` is the
    /// squared distance from the query's subspace-m to centroid k.
    pub fn asymmetric_lut(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let expected = self.m * self.sub_dim;
        debug_assert_eq!(
            query.len(),
            expected,
            "Pq::asymmetric_lut: query length {} does not match codebook dimensionality {}",
            query.len(),
            expected
        );
        let mut lut = Vec::with_capacity(self.m);
        for m_idx in 0..self.m {
            let q = &query[m_idx * self.sub_dim..(m_idx + 1) * self.sub_dim];
            let cents = &self.centroids[m_idx];
            let mut row = vec![0f32; self.k];
            for ki in 0..self.k {
                let c = &cents[ki * self.sub_dim..(ki + 1) * self.sub_dim];
                row[ki] = sq_dist_f32(q, c);
            }
            lut.push(row);
        }
        lut
    }
}

/// One coded entry in the index.
#[derive(Debug, Clone)]
pub struct PqEntry {
    pub id: String,
    pub codes: Vec<u8>,
}

/// Pluggable backend for a `PqIndex`. All implementations must preserve
/// insertion order so that integer indices used by callers stay stable.
pub trait PqStore: Send + Sync {
    fn insert(&mut self, id: &str, codes: &[u8]) -> Result<(), PqError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Iterate all (id, codes) pairs. Implementations should produce them
    /// in insertion order — search ranks rely on stable indexing.
    fn for_each(&self, f: &mut dyn FnMut(&str, &[u8]));
    /// Optional fast random access by id.
    fn get(&self, id: &str) -> Option<Vec<u8>>;
}

/// In-memory `PqStore`. Default backend; zero extra dependencies.
#[derive(Debug, Default, Clone)]
pub struct InMemoryPqStore {
    entries: Vec<PqEntry>,
    by_id: HashMap<String, usize>,
}

impl InMemoryPqStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PqStore for InMemoryPqStore {
    fn insert(&mut self, id: &str, codes: &[u8]) -> Result<(), PqError> {
        if let Some(&idx) = self.by_id.get(id) {
            self.entries[idx].codes = codes.to_vec();
            return Ok(());
        }
        let idx = self.entries.len();
        self.entries.push(PqEntry {
            id: id.to_string(),
            codes: codes.to_vec(),
        });
        self.by_id.insert(id.to_string(), idx);
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn for_each(&self, f: &mut dyn FnMut(&str, &[u8])) {
        for e in &self.entries {
            f(&e.id, &e.codes);
        }
    }

    fn get(&self, id: &str) -> Option<Vec<u8>> {
        self.by_id
            .get(id)
            .map(|&idx| self.entries[idx].codes.clone())
    }
}

/// Top-level PQ index. Owns the codebook and a backing store of
/// per-item codes. Search returns ids ranked by the asymmetric distance
/// from the query to each item's reconstructed embedding.
pub struct PqIndex {
    pub codebook: PqCodebook,
    pub store: Box<dyn PqStore>,
}

impl PqIndex {
    /// Build an index from a trained codebook + an empty store.
    pub fn new(codebook: PqCodebook, store: Box<dyn PqStore>) -> Self {
        Self { codebook, store }
    }

    /// Convenience: train a codebook and pre-load the same corpus into
    /// an in-memory store.
    pub fn build(ids: &[String], corpus: &[Vec<f32>], config: &PqConfig) -> Result<Self, PqError> {
        if ids.len() != corpus.len() {
            return Err(PqError::Store(format!(
                "ids len {} != corpus len {}",
                ids.len(),
                corpus.len()
            )));
        }
        let codebook = PqCodebook::train(corpus, config)?;
        let mut store: Box<dyn PqStore> = Box::new(InMemoryPqStore::new());
        for (id, e) in ids.iter().zip(corpus.iter()) {
            let codes = codebook.encode(e);
            store.insert(id, &codes)?;
        }
        Ok(Self { codebook, store })
    }

    /// Insert (or overwrite) one item.
    pub fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), PqError> {
        let expected = self.codebook.m * self.codebook.sub_dim;
        if embedding.len() != expected {
            return Err(PqError::DimensionMismatch {
                expected,
                got: embedding.len(),
            });
        }
        let codes = self.codebook.encode(embedding);
        self.store.insert(id, &codes)
    }

    /// Brute-force PQ search over every coded item. `O(N · M)` per query
    /// with the per-query LUT amortizing the expensive distance work.
    /// Returns an empty `Vec` if `query.len()` does not match the
    /// codebook dimensionality.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if k == 0 || query.len() != self.codebook.m * self.codebook.sub_dim {
            return Vec::new();
        }
        let lut = self.codebook.asymmetric_lut(query);
        let m = self.codebook.m;
        let mut heap: Vec<(String, f32)> = Vec::with_capacity(k + 1);
        self.store.for_each(&mut |id, codes| {
            // Defense against corrupted store entries: skip items whose
            // code length doesn't match the codebook. The store should
            // never contain these (insert validates), but a stale on-disk
            // sidecar from an older codebook would otherwise panic here.
            if codes.len() != m {
                debug_assert_eq!(codes.len(), m, "store returned malformed code");
                return;
            }
            let mut d = 0f32;
            for mi in 0..m {
                d += lut[mi][codes[mi] as usize];
            }
            // Insertion-sort top-k. Faster than a full sort for the
            // common k ≪ N case (k = 10 on N = 1e6).
            if heap.len() < k {
                heap.push((id.to_string(), d));
                heap.sort_by(|a, b| a.1.total_cmp(&b.1));
            } else if d < heap[k - 1].1 {
                heap[k - 1] = (id.to_string(), d);
                heap.sort_by(|a, b| a.1.total_cmp(&b.1));
            }
        });
        heap
    }

    /// Re-rank a candidate id list against the query using PQ distances.
    /// Returns a slice of size `min(k, candidates.len())` ranked
    /// ascending by asymmetric distance. Items missing from the store
    /// are dropped silently — they're either un-indexed or stale.
    /// Returns an empty `Vec` if `query.len()` does not match the
    /// codebook dimensionality.
    pub fn rerank(&self, query: &[f32], candidates: &[String], k: usize) -> Vec<(String, f32)> {
        if query.len() != self.codebook.m * self.codebook.sub_dim {
            return Vec::new();
        }
        let lut = self.codebook.asymmetric_lut(query);
        let m = self.codebook.m;
        let mut scored: Vec<(String, f32)> = Vec::with_capacity(candidates.len());
        for id in candidates {
            let Some(codes) = self.store.get(id) else {
                continue;
            };
            if codes.len() != m {
                debug_assert_eq!(codes.len(), m, "store returned malformed code");
                continue;
            }
            let mut d = 0f32;
            for mi in 0..m {
                d += lut[mi][codes[mi] as usize];
            }
            scored.push((id.clone(), d));
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }
}

// ── Internals ─────────────────────────────────────────────────────────

fn sq_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// k-means++ init then Lloyd iterations.
///
/// `points[i]` is a slice of `sub_dim` floats. Returns a flat
/// `k * sub_dim` centroid buffer.
fn kmeans_subspace(
    points: &[&[f32]],
    sub_dim: usize,
    k: usize,
    config: &PqConfig,
    subspace_idx: usize,
) -> Vec<f32> {
    let n = points.len();
    let mut rng = SplitMix64::new(config.seed.wrapping_add(0x1234_5678 ^ subspace_idx as u64));

    // ── k-means++ initialization ──────────────────────────────────────
    let mut centroids = vec![0f32; k * sub_dim];
    let first = (rng.next_u64() as usize) % n;
    centroids[..sub_dim].copy_from_slice(points[first]);

    let mut min_dist = vec![f32::INFINITY; n];
    for i in 0..n {
        min_dist[i] = sq_dist_f32(points[i], &centroids[..sub_dim]);
    }

    for c in 1..k {
        let total: f32 = min_dist.iter().sum();
        let pick = if total > 0.0 {
            let target = rng.next_f32() * total;
            let mut acc = 0f32;
            let mut chosen = n - 1;
            for (i, &d) in min_dist.iter().enumerate() {
                acc += d;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            (rng.next_u64() as usize) % n
        };
        centroids[c * sub_dim..(c + 1) * sub_dim].copy_from_slice(points[pick]);
        for i in 0..n {
            let d = sq_dist_f32(points[i], &centroids[c * sub_dim..(c + 1) * sub_dim]);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    // ── Lloyd iterations ──────────────────────────────────────────────
    let mut assign = vec![0u32; n];
    let mut prev_inertia = f64::INFINITY;
    for _ in 0..config.kmeans_iters {
        // Assign
        let mut inertia = 0f64;
        for i in 0..n {
            let mut best_d = f32::INFINITY;
            let mut best_k = 0u32;
            for ki in 0..k {
                let d = sq_dist_f32(points[i], &centroids[ki * sub_dim..(ki + 1) * sub_dim]);
                if d < best_d {
                    best_d = d;
                    best_k = ki as u32;
                }
            }
            assign[i] = best_k;
            inertia += best_d as f64;
        }

        // Update
        let mut sums = vec![0f32; k * sub_dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let ki = assign[i] as usize;
            counts[ki] += 1;
            let dst = &mut sums[ki * sub_dim..(ki + 1) * sub_dim];
            let src = points[i];
            for d in 0..sub_dim {
                dst[d] += src[d];
            }
        }
        for ki in 0..k {
            if counts[ki] == 0 {
                continue;
            }
            let inv = 1.0 / counts[ki] as f32;
            for d in 0..sub_dim {
                centroids[ki * sub_dim + d] = sums[ki * sub_dim + d] * inv;
            }
        }

        // Convergence
        if prev_inertia.is_finite() {
            let denom = prev_inertia.max(1e-12);
            let rel = (prev_inertia - inertia).abs() / denom;
            if rel < config.kmeans_tol {
                break;
            }
        }
        prev_inertia = inertia;
    }

    centroids
}

// ── Deterministic PRNG (mirrors sphereql-embed's SplitMix64). ─────────
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_corpus(n: usize, dim: usize) -> Vec<Vec<f32>> {
        // Two clusters, well-separated, with mild jitter so PQ is
        // forced to actually use multiple centroids.
        let mut out = Vec::with_capacity(n);
        let mut rng = SplitMix64::new(42);
        for i in 0..n {
            let mut v = vec![0f32; dim];
            let cluster = i % 4;
            for (d, slot) in v.iter_mut().enumerate() {
                let base = if (d / (dim / 4)) == cluster { 1.0 } else { 0.0 };
                *slot = base + (rng.next_f32() - 0.5) * 0.05;
            }
            out.push(v);
        }
        out
    }

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("id-{i:04}")).collect()
    }

    #[test]
    fn train_and_encode_roundtrip() {
        let corpus = make_corpus(300, 16);
        let config = PqConfig {
            m: 4,
            bits_per_code: 4, // K=16 is plenty for 300 toy points
            kmeans_iters: 10,
            kmeans_tol: 1e-4,
            seed: 1,
        };
        let cb = PqCodebook::train(&corpus, &config).unwrap();
        assert_eq!(cb.m, 4);
        assert_eq!(cb.k, 16);
        assert_eq!(cb.sub_dim, 4);

        let codes = cb.encode(&corpus[0]);
        assert_eq!(codes.len(), 4);
        // All codes must be valid centroid indices.
        for c in codes {
            assert!((c as usize) < cb.k);
        }
    }

    #[test]
    fn search_returns_query_itself_first() {
        let corpus = make_corpus(200, 16);
        let id_list = ids(corpus.len());
        let config = PqConfig {
            m: 4,
            bits_per_code: 4,
            kmeans_iters: 10,
            kmeans_tol: 1e-4,
            seed: 7,
        };
        let index = PqIndex::build(&id_list, &corpus, &config).unwrap();
        let results = index.search(&corpus[5], 3);
        assert_eq!(results.len(), 3);
        // The query is in the corpus → it ranks first (its own codes
        // give it a self-distance of zero up to centroid quantization).
        assert_eq!(results[0].0, "id-0005");
    }

    #[test]
    fn rerank_orders_existing_candidates() {
        let corpus = make_corpus(60, 16);
        let id_list = ids(corpus.len());
        let config = PqConfig {
            m: 4,
            bits_per_code: 4,
            kmeans_iters: 10,
            kmeans_tol: 1e-4,
            seed: 3,
        };
        let index = PqIndex::build(&id_list, &corpus, &config).unwrap();

        // Hand the rerank a deliberately mis-ordered candidate list.
        let candidates = vec!["id-0040".into(), "id-0001".into(), "id-0000".into()];
        let result = index.rerank(&corpus[0], &candidates, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, "id-0000"); // best match → query itself
        // Distances must be monotonically non-decreasing.
        assert!(result[0].1 <= result[1].1);
        assert!(result[1].1 <= result[2].1);
    }

    #[test]
    fn empty_corpus_errors() {
        let result = PqCodebook::train(&[], &PqConfig::default());
        assert!(matches!(result, Err(PqError::EmptyCorpus)));
    }

    #[test]
    fn non_divisible_dim_errors() {
        let bogus = vec![vec![1.0_f32; 17]; 50];
        let cfg = PqConfig {
            m: 4,
            bits_per_code: 4,
            ..Default::default()
        };
        let result = PqCodebook::train(&bogus, &cfg);
        assert!(matches!(result, Err(PqError::NonDivisibleDim { .. })));
    }

    #[test]
    fn bits_too_large_errors() {
        let cfg = PqConfig {
            bits_per_code: 9,
            ..Default::default()
        };
        let result = PqCodebook::train(&[vec![0f32; 8]], &cfg);
        assert!(matches!(result, Err(PqError::BitsTooLarge(9))));
    }

    #[test]
    fn store_overwrite_keeps_index_stable() {
        let mut store = InMemoryPqStore::new();
        store.insert("a", &[0, 1, 2]).unwrap();
        store.insert("b", &[3, 4, 5]).unwrap();
        store.insert("a", &[9, 9, 9]).unwrap(); // overwrite
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a").unwrap(), vec![9, 9, 9]);
        assert_eq!(store.get("b").unwrap(), vec![3, 4, 5]);
    }
}
