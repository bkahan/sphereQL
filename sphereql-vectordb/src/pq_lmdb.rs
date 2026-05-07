//! LMDB-backed [`PqStore`] (heed wrapper).
//!
//! Uses two LMDB databases under one environment:
//! - `codes`: id (UTF-8) → codes (`Vec<u8>` of length `m`).
//! - `order`: insertion-order index (big-endian u32) → id, so iteration
//!   is stable across process restarts. The dual-index layout costs one
//!   extra write per insert and a few bytes per item; in exchange,
//!   `for_each` returns results in deterministic order without sorting,
//!   which matters because PqIndex search ranks rely on it.
//!
//! Behind the `pq-lmdb` feature so the crate's default build doesn't
//! pull in a C dependency.

use std::path::Path;

use heed::types::{Bytes, Str, U32};
use heed::{Database, Env, EnvOpenOptions};

use crate::pq::{PqError, PqStore};

/// LMDB-backed store. Open once, reuse across queries.
pub struct LmdbPqStore {
    env: Env,
    codes: Database<Str, Bytes>,
    order: Database<U32<heed::byteorder::BigEndian>, Str>,
    next_index: u32,
}

impl LmdbPqStore {
    /// Open or create the database under `path`. `map_size_bytes`
    /// caps the on-disk file (LMDB requires this up front); pick
    /// generously — the file is sparse on most filesystems.
    pub fn open<P: AsRef<Path>>(path: P, map_size_bytes: usize) -> Result<Self, PqError> {
        std::fs::create_dir_all(&path).map_err(|e| PqError::Store(e.to_string()))?;
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size_bytes)
                .max_dbs(2)
                .open(path)
        }
        .map_err(|e| PqError::Store(e.to_string()))?;

        let mut wtxn = env.write_txn().map_err(|e| PqError::Store(e.to_string()))?;
        let codes: Database<Str, Bytes> = env
            .create_database(&mut wtxn, Some("codes"))
            .map_err(|e| PqError::Store(e.to_string()))?;
        let order: Database<U32<heed::byteorder::BigEndian>, Str> = env
            .create_database(&mut wtxn, Some("order"))
            .map_err(|e| PqError::Store(e.to_string()))?;
        wtxn.commit().map_err(|e| PqError::Store(e.to_string()))?;

        // Seed `next_index` from the highest existing entry so reopens
        // resume the order monotonically.
        let rtxn = env.read_txn().map_err(|e| PqError::Store(e.to_string()))?;
        let mut next_index: u32 = 0;
        if let Some((max_idx, _)) = order
            .last(&rtxn)
            .map_err(|e| PqError::Store(e.to_string()))?
        {
            next_index = max_idx.saturating_add(1);
        }
        drop(rtxn);

        Ok(Self {
            env,
            codes,
            order,
            next_index,
        })
    }
}

impl PqStore for LmdbPqStore {
    fn insert(&mut self, id: &str, codes: &[u8]) -> Result<(), PqError> {
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|e| PqError::Store(e.to_string()))?;
        let existed = self
            .codes
            .get(&wtxn, id)
            .map_err(|e| PqError::Store(e.to_string()))?
            .is_some();
        self.codes
            .put(&mut wtxn, id, codes)
            .map_err(|e| PqError::Store(e.to_string()))?;
        if !existed {
            let idx = self.next_index;
            self.order
                .put(&mut wtxn, &idx, id)
                .map_err(|e| PqError::Store(e.to_string()))?;
            self.next_index = idx.saturating_add(1);
        }
        wtxn.commit().map_err(|e| PqError::Store(e.to_string()))?;
        Ok(())
    }

    fn len(&self) -> usize {
        let rtxn = match self.env.read_txn() {
            Ok(t) => t,
            Err(_) => return 0,
        };
        self.codes.len(&rtxn).unwrap_or(0) as usize
    }

    fn for_each(&self, f: &mut dyn FnMut(&str, &[u8])) {
        let Ok(rtxn) = self.env.read_txn() else {
            return;
        };
        // Walk the order index, look up each id in `codes`. Stable
        // insertion-order traversal regardless of LMDB's lexicographic
        // key sort.
        let Ok(iter) = self.order.iter(&rtxn) else {
            return;
        };
        for entry in iter.flatten() {
            let (_idx, id) = entry;
            if let Ok(Some(c)) = self.codes.get(&rtxn, id) {
                f(id, c);
            }
        }
    }

    fn get(&self, id: &str) -> Option<Vec<u8>> {
        let rtxn = self.env.read_txn().ok()?;
        self.codes.get(&rtxn, id).ok().flatten().map(|c| c.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pq::{PqConfig, PqIndex};
    use tempfile::TempDir;

    fn small_corpus() -> (Vec<String>, Vec<Vec<f32>>) {
        let n = 60;
        let dim = 16;
        let mut corpus = Vec::with_capacity(n);
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0f32; dim];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = ((i + d) as f32 * 0.1).sin();
            }
            corpus.push(v);
            ids.push(format!("id-{i:04}"));
        }
        (ids, corpus)
    }

    #[test]
    fn open_insert_search_persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pq");
        let (ids, corpus) = small_corpus();
        let cfg = PqConfig {
            m: 4,
            bits_per_code: 4,
            kmeans_iters: 10,
            kmeans_tol: 1e-4,
            seed: 1,
        };

        let codebook = crate::pq::PqCodebook::train(&corpus, &cfg).unwrap();
        {
            let store = LmdbPqStore::open(&path, 64 * 1024 * 1024).unwrap();
            let mut index = PqIndex::new(codebook.clone(), Box::new(store));
            for (id, e) in ids.iter().zip(corpus.iter()) {
                index.insert(id, e).unwrap();
            }
            let hits = index.search(&corpus[5], 1);
            assert_eq!(hits[0].0, "id-0005");
        }

        // Reopen and verify codes are still there.
        let store2 = LmdbPqStore::open(&path, 64 * 1024 * 1024).unwrap();
        assert_eq!(store2.len(), ids.len());
        let index2 = PqIndex::new(codebook, Box::new(store2));
        let hits = index2.search(&corpus[5], 1);
        assert_eq!(hits[0].0, "id-0005");
    }
}
