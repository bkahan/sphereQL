//! Tiny shared helpers used across multiple modules.
//!
//! Kept deliberately small — when in doubt, inline instead of growing
//! this module. The helpers live here because they'd otherwise duplicate
//! across `meta_model.rs` and `feedback.rs` (both need timestamps on
//! persisted records, both default their storage to `~/.sphereql/`, and
//! both migrate legacy JSON-array stores to JSONL on first append).

use std::fs;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use indexmap::IndexMap;

/// Default persisted-record timestamp: seconds since Unix epoch, as a
/// string. Sortable, unambiguous, and dependency-free. Callers that
/// want a human-readable format should overwrite the timestamp field
/// themselves (e.g. via `with_timestamp`).
pub fn default_timestamp() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_secs().to_string(),
        Err(_) => "0".to_string(),
    }
}

/// Resolve `~/.sphereql/` — the on-disk convention for SphereQL's
/// persistent training stores (meta_records.json, feedback_events.json).
///
/// Returns `$HOME/.sphereql` on Unix, `$USERPROFILE\.sphereql` on
/// Windows. Returns an error only when neither env var is set — rare,
/// would mean the process is running without a user profile.
pub fn sphereql_home_dir() -> io::Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "neither HOME nor USERPROFILE is set",
            )
        })?;
    Ok(PathBuf::from(home).join(".sphereql"))
}

/// Hard cap on the number of distinct canonicalized paths we keep
/// migration locks for. Long-running processes that touch many
/// stores would otherwise grow the lock map unboundedly. The
/// per-path lock is only contended during the one-shot legacy-JSON →
/// JSONL migration, so evicting an idle entry is safe — the next
/// migration on that path just allocates a fresh lock.
const MIGRATION_LOCK_CAPACITY: usize = 128;

/// In-process serialization for the legacy-JSON → JSONL migration in
/// [`migrate_legacy_array_to_jsonl`]. Two threads racing the migration
/// would otherwise duplicate the migrated bytes (each reads the legacy
/// array, each writes its own copy back). Keyed per canonical path so
/// unrelated stores don't contend. Bounded LRU so a long-running daemon
/// that rotates through many files doesn't leak.
fn migration_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<IndexMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let key = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let map = LOCKS.get_or_init(|| Mutex::new(IndexMap::new()));
    // Panics only if another thread panicked while holding this lock, which
    // indicates an unrecoverable process state — re-panicking is correct.
    let mut guard = map.lock().expect("migration lock map poisoned");
    if let Some(existing) = guard.shift_remove(&key) {
        guard.insert(key, existing.clone());
        return existing;
    }
    while guard.len() >= MIGRATION_LOCK_CAPACITY {
        guard.shift_remove_index(0);
    }
    let lock = Arc::new(Mutex::new(()));
    guard.insert(key, lock.clone());
    lock
}

fn first_non_ws_byte(path: &Path) -> io::Result<Option<u8>> {
    let mut f = fs::File::open(path)?;
    let mut buf = [0u8; 64];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        if let Some(&b) = buf[..n].iter().find(|b| !b.is_ascii_whitespace()) {
            return Ok(Some(b));
        }
    }
}

/// One-time migration of a legacy JSON-array store to JSONL, shared by
/// the append paths in `meta_model.rs` and `feedback.rs`.
///
/// No-op unless `path` exists and its first non-whitespace byte is `[`
/// — only that byte is needed to disambiguate, so the non-legacy hot
/// path never reads the whole file. When a legacy array is detected,
/// `to_jsonl` receives the full file text and must return the JSONL
/// replacement (one record per line, trailing newline included).
///
/// Concurrency-safe: the migration is serialized per canonical path
/// and the format is re-checked under the lock, so concurrent
/// appenders don't double-migrate. The rewrite goes through a sibling
/// temp file + rename, so a crash mid-migration leaves either the
/// legacy array or the migrated JSONL — never a half-written mix.
pub fn migrate_legacy_array_to_jsonl(
    path: &Path,
    to_jsonl: impl FnOnce(&str) -> io::Result<String>,
) -> io::Result<()> {
    if !path.exists() || first_non_ws_byte(path)? != Some(b'[') {
        return Ok(());
    }
    let lock = migration_lock(path);
    // Same reasoning as in migration_lock — mutex poisoning means
    // unrecoverable state.
    let _g = lock.lock().expect("migration lock poisoned");
    if path.exists() && first_non_ws_byte(path)? == Some(b'[') {
        let head = fs::read_to_string(path)?;
        let migrated = to_jsonl(&head)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
        io::Write::write_all(&mut tmp, migrated.as_bytes())?;
        tmp.persist(path).map_err(io::Error::other)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_timestamp_is_parseable_epoch_seconds() {
        let ts = default_timestamp();
        assert!(!ts.is_empty());
        assert!(ts.parse::<u64>().is_ok());
    }

    #[test]
    fn sphereql_home_dir_ends_in_dot_sphereql() {
        let p = sphereql_home_dir().unwrap();
        assert_eq!(p.file_name().and_then(|s| s.to_str()), Some(".sphereql"));
    }
}
