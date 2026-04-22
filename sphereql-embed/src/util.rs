//! Tiny shared helpers used across multiple modules.
//!
//! Kept deliberately small — when in doubt, inline instead of growing
//! this module. Two helpers live here because they'd otherwise duplicate
//! across `meta_model.rs` and `feedback.rs` (both need timestamps on
//! persisted records and both default their storage to `~/.sphereql/`).

use std::io;
use std::path::PathBuf;

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
