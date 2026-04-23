//! Secret wrapper that hides its payload from `{:?}` / `{}` and any
//! transitively-derived `Debug` / `Display` on owning structs.
//!
//! `#[derive(Debug)]` on a config struct that holds a plain `String`
//! `api_key` field silently emits the raw key in every panic backtrace,
//! `tracing::error!("{:?}", config)` call, and test-failure message.
//! Callers get no warning and the leak path is invisible at the use
//! site. Wrapping the field in [`Redacted`] turns that into a
//! `"<redacted>"` placeholder by construction.
//!
//! We keep our own newtype instead of pulling in `secrecy` because the
//! full `SecretString` machinery (exposing via `ExposeSecret`, zero-on-
//! drop via `zeroize`) is overkill — we need hide-from-logs, not an
//! adversary-resistant secret type. If the threat model ever grows,
//! this is a one-line swap.

use std::fmt;

/// Stores a `String` that must not appear in `Debug` / `Display`
/// output. Construct via [`Redacted::new`]; read the underlying value
/// with [`Redacted::expose`] at the (single, explicit) point that
/// actually needs it.
#[derive(Clone)]
pub struct Redacted(String);

impl Redacted {
    /// Wrap a secret. Accepts any `Into<String>` so call sites read
    /// `Redacted::new(key)` without extra conversions.
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// Read the underlying value. Every call site is intentionally
    /// visible in a grep, so leaks are easier to review.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// True if the wrapped string is empty. Lets callers gate "is a
    /// key configured?" logic without calling [`Self::expose`].
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for Redacted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Redacted(\"<redacted>\")")
    }
}

impl fmt::Display for Redacted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_does_not_leak_secret() {
        let r = Redacted::new("sk-live-extremely-secret-key");
        let rendered = format!("{r:?}");
        assert!(!rendered.contains("sk-live"));
        assert!(!rendered.contains("secret"));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn display_does_not_leak_secret() {
        let r = Redacted::new("sk-live-extremely-secret-key");
        let rendered = format!("{r}");
        assert!(!rendered.contains("sk-live"));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn expose_returns_original() {
        let r = Redacted::new("top-secret");
        assert_eq!(r.expose(), "top-secret");
    }

    #[test]
    fn owning_struct_debug_also_redacts() {
        // The whole point: callers `#[derive(Debug)]` on their config;
        // this is the transitive property that must hold.
        #[derive(Debug)]
        struct Config {
            api_key: Redacted,
            host: String,
        }
        let c = Config {
            api_key: Redacted::new("sk-live-leaky"),
            host: "example.com".into(),
        };
        let rendered = format!("{c:?}");
        assert!(!rendered.contains("sk-live-leaky"));
        assert!(rendered.contains("redacted"));
        assert!(rendered.contains("example.com"));
    }
}
