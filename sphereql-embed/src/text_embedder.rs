//! Pluggable text-to-embedding hook.
//!
//! Consumers that want to query the pipeline from natural-language text
//! (e.g. the GraphQL crate's `drillDown(queryText: ...)`, a REPL, a Python
//! shell helper) need a way to turn a `&str` into an [`Embedding`] without
//! this crate taking a dependency on any specific embedder. `TextEmbedder`
//! is that hook.
//!
//! The trait is deliberately minimal: one fallible method, `Send + Sync`,
//! no lifetime parameters. Users implement it on their own newtype — a
//! thin wrapper around `sentence-transformers`, an OpenAI client, a local
//! ONNX model, or a deterministic hash — and hand an `Arc<dyn TextEmbedder>`
//! to whatever schema / pipeline / REPL consumes it.
//!
//! Convenience types:
//!
//! - [`NoEmbedder`] — the default; returns an error on `embed()`. Wired
//!   into GraphQL schemas that haven't configured a real embedder so text
//!   query resolvers fail with a clear message rather than panicking.
//! - [`FnEmbedder`] — a zero-cost wrapper that lifts a closure into the
//!   trait, for quick wiring in tests and examples.
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use sphereql_embed::text_embedder::{TextEmbedder, FnEmbedder, EmbedderError};
//! use sphereql_embed::types::Embedding;
//!
//! let embedder: Arc<dyn TextEmbedder> = Arc::new(FnEmbedder::new(|text: &str| {
//!     let len = text.len() as f64;
//!     Ok(Embedding::new(vec![len, len.sqrt(), len.ln().max(0.0)]))
//! }));
//!
//! let vec = embedder.embed("hello").unwrap();
//! assert_eq!(vec.dimension(), 3);
//! ```

use thiserror::Error;

use crate::types::Embedding;

/// Errors surfaced from a [`TextEmbedder`] implementation.
///
/// Kept intentionally small: the single stringly-typed variant lets any
/// backend (HTTP, ONNX runtime, pure-Rust model) funnel its native error
/// into the trait without forcing this crate to enumerate every failure
/// mode. Consumers that need typed errors can match on `.to_string()` or
/// wrap this in their own error enum.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// Embedder implementation failed while embedding the input.
    #[error("embedder failed: {0}")]
    Embedding(String),

    /// Input was rejected before reaching the model (empty string,
    /// too long, invalid UTF-8 boundary, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl EmbedderError {
    /// Construct an [`EmbedderError::Embedding`] from any displayable error.
    pub fn embedding<E: std::fmt::Display>(err: E) -> Self {
        EmbedderError::Embedding(err.to_string())
    }

    /// Construct an [`EmbedderError::InvalidInput`] from any displayable error.
    pub fn invalid_input<E: std::fmt::Display>(err: E) -> Self {
        EmbedderError::InvalidInput(err.to_string())
    }
}

/// Turns free-form text into an [`Embedding`] suitable for projection
/// through the sphereQL pipeline.
///
/// Implement on a newtype that owns the embedder's state (HTTP client,
/// tokenizer, model weights). The trait is `Send + Sync` so a single
/// embedder can be shared across async request handlers via
/// `Arc<dyn TextEmbedder>`.
pub trait TextEmbedder: Send + Sync {
    /// Embed a text query.
    ///
    /// Returns an [`Embedding`] whose dimensionality must match whatever
    /// pipeline the result will be fed into. Implementations should
    /// surface upstream failures as [`EmbedderError::Embedding`] rather
    /// than panicking.
    fn embed(&self, text: &str) -> Result<Embedding, EmbedderError>;
}

impl<T: TextEmbedder + ?Sized> TextEmbedder for std::sync::Arc<T> {
    fn embed(&self, text: &str) -> Result<Embedding, EmbedderError> {
        (**self).embed(text)
    }
}

impl<T: TextEmbedder + ?Sized> TextEmbedder for Box<T> {
    fn embed(&self, text: &str) -> Result<Embedding, EmbedderError> {
        (**self).embed(text)
    }
}

/// Default embedder that always fails with a descriptive error.
///
/// Wired into schemas that haven't been given a real embedder, so a text
/// query hits an actionable "no TextEmbedder configured" error rather
/// than panicking or silently returning empty results.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoEmbedder;

impl TextEmbedder for NoEmbedder {
    fn embed(&self, _text: &str) -> Result<Embedding, EmbedderError> {
        Err(EmbedderError::Embedding(
            "no TextEmbedder configured — supply one to build_category_schema \
             (or equivalent) before issuing text queries"
                .into(),
        ))
    }
}

/// Zero-cost wrapper that lifts a closure into [`TextEmbedder`].
///
/// Useful for tests and examples where the full trait-impl-on-newtype
/// ceremony is overkill:
///
/// ```ignore
/// use sphereql_embed::text_embedder::FnEmbedder;
/// use sphereql_embed::types::Embedding;
///
/// let embedder = FnEmbedder::new(|text: &str| {
///     Ok(Embedding::new(vec![text.len() as f64; 128]))
/// });
/// ```
pub struct FnEmbedder<F> {
    inner: F,
}

impl<F> FnEmbedder<F>
where
    F: Fn(&str) -> Result<Embedding, EmbedderError> + Send + Sync,
{
    /// Wrap a closure. The closure must be `Send + Sync` so the resulting
    /// embedder can be shared across threads.
    pub fn new(f: F) -> Self {
        Self { inner: f }
    }
}

impl<F> TextEmbedder for FnEmbedder<F>
where
    F: Fn(&str) -> Result<Embedding, EmbedderError> + Send + Sync,
{
    fn embed(&self, text: &str) -> Result<Embedding, EmbedderError> {
        (self.inner)(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_embedder_errors_descriptively() {
        let err = NoEmbedder.embed("hello").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no TextEmbedder configured"), "got: {msg}");
    }

    #[test]
    fn fn_embedder_round_trips() {
        let embedder = FnEmbedder::new(|text: &str| {
            Ok(Embedding::new(vec![text.len() as f64, 0.0, 0.0]))
        });
        let v = embedder.embed("hello world").unwrap();
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.values[0], 11.0);
    }

    #[test]
    fn arc_forwards() {
        let arc: std::sync::Arc<dyn TextEmbedder> =
            std::sync::Arc::new(FnEmbedder::new(|_| Ok(Embedding::new(vec![1.0]))));
        assert_eq!(arc.embed("x").unwrap().values, vec![1.0]);
    }

    #[test]
    fn error_constructors_format() {
        let e = EmbedderError::embedding("upstream blew up");
        assert_eq!(e.to_string(), "embedder failed: upstream blew up");
        let e = EmbedderError::invalid_input("empty");
        assert_eq!(e.to_string(), "invalid input: empty");
    }
}
