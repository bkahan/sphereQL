use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("connection failed: {0}")]
    Connection(String),

    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("record not found: {0}")]
    NotFound(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("pipeline not built — call build_pipeline() first")]
    PipelineNotBuilt,

    #[error("insufficient data: {0}")]
    InsufficientData(String),

    /// The backend rate-limited us (Pinecone 429, Qdrant
    /// `ResourceExhausted`). `retry_after` carries the server's hint
    /// when one was returned — callers should sleep for at least that
    /// long before retrying. `None` means the backend declined to
    /// specify.
    #[error(
        "rate limited by backend{}",
        retry_after
            .map(|d| format!(" (retry after {}ms)", d.as_millis()))
            .unwrap_or_default()
    )]
    RateLimited { retry_after: Option<Duration> },

    /// The backend's response exceeds the configured
    /// [`BridgeConfig::max_response_bytes`] cap. Rejected without
    /// allocation to guard against OOM from a buggy or malicious
    /// server. Raise the cap in `BridgeConfig` if your corpus has
    /// legitimately large payloads.
    #[error("response size {bytes} exceeds cap of {cap} bytes")]
    ResponseTooLarge { bytes: u64, cap: u64 },
}
