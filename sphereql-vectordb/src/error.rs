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

    #[error("pipeline not built \u2014 call build_pipeline() first")]
    PipelineNotBuilt,

    #[error("insufficient data: {0}")]
    InsufficientData(String),
}
