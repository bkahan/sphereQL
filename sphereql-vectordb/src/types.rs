use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A single vector record: id + dense vector + arbitrary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VectorRecord {
    pub fn new(id: impl Into<String>, vector: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

/// A scored search result returned by nearest-neighbor queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub vector: Option<Vec<f64>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A page of records from a scroll/list operation.
#[derive(Debug, Clone)]
pub struct VectorPage {
    pub records: Vec<VectorRecord>,
    /// Opaque cursor for the next page. `None` means no more pages.
    pub next_offset: Option<String>,
}

/// Configuration for creating or connecting to a vector collection.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub distance: DistanceMetric,
}

/// Distance metric used by the vector index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

/// Payload-only update: sets metadata fields on an existing record
/// without re-uploading the vector.
#[derive(Debug, Clone)]
pub struct PayloadUpdate {
    pub id: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Reserved payload keys written by sphereQL during projection sync.
pub const SPHEREQL_ID_KEY: &str = "_sphereql_id";
pub const SPHEREQL_R_KEY: &str = "_sphereql_r";
pub const SPHEREQL_THETA_KEY: &str = "_sphereql_theta";
pub const SPHEREQL_PHI_KEY: &str = "_sphereql_phi";
