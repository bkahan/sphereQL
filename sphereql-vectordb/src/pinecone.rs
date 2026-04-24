use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::VectorStoreError;
use crate::redacted::Redacted;
use crate::store::VectorStore;
use crate::types::{PayloadUpdate, SearchResult, VectorPage, VectorRecord};

// ── Config ──────────────────────────────────────────────────────────────

/// Configuration for connecting to a Pinecone index.
///
/// `api_key` is a [`Redacted`] wrapper so the struct's `Debug` impl
/// never leaks the key into logs, panic backtraces, or test output.
#[derive(Debug, Clone)]
pub struct PineconeConfig {
    pub api_key: Redacted,
    /// Index host, e.g. "my-index-abc123.svc.us-east1-gcp.pinecone.io"
    pub host: String,
    /// Pinecone namespace. Default "".
    pub namespace: String,
    pub dimension: usize,
    /// Pinecone caps top_k at 10,000.
    pub top_k_limit: usize,
}

impl PineconeConfig {
    pub fn new(api_key: impl Into<String>, host: impl Into<String>, dimension: usize) -> Self {
        Self {
            api_key: Redacted::new(api_key),
            host: host.into(),
            namespace: String::new(),
            dimension,
            top_k_limit: 10_000,
        }
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn with_top_k_limit(mut self, limit: usize) -> Self {
        self.top_k_limit = limit;
        self
    }
}

// ── Store ───────────────────────────────────────────────────────────────

/// Pinecone vector database backend.
///
/// Connects to a Pinecone index via its REST API. All operations use
/// `reqwest` with the `Api-Key` header for authentication.
///
/// # Precision
///
/// Pinecone stores vectors as `f32`. sphereQL uses `f64` internally.
/// Vectors are converted f64→f32 on write and f32→f64 on read.
///
/// # Metadata updates
///
/// `set_payload` calls Pinecone's update endpoint once per record,
/// since Pinecone doesn't support batch metadata updates. This can be
/// slow for large datasets.
pub struct PineconeStore {
    config: PineconeConfig,
    client: Client,
    base_url: String,
}

impl PineconeStore {
    pub fn new(config: PineconeConfig) -> Result<Self, VectorStoreError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        let host = config.host.trim_end_matches('/');
        let base_url = if host.starts_with("https://") {
            host.to_string()
        } else if let Some(after_scheme) = host.strip_prefix("http://") {
            // Allow plain HTTP only for localhost (dev/testing)
            let host_part = after_scheme.split(':').next().unwrap_or(after_scheme);
            if host_part == "localhost" || host_part == "127.0.0.1" || host_part == "[::1]" {
                host.to_string()
            } else {
                return Err(VectorStoreError::InvalidConfig(
                    "Pinecone host must use HTTPS (HTTP is only allowed for localhost)".into(),
                ));
            }
        } else {
            format!("https://{host}")
        };

        Ok(Self {
            config,
            client,
            base_url,
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, format!("{}{path}", self.base_url))
            .header("Api-Key", self.config.api_key.expose())
            .header("Content-Type", "application/json")
    }

    async fn check_response(
        resp: reqwest::Response,
    ) -> Result<reqwest::Response, VectorStoreError> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        let status = resp.status();

        // Map 429 to RateLimited so callers can back off and retry.
        // Pinecone returns `Retry-After` in seconds; RFC 7231 also
        // allows an HTTP-date, but Pinecone doesn't use that form.
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(std::time::Duration::from_secs);
            return Err(VectorStoreError::RateLimited { retry_after });
        }

        let body = resp.text().await.unwrap_or_default();
        if let Ok(err) = serde_json::from_str::<PineconeError>(&body) {
            Err(VectorStoreError::Backend(format!(
                "Pinecone {status}: {} (code {})",
                err.message, err.code
            )))
        } else {
            Err(VectorStoreError::Backend(format!(
                "Pinecone {status}: {body}"
            )))
        }
    }

    /// Apply a single payload update. Split out so
    /// [`VectorStore::set_payload`]'s `FuturesUnordered` pool can
    /// pattern-borrow `&self` through the method instead of fighting
    /// the closure's higher-ranked lifetimes in `buffer_unordered`.
    async fn set_payload_one(&self, update: &PayloadUpdate) -> Result<(), VectorStoreError> {
        let body = UpdateRequest {
            id: update.id.clone(),
            set_metadata: update.metadata.clone(),
            namespace: ns_or_none(&self.config.namespace),
        };
        let resp = self
            .request(reqwest::Method::POST, "/vectors/update")
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        Self::check_response(resp).await?;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for PineconeStore {
    async fn upsert(&self, records: &[VectorRecord]) -> Result<(), VectorStoreError> {
        if records.is_empty() {
            return Ok(());
        }

        for chunk in records.chunks(100) {
            let vectors: Vec<PineconeVector> = chunk
                .iter()
                .map(|r| {
                    if r.vector.len() != self.config.dimension {
                        return Err(VectorStoreError::DimensionMismatch {
                            expected: self.config.dimension,
                            got: r.vector.len(),
                        });
                    }
                    Ok(PineconeVector {
                        id: r.id.clone(),
                        values: f64_to_f32(&r.vector),
                        metadata: if r.metadata.is_empty() {
                            None
                        } else {
                            Some(r.metadata.clone())
                        },
                    })
                })
                .collect::<Result<_, _>>()?;

            let body = UpsertRequest {
                vectors,
                namespace: ns_or_none(&self.config.namespace),
            };

            let resp = self
                .request(reqwest::Method::POST, "/vectors/upsert")
                .json(&body)
                .send()
                .await
                .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

            Self::check_response(resp).await?;
        }

        Ok(())
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<VectorRecord>, VectorStoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut query_pairs: Vec<(&str, &str)> =
            ids.iter().map(|id| ("ids", id.as_str())).collect();
        if !self.config.namespace.is_empty() {
            query_pairs.push(("namespace", &self.config.namespace));
        }

        let resp = self
            .request(reqwest::Method::GET, "/vectors/fetch")
            .query(&query_pairs)
            .send()
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        let resp = Self::check_response(resp).await?;
        let fetch: FetchResponse = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;

        let records = fetch
            .vectors
            .into_values()
            .map(|v| VectorRecord {
                id: v.id,
                vector: f32_to_f64(&v.values),
                metadata: v.metadata.unwrap_or_default(),
            })
            .collect();

        Ok(records)
    }

    async fn delete(&self, ids: &[String]) -> Result<(), VectorStoreError> {
        if ids.is_empty() {
            return Ok(());
        }

        let body = DeleteRequest {
            ids: ids.to_vec(),
            namespace: ns_or_none(&self.config.namespace),
        };

        let resp = self
            .request(reqwest::Method::POST, "/vectors/delete")
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        Self::check_response(resp).await?;
        Ok(())
    }

    async fn search(
        &self,
        vector: &[f64],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if vector.len() != self.config.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        let top_k = k.min(self.config.top_k_limit);
        let body = QueryRequest {
            vector: f64_to_f32(vector),
            top_k,
            namespace: ns_or_none(&self.config.namespace),
            include_values: true,
            include_metadata: true,
        };

        let resp = self
            .request(reqwest::Method::POST, "/query")
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        let resp = Self::check_response(resp).await?;
        let query_resp: QueryResponse = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;

        Ok(query_resp
            .matches
            .into_iter()
            .map(|m| SearchResult {
                id: m.id,
                score: m.score as f64,
                vector: m.values.map(|v| f32_to_f64(&v)),
                metadata: m.metadata.unwrap_or_default(),
            })
            .collect())
    }

    async fn list(
        &self,
        limit: usize,
        offset: Option<&str>,
    ) -> Result<VectorPage, VectorStoreError> {
        // Step 1: List IDs
        let mut query_params: Vec<(&str, String)> = vec![("limit", limit.to_string())];
        if !self.config.namespace.is_empty() {
            query_params.push(("namespace", self.config.namespace.clone()));
        }
        if let Some(token) = offset {
            query_params.push(("paginationToken", token.to_string()));
        }

        let resp = self
            .request(reqwest::Method::GET, "/vectors/list")
            .query(&query_params)
            .send()
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        let resp = Self::check_response(resp).await?;
        let list_resp: ListResponse = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;

        let ids: Vec<String> = list_resp
            .vectors
            .unwrap_or_default()
            .into_iter()
            .map(|v| v.id)
            .collect();

        // Step 2: Fetch full records for these IDs
        let records = if ids.is_empty() {
            Vec::new()
        } else {
            self.get(&ids).await?
        };

        Ok(VectorPage {
            records,
            next_offset: list_resp.pagination.and_then(|p| p.next),
        })
    }

    async fn count(&self) -> Result<usize, VectorStoreError> {
        let resp = self
            .request(reqwest::Method::POST, "/describe_index_stats")
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        let resp = Self::check_response(resp).await?;
        let stats: DescribeIndexStatsResponse = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;

        let count = if self.config.namespace.is_empty() {
            stats.total_vector_count.unwrap_or(0)
        } else {
            stats
                .namespaces
                .and_then(|ns| ns.get(&self.config.namespace).cloned())
                .map(|n| n.vector_count)
                .unwrap_or(0)
        };

        Ok(count)
    }

    async fn set_payload(&self, updates: &[PayloadUpdate]) -> Result<(), VectorStoreError> {
        // Pinecone's `/vectors/update` endpoint is per-record — there
        // is no batch metadata update. Previous implementation posted
        // N requests serially; at 500k records that's tens of minutes.
        // Sliding-window `FuturesUnordered` pool (same shape as
        // QdrantStore::set_payload) keeps up to CONCURRENCY requests
        // in flight. 16 is a conservative default below Pinecone's
        // typical rate-limit budget — anything higher risks tripping
        // 429, which now maps to VectorStoreError::RateLimited.
        use futures::stream::{FuturesUnordered, StreamExt};

        const CONCURRENCY: usize = 16;
        let mut in_flight = FuturesUnordered::new();
        let mut iter = updates.iter();

        for update in iter.by_ref().take(CONCURRENCY) {
            in_flight.push(self.set_payload_one(update));
        }
        while let Some(result) = in_flight.next().await {
            result?;
            if let Some(update) = iter.next() {
                in_flight.push(self.set_payload_one(update));
            }
        }
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn collection_name(&self) -> &str {
        if self.config.namespace.is_empty() {
            &self.config.host
        } else {
            &self.config.namespace
        }
    }
}

// ── Precision conversion ────────────────────────────────────────────────

fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn f32_to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn ns_or_none(ns: &str) -> Option<String> {
    if ns.is_empty() {
        None
    } else {
        Some(ns.to_string())
    }
}

// ── Pinecone API types ──────────────────────────────────────────────────

#[derive(Serialize)]
struct PineconeVector {
    id: String,
    values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct UpsertRequest {
    vectors: Vec<PineconeVector>,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,
}

#[derive(Deserialize)]
struct FetchResponse {
    #[serde(default)]
    vectors: HashMap<String, FetchedVector>,
}

#[derive(Deserialize)]
struct FetchedVector {
    id: String,
    values: Vec<f32>,
    metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct DeleteRequest {
    ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,
}

#[derive(Serialize)]
struct QueryRequest {
    vector: Vec<f32>,
    #[serde(rename = "topK")]
    top_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,
    #[serde(rename = "includeValues")]
    include_values: bool,
    #[serde(rename = "includeMetadata")]
    include_metadata: bool,
}

#[derive(Deserialize)]
struct QueryResponse {
    #[serde(default)]
    matches: Vec<QueryMatch>,
}

#[derive(Deserialize)]
struct QueryMatch {
    id: String,
    score: f32,
    values: Option<Vec<f32>>,
    metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct UpdateRequest {
    id: String,
    #[serde(rename = "setMetadata")]
    set_metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<String>,
}

#[derive(Deserialize)]
struct ListResponse {
    vectors: Option<Vec<ListVector>>,
    pagination: Option<ListPagination>,
}

#[derive(Deserialize)]
struct ListVector {
    id: String,
}

#[derive(Deserialize)]
struct ListPagination {
    next: Option<String>,
}

#[derive(Deserialize)]
struct DescribeIndexStatsResponse {
    namespaces: Option<HashMap<String, NamespaceStats>>,
    #[serde(rename = "totalVectorCount")]
    total_vector_count: Option<usize>,
}

#[derive(Deserialize, Clone)]
struct NamespaceStats {
    #[serde(rename = "vectorCount")]
    vector_count: usize,
}

#[derive(Deserialize)]
struct PineconeError {
    code: i32,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = PineconeConfig::new("key", "my-host.pinecone.io", 384);
        assert_eq!(config.api_key.expose(), "key");
        assert_eq!(config.host, "my-host.pinecone.io");
        assert_eq!(config.dimension, 384);

        // `{:?}` on the config must not leak the key — this is the
        // whole reason `api_key` is wrapped in `Redacted`.
        let rendered = format!("{config:?}");
        assert!(
            !rendered.contains("\"key\""),
            "PineconeConfig Debug leaked the api_key: {rendered}"
        );
        assert!(rendered.contains("redacted"));
        assert!(config.namespace.is_empty());
        assert_eq!(config.top_k_limit, 10_000);
    }

    #[test]
    fn config_builder() {
        let config = PineconeConfig::new("key", "host", 128)
            .with_namespace("my-ns")
            .with_top_k_limit(5000);
        assert_eq!(config.namespace, "my-ns");
        assert_eq!(config.top_k_limit, 5000);
    }

    #[test]
    fn store_base_url_no_scheme() {
        let config = PineconeConfig::new("key", "my-index.svc.pinecone.io", 3);
        let store = PineconeStore::new(config).unwrap();
        assert_eq!(store.base_url, "https://my-index.svc.pinecone.io");
    }

    #[test]
    fn store_base_url_with_scheme() {
        let config = PineconeConfig::new("key", "http://localhost:8080", 3);
        let store = PineconeStore::new(config).unwrap();
        assert_eq!(store.base_url, "http://localhost:8080");
    }

    #[test]
    fn store_base_url_trailing_slash() {
        let config = PineconeConfig::new("key", "https://host.io/", 3);
        let store = PineconeStore::new(config).unwrap();
        assert_eq!(store.base_url, "https://host.io");
    }

    #[test]
    fn collection_name_uses_namespace() {
        let config = PineconeConfig::new("key", "host.io", 3).with_namespace("prod");
        let store = PineconeStore::new(config).unwrap();
        assert_eq!(store.collection_name(), "prod");
    }

    #[test]
    fn collection_name_falls_back_to_host() {
        let config = PineconeConfig::new("key", "host.io", 3);
        let store = PineconeStore::new(config).unwrap();
        assert_eq!(store.collection_name(), "host.io");
    }

    #[test]
    fn f64_f32_roundtrip() {
        let original = vec![1.0, 0.123_456_789, -99.99];
        let f32_vec = f64_to_f32(&original);
        let recovered = f32_to_f64(&f32_vec);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn ns_or_none_empty() {
        assert!(ns_or_none("").is_none());
    }

    #[test]
    fn ns_or_none_non_empty() {
        assert_eq!(ns_or_none("prod"), Some("prod".to_string()));
    }

    #[test]
    fn upsert_request_serialization() {
        let req = UpsertRequest {
            vectors: vec![PineconeVector {
                id: "v1".into(),
                values: vec![1.0, 2.0, 3.0],
                metadata: Some(HashMap::from([(
                    "category".into(),
                    serde_json::json!("science"),
                )])),
            }],
            namespace: Some("test-ns".into()),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["vectors"][0]["id"], "v1");
        assert_eq!(json["namespace"], "test-ns");
    }

    #[test]
    fn query_request_serialization() {
        let req = QueryRequest {
            vector: vec![0.1, 0.2],
            top_k: 5,
            namespace: None,
            include_values: true,
            include_metadata: true,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["topK"], 5);
        assert_eq!(json["includeValues"], true);
        assert!(json.get("namespace").is_none());
    }

    #[test]
    fn fetch_response_deserialization() {
        let json = serde_json::json!({
            "vectors": {
                "v1": {
                    "id": "v1",
                    "values": [1.0, 2.0, 3.0],
                    "metadata": {"category": "science"}
                }
            }
        });
        let resp: FetchResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.vectors.len(), 1);
        let v = &resp.vectors["v1"];
        assert_eq!(v.id, "v1");
        assert_eq!(v.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn query_response_deserialization() {
        let json = serde_json::json!({
            "matches": [
                {
                    "id": "v1",
                    "score": 0.95,
                    "values": [1.0, 2.0],
                    "metadata": {"k": "v"}
                },
                {
                    "id": "v2",
                    "score": 0.80
                }
            ]
        });
        let resp: QueryResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.matches.len(), 2);
        assert_eq!(resp.matches[0].id, "v1");
        assert!((resp.matches[0].score - 0.95).abs() < 1e-6);
        assert!(resp.matches[1].values.is_none());
        assert!(resp.matches[1].metadata.is_none());
    }

    #[test]
    fn list_response_deserialization() {
        let json = serde_json::json!({
            "vectors": [{"id": "a"}, {"id": "b"}],
            "pagination": {"next": "token123"}
        });
        let resp: ListResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.vectors.as_ref().unwrap().len(), 2);
        assert_eq!(
            resp.pagination.as_ref().unwrap().next,
            Some("token123".into())
        );
    }

    #[test]
    fn list_response_no_pagination() {
        let json = serde_json::json!({"vectors": []});
        let resp: ListResponse = serde_json::from_value(json).unwrap();
        assert!(resp.pagination.is_none());
    }

    #[test]
    fn describe_stats_deserialization() {
        let json = serde_json::json!({
            "namespaces": {
                "": {"vectorCount": 100},
                "prod": {"vectorCount": 50}
            },
            "totalVectorCount": 150
        });
        let resp: DescribeIndexStatsResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.total_vector_count, Some(150));
        let ns = resp.namespaces.unwrap();
        assert_eq!(ns["prod"].vector_count, 50);
    }

    #[test]
    fn pinecone_error_deserialization() {
        let json = serde_json::json!({"code": 3, "message": "not found"});
        let err: PineconeError = serde_json::from_value(json).unwrap();
        assert_eq!(err.code, 3);
        assert_eq!(err.message, "not found");
    }
}
