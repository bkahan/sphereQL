use std::collections::HashMap;

use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CountPointsBuilder, CreateCollectionBuilder, DeletePointsBuilder, Distance, GetPointsBuilder,
    PointId, PointStruct, PointsIdsList, ScrollPointsBuilder, SearchPointsBuilder,
    SetPayloadPointsBuilder, UpsertPointsBuilder, Value as QdrantValue, VectorParamsBuilder,
    VectorsOutput, point_id::PointIdOptions, value::Kind,
    vector_output::Vector as VectorOutputVariant, vectors_output::VectorsOptions,
};

use crate::error::VectorStoreError;
use crate::store::VectorStore;
use crate::types::{
    DistanceMetric, PayloadUpdate, SPHEREQL_ID_KEY, SearchResult, VectorPage, VectorRecord,
};

/// Configuration for connecting to a Qdrant instance.
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection: String,
    pub dimension: usize,
    pub distance: DistanceMetric,
    /// Create the collection if it doesn't exist.
    pub create_if_missing: bool,
}

impl QdrantConfig {
    pub fn new(url: impl Into<String>, collection: impl Into<String>, dimension: usize) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            collection: collection.into(),
            dimension,
            distance: DistanceMetric::Cosine,
            create_if_missing: true,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn create_if_missing(mut self, create: bool) -> Self {
        self.create_if_missing = create;
        self
    }
}

/// Qdrant vector database backend.
///
/// Connects to a Qdrant instance over gRPC. All operations map directly
/// to Qdrant's point CRUD and search APIs.
///
/// # Vector precision
///
/// Qdrant stores vectors as `f32`. sphereQL uses `f64` internally.
/// Vectors are converted f64\u2192f32 on write and f32\u2192f64 on read.
/// The precision loss is negligible for similarity search.
///
/// # Point IDs
///
/// String IDs are mapped to `u64` via FNV-1a hashing. The original
/// string ID is always stored in the payload under `_sphereql_id`
/// so it can be recovered on read.
pub struct QdrantStore {
    client: Qdrant,
    collection: String,
    dimension: usize,
}

impl QdrantStore {
    /// Connect to Qdrant and optionally create the collection.
    pub async fn connect(config: QdrantConfig) -> Result<Self, VectorStoreError> {
        let mut builder = Qdrant::from_url(&config.url);
        if let Some(ref key) = config.api_key {
            builder = builder.api_key(key.as_str());
        }
        let client = builder
            .build()
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        if config.create_if_missing {
            let exists = client
                .collection_exists(&config.collection)
                .await
                .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

            if !exists {
                let distance = match config.distance {
                    DistanceMetric::Cosine => Distance::Cosine,
                    DistanceMetric::Euclidean => Distance::Euclid,
                    DistanceMetric::DotProduct => Distance::Dot,
                };
                client
                    .create_collection(
                        CreateCollectionBuilder::new(&config.collection).vectors_config(
                            VectorParamsBuilder::new(config.dimension as u64, distance),
                        ),
                    )
                    .await
                    .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
            }
        }

        Ok(Self {
            client,
            collection: config.collection,
            dimension: config.dimension,
        })
    }

    /// Wrap an existing Qdrant client \u2014 for when the caller manages
    /// the connection and collection lifecycle themselves.
    pub fn from_client(client: Qdrant, collection: impl Into<String>, dimension: usize) -> Self {
        Self {
            client,
            collection: collection.into(),
            dimension,
        }
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn upsert(&self, records: &[VectorRecord]) -> Result<(), VectorStoreError> {
        if records.is_empty() {
            return Ok(());
        }

        let points: Vec<PointStruct> = records
            .iter()
            .map(|r| {
                if r.vector.len() != self.dimension {
                    return Err(VectorStoreError::DimensionMismatch {
                        expected: self.dimension,
                        got: r.vector.len(),
                    });
                }
                let point_id = string_to_point_id(&r.id);
                let vector_f32 = f64_to_f32(&r.vector);

                let mut payload = metadata_to_payload(&r.metadata);
                payload.insert(SPHEREQL_ID_KEY.to_string(), QdrantValue::from(r.id.clone()));

                Ok(PointStruct::new(point_id, vector_f32, payload))
            })
            .collect::<Result<_, _>>()?;

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, points).wait(true))
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        Ok(())
    }

    async fn get(&self, ids: &[String]) -> Result<Vec<VectorRecord>, VectorStoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let point_ids: Vec<PointId> = ids.iter().map(|id| string_to_point_id(id)).collect();

        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection, point_ids)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        Ok(response
            .result
            .into_iter()
            .filter_map(|point| {
                let id = extract_original_id(&point.payload)
                    .or_else(|| point.id.as_ref().map(point_id_to_string))?;
                let vector = extract_vector_f64(&point.vectors)?;
                let metadata = payload_to_metadata(&point.payload);
                Some(VectorRecord {
                    id,
                    vector,
                    metadata,
                })
            })
            .collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<(), VectorStoreError> {
        if ids.is_empty() {
            return Ok(());
        }

        let point_ids: Vec<PointId> = ids.iter().map(|id| string_to_point_id(id)).collect();

        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection).points(PointsIdsList { ids: point_ids }),
            )
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        Ok(())
    }

    async fn search(
        &self,
        vector: &[f64],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if vector.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let query_f32 = f64_to_f32(vector);

        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.collection, query_f32, k as u64)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        Ok(response
            .result
            .into_iter()
            .filter_map(|scored| {
                let id = extract_original_id(&scored.payload)
                    .or_else(|| scored.id.as_ref().map(point_id_to_string))?;
                let vector = extract_vector_f64(&scored.vectors);
                let metadata = payload_to_metadata(&scored.payload);
                Some(SearchResult {
                    id,
                    score: scored.score as f64,
                    vector,
                    metadata,
                })
            })
            .collect())
    }

    async fn list(
        &self,
        limit: usize,
        offset: Option<&str>,
    ) -> Result<VectorPage, VectorStoreError> {
        let mut builder = ScrollPointsBuilder::new(&self.collection)
            .with_payload(true)
            .with_vectors(true)
            .limit(limit as u32);

        if let Some(off) = offset
            && let Ok(num) = off.parse::<u64>()
        {
            builder = builder.offset(PointId::from(num));
        }

        let response = self
            .client
            .scroll(builder)
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        let records: Vec<VectorRecord> = response
            .result
            .into_iter()
            .filter_map(|point| {
                let id = extract_original_id(&point.payload)
                    .or_else(|| point.id.as_ref().map(point_id_to_string))?;
                let vector = extract_vector_f64(&point.vectors)?;
                let metadata = payload_to_metadata(&point.payload);
                Some(VectorRecord {
                    id,
                    vector,
                    metadata,
                })
            })
            .collect();

        let next_offset = response.next_page_offset.as_ref().map(point_id_to_string);

        Ok(VectorPage {
            records,
            next_offset,
        })
    }

    async fn count(&self) -> Result<usize, VectorStoreError> {
        let response = self
            .client
            .count(CountPointsBuilder::new(&self.collection))
            .await
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;

        Ok(response.result.map(|r| r.count as usize).unwrap_or(0))
    }

    async fn set_payload(&self, updates: &[PayloadUpdate]) -> Result<(), VectorStoreError> {
        for update in updates {
            let point_id = string_to_point_id(&update.id);
            let payload = metadata_to_payload(&update.metadata);

            self.client
                .set_payload(
                    SetPayloadPointsBuilder::new(&self.collection, payload)
                        .points_selector(vec![point_id]),
                )
                .await
                .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn collection_name(&self) -> &str {
        &self.collection
    }
}

// \u2500\u2500 Precision conversion \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn f32_to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

// \u2500\u2500 Point ID mapping \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

/// Deterministic SHA-256 based UUID PointId.
/// Uses first 128 bits of SHA-256, formatted as a UUID string that
/// Qdrant accepts natively. Collision-resistant unlike the prior FNV-1a.
fn string_to_point_id(id: &str) -> PointId {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(id.as_bytes());
    let uuid = format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]),
        u16::from_be_bytes([hash[4], hash[5]]),
        u16::from_be_bytes([hash[6], hash[7]]),
        u16::from_be_bytes([hash[8], hash[9]]),
        u64::from_be_bytes([0, 0, hash[10], hash[11], hash[12], hash[13], hash[14], hash[15]]),
    );
    PointId::from(uuid)
}

fn point_id_to_string(id: &PointId) -> String {
    match &id.point_id_options {
        Some(PointIdOptions::Num(n)) => n.to_string(),
        Some(PointIdOptions::Uuid(u)) => u.clone(),
        None => String::new(),
    }
}

fn extract_original_id(payload: &HashMap<String, QdrantValue>) -> Option<String> {
    payload.get(SPHEREQL_ID_KEY).and_then(|v| match &v.kind {
        Some(Kind::StringValue(s)) => Some(s.clone()),
        _ => None,
    })
}

// \u2500\u2500 Vector extraction \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

fn extract_vector_f64(vectors: &Option<VectorsOutput>) -> Option<Vec<f64>> {
    let v = vectors.as_ref()?;
    match &v.vectors_options {
        Some(VectorsOptions::Vector(vec)) => match vec.clone().into_vector() {
            VectorOutputVariant::Dense(dense) => Some(f32_to_f64(&dense.data)),
            _ => None,
        },
        _ => None,
    }
}

// \u2500\u2500 Payload conversion \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

fn metadata_to_payload(
    metadata: &HashMap<String, serde_json::Value>,
) -> HashMap<String, QdrantValue> {
    metadata
        .iter()
        .map(|(k, v)| (k.clone(), json_to_qdrant_value(v)))
        .collect()
}

fn payload_to_metadata(
    payload: &HashMap<String, QdrantValue>,
) -> HashMap<String, serde_json::Value> {
    payload
        .iter()
        .filter(|(k, _)| !k.starts_with("_sphereql_"))
        .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
        .collect()
}

fn json_to_qdrant_value(v: &serde_json::Value) -> QdrantValue {
    match v {
        serde_json::Value::Null => QdrantValue {
            kind: Some(Kind::NullValue(0)),
        },
        serde_json::Value::Bool(b) => QdrantValue {
            kind: Some(Kind::BoolValue(*b)),
        },
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                QdrantValue {
                    kind: Some(Kind::IntegerValue(i)),
                }
            } else {
                QdrantValue {
                    kind: Some(Kind::DoubleValue(n.as_f64().unwrap_or(0.0))),
                }
            }
        }
        serde_json::Value::String(s) => QdrantValue {
            kind: Some(Kind::StringValue(s.clone())),
        },
        serde_json::Value::Array(arr) => QdrantValue {
            kind: Some(Kind::ListValue(qdrant_client::qdrant::ListValue {
                values: arr.iter().map(json_to_qdrant_value).collect(),
            })),
        },
        serde_json::Value::Object(map) => QdrantValue {
            kind: Some(Kind::StructValue(qdrant_client::qdrant::Struct {
                fields: map
                    .iter()
                    .map(|(k, v)| (k.clone(), json_to_qdrant_value(v)))
                    .collect(),
            })),
        },
    }
}

fn qdrant_value_to_json(v: &QdrantValue) -> serde_json::Value {
    match &v.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::json!(b),
        Some(Kind::IntegerValue(i)) => serde_json::json!(i),
        Some(Kind::DoubleValue(d)) => serde_json::Value::Number(
            serde_json::Number::from_f64(*d).unwrap_or_else(|| serde_json::Number::from(0)),
        ),
        Some(Kind::StringValue(s)) => serde_json::json!(s),
        Some(Kind::ListValue(l)) => {
            serde_json::Value::Array(l.values.iter().map(qdrant_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => {
            let map: serde_json::Map<String, serde_json::Value> = s
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        None => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_id_deterministic() {
        let a = string_to_point_id("hello");
        let b = string_to_point_id("hello");
        let c = string_to_point_id("world");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn point_id_is_uuid_format() {
        let id = string_to_point_id("my-record-id");
        let s = point_id_to_string(&id);
        assert_eq!(s.len(), 36);
        assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
    }

    #[test]
    fn f64_f32_roundtrip_precision() {
        let original = vec![1.0, 0.123_456_789_012_345_6, -99.99];
        let f32_vec = f64_to_f32(&original);
        let recovered = f32_to_f64(&f32_vec);

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn json_to_qdrant_roundtrip() {
        let original = serde_json::json!({
            "string": "hello",
            "int": 42,
            "float": 3.5,
            "bool": true,
            "null": null,
            "array": [1, 2, 3],
            "nested": { "key": "value" }
        });

        if let serde_json::Value::Object(map) = &original {
            for (k, v) in map {
                let qv = json_to_qdrant_value(v);
                let back = qdrant_value_to_json(&qv);
                assert_eq!(
                    v, &back,
                    "roundtrip failed for key '{k}': {v:?} != {back:?}"
                );
            }
        }
    }

    #[test]
    fn payload_strips_sphereql_keys_on_read() {
        let mut payload = HashMap::new();
        payload.insert(
            "_sphereql_id".to_string(),
            QdrantValue {
                kind: Some(Kind::StringValue("test-id".to_string())),
            },
        );
        payload.insert(
            "_sphereql_r".to_string(),
            QdrantValue {
                kind: Some(Kind::DoubleValue(1.5)),
            },
        );
        payload.insert(
            "user_key".to_string(),
            QdrantValue {
                kind: Some(Kind::StringValue("user_value".to_string())),
            },
        );

        let meta = payload_to_metadata(&payload);
        assert!(!meta.contains_key("_sphereql_id"));
        assert!(!meta.contains_key("_sphereql_r"));
        assert_eq!(meta["user_key"], serde_json::json!("user_value"));
    }

    #[test]
    fn extract_original_id_from_payload() {
        let mut payload = HashMap::new();
        payload.insert(
            SPHEREQL_ID_KEY.to_string(),
            QdrantValue {
                kind: Some(Kind::StringValue("my-id".to_string())),
            },
        );
        assert_eq!(extract_original_id(&payload), Some("my-id".to_string()));
    }

    #[test]
    fn extract_original_id_missing() {
        let payload = HashMap::new();
        assert_eq!(extract_original_id(&payload), None);
    }
}
