use wasm_bindgen::prelude::*;

use sphereql_embed::pipeline::{
    GlobSummary, NearestResult, PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline,
    SphereQLQuery,
};
use sphereql_embed::projection::Projection;

/// WASM-exposed pipeline. Constructed once with corpus data, then queried
/// repeatedly from JavaScript.
#[wasm_bindgen]
pub struct Pipeline {
    inner: SphereQLPipeline,
}

#[wasm_bindgen]
impl Pipeline {
    /// Create a new pipeline from JSON input.
    ///
    /// Expected JSON shape:
    /// ```json
    /// {
    ///   "categories": ["science", "cooking", ...],
    ///   "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    /// }
    /// ```
    ///
    /// categories.length must equal embeddings.length.
    /// All embedding sub-arrays must have the same length (>= 3).
    #[wasm_bindgen(constructor)]
    pub fn new(input_json: &str) -> Result<Pipeline, JsError> {
        let parsed: serde_json::Value =
            serde_json::from_str(input_json).map_err(|e| JsError::new(&e.to_string()))?;

        let categories: Vec<String> = parsed["categories"]
            .as_array()
            .ok_or_else(|| JsError::new("missing 'categories' array"))?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_str()
                    .ok_or_else(|| {
                        JsError::new(&format!("category at index {i} must be a string"))
                    })
                    .map(|s| s.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let embeddings: Vec<Vec<f64>> = parsed["embeddings"]
            .as_array()
            .ok_or_else(|| JsError::new("missing 'embeddings' array"))?
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let arr = row.as_array().ok_or_else(|| {
                    JsError::new(&format!("embedding at index {i} must be an array"))
                })?;
                arr.iter()
                    .enumerate()
                    .map(|(j, v)| {
                        v.as_f64().ok_or_else(|| {
                            JsError::new(&format!("embedding[{i}][{j}] must be a number"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        if categories.len() != embeddings.len() {
            return Err(JsError::new(&format!(
                "categories.len ({}) != embeddings.len ({})",
                categories.len(),
                embeddings.len()
            )));
        }

        let input = PipelineInput {
            categories,
            embeddings,
        };

        Ok(Pipeline {
            inner: SphereQLPipeline::new(input).map_err(|e| JsError::new(&e.to_string()))?,
        })
    }

    /// Query: k nearest neighbors.
    /// Returns JSON: `[{id, category, distance}, ...]`
    pub fn nearest(&self, query_json: &str, k: usize) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.pca().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let result = self.inner.query(SphereQLQuery::Nearest { k }, &emb);
        match result {
            SphereQLOutput::Nearest(items) => {
                serde_json::to_string(&items.iter().map(NearestOut::from).collect::<Vec<_>>())
                    .map_err(|e| JsError::new(&e.to_string()))
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Query: all items above a cosine similarity threshold.
    /// Returns JSON: `[{id, category, distance}, ...]`
    pub fn similar_above(&self, query_json: &str, min_cosine: f64) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.pca().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let result = self
            .inner
            .query(SphereQLQuery::SimilarAbove { min_cosine }, &emb);
        match result {
            SphereQLOutput::KNearest(items) => {
                serde_json::to_string(&items.iter().map(NearestOut::from).collect::<Vec<_>>())
                    .map_err(|e| JsError::new(&e.to_string()))
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Query: shortest concept path between two indexed items.
    /// Returns JSON: `{steps: [{id, category, cumulative_distance}], total_distance}` or `null`.
    pub fn concept_path(
        &self,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query_json: &str,
    ) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.pca().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let result = self.inner.query(
            SphereQLQuery::ConceptPath {
                source_id,
                target_id,
                graph_k,
            },
            &emb,
        );
        match result {
            SphereQLOutput::ConceptPath(path) => serde_json::to_string(&path.map(|p| {
                PathOut {
                    total_distance: p.total_distance,
                    steps: p
                        .steps
                        .iter()
                        .map(|s| PathStepOut {
                            id: s.id.clone(),
                            category: s.category.clone(),
                            cumulative_distance: s.cumulative_distance,
                        })
                        .collect(),
                }
            }))
            .map_err(|e| JsError::new(&e.to_string())),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Detect concept globs.
    /// k=0 for auto-detection (silhouette), k>0 for fixed count.
    /// Returns JSON: `[{id, centroid, member_count, radius, top_categories}, ...]`
    pub fn detect_globs(
        &self,
        k: usize,
        max_k: usize,
        query_json: &str,
    ) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.pca().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let k_opt = if k == 0 { None } else { Some(k) };
        let result = self
            .inner
            .query(SphereQLQuery::DetectGlobs { k: k_opt, max_k }, &emb);
        match result {
            SphereQLOutput::Globs(globs) => {
                serde_json::to_string(&globs.iter().map(GlobOut::from).collect::<Vec<_>>())
                    .map_err(|e| JsError::new(&e.to_string()))
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Fit a local manifold around the query point.
    /// Returns JSON: `{centroid, normal, variance_ratio}`
    pub fn local_manifold(
        &self,
        query_json: &str,
        neighborhood_k: usize,
    ) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.pca().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let result = self
            .inner
            .query(SphereQLQuery::LocalManifold { neighborhood_k }, &emb);
        match result {
            SphereQLOutput::LocalManifold(m) => serde_json::to_string(&ManifoldOut {
                centroid: m.centroid,
                normal: m.normal,
                variance_ratio: m.variance_ratio,
            })
            .map_err(|e| JsError::new(&e.to_string())),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Export all projected points as JSON.
    pub fn export_json(&self) -> Result<String, JsError> {
        Ok(self.inner.to_json())
    }

    /// PCA explained variance ratio.
    pub fn explained_variance(&self) -> f64 {
        self.inner.explained_variance_ratio()
    }

    /// Number of indexed items.
    pub fn len(&self) -> usize {
        self.inner.num_items()
    }

    /// Whether the pipeline has no indexed items.
    pub fn is_empty(&self) -> bool {
        self.inner.num_items() == 0
    }
}

// ── JSON helpers ────────────────────────────────────────────────────────

fn parse_query(json: &str) -> Result<PipelineQuery, JsError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))?;
    let embedding: Vec<f64> = v
        .as_array()
        .ok_or_else(|| JsError::new("query must be a JSON array of numbers"))?
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let val = x
                .as_f64()
                .ok_or_else(|| JsError::new(&format!("query[{i}] must be a number")))?;
            if !val.is_finite() {
                return Err(JsError::new(&format!(
                    "query[{i}] must be finite (no NaN or Inf)"
                )));
            }
            Ok(val)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PipelineQuery { embedding })
}

// ── Serde output types (mirrors pipeline types for JSON serialization) ──

#[derive(serde::Serialize)]
struct NearestOut {
    id: String,
    category: String,
    distance: f64,
    certainty: f64,
    intensity: f64,
}

impl From<&NearestResult> for NearestOut {
    fn from(r: &NearestResult) -> Self {
        Self {
            id: r.id.clone(),
            category: r.category.clone(),
            distance: r.distance,
            certainty: r.certainty,
            intensity: r.intensity,
        }
    }
}

#[derive(serde::Serialize)]
struct PathOut {
    total_distance: f64,
    steps: Vec<PathStepOut>,
}

#[derive(serde::Serialize)]
struct PathStepOut {
    id: String,
    category: String,
    cumulative_distance: f64,
}

#[derive(serde::Serialize)]
struct GlobOut {
    id: usize,
    centroid: [f64; 3],
    member_count: usize,
    radius: f64,
    top_categories: Vec<(String, usize)>,
}

impl From<&GlobSummary> for GlobOut {
    fn from(g: &GlobSummary) -> Self {
        Self {
            id: g.id,
            centroid: g.centroid,
            member_count: g.member_count,
            radius: g.radius,
            top_categories: g.top_categories.clone(),
        }
    }
}

#[derive(serde::Serialize)]
struct ManifoldOut {
    centroid: [f64; 3],
    normal: [f64; 3],
    variance_ratio: f64,
}

// ── Server-side cache (Node.js only, not available in browser WASM) ────

fn validate_cache_filename(path: &str) -> Result<std::path::PathBuf, JsError> {
    let name = std::path::Path::new(path)
        .file_name()
        .ok_or_else(|| JsError::new("cache path must be a plain filename, not a directory path"))?;
    if name != std::ffi::OsStr::new(path) {
        return Err(JsError::new(
            "cache path must not contain path separators (no directory traversal)",
        ));
    }
    let cache_dir = std::path::PathBuf::from(".sphereql_cache");
    Ok(cache_dir.join(name))
}

/// Read a cached SphereQL result from disk.
/// `path` must be a plain filename (no directory separators). Files are
/// resolved relative to `.sphereql_cache/`.
#[wasm_bindgen]
pub fn cache_read(path: &str) -> Result<Option<String>, JsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = path;
        Ok(None)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let resolved = validate_cache_filename(path)?;
        match std::fs::read_to_string(&resolved) {
            Ok(contents) => Ok(Some(contents)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }
}

/// Write a SphereQL result to disk for caching.
/// `path` must be a plain filename (no directory separators). Files are
/// written to `.sphereql_cache/`.
#[wasm_bindgen]
pub fn cache_write(path: &str, json: &str) -> Result<(), JsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (path, json);
        Err(JsError::new(
            "cache_write is not available in browser WASM — use localStorage or IndexedDB instead",
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let resolved = validate_cache_filename(path)?;
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(|e| JsError::new(&e.to_string()))?;
        }
        std::fs::write(&resolved, json).map_err(|e| JsError::new(&e.to_string()))
    }
}
