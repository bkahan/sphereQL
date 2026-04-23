use wasm_bindgen::prelude::*;

use sphereql_embed::category::{
    BridgeItem, CategoryPath, CategoryPathStep, CategorySummary, DrillDownResult, InnerSphereReport,
};
use sphereql_embed::config::PipelineConfig;
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
        let input = parse_input(input_json)?;
        Ok(Pipeline {
            inner: SphereQLPipeline::new(input).map_err(|e| JsError::new(&e.to_string()))?,
        })
    }

    /// Create a new pipeline with an explicit [`PipelineConfig`].
    ///
    /// `config_json` is the serde JSON representation of `PipelineConfig`.
    /// Any field may be omitted — missing keys fall back to `PipelineConfig::default`.
    /// This is the entry point for selecting a non-PCA projection family
    /// (e.g. `{"projection_kind": "LaplacianEigenmap"}`) or for overriding
    /// bridge / inner-sphere / routing thresholds from the browser.
    #[wasm_bindgen(js_name = newWithConfig)]
    pub fn new_with_config(input_json: &str, config_json: &str) -> Result<Pipeline, JsError> {
        let input = parse_input(input_json)?;
        let config: PipelineConfig = serde_json::from_str(config_json)
            .map_err(|e| JsError::new(&format!("invalid PipelineConfig JSON: {e}")))?;
        Ok(Pipeline {
            inner: SphereQLPipeline::new_with_config(input, config)
                .map_err(|e| JsError::new(&e.to_string()))?,
        })
    }

    /// Return the [`PipelineConfig`] this pipeline was built with, as a JSON string.
    pub fn config(&self) -> Result<String, JsError> {
        serde_json::to_string(self.inner.config()).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Short stable name of the projection family — "pca", "kernel_pca",
    /// or "laplacian_eigenmap".
    #[wasm_bindgen(js_name = projectionKind)]
    pub fn projection_kind(&self) -> String {
        self.inner.projection_kind().name().to_string()
    }

    /// Query: k nearest neighbors.
    /// Returns JSON: `[{id, category, distance}, ...]`
    pub fn nearest(&self, query_json: &str, k: usize) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.projection().dimensionality();
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
        let expected_dim = self.inner.projection().dimensionality();
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
        let expected_dim = self.inner.projection().dimensionality();
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
                            hop_distance: s.hop_distance,
                            bridge_strength: s.bridge_strength,
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
        let expected_dim = self.inner.projection().dimensionality();
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
        let expected_dim = self.inner.projection().dimensionality();
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

    // ── Category Enrichment Layer ──────────────────────────────────────

    /// Find the shortest path between two categories through the category graph.
    /// Returns JSON: `{steps: [{category_index, category_name, cumulative_distance, bridges_to_next}], total_distance}` or `null`.
    pub fn category_concept_path(
        &self,
        source_category: &str,
        target_category: &str,
    ) -> Result<String, JsError> {
        let pq = self.dummy_query();
        let result = self.inner.query(
            SphereQLQuery::CategoryConceptPath {
                source_category,
                target_category,
            },
            &pq,
        );
        match result {
            SphereQLOutput::CategoryConceptPath(path) => {
                serde_json::to_string(&path.map(CategoryPathOut::from))
                    .map_err(|e| JsError::new(&e.to_string()))
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Find the k nearest neighbor categories to the given category.
    /// Returns JSON: `[{name, member_count, centroid_theta, centroid_phi, angular_spread, cohesion}, ...]`
    pub fn category_neighbors(&self, category: &str, k: usize) -> Result<String, JsError> {
        let pq = self.dummy_query();
        let result = self
            .inner
            .query(SphereQLQuery::CategoryNeighbors { category, k }, &pq);
        match result {
            SphereQLOutput::CategoryNeighbors(summaries) => serde_json::to_string(
                &summaries
                    .iter()
                    .map(CategorySummaryOut::from)
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| JsError::new(&e.to_string())),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Drill down into a category: k-NN within the category using the
    /// inner sphere's projection if available.
    /// Returns JSON: `[{item_index, distance, used_inner_sphere}, ...]`
    pub fn drill_down(
        &self,
        category: &str,
        k: usize,
        query_json: &str,
    ) -> Result<String, JsError> {
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.projection().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let result = self
            .inner
            .query(SphereQLQuery::DrillDown { category, k }, &emb);
        match result {
            SphereQLOutput::DrillDown(results) => {
                serde_json::to_string(&results.iter().map(DrillDownOut::from).collect::<Vec<_>>())
                    .map_err(|e| JsError::new(&e.to_string()))
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Get summary statistics for all categories and inner spheres.
    /// Returns JSON: `{summaries: [...], inner_sphere_reports: [...]}`
    pub fn category_stats(&self) -> Result<String, JsError> {
        let pq = self.dummy_query();
        let result = self.inner.query(SphereQLQuery::CategoryStats, &pq);
        match result {
            SphereQLOutput::CategoryStats {
                summaries,
                inner_sphere_reports,
            } => serde_json::to_string(&CategoryStatsOut {
                summaries: summaries.iter().map(CategorySummaryOut::from).collect(),
                inner_sphere_reports: inner_sphere_reports
                    .iter()
                    .map(InnerSphereReportOut::from)
                    .collect(),
            })
            .map_err(|e| JsError::new(&e.to_string())),
            _ => Err(JsError::new("unexpected output type")),
        }
    }
}

impl Pipeline {
    fn dummy_query(&self) -> PipelineQuery {
        PipelineQuery {
            embedding: vec![0.0; self.inner.projection().dimensionality()],
        }
    }
}

// ── JSON helpers ────────────────────────────────────────────────────────

fn parse_input(input_json: &str) -> Result<PipelineInput, JsError> {
    let parsed: serde_json::Value =
        serde_json::from_str(input_json).map_err(|e| JsError::new(&e.to_string()))?;

    let categories: Vec<String> = parsed["categories"]
        .as_array()
        .ok_or_else(|| JsError::new("missing 'categories' array"))?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_str()
                .ok_or_else(|| JsError::new(&format!("category at index {i} must be a string")))
                .map(|s| s.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let embeddings: Vec<Vec<f64>> = parsed["embeddings"]
        .as_array()
        .ok_or_else(|| JsError::new("missing 'embeddings' array"))?
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let arr = row
                .as_array()
                .ok_or_else(|| JsError::new(&format!("embedding at index {i} must be an array")))?;
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

    Ok(PipelineInput {
        categories,
        embeddings,
    })
}

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
    hop_distance: f64,
    bridge_strength: Option<f64>,
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

// ── Category Enrichment output types ─────────────────────────────────

#[derive(serde::Serialize)]
struct CategorySummaryOut {
    name: String,
    member_count: usize,
    centroid_theta: f64,
    centroid_phi: f64,
    angular_spread: f64,
    cohesion: f64,
}

impl From<&CategorySummary> for CategorySummaryOut {
    fn from(s: &CategorySummary) -> Self {
        Self {
            name: s.name.clone(),
            member_count: s.member_count,
            centroid_theta: s.centroid_position.theta,
            centroid_phi: s.centroid_position.phi,
            angular_spread: s.angular_spread,
            cohesion: s.cohesion,
        }
    }
}

#[derive(serde::Serialize)]
struct BridgeItemOut {
    item_index: usize,
    source_category: usize,
    target_category: usize,
    affinity_to_source: f64,
    affinity_to_target: f64,
    bridge_strength: f64,
}

impl From<&BridgeItem> for BridgeItemOut {
    fn from(b: &BridgeItem) -> Self {
        Self {
            item_index: b.item_index,
            source_category: b.source_category,
            target_category: b.target_category,
            affinity_to_source: b.affinity_to_source,
            affinity_to_target: b.affinity_to_target,
            bridge_strength: b.bridge_strength,
        }
    }
}

#[derive(serde::Serialize)]
struct CategoryPathStepOut {
    category_index: usize,
    category_name: String,
    cumulative_distance: f64,
    bridges_to_next: Vec<BridgeItemOut>,
}

impl From<&CategoryPathStep> for CategoryPathStepOut {
    fn from(s: &CategoryPathStep) -> Self {
        Self {
            category_index: s.category_index,
            category_name: s.category_name.clone(),
            cumulative_distance: s.cumulative_distance,
            bridges_to_next: s.bridges_to_next.iter().map(BridgeItemOut::from).collect(),
        }
    }
}

#[derive(serde::Serialize)]
struct CategoryPathOut {
    total_distance: f64,
    steps: Vec<CategoryPathStepOut>,
}

impl From<CategoryPath> for CategoryPathOut {
    fn from(p: CategoryPath) -> Self {
        Self {
            total_distance: p.total_distance,
            steps: p.steps.iter().map(CategoryPathStepOut::from).collect(),
        }
    }
}

#[derive(serde::Serialize)]
struct DrillDownOut {
    item_index: usize,
    distance: f64,
    used_inner_sphere: bool,
}

impl From<&DrillDownResult> for DrillDownOut {
    fn from(r: &DrillDownResult) -> Self {
        Self {
            item_index: r.item_index,
            distance: r.distance,
            used_inner_sphere: r.used_inner_sphere,
        }
    }
}

#[derive(serde::Serialize)]
struct InnerSphereReportOut {
    category_name: String,
    category_index: usize,
    member_count: usize,
    projection_type: String,
    inner_evr: f64,
    global_subset_evr: f64,
    evr_improvement: f64,
}

impl From<&InnerSphereReport> for InnerSphereReportOut {
    fn from(r: &InnerSphereReport) -> Self {
        Self {
            category_name: r.category_name.clone(),
            category_index: r.category_index,
            member_count: r.member_count,
            projection_type: r.projection_type.to_string(),
            inner_evr: r.inner_evr,
            global_subset_evr: r.global_subset_evr,
            evr_improvement: r.evr_improvement,
        }
    }
}

#[derive(serde::Serialize)]
struct CategoryStatsOut {
    summaries: Vec<CategorySummaryOut>,
    inner_sphere_reports: Vec<InnerSphereReportOut>,
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
