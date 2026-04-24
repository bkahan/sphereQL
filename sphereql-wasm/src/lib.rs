use wasm_bindgen::prelude::*;

use sphereql_embed::category::{
    BridgeClassification, BridgeItem, CategoryPath, CategoryPathStep, CategorySummary,
    DrillDownResult, InnerSphereReport,
};
use sphereql_embed::confidence::{ProjectionWarning, WarningSeverity};
use sphereql_embed::config::PipelineConfig;
use sphereql_embed::corpus_features::CorpusFeatures;
use sphereql_embed::domain_groups::DomainGroup;
use sphereql_embed::pipeline::{
    GlobSummary, NearestResult, PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline,
    SphereQLQuery,
};
use sphereql_embed::projection::Projection;
use sphereql_embed::feedback::{FeedbackAggregator, FeedbackEvent};
use sphereql_embed::meta_model::{
    DistanceWeightedMetaModel, MetaModel, MetaTrainingRecord, NearestNeighborMetaModel,
};
use sphereql_embed::quality_metric::{
    BridgeCoherence, ClusterSilhouette, CompositeMetric, GraphModularity, QualityMetric,
    TerritorialHealth,
};
use sphereql_embed::tuner::{SearchSpace, SearchStrategy, TuneReport, auto_tune as rust_auto_tune};

fn classification_name(c: BridgeClassification) -> &'static str {
    match c {
        BridgeClassification::Genuine => "Genuine",
        BridgeClassification::OverlapArtifact => "OverlapArtifact",
        BridgeClassification::Weak => "Weak",
    }
}

fn severity_name(s: WarningSeverity) -> &'static str {
    match s {
        WarningSeverity::Info => "Info",
        WarningSeverity::Warning => "Warning",
        WarningSeverity::Critical => "Critical",
    }
}

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
        let result = self
            .inner
            .query(SphereQLQuery::Nearest { k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
            .query(SphereQLQuery::SimilarAbove { min_cosine }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
        let result = self
            .inner
            .query(
                SphereQLQuery::ConceptPath {
                    source_id,
                    target_id,
                    graph_k,
                },
                &emb,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;
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
            .query(SphereQLQuery::DetectGlobs { k: k_opt, max_k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
            .query(SphereQLQuery::LocalManifold { neighborhood_k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
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

    // ── Hierarchical routing ─────────────────────────────────────────

    /// Hierarchical nearest-neighbor search: group → category → items.
    ///
    /// Falls back to plain k-NN when EVR is above the configured
    /// `low_evr_threshold`; otherwise routes the query through a coarse
    /// domain group before drilling into member categories.
    /// Returns JSON: `[{id, category, distance, certainty, intensity}, ...]`.
    #[wasm_bindgen(js_name = hierarchicalNearest)]
    pub fn hierarchical_nearest(&self, query_json: &str, k: usize) -> Result<String, JsError> {
        use sphereql_embed::types::Embedding;
        let emb = parse_query(query_json)?;
        let expected_dim = self.inner.projection().dimensionality();
        if emb.embedding.len() != expected_dim {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected_dim}, got {}",
                emb.embedding.len()
            )));
        }
        let embedding = Embedding::new(emb.embedding);
        let results = self.inner.hierarchical_nearest(&embedding, k);
        serde_json::to_string(&results.iter().map(NearestOut::from).collect::<Vec<_>>())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Coarse domain groups detected from category geometry.
    /// Returns JSON: `[{member_categories, category_names, centroid_theta, centroid_phi, angular_spread, cohesion, total_items}, ...]`.
    #[wasm_bindgen(js_name = domainGroups)]
    pub fn domain_groups(&self) -> Result<String, JsError> {
        let groups: Vec<DomainGroupOut> =
            self.inner.domain_groups().iter().map(Into::into).collect();
        serde_json::to_string(&groups).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Structured projection-quality warnings. Empty array when EVR
    /// is above `warn_below_evr`.
    /// Returns JSON: `[{message, evr, severity}, ...]`.
    #[wasm_bindgen(js_name = projectionWarnings)]
    pub fn projection_warnings(&self) -> Result<String, JsError> {
        let warnings: Vec<ProjectionWarningOut> = self
            .inner
            .projection_warnings()
            .iter()
            .map(Into::into)
            .collect();
        serde_json::to_string(&warnings).map_err(|e| JsError::new(&e.to_string()))
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
        let result = self
            .inner
            .query(
                SphereQLQuery::CategoryConceptPath {
                    source_category,
                    target_category,
                },
                &pq,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;
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
            .query(SphereQLQuery::CategoryNeighbors { category, k }, &pq)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
            .query(SphereQLQuery::DrillDown { category, k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
        let result = self
            .inner
            .query(SphereQLQuery::CategoryStats, &pq)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
    bridge_quality: f64,
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
            bridge_quality: s.bridge_quality,
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
    classification: &'static str,
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
            classification: classification_name(b.classification),
        }
    }
}

#[derive(serde::Serialize)]
struct CategoryPathStepOut {
    category_index: usize,
    category_name: String,
    cumulative_distance: f64,
    bridges_to_next: Vec<BridgeItemOut>,
    hop_confidence: f64,
}

impl From<&CategoryPathStep> for CategoryPathStepOut {
    fn from(s: &CategoryPathStep) -> Self {
        Self {
            category_index: s.category_index,
            category_name: s.category_name.clone(),
            cumulative_distance: s.cumulative_distance,
            bridges_to_next: s.bridges_to_next.iter().map(BridgeItemOut::from).collect(),
            hop_confidence: s.hop_confidence,
        }
    }
}

#[derive(serde::Serialize)]
struct CategoryPathOut {
    total_distance: f64,
    steps: Vec<CategoryPathStepOut>,
    path_confidence: f64,
}

impl From<CategoryPath> for CategoryPathOut {
    fn from(p: CategoryPath) -> Self {
        Self {
            total_distance: p.total_distance,
            steps: p.steps.iter().map(CategoryPathStepOut::from).collect(),
            path_confidence: p.path_confidence,
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

#[derive(serde::Serialize)]
struct DomainGroupOut {
    member_categories: Vec<usize>,
    category_names: Vec<String>,
    centroid_theta: f64,
    centroid_phi: f64,
    angular_spread: f64,
    cohesion: f64,
    total_items: usize,
}

impl From<&DomainGroup> for DomainGroupOut {
    fn from(g: &DomainGroup) -> Self {
        Self {
            member_categories: g.member_categories.clone(),
            category_names: g.category_names.clone(),
            centroid_theta: g.centroid.theta,
            centroid_phi: g.centroid.phi,
            angular_spread: g.angular_spread,
            cohesion: g.cohesion,
            total_items: g.total_items,
        }
    }
}

#[derive(serde::Serialize)]
struct ProjectionWarningOut {
    message: String,
    evr: f64,
    severity: &'static str,
}

impl From<&ProjectionWarning> for ProjectionWarningOut {
    fn from(w: &ProjectionWarning) -> Self {
        Self {
            message: w.message.clone(),
            evr: w.evr,
            severity: severity_name(w.severity),
        }
    }
}

// ── Metalearning surface ──────────────────────────────────────────────

/// Extract a [`CorpusFeatures`] profile from categorized embeddings.
///
/// `input_json` has the same shape as [`Pipeline::new`]:
/// `{ "categories": [...], "embeddings": [[...], ...] }`.
///
/// Returns the full serde JSON of `CorpusFeatures` — suitable as input
/// to a `MetaModel` on the Rust side, or for logging/audit.
#[wasm_bindgen(js_name = corpusFeatures)]
pub fn corpus_features(input_json: &str) -> Result<String, JsError> {
    let input = parse_input(input_json)?;
    let features = CorpusFeatures::extract(&input.categories, &input.embeddings);
    serde_json::to_string(&features).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(serde::Deserialize, Default)]
struct AutoTuneOpts {
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default = "default_strategy")]
    strategy: String,
    #[serde(default = "default_budget")]
    budget: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default = "default_warmup")]
    warmup: usize,
    #[serde(default = "default_gamma")]
    gamma: f64,
    #[serde(default)]
    base_config: Option<PipelineConfig>,
    #[serde(default)]
    search_space: Option<SearchSpace>,
}

fn default_metric() -> String {
    "default_composite".to_string()
}
fn default_strategy() -> String {
    "random".to_string()
}
fn default_budget() -> usize {
    24
}
fn default_warmup() -> usize {
    4
}
fn default_gamma() -> f64 {
    0.25
}

#[derive(serde::Serialize)]
struct TuneReportOut {
    metric_name: String,
    best_score: f64,
    best_config: PipelineConfig,
    trials_count: usize,
    failures_count: usize,
    mean_score: f64,
}

impl TuneReportOut {
    fn from_report(report: &TuneReport) -> Self {
        Self {
            metric_name: report.metric_name.clone(),
            best_score: report.best_score,
            best_config: report.best_config.clone(),
            trials_count: report.trials.len(),
            failures_count: report.failures.len(),
            mean_score: report.mean_score(),
        }
    }
}

fn resolve_strategy_wasm(
    kind: &str,
    budget: usize,
    seed: u64,
    warmup: usize,
    gamma: f64,
) -> Result<SearchStrategy, JsError> {
    match kind {
        "grid" => Ok(SearchStrategy::Grid),
        "random" => Ok(SearchStrategy::Random { budget, seed }),
        "bayesian" => Ok(SearchStrategy::Bayesian {
            budget,
            warmup,
            gamma,
            seed,
        }),
        other => Err(JsError::new(&format!(
            "unknown strategy {other:?}; expected grid, random, or bayesian"
        ))),
    }
}

/// Resolve a metric name into a boxed `QualityMetric` trait object.
///
/// Mirrors `sphereql-python/src/meta.rs::resolve_metric` — `auto_tune`
/// now accepts `&dyn QualityMetric` (via a `?Sized` bound), so both
/// bindings can share a single match-then-hand-off form.
fn resolve_metric(name: &str) -> Result<Box<dyn QualityMetric>, JsError> {
    match name {
        "territorial_health" => Ok(Box::new(TerritorialHealth)),
        "bridge_coherence" => Ok(Box::new(BridgeCoherence)),
        "cluster_silhouette" => Ok(Box::new(ClusterSilhouette)),
        "graph_modularity" => Ok(Box::new(GraphModularity::default())),
        "default_composite" => Ok(Box::new(CompositeMetric::default_composite())),
        "connectivity_composite" => Ok(Box::new(CompositeMetric::connectivity_composite())),
        other => Err(JsError::new(&format!(
            "unknown metric {other:?}; expected one of: \
             territorial_health, bridge_coherence, cluster_silhouette, \
             graph_modularity, default_composite, connectivity_composite"
        ))),
    }
}

/// Run the auto-tuner and return a summary report.
///
/// `input_json` has the same shape as [`Pipeline::new`]. `opts_json`
/// may be `"{}"` or a JSON object with any of:
/// `metric`, `strategy`, `budget`, `seed`, `warmup`, `gamma`, `base_config`.
///
/// To construct the tuned pipeline, parse `best_config` from the
/// returned report and pass it to [`Pipeline::newWithConfig`].
#[wasm_bindgen(js_name = autoTune)]
pub fn auto_tune(input_json: &str, opts_json: &str) -> Result<String, JsError> {
    let input = parse_input(input_json)?;
    let opts: AutoTuneOpts = serde_json::from_str(opts_json)
        .map_err(|e| JsError::new(&format!("invalid options JSON: {e}")))?;

    let metric = resolve_metric(&opts.metric)?;
    let strategy = resolve_strategy_wasm(
        &opts.strategy,
        opts.budget,
        opts.seed,
        opts.warmup,
        opts.gamma,
    )?;
    let space = opts.search_space.unwrap_or_default();
    let base = opts.base_config.unwrap_or_default();

    let (_pipeline, report) =
        rust_auto_tune(input, &space, metric.as_ref(), strategy, &base)
            .map_err(|e| JsError::new(&e.to_string()))?;

    serde_json::to_string(&TuneReportOut::from_report(&report))
        .map_err(|e| JsError::new(&e.to_string()))
}

// ── MetaModel ────────────────────────────────────────────────────────

fn parse_records(records_json: &str) -> Result<Vec<MetaTrainingRecord>, JsError> {
    serde_json::from_str(records_json)
        .map_err(|e| JsError::new(&format!("invalid training records JSON: {e}")))
}

fn parse_features(features_json: &str) -> Result<CorpusFeatures, JsError> {
    serde_json::from_str(features_json)
        .map_err(|e| JsError::new(&format!("invalid CorpusFeatures JSON: {e}")))
}

fn serialize_config(cfg: &PipelineConfig) -> Result<String, JsError> {
    serde_json::to_string(cfg).map_err(|e| JsError::new(&e.to_string()))
}

/// Nearest-neighbor meta-model: picks the training record whose corpus
/// feature vector is closest to the query.
#[wasm_bindgen(js_name = NearestNeighborMetaModel)]
pub struct WasmNearestNeighborMetaModel {
    inner: NearestNeighborMetaModel,
}

#[wasm_bindgen(js_class = NearestNeighborMetaModel)]
impl WasmNearestNeighborMetaModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: NearestNeighborMetaModel::new(),
        }
    }

    /// Fit on a JSON array of training records.
    pub fn fit(&mut self, records_json: &str) -> Result<(), JsError> {
        let records = parse_records(records_json)?;
        self.inner.fit(&records);
        Ok(())
    }

    /// Predict the PipelineConfig for a new corpus profile. Returns
    /// `PipelineConfig` JSON ready to pass to [`Pipeline::newWithConfig`].
    pub fn predict(&self, features_json: &str) -> Result<String, JsError> {
        let features = parse_features(features_json)?;
        serialize_config(&self.inner.predict(&features))
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

impl Default for WasmNearestNeighborMetaModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Distance-weighted meta-model: balances similarity against observed
/// quality.
#[wasm_bindgen(js_name = DistanceWeightedMetaModel)]
pub struct WasmDistanceWeightedMetaModel {
    inner: DistanceWeightedMetaModel,
}

#[wasm_bindgen(js_class = DistanceWeightedMetaModel)]
impl WasmDistanceWeightedMetaModel {
    #[wasm_bindgen(constructor)]
    pub fn new(epsilon: Option<f64>) -> Self {
        let epsilon = epsilon.unwrap_or(0.1);
        Self {
            inner: DistanceWeightedMetaModel::new().with_epsilon(epsilon),
        }
    }

    pub fn fit(&mut self, records_json: &str) -> Result<(), JsError> {
        let records = parse_records(records_json)?;
        self.inner.fit(&records);
        Ok(())
    }

    pub fn predict(&self, features_json: &str) -> Result<String, JsError> {
        let features = parse_features(features_json)?;
        serialize_config(&self.inner.predict(&features))
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

impl Default for WasmDistanceWeightedMetaModel {
    fn default() -> Self {
        Self::new(None)
    }
}

// ── Feedback primitives ─────────────────────────────────────────────

/// Accumulates [`FeedbackEvent`]s and summarizes them by `corpus_id`.
///
/// Events are passed as JSON objects matching [`FeedbackEvent`]'s serde
/// shape; summaries are returned the same way. Browser WASM has no
/// filesystem — persistence is the caller's responsibility (e.g.
/// localStorage, IndexedDB, or shipping JSON back to a server).
#[wasm_bindgen(js_name = FeedbackAggregator)]
pub struct WasmFeedbackAggregator {
    inner: FeedbackAggregator,
}

#[wasm_bindgen(js_class = FeedbackAggregator)]
impl WasmFeedbackAggregator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: FeedbackAggregator::new(),
        }
    }

    /// Load from a JSON array of events. `"[]"` yields an empty aggregator.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(events_json: &str) -> Result<WasmFeedbackAggregator, JsError> {
        let events: Vec<FeedbackEvent> = serde_json::from_str(events_json)
            .map_err(|e| JsError::new(&format!("invalid events JSON: {e}")))?;
        let mut inner = FeedbackAggregator::new();
        for e in events {
            inner.record(e);
        }
        Ok(Self { inner })
    }

    /// Serialize the full event log to JSON.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Record one event (JSON with `corpus_id`, `query_id`, `score`,
    /// `timestamp`).
    pub fn record(&mut self, event_json: &str) -> Result<(), JsError> {
        let event: FeedbackEvent = serde_json::from_str(event_json)
            .map_err(|e| JsError::new(&format!("invalid event JSON: {e}")))?;
        self.inner.record(event);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[wasm_bindgen(js_name = corpusIds)]
    pub fn corpus_ids(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner.corpus_ids()).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Summarize one corpus. Returns `null` if no events exist.
    pub fn summarize(&self, corpus_id: &str) -> Result<String, JsError> {
        serde_json::to_string(&self.inner.summarize(corpus_id))
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Summarize all corpora as an array of summaries.
    #[wasm_bindgen(js_name = summarizeAll)]
    pub fn summarize_all(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner.summarize_all())
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

impl Default for WasmFeedbackAggregator {
    fn default() -> Self {
        Self::new()
    }
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
