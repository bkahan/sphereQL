use wasm_bindgen::prelude::*;

use sphereql_embed::category::{
    BridgeClassification, BridgeItem, CategoryPath, CategoryPathStep, CategorySummary,
    DrillDownResult, InnerSphereReport,
};
use sphereql_embed::confidence::{ProjectionWarning, WarningSeverity};
use sphereql_embed::config::PipelineConfig;
use sphereql_embed::corpus_features::CorpusFeatures;
use sphereql_embed::domain_groups::DomainGroup;
use sphereql_embed::feedback::{FeedbackAggregator, FeedbackEvent, FeedbackSummary};
use sphereql_embed::laplacian::{
    DEFAULT_ACTIVE_THRESHOLD, DEFAULT_K_NEIGHBORS, LaplacianEigenmapProjection,
};
use sphereql_embed::meta_model::{
    DistanceWeightedMetaModel, MetaModel, MetaTrainingRecord, NearestNeighborMetaModel,
};
use sphereql_embed::pipeline::{
    GlobSummary, NearestResult, PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline,
    SphereQLQuery,
};
use sphereql_embed::projection::Projection;
use sphereql_embed::quality_metric::{
    BridgeCoherence, ClusterSilhouette, CompositeMetric, GraphModularity, QualityMetric,
    TerritorialHealth,
};
use sphereql_embed::tuner::{SearchSpace, SearchStrategy, TuneReport, auto_tune as rust_auto_tune};
use sphereql_embed::types::{Embedding, RadialStrategy};

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

    /// k nearest neighbors of the query.
    pub fn nearest(&self, query_json: &str, k: usize) -> Result<Vec<NearestOut>, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
        let result = self
            .inner
            .query(SphereQLQuery::Nearest { k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::Nearest(items) => Ok(items.iter().map(NearestOut::from).collect()),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// All items above a cosine similarity threshold.
    pub fn similar_above(
        &self,
        query_json: &str,
        min_cosine: f64,
    ) -> Result<Vec<NearestOut>, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
        let result = self
            .inner
            .query(SphereQLQuery::SimilarAbove { min_cosine }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::KNearest(items) => Ok(items.iter().map(NearestOut::from).collect()),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Shortest concept path between two indexed items; `None` when no
    /// path exists.
    pub fn concept_path(
        &self,
        source_id: &str,
        target_id: &str,
        graph_k: usize,
        query_json: &str,
    ) -> Result<Option<PathOut>, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
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
            SphereQLOutput::ConceptPath(path) => Ok(path.map(|p| PathOut {
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
            })),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Detect concept globs. `k = 0` → auto-detect via silhouette;
    /// otherwise fit exactly `k` clusters.
    pub fn detect_globs(
        &self,
        k: usize,
        max_k: usize,
        query_json: &str,
    ) -> Result<Vec<GlobOut>, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
        let k_opt = if k == 0 { None } else { Some(k) };
        let result = self
            .inner
            .query(SphereQLQuery::DetectGlobs { k: k_opt, max_k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::Globs(globs) => Ok(globs.iter().map(GlobOut::from).collect()),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Fit a local manifold around the query point.
    pub fn local_manifold(
        &self,
        query_json: &str,
        neighborhood_k: usize,
    ) -> Result<ManifoldOut, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
        let result = self
            .inner
            .query(SphereQLQuery::LocalManifold { neighborhood_k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::LocalManifold(m) => Ok(ManifoldOut {
                centroid: m.centroid,
                normal: m.normal,
                variance_ratio: m.variance_ratio,
            }),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Shared dimension-mismatch guard for every query-taking method —
    /// the pipeline panics internally on a dim mismatch; we catch it at
    /// the WASM boundary and surface a clean error instead.
    fn require_matching_dim(&self, query: &PipelineQuery) -> Result<(), JsError> {
        let expected = self.inner.projection().dimensionality();
        if query.embedding.len() != expected {
            return Err(JsError::new(&format!(
                "query dimension mismatch: expected {expected}, got {}",
                query.embedding.len()
            )));
        }
        Ok(())
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
    #[wasm_bindgen(js_name = hierarchicalNearest)]
    pub fn hierarchical_nearest(
        &self,
        query_json: &str,
        k: usize,
    ) -> Result<Vec<NearestOut>, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
        let embedding = Embedding::new(emb.embedding);
        let results = self.inner.hierarchical_nearest(&embedding, k);
        Ok(results.iter().map(NearestOut::from).collect())
    }

    /// Coarse domain groups detected from category geometry.
    #[wasm_bindgen(js_name = domainGroups)]
    pub fn domain_groups(&self) -> Vec<DomainGroupOut> {
        self.inner.domain_groups().iter().map(Into::into).collect()
    }

    /// Structured projection-quality warnings. Empty array when EVR
    /// is above `warn_below_evr`.
    #[wasm_bindgen(js_name = projectionWarnings)]
    pub fn projection_warnings(&self) -> Vec<ProjectionWarningOut> {
        self.inner
            .projection_warnings()
            .iter()
            .map(Into::into)
            .collect()
    }

    // ── Category Enrichment Layer ──────────────────────────────────────

    /// Shortest path between two categories through the category graph.
    /// `None` when no path exists.
    pub fn category_concept_path(
        &self,
        source_category: &str,
        target_category: &str,
    ) -> Result<Option<CategoryPathOut>, JsError> {
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
            SphereQLOutput::CategoryConceptPath(path) => Ok(path.map(CategoryPathOut::from)),
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// The k nearest neighbor categories to the given category.
    pub fn category_neighbors(
        &self,
        category: &str,
        k: usize,
    ) -> Result<Vec<CategorySummaryOut>, JsError> {
        let pq = self.dummy_query();
        let result = self
            .inner
            .query(SphereQLQuery::CategoryNeighbors { category, k }, &pq)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::CategoryNeighbors(summaries) => {
                Ok(summaries.iter().map(CategorySummaryOut::from).collect())
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Drill down into a category: k-NN within the category using the
    /// inner sphere's projection when one is available.
    pub fn drill_down(
        &self,
        category: &str,
        k: usize,
        query_json: &str,
    ) -> Result<Vec<DrillDownOut>, JsError> {
        let emb = parse_query(query_json)?;
        self.require_matching_dim(&emb)?;
        let result = self
            .inner
            .query(SphereQLQuery::DrillDown { category, k }, &emb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::DrillDown(results) => {
                Ok(results.iter().map(DrillDownOut::from).collect())
            }
            _ => Err(JsError::new("unexpected output type")),
        }
    }

    /// Summary statistics for all categories and inner spheres.
    pub fn category_stats(&self) -> Result<CategoryStatsOut, JsError> {
        let pq = self.dummy_query();
        let result = self
            .inner
            .query(SphereQLQuery::CategoryStats, &pq)
            .map_err(|e| JsError::new(&e.to_string()))?;
        match result {
            SphereQLOutput::CategoryStats {
                summaries,
                inner_sphere_reports,
            } => Ok(CategoryStatsOut {
                summaries: summaries.iter().map(CategorySummaryOut::from).collect(),
                inner_sphere_reports: inner_sphere_reports
                    .iter()
                    .map(InnerSphereReportOut::from)
                    .collect(),
            }),
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
                    let f = v.as_f64().ok_or_else(|| {
                        JsError::new(&format!("embedding[{i}][{j}] must be a number"))
                    })?;
                    if !f.is_finite() {
                        return Err(JsError::new(&format!(
                            "embedding[{i}][{j}] must be finite"
                        )));
                    }
                    Ok(f)
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

// ── Pipeline output types (typed, tsify-exported) ────────────────────
//
// Each struct below is mirrored as a TypeScript `interface` via `tsify`
// with `into_wasm_abi`, so methods that used to return JSON-stringified
// blobs can return strongly-typed values directly. Callers receive a
// plain JS object whose shape matches the TS interface — no JSON.parse
// step required.

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct NearestOut {
    pub id: String,
    pub category: String,
    pub distance: f64,
    pub certainty: f64,
    pub intensity: f64,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct PathOut {
    pub total_distance: f64,
    pub steps: Vec<PathStepOut>,
}

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct PathStepOut {
    pub id: String,
    pub category: String,
    pub cumulative_distance: f64,
    pub hop_distance: f64,
    pub bridge_strength: Option<f64>,
}

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct GlobOut {
    pub id: usize,
    pub centroid: [f64; 3],
    pub member_count: usize,
    pub radius: f64,
    pub top_categories: Vec<(String, usize)>,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct ManifoldOut {
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub variance_ratio: f64,
}

// ── Category Enrichment output types ─────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct CategorySummaryOut {
    pub name: String,
    pub member_count: usize,
    pub centroid_theta: f64,
    pub centroid_phi: f64,
    pub angular_spread: f64,
    pub cohesion: f64,
    pub bridge_quality: f64,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct BridgeItemOut {
    pub item_index: usize,
    pub source_category: usize,
    pub target_category: usize,
    pub affinity_to_source: f64,
    pub affinity_to_target: f64,
    pub bridge_strength: f64,
    pub classification: String,
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
            classification: classification_name(b.classification).to_string(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct CategoryPathStepOut {
    pub category_index: usize,
    pub category_name: String,
    pub cumulative_distance: f64,
    pub bridges_to_next: Vec<BridgeItemOut>,
    pub hop_confidence: f64,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct CategoryPathOut {
    pub total_distance: f64,
    pub steps: Vec<CategoryPathStepOut>,
    pub path_confidence: f64,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct DrillDownOut {
    pub item_index: usize,
    pub distance: f64,
    pub used_inner_sphere: bool,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct InnerSphereReportOut {
    pub category_name: String,
    pub category_index: usize,
    pub member_count: usize,
    pub projection_type: String,
    pub inner_evr: f64,
    pub global_subset_evr: f64,
    pub evr_improvement: f64,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct CategoryStatsOut {
    pub summaries: Vec<CategorySummaryOut>,
    pub inner_sphere_reports: Vec<InnerSphereReportOut>,
}

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct DomainGroupOut {
    pub member_categories: Vec<usize>,
    pub category_names: Vec<String>,
    pub centroid_theta: f64,
    pub centroid_phi: f64,
    pub angular_spread: f64,
    pub cohesion: f64,
    pub total_items: usize,
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct ProjectionWarningOut {
    pub message: String,
    pub evr: f64,
    pub severity: String,
}

impl From<&ProjectionWarning> for ProjectionWarningOut {
    fn from(w: &ProjectionWarning) -> Self {
        Self {
            message: w.message.clone(),
            evr: w.evr,
            severity: severity_name(w.severity).to_string(),
        }
    }
}

// ── Metalearning surface ──────────────────────────────────────────────

/// Extract a [`CorpusFeatures`] profile from categorized embeddings.
///
/// `input_json` has the same shape as [`Pipeline::new`]:
/// `{ "categories": [...], "embeddings": [[...], ...] }`.
#[wasm_bindgen(js_name = corpusFeatures)]
pub fn corpus_features(input_json: &str) -> Result<CorpusFeaturesOut, JsError> {
    let input = parse_input(input_json)?;
    let features = CorpusFeatures::extract(&input.categories, &input.embeddings);
    Ok(CorpusFeaturesOut::from(&features))
}

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct CorpusFeaturesOut {
    pub n_items: usize,
    pub n_categories: usize,
    pub dim: usize,
    pub mean_members_per_category: f64,
    pub category_size_entropy: f64,
    pub mean_sparsity: f64,
    pub axis_utilization_entropy: f64,
    pub noise_estimate: f64,
    pub mean_intra_category_similarity: f64,
    pub mean_inter_category_similarity: f64,
    pub category_separation_ratio: f64,
}

impl From<&CorpusFeatures> for CorpusFeaturesOut {
    fn from(f: &CorpusFeatures) -> Self {
        Self {
            n_items: f.n_items,
            n_categories: f.n_categories,
            dim: f.dim,
            mean_members_per_category: f.mean_members_per_category,
            category_size_entropy: f.category_size_entropy,
            mean_sparsity: f.mean_sparsity,
            axis_utilization_entropy: f.axis_utilization_entropy,
            noise_estimate: f.noise_estimate,
            mean_intra_category_similarity: f.mean_intra_category_similarity,
            mean_inter_category_similarity: f.mean_inter_category_similarity,
            category_separation_ratio: f.category_separation_ratio,
        }
    }
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

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct TuneReportOut {
    pub metric_name: String,
    pub best_score: f64,
    // `PipelineConfig` is a foreign type (lives in `sphereql-embed`) so
    // we can't derive Tsify on it here. Override the TS type to a
    // structural `object` — the JSON round-trips fine and callers can
    // pass the whole value straight back into `Pipeline.newWithConfig`.
    #[tsify(type = "object")]
    pub best_config: PipelineConfig,
    pub trials_count: usize,
    pub failures_count: usize,
    pub mean_score: f64,
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
/// To construct the tuned pipeline, pass the returned `best_config`
/// object straight back into [`Pipeline::newWithConfig`].
#[wasm_bindgen(js_name = autoTune)]
pub fn auto_tune(input_json: &str, opts_json: &str) -> Result<TuneReportOut, JsError> {
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

    let (_pipeline, report) = rust_auto_tune(input, &space, metric.as_ref(), strategy, &base)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(TuneReportOut::from_report(&report))
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
    pub fn corpus_ids(&self) -> Vec<String> {
        self.inner.corpus_ids()
    }

    /// Summarize one corpus. `None` when no events exist.
    pub fn summarize(&self, corpus_id: &str) -> Option<FeedbackSummaryOut> {
        self.inner
            .summarize(corpus_id)
            .map(FeedbackSummaryOut::from)
    }

    /// Summarize every corpus with recorded events.
    #[wasm_bindgen(js_name = summarizeAll)]
    pub fn summarize_all(&self) -> Vec<FeedbackSummaryOut> {
        self.inner
            .summarize_all()
            .into_iter()
            .map(FeedbackSummaryOut::from)
            .collect()
    }
}

#[derive(serde::Serialize, serde::Deserialize, tsify::Tsify)]
#[tsify(into_wasm_abi)]
pub struct FeedbackSummaryOut {
    pub corpus_id: String,
    pub n_events: usize,
    pub mean_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

impl From<FeedbackSummary> for FeedbackSummaryOut {
    fn from(s: FeedbackSummary) -> Self {
        Self {
            corpus_id: s.corpus_id,
            n_events: s.n_events,
            mean_score: s.mean_score,
            min_score: s.min_score,
            max_score: s.max_score,
        }
    }
}

impl Default for WasmFeedbackAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ── LaplacianEigenmapProjection ────────────────────────────────────────
//
// Standalone projection class — mirrors the Rust `LaplacianEigenmapProjection`
// API for callers that want to project embeddings without standing up a full
// pipeline. Inputs arrive as JSON arrays so the binding works uniformly from
// JS, Node and browsers without a numpy-equivalent type.

fn parse_embeddings_2d(json: &str) -> Result<Vec<Embedding>, JsError> {
    let parsed: Vec<Vec<f64>> =
        serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))?;
    for (i, row) in parsed.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(JsError::new(&format!(
                    "embedding[{i}][{j}] must be finite (no NaN or Inf)"
                )));
            }
        }
    }
    Ok(parsed.into_iter().map(Embedding::new).collect())
}

fn parse_embedding_1d(json: &str) -> Result<Embedding, JsError> {
    let parsed: Vec<f64> = serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))?;
    for (i, v) in parsed.iter().enumerate() {
        if !v.is_finite() {
            return Err(JsError::new(&format!(
                "embedding[{i}] must be finite (no NaN or Inf)"
            )));
        }
    }
    Ok(Embedding::new(parsed))
}

fn parse_radial_strategy(json: Option<&str>) -> Result<RadialStrategy, JsError> {
    let Some(raw) = json else {
        return Ok(RadialStrategy::Magnitude);
    };
    if let Ok(s) = serde_json::from_str::<String>(raw) {
        match s.as_str() {
            "magnitude" => Ok(RadialStrategy::Magnitude),
            other => Err(JsError::new(&format!(
                "unknown radial strategy '{other}': expected 'magnitude' or a number"
            ))),
        }
    } else if let Ok(v) = serde_json::from_str::<f64>(raw) {
        Ok(RadialStrategy::Fixed(v))
    } else {
        Err(JsError::new(
            "radial must be the string \"magnitude\" or a JSON number",
        ))
    }
}

#[derive(serde::Serialize)]
struct SphericalPointOut {
    r: f64,
    theta: f64,
    phi: f64,
}

/// Laplacian-eigenmap projection — connectivity-preserving embedding on S².
///
/// Useful when the corpus is sparse and noisy enough that variance-maximizing
/// projections (PCA / kernel PCA) collapse into the noise axes. See the Rust
/// `sphereql_embed::laplacian` module for algorithmic detail.
#[wasm_bindgen(js_name = LaplacianEigenmapProjection)]
pub struct WasmLaplacianEigenmapProjection {
    inner: LaplacianEigenmapProjection,
}

#[wasm_bindgen(js_class = LaplacianEigenmapProjection)]
impl WasmLaplacianEigenmapProjection {
    /// Fit a projection from a JSON 2-D array of embeddings.
    ///
    /// - `embeddings_json`: `[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]`, at least
    ///   4 rows; all rows must share a dimensionality >= 1.
    /// - `radial_json` (optional): `"\"magnitude\""` or a JSON number.
    /// - `k_neighbors` (optional): k-NN graph density; defaults to 15.
    /// - `active_threshold` (optional): absolute-weight cutoff; defaults to 0.05.
    #[wasm_bindgen(constructor)]
    pub fn new(
        embeddings_json: &str,
        radial_json: Option<String>,
        k_neighbors: Option<usize>,
        active_threshold: Option<f64>,
    ) -> Result<WasmLaplacianEigenmapProjection, JsError> {
        let embs = parse_embeddings_2d(embeddings_json)?;
        let radial = parse_radial_strategy(radial_json.as_deref())?;
        let k = k_neighbors.unwrap_or(DEFAULT_K_NEIGHBORS);
        let thresh = active_threshold.unwrap_or(DEFAULT_ACTIVE_THRESHOLD);
        let inner = LaplacianEigenmapProjection::fit_with_params(&embs, k, thresh, radial)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn dimensionality(&self) -> usize {
        self.inner.dimensionality()
    }

    /// Mean of `|μ_k|` across the three retained eigenvalues, in `[0, 1]`.
    #[wasm_bindgen(getter, js_name = connectivityRatio)]
    pub fn connectivity_ratio(&self) -> f64 {
        self.inner.connectivity_ratio()
    }

    /// Same scalar as `connectivityRatio`, exposed under PCA's name for
    /// adaptive-threshold compatibility.
    #[wasm_bindgen(getter, js_name = explainedVarianceRatio)]
    pub fn explained_variance_ratio(&self) -> f64 {
        self.inner.explained_variance_ratio()
    }

    /// Returns `[μ_1, μ_2, μ_3]` as a JSON array string.
    pub fn eigenvalues(&self) -> Result<String, JsError> {
        let vals = self.inner.eigenvalues();
        serde_json::to_string(&vals).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Project a single embedding (JSON array of numbers) to a spherical
    /// point. Returns `{ r, theta, phi }` as JSON.
    pub fn project(&self, embedding_json: &str) -> Result<String, JsError> {
        let emb = parse_embedding_1d(embedding_json)?;
        let p = self.inner.project(&emb);
        let out = SphericalPointOut {
            r: p.r,
            theta: p.theta,
            phi: p.phi,
        };
        serde_json::to_string(&out).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Project a batch (JSON 2-D array). Returns a JSON array of
    /// `{ r, theta, phi }`.
    #[wasm_bindgen(js_name = projectBatch)]
    pub fn project_batch(&self, embeddings_json: &str) -> Result<String, JsError> {
        let embs = parse_embeddings_2d(embeddings_json)?;
        let out: Vec<SphericalPointOut> = embs
            .iter()
            .map(|e| {
                let p = self.inner.project(e);
                SphericalPointOut {
                    r: p.r,
                    theta: p.theta,
                    phi: p.phi,
                }
            })
            .collect();
        serde_json::to_string(&out).map_err(|e| JsError::new(&e.to_string()))
    }
}

// ── Server-side cache (Node.js only, not available in browser WASM) ────
//
// `validate_cache_filename` is only consumed by the `cfg(not(wasm32))`
// arms of `cache_read` / `cache_write`. Gate the helper itself with the
// same cfg so wasm32 builds don't trip `-D dead_code`.

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod laplacian_tests {
    use super::*;

    fn corpus_json() -> &'static str {
        r#"{
            "categories": ["a","b","a","b","a","b","a","b","a","b"],
            "embeddings": [
                [1.0, 0.1, 0.0, 0.2],
                [0.1, 1.0, 0.0, 0.2],
                [0.9, 0.2, 0.1, 0.3],
                [0.2, 0.9, 0.1, 0.3],
                [0.8, 0.3, 0.2, 0.1],
                [0.3, 0.8, 0.2, 0.1],
                [0.85, 0.15, 0.05, 0.25],
                [0.15, 0.85, 0.05, 0.25],
                [0.95, 0.05, 0.1, 0.15],
                [0.05, 0.95, 0.1, 0.15]
            ]
        }"#
    }

    #[test]
    fn standalone_laplacian_fits_and_projects() {
        let embeddings = r#"[
            [1.0, 0.1, 0.0, 0.2],
            [0.1, 1.0, 0.0, 0.2],
            [0.9, 0.2, 0.1, 0.3],
            [0.2, 0.9, 0.1, 0.3],
            [0.8, 0.3, 0.2, 0.1],
            [0.3, 0.8, 0.2, 0.1]
        ]"#;
        let proj = WasmLaplacianEigenmapProjection::new(embeddings, None, Some(3), None)
            .expect("fit failed");
        // `dimensionality` reports input dim (4 here), not the projected
        // output dim — output is always 3 (S²) for any projection.
        assert_eq!(proj.dimensionality(), 4);
        let cr = proj.connectivity_ratio();
        assert!(
            (0.0..=1.0).contains(&cr),
            "connectivity_ratio out of range: {cr}"
        );

        // Default radial strategy is Magnitude → r equals the input
        // embedding's L2 norm (sqrt(0.86) ≈ 0.927 here), not 1.
        let point_json = proj
            .project("[0.9, 0.1, 0.0, 0.2]")
            .expect("project failed");
        let v: serde_json::Value = serde_json::from_str(&point_json).unwrap();
        let r = v["r"].as_f64().unwrap();
        let expected = (0.81f64 + 0.01 + 0.0 + 0.04).sqrt();
        assert!(
            (r - expected).abs() < 1e-6,
            "got r={r}, expected ≈{expected}"
        );
        assert!(v["theta"].as_f64().unwrap().is_finite());
        assert!(v["phi"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn standalone_laplacian_fixed_radial_lands_on_unit_sphere() {
        let embeddings = r#"[
            [1.0, 0.1, 0.0, 0.2],
            [0.1, 1.0, 0.0, 0.2],
            [0.9, 0.2, 0.1, 0.3],
            [0.2, 0.9, 0.1, 0.3],
            [0.8, 0.3, 0.2, 0.1],
            [0.3, 0.8, 0.2, 0.1]
        ]"#;
        let proj =
            WasmLaplacianEigenmapProjection::new(embeddings, Some("1.0".into()), Some(3), None)
                .expect("fit failed");
        let p: serde_json::Value =
            serde_json::from_str(&proj.project("[0.9, 0.1, 0.0, 0.2]").unwrap()).unwrap();
        assert!((p["r"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pipeline_with_laplacian_config_partial() {
        // Just the projection_kind override — rest defaulted.
        let cfg = r#"{"projection_kind": "LaplacianEigenmap"}"#;
        let p = Pipeline::new_with_config(corpus_json(), cfg).expect("pipeline build failed");
        assert_eq!(p.projection_kind(), "laplacian_eigenmap");
    }

    #[test]
    fn pipeline_with_laplacian_config_full() {
        let cfg = r#"{
            "projection_kind": "LaplacianEigenmap",
            "laplacian": { "k_neighbors": 4, "active_threshold": 0.01 }
        }"#;
        let p = Pipeline::new_with_config(corpus_json(), cfg).expect("pipeline build failed");
        assert_eq!(p.projection_kind(), "laplacian_eigenmap");
    }

    #[test]
    fn pipeline_with_default_pca_unchanged() {
        // Regression: omitting projection_kind preserves PCA default.
        let cfg = r#"{}"#;
        let p = Pipeline::new_with_config(corpus_json(), cfg).expect("pipeline build failed");
        assert_eq!(p.projection_kind(), "pca");
    }

    #[test]
    fn nearest_returns_typed_tsify_structs() {
        // Phase 6a: `nearest` now returns `Vec<NearestOut>` directly
        // (typed through tsify on wasm32, plain Rust on native). Verify
        // the shape survives the migration end-to-end.
        let p = Pipeline::new(corpus_json()).expect("pipeline build failed");
        let results = p
            .nearest("[0.9, 0.1, 0.0, 0.2]", 3)
            .expect("nearest failed");
        assert_eq!(results.len(), 3);
        assert!(!results[0].id.is_empty());
        assert!(!results[0].category.is_empty());
        // Distances must be sorted ascending.
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }
}
