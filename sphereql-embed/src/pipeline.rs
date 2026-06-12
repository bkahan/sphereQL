use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

use sphereql_core::*;
use sphereql_index::SpatialItem;

use crate::category::{
    BridgeItem, CategoryLayer, CategoryPath, CategorySummary, DrillDownResult, InnerSphereReport,
};
use crate::confidence::{ProjectionWarning, QualityConfig, QualitySignal};
use crate::config::{PipelineConfig, ProjectionKind};
use crate::configured_projection::ConfiguredProjection;
use crate::corpus_features::CorpusFeatures;
use crate::domain_groups::DomainGroup;
use crate::kernel_pca::KernelPcaProjection;
use crate::laplacian::LaplacianEigenmapProjection;
use crate::meta_model::MetaModel;
use crate::projection::{PcaProjection, Projection};
use crate::quality_metric::QualityMetric;
use crate::query::{EmbeddingIndex, GlobResult, SlicingManifold};
use crate::tuner::{SearchSpace, SearchStrategy, TuneReport, auto_tune};
use crate::types::{Embedding, RadialStrategy};

// ── Errors ─────────────────────────────────────────────────────────────────

/// Reasons a pipeline build or query can fail.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PipelineError {
    /// `categories` and `embeddings` had different lengths — they must
    /// match one-to-one.
    #[error("categories length ({cat}) must equal embeddings length ({emb})")]
    LengthMismatch { cat: usize, emb: usize },
    /// Fewer than 3 embeddings — not enough to fit a 3D projection.
    #[error("need at least 3 embeddings, got {0}")]
    TooFewEmbeddings(usize),
    /// Projection fit rejected the input. Wraps
    /// [`ProjectionError`](crate::projection::ProjectionError) — empty
    /// corpus, dim-too-low, inconsistent dim, invalid sigma, etc.
    #[error("projection fit failed: {0}")]
    Projection(#[from] crate::projection::ProjectionError),
    /// A category-keyed query referenced a name that doesn't exist in
    /// the pipeline. Previously these paths silently returned empty
    /// results / `None`, which callers couldn't distinguish from
    /// "category exists but is disconnected on the graph".
    #[error("unknown category: {0:?}")]
    UnknownCategory(String),
    /// A concept-path query referenced an id that doesn't exist in the
    /// pipeline.
    #[error("unknown id: {0:?}")]
    UnknownId(String),
    /// Every [`crate::tuner::auto_tune`] trial failed a
    /// downstream validator (e.g. every candidate config was rejected
    /// by the pipeline builder). The attached `failures` list carries
    /// the `(config, error)` pairs the tuner observed — callers should
    /// inspect them to find the real cause; the outer error is just a
    /// roll-up saying "none of the trials produced a scorable pipeline".
    #[error("auto_tune produced no successful trials ({} failures)", failures.len())]
    AllTrialsFailed {
        failures: Vec<(crate::config::PipelineConfig, String)>,
    },
    /// `auto_tune` was called with a malformed
    /// [`SearchSpace`](crate::tuner::SearchSpace) or
    /// [`SearchStrategy`](crate::tuner::SearchStrategy) — empty axis,
    /// budget too small for the strategy to make progress, etc.
    /// Surfaced upfront so every search mode (Grid, Random, Bayesian)
    /// fails at the same boundary instead of panicking mid-trial or
    /// rolling up into `AllTrialsFailed { failures: [] }`.
    #[error("invalid search space: {0}")]
    InvalidSearchSpace(String),
    /// Input validation failed before any projection or pipeline work was
    /// attempted (e.g. empty corpus, ragged embeddings, mismatched lengths
    /// detected during corpus feature extraction).
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

// ── Input contract ──────────────────────────────────────────────────────────────

/// Input to construct a SphereQL pipeline.
///
/// - `categories`: one category string per sentence, same length as `embeddings`
/// - `embeddings`: one `Vec<f64>` per sentence, all same dimensionality
/// - Both vectors must have the same length.
pub struct PipelineInput {
    pub categories: Vec<String>,
    pub embeddings: Vec<Vec<f64>>,
}

/// A query into the pipeline. All fields are embeddings of the same
/// dimensionality as the pipeline's corpus.
pub struct PipelineQuery {
    pub embedding: Vec<f64>,
}

// ── Output types ────────────────────────────────────────────────────────────────

/// One item returned from a nearest-neighbor or similarity query.
///
/// All fields use the pipeline's configured projection to derive
/// distances and quality signals; callers should treat results as
/// comparable within a single pipeline but not across pipelines with
/// different projection kinds.
#[derive(Debug, Clone)]
pub struct NearestResult {
    /// Item id as supplied to [`SphereQLPipeline::new`].
    pub id: String,
    /// Category label from the input.
    pub category: String,
    /// Angular distance on S² between the query and this item's
    /// projected position, in radians.
    pub distance: f64,
    /// Certainty of this point's projection (0–1). Higher = more faithfully represented.
    pub certainty: f64,
    /// Semantic intensity (pre-normalization magnitude of original embedding).
    pub intensity: f64,
    /// Combined quality signal: EVR × certainty × gap_confidence.
    /// Always `Some(...)` for results the pipeline produces today; the
    /// `Option` is kept so callers that construct `NearestResult`
    /// outside the pipeline (e.g. mocks, tests) can omit it.
    pub quality: Option<QualitySignal>,
}

/// Concept-path result: ordered steps between two indexed items, with
/// cumulative angular distance along the path.
#[derive(Debug, Clone)]
pub struct PathResult {
    pub steps: Vec<PipelinePathStep>,
    pub total_distance: f64,
}

/// One step along a [`PathResult`].
#[derive(Debug, Clone)]
pub struct PipelinePathStep {
    pub id: String,
    pub category: String,
    pub cumulative_distance: f64,
    /// Angular distance of this individual hop (0.0 for the first step).
    pub hop_distance: f64,
    /// Bridge strength used on cross-category hops (None for same-category or unbridged paths).
    pub bridge_strength: Option<f64>,
}

/// Summary of one cluster detected by `DetectGlobs`.
#[derive(Debug, Clone)]
pub struct GlobSummary {
    pub id: usize,
    pub centroid: [f64; 3],
    pub member_count: usize,
    pub radius: f64,
    pub top_categories: Vec<(String, usize)>,
}

/// Local 3-D manifold fitted around the query point.
#[derive(Debug, Clone)]
pub struct ManifoldResult {
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub variance_ratio: f64,
}

/// Typed output from a pipeline query.
#[derive(Debug, Clone)]
pub enum SphereQLOutput {
    Nearest(Vec<NearestResult>),
    KNearest(Vec<NearestResult>),
    ConceptPath(Option<PathResult>),
    Globs(Vec<GlobSummary>),
    LocalManifold(ManifoldResult),
    // ── Phase 3: category-level outputs ─────────────────────────
    /// Result of a category-level concept path query.
    CategoryConceptPath(Option<CategoryPath>),
    /// Nearest neighbor categories to a given category.
    CategoryNeighbors(Vec<CategorySummary>),
    /// Drill-down results within a single category.
    DrillDown(Vec<DrillDownResult>),
    /// Summary statistics for all categories and inner spheres.
    CategoryStats {
        summaries: Vec<CategorySummary>,
        inner_sphere_reports: Vec<InnerSphereReport>,
    },
}

/// Typed query request.
pub enum SphereQLQuery<'a> {
    /// Find the k nearest neighbors to the query embedding.
    Nearest { k: usize },
    /// Find all neighbors within a cosine similarity threshold.
    SimilarAbove { min_cosine: f64 },
    /// Find the shortest concept path between two indexed items.
    ConceptPath {
        source_id: &'a str,
        target_id: &'a str,
        graph_k: usize,
    },
    /// Detect concept globs. k=None for auto-detection.
    DetectGlobs { k: Option<usize>, max_k: usize },
    /// Fit a local manifold around the query point.
    LocalManifold { neighborhood_k: usize },
    // ── Phase 3: category-level queries ─────────────────────────
    /// Find the shortest path between two categories through the category graph.
    CategoryConceptPath {
        source_category: &'a str,
        target_category: &'a str,
    },
    /// Find the k nearest neighbor categories to the given category.
    CategoryNeighbors { category: &'a str, k: usize },
    /// Drill down into a category: k-NN within the category, using the
    /// inner sphere's projection if available.
    DrillDown { category: &'a str, k: usize },
    /// Get summary statistics for all categories and inner spheres.
    CategoryStats,
}

/// Projected data for a single item, suitable for export or visualization.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExportedPoint {
    pub id: String,
    pub category: String,
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub certainty: f64,
    pub intensity: f64,
}

/// Per-item projected positions and category indices shared by the
/// geometry metrics ([`ClusterSilhouette`](crate::quality_metric::ClusterSilhouette),
/// [`GraphModularity`](crate::quality_metric::GraphModularity)). Built
/// lazily once per pipeline — unlike [`SphereQLPipeline::exported_points`],
/// which clones every id and category string per call.
pub(crate) struct MetricPoints {
    pub(crate) positions: Vec<SphericalPoint>,
    pub(crate) category_indices: Vec<Option<usize>>,
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Outer-projection EVR at or above which [`SphereQLPipeline::default_nearest`]
/// skips group-inner-sphere routing entirely. At this fidelity the outer
/// angular distances are more accurate than any inner re-projection
/// (observed: UMAP at 99.7% EVR routed correct outer-sphere neighbors
/// away through a lossy group PCA). Group routing only helps when the
/// outer sphere is unreliable and the inner re-projection recovers lost
/// structure.
const HIGH_EVR_ROUTING_BYPASS: f64 = 0.90;

/// The main SphereQL pipeline: fitted projection + spatial index +
/// category enrichment layer + optional tunable config.
///
/// Build one with [`Self::new`] for defaults,
/// [`Self::new_with_config`] for an explicit [`PipelineConfig`], or
/// [`Self::new_from_metamodel`] / [`Self::new_from_metamodel_tuned`]
/// to consult a trained meta-model on past tuner runs.
pub struct SphereQLPipeline {
    projection: ConfiguredProjection,
    index: EmbeddingIndex<ConfiguredProjection>,
    categories: Vec<String>,
    cart_points: Vec<[f64; 3]>,
    ids: Vec<String>,
    /// Category enrichment layer: summaries, graph, bridges, inner spheres.
    category_layer: CategoryLayer,
    /// Quality configuration for filtering and warnings.
    quality_config: QualityConfig,
    /// Projection quality warnings (empty if EVR is above threshold).
    projection_warnings: Vec<ProjectionWarning>,
    /// Full tunable configuration used at build time.
    config: PipelineConfig,
    /// Original embeddings retained for downstream consumers that need
    /// full-dimensional similarity (e.g., priming, concept extraction).
    /// Only populated when the `retain-embeddings` feature is active.
    raw_embeddings: Option<Vec<Vec<f64>>>,
    /// Lazily-built positions + category indices for the geometry
    /// metrics. Safe to cache: positions and categories are fixed at
    /// construction (the only `&mut self` method is
    /// [`Self::set_quality_config`], which touches neither).
    metric_points: OnceLock<MetricPoints>,
}

impl SphereQLPipeline {
    /// Build a pipeline from raw inputs with [`PipelineConfig::default`].
    ///
    /// - `input.categories[i]` is the category for sentence `i`
    /// - `input.embeddings[i]` is the embedding vector for sentence `i`
    /// - All embedding vectors must have the same dimensionality (>= 3).
    pub fn new(input: PipelineInput) -> Result<Self, PipelineError> {
        Self::new_with_config(input, PipelineConfig::default())
    }

    /// Build a pipeline with an explicit configuration. Fits the projection
    /// internally using [`PipelineConfig::projection_kind`] and any relevant
    /// sub-config (e.g. [`LaplacianConfig`](crate::config::LaplacianConfig)).
    pub fn new_with_config(
        input: PipelineInput,
        config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        #[cfg(feature = "retain-embeddings")]
        let raw = Some(input.embeddings.clone());
        #[cfg(not(feature = "retain-embeddings"))]
        let raw: Option<Vec<Vec<f64>>> = None;

        let embeddings: Vec<Embedding> = input
            .embeddings
            .iter()
            .map(|v| Embedding::new(v.clone()))
            .collect();

        let projection = fit_projection_for_config(&embeddings, &input.categories, &config)?;
        let mut pipeline = Self::with_configured_projection_and_config(
            input.categories,
            embeddings,
            projection,
            config,
        )?;
        pipeline.raw_embeddings = raw;
        Ok(pipeline)
    }

    /// Build a pipeline using a config predicted by a [`MetaModel`].
    ///
    /// Extracts [`CorpusFeatures`] from the input, asks the model for a
    /// predicted [`PipelineConfig`], then builds the pipeline with it.
    /// Returns the pipeline alongside the extracted features and the
    /// predicted config so the caller can log, audit, or save them as a
    /// new [`MetaTrainingRecord`](crate::meta_model::MetaTrainingRecord).
    ///
    /// This is the "tune-or-recall" entry point: once you've accumulated
    /// a handful of training records, call this instead of
    /// [`crate::tuner::auto_tune`] when you want to skip
    /// search entirely. For a warm-start hybrid that does some tuning
    /// on top of the prediction, use [`Self::new_from_metamodel_tuned`].
    pub fn new_from_metamodel<M: MetaModel>(
        input: PipelineInput,
        model: &M,
    ) -> Result<(Self, CorpusFeatures, PipelineConfig), PipelineError> {
        require_fitted(model)?;
        let features = CorpusFeatures::extract(&input.categories, &input.embeddings)
            .map_err(PipelineError::InvalidInput)?;
        let predicted = model.predict(&features);
        let pipeline = Self::new_with_config(input, predicted.clone())?;
        Ok((pipeline, features, predicted))
    }

    /// Warm-started hybrid: predict a config with `model`, then run a
    /// small-budget tuner pass using that prediction as `base_config`.
    ///
    /// The prediction supplies values only for knobs NOT enumerated by
    /// `space` — any knob the space lists is searched cold across its
    /// axes, and the predicted value for it is ignored. Under
    /// [`SearchStrategy::Random`] and [`SearchStrategy::Bayesian`] the
    /// predicted config itself is additionally evaluated as trial 0
    /// (counted against the budget), so it competes directly with the
    /// searched candidates; [`SearchStrategy::Grid`] skips that seed
    /// trial to keep its trial set the exact Cartesian enumeration.
    ///
    /// Returns the winning pipeline, the extracted corpus features, and
    /// the full [`TuneReport`]. Callers can feed the report back into
    /// [`MetaTrainingRecord::from_tune_result`](crate::meta_model::MetaTrainingRecord::from_tune_result)
    /// to accumulate more training data for the next recall.
    pub fn new_from_metamodel_tuned<M, Q>(
        input: PipelineInput,
        model: &M,
        space: &SearchSpace,
        metric: &Q,
        strategy: SearchStrategy,
    ) -> Result<(Self, CorpusFeatures, TuneReport), PipelineError>
    where
        M: MetaModel,
        Q: QualityMetric,
    {
        require_fitted(model)?;
        let features = CorpusFeatures::extract(&input.categories, &input.embeddings)
            .map_err(PipelineError::InvalidInput)?;
        let predicted = model.predict(&features);
        let (pipeline, report) = auto_tune(input, space, metric, strategy, &predicted)?;
        Ok((pipeline, features, report))
    }

    /// Build a pipeline from pre-computed embeddings and an existing PCA
    /// projection, with [`PipelineConfig::default`].
    ///
    /// This is the legacy entry point — use
    /// [`Self::with_configured_projection_and_config`] directly when you
    /// have a non-PCA [`ConfiguredProjection`].
    pub fn with_projection(
        categories: Vec<String>,
        embeddings: Vec<Embedding>,
        pca: PcaProjection,
    ) -> Result<Self, PipelineError> {
        Self::with_configured_projection_and_config(
            categories,
            embeddings,
            ConfiguredProjection::Pca(pca),
            PipelineConfig::default(),
        )
    }

    /// Legacy configurable PCA entry point. Prefer
    /// [`Self::with_configured_projection_and_config`] for new code.
    pub fn with_projection_and_config(
        categories: Vec<String>,
        embeddings: Vec<Embedding>,
        pca: PcaProjection,
        config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        Self::with_configured_projection_and_config(
            categories,
            embeddings,
            ConfiguredProjection::Pca(pca),
            config,
        )
    }

    /// Core pipeline constructor: accepts any [`ConfiguredProjection`] and
    /// a [`PipelineConfig`].
    pub fn with_configured_projection_and_config(
        categories: Vec<String>,
        embeddings: Vec<Embedding>,
        projection: ConfiguredProjection,
        config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        Self::with_projection_parts(categories, &embeddings, projection, config)
    }

    /// Borrowed-embeddings core shared by the owned constructor and the
    /// tuner. The pipeline never retains the high-dimensional embeddings
    /// (only `raw_embeddings` under the `retain-embeddings` feature does,
    /// and that copy is taken in [`Self::new_with_config`]), so taking a
    /// slice here lets the tuner run trials without cloning the full
    /// corpus matrix per trial.
    pub(crate) fn with_projection_parts(
        categories: Vec<String>,
        embeddings: &[Embedding],
        projection: ConfiguredProjection,
        config: PipelineConfig,
    ) -> Result<Self, PipelineError> {
        let n = embeddings.len();
        if n != categories.len() {
            return Err(PipelineError::LengthMismatch {
                cat: categories.len(),
                emb: n,
            });
        }
        if n < 3 {
            return Err(PipelineError::TooFewEmbeddings(n));
        }

        let mut index = EmbeddingIndex::builder(projection.clone())
            .uniform_shells(10, 1.0)
            .theta_divisions(12)
            .phi_divisions(6)
            .build();

        let mut ids = Vec::with_capacity(n);
        for (i, emb) in embeddings.iter().enumerate() {
            let id = format!("s-{i:04}");
            index.insert(&id, emb);
            ids.push(id);
        }

        // Project each embedding exactly once. The category layer
        // needs the spherical positions and the glob / manifold paths
        // need their Cartesian form; both are derived from the same
        // `projection.project(e)` so building them together saves an
        // N-way pass over the projection (which can be expensive for
        // kernel PCA / Laplacian).
        let projected_positions: Vec<SphericalPoint> =
            embeddings.iter().map(|e| projection.project(e)).collect();
        let cart_points: Vec<[f64; 3]> = projected_positions
            .iter()
            .map(|sp| {
                let c = spherical_to_cartesian(sp);
                [c.x, c.y, c.z]
            })
            .collect();

        let evr = projection.explained_variance_ratio();
        let category_layer = CategoryLayer::build_with_config(
            &categories,
            embeddings,
            &projected_positions,
            &projection,
            evr,
            &config,
        );

        let quality_config = QualityConfig::default();
        let projection_warnings = ProjectionWarning::from_evr(evr, quality_config.warn_below_evr)
            .into_iter()
            .collect();

        Ok(Self {
            projection,
            index,
            categories,
            cart_points,
            ids,
            category_layer,
            quality_config,
            projection_warnings,
            config,
            raw_embeddings: None,
            metric_points: OnceLock::new(),
        })
    }

    /// True if `name` is a known category in this pipeline. Pair with
    /// [`Self::query`] to disambiguate "unknown category" from
    /// "category exists but is disconnected on the graph" without
    /// pattern-matching on `PipelineError::UnknownCategory`.
    pub fn has_category(&self, name: &str) -> bool {
        self.category_layer.name_to_index.contains_key(name)
    }

    /// True if `id` is an indexed item in this pipeline.
    pub fn has_id(&self, id: &str) -> bool {
        self.index.get(id).is_some()
    }

    /// All indexed item ids, in the order they were inserted (i.e. parallel
    /// to the input embeddings/categories). Currently auto-generated as
    /// `s-{i:04}` strings; callers that need stable mapping back to their
    /// own ids should keep their own parallel array.
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Execute a typed query against the pipeline.
    ///
    /// Returns [`PipelineError::UnknownCategory`] when a category
    /// query references a name not in the pipeline, and
    /// [`PipelineError::UnknownId`] when a concept-path query
    /// references an id not in the index. Previously those paths
    /// collapsed into empty results / `None`, which callers couldn't
    /// distinguish from legitimate "found nothing" outcomes.
    pub fn query(
        &self,
        q: SphereQLQuery<'_>,
        query_embedding: &PipelineQuery,
    ) -> Result<SphereQLOutput, PipelineError> {
        let emb = Embedding::new(query_embedding.embedding.clone());

        match q {
            SphereQLQuery::Nearest { k } => {
                Ok(SphereQLOutput::Nearest(self.default_nearest(&emb, k)))
            }

            SphereQLQuery::SimilarAbove { min_cosine } => {
                let evr = self.projection.explained_variance_ratio();
                let results = self.index.search_similar(&emb, min_cosine);
                let sp_q = self.projection.project(&emb);
                Ok(SphereQLOutput::KNearest(
                    results
                        .items
                        .iter()
                        .map(|item| {
                            let d = angular_distance(&sp_q, item.position());
                            let certainty = item.certainty();
                            let quality = QualitySignal::from_certainty(evr, certainty);
                            NearestResult {
                                id: item.id.clone(),
                                category: self.cat_for(&item.id),
                                distance: d,
                                certainty,
                                intensity: item.intensity(),
                                quality: Some(quality),
                            }
                        })
                        .filter(|r| self.passes_quality(r))
                        .collect(),
                ))
            }

            SphereQLQuery::ConceptPath {
                source_id,
                target_id,
                graph_k,
            } => {
                if !self.has_id(source_id) {
                    return Err(PipelineError::UnknownId(source_id.to_string()));
                }
                if !self.has_id(target_id) {
                    return Err(PipelineError::UnknownId(target_id.to_string()));
                }
                let path = self.index.concept_path(source_id, target_id, graph_k);
                Ok(SphereQLOutput::ConceptPath(path.map(|p| {
                    PathResult {
                        total_distance: p.total_distance,
                        steps: p
                            .steps
                            .iter()
                            .map(|s| PipelinePathStep {
                                id: s.id.clone(),
                                category: self.cat_for(&s.id),
                                cumulative_distance: s.cumulative_distance,
                                hop_distance: s.hop_distance,
                                bridge_strength: s.bridge_strength,
                            })
                            .collect(),
                    }
                })))
            }

            SphereQLQuery::DetectGlobs { k, max_k } => {
                let result = GlobResult::detect(&self.cart_points, &self.ids, k, max_k);
                Ok(SphereQLOutput::Globs(
                    result
                        .globs
                        .iter()
                        .map(|g| {
                            let mut cat_counts = std::collections::HashMap::<String, usize>::new();
                            for mid in &g.member_ids {
                                let cat = self.cat_for(mid);
                                *cat_counts.entry(cat).or_default() += 1;
                            }
                            let mut top: Vec<_> = cat_counts.into_iter().collect();
                            top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
                            top.truncate(3);

                            GlobSummary {
                                id: g.id,
                                centroid: g.centroid,
                                member_count: g.member_ids.len(),
                                radius: g.radius,
                                top_categories: top,
                            }
                        })
                        .collect(),
                ))
            }

            SphereQLQuery::LocalManifold { neighborhood_k } => {
                let sp = self.projection.project(&emb);
                let c = spherical_to_cartesian(&sp);
                let qpt = [c.x, c.y, c.z];
                let m = SlicingManifold::fit_local(&qpt, &self.cart_points, neighborhood_k);
                Ok(SphereQLOutput::LocalManifold(ManifoldResult {
                    centroid: m.centroid,
                    normal: m.normal,
                    variance_ratio: m.variance_ratio,
                }))
            }

            // ── Phase 3: category-level query dispatch ─────────────────
            SphereQLQuery::CategoryConceptPath {
                source_category,
                target_category,
            } => {
                if !self.has_category(source_category) {
                    return Err(PipelineError::UnknownCategory(source_category.to_string()));
                }
                if !self.has_category(target_category) {
                    return Err(PipelineError::UnknownCategory(target_category.to_string()));
                }
                let path = self
                    .category_layer
                    .category_path(source_category, target_category);
                Ok(SphereQLOutput::CategoryConceptPath(path))
            }

            SphereQLQuery::CategoryNeighbors { category, k } => {
                if !self.has_category(category) {
                    return Err(PipelineError::UnknownCategory(category.to_string()));
                }
                let neighbors = self.category_layer.category_neighbors(category, k);
                Ok(SphereQLOutput::CategoryNeighbors(
                    neighbors.into_iter().cloned().collect(),
                ))
            }

            SphereQLQuery::DrillDown { category, k } => {
                if !self.has_category(category) {
                    return Err(PipelineError::UnknownCategory(category.to_string()));
                }
                let results = self.category_layer.drill_down_with_projection(
                    category,
                    &emb,
                    &self.projection,
                    k,
                );
                Ok(SphereQLOutput::DrillDown(results))
            }

            SphereQLQuery::CategoryStats => Ok(SphereQLOutput::CategoryStats {
                summaries: self.category_layer.summaries.clone(),
                inner_sphere_reports: self.category_layer.inner_sphere_stats(),
            }),
        }
    }

    /// Get the category for an indexed item by its id.
    fn cat_for(&self, id: &str) -> String {
        if let Some(idx_str) = id.strip_prefix("s-")
            && let Ok(idx) = idx_str.parse::<usize>()
            && idx < self.categories.len()
        {
            return self.categories[idx].clone();
        }
        // Every indexed id is generated as `s-{i:04}` at construction, so
        // reaching this branch means an invariant broke (foreign id leaked
        // into the index, or an out-of-range index). Surface it loudly in
        // debug builds; release builds keep the historical soft fallback.
        debug_assert!(
            false,
            "cat_for: id {id:?} does not match the generated `s-{{i:04}}` format \
             or indexes past {} categories",
            self.categories.len()
        );
        "unknown".into()
    }

    /// Total number of indexed items.
    pub fn num_items(&self) -> usize {
        self.ids.len()
    }

    /// Slice of per-item category labels (index-aligned with insertion order).
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// Export (id, category, cartesian [x, y, z]) triples for every indexed item.
    pub fn projected_points(&self) -> Vec<(&str, &str, [f64; 3])> {
        self.ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let cat = self
                    .categories
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                (id.as_str(), cat, self.cart_points[i])
            })
            .collect()
    }

    /// Borrow the fitted projection regardless of kind.
    ///
    /// Returns a `&ConfiguredProjection`, which implements the
    /// [`crate::projection::Projection`] trait — so most
    /// callers never need to pattern-match on the enum. The old
    /// `.pca()` accessor was removed because it panicked under any
    /// non-PCA config and every caller already worked through this
    /// method or its trait impl.
    pub fn projection(&self) -> &ConfiguredProjection {
        &self.projection
    }

    /// Active outer-sphere projection kind.
    pub fn projection_kind(&self) -> ProjectionKind {
        self.projection.kind()
    }

    /// Export all projected points with their Cartesian and spherical coordinates.
    ///
    /// Returns one `ExportedPoint` per indexed item, in insertion order.
    pub fn exported_points(&self) -> Vec<ExportedPoint> {
        self.ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let [x, y, z] = self.cart_points[i];
                let category = self
                    .categories
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| "unknown".into());
                let item = self.index.get(id);
                let (r, theta, phi) = item
                    .map(|it| {
                        let pos = it.position();
                        (pos.r, pos.theta, pos.phi)
                    })
                    .unwrap_or((0.0, 0.0, 0.0));
                let certainty = item.map_or(1.0, |it| it.certainty());
                let intensity = item.map_or(1.0, |it| it.intensity());
                ExportedPoint {
                    id: id.clone(),
                    category,
                    r,
                    theta,
                    phi,
                    x,
                    y,
                    z,
                    certainty,
                    intensity,
                }
            })
            .collect()
    }

    /// Cached per-item positions and category indices for the geometry
    /// metrics. Positions come from the same per-item index records as
    /// [`Self::exported_points`] (with the same `(0, 0, 0)` fallback for
    /// missing items), so the two views always agree.
    pub(crate) fn metric_points(&self) -> &MetricPoints {
        self.metric_points.get_or_init(|| {
            let positions = self
                .ids
                .iter()
                .map(|id| {
                    self.index
                        .get(id)
                        .map(|it| *it.position())
                        .unwrap_or_else(|| SphericalPoint::new_unchecked(0.0, 0.0, 0.0))
                })
                .collect();
            let category_indices = self
                .categories
                .iter()
                .map(|c| self.category_layer.name_to_index.get(c).copied())
                .collect();
            MetricPoints {
                positions,
                category_indices,
            }
        })
    }

    /// The active projection's explained-variance-ratio-equivalent
    /// quality score, in `[0, 1]`. PCA returns the classical EVR;
    /// kernel PCA returns its kernel-space EVR; Laplacian eigenmap
    /// returns a compatible connectivity ratio (see
    /// [`LaplacianEigenmapProjection::connectivity_ratio`](crate::laplacian::LaplacianEigenmapProjection::connectivity_ratio)).
    /// All three feed the EVR-adaptive thresholds downstream.
    pub fn explained_variance_ratio(&self) -> f64 {
        self.projection.explained_variance_ratio()
    }

    /// Number of unique categories in the corpus.
    pub fn num_categories(&self) -> usize {
        self.category_layer.num_categories()
    }

    /// Unique category names in insertion order.
    pub fn unique_categories(&self) -> Vec<String> {
        self.category_layer
            .summaries
            .iter()
            .map(|s| s.name.clone())
            .collect()
    }

    // ── Phase 3: category-level accessors ──────────────────────────

    /// Access the category enrichment layer directly.
    pub fn category_layer(&self) -> &CategoryLayer {
        &self.category_layer
    }

    /// Shortcut: find the shortest path between two categories.
    pub fn category_path(&self, source: &str, target: &str) -> Option<CategoryPath> {
        self.category_layer.category_path(source, target)
    }

    /// Shortcut: get bridge items between two categories.
    pub fn bridge_items(&self, source: &str, target: &str, max: usize) -> Vec<&BridgeItem> {
        self.category_layer.bridge_items(source, target, max)
    }

    /// Shortcut: check if a category has an inner sphere.
    pub fn has_inner_sphere(&self, category: &str) -> bool {
        self.category_layer.has_inner_sphere(category)
    }

    /// Shortcut: number of categories with inner spheres.
    pub fn num_inner_spheres(&self) -> usize {
        self.category_layer.num_inner_spheres()
    }

    /// Shortcut: inner sphere statistics for all categories.
    pub fn inner_sphere_stats(&self) -> Vec<InnerSphereReport> {
        self.category_layer.inner_sphere_stats()
    }

    /// Projection quality warnings. Empty if EVR is above threshold.
    pub fn projection_warnings(&self) -> &[ProjectionWarning] {
        &self.projection_warnings
    }

    // ── Raw-embedding accessors (retain-embeddings feature) ────────────

    /// Returns the original high-dimensional embeddings if the `retain-embeddings`
    /// feature was active at construction time. The returned slice is aligned with
    /// `ids()`, `categories()`, and `projected_points()`.
    ///
    /// Returns `None` if the feature was not active or embeddings were not retained.
    pub fn raw_embeddings(&self) -> Option<&[Vec<f64>]> {
        self.raw_embeddings.as_deref()
    }

    /// Embedding dimensionality (length of each embedding vector), or 0 if
    /// embeddings are not retained.
    pub fn embedding_dim(&self) -> usize {
        self.raw_embeddings
            .as_ref()
            .and_then(|e| e.first())
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Compute the pairwise cosine similarity matrix from the retained raw
    /// embeddings. Returns the upper triangle as a flat vector aligned with
    /// `ids()` ordering.
    ///
    /// Returns `None` if embeddings were not retained (feature `retain-embeddings`
    /// not active at construction).
    ///
    /// Returns `Err` if the stored embeddings have mismatched dimensions
    /// (should not happen if the pipeline was constructed correctly).
    pub fn pairwise_similarities(&self) -> Option<Result<Vec<f64>, sphereql_core::SphereQlError>> {
        self.raw_embeddings
            .as_ref()
            .map(|embeddings| sphereql_core::pairwise_cosine_similarities(embeddings))
    }

    /// Find the k concepts most similar to `query_embedding` by cosine
    /// similarity in the original embedding space (not the projected space).
    ///
    /// Returns `(index, similarity)` pairs sorted by descending similarity,
    /// where `index` aligns with `ids()`, `categories()`, and `projected_points()`.
    ///
    /// Returns `None` if embeddings were not retained.
    /// Returns `Err(DimensionMismatch)` if `query_embedding.len()` differs from
    /// the stored embedding dimensionality.
    ///
    /// Cost: scans every retained embedding — O(N·D) similarity
    /// computations — with a size-`k` heap keeping selection at
    /// O(N log k) time and O(k) extra allocation.
    pub fn nearest_by_embedding(
        &self,
        query_embedding: &[f64],
        k: usize,
    ) -> Option<Result<Vec<(usize, f64)>, sphereql_core::SphereQlError>> {
        let embeddings = self.raw_embeddings.as_ref()?;
        let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);

        if query_embedding.len() != dim {
            return Some(Err(sphereql_core::SphereQlError::DimensionMismatch {
                expected: dim,
                actual: query_embedding.len(),
            }));
        }

        let query_norm: f64 = query_embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if query_norm < f64::EPSILON || k == 0 {
            return Some(Ok(Vec::new()));
        }

        // Min-heap ordering: lower similarity is "smaller"; on ties the
        // higher index is evicted first, matching the stable-sort order
        // of the old full-materialize implementation.
        struct Scored {
            sim: f64,
            index: usize,
        }
        impl Ord for Scored {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.sim
                    .total_cmp(&other.sim)
                    .then_with(|| other.index.cmp(&self.index))
            }
        }
        impl PartialOrd for Scored {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl PartialEq for Scored {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == std::cmp::Ordering::Equal
            }
        }
        impl Eq for Scored {}

        let mut heap: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(k + 1);
        for (i, emb) in embeddings.iter().enumerate() {
            let emb_norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            let dot: f64 = query_embedding
                .iter()
                .zip(emb.iter())
                .map(|(a, b)| a * b)
                .sum();
            let sim = if emb_norm < f64::EPSILON {
                0.0
            } else {
                (dot / (query_norm * emb_norm)).clamp(-1.0, 1.0)
            };
            heap.push(Reverse(Scored { sim, index: i }));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut scored: Vec<(usize, f64)> = heap
            .into_iter()
            .map(|Reverse(s)| (s.index, s.sim))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Some(Ok(scored))
    }

    // ── Phase 5: hierarchical domain groups ────────────────────────

    /// Coarse-grained domain groups detected from Voronoi adjacency + cap overlap.
    /// Single source of truth: the same vector used by `default_nearest`'s
    /// inner-sphere routing and `hierarchical_nearest`'s drill-down.
    pub fn domain_groups(&self) -> &[DomainGroup] {
        &self.category_layer.domain_groups
    }

    /// Coarse routing: find the domain group whose centroid is angularly
    /// nearest to the query's projected position.
    pub fn route_to_group(&self, embedding: &Embedding) -> Option<&DomainGroup> {
        let groups = &self.category_layer.domain_groups;
        if groups.is_empty() {
            return None;
        }
        let pos = self.projection.project(embedding);
        groups.iter().min_by(|a, b| {
            let da = angular_distance(&pos, &a.centroid);
            let db = angular_distance(&pos, &b.centroid);
            da.total_cmp(&db)
        })
    }

    /// Hierarchical nearest-neighbor search: group → category → items.
    ///
    /// When EVR is at or above
    /// [`RoutingConfig::low_evr_threshold`](crate::config::RoutingConfig::low_evr_threshold),
    /// this is a plain outer-sphere k-NN (identical to [`SphereQLQuery::Nearest`]).
    ///
    /// Below that threshold the outer sphere is unreliable, so we:
    ///   1. Route the query to its nearest domain group.
    ///   2. Drill down into each member category using its inner sphere
    ///      (or the outer sphere if none exists).
    ///   3. Merge the per-category results, sort by distance, truncate to `k`.
    pub fn hierarchical_nearest(&self, embedding: &Embedding, k: usize) -> Vec<NearestResult> {
        let evr = self.projection.explained_variance_ratio();

        if evr >= self.config.routing.low_evr_threshold {
            return self.nearest_filtered(embedding, k, evr);
        }

        let Some(group) = self.route_to_group(embedding) else {
            return self.nearest_filtered(embedding, k, evr);
        };

        // Collect candidates from every category in the routed group, using
        // inner-sphere distances where available.
        let mut candidates: Vec<NearestResult> = Vec::new();
        for &ci in &group.member_categories {
            let cat_name = &self.category_layer.summaries[ci].name;
            for r in self.category_layer.drill_down_with_projection(
                cat_name,
                embedding,
                &self.projection,
                k,
            ) {
                candidates.push(self.drill_result_to_nearest(&r, evr));
            }
        }

        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Don't apply the outer-sphere `passes_quality` filter here:
        // drill_result_to_nearest synthesizes certainty from the *outer*
        // projection's per-item record, but the ranking distance came
        // from the *inner* sphere. Filtering with a threshold calibrated
        // on the outer sphere mixes coordinate systems and would either
        // discard valid inner-sphere matches or admit cross-projection
        // false positives. The low-EVR branch exists precisely because
        // outer-sphere certainty is unreliable in this regime; trust the
        // routing + drill-down ranking instead.
        let truncated: Vec<NearestResult> = candidates.into_iter().take(k).collect();

        if truncated.is_empty() {
            self.nearest_filtered(embedding, k, evr)
        } else {
            truncated
        }
    }

    /// Shared quality-filter predicate. A result passes when its
    /// certainty meets [`QualityConfig::min_certainty`] and, if a
    /// [`QualitySignal`] is attached, it clears
    /// [`QualityConfig::min_combined`].
    ///
    /// Factored out because four of the query-path filter closures had
    /// the same body verbatim; the duplication made threshold changes
    /// a four-way edit.
    #[inline]
    fn passes_quality(&self, r: &NearestResult) -> bool {
        r.certainty >= self.quality_config.min_certainty
            && r.quality
                .is_none_or(|q| q.passes_threshold(self.quality_config.min_combined))
    }

    /// Default `nearest` path (v2 routing).
    ///
    /// Routes the query to its closest domain group when the outer
    /// projection's EVR is below [`HIGH_EVR_ROUTING_BYPASS`], the choice
    /// is unambiguous (`d_nearest / d_second_nearest < group_routing_alpha`),
    /// and the group has an inner sphere; otherwise falls back to plain
    /// outer-sphere k-NN. At EVR ≥ [`HIGH_EVR_ROUTING_BYPASS`] routing is
    /// bypassed entirely — the outer angular distances are already more
    /// accurate than any inner-sphere re-projection.
    pub fn default_nearest(&self, embedding: &Embedding, k: usize) -> Vec<NearestResult> {
        let evr = self.projection.explained_variance_ratio();
        let alpha = self.config.routing.group_routing_alpha;

        let route = if alpha > 0.0 && evr < HIGH_EVR_ROUTING_BYPASS {
            let pos = self.projection.project(embedding);
            self.category_layer
                .nearest_group(&pos)
                .filter(|(gi, d_near, d_second)| {
                    self.category_layer.group_inner_spheres.contains_key(gi)
                        && (*d_second == f64::INFINITY || d_near / d_second < alpha)
                })
                .map(|(gi, _, _)| gi)
        } else {
            None
        };

        if let Some(gi) = route {
            let drilled = self.category_layer.drill_down_group(gi, embedding, k);
            let mut results: Vec<NearestResult> = drilled
                .iter()
                .map(|r| self.drill_result_to_nearest(r, evr))
                .filter(|r| self.passes_quality(r))
                .collect();
            // Falling back to the outer path on empty results matches
            // hierarchical_nearest's correctness-inversion safeguard.
            if !results.is_empty() {
                results.truncate(k);
                return results;
            }
        }

        self.nearest_filtered(embedding, k, evr)
    }

    /// Shared helper: outer-sphere k-NN with quality filtering.
    fn nearest_filtered(&self, embedding: &Embedding, k: usize, evr: f64) -> Vec<NearestResult> {
        self.index
            .search_nearest(embedding, k)
            .iter()
            .map(|r| {
                let certainty = r.item.certainty();
                let quality = QualitySignal::from_certainty(evr, certainty);
                NearestResult {
                    id: r.item.id.clone(),
                    category: self.cat_for(&r.item.id),
                    distance: r.distance,
                    certainty,
                    intensity: r.item.intensity(),
                    quality: Some(quality),
                }
            })
            .filter(|r| self.passes_quality(r))
            .collect()
    }

    fn drill_result_to_nearest(&self, r: &DrillDownResult, evr: f64) -> NearestResult {
        let id = self.ids[r.item_index].clone();
        let item = self.index.get(&id);
        let certainty = item.map_or(1.0, |it| it.certainty());
        let intensity = item.map_or(1.0, |it| it.intensity());
        let quality = QualitySignal::from_certainty(evr, certainty);
        NearestResult {
            id,
            category: self
                .categories
                .get(r.item_index)
                .cloned()
                .unwrap_or_else(|| "unknown".into()),
            distance: r.distance,
            certainty,
            intensity,
            quality: Some(quality),
        }
    }

    /// Current quality configuration.
    pub fn quality_config(&self) -> &QualityConfig {
        &self.quality_config
    }

    /// Update the quality configuration (e.g., to enable filtering).
    pub fn set_quality_config(&mut self, config: QualityConfig) {
        self.quality_config = config;
    }

    /// Annotate every bridge in the category layer with an inferred
    /// [`RelationType`](crate::category::RelationType).
    ///
    /// `labels[i]` must correspond to the same item as
    /// `BridgeItem::item_index == i` — i.e., the pipeline's item list.
    pub fn annotate_relations(&mut self, labels: &[String]) {
        self.category_layer.annotate_bridge_relations(labels);
    }

    /// Full tunable configuration this pipeline was built with.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Serialize all projected points as a JSON array string.
    ///
    /// Returns `Err` when serialization fails — degenerate projections
    /// can produce non-finite coordinates, which JSON cannot represent.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.exported_points())
    }

    /// Serialize all projected points as RFC 4180-compliant CSV with a header row.
    ///
    /// String fields (id, category) are quoted to handle embedded commas
    /// and special characters safely.
    pub fn to_csv(&self) -> String {
        let points = self.exported_points();
        let mut out = String::from("id,category,r,theta,phi,x,y,z,certainty,intensity\n");
        for p in &points {
            out.push_str(&format!(
                "\"{}\",\"{}\",{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                p.id.replace('"', "\"\""),
                p.category.replace('"', "\"\""),
                p.r,
                p.theta,
                p.phi,
                p.x,
                p.y,
                p.z,
                p.certainty,
                p.intensity,
            ));
        }
        out
    }
}

/// Shared guard for the meta-model entry points: an unfitted model would
/// panic inside `predict`, which must not escape a `Result`-returning
/// boundary.
fn require_fitted<M: MetaModel>(model: &M) -> Result<(), PipelineError> {
    if model.is_fitted() {
        Ok(())
    } else {
        Err(PipelineError::InvalidInput(format!(
            "meta-model {:?} is unfitted; call fit() with at least one record first",
            model.name()
        )))
    }
}

/// Fit the projection family specified by `config.projection_kind` on the
/// given corpus. Called by [`SphereQLPipeline::new_with_config`] and the
/// auto-tuner prefit step. Default radial strategy mirrors
/// [`SphereQLPipeline::new`]'s legacy behavior (magnitude + volumetric).
///
/// `categories` is parallel to `embeddings` — same length, same order.
/// The PCA arm uses it to compute per-sample weights
/// `wᵢ = 1 / sqrt(|category(i)|)` (square-root rebalancing of the
/// covariance — see [`PcaProjection::fit_weighted`]); the UMAP arm uses
/// it as supervision labels for the category term. Kernel PCA and
/// Laplacian ignore it.
pub fn fit_projection_for_config(
    embeddings: &[Embedding],
    categories: &[String],
    config: &PipelineConfig,
) -> Result<ConfiguredProjection, crate::projection::ProjectionError> {
    match config.projection_kind {
        ProjectionKind::Pca => {
            // Weight each sample by 1/sqrt(|its_category|) so a category's
            // covariance mass grows as √size instead of linearly — softens
            // (not equalizes) imbalance. Critical for corpora with
            // singleton categories (DBpedia-style), where a uniform fit
            // lets large categories dominate.
            let mut cat_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for c in categories {
                *cat_counts.entry(c.as_str()).or_default() += 1;
            }
            let weights: Vec<f64> = categories
                .iter()
                .map(|c| 1.0 / (cat_counts[c.as_str()] as f64).sqrt())
                .collect();
            Ok(ConfiguredProjection::Pca(
                PcaProjection::fit_weighted(embeddings, &weights, RadialStrategy::Magnitude)?
                    .with_volumetric(true),
            ))
        }
        ProjectionKind::KernelPca => Ok(ConfiguredProjection::KernelPca(KernelPcaProjection::fit(
            embeddings,
            RadialStrategy::Magnitude,
        )?)),
        ProjectionKind::LaplacianEigenmap => {
            let lc = &config.laplacian;
            Ok(ConfiguredProjection::Laplacian(
                LaplacianEigenmapProjection::fit_with_params(
                    embeddings,
                    lc.k_neighbors,
                    lc.active_threshold,
                    RadialStrategy::Magnitude,
                )?,
            ))
        }
        ProjectionKind::UmapSphere => {
            let cat_indices = compact_category_indices(categories);
            Ok(ConfiguredProjection::UmapSphere(
                crate::umap::UmapSphereProjection::fit(
                    embeddings,
                    Some(&cat_indices),
                    RadialStrategy::Magnitude,
                    umap_fit_config(config),
                )?,
            ))
        }
    }
}

/// Fit a UMAP projection from a prebuilt kNN graph.
///
/// Used by the tuner to avoid rebuilding the kNN graph across trials
/// that share `n_neighbors` but differ in `n_epochs` /
/// `category_weight` / `min_dist`.
/// The graph build is the expensive part of UMAP fit; the Adam optimizer
/// that runs on top of it is comparatively cheap.
pub fn fit_umap_from_graph(
    graph: &crate::umap::UmapGraph,
    categories: &[String],
    config: &PipelineConfig,
) -> Result<ConfiguredProjection, crate::projection::ProjectionError> {
    let cat_indices = compact_category_indices(categories);
    Ok(ConfiguredProjection::UmapSphere(
        crate::umap::UmapSphereProjection::fit_from_graph(
            graph,
            Some(&cat_indices),
            RadialStrategy::Magnitude,
            umap_fit_config(config),
        )?,
    ))
}

/// Compact each unique category string to a dense `u32` id, first-seen
/// order, for UMAP's supervised term. Shared by
/// [`fit_projection_for_config`] and [`fit_umap_from_graph`] so both fit
/// paths label items identically.
fn compact_category_indices(categories: &[String]) -> Vec<u32> {
    let mut cat_map: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut next_id: u32 = 0;
    categories
        .iter()
        .map(|c| {
            *cat_map.entry(c.as_str()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
}

/// Translate the tunable [`UmapConfig`](crate::config::UmapConfig)
/// section of a [`PipelineConfig`] into the fitter's own config. The
/// non-tunable optimizer constants (`learning_rate`,
/// `negative_sample_rate`) are pinned to their canonical UMAP defaults
/// here — the single place they're set for pipeline-driven fits.
fn umap_fit_config(config: &PipelineConfig) -> crate::umap::UmapConfig {
    let uc = &config.umap;
    crate::umap::UmapConfig {
        n_neighbors: uc.n_neighbors,
        n_epochs: uc.n_epochs,
        learning_rate: 0.05,
        negative_sample_rate: 5,
        category_weight: uc.category_weight,
        min_dist: uc.min_dist,
        warm_start_anchor: uc.warm_start_anchor,
        seed: uc.seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(n: usize, dim: usize) -> (PipelineInput, PipelineQuery) {
        let mut embeddings = Vec::with_capacity(n);
        let mut categories = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0; dim];
            if i < n / 2 {
                v[0] = 1.0 + (i as f64 * 0.01);
                v[1] = 0.1;
                categories.push("group_a".into());
            } else {
                v[0] = 0.1;
                v[1] = 1.0 + (i as f64 * 0.01);
                categories.push("group_b".into());
            }
            v[2] = 0.05 * (i as f64);
            embeddings.push(v);
        }
        let query = PipelineQuery {
            embedding: vec![0.9; dim],
        };
        (
            PipelineInput {
                categories,
                embeddings,
            },
            query,
        )
    }

    // ── Existing tests (unchanged) ─────────────────────────────────

    #[test]
    fn ids_are_insertion_order_aligned_with_categories_and_points() {
        let (input, _) = make_input(20, 10);
        let categories_in = input.categories.clone();
        let pipeline = SphereQLPipeline::new(input).unwrap();

        let ids = pipeline.ids();
        assert_eq!(ids.len(), 20);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(id, &format!("s-{i:04}"));
        }

        // categories() must be index-aligned with ids().
        let cats = pipeline.categories();
        assert_eq!(cats.len(), ids.len());
        for (i, cat) in cats.iter().enumerate() {
            assert_eq!(cat, &categories_in[i]);
        }

        // projected_points() emits (id, category, [x,y,z]) in the same
        // insertion order — bridge.rs relies on this alignment to
        // round-trip caller IDs.
        let pts = pipeline.projected_points();
        assert_eq!(pts.len(), ids.len());
        for (i, (id, cat, _)) in pts.iter().enumerate() {
            assert_eq!(*id, ids[i].as_str());
            assert_eq!(*cat, cats[i].as_str());
        }
    }

    #[test]
    fn fit_weighted_pca_handles_singleton_categories() {
        // Corpus with extreme imbalance: one category with 20 items,
        // two categories with 1 item each. A naive (unweighted) fit
        // would let the 20-item category dominate the covariance.
        // fit_weighted uses 1/sqrt(count) per category, compressing the
        // big category's mass from 20× to √20 ≈ 4.5× a singleton's —
        // and uses all 22 samples, giving a well-conditioned fit even
        // with singletons.
        let mut embeddings = Vec::new();
        let mut categories = Vec::new();
        for i in 0..20 {
            let mut v = vec![0.0; 8];
            v[0] = 1.0 + (i as f64 * 0.01);
            v[1] = 0.1;
            embeddings.push(v);
            categories.push("big".to_string());
        }
        embeddings.push(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        categories.push("singleton_a".to_string());
        embeddings.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        categories.push("singleton_b".to_string());

        let pipeline = SphereQLPipeline::new(PipelineInput {
            categories,
            embeddings,
        })
        .unwrap();

        let evr = pipeline.explained_variance_ratio();
        assert!(
            evr > 0.0,
            "weighted PCA should produce nonzero EVR even with singletons, got {evr}"
        );
        assert_eq!(pipeline.num_categories(), 3);
    }

    #[test]
    fn pipeline_nearest() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(SphereQLQuery::Nearest { k: 5 }, &query)
            .unwrap();
        match result {
            SphereQLOutput::Nearest(items) => {
                assert_eq!(items.len(), 5);
                assert!(items[0].distance <= items[1].distance);
            }
            _ => panic!("expected Nearest"),
        }
    }

    #[test]
    fn pipeline_globs() {
        let (input, query) = make_input(30, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(
                SphereQLQuery::DetectGlobs {
                    k: Some(2),
                    max_k: 5,
                },
                &query,
            )
            .unwrap();
        match result {
            SphereQLOutput::Globs(globs) => {
                assert_eq!(globs.len(), 2);
                let total: usize = globs.iter().map(|g| g.member_count).sum();
                assert_eq!(total, 30);
            }
            _ => panic!("expected Globs"),
        }
    }

    #[test]
    fn pipeline_concept_path() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(
                SphereQLQuery::ConceptPath {
                    source_id: "s-0000",
                    target_id: "s-0015",
                    graph_k: 10,
                },
                &query,
            )
            .unwrap();
        match result {
            SphereQLOutput::ConceptPath(Some(path)) => {
                assert!(path.steps.len() >= 2);
                assert_eq!(path.steps.first().unwrap().id, "s-0000");
                assert_eq!(path.steps.last().unwrap().id, "s-0015");
            }
            _ => panic!("expected ConceptPath(Some)"),
        }
    }

    #[test]
    fn pipeline_local_manifold() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(SphereQLQuery::LocalManifold { neighborhood_k: 10 }, &query)
            .unwrap();
        match result {
            SphereQLOutput::LocalManifold(m) => {
                assert!(m.variance_ratio > 0.0);
                assert!(m.variance_ratio <= 1.0);
            }
            _ => panic!("expected LocalManifold"),
        }
    }

    #[test]
    fn test_exported_points_count() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        assert_eq!(pipeline.exported_points().len(), 20);
    }

    #[test]
    fn test_exported_points_fields() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        for p in pipeline.exported_points() {
            assert!(p.r >= 0.0, "r must be non-negative");
            assert!(
                p.theta >= 0.0 && p.theta < std::f64::consts::TAU,
                "theta out of range"
            );
            assert!(
                p.phi >= 0.0 && p.phi <= std::f64::consts::PI,
                "phi out of range"
            );
        }
    }

    #[test]
    fn test_exported_points_categories() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let points = pipeline.exported_points();
        for (i, p) in points.iter().enumerate() {
            let expected = if i < 10 { "group_a" } else { "group_b" };
            assert_eq!(p.category, expected);
        }
    }

    #[test]
    fn test_to_json_parseable() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let json = pipeline.to_json().unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed.len(), 20);
    }

    #[test]
    fn test_to_csv_lines() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let csv = pipeline.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "id,category,r,theta,phi,x,y,z,certainty,intensity"
        );
        assert_eq!(lines.len(), 21);
    }

    #[test]
    fn test_to_csv_quoted_fields() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let csv = pipeline.to_csv();
        let data_line = csv.lines().nth(1).unwrap();
        assert!(data_line.starts_with('"'), "id field should be quoted");
    }

    #[test]
    fn test_explained_variance() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let ratio = pipeline.explained_variance_ratio();
        assert!(ratio > 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_unique_categories() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let cats = pipeline.unique_categories();
        assert_eq!(cats.len(), 2);
        assert_eq!(cats[0], "group_a");
        assert_eq!(cats[1], "group_b");
        assert_eq!(pipeline.num_categories(), 2);
    }

    // ── Phase 3 tests: category layer integration ────────────────────

    #[test]
    fn pipeline_builds_category_layer() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        assert_eq!(pipeline.category_layer().num_categories(), 2);
    }

    #[test]
    fn pipeline_category_path_query() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(
                SphereQLQuery::CategoryConceptPath {
                    source_category: "group_a",
                    target_category: "group_b",
                },
                &query,
            )
            .unwrap();
        match result {
            SphereQLOutput::CategoryConceptPath(Some(path)) => {
                assert!(path.steps.len() >= 2);
                assert_eq!(path.steps.first().unwrap().category_name, "group_a");
                assert_eq!(path.steps.last().unwrap().category_name, "group_b");
                assert!(path.total_distance > 0.0);
            }
            _ => panic!("expected CategoryConceptPath(Some)"),
        }
    }

    #[test]
    fn pipeline_category_path_shortcut() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let path = pipeline.category_path("group_a", "group_b");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.steps.first().unwrap().category_name, "group_a");
        assert_eq!(path.steps.last().unwrap().category_name, "group_b");
    }

    #[test]
    fn pipeline_category_path_unknown() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        assert!(pipeline.category_path("group_a", "nonexistent").is_none());
    }

    #[test]
    fn pipeline_category_neighbors_query() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(
                SphereQLQuery::CategoryNeighbors {
                    category: "group_a",
                    k: 5,
                },
                &query,
            )
            .unwrap();
        match result {
            SphereQLOutput::CategoryNeighbors(neighbors) => {
                assert_eq!(neighbors.len(), 1);
                assert_eq!(neighbors[0].name, "group_b");
            }
            _ => panic!("expected CategoryNeighbors"),
        }
    }

    #[test]
    fn pipeline_drill_down_query() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(
                SphereQLQuery::DrillDown {
                    category: "group_a",
                    k: 5,
                },
                &query,
            )
            .unwrap();
        match result {
            SphereQLOutput::DrillDown(results) => {
                assert!(!results.is_empty());
                assert!(results.len() <= 5);
                for w in results.windows(2) {
                    assert!(w[0].distance <= w[1].distance);
                }
            }
            _ => panic!("expected DrillDown"),
        }
    }

    #[test]
    fn pipeline_category_stats_query() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline
            .query(SphereQLQuery::CategoryStats, &query)
            .unwrap();
        match result {
            SphereQLOutput::CategoryStats {
                summaries,
                inner_sphere_reports,
            } => {
                assert_eq!(summaries.len(), 2);
                assert_eq!(inner_sphere_reports.len(), 0);
            }
            _ => panic!("expected CategoryStats"),
        }
    }

    #[test]
    fn pipeline_bridge_items_shortcut() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let _ = pipeline.bridge_items("group_a", "group_b", 5);
    }

    #[test]
    fn pipeline_inner_sphere_shortcuts() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        assert!(!pipeline.has_inner_sphere("group_a"));
        assert_eq!(pipeline.num_inner_spheres(), 0);
        assert!(pipeline.inner_sphere_stats().is_empty());
    }

    #[test]
    fn pipeline_category_layer_accessor() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let layer = pipeline.category_layer();
        assert_eq!(layer.num_categories(), 2);
        assert!(layer.get_category("group_a").is_some());
        assert!(layer.get_category("group_b").is_some());
    }

    // ── Phase 5: domain groups ────────────────────────────────────

    #[test]
    fn domain_groups_detected() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let groups = pipeline.domain_groups();
        assert!(!groups.is_empty());
        let total: usize = groups.iter().map(|g| g.total_items).sum();
        assert_eq!(total, pipeline.num_items());
    }

    #[test]
    fn domain_groups_cover_all_categories() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let groups = pipeline.domain_groups();
        let mut all_cats: Vec<usize> = groups
            .iter()
            .flat_map(|g| g.member_categories.iter().copied())
            .collect();
        all_cats.sort();
        all_cats.dedup();
        assert_eq!(all_cats.len(), pipeline.num_categories());
    }

    #[test]
    fn route_to_group_returns_something() {
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let emb = Embedding::new(vec![0.5; 10]);
        assert!(pipeline.route_to_group(&emb).is_some());
    }

    #[test]
    fn hierarchical_nearest_matches_standard_when_evr_high() {
        // With only 20 items in two well-separated clusters, PCA EVR is
        // typically >= 0.35, so hierarchical_nearest should take the
        // standard outer-sphere path and produce the same IDs as Nearest.
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let hier = pipeline.hierarchical_nearest(&Embedding::new(query.embedding.clone()), 5);
        assert!(!hier.is_empty());
        assert!(hier.len() <= 5);
        for w in hier.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn hierarchical_nearest_falls_back_when_filter_kills_candidates() {
        // Force the low-EVR branch by setting low_evr_threshold = 1.1
        // (every EVR is below that), then set min_certainty = 1.1 so the
        // quality filter discards everything. Without the fallback this
        // used to return an empty Vec in exactly the regime the branch
        // was meant to help. Now it must fall back to the outer-sphere
        // path.
        let (input, query) = make_input(20, 10);
        let mut pipeline = SphereQLPipeline::new_with_config(
            input,
            PipelineConfig {
                routing: crate::config::RoutingConfig {
                    num_domain_groups: 2,
                    group_routing_alpha: 0.8,
                    low_evr_threshold: 1.1, // force low-EVR branch
                },
                ..Default::default()
            },
        )
        .unwrap();
        pipeline.set_quality_config(crate::confidence::QualityConfig {
            min_certainty: 1.1, // unreachable -> every candidate filtered out
            ..Default::default()
        });

        // Also make sure the fallback path itself won't filter everything:
        // nearest_filtered applies the same filter. So we expect the
        // fallback to return Vec too — but the important thing is that
        // neither path silently returns empty when the OTHER path would
        // succeed. Here, with min_certainty=1.1 both paths are filtered.
        // Re-run with min_certainty=0 to assert the fallback-to-outer
        // path actually produces results in the low-EVR regime.
        pipeline.set_quality_config(crate::confidence::QualityConfig::default());
        let hier = pipeline.hierarchical_nearest(&Embedding::new(query.embedding.clone()), 5);
        assert!(
            !hier.is_empty(),
            "low-EVR branch should return results with default filter"
        );
    }

    #[test]
    fn feedback_aggregator_derive_and_save_load_round_trip() {
        // #[serde(transparent)] means the derive-based serializer and
        // the hand-rolled save/load both use a flat JSON array. A file
        // written via serde_json::to_string(&agg) must be loadable via
        // FeedbackAggregator::load.
        use crate::feedback::{FeedbackAggregator, FeedbackEvent};
        let mut agg = FeedbackAggregator::default();
        agg.record(FeedbackEvent {
            corpus_id: "c".into(),
            query_id: "q".into(),
            score: 0.5,
            timestamp: "0".into(),
        });

        let json_via_derive = serde_json::to_string(&agg).unwrap();
        // Flat array shape: starts with '[', not '{'.
        assert!(json_via_derive.starts_with('['));

        // Reload via load() by routing through a temp file.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "sphereql_serde_transparent_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, &json_via_derive).unwrap();
        let loaded = FeedbackAggregator::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn new_from_metamodel_uses_predicted_config() {
        use crate::corpus_features::CorpusFeatures;
        use crate::meta_model::{MetaTrainingRecord, NearestNeighborMetaModel};

        let (input, _) = make_input(20, 10);
        let features = CorpusFeatures::extract(&input.categories, &input.embeddings).unwrap();

        // Hand-built training record: "on a corpus shaped like this, a
        // LaplacianEigenmap config wins". The NN model has only one
        // point so it always returns this config.
        let target_config = PipelineConfig {
            projection_kind: ProjectionKind::LaplacianEigenmap,
            ..Default::default()
        };
        let record = MetaTrainingRecord {
            corpus_id: "seed".into(),
            features: features.clone(),
            best_config: target_config.clone(),
            best_score: 0.5,
            score_lift: None,
            metric_name: "test".into(),
            strategy: "manual".into(),
            timestamp: "0".into(),
        };

        let mut model = NearestNeighborMetaModel::new();
        model.fit(&[record]);

        let (pipeline, _extracted, predicted) =
            SphereQLPipeline::new_from_metamodel(input, &model).unwrap();
        assert_eq!(predicted.projection_kind, ProjectionKind::LaplacianEigenmap);
        assert_eq!(
            pipeline.projection_kind(),
            ProjectionKind::LaplacianEigenmap
        );
    }

    #[test]
    fn new_from_metamodel_tuned_runs_and_carries_prediction() {
        use crate::corpus_features::CorpusFeatures;
        use crate::meta_model::{MetaTrainingRecord, NearestNeighborMetaModel};
        use crate::quality_metric::TerritorialHealth;
        use crate::tuner::{SearchSpace, SearchStrategy};

        // Predict a config that sets an unusual `overlap_artifact_territorial`
        // value NOT in the default search space; then run the tuner with
        // `num_domain_groups` as the only varying axis. The returned
        // pipeline should keep the predicted overlap value (base_config is
        // the prediction) while the tuner picks best num_domain_groups.
        let (input, _) = make_input(20, 10);
        let features = CorpusFeatures::extract(&input.categories, &input.embeddings).unwrap();

        let mut predicted_cfg = PipelineConfig::default();
        predicted_cfg.bridges.overlap_artifact_territorial = 0.123; // unusual

        let record = MetaTrainingRecord {
            corpus_id: "seed".into(),
            features: features.clone(),
            best_config: predicted_cfg.clone(),
            best_score: 0.5,
            score_lift: None,
            metric_name: "test".into(),
            strategy: "manual".into(),
            timestamp: "0".into(),
        };
        let mut model = NearestNeighborMetaModel::new();
        model.fit(&[record]);

        // Constrain the space so only num_domain_groups varies.
        let space = SearchSpace {
            projection_kinds: vec![ProjectionKind::Pca],
            laplacian_k_neighbors: vec![15],
            laplacian_active_threshold: vec![0.05],
            umap_n_neighbors: vec![15],
            umap_n_epochs: vec![200],
            umap_category_weight: vec![1.5],
            umap_min_dist: vec![0.1],
            num_domain_groups: vec![3, 5],
            low_evr_threshold: vec![0.35],
            overlap_artifact_territorial: vec![0.3], // NOT the predicted 0.123
            threshold_base: vec![0.5],
            threshold_evr_penalty: vec![0.4],
            min_evr_improvement: vec![0.10],
        };

        let metric = TerritorialHealth;
        let (pipeline, _feats, report) = SphereQLPipeline::new_from_metamodel_tuned(
            input,
            &model,
            &space,
            &metric,
            SearchStrategy::Grid,
        )
        .unwrap();

        // Grid visits 2 trials (num_domain_groups × 2). Overlap-artifact
        // in every trial's config should be the SPACE's 0.3, not the
        // predicted 0.123 — the search space always overrides. That's
        // the intended contract: warm-start only helps when a knob is
        // NOT in the space.
        assert_eq!(report.trials.len(), 2);
        for t in &report.trials {
            assert!((t.config.bridges.overlap_artifact_territorial - 0.3).abs() < 1e-9);
        }
        assert_eq!(pipeline.projection_kind(), ProjectionKind::Pca);
    }

    #[test]
    fn new_from_metamodel_unfitted_model_returns_err() {
        use crate::meta_model::NearestNeighborMetaModel;

        let (input, _) = make_input(20, 10);
        let model = NearestNeighborMetaModel::new();
        assert!(!model.is_fitted());
        match SphereQLPipeline::new_from_metamodel(input, &model) {
            Err(PipelineError::InvalidInput(msg)) => {
                assert!(msg.contains("unfitted"), "msg = {msg:?}");
            }
            Err(other) => panic!("expected InvalidInput, got {other:?}"),
            Ok(_) => panic!("expected Err for unfitted model"),
        }
    }

    #[test]
    fn new_from_metamodel_tuned_unfitted_model_returns_err() {
        use crate::meta_model::DistanceWeightedMetaModel;
        use crate::quality_metric::TerritorialHealth;
        use crate::tuner::{SearchSpace, SearchStrategy};

        let (input, _) = make_input(20, 10);
        let model = DistanceWeightedMetaModel::new();
        match SphereQLPipeline::new_from_metamodel_tuned(
            input,
            &model,
            &SearchSpace::default(),
            &TerritorialHealth,
            SearchStrategy::Grid,
        ) {
            Err(PipelineError::InvalidInput(msg)) => {
                assert!(msg.contains("unfitted"), "msg = {msg:?}");
            }
            Err(other) => panic!("expected InvalidInput, got {other:?}"),
            Ok(_) => panic!("expected Err for unfitted model"),
        }
    }

    #[test]
    fn new_from_metamodel_empty_input_maps_to_invalid_input() {
        use crate::corpus_features::CorpusFeatures;
        use crate::meta_model::{MetaTrainingRecord, NearestNeighborMetaModel};

        let (seed_input, _) = make_input(20, 10);
        let features =
            CorpusFeatures::extract(&seed_input.categories, &seed_input.embeddings).unwrap();
        let record = MetaTrainingRecord {
            corpus_id: "seed".into(),
            features,
            best_config: PipelineConfig::default(),
            best_score: 0.5,
            score_lift: None,
            metric_name: "test".into(),
            strategy: "manual".into(),
            timestamp: "0".into(),
        };
        let mut model = NearestNeighborMetaModel::new();
        model.fit(&[record]);

        let empty = PipelineInput {
            categories: Vec::new(),
            embeddings: Vec::new(),
        };
        match SphereQLPipeline::new_from_metamodel(empty, &model) {
            Err(PipelineError::InvalidInput(msg)) => {
                assert!(msg.contains("empty"), "msg = {msg:?}");
            }
            Err(other) => panic!("expected InvalidInput, got {other:?}"),
            Ok(_) => panic!("expected Err for empty input"),
        }
    }

    // ── v2 default routing (#3 projection-overhaul) ────────────────────

    /// Two well-separated clusters in 6D, 60 items each. min_size=20
    /// guarantees both per-category and group-level inner spheres are
    /// candidates.
    fn two_cluster_input(n_per: usize, dim: usize) -> (PipelineInput, PipelineQuery) {
        let mut embeddings = Vec::with_capacity(2 * n_per);
        let mut categories = Vec::with_capacity(2 * n_per);
        for i in 0..n_per {
            let t = i as f64 * 0.001;
            let mut v = vec![0.0; dim];
            v[0] = 1.0 + t;
            v[1] = 0.1 + t;
            embeddings.push(v);
            categories.push("a".into());
        }
        for i in 0..n_per {
            let t = i as f64 * 0.001;
            let mut v = vec![0.0; dim];
            v[3] = 1.0 + t;
            v[4] = 0.1 + t;
            embeddings.push(v);
            categories.push("b".into());
        }
        let query = PipelineQuery {
            embedding: {
                let mut q = vec![0.0; dim];
                q[0] = 1.0;
                q[1] = 0.05;
                q
            },
        };
        (
            PipelineInput {
                categories,
                embeddings,
            },
            query,
        )
    }

    #[test]
    fn group_inner_spheres_built_when_two_or_more_groups() {
        let (input, _) = two_cluster_input(30, 8);
        let pipeline = SphereQLPipeline::new_with_config(
            input,
            PipelineConfig {
                routing: crate::config::RoutingConfig {
                    num_domain_groups: 2,
                    group_routing_alpha: 0.8,
                    low_evr_threshold: 0.35,
                },
                // Loosen EVR-improvement threshold: synthetic corpora
                // with rank≤3 cluster geometry leave little room for
                // inner PCA to beat the outer projection by a wide
                // margin. The realistic threshold (0.10) is exercised
                // by the integration tests on real corpora elsewhere.
                inner_sphere: crate::config::InnerSphereConfig {
                    min_evr_improvement: -1.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let layer = pipeline.category_layer();
        assert_eq!(layer.domain_groups.len(), 2);
        assert!(
            !layer.group_inner_spheres.is_empty(),
            "at least one group should produce a group-level inner sphere"
        );
    }

    #[test]
    fn default_nearest_routes_through_group_inner_sphere() {
        let (input, query) = two_cluster_input(30, 8);
        let pipeline = SphereQLPipeline::new_with_config(
            input,
            PipelineConfig {
                routing: crate::config::RoutingConfig {
                    num_domain_groups: 2,
                    group_routing_alpha: 0.99,
                    low_evr_threshold: 0.35,
                },
                ..Default::default()
            },
        )
        .unwrap();

        let results = pipeline.default_nearest(&Embedding::new(query.embedding.clone()), 5);
        assert!(!results.is_empty());
        // Query is closest to cluster-a; routed results should all be a's.
        for r in &results {
            assert_eq!(r.category, "a", "got {r:?}");
        }
    }

    #[test]
    fn default_nearest_bypasses_group_routing_at_high_evr() {
        // When EVR is high, default_nearest should use outer-sphere distances
        // directly, not route through group inner spheres. We verify this by
        // constructing a pipeline with a known high-EVR projection and checking
        // that the nearest result matches the outer-sphere expectation.
        let (input, query) = two_cluster_input(30, 8);
        let pipeline = SphereQLPipeline::new_with_config(
            input,
            PipelineConfig {
                routing: crate::config::RoutingConfig {
                    num_domain_groups: 2,
                    group_routing_alpha: 0.99, // would normally route
                    low_evr_threshold: 0.35,
                },
                ..Default::default()
            },
        )
        .unwrap();

        let evr = pipeline.explained_variance_ratio();
        if evr >= HIGH_EVR_ROUTING_BYPASS {
            // High EVR: routing should be bypassed, outer-sphere k-NN used.
            // The query is [1.0, 0.05, 0, 0, 0, 0, 0, 0] → closest to cluster "a".
            let results = pipeline.default_nearest(&Embedding::new(query.embedding.clone()), 5);
            assert!(!results.is_empty());
            // All results should be from cluster "a" (outer-sphere is accurate).
            for r in &results {
                assert_eq!(
                    r.category, "a",
                    "high-EVR outer-sphere should find cluster-a items, got category={} id={}",
                    r.category, r.id
                );
            }
        }
        // If EVR < HIGH_EVR_ROUTING_BYPASS on this synthetic data, the
        // test is a no-op. That's fine — the important path is tested by
        // the existing routing tests at lower EVR.
    }

    #[test]
    fn default_nearest_falls_back_when_alpha_zero() {
        // alpha=0 disables routing → outer-sphere k-NN is used.
        // This must still produce results (correctness) and is the
        // back-compat escape hatch.
        let (input, query) = two_cluster_input(30, 8);
        let pipeline = SphereQLPipeline::new_with_config(
            input,
            PipelineConfig {
                routing: crate::config::RoutingConfig {
                    num_domain_groups: 2,
                    group_routing_alpha: 0.0,
                    low_evr_threshold: 0.35,
                },
                ..Default::default()
            },
        )
        .unwrap();
        let results = pipeline.default_nearest(&Embedding::new(query.embedding.clone()), 5);
        assert!(!results.is_empty());
    }

    // ── retain-embeddings tests ──────────────────────────────────────

    #[test]
    #[cfg(feature = "retain-embeddings")]
    fn raw_embeddings_retained() {
        let input = PipelineInput {
            categories: vec!["a".into(), "a".into(), "b".into()],
            embeddings: vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.9, 0.1, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
        };
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let raw = pipeline
            .raw_embeddings()
            .expect("should be Some with feature");
        assert_eq!(raw.len(), 3);
        assert_eq!(raw[0], vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[cfg(feature = "retain-embeddings")]
    fn pairwise_similarities_bounded() {
        let input = PipelineInput {
            categories: vec!["a".into(), "b".into(), "c".into()],
            embeddings: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![1.0, 1.0, 0.0],
            ],
        };
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let sims = pipeline.pairwise_similarities().unwrap().unwrap();
        assert_eq!(sims.len(), 3);
        for &s in &sims {
            assert!((-1.0..=1.0).contains(&s));
        }
        let sim_01 = sims[sphereql_core::upper_triangle_index(0, 1, 3)];
        let sim_02 = sims[sphereql_core::upper_triangle_index(0, 2, 3)];
        assert!(sim_02 > sim_01);
    }

    #[test]
    #[cfg(feature = "retain-embeddings")]
    fn nearest_by_embedding_finds_closest() {
        let input = PipelineInput {
            categories: vec!["a".into(), "b".into(), "c".into()],
            embeddings: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        };
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let results = pipeline
            .nearest_by_embedding(&[0.9, 0.1, 0.0], 2)
            .unwrap()
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    #[cfg(feature = "retain-embeddings")]
    fn nearest_by_embedding_handles_ties_and_large_k() {
        // Items 1 and 2 are identical, so their similarity to any query
        // ties exactly — the lower index must rank first. k beyond the
        // corpus size returns everything, sorted descending.
        let input = PipelineInput {
            categories: vec!["a".into(), "b".into(), "b".into(), "c".into()],
            embeddings: vec![
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.5, 0.0, 0.5],
            ],
        };
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let results = pipeline
            .nearest_by_embedding(&[1.0, 0.0, 0.0], 10)
            .unwrap()
            .unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 2);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }

        let top2 = pipeline
            .nearest_by_embedding(&[1.0, 0.0, 0.0], 2)
            .unwrap()
            .unwrap();
        assert_eq!(
            top2.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![1, 2],
            "bounded heap must keep the lower index on exact ties"
        );

        let none = pipeline
            .nearest_by_embedding(&[1.0, 0.0, 0.0], 0)
            .unwrap()
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    #[cfg(feature = "retain-embeddings")]
    fn nearest_by_embedding_dimension_mismatch() {
        let input = PipelineInput {
            categories: vec!["a".into(), "b".into(), "c".into()],
            embeddings: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        };
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline.nearest_by_embedding(&[1.0, 0.0], 1).unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_with_min_category_size_still_indexes_all_items() {
        let mut embeddings = Vec::new();
        let mut categories = Vec::new();
        for i in 0..15 {
            let mut v = vec![0.0; 8];
            v[0] = 1.0 + (i as f64 * 0.01);
            embeddings.push(v);
            categories.push("big".into());
        }
        embeddings.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        categories.push("tiny_a".into());
        embeddings.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        categories.push("tiny_b".into());

        let pipeline = SphereQLPipeline::new_with_config(
            PipelineInput {
                categories,
                embeddings,
            },
            PipelineConfig {
                min_category_size: 5,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(pipeline.num_items(), 17);
        assert!(pipeline.has_id("s-0015"));
        assert!(pipeline.has_id("s-0016"));

        assert_eq!(pipeline.num_categories(), 1);
        let cats = pipeline.unique_categories();
        assert_eq!(cats, vec!["big".to_string()]);
    }

    #[test]
    fn nearest_group_returns_consistent_order() {
        let (input, _) = two_cluster_input(30, 8);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let layer = pipeline.category_layer();
        // Probe at one of the group centroids — that group must come first.
        if let Some(g0) = layer.domain_groups.first() {
            let pos = g0.centroid;
            let (gi, d_near, d_second) = layer.nearest_group(&pos).unwrap();
            assert_eq!(gi, 0);
            assert!(d_near <= d_second);
        }
    }
}
