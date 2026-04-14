use sphereql_core::*;
use sphereql_index::SpatialItem;

use crate::projection::{PcaProjection, Projection};
use crate::query::{EmbeddingIndex, GlobResult, SlicingManifold};
use crate::types::{Embedding, RadialStrategy};

// ── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, thiserror::Error)]
pub enum PipelineError {
    #[error("categories length ({cat}) must equal embeddings length ({emb})")]
    LengthMismatch { cat: usize, emb: usize },
    #[error("need at least 3 embeddings, got {0}")]
    TooFewEmbeddings(usize),
}

// ── Input contract ──────────────────────────────────────────────────────────

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

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NearestResult {
    pub id: String,
    pub category: String,
    pub distance: f64,
    /// Certainty of this point's projection (0–1). Higher = more faithfully represented.
    pub certainty: f64,
    /// Semantic intensity (pre-normalization magnitude of original embedding).
    pub intensity: f64,
}

#[derive(Debug, Clone)]
pub struct PathResult {
    pub steps: Vec<PipelinePathStep>,
    pub total_distance: f64,
}

#[derive(Debug, Clone)]
pub struct PipelinePathStep {
    pub id: String,
    pub category: String,
    pub cumulative_distance: f64,
}

#[derive(Debug, Clone)]
pub struct GlobSummary {
    pub id: usize,
    pub centroid: [f64; 3],
    pub member_count: usize,
    pub radius: f64,
    pub top_categories: Vec<(String, usize)>,
}

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

// ── Pipeline ──────────────────────────────────────────────────────────────

pub struct SphereQLPipeline {
    pca: PcaProjection,
    index: EmbeddingIndex<PcaProjection>,
    categories: Vec<String>,
    cart_points: Vec<[f64; 3]>,
    ids: Vec<String>,
}

impl SphereQLPipeline {
    /// Build a pipeline from raw inputs, fitting a new PCA internally.
    ///
    /// - `input.categories[i]` is the category for sentence `i`
    /// - `input.embeddings[i]` is the embedding vector for sentence `i`
    /// - All embedding vectors must have the same dimensionality (>= 3).
    pub fn new(input: PipelineInput) -> Result<Self, PipelineError> {
        let embeddings: Vec<Embedding> = input
            .embeddings
            .iter()
            .map(|v| Embedding::new(v.clone()))
            .collect();

        let pca = PcaProjection::fit(&embeddings, RadialStrategy::Magnitude).with_volumetric(true);
        Self::with_projection(input.categories, embeddings, pca)
    }

    /// Build a pipeline from pre-computed embeddings and an existing PCA projection.
    ///
    /// Use this when the projection has already been fitted externally (e.g.,
    /// by `VectorStoreBridge`) to avoid fitting a second PCA on the same data.
    pub fn with_projection(
        categories: Vec<String>,
        embeddings: Vec<Embedding>,
        pca: PcaProjection,
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

        let mut index = EmbeddingIndex::builder(pca.clone())
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

        let cart_points: Vec<[f64; 3]> = embeddings
            .iter()
            .map(|e| {
                let sp = pca.project(e);
                let c = spherical_to_cartesian(&sp);
                [c.x, c.y, c.z]
            })
            .collect();

        Ok(Self {
            pca,
            index,
            categories,
            cart_points,
            ids,
        })
    }

    /// Execute a typed query against the pipeline.
    pub fn query(&self, q: SphereQLQuery<'_>, query_embedding: &PipelineQuery) -> SphereQLOutput {
        let emb = Embedding::new(query_embedding.embedding.clone());

        match q {
            SphereQLQuery::Nearest { k } => {
                let results = self.index.search_nearest(&emb, k);
                SphereQLOutput::Nearest(
                    results
                        .iter()
                        .map(|r| NearestResult {
                            id: r.item.id.clone(),
                            category: self.cat_for(&r.item.id),
                            distance: r.distance,
                            certainty: r.item.certainty(),
                            intensity: r.item.intensity(),
                        })
                        .collect(),
                )
            }

            SphereQLQuery::SimilarAbove { min_cosine } => {
                let results = self.index.search_similar(&emb, min_cosine);
                let sp_q = self.pca.project(&emb);
                SphereQLOutput::KNearest(
                    results
                        .items
                        .iter()
                        .map(|item| {
                            let d = angular_distance(&sp_q, item.position());
                            NearestResult {
                                id: item.id.clone(),
                                category: self.cat_for(&item.id),
                                distance: d,
                                certainty: item.certainty(),
                                intensity: item.intensity(),
                            }
                        })
                        .collect(),
                )
            }

            SphereQLQuery::ConceptPath {
                source_id,
                target_id,
                graph_k,
            } => {
                let path = self.index.concept_path(source_id, target_id, graph_k);
                SphereQLOutput::ConceptPath(path.map(|p| {
                    PathResult {
                        total_distance: p.total_distance,
                        steps: p
                            .steps
                            .iter()
                            .map(|s| PipelinePathStep {
                                id: s.id.clone(),
                                category: self.cat_for(&s.id),
                                cumulative_distance: s.cumulative_distance,
                            })
                            .collect(),
                    }
                }))
            }

            SphereQLQuery::DetectGlobs { k, max_k } => {
                let result = GlobResult::detect(&self.cart_points, &self.ids, k, max_k);
                SphereQLOutput::Globs(
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
                )
            }

            SphereQLQuery::LocalManifold { neighborhood_k } => {
                let sp = self.pca.project(&emb);
                let c = spherical_to_cartesian(&sp);
                let qpt = [c.x, c.y, c.z];
                let m = SlicingManifold::fit_local(&qpt, &self.cart_points, neighborhood_k);
                SphereQLOutput::LocalManifold(ManifoldResult {
                    centroid: m.centroid,
                    normal: m.normal,
                    variance_ratio: m.variance_ratio,
                })
            }
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
        "unknown".into()
    }

    pub fn num_items(&self) -> usize {
        self.ids.len()
    }

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

    /// Access the fitted PCA projection.
    pub fn pca(&self) -> &PcaProjection {
        &self.pca
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

    /// The PCA projection's explained variance ratio (0.0–1.0).
    pub fn explained_variance_ratio(&self) -> f64 {
        self.pca.explained_variance_ratio()
    }

    /// Number of unique categories in the corpus.
    pub fn num_categories(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for cat in &self.categories {
            seen.insert(cat.as_str());
        }
        seen.len()
    }

    /// Unique category names in insertion order.
    pub fn unique_categories(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for cat in &self.categories {
            if seen.insert(cat.as_str()) {
                result.push(cat.clone());
            }
        }
        result
    }

    /// Serialize all projected points as a JSON array string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.exported_points())
            .expect("ExportedPoint is always serializable")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(n: usize, dim: usize) -> (PipelineInput, PipelineQuery) {
        let mut embeddings = Vec::with_capacity(n);
        let mut categories = Vec::with_capacity(n);
        // Create n embeddings in dim dimensions with some structure
        for i in 0..n {
            let mut v = vec![0.0; dim];
            // Put half in "group_a" (strong first component) and half in "group_b" (strong second)
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

    #[test]
    fn pipeline_nearest() {
        let (input, query) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let result = pipeline.query(SphereQLQuery::Nearest { k: 5 }, &query);
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
        let result = pipeline.query(
            SphereQLQuery::DetectGlobs {
                k: Some(2),
                max_k: 5,
            },
            &query,
        );
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
        let result = pipeline.query(
            SphereQLQuery::ConceptPath {
                source_id: "s-0000",
                target_id: "s-0015",
                graph_k: 10,
            },
            &query,
        );
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
        let result = pipeline.query(SphereQLQuery::LocalManifold { neighborhood_k: 10 }, &query);
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
        let json = pipeline.to_json();
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
        assert_eq!(lines.len(), 21); // header + 20 data lines
    }

    #[test]
    fn test_to_csv_quoted_fields() {
        // Verify that the CSV output properly quotes string fields
        let (input, _) = make_input(20, 10);
        let pipeline = SphereQLPipeline::new(input).unwrap();
        let csv = pipeline.to_csv();
        let data_line = csv.lines().nth(1).unwrap();
        // First field (id) should be quoted
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
}
