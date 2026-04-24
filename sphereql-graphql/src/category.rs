//! GraphQL root for the category-enrichment surface.
//!
//! Seven resolvers cover the full category API exposed by the Rust
//! `SphereQLPipeline`:
//!
//! - `conceptPath` — item-to-item path through the concept graph
//! - `categoryConceptPath` — category-to-category path through the graph
//! - `categoryNeighbors` — k-NN over categories
//! - `drillDown` — within-category search using inner-sphere projection
//!   when available
//! - `hierarchicalNearest` — domain-group routing fallback for low-EVR
//!   projections
//! - `categoryStats` — summaries + inner-sphere reports
//! - `domainGroups` — coarse category clusters from Voronoi geometry
//!
//! Resolvers that take a natural-language `queryText` embed it through
//! the `EmbedderHandle` stored in context. Schemas that haven't been
//! given a real embedder use [`crate::default_no_embedder_handle`],
//! which returns a descriptive error when `embed()` is called.

use async_graphql::{Context, Object, Result};

use sphereql_embed::pipeline::{SphereQLOutput, SphereQLQuery};
use sphereql_embed::text_embedder::TextEmbedder;
use sphereql_embed::types::Embedding;

use crate::category_types::{
    CategoryNearestResultOutput, CategoryPathOutput, CategoryStatsOutput, CategorySummaryOutput,
    ConceptPathOutput, DomainGroupOutput, DrillDownOutput,
};
use crate::context::{CategoryPipelineHandle, EmbedderHandle};

pub struct CategoryQueryRoot;

#[Object]
impl CategoryQueryRoot {
    /// Shortest path between two indexed items through the concept graph,
    /// anchored at the supplied `queryText` embedding.
    async fn concept_path(
        &self,
        ctx: &Context<'_>,
        source_id: String,
        target_id: String,
        graph_k: i32,
        query_text: String,
    ) -> Result<Option<ConceptPathOutput>> {
        let embedding = embed_query_text(ctx, &query_text)?;
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        let query = sphereql_embed::pipeline::PipelineQuery {
            embedding: embedding.values,
        };
        let out = guard
            .query(
                SphereQLQuery::ConceptPath {
                    source_id: &source_id,
                    target_id: &target_id,
                    graph_k: graph_k.max(0) as usize,
                },
                &query,
            )
            .map_err(gql_err)?;
        match out {
            SphereQLOutput::ConceptPath(path) => Ok(path.as_ref().map(ConceptPathOutput::from)),
            _ => Err(unexpected("ConceptPath")),
        }
    }

    /// Shortest path between two categories through the category-level graph.
    async fn category_concept_path(
        &self,
        ctx: &Context<'_>,
        source_category: String,
        target_category: String,
    ) -> Result<Option<CategoryPathOutput>> {
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        // CategoryConceptPath doesn't consult the query embedding, but the
        // pipeline's uniform `query` entry point requires one. Pass a
        // dimensionality-matched zero vector.
        let query = zero_query(&guard);
        let out = guard
            .query(
                SphereQLQuery::CategoryConceptPath {
                    source_category: &source_category,
                    target_category: &target_category,
                },
                &query,
            )
            .map_err(gql_err)?;
        match out {
            SphereQLOutput::CategoryConceptPath(path) => {
                Ok(path.as_ref().map(CategoryPathOutput::from))
            }
            _ => Err(unexpected("CategoryConceptPath")),
        }
    }

    /// k nearest neighbor categories to the given category.
    async fn category_neighbors(
        &self,
        ctx: &Context<'_>,
        category: String,
        k: i32,
    ) -> Result<Vec<CategorySummaryOutput>> {
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        let query = zero_query(&guard);
        let out = guard
            .query(
                SphereQLQuery::CategoryNeighbors {
                    category: &category,
                    k: k.max(0) as usize,
                },
                &query,
            )
            .map_err(gql_err)?;
        match out {
            SphereQLOutput::CategoryNeighbors(items) => {
                Ok(items.iter().map(CategorySummaryOutput::from).collect())
            }
            _ => Err(unexpected("CategoryNeighbors")),
        }
    }

    /// k-NN within a single category, using the category's inner-sphere
    /// projection when one exists.
    async fn drill_down(
        &self,
        ctx: &Context<'_>,
        category: String,
        query_text: String,
        k: i32,
    ) -> Result<Vec<DrillDownOutput>> {
        let embedding = embed_query_text(ctx, &query_text)?;
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        let query = sphereql_embed::pipeline::PipelineQuery {
            embedding: embedding.values,
        };
        let out = guard
            .query(
                SphereQLQuery::DrillDown {
                    category: &category,
                    k: k.max(0) as usize,
                },
                &query,
            )
            .map_err(gql_err)?;
        match out {
            SphereQLOutput::DrillDown(items) => {
                Ok(items.iter().map(DrillDownOutput::from).collect())
            }
            _ => Err(unexpected("DrillDown")),
        }
    }

    /// Domain-group-routed nearest-neighbor search. Falls back to a plain
    /// outer-sphere k-NN when the projection EVR is above the configured
    /// routing threshold.
    async fn hierarchical_nearest(
        &self,
        ctx: &Context<'_>,
        query_text: String,
        k: i32,
    ) -> Result<Vec<CategoryNearestResultOutput>> {
        let embedding = embed_query_text(ctx, &query_text)?;
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        let items = guard.hierarchical_nearest(&embedding, k.max(0) as usize);
        Ok(items
            .iter()
            .map(CategoryNearestResultOutput::from)
            .collect())
    }

    /// Per-category summaries + inner-sphere reports.
    async fn category_stats(&self, ctx: &Context<'_>) -> Result<CategoryStatsOutput> {
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        let query = zero_query(&guard);
        let out = guard
            .query(SphereQLQuery::CategoryStats, &query)
            .map_err(gql_err)?;
        match out {
            SphereQLOutput::CategoryStats {
                summaries,
                inner_sphere_reports,
            } => Ok(CategoryStatsOutput {
                summaries: summaries.iter().map(CategorySummaryOutput::from).collect(),
                inner_sphere_reports: inner_sphere_reports
                    .iter()
                    .map(crate::category_types::InnerSphereReportOutput::from)
                    .collect(),
            }),
            _ => Err(unexpected("CategoryStats")),
        }
    }

    /// Coarse domain groups detected from category geometry.
    async fn domain_groups(&self, ctx: &Context<'_>) -> Result<Vec<DomainGroupOutput>> {
        let pipeline = pipeline_handle(ctx)?;
        let guard = pipeline.read().await;
        Ok(guard
            .domain_groups()
            .iter()
            .map(DomainGroupOutput::from)
            .collect())
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn pipeline_handle<'c>(ctx: &Context<'c>) -> Result<&'c CategoryPipelineHandle> {
    ctx.data::<CategoryPipelineHandle>()
        .map_err(|_| async_graphql::Error::new("SphereQLPipeline not found in context"))
}

fn embed_query_text(ctx: &Context<'_>, text: &str) -> Result<Embedding> {
    let embedder = ctx
        .data::<EmbedderHandle>()
        .map_err(|_| async_graphql::Error::new("TextEmbedder not found in context"))?;
    embedder
        .embed(text)
        .map_err(|e| async_graphql::Error::new(e.to_string()))
}

/// Build a dimensionality-matched zero query for resolvers that dispatch
/// category-only `SphereQLQuery` variants (those variants never touch the
/// embedding, but the pipeline's uniform entry point still requires one
/// of matching dimension).
fn zero_query(
    pipeline: &sphereql_embed::pipeline::SphereQLPipeline,
) -> sphereql_embed::pipeline::PipelineQuery {
    use sphereql_embed::projection::Projection;
    let dim = pipeline.projection().dimensionality();
    sphereql_embed::pipeline::PipelineQuery {
        embedding: vec![0.0; dim],
    }
}

fn gql_err<E: std::fmt::Display>(e: E) -> async_graphql::Error {
    async_graphql::Error::new(e.to_string())
}

fn unexpected(expected: &str) -> async_graphql::Error {
    async_graphql::Error::new(format!(
        "unexpected SphereQLOutput variant (expected {expected})"
    ))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_graphql::{EmptyMutation, EmptySubscription, Schema};
    use sphereql_embed::text_embedder::{EmbedderError, FnEmbedder};
    use sphereql_embed::types::Embedding;

    use crate::context::{build_pipeline_handle_from_items, default_no_embedder_handle};
    use crate::types::CategorizedItemInput;

    fn items() -> Vec<CategorizedItemInput> {
        // 12 items, 4-dim embeddings, 3 categories with 4 each — enough
        // for category neighbors, stats, and domain groups to have
        // content without triggering inner-sphere thresholds.
        let rows: &[(&str, &str, [f64; 4])] = &[
            ("a1", "alpha", [1.0, 0.1, 0.0, 0.2]),
            ("a2", "alpha", [0.9, 0.2, 0.0, 0.3]),
            ("a3", "alpha", [1.0, 0.15, 0.05, 0.25]),
            ("a4", "alpha", [0.95, 0.1, 0.05, 0.2]),
            ("b1", "beta", [0.1, 1.0, 0.0, 0.2]),
            ("b2", "beta", [0.2, 0.9, 0.1, 0.3]),
            ("b3", "beta", [0.15, 0.95, 0.05, 0.25]),
            ("b4", "beta", [0.05, 0.85, 0.05, 0.2]),
            ("g1", "gamma", [0.3, 0.3, 0.9, 0.1]),
            ("g2", "gamma", [0.25, 0.35, 0.85, 0.15]),
            ("g3", "gamma", [0.35, 0.25, 0.9, 0.1]),
            ("g4", "gamma", [0.3, 0.3, 0.95, 0.05]),
        ];
        rows.iter()
            .map(|(id, cat, emb)| CategorizedItemInput {
                id: (*id).into(),
                category: (*cat).into(),
                embedding: emb.to_vec(),
            })
            .collect()
    }

    fn test_embedder() -> EmbedderHandle {
        // Deterministic closure: pad-or-truncate the text's bytes to a 4-d
        // vector. Enough to exercise the resolver plumbing; the embedding
        // correlates weakly with corpus items so queries produce real
        // results without being hand-tuned.
        Arc::new(FnEmbedder::new(|text: &str| {
            let b = text.as_bytes();
            let mut v = [0.0f64; 4];
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = *b.get(i).unwrap_or(&1) as f64 / 128.0;
            }
            Ok::<_, EmbedderError>(Embedding::new(v.to_vec()))
        }))
    }

    fn build_schema_for_tests(
        embedder: EmbedderHandle,
    ) -> Schema<CategoryQueryRoot, EmptyMutation, EmptySubscription> {
        let pipeline = build_pipeline_handle_from_items(&items()).expect("pipeline build failed");
        Schema::build(CategoryQueryRoot, EmptyMutation, EmptySubscription)
            .data(pipeline)
            .data(embedder)
            .finish()
    }

    #[tokio::test]
    async fn category_neighbors_returns_results() {
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(r#"{ categoryNeighbors(category: "alpha", k: 2) { name memberCount } }"#)
            .await;
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let arr = data["categoryNeighbors"].as_array().unwrap();
        assert!(!arr.is_empty());
    }

    #[tokio::test]
    async fn category_stats_returns_summaries() {
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(r#"{ categoryStats { summaries { name memberCount } innerSphereReports { categoryName } } }"#)
            .await;
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let summaries = data["categoryStats"]["summaries"].as_array().unwrap();
        assert_eq!(summaries.len(), 3);
    }

    #[tokio::test]
    async fn category_concept_path_finds_path() {
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(
                r#"{ categoryConceptPath(sourceCategory: "alpha", targetCategory: "gamma") { totalDistance steps { categoryName } } }"#,
            )
            .await;
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        assert!(data["categoryConceptPath"].is_object());
    }

    #[tokio::test]
    async fn drill_down_uses_embedder() {
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(
                r#"{ drillDown(category: "alpha", queryText: "alpha query", k: 3) { itemIndex distance usedInnerSphere } }"#,
            )
            .await;
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let arr = data["drillDown"].as_array().unwrap();
        assert!(!arr.is_empty());
    }

    #[tokio::test]
    async fn hierarchical_nearest_uses_embedder() {
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(
                r#"{ hierarchicalNearest(queryText: "something", k: 3) { id category distance } }"#,
            )
            .await;
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        let arr = data["hierarchicalNearest"].as_array().unwrap();
        assert!(!arr.is_empty());
    }

    #[tokio::test]
    async fn domain_groups_returns_groups() {
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(r#"{ domainGroups { categoryNames totalItems } }"#)
            .await;
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
        let data = res.data.into_json().unwrap();
        assert!(data["domainGroups"].is_array());
    }

    #[tokio::test]
    async fn concept_path_uses_embedder() {
        // Pipeline assigns its own ids of the form `s-NNNN` (see
        // `items_to_pipeline_input` docs) — use those, not the values
        // passed on `CategorizedItemInput.id`.
        let schema = build_schema_for_tests(test_embedder());
        let res = schema
            .execute(
                r#"{ conceptPath(sourceId: "s-0000", targetId: "s-0008", graphK: 4, queryText: "bridge") { totalDistance steps { id category } } }"#,
            )
            .await;
        // ConceptPath may return null if no path is found — both shapes
        // are schema-valid. Just assert no error.
        assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
    }

    #[tokio::test]
    async fn text_query_without_embedder_errors_descriptively() {
        let schema = build_schema_for_tests(default_no_embedder_handle());
        let res = schema
            .execute(r#"{ hierarchicalNearest(queryText: "whatever", k: 3) { id } }"#)
            .await;
        assert!(!res.errors.is_empty(), "expected an error");
        let msg = res.errors[0].message.clone();
        assert!(msg.contains("no TextEmbedder configured"), "got: {msg}");
    }
}
