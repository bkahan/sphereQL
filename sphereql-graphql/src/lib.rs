//! GraphQL integration for sphereQL spatial queries.
//!
//! Provides an [`async-graphql`] schema with queries for cone, shell, band,
//! wedge, and region lookups, k-nearest-neighbor search, real-time
//! subscriptions via a broadcast event bus, and the full category
//! enrichment surface (concept paths, drill-down, domain groups, stats).

pub mod category;
pub mod category_types;
pub mod context;
pub mod query;
pub mod subscription;
pub mod types;

pub use category::*;
pub use category_types::*;
pub use context::*;
pub use query::*;
pub use subscription::*;
pub use types::*;

/// Merged GraphQL query root combining the spatial-only resolvers and
/// the category-enrichment resolvers.
#[derive(async_graphql::MergedObject, Default)]
pub struct MergedQueryRoot(SphericalQueryRoot, CategoryQueryRoot);

impl MergedQueryRoot {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for SphericalQueryRoot {
    fn default() -> Self {
        SphericalQueryRoot
    }
}

impl Default for CategoryQueryRoot {
    fn default() -> Self {
        CategoryQueryRoot
    }
}

/// Schema flavor that exposes both spatial and category queries.
pub type UnifiedSchema =
    async_graphql::Schema<MergedQueryRoot, async_graphql::EmptyMutation, SphericalSubscriptionRoot>;

/// Build a [`UnifiedSchema`] from all four context resources: the
/// spatial point index, the spatial event bus, the category-enrichment
/// pipeline, and a [`TextEmbedder`](sphereql_embed::text_embedder::TextEmbedder)
/// for resolvers that take text queries.
///
/// To run a spatial-only deployment, keep using [`build_schema`]; the
/// pipeline + embedder context entries are unused there.
///
/// To run a category-only deployment, pass a no-op
/// [`PointIndex`](crate::query::PointIndex) created via
/// [`create_default_index`] alongside the real pipeline; spatial
/// resolvers will return empty results but won't error.
pub fn build_unified_schema(
    index: PointIndex,
    event_bus: SpatialEventBus,
    pipeline: CategoryPipelineHandle,
    embedder: EmbedderHandle,
) -> UnifiedSchema {
    async_graphql::Schema::build(
        MergedQueryRoot::new(),
        async_graphql::EmptyMutation,
        SphericalSubscriptionRoot,
    )
    .data(index)
    .data(event_bus)
    .data(pipeline)
    .data(embedder)
    .finish()
}

/// Convenience wrapper: build a unified schema from an in-memory list of
/// [`CategorizedItemInput`]s and the default no-op embedder.
///
/// Intended for tests, examples, and quickstarts. Production callers
/// should construct the pipeline themselves (so they control projection
/// kind / config) and supply a real
/// [`TextEmbedder`](sphereql_embed::text_embedder::TextEmbedder) before
/// calling [`build_unified_schema`].
pub fn build_unified_schema_from_items(
    items: &[CategorizedItemInput],
) -> Result<UnifiedSchema, sphereql_embed::pipeline::PipelineError> {
    let pipeline = build_pipeline_handle_from_items(items)?;
    Ok(build_unified_schema(
        create_default_index(),
        SpatialEventBus::new(16),
        pipeline,
        default_no_embedder_handle(),
    ))
}
