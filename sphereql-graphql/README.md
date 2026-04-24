# sphereql-graphql

GraphQL integration for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Provides an `async-graphql` schema with two surfaces unified behind a
`MergedQueryRoot`:

- **Spatial** — cone, shell, band, wedge, and region lookups,
  k-nearest-neighbor search, distance calculations, and real-time
  subscriptions via a broadcast event bus.
- **Category enrichment** — seven resolvers covering the full
  `sphereql-embed` category surface: `conceptPath`,
  `categoryConceptPath`, `categoryNeighbors`, `drillDown`,
  `hierarchicalNearest`, `categoryStats`, `domainGroups`.

Resolvers that take a `queryText: String` embed it server-side through
a pluggable `TextEmbedder` trait (from `sphereql-embed`). The default
`NoEmbedder` returns a descriptive error until a real embedder is
wired in, so text-query deployments never silently no-op.

```rust
use std::sync::Arc;
use sphereql_embed::{Embedding, EmbedderError, FnEmbedder};
use sphereql_graphql::{
    build_unified_schema, build_pipeline_handle_from_items,
    create_default_index, SpatialEventBus,
};

let embedder = Arc::new(FnEmbedder::new(|text: &str| {
    Ok::<_, EmbedderError>(Embedding::new(embed(text)))
}));
let pipeline = build_pipeline_handle_from_items(&items)?;
let schema = build_unified_schema(
    create_default_index(),
    SpatialEventBus::new(256),
    pipeline,
    embedder,
);
```

Spatial-only deployments keep using the existing `build_schema(index,
event_bus)` entry point — the pipeline and embedder resources are
additive.

See the [main repository](https://github.com/bkahan/sphereQL) for full
documentation, examples, and architecture overview.
