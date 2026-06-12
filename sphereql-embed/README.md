# sphereql-embed

Vector embedding projection engine for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Projects high-dimensional embeddings onto S² via one of four families — PCA, Kernel PCA (Gaussian/RBF), Laplacian eigenmap (connectivity-preserving), or UMAP-on-sphere (tangent-bundle Adam optimizer, kNN attractive + uniform-negative repulsive, optional category supervision) — unified behind a `ConfiguredProjection` enum so the pipeline can switch families without touching generics.

Provides a query pipeline (`SphereQLPipeline`) with k-NN search, similarity thresholds, concept paths, glob detection, local manifold fitting, and a Category Enrichment Layer: inter-category graph, bridge detection with `Genuine` / `OverlapArtifact` / `Weak` classification, automatic inner spheres, drill-down, and hierarchical domain-group routing for low-EVR regimes (`hierarchical_nearest`).

Ships a metalearning framework on top: a `PipelineConfig` hierarchy for every tunable constant (with `#[serde(default)]` so partial overrides work), a `QualityMetric` trait plus four concrete metrics (territorial health, bridge coherence, cluster silhouette, graph modularity) with composite presets, an `auto_tune` sweep over a discrete `SearchSpace` (Grid / Random / Bayesian TPE-lite), a `MetaModel` layer (`NearestNeighbor`, `DistanceWeighted`) with an on-disk store at `~/.sphereql/meta_records.json`, and `FeedbackEvent` / `FeedbackAggregator` primitives for blending user satisfaction into the training record.

Includes a `TextEmbedder` trait (plus `NoEmbedder` default and `FnEmbedder` closure wrapper) so downstream crates — GraphQL, REPLs, custom harnesses — can accept natural-language queries without `sphereql-embed` depending on any specific embedder backend.

## Example

```rust
use sphereql_embed::{
    PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};

let pipeline = SphereQLPipeline::new(PipelineInput {
    categories: vec![
        "science".into(), "science".into(),
        "cooking".into(), "cooking".into(),
    ],
    embeddings: vec![
        vec![0.1, 0.9, 0.3, 0.0],
        vec![0.2, 0.8, 0.4, 0.1],
        vec![0.9, 0.1, 0.0, 0.5],
        vec![0.8, 0.2, 0.1, 0.4],
    ],
})?;

let out = pipeline.query(
    SphereQLQuery::Nearest { k: 3 },
    &PipelineQuery { embedding: vec![0.15, 0.85, 0.35, 0.05] },
)?;

if let SphereQLOutput::Nearest(results) = out {
    for r in results {
        println!("{} ({}) at {:.3} rad", r.id, r.category, r.distance);
    }
}
```

To pick a non-default projection family, build with `SphereQLPipeline::new_with_config` and set `PipelineConfig::projection_kind` (`Pca`, `KernelPca`, `LaplacianEigenmap`, or `UmapSphere`) — or let `auto_tune` sweep the kind as a tuner axis.

## Versioning

Part of the sphereQL workspace, currently `0.2.0-alpha`; API may change before 1.0. Notable recent changes (see the workspace [CHANGELOG](https://github.com/bkahan/sphereQL/blob/main/CHANGELOG.md)): `run_self_tune` and `SphereQLPipeline::to_json` now return `Result`, `MetaModel` gained `is_fitted`, `auto_tune` warm-starts trial 0 from the meta-model prediction, and for UMAP projections `explained_variance_ratio` now reports kNN-recall trustworthiness rather than the old variance proxy (records stored under the old proxy are not score-comparable).

See the [main repository](https://github.com/bkahan/sphereQL) for full documentation, examples (`auto_tune`, `meta_learn`, `meta_warm_start`, `meta_feedback`, `spatial_analysis`, `category_enrichment`), and architecture overview.
