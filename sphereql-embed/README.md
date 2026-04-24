# sphereql-embed

Vector embedding projection engine for the [sphereQL](https://github.com/bkahan/sphereQL) project.

Projects high-dimensional embeddings onto S² via one of four families — PCA, Kernel PCA (Gaussian/RBF), Laplacian eigenmap (connectivity-preserving), or random projection — unified behind a `ConfiguredProjection` enum so the pipeline can switch families without touching generics.

Provides a query pipeline (`SphereQLPipeline`) with k-NN search, similarity thresholds, concept paths, glob detection, local manifold fitting, and a Category Enrichment Layer: inter-category graph, bridge detection with `Genuine` / `OverlapArtifact` / `Weak` classification, automatic inner spheres, drill-down, and hierarchical domain-group routing for low-EVR regimes (`hierarchical_nearest`).

Ships a metalearning framework on top: a `PipelineConfig` hierarchy for every tunable constant (with `#[serde(default)]` so partial overrides work), a `QualityMetric` trait plus four concrete metrics (territorial health, bridge coherence, cluster silhouette, graph modularity) with composite presets, an `auto_tune` sweep over a discrete `SearchSpace` (Grid / Random / Bayesian TPE-lite), a `MetaModel` layer (`NearestNeighbor`, `DistanceWeighted`) with an on-disk store at `~/.sphereql/meta_records.json`, and `FeedbackEvent` / `FeedbackAggregator` primitives for blending user satisfaction into the training record.

Includes a `TextEmbedder` trait (plus `NoEmbedder` default and `FnEmbedder` closure wrapper) so downstream crates — GraphQL, REPLs, custom harnesses — can accept natural-language queries without `sphereql-embed` depending on any specific embedder backend.

See the [main repository](https://github.com/bkahan/sphereQL) for full documentation, examples (`auto_tune`, `meta_learn`, `meta_warm_start`, `meta_feedback`, `spatial_analysis`, `category_enrichment`), and architecture overview.
