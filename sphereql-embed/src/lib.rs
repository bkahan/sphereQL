//! Vector embedding projection engine.
//!
//! Projects high-dimensional embeddings onto S² via one of several
//! projection families (PCA, kernel PCA, Laplacian eigenmap, random)
//! and offers a query pipeline for k-NN search, similarity thresholds,
//! concept paths, glob detection, local manifold fitting, and a
//! category-level enrichment layer (inter-category graph, bridge
//! detection with `Genuine`/`OverlapArtifact`/`Weak` classification,
//! inner spheres, hierarchical concept paths, domain-group routing
//! for low-EVR regimes).
//!
//! On top of that, the crate ships a **metalearning framework**:
//!
//! - [`config`] — a single [`PipelineConfig`] hierarchy for every
//!   tunable constant.
//! - [`quality_metric`] — a [`QualityMetric`] trait plus four concrete
//!   metrics (territorial health, bridge coherence, cluster silhouette,
//!   graph modularity) and composite presets.
//! - [`tuner`] — [`auto_tune`] over a discrete [`SearchSpace`] with
//!   Grid / Random / Bayesian (TPE-lite) strategies. Projection kind
//!   is a first-class tuner axis.
//! - [`corpus_features`] — a 10-feature corpus profile suitable as
//!   input to a meta-model.
//! - [`meta_model`] — [`MetaTrainingRecord`] with an on-disk store
//!   under `~/.sphereql/meta_records.json`, the [`MetaModel`] trait,
//!   and two concrete implementations ([`NearestNeighborMetaModel`],
//!   [`DistanceWeightedMetaModel`]).
//! - [`feedback`] — per-query user-satisfaction primitives
//!   ([`FeedbackEvent`] + [`FeedbackAggregator`]) for L3 online
//!   refinement of stored records.
//!
//! See [`SphereQLPipeline::new_with_config`],
//! [`SphereQLPipeline::new_from_metamodel`], and
//! [`SphereQLPipeline::new_from_metamodel_tuned`] for the
//! tune-or-recall entry points.

pub mod category;
pub mod confidence;
pub mod config;
pub mod configured_projection;
pub mod corpus_features;
pub mod domain_groups;
pub mod feedback;
pub mod kernel_pca;
pub mod laplacian;
pub mod mapper;
pub mod meta_model;
pub mod navigator;
pub mod pipeline;
pub mod projection;
pub mod quality_metric;
pub mod query;
pub mod spatial_quality;
pub mod text_embedder;
pub mod tuner;
pub mod types;
pub mod util;

pub use category::*;
pub use confidence::*;
pub use config::*;
pub use configured_projection::*;
pub use corpus_features::*;
pub use domain_groups::*;
pub use feedback::*;
pub use kernel_pca::*;
pub use laplacian::*;
pub use mapper::*;
pub use meta_model::*;
pub use navigator::*;
pub use pipeline::*;
pub use projection::*;
pub use quality_metric::*;
pub use query::*;
pub use spatial_quality::*;
pub use text_embedder::*;
pub use tuner::*;
pub use types::*;
pub use util::*;
