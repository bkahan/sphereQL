//! Language to native SphereQL coordinates.
//!
//! Six-stage pipeline: text -> concepts -> theta -> phi -> r -> relations -> ConceptGraph.
//! Each stage is a trait with a default heuristic implementation. Plug in
//! LLM-backed extractors for production.
//!
//! Reuses [`sphereql_core::SphericalPoint`] as the coordinate type and
//! delegates all spherical math (Vincenty distance, SLERP, conversions)
//! to `sphereql-core`.

pub mod concept;
pub mod taxonomy;
pub mod abstraction;
pub mod salience;
pub mod relation;
pub mod graph;
pub mod pipeline;

pub use concept::{Concept, ConceptExtractor, RegexExtractor};
pub use taxonomy::{DomainAnchor, DomainTaxonomy};
pub use abstraction::AbstractionResolver;
pub use salience::SalienceScorer;
pub use relation::{Relation, RelationType, RelationEncoder};
pub use graph::ConceptGraph;
pub use pipeline::LinguaPipeline;
