//! Language to native SphereQL coordinates.
//!
//! Six-stage pipeline: text -> concepts -> theta -> phi -> r -> relations -> ConceptGraph.
//! Each stage is a trait with a default heuristic implementation. Plug in
//! LLM-backed extractors for production.
//!
//! Reuses [`sphereql_core::SphericalPoint`] as the coordinate type and
//! delegates all spherical math (Vincenty distance, SLERP, conversions)
//! to `sphereql-core`.

pub mod abstraction;
pub mod concept;
pub mod graph;
pub mod pipeline;
pub mod relation;
pub mod salience;
pub mod taxonomy;

pub use abstraction::AbstractionResolver;
pub use concept::{Concept, ConceptExtractor, RegexExtractor};
pub use graph::ConceptGraph;
pub use pipeline::LinguaPipeline;
pub use relation::{Relation, RelationEncoder, RelationType};
pub use salience::SalienceScorer;
pub use taxonomy::{DomainAnchor, DomainTaxonomy};
