//! Vector embedding projection engine.
//!
//! Projects high-dimensional embeddings onto S² via PCA, kernel PCA,
//! or random projection, then provides a query pipeline for k-NN search,
//! similarity thresholds, concept paths, glob detection, local manifold
//! fitting, and category-level enrichment (inter-category graph, bridge
//! detection, hierarchical concept paths).

pub mod category;
pub mod confidence;
pub mod kernel_pca;
pub mod mapper;
pub mod navigator;
pub mod pipeline;
pub mod projection;
pub mod query;
pub mod spatial_quality;
pub mod types;

pub use category::*;
pub use confidence::*;
pub use kernel_pca::*;
pub use mapper::*;
pub use navigator::*;
pub use pipeline::*;
pub use projection::*;
pub use query::*;
pub use spatial_quality::*;
pub use types::*;
