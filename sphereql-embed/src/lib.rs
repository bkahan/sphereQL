//! Vector embedding projection engine.
//!
//! Projects high-dimensional embeddings onto S² via PCA, kernel PCA,
//! or random projection, then provides a query pipeline for k-NN search,
//! similarity thresholds, concept paths, glob detection, and local
//! manifold fitting.

pub mod kernel_pca;
pub mod mapper;
pub mod pipeline;
pub mod projection;
pub mod query;
pub mod types;

pub use kernel_pca::*;
pub use mapper::*;
pub use pipeline::*;
pub use projection::*;
pub use query::*;
pub use types::*;
