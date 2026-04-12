//! Layout engines for distributing items on S².
//!
//! Strategies: Fibonacci spiral ([`UniformLayout`]), k-means clustering
//! ([`ClusteredLayout`]), force-directed simulation ([`ForceDirectedLayout`]),
//! and incremental managed layouts ([`ManagedLayout`]).

pub mod clustered;
pub mod force;
pub mod managed;
pub mod quality;
pub mod traits;
pub mod types;
pub mod uniform;

pub use clustered::*;
pub use force::*;
pub use managed::*;
pub use quality::*;
pub use traits::*;
pub use types::*;
pub use uniform::*;
