//! Spatial indexing with shell/sector partitioning.
//!
//! Partitions S² into radial shells and angular sectors for fast spatial
//! queries: cone, cap, shell, band, wedge lookups and k-nearest-neighbor search.

pub mod cache;
pub mod composite;
pub mod item;
pub mod sector;
pub mod shell;

pub use cache::*;
pub use composite::*;
pub use item::*;
pub use sector::*;
pub use shell::*;
