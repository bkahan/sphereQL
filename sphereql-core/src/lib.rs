//! Spherical coordinate math primitives.
//!
//! Provides point types ([`SphericalPoint`], [`CartesianPoint`], [`GeoPoint`]),
//! coordinate conversions, distance functions, interpolation (slerp), and
//! spatial region primitives (cones, caps, shells, bands, wedges).

pub mod conversions;
pub mod distance;
pub mod error;
pub mod interpolation;
pub mod regions;
pub mod spatial;
pub mod types;

pub use conversions::*;
pub use distance::*;
pub use error::*;
pub use interpolation::*;
pub use regions::*;
pub use spatial::*;
pub use types::*;
