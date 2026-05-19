//! Umbrella crate for SphereQL.
//!
//! Re-exports each sub-crate behind a feature gate of the same name.
//! Enable only the features your binary needs to keep compile times short.
//!
//! | Feature | Crate |
//! |---------|-------|
//! | `core` | [`sphereql_core`] — spherical geometry primitives |
//! | `index` | [`sphereql_index`] — spatial index structures |
//! | `layout` | [`sphereql_layout`] — projection & layout utilities |
//! | `embed` | [`sphereql_embed`] — embedding pipeline + tuner |
//! | `graphql` | [`sphereql_graphql`] — GraphQL schema layer |
//! | `vectordb` | [`sphereql_vectordb`] — vector store backend |

#[cfg(feature = "core")]
pub mod core {
    pub use sphereql_core::*;
}

#[cfg(feature = "index")]
pub mod index {
    pub use sphereql_index::*;
}

#[cfg(feature = "layout")]
pub mod layout {
    pub use sphereql_layout::*;
}

#[cfg(feature = "embed")]
pub mod embed {
    pub use sphereql_embed::*;
}

#[cfg(feature = "graphql")]
pub mod graphql {
    pub use sphereql_graphql::*;
}

#[cfg(feature = "vectordb")]
pub mod vectordb {
    pub use sphereql_vectordb::*;
}
