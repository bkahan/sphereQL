use sphereql_core::SphericalPoint;
use std::hash::Hash;

pub trait SpatialItem: Clone + Send + Sync {
    type Id: Eq + Hash + Clone + Send + Sync + std::fmt::Debug;
    fn id(&self) -> &Self::Id;
    fn position(&self) -> &SphericalPoint;
}

#[derive(Debug, Clone)]
pub struct NearestResult<T: SpatialItem> {
    pub item: T,
    pub distance: f64,
}

#[derive(Debug, Clone)]
pub struct SpatialQueryResult<T: SpatialItem> {
    pub items: Vec<T>,
    pub total_scanned: usize,
}
