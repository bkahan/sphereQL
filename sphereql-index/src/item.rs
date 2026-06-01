use sphereql_core::SphericalPoint;
use std::hash::Hash;

/// Trait for items that can be stored in a [`SpatialIndex`](crate::SpatialIndex).
///
/// Implementors must provide an id and a spherical position.
///
/// ```
/// use sphereql_index::SpatialItem;
/// use sphereql_core::SphericalPoint;
///
/// #[derive(Debug, Clone)]
/// struct MyItem { id: u32, pos: SphericalPoint }
///
/// impl SpatialItem for MyItem {
///     type Id = u32;
///     fn id(&self) -> &u32 { &self.id }
///     fn position(&self) -> &SphericalPoint { &self.pos }
/// }
///
/// let item = MyItem { id: 1, pos: SphericalPoint::new_unchecked(1.0, 0.5, 0.7) };
/// assert_eq!(*item.id(), 1);
/// ```
pub trait SpatialItem: Clone + Send + Sync {
    type Id: Eq + Hash + Clone + Send + Sync + std::fmt::Debug;
    fn id(&self) -> &Self::Id;
    fn position(&self) -> &SphericalPoint;
}

/// One result from a k-nearest-neighbor query.
#[derive(Debug, Clone)]
pub struct NearestResult<T: SpatialItem> {
    pub item: T,
    /// Angular distance in radians from the query point to this item.
    pub distance: f64,
}

/// Result of a spatial region query.
#[derive(Debug, Clone)]
pub struct SpatialQueryResult<T: SpatialItem> {
    pub items: Vec<T>,
    /// Number of items inspected (including those that did not match). Useful for profiling index efficiency.
    pub total_scanned: usize,
}
