use sphereql_core::SphericalPoint;

use crate::types::LayoutResult;

/// Projects an item from its native representation to a position on S².
pub trait DimensionMapper: Send + Sync {
    type Item;
    /// Maps `item` to a spherical position.
    fn map(&self, item: &Self::Item) -> SphericalPoint;
}

/// A layout algorithm that assigns positions on S² to a set of items.
pub trait LayoutStrategy<T>: Send + Sync {
    /// Computes positions for all `items` using `mapper` to derive their natural positions.
    fn layout(&self, items: &[T], mapper: &dyn DimensionMapper<Item = T>) -> LayoutResult<T>;
}
