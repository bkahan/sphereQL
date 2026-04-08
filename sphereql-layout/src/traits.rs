use sphereql_core::SphericalPoint;

use crate::types::LayoutResult;

pub trait DimensionMapper: Send + Sync {
    type Item;
    fn map(&self, item: &Self::Item) -> SphericalPoint;
}

pub trait LayoutStrategy<T>: Send + Sync {
    fn layout(&self, items: &[T], mapper: &dyn DimensionMapper<Item = T>) -> LayoutResult<T>;
}
