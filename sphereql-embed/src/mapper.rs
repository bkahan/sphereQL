use sphereql_core::SphericalPoint;
use sphereql_layout::DimensionMapper;

use crate::projection::Projection;
use crate::types::Embedding;

/// Adapts any [`Projection`] into a [`DimensionMapper`] for use with
/// sphereql-layout's layout strategies (Uniform, Clustered, ForceDirected).
pub struct EmbeddingMapper<P> {
    projection: P,
}

impl<P> EmbeddingMapper<P> {
    pub fn new(projection: P) -> Self {
        Self { projection }
    }

    pub fn projection(&self) -> &P {
        &self.projection
    }
}

impl<P: Projection> DimensionMapper for EmbeddingMapper<P> {
    type Item = Embedding;

    fn map(&self, item: &Embedding) -> SphericalPoint {
        self.projection.project(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::RandomProjection;
    use crate::types::RadialStrategy;
    use sphereql_layout::{LayoutStrategy, UniformLayout};

    #[test]
    fn mapper_delegates_to_projection() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mapper = EmbeddingMapper::new(rp);

        let e = Embedding::new(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let sp = mapper.map(&e);
        assert!((sp.r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mapper_with_layout_strategy() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mapper = EmbeddingMapper::new(rp);

        let embeddings = vec![
            Embedding::new(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
            Embedding::new(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
            Embedding::new(vec![0.0, 0.0, 1.0, 0.0, 0.0]),
        ];

        let layout = UniformLayout::new();
        let result = layout.layout(&embeddings, &mapper);
        assert_eq!(result.entries.len(), 3);

        for entry in &result.entries {
            assert!((entry.position.r - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mapper_exposes_projection() {
        let rp = RandomProjection::new(8, RadialStrategy::Fixed(1.0), 99);
        let mapper = EmbeddingMapper::new(rp);
        assert_eq!(mapper.projection().dimensionality(), 8);
    }
}
