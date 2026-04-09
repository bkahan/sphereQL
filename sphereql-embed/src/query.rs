use sphereql_core::*;
use sphereql_index::*;

use crate::projection::Projection;
use crate::types::Embedding;

#[derive(Debug, Clone)]
pub struct EmbeddingItem {
    pub id: String,
    pub position: SphericalPoint,
    pub original_magnitude: f64,
}

impl SpatialItem for EmbeddingItem {
    type Id = String;
    fn id(&self) -> &String {
        &self.id
    }
    fn position(&self) -> &SphericalPoint {
        &self.position
    }
}

pub struct EmbeddingIndexBuilder<P> {
    projection: P,
    inner: SpatialIndexBuilder,
}

impl<P: Projection> EmbeddingIndexBuilder<P> {
    pub fn new(projection: P) -> Self {
        Self {
            projection,
            inner: SpatialIndexBuilder::new(),
        }
    }

    pub fn shell_boundary(mut self, r: f64) -> Self {
        self.inner = self.inner.shell_boundary(r);
        self
    }

    pub fn uniform_shells(mut self, count: usize, max_r: f64) -> Self {
        self.inner = self.inner.uniform_shells(count, max_r);
        self
    }

    pub fn theta_divisions(mut self, n: usize) -> Self {
        self.inner = self.inner.theta_divisions(n);
        self
    }

    pub fn phi_divisions(mut self, n: usize) -> Self {
        self.inner = self.inner.phi_divisions(n);
        self
    }

    pub fn build(self) -> EmbeddingIndex<P> {
        EmbeddingIndex {
            projection: self.projection,
            index: self.inner.build(),
        }
    }
}

pub struct EmbeddingIndex<P> {
    projection: P,
    index: SpatialIndex<EmbeddingItem>,
}

impl<P: Projection> EmbeddingIndex<P> {
    pub fn builder(projection: P) -> EmbeddingIndexBuilder<P> {
        EmbeddingIndexBuilder::new(projection)
    }

    pub fn insert(&mut self, id: impl Into<String>, embedding: &Embedding) {
        let position = self.projection.project(embedding);
        self.index.insert(EmbeddingItem {
            id: id.into(),
            position,
            original_magnitude: embedding.magnitude(),
        });
    }

    /// Insert with an explicit radial value, overriding the projection's RadialStrategy.
    /// The angular coordinates (theta, phi) are still determined by the projection.
    /// Use this for metadata-driven radius: recency scores, importance weights, etc.
    pub fn insert_with_radius(&mut self, id: impl Into<String>, embedding: &Embedding, r: f64) {
        let projected = self.projection.project(embedding);
        self.index.insert(EmbeddingItem {
            id: id.into(),
            position: SphericalPoint::new_unchecked(r, projected.theta, projected.phi),
            original_magnitude: embedding.magnitude(),
        });
    }

    /// Find the k embeddings whose projected directions are closest to the query.
    pub fn search_nearest(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Vec<NearestResult<EmbeddingItem>> {
        let projected = self.projection.project(query);
        self.index.nearest(&projected, k)
    }

    /// Find all embeddings whose projected cosine similarity to the query
    /// is at least `min_cosine_similarity`.
    ///
    /// Internally maps cos(sim) → angular distance and uses `within_distance`.
    pub fn search_similar(
        &self,
        query: &Embedding,
        min_cosine_similarity: f64,
    ) -> SpatialQueryResult<EmbeddingItem> {
        let projected = self.projection.project(query);
        let max_angle = min_cosine_similarity.clamp(-1.0, 1.0).acos();
        self.index.within_distance(&projected, max_angle)
    }

    pub fn search_region(&self, region: &Region) -> SpatialQueryResult<EmbeddingItem> {
        self.index.query_region(region)
    }

    pub fn remove(&mut self, id: &str) -> Option<EmbeddingItem> {
        self.index.remove(&id.to_string())
    }

    pub fn get(&self, id: &str) -> Option<&EmbeddingItem> {
        self.index.get(&id.to_string())
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn projection(&self) -> &P {
        &self.projection
    }
}

/// Builds SphereQL [`Region`]s from semantic constraints on embeddings.
pub struct SemanticQuery;

impl SemanticQuery {
    /// Spherical cap: all points within `max_angular_distance` radians of the query.
    pub fn within_angle<P: Projection>(
        query: &Embedding,
        projection: &P,
        max_angular_distance: f64,
    ) -> Region {
        let point = projection.project(query);
        let half_angle = max_angular_distance.clamp(1e-10, std::f64::consts::PI);
        Region::Cap(
            Cap::new(
                SphericalPoint::new_unchecked(1.0, point.theta, point.phi),
                half_angle,
            )
            .unwrap(),
        )
    }

    /// Spherical cap from a cosine similarity threshold.
    /// cos_sim >= threshold ↔ angular_distance <= arccos(threshold).
    pub fn above_similarity<P: Projection>(
        query: &Embedding,
        projection: &P,
        min_similarity: f64,
    ) -> Region {
        let half_angle = min_similarity.clamp(-1.0, 1.0).acos();
        Self::within_angle(query, projection, half_angle)
    }

    /// Radial shell: embeddings whose projected radius falls in [inner, outer].
    pub fn in_shell(inner: f64, outer: f64) -> Region {
        Region::Shell(Shell::new(inner, outer).expect("invalid shell bounds"))
    }

    /// Intersection of a similarity cap with a radial shell.
    /// "Semantically similar AND within a magnitude/metadata range."
    pub fn similar_in_shell<P: Projection>(
        query: &Embedding,
        projection: &P,
        min_similarity: f64,
        shell_inner: f64,
        shell_outer: f64,
    ) -> Region {
        Region::intersection(vec![
            Self::above_similarity(query, projection, min_similarity),
            Self::in_shell(shell_inner, shell_outer),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{PcaProjection, RandomProjection};
    use crate::types::RadialStrategy;
    use sphereql_core::angular_distance;

    fn emb(vals: &[f64]) -> Embedding {
        Embedding::new(vals.to_vec())
    }

    fn test_corpus() -> Vec<Embedding> {
        vec![
            emb(&[1.0, 0.0, 0.0, 0.1, 0.0]),
            emb(&[0.0, 1.0, 0.0, 0.0, 0.1]),
            emb(&[0.0, 0.0, 1.0, 0.1, 0.0]),
            emb(&[1.0, 1.0, 0.0, 0.05, 0.05]),
            emb(&[-1.0, 0.0, 0.0, -0.1, 0.0]),
            emb(&[0.0, -1.0, 0.0, 0.0, -0.1]),
        ]
    }

    // --- EmbeddingIndex ---

    #[test]
    fn insert_and_get() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        idx.insert("a", &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]));
        idx.insert("b", &emb(&[0.0, 1.0, 0.0, 0.0, 0.0]));

        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
        assert!(idx.get("a").is_some());
        assert!(idx.get("b").is_some());
        assert!(idx.get("c").is_none());
    }

    #[test]
    fn remove() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();

        idx.insert("a", &emb(&[1.0; 5]));
        assert_eq!(idx.len(), 1);

        let removed = idx.remove("a");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "a");
        assert_eq!(idx.len(), 0);
        assert!(idx.get("a").is_none());
    }

    #[test]
    fn remove_nonexistent() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();
        assert!(idx.remove("nope").is_none());
    }

    #[test]
    fn search_nearest_returns_sorted() {
        let corpus = test_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let mut idx = EmbeddingIndex::builder(pca)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        for (i, e) in corpus.iter().enumerate() {
            idx.insert(format!("item-{i}"), e);
        }

        let query = emb(&[0.95, 0.1, 0.0, 0.05, 0.0]);
        let results = idx.search_nearest(&query, 3);

        assert_eq!(results.len(), 3);
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }

    #[test]
    fn search_similar_respects_threshold() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        idx.insert("close_a", &emb(&[1.0, 0.1, 0.0, 0.0, 0.0]));
        idx.insert("close_b", &emb(&[0.9, 0.2, 0.0, 0.0, 0.0]));
        idx.insert("far", &emb(&[-1.0, 0.0, 0.0, 0.0, 0.0]));

        let query = emb(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let projected_query = idx.projection().project(&query);
        let result = idx.search_similar(&query, 0.5);

        let max_angle = 0.5_f64.acos();
        for item in &result.items {
            let d = angular_distance(&projected_query, item.position());
            assert!(d <= max_angle + 1e-10, "item {} too far: {d}", item.id);
        }
    }

    #[test]
    fn insert_with_radius_overrides() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();

        idx.insert_with_radius("custom", &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]), 42.0);
        let item = idx.get("custom").unwrap();
        assert!((item.position.r - 42.0).abs() < 1e-12);
    }

    #[test]
    fn original_magnitude_stored() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let mut idx = EmbeddingIndex::builder(rp).build();

        let e = emb(&[3.0, 4.0, 0.0, 0.0, 0.0]);
        idx.insert("vec", &e);
        let item = idx.get("vec").unwrap();
        assert!((item.original_magnitude - 5.0).abs() < 1e-10);
    }

    #[test]
    fn magnitude_radial_with_shell_query() {
        let corpus = test_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Magnitude);
        let mut idx = EmbeddingIndex::builder(pca)
            .uniform_shells(5, 10.0)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        idx.insert("small", &emb(&[0.1, 0.0, 0.0, 0.0, 0.0]));
        idx.insert("medium", &emb(&[1.0, 0.0, 0.0, 0.0, 0.0]));
        idx.insert("large", &emb(&[5.0, 0.0, 0.0, 0.0, 0.0]));

        let shell = Shell::new(0.5, 2.0).unwrap();
        let result = idx.search_region(&Region::Shell(shell));

        let ids: Vec<&str> = result.items.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"medium"), "medium (mag=1.0) should be in [0.5, 2.0]");
        assert!(!ids.contains(&"large"), "large (mag=5.0) should not be in [0.5, 2.0]");
    }

    #[test]
    fn empty_index() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let idx = EmbeddingIndex::builder(rp).build();

        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.get("x").is_none());

        let results = idx.search_nearest(&emb(&[1.0; 5]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn projection_accessor() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let idx = EmbeddingIndex::builder(rp).build();
        assert_eq!(idx.projection().dimensionality(), 5);
    }

    // --- SemanticQuery ---

    #[test]
    fn above_similarity_creates_cap() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let region = SemanticQuery::above_similarity(&emb(&[1.0; 5]), &rp, 0.8);
        assert!(matches!(region, Region::Cap(_)));
    }

    #[test]
    fn within_angle_creates_cap() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let region = SemanticQuery::within_angle(&emb(&[1.0; 5]), &rp, 0.5);
        assert!(matches!(region, Region::Cap(_)));
    }

    #[test]
    fn in_shell_creates_shell() {
        let region = SemanticQuery::in_shell(1.0, 5.0);
        assert!(matches!(region, Region::Shell(_)));
    }

    #[test]
    fn similar_in_shell_creates_intersection() {
        let rp = RandomProjection::new(5, RadialStrategy::Fixed(1.0), 42);
        let region = SemanticQuery::similar_in_shell(&emb(&[1.0; 5]), &rp, 0.7, 1.0, 5.0);

        match region {
            Region::Intersection(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(parts[0], Region::Cap(_)));
                assert!(matches!(parts[1], Region::Shell(_)));
            }
            other => panic!("expected Intersection, got {other:?}"),
        }
    }

    #[test]
    fn semantic_query_region_used_in_index() {
        let corpus = test_corpus();
        let pca = PcaProjection::fit(&corpus, RadialStrategy::Fixed(1.0));
        let projection_clone = pca.clone();
        let mut idx = EmbeddingIndex::builder(pca)
            .theta_divisions(4)
            .phi_divisions(3)
            .build();

        for (i, e) in corpus.iter().enumerate() {
            idx.insert(format!("item-{i}"), e);
        }

        let query = emb(&[1.0, 0.0, 0.0, 0.05, 0.0]);
        let region = SemanticQuery::above_similarity(&query, &projection_clone, 0.5);
        let result = idx.search_region(&region);

        for item in &result.items {
            assert!(region.contains(item.position()));
        }
    }
}
