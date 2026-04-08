use sphereql_core::SphericalPoint;

#[derive(Debug, Clone)]
pub struct LayoutEntry<T> {
    pub item: T,
    pub position: SphericalPoint,
}

#[derive(Debug, Clone)]
pub struct LayoutResult<T> {
    pub entries: Vec<LayoutEntry<T>>,
    pub quality: LayoutQuality,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LayoutQuality {
    /// How evenly spread points are (0.0 = clustered, 1.0 = maximally dispersed)
    pub dispersion_score: f64,
    /// Fraction of point pairs closer than a threshold (0.0 = no overlaps, 1.0 = all overlapping)
    pub overlap_score: f64,
    /// Silhouette coefficient for cluster quality (-1.0 to 1.0, higher = better separated clusters)
    pub silhouette_score: f64,
}
