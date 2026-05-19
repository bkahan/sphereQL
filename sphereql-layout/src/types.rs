use sphereql_core::SphericalPoint;

/// An item paired with its assigned position on S².
#[derive(Debug, Clone)]
pub struct LayoutEntry<T> {
    pub item: T,
    pub position: SphericalPoint,
}

/// The output of a layout computation.
#[derive(Debug, Clone)]
pub struct LayoutResult<T> {
    pub entries: Vec<LayoutEntry<T>>,
    pub quality: LayoutQuality,
}

/// Quality metrics for a layout.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayoutQuality {
    /// How evenly spread points are (0.0 = clustered, 1.0 = maximally dispersed).
    pub dispersion_score: f64,
    /// Fraction of point pairs closer than the overlap threshold (0.0 = none, 1.0 = all).
    pub overlap_score: f64,
    /// Silhouette coefficient for cluster quality (-1.0 to 1.0, higher = better separated).
    pub silhouette_score: f64,
}
