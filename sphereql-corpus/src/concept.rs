/// A concept in the test corpus.
pub struct Concept {
    pub label: &'static str,
    pub category: &'static str,
    pub features: Vec<(usize, f64)>,
}
