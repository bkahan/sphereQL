//! Low-dimensional profile of a corpus — features that meta-learning can
//! map to an optimal [`PipelineConfig`](crate::config::PipelineConfig).
//!
//! Meta-learning across corpora needs a stable, compact characterization
//! of "what kind of data is this?" so that past (corpus, best_config)
//! pairs can be indexed and retrieved for prediction on new corpora.
//! [`CorpusFeatures`] is that characterization. Extraction is O(N² · d)
//! for pairwise-similarity features and O(N · d) for shape features;
//! ~100ms at N = 775, d = 128.
//!
//! Every field is a scalar [0, 1] or an unbounded non-negative number.
//! [`CorpusFeatures::to_vec`] flattens to a fixed-length `Vec<f64>` in a
//! stable order matching [`CorpusFeatures::feature_names`].

use std::collections::HashMap;

use sphereql_core::cosine_similarity;

use crate::config::LaplacianConfig;

/// Low-dimensional profile of a corpus. Computed once per corpus; fed
/// into any [`MetaModel`](crate::meta_model::MetaModel) to predict the
/// pipeline config that's likely to work best on it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorpusFeatures {
    /// Total item count.
    pub n_items: usize,
    /// Unique category count.
    pub n_categories: usize,
    /// Embedding dimensionality.
    pub dim: usize,
    /// `n_items / n_categories`.
    pub mean_members_per_category: f64,
    /// Shannon entropy of the category-size distribution, normalized to
    /// `[0, 1]` by dividing by `log(n_categories)`. High = balanced
    /// category sizes; low = heavily skewed.
    pub category_size_entropy: f64,
    /// Mean per-item active-axis fraction: `|active axes| / dim`, averaged
    /// over items. An axis is active when `|v_i| > active_threshold`.
    pub mean_sparsity: f64,
    /// Entropy of how often each axis is active across the corpus,
    /// normalized by `log(dim)`. High = all axes used similarly; low =
    /// a few axes dominate.
    pub axis_utilization_entropy: f64,
    /// Median of `|v_i|` across inactive entries (|v_i| ≤ threshold),
    /// averaged across items. A proxy for the noise floor.
    pub noise_estimate: f64,
    /// Mean intra-category cosine similarity in embedding space. High =
    /// items within a category are tightly clustered.
    pub mean_intra_category_similarity: f64,
    /// Mean inter-category cosine similarity in embedding space. High =
    /// categories overlap heavily; low = categories are semantically
    /// distinct.
    pub mean_inter_category_similarity: f64,
    /// `mean_intra / max(mean_inter, eps)`. A ratio-based separation
    /// signal: values > 1 mean categories separate well in embedding
    /// space; values near 1 mean the corpus is difficult to partition.
    pub category_separation_ratio: f64,
}

/// Length of the vector returned by [`CorpusFeatures::to_vec`].
pub const CORPUS_FEATURE_COUNT: usize = 10;

impl CorpusFeatures {
    /// Stable feature names aligned with [`Self::to_vec`]. Useful for
    /// logging, feature importance reports, and CSV headers.
    ///
    /// Note: `category_separation_ratio` is deliberately excluded — it's
    /// a derived ratio of two features already named here, so including
    /// it would double-count under any distance metric. See
    /// [`Self::to_vec`].
    pub fn feature_names() -> [&'static str; CORPUS_FEATURE_COUNT] {
        [
            "n_items",
            "n_categories",
            "dim",
            "mean_members_per_category",
            "category_size_entropy",
            "mean_sparsity",
            "axis_utilization_entropy",
            "noise_estimate",
            "mean_intra_category_similarity",
            "mean_inter_category_similarity",
        ]
    }

    /// Fixed-length flattened representation in the order declared by
    /// [`Self::feature_names`]. Suitable as input to any nearest-neighbor
    /// or regression meta-model. `category_separation_ratio` is
    /// intentionally *excluded* because it's a derived ratio of two
    /// features already in the vector — keeping it in would double-count.
    pub fn to_vec(&self) -> [f64; CORPUS_FEATURE_COUNT] {
        [
            self.n_items as f64,
            self.n_categories as f64,
            self.dim as f64,
            self.mean_members_per_category,
            self.category_size_entropy,
            self.mean_sparsity,
            self.axis_utilization_entropy,
            self.noise_estimate,
            self.mean_intra_category_similarity,
            self.mean_inter_category_similarity,
        ]
    }

    /// Extract features from a corpus using default Laplacian config
    /// (for the `active_threshold` used in sparsity/noise estimation).
    pub fn extract(categories: &[String], embeddings: &[Vec<f64>]) -> Self {
        Self::extract_with_threshold(
            categories,
            embeddings,
            LaplacianConfig::default().active_threshold,
        )
    }

    /// Extract features with an explicit active-axis threshold. Use this
    /// when you want feature values comparable across different Laplacian
    /// configurations.
    pub fn extract_with_threshold(
        categories: &[String],
        embeddings: &[Vec<f64>],
        active_threshold: f64,
    ) -> Self {
        assert_eq!(
            categories.len(),
            embeddings.len(),
            "categories and embeddings must have matching length"
        );
        let n = embeddings.len();
        let dim = if n > 0 { embeddings[0].len() } else { 0 };
        assert!(n > 0, "cannot extract features from an empty corpus");
        assert!(dim > 0, "embeddings must have positive dimensionality");

        // 1. Category bookkeeping.
        let mut cat_counts: HashMap<&str, usize> = HashMap::new();
        for c in categories {
            *cat_counts.entry(c.as_str()).or_insert(0) += 1;
        }
        let n_categories = cat_counts.len();
        let mean_members_per_category = n as f64 / n_categories.max(1) as f64;

        let category_size_entropy = if n_categories > 1 {
            let h: f64 = cat_counts
                .values()
                .map(|&c| {
                    let p = c as f64 / n as f64;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            // Normalize by ln(n_categories), the maximum for n_categories bins.
            h / (n_categories as f64).ln().max(f64::EPSILON)
        } else {
            0.0
        };

        // 2. Per-axis usage + per-item active counts + noise estimate.
        let mut axis_usage = vec![0usize; dim];
        let mut active_per_item = vec![0usize; n];
        let mut noise_sum = 0.0f64;
        let mut noise_count = 0usize;

        for (i, e) in embeddings.iter().enumerate() {
            let mut inactive_magnitudes: Vec<f64> = Vec::with_capacity(dim);
            for (d, &v) in e.iter().enumerate() {
                if v.abs() > active_threshold {
                    axis_usage[d] += 1;
                    active_per_item[i] += 1;
                } else {
                    inactive_magnitudes.push(v.abs());
                }
            }
            if !inactive_magnitudes.is_empty() {
                inactive_magnitudes
                    .sort_by(|a, b| a.total_cmp(b));
                let median = inactive_magnitudes[inactive_magnitudes.len() / 2];
                noise_sum += median;
                noise_count += 1;
            }
        }

        let mean_sparsity: f64 =
            active_per_item.iter().map(|&a| a as f64).sum::<f64>() / (n * dim) as f64;

        let axis_utilization_entropy = {
            let total: f64 = axis_usage.iter().map(|&c| c as f64).sum();
            if total > 0.0 && dim > 1 {
                let h: f64 = axis_usage
                    .iter()
                    .map(|&c| {
                        let p = c as f64 / total;
                        if p > 0.0 { -p * p.ln() } else { 0.0 }
                    })
                    .sum();
                h / (dim as f64).ln().max(f64::EPSILON)
            } else {
                0.0
            }
        };

        let noise_estimate = if noise_count > 0 {
            noise_sum / noise_count as f64
        } else {
            0.0
        };

        // 3. Pairwise intra/inter category similarity.
        let mean_intra_category_similarity =
            pairwise_similarity(embeddings, categories, SimilarityMode::IntraCategory);
        let mean_inter_category_similarity =
            pairwise_similarity(embeddings, categories, SimilarityMode::InterCategory);
        let category_separation_ratio =
            mean_intra_category_similarity / mean_inter_category_similarity.abs().max(1e-12);

        Self {
            n_items: n,
            n_categories,
            dim,
            mean_members_per_category,
            category_size_entropy,
            mean_sparsity,
            axis_utilization_entropy,
            noise_estimate,
            mean_intra_category_similarity,
            mean_inter_category_similarity,
            category_separation_ratio,
        }
    }
}

// ── Similarity helpers ─────────────────────────────────────────────────

enum SimilarityMode {
    IntraCategory,
    InterCategory,
}

fn pairwise_similarity(
    embeddings: &[Vec<f64>],
    categories: &[String],
    mode: SimilarityMode,
) -> f64 {
    let n = embeddings.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count: usize = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let same = categories[i] == categories[j];
            let use_pair = match mode {
                SimilarityMode::IntraCategory => same,
                SimilarityMode::InterCategory => !same,
            };
            if !use_pair {
                continue;
            }
            sum += cosine_similarity(&embeddings[i], &embeddings[j]);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_corpus() -> (Vec<String>, Vec<Vec<f64>>) {
        let categories: Vec<String> = vec![
            "a".into(),
            "a".into(),
            "a".into(),
            "b".into(),
            "b".into(),
            "b".into(),
        ];
        let embeddings = vec![
            vec![1.0, 0.1, 0.0, 0.0, 0.02],
            vec![0.9, 0.15, 0.0, 0.0, 0.01],
            vec![0.95, 0.05, 0.0, 0.0, 0.03],
            vec![0.1, 0.0, 1.0, 0.0, 0.02],
            vec![0.15, 0.0, 0.9, 0.0, 0.01],
            vec![0.05, 0.0, 0.95, 0.0, 0.03],
        ];
        (categories, embeddings)
    }

    #[test]
    fn extract_basic_shape() {
        let (cats, embs) = toy_corpus();
        let cf = CorpusFeatures::extract(&cats, &embs);
        assert_eq!(cf.n_items, 6);
        assert_eq!(cf.n_categories, 2);
        assert_eq!(cf.dim, 5);
        assert!((cf.mean_members_per_category - 3.0).abs() < 1e-12);
    }

    #[test]
    fn category_size_entropy_balanced() {
        // Balanced 3/3 split → maximum entropy = 1.0 (after log normalization).
        let (cats, embs) = toy_corpus();
        let cf = CorpusFeatures::extract(&cats, &embs);
        assert!(
            (cf.category_size_entropy - 1.0).abs() < 1e-10,
            "balanced split should give entropy = 1.0, got {}",
            cf.category_size_entropy
        );
    }

    #[test]
    fn category_size_entropy_skewed() {
        // 5/1 split → lower entropy than balanced.
        let cats: Vec<String> = vec!["a", "a", "a", "a", "a", "b"]
            .into_iter()
            .map(Into::into)
            .collect();
        let embs = vec![vec![1.0, 0.0, 0.0]; 6];
        let cf = CorpusFeatures::extract(&cats, &embs);
        assert!(
            cf.category_size_entropy < 0.9,
            "skewed split should give entropy < 0.9, got {}",
            cf.category_size_entropy
        );
    }

    #[test]
    fn sparsity_matches_threshold() {
        let (cats, embs) = toy_corpus();
        // With threshold 0.05: each 5-dim vector has 2 active axes → sparsity 2/5 = 0.4
        let cf = CorpusFeatures::extract_with_threshold(&cats, &embs, 0.05);
        assert!(
            (cf.mean_sparsity - 0.4).abs() < 0.11,
            "expected ~0.4, got {}",
            cf.mean_sparsity
        );
    }

    #[test]
    fn intra_higher_than_inter_for_well_separated() {
        let (cats, embs) = toy_corpus();
        let cf = CorpusFeatures::extract(&cats, &embs);
        assert!(
            cf.mean_intra_category_similarity > cf.mean_inter_category_similarity,
            "expected intra > inter on well-separated corpus"
        );
        assert!(cf.category_separation_ratio > 1.0);
    }

    #[test]
    fn to_vec_length_matches_feature_names() {
        let (cats, embs) = toy_corpus();
        let cf = CorpusFeatures::extract(&cats, &embs);
        assert_eq!(cf.to_vec().len(), CorpusFeatures::feature_names().len());
        assert_eq!(cf.to_vec().len(), CORPUS_FEATURE_COUNT);
    }

    #[test]
    fn features_serialize_json_roundtrip() {
        let (cats, embs) = toy_corpus();
        let cf = CorpusFeatures::extract(&cats, &embs);
        let json = serde_json::to_string(&cf).unwrap();
        let back: CorpusFeatures = serde_json::from_str(&json).unwrap();
        assert_eq!(cf.n_items, back.n_items);
        assert_eq!(cf.n_categories, back.n_categories);
        assert!(
            (cf.mean_intra_category_similarity - back.mean_intra_category_similarity).abs() < 1e-12
        );
    }

    #[test]
    fn empty_inactive_sets_produce_zero_noise() {
        // All axes active — noise_estimate defaults to 0.
        let cats: Vec<String> = vec!["a".into(), "a".into()];
        let embs = vec![vec![1.0, 1.0], vec![0.9, 0.9]];
        let cf = CorpusFeatures::extract_with_threshold(&cats, &embs, 0.05);
        assert_eq!(cf.noise_estimate, 0.0);
    }
}
