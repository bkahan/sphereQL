//! Python bindings for the metalearning surface: corpus features,
//! auto-tuner, meta-model. Kept separate from `pipeline.rs` because
//! these live at module level rather than on the [`Pipeline`] class.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sphereql_embed::corpus_features::CorpusFeatures;

use crate::projection::extract_embeddings_2d;

/// Extract a corpus feature profile.
///
/// Returns a dict matching [`CorpusFeatures`]'s serde representation:
/// `n_items`, `n_categories`, `dim`, `mean_members_per_category`,
/// `category_size_entropy`, `mean_sparsity`, `axis_utilization_entropy`,
/// `noise_estimate`, `mean_intra_category_similarity`,
/// `mean_inter_category_similarity`, `category_separation_ratio`.
///
/// This is the input any [`MetaModel`] uses to predict a `PipelineConfig`
/// for a previously-unseen corpus.
///
/// Args:
///     categories: Category label per embedding.
///     embeddings: List of embedding vectors (list[list[float]] or np.ndarray).
///
/// Returns:
///     dict of corpus features.
#[pyfunction]
#[pyo3(signature = (categories, embeddings))]
pub fn corpus_features<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    embeddings: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let embs = extract_embeddings_2d(embeddings)?;
    if categories.len() != embs.len() {
        return Err(PyValueError::new_err(format!(
            "categories length ({}) != embeddings length ({})",
            categories.len(),
            embs.len()
        )));
    }
    let raw: Vec<Vec<f64>> = embs.iter().map(|e| e.values.clone()).collect();
    let features = py.detach(|| CorpusFeatures::extract(&categories, &raw));
    pythonize::pythonize(py, &features)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize CorpusFeatures: {e}")))
}
