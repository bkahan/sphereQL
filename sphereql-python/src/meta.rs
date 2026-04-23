//! Python bindings for the metalearning surface: corpus features,
//! auto-tuner, meta-model. Kept separate from `pipeline.rs` because
//! these live at module level rather than on the [`Pipeline`] class.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sphereql_embed::config::PipelineConfig;
use sphereql_embed::corpus_features::CorpusFeatures;
use sphereql_embed::pipeline::PipelineInput;
use sphereql_embed::quality_metric::{
    BridgeCoherence, ClusterSilhouette, CompositeMetric, GraphModularity, TerritorialHealth,
};
use sphereql_embed::tuner::{SearchSpace, SearchStrategy, auto_tune as rust_auto_tune};

use crate::pipeline::Pipeline;
use crate::projection::extract_embeddings_2d;

fn run_auto_tune(
    metric: &str,
    input: PipelineInput,
    space: &SearchSpace,
    strategy: SearchStrategy,
    base: &PipelineConfig,
) -> Result<
    (
        sphereql_embed::pipeline::SphereQLPipeline,
        sphereql_embed::tuner::TuneReport,
    ),
    sphereql_embed::pipeline::PipelineError,
> {
    match metric {
        "territorial_health" => {
            rust_auto_tune(input, space, &TerritorialHealth, strategy, base)
        }
        "bridge_coherence" => rust_auto_tune(input, space, &BridgeCoherence, strategy, base),
        "cluster_silhouette" => {
            rust_auto_tune(input, space, &ClusterSilhouette, strategy, base)
        }
        "graph_modularity" => {
            rust_auto_tune(input, space, &GraphModularity::default(), strategy, base)
        }
        "default_composite" => rust_auto_tune(
            input,
            space,
            &CompositeMetric::default_composite(),
            strategy,
            base,
        ),
        "connectivity_composite" => rust_auto_tune(
            input,
            space,
            &CompositeMetric::connectivity_composite(),
            strategy,
            base,
        ),
        _ => unreachable!("metric name validated before dispatch"),
    }
}

fn validate_metric(name: &str) -> PyResult<()> {
    match name {
        "territorial_health"
        | "bridge_coherence"
        | "cluster_silhouette"
        | "graph_modularity"
        | "default_composite"
        | "connectivity_composite" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?}; expected one of: \
             territorial_health, bridge_coherence, cluster_silhouette, \
             graph_modularity, default_composite, connectivity_composite"
        ))),
    }
}

fn resolve_strategy(
    kind: &str,
    budget: usize,
    seed: u64,
    warmup: usize,
    gamma: f64,
) -> PyResult<SearchStrategy> {
    match kind {
        "grid" => Ok(SearchStrategy::Grid),
        "random" => Ok(SearchStrategy::Random { budget, seed }),
        "bayesian" => Ok(SearchStrategy::Bayesian {
            budget,
            warmup,
            gamma,
            seed,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown strategy {other:?}; expected one of: grid, random, bayesian"
        ))),
    }
}

/// Run the auto-tuner over a default [`SearchSpace`] and return the
/// best-scoring pipeline alongside a summary report.
///
/// Args:
///     categories: Category label per embedding.
///     embeddings: List of embedding vectors.
///     metric: Quality metric name. One of `territorial_health`,
///         `bridge_coherence`, `cluster_silhouette`, `graph_modularity`,
///         `default_composite`, `connectivity_composite`.
///         Default: `default_composite`.
///     strategy: One of `grid`, `random`, `bayesian`. Default: `random`.
///     budget: Number of trials for `random` / `bayesian`. Default: 24.
///     seed: RNG seed for `random` / `bayesian`. Default: 0.
///     warmup: Warmup trials before acquisition kicks in (Bayesian only).
///         Default: 4.
///     gamma: Fraction of trials treated as "good" (Bayesian only).
///         Default: 0.25.
///     base_config: Optional dict of PipelineConfig. Non-tuned knobs
///         stay at these values; tuned knobs explore the search space
///         starting from them. Default: PipelineConfig.default().
///
/// Returns:
///     Tuple `(Pipeline, report)` where `report` is a dict with
///     `metric_name`, `best_score`, `best_config`, `trials_count`,
///     `failures_count`, `mean_score`.
#[pyfunction]
#[pyo3(signature = (
    categories,
    embeddings,
    *,
    metric = "default_composite",
    strategy = "random",
    budget = 24,
    seed = 0,
    warmup = 4,
    gamma = 0.25,
    base_config = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn auto_tune<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    embeddings: &Bound<'_, PyAny>,
    metric: &str,
    strategy: &str,
    budget: usize,
    seed: u64,
    warmup: usize,
    gamma: f64,
    base_config: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Pipeline, Bound<'py, PyAny>)> {
    let embs = extract_embeddings_2d(embeddings)?;
    if categories.len() != embs.len() {
        return Err(PyValueError::new_err(format!(
            "categories length ({}) != embeddings length ({})",
            categories.len(),
            embs.len()
        )));
    }
    let dim = embs.first().map(|e| e.dimension()).unwrap_or(0);
    let raw: Vec<Vec<f64>> = embs.iter().map(|e| e.values.clone()).collect();

    let base = match base_config {
        Some(obj) => pythonize::depythonize::<PipelineConfig>(obj).map_err(|e| {
            PyValueError::new_err(format!("invalid base_config dict: {e}"))
        })?,
        None => PipelineConfig::default(),
    };

    validate_metric(metric)?;
    let search_strategy = resolve_strategy(strategy, budget, seed, warmup, gamma)?;
    let space = SearchSpace::default();
    let metric_owned = metric.to_string();

    let (pipeline, report) = py
        .detach(move || {
            run_auto_tune(
                &metric_owned,
                PipelineInput {
                    categories,
                    embeddings: raw,
                },
                &space,
                search_strategy,
                &base,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let report_dict = PyDict::new(py);
    report_dict.set_item("metric_name", &report.metric_name)?;
    report_dict.set_item("best_score", report.best_score)?;
    report_dict.set_item(
        "best_config",
        pythonize::pythonize(py, &report.best_config).map_err(|e| {
            PyValueError::new_err(format!("failed to serialize best_config: {e}"))
        })?,
    )?;
    report_dict.set_item("trials_count", report.trials.len())?;
    report_dict.set_item("failures_count", report.failures.len())?;
    report_dict.set_item("mean_score", report.mean_score())?;

    Ok((Pipeline::from_inner(pipeline, dim), report_dict.into_any()))
}

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
