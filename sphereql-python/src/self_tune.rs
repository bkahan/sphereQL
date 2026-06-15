//! Python binding for the Phase-7 corpus self-tune surface.
//!
//! Mirrors the patterns in [`crate::meta::auto_tune`]: optional config
//! dicts are depythonized into the corresponding `sphereql-embed` structs
//! (falling back to their `Default`), the heavy loop runs inside
//! `py.detach`, and the report is hand-assembled into a `PyDict` so the
//! serde-free `SelfTuneReport` types surface as plain Python dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use sphereql_embed::config::PipelineConfig;
use sphereql_embed::corpus_quality::{CorpusQuality, CorpusQualityWeights};
use sphereql_embed::self_tune::{
    SelfTuneConfig, SelfTuneReport, StopReason, TunableConcept, run_self_tune as rust_run_self_tune,
};

/// Default embed seed — the value Phase 6's binary uses so the synthetic
/// noise is reproducible across runs.
const DEFAULT_SEED: u64 = 0xDEADBEEF;

fn stop_reason_str(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Plateau => "plateau",
        StopReason::MaxIterations => "max_iterations",
        StopReason::PruneFloorHit => "prune_floor_hit",
    }
}

/// Build the per-iteration / final report as a Python dict.
///
/// Each iteration flattens its `CorpusQualityBreakdown` into the five
/// sub-score keys alongside the per-iteration counters, matching the
/// dict shape `auto_tune` returns for its tune report.
fn build_report_dict<'py>(
    py: Python<'py>,
    report: &SelfTuneReport,
) -> PyResult<Bound<'py, PyDict>> {
    let iterations = PyList::empty(py);
    for it in &report.iterations {
        let d = PyDict::new(py);
        d.set_item("iteration", it.iteration)?;
        d.set_item("n_concepts", it.n_concepts)?;
        d.set_item("composite_score", it.composite_score)?;
        d.set_item("n_pruned", it.n_pruned)?;
        d.set_item("mean_quality", it.mean_quality)?;
        d.set_item("mean_quality_delta", it.mean_quality_delta)?;
        d.set_item("evr", it.breakdown.evr)?;
        d.set_item("bridge_coherence", it.breakdown.bridge_coherence)?;
        d.set_item("curvature_health", it.breakdown.curvature_health)?;
        d.set_item("category_balance", it.breakdown.category_balance)?;
        d.set_item("composite", it.breakdown.composite)?;
        iterations.append(d)?;
    }

    let report_dict = PyDict::new(py);
    report_dict.set_item("iterations", iterations)?;
    report_dict.set_item("stopped_reason", stop_reason_str(report.stopped_reason))?;
    report_dict.set_item("final_composite", report.final_composite)?;
    Ok(report_dict)
}

/// Run one corpus self-tune loop and return the mutated corpus and report.
///
/// Mirrors the standalone `sphereql_embed::run_self_tune`, owning the
/// embed closure (fixed to the deterministic synthetic embedder so the
/// run is reproducible) rather than taking an arbitrary callback.
///
/// Args:
///     concepts: List of concept dicts. Each must carry `label`,
///         `category`, `features` (list of `[axis, weight]` pairs),
///         `quality`, `axis_coherence`, `bridge_degree`,
///         `source_confidence`, `home_affinity`, and the optional
///         `source` / `openalex_id`.
///     base_config: Optional dict of PipelineConfig. Default:
///         PipelineConfig.default().
///     cfg: Optional dict of SelfTuneConfig (max_iterations,
///         plateau_epsilon, min_quality_to_keep, etc.). Any field may be
///         omitted. Default: SelfTuneConfig.default().
///     weights: Optional dict of CorpusQualityWeights (`w_evr`,
///         `w_bridge`, `w_curvature`, `w_balance`). Default:
///         0.30 / 0.30 / 0.20 / 0.20.
///     seed: Embed-noise seed for the deterministic synthetic embedder.
///         Default: 0xDEADBEEF.
///
/// Returns:
///     Tuple `(concepts, report)` where `concepts` is the mutated
///     (possibly pruned) corpus as a list of dicts, and `report` is a
///     dict with `iterations` (each carrying the iteration counters plus
///     the five flattened breakdown sub-scores), `stopped_reason`
///     (`"plateau"`, `"max_iterations"`, or `"prune_floor_hit"`), and
///     `final_composite` (`float` or `None`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (concepts, base_config = None, cfg = None, weights = None, seed = DEFAULT_SEED))]
pub fn run_self_tune<'py>(
    py: Python<'py>,
    concepts: &Bound<'_, PyAny>,
    base_config: Option<&Bound<'_, PyAny>>,
    cfg: Option<&Bound<'_, PyAny>>,
    weights: Option<&Bound<'_, PyAny>>,
    seed: u64,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let corpus: Vec<TunableConcept> = pythonize::depythonize(concepts)
        .map_err(|e| PyValueError::new_err(format!("invalid concepts list: {e}")))?;

    let base = match base_config {
        Some(obj) => pythonize::depythonize::<PipelineConfig>(obj)
            .map_err(|e| PyValueError::new_err(format!("invalid base_config dict: {e}")))?,
        None => PipelineConfig::default(),
    };
    let tune_cfg = match cfg {
        Some(obj) => pythonize::depythonize::<SelfTuneConfig>(obj)
            .map_err(|e| PyValueError::new_err(format!("invalid cfg dict: {e}")))?,
        None => SelfTuneConfig::default(),
    };
    let quality_weights = match weights {
        Some(obj) => pythonize::depythonize::<CorpusQualityWeights>(obj)
            .map_err(|e| PyValueError::new_err(format!("invalid weights dict: {e}")))?,
        None => CorpusQualityWeights::default(),
    };
    quality_weights
        .validate()
        .map_err(|e| PyValueError::new_err(format!("invalid weights: {e}")))?;

    let (tuned, report) = py
        .detach(move || {
            let quality = CorpusQuality::new(quality_weights);
            rust_run_self_tune(
                corpus,
                |f: &[(usize, f64)]| sphereql_core::synthetic::embed(f, seed),
                base,
                &quality,
                &tune_cfg,
            )
        })
        .map_err(PyValueError::new_err)?;

    let tuned_list = pythonize::pythonize(py, &tuned)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize tuned corpus: {e}")))?;
    let report_dict = build_report_dict(py, &report)?;

    Ok((tuned_list, report_dict))
}
