use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use sphereql_embed::{Embedding, PcaProjection, Projection, RadialStrategy};
use sphereql_vis::{Scene, ScenePoint, SceneStats};

use crate::pipeline::Pipeline;

/// Human-readable label for the headline quality number, per projection
/// family. Drives the stats-panel label so a UMAP-fitted pipeline no longer
/// reports its quality as "PCA variance".
fn evr_label_for(kind: &str) -> &'static str {
    match kind {
        "pca" => "PCA variance",
        "kernel_pca" => "Kernel EVR",
        "laplacian_eigenmap" => "Connectivity ratio",
        "umap_sphere" => "UMAP kNN-recall",
        _ => "Explained variance ratio",
    }
}

fn write_and_maybe_open(
    py: Python<'_>,
    scene: &Scene,
    output: &str,
    open_browser: bool,
) -> PyResult<String> {
    std::fs::write(output, scene.to_html())
        .map_err(|e| PyValueError::new_err(format!("failed to write {output}: {e}")))?;
    let abs_path = std::fs::canonicalize(output)
        .map_err(|e| PyValueError::new_err(format!("failed to resolve path: {e}")))?
        .to_string_lossy()
        .to_string();
    if open_browser {
        open_in_browser(py, &abs_path)?;
    }
    Ok(abs_path)
}

fn open_in_browser(py: Python<'_>, path: &str) -> PyResult<()> {
    let url = format!("file://{path}");
    let wb = py.import("webbrowser")?;
    wb.call_method1("open", (&url,))?;
    Ok(())
}

/// Generate an interactive 3D sphere visualization from embeddings.
///
/// Fits a PCA projection, projects embeddings to 3D spherical coordinates,
/// and writes a self-contained HTML file with a Three.js scene (the runtime
/// is inlined, so the file works offline).
///
/// Args:
///     categories: Category label for each embedding.
///     embeddings: List of embedding vectors (list[list[float]]).
///     output: Output HTML file path. Default "sphere_viz.html".
///     labels: Optional labels for each point (shown on hover).
///     title: Title shown in the visualization.
///     open_browser: Whether to open the result in a browser. Default True.
///
/// Returns:
///     Absolute path of the generated HTML file.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (categories, embeddings, output="sphere_viz.html", labels=None, title=None, open_browser=true))]
pub fn visualize(
    py: Python<'_>,
    categories: Vec<String>,
    embeddings: Vec<Vec<f64>>,
    output: &str,
    labels: Option<Vec<String>>,
    title: Option<&str>,
    open_browser: bool,
) -> PyResult<String> {
    if categories.len() != embeddings.len() {
        return Err(PyValueError::new_err(format!(
            "categories length ({}) != embeddings length ({})",
            categories.len(),
            embeddings.len()
        )));
    }
    if embeddings.len() < 3 {
        return Err(PyValueError::new_err("need at least 3 embeddings"));
    }

    for (i, row) in embeddings.iter().enumerate() {
        if let Some(j) = row.iter().position(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "embeddings[{i}][{j}] must be finite (no NaN or Inf)"
            )));
        }
    }
    let embs: Vec<Embedding> = embeddings
        .iter()
        .map(|v| Embedding::from(v.as_slice()))
        .collect();

    let pca = PcaProjection::fit(&embs, RadialStrategy::Magnitude)
        .map_err(|e| PyValueError::new_err(format!("PCA fit failed: {e}")))?
        .with_volumetric(true);
    let evr = pca.explained_variance_ratio();

    let points: Vec<ScenePoint> = embs
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let sp = pca.project(emb);
            let label = labels
                .as_ref()
                .and_then(|l| l.get(i))
                .cloned()
                .unwrap_or_default();
            ScenePoint::from_spherical(categories[i].clone(), label, sp.r, sp.theta, sp.phi)
        })
        .collect();

    let scene = Scene::builder()
        .title(title.unwrap_or("SphereQL Visualization"))
        .points(points)
        .stats(SceneStats::new("pca", evr).with_label("PCA variance"))
        .build();

    write_and_maybe_open(py, &scene, output, open_browser)
}

/// Generate a visualization from an already-built Pipeline.
///
/// Reuses the projection fitted inside the pipeline, avoiding re-fitting. The
/// stats panel reports the pipeline's actual projection family and its
/// matching quality metric (e.g. "UMAP kNN-recall"), not a hardcoded label.
/// The pipeline's internal IDs (s-0000, s-0001, ...) are used as labels.
///
/// Args:
///     pipeline: A built sphereql.Pipeline instance.
///     output: Output HTML file path. Default "sphere_viz.html".
///     title: Title shown in the visualization.
///     open_browser: Whether to open the result in a browser. Default True.
///
/// Returns:
///     Absolute path of the generated HTML file.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (pipeline, output="sphere_viz.html", title=None, open_browser=true))]
pub fn visualize_pipeline(
    py: Python<'_>,
    pipeline: &Pipeline,
    output: &str,
    title: Option<&str>,
    open_browser: bool,
) -> PyResult<String> {
    let projected = pipeline.inner.projected_points();
    let evr = pipeline.inner.projection().explained_variance_ratio();
    let kind = pipeline.inner.projection_kind().name();

    let points: Vec<ScenePoint> = projected
        .iter()
        // from_cartesian derives the spherical readout so it matches the
        // pipeline's stored geometry.
        .map(|(id, cat, xyz)| ScenePoint::from_cartesian(cat.to_string(), id.to_string(), *xyz))
        .collect();

    let scene = Scene::builder()
        .title(title.unwrap_or("SphereQL Visualization"))
        .points(points)
        .stats(SceneStats::new(kind, evr).with_label(evr_label_for(kind)))
        .build();

    write_and_maybe_open(py, &scene, output, open_browser)
}
