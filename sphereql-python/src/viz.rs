use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use sphereql_core::conversions::{cartesian_to_spherical, spherical_to_cartesian};
use sphereql_core::types::CartesianPoint;
use sphereql_embed::{Embedding, PcaProjection, Projection, RadialStrategy};

use crate::pipeline::Pipeline;

const TEMPLATE: &str = include_str!("viz_template.html");

#[derive(serde::Serialize)]
struct VizPoint<'a> {
    x: f64,
    y: f64,
    z: f64,
    r: f64,
    theta: f64,
    phi: f64,
    cat: &'a str,
    label: &'a str,
}

#[derive(serde::Serialize)]
struct VizData<'a> {
    evr: f64,
    points: Vec<VizPoint<'a>>,
}

fn build_data_json(
    categories: &[String],
    cart_points: &[[f64; 3]],
    spherical: &[(f64, f64, f64)],
    labels: Option<&[String]>,
    explained_variance: f64,
) -> String {
    let points = cart_points
        .iter()
        .zip(spherical.iter())
        .zip(categories.iter())
        .enumerate()
        .map(|(i, ((xyz, sph), cat))| VizPoint {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
            r: sph.0,
            theta: sph.1,
            phi: sph.2,
            cat: cat.as_str(),
            label: labels
                .and_then(|l| l.get(i))
                .map(String::as_str)
                .unwrap_or(""),
        })
        .collect();
    let data = VizData {
        evr: explained_variance,
        points,
    };
    // Infallible: every field is a finite f64 or a borrowed &str.
    serde_json::to_string(&data).unwrap_or_else(|_| "{}".into())
}

fn render_html(data_json: &str, title: &str) -> String {
    TEMPLATE
        .replace("/*__SPHEREQL_DATA__*/", data_json)
        .replace("__SPHEREQL_TITLE__", title)
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
/// and writes a self-contained HTML file with a Three.js scene.
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

    let embs: Vec<Embedding> = embeddings
        .iter()
        .map(|v| Embedding::from(v.as_slice()))
        .collect();

    let pca = PcaProjection::fit(&embs, RadialStrategy::Magnitude)
        .map_err(|e| PyValueError::new_err(format!("PCA fit failed: {e}")))?
        .with_volumetric(true);
    let evr = pca.explained_variance_ratio();

    let mut cart_points = Vec::with_capacity(embs.len());
    let mut spherical = Vec::with_capacity(embs.len());
    for emb in &embs {
        let sp = pca.project(emb);
        let c = spherical_to_cartesian(&sp);
        cart_points.push([c.x, c.y, c.z]);
        spherical.push((sp.r, sp.theta, sp.phi));
    }

    let data_json = build_data_json(
        &categories,
        &cart_points,
        &spherical,
        labels.as_deref(),
        evr,
    );
    let title_str = title.unwrap_or("SphereQL Visualization");
    let html = render_html(&data_json, title_str);

    std::fs::write(output, &html)
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

/// Generate a visualization from an already-built Pipeline.
///
/// Reuses the PCA projection fitted inside the pipeline, avoiding
/// re-fitting. The pipeline's internal IDs (s-0000, s-0001, ...) are
/// used as labels unless the pipeline items have associated text.
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
    let points = pipeline.inner.projected_points();
    let evr = pipeline.inner.projection().explained_variance_ratio();

    let mut categories = Vec::with_capacity(points.len());
    let mut cart_points = Vec::with_capacity(points.len());
    let mut labels = Vec::with_capacity(points.len());
    let mut spherical = Vec::with_capacity(points.len());

    for (id, cat, xyz) in &points {
        categories.push(cat.to_string());
        cart_points.push(*xyz);
        labels.push(id.to_string());
        let cp = CartesianPoint {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
        };
        let sp = cartesian_to_spherical(&cp);
        spherical.push((sp.r, sp.theta, sp.phi));
    }

    let data_json = build_data_json(&categories, &cart_points, &spherical, Some(&labels), evr);
    let title_str = title.unwrap_or("SphereQL Visualization");
    let html = render_html(&data_json, title_str);

    std::fs::write(output, &html)
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
