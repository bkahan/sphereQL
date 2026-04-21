use pyo3::prelude::*;

use sphereql_core::SphericalPoint;
use sphereql_embed::navigator;
use sphereql_embed::pipeline::SphereQLPipeline;

use crate::pipeline::Pipeline;

#[pyclass(name = "NavigatorConfig", from_py_object)]
#[derive(Clone)]
pub struct PyNavigatorConfig {
    #[pyo3(get, set)]
    pub antipodal_radius: f64,
    #[pyo3(get, set)]
    pub coverage_samples: usize,
    #[pyo3(get, set)]
    pub geodesic_epsilon: f64,
    #[pyo3(get, set)]
    pub density_bins: usize,
    #[pyo3(get, set)]
    pub voronoi_samples: usize,
    #[pyo3(get, set)]
    pub exclusivity_samples: usize,
    #[pyo3(get, set)]
    pub curvature_top_n: usize,
    #[pyo3(get, set)]
    pub gap_sharpness: f64,
}

#[pymethods]
impl PyNavigatorConfig {
    #[new]
    #[pyo3(signature = (
        antipodal_radius=0.5, coverage_samples=200_000,
        geodesic_epsilon=0.3, density_bins=20,
        voronoi_samples=200_000, exclusivity_samples=50_000,
        curvature_top_n=20, gap_sharpness=5.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        antipodal_radius: f64,
        coverage_samples: usize,
        geodesic_epsilon: f64,
        density_bins: usize,
        voronoi_samples: usize,
        exclusivity_samples: usize,
        curvature_top_n: usize,
        gap_sharpness: f64,
    ) -> Self {
        Self {
            antipodal_radius,
            coverage_samples,
            geodesic_epsilon,
            density_bins,
            voronoi_samples,
            exclusivity_samples,
            curvature_top_n,
            gap_sharpness,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NavigatorConfig(coverage_samples={}, voronoi_samples={})",
            self.coverage_samples, self.voronoi_samples
        )
    }
}

impl From<&PyNavigatorConfig> for navigator::NavigatorConfig {
    fn from(py: &PyNavigatorConfig) -> Self {
        navigator::NavigatorConfig {
            antipodal_radius: py.antipodal_radius,
            coverage_samples: py.coverage_samples,
            geodesic_epsilon: py.geodesic_epsilon,
            density_bins: py.density_bins,
            voronoi_samples: py.voronoi_samples,
            exclusivity_samples: py.exclusivity_samples,
            curvature_top_n: py.curvature_top_n,
            gap_sharpness: py.gap_sharpness,
        }
    }
}

#[pyclass(name = "AntipodalReport", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyAntipodalReport {
    #[pyo3(get)]
    pub category_name: String,
    #[pyo3(get)]
    pub antipodal_coherence: f64,
    #[pyo3(get)]
    pub antipodal_item_count: usize,
    #[pyo3(get)]
    pub dominant_antipodal_category: Option<String>,
}

#[pyclass(name = "CoverageReport", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCoverageReport {
    #[pyo3(get)]
    pub coverage_fraction: f64,
    #[pyo3(get)]
    pub covered_area: f64,
    #[pyo3(get)]
    pub overlap_area: f64,
    #[pyo3(get)]
    pub void_samples: usize,
    #[pyo3(get)]
    pub total_samples: usize,
}

#[pyclass(name = "VoronoiCell", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyVoronoiCell {
    #[pyo3(get)]
    pub category_name: String,
    #[pyo3(get)]
    pub cell_area: f64,
    #[pyo3(get)]
    pub item_count: usize,
    #[pyo3(get)]
    pub territorial_efficiency: f64,
    #[pyo3(get)]
    pub graph_neighbor_overlap: f64,
    #[pyo3(get)]
    pub voronoi_neighbors: Vec<String>,
}

#[pyclass(name = "CurvatureTriple", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCurvatureTriple {
    #[pyo3(get)]
    pub categories: Vec<String>,
    #[pyo3(get)]
    pub excess: f64,
}

#[pyclass(name = "LuneReport", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyLuneReport {
    #[pyo3(get)]
    pub category_a: String,
    #[pyo3(get)]
    pub category_b: String,
    #[pyo3(get)]
    pub a_leaning_count: usize,
    #[pyo3(get)]
    pub b_leaning_count: usize,
    #[pyo3(get)]
    pub on_bisector_count: usize,
    #[pyo3(get)]
    pub asymmetry: f64,
}

#[pyclass(name = "NavigatorReport", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyNavigatorReport {
    #[pyo3(get)]
    pub antipodal: Vec<PyAntipodalReport>,
    #[pyo3(get)]
    pub coverage: PyCoverageReport,
    #[pyo3(get)]
    pub voronoi: Vec<PyVoronoiCell>,
    #[pyo3(get)]
    pub top_curvature_triples: Vec<PyCurvatureTriple>,
    #[pyo3(get)]
    pub lunes: Vec<PyLuneReport>,
    #[pyo3(get)]
    pub num_categories: usize,
    #[pyo3(get)]
    pub num_items: usize,
    #[pyo3(get)]
    pub explained_variance_ratio: f64,
}

#[pyfunction]
#[pyo3(signature = (pipeline, *, config=None))]
pub fn run_navigator(
    pipeline: &Pipeline,
    config: Option<&PyNavigatorConfig>,
) -> PyResult<PyNavigatorReport> {
    let cfg = config
        .map(navigator::NavigatorConfig::from)
        .unwrap_or_default();

    let inner: &SphereQLPipeline = &pipeline.inner;
    let layer = inner.category_layer();
    let positions: Vec<SphericalPoint> = inner
        .exported_points()
        .iter()
        .map(|p| SphericalPoint::new_unchecked(p.r, p.theta, p.phi))
        .collect();
    let categories: Vec<String> = inner.categories().to_vec();
    let evr = inner.explained_variance_ratio();

    let report = navigator::run_full_analysis(layer, &positions, &categories, evr, &cfg);

    Ok(PyNavigatorReport {
        antipodal: report
            .antipodal
            .into_iter()
            .map(|a| PyAntipodalReport {
                category_name: a.category_name,
                antipodal_coherence: a.antipodal_coherence,
                antipodal_item_count: a.antipodal_items.len(),
                dominant_antipodal_category: a.dominant_antipodal_category,
            })
            .collect(),
        coverage: PyCoverageReport {
            coverage_fraction: report.coverage.coverage_fraction,
            covered_area: report.coverage.covered_area,
            overlap_area: report.coverage.overlap_area,
            void_samples: report.coverage.void_samples,
            total_samples: report.coverage.total_samples,
        },
        voronoi: report
            .voronoi
            .cells
            .into_iter()
            .map(|c| PyVoronoiCell {
                category_name: c.category_name,
                cell_area: c.cell_area,
                item_count: c.item_count,
                territorial_efficiency: c.territorial_efficiency,
                graph_neighbor_overlap: c.graph_neighbor_overlap,
                voronoi_neighbors: c.voronoi_neighbors,
            })
            .collect(),
        top_curvature_triples: report
            .curvature
            .top_triples
            .into_iter()
            .map(|t| PyCurvatureTriple {
                categories: t.categories.to_vec(),
                excess: t.excess,
            })
            .collect(),
        lunes: report
            .lunes
            .into_iter()
            .map(|l| PyLuneReport {
                category_a: l.category_a,
                category_b: l.category_b,
                a_leaning_count: l.a_leaning_count,
                b_leaning_count: l.b_leaning_count,
                on_bisector_count: l.on_bisector_count,
                asymmetry: l.asymmetry,
            })
            .collect(),
        num_categories: report.num_categories,
        num_items: report.num_items,
        explained_variance_ratio: report.explained_variance_ratio,
    })
}
