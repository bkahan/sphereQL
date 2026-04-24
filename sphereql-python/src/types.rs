use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use sphereql_embed::category::{
    BridgeClassification, BridgeItem, CategoryPath, CategoryPathStep, CategorySummary,
    DrillDownResult, InnerSphereReport,
};
use sphereql_embed::confidence::{ProjectionWarning, WarningSeverity};
use sphereql_embed::domain_groups::DomainGroup;
use sphereql_embed::pipeline::{GlobSummary, ManifoldResult, NearestResult, PathResult};

fn classification_name(c: BridgeClassification) -> &'static str {
    match c {
        BridgeClassification::Genuine => "Genuine",
        BridgeClassification::OverlapArtifact => "OverlapArtifact",
        BridgeClassification::Weak => "Weak",
    }
}

fn severity_name(s: WarningSeverity) -> &'static str {
    match s {
        WarningSeverity::Info => "Info",
        WarningSeverity::Warning => "Warning",
        WarningSeverity::Critical => "Critical",
    }
}

// ── Nearest ────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "NearestHit", frozen, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct Nearest {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub category: String,
    #[pyo3(get)]
    pub distance: f64,
    #[pyo3(get)]
    pub certainty: f64,
    #[pyo3(get)]
    pub intensity: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl Nearest {
    fn __repr__(&self) -> String {
        format!(
            "NearestHit(id={:?}, category={:?}, distance={:.4})",
            self.id, self.category, self.distance
        )
    }

    fn __eq__(&self, other: &Nearest) -> bool {
        self == other
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "id": self.id,
            "category": self.category,
            "distance": self.distance,
            "certainty": self.certainty,
            "intensity": self.intensity,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            id: v["id"].as_str().unwrap_or("").to_string(),
            category: v["category"].as_str().unwrap_or("").to_string(),
            distance: v["distance"].as_f64().unwrap_or(0.0),
            certainty: v["certainty"].as_f64().unwrap_or(0.0),
            intensity: v["intensity"].as_f64().unwrap_or(0.0),
        })
    }
}

impl From<&NearestResult> for Nearest {
    fn from(r: &NearestResult) -> Self {
        Self {
            id: r.id.clone(),
            category: r.category.clone(),
            distance: r.distance,
            certainty: r.certainty,
            intensity: r.intensity,
        }
    }
}

// ── PathStep ───────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(frozen, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PathStep {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub category: String,
    #[pyo3(get)]
    pub cumulative_distance: f64,
    #[pyo3(get)]
    pub hop_distance: f64,
    #[pyo3(get)]
    pub bridge_strength: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PathStep {
    fn __repr__(&self) -> String {
        format!(
            "PathStep(id={:?}, category={:?}, cumulative_distance={:.4}, hop={:.4})",
            self.id, self.category, self.cumulative_distance, self.hop_distance
        )
    }

    fn __eq__(&self, other: &PathStep) -> bool {
        self == other
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "id": self.id,
            "category": self.category,
            "cumulative_distance": self.cumulative_distance,
            "hop_distance": self.hop_distance,
            "bridge_strength": self.bridge_strength,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            id: v["id"].as_str().unwrap_or("").to_string(),
            category: v["category"].as_str().unwrap_or("").to_string(),
            cumulative_distance: v["cumulative_distance"].as_f64().unwrap_or(0.0),
            hop_distance: v["hop_distance"].as_f64().unwrap_or(0.0),
            bridge_strength: v["bridge_strength"].as_f64(),
        })
    }
}

// ── Path ───────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "PathResult", frozen, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct Path {
    #[pyo3(get)]
    pub total_distance: f64,
    #[pyo3(get)]
    pub steps: Vec<PathStep>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Path {
    fn __repr__(&self) -> String {
        format!(
            "PathResult(steps={}, total_distance={:.4})",
            self.steps.len(),
            self.total_distance
        )
    }

    fn __eq__(&self, other: &Path) -> bool {
        self == other
    }

    fn to_json(&self) -> PyResult<String> {
        let steps: Vec<_> = self
            .steps
            .iter()
            .map(|s| {
                serde_json::json!({
                    "id": s.id,
                    "category": s.category,
                    "cumulative_distance": s.cumulative_distance,
                })
            })
            .collect();
        serde_json::to_string(&serde_json::json!({
            "total_distance": self.total_distance,
            "steps": steps,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let steps = v["steps"]
            .as_array()
            .ok_or_else(|| PyValueError::new_err("missing 'steps' array"))?
            .iter()
            .map(|s| PathStep {
                id: s["id"].as_str().unwrap_or("").to_string(),
                category: s["category"].as_str().unwrap_or("").to_string(),
                cumulative_distance: s["cumulative_distance"].as_f64().unwrap_or(0.0),
                hop_distance: s["hop_distance"].as_f64().unwrap_or(0.0),
                bridge_strength: s["bridge_strength"].as_f64(),
            })
            .collect();
        Ok(Self {
            total_distance: v["total_distance"].as_f64().unwrap_or(0.0),
            steps,
        })
    }
}

impl From<&PathResult> for Path {
    fn from(p: &PathResult) -> Self {
        Self {
            total_distance: p.total_distance,
            steps: p
                .steps
                .iter()
                .map(|s| PathStep {
                    id: s.id.clone(),
                    category: s.category.clone(),
                    cumulative_distance: s.cumulative_distance,
                    hop_distance: s.hop_distance,
                    bridge_strength: s.bridge_strength,
                })
                .collect(),
        }
    }
}

// ── Glob ───────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "GlobInfo", frozen, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct Glob {
    #[pyo3(get)]
    pub id: usize,
    #[pyo3(get)]
    pub centroid: [f64; 3],
    #[pyo3(get)]
    pub member_count: usize,
    #[pyo3(get)]
    pub radius: f64,
    #[pyo3(get)]
    pub top_categories: Vec<(String, usize)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Glob {
    fn __repr__(&self) -> String {
        format!(
            "GlobInfo(id={}, members={}, radius={:.4})",
            self.id, self.member_count, self.radius
        )
    }

    fn __eq__(&self, other: &Glob) -> bool {
        self == other
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "id": self.id,
            "centroid": self.centroid,
            "member_count": self.member_count,
            "radius": self.radius,
            "top_categories": self.top_categories,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let top_categories = v["top_categories"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|pair| {
                let arr = pair.as_array()?;
                Some((
                    arr.first()?.as_str()?.to_string(),
                    arr.get(1)?.as_u64()? as usize,
                ))
            })
            .collect();
        Ok(Self {
            id: v["id"].as_u64().unwrap_or(0) as usize,
            centroid: parse_f64_3(&v["centroid"]),
            member_count: v["member_count"].as_u64().unwrap_or(0) as usize,
            radius: v["radius"].as_f64().unwrap_or(0.0),
            top_categories,
        })
    }
}

impl From<&GlobSummary> for Glob {
    fn from(g: &GlobSummary) -> Self {
        Self {
            id: g.id,
            centroid: g.centroid,
            member_count: g.member_count,
            radius: g.radius,
            top_categories: g.top_categories.clone(),
        }
    }
}

// ── Manifold ───────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "ManifoldInfo", frozen, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct Manifold {
    #[pyo3(get)]
    pub centroid: [f64; 3],
    #[pyo3(get)]
    pub normal: [f64; 3],
    #[pyo3(get)]
    pub variance_ratio: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl Manifold {
    fn __repr__(&self) -> String {
        format!("ManifoldInfo(variance_ratio={:.4})", self.variance_ratio)
    }

    fn __eq__(&self, other: &Manifold) -> bool {
        self == other
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&serde_json::json!({
            "centroid": self.centroid,
            "normal": self.normal,
            "variance_ratio": self.variance_ratio,
        }))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            centroid: parse_f64_3(&v["centroid"]),
            normal: parse_f64_3(&v["normal"]),
            variance_ratio: v["variance_ratio"].as_f64().unwrap_or(0.0),
        })
    }
}

impl From<&ManifoldResult> for Manifold {
    fn from(m: &ManifoldResult) -> Self {
        Self {
            centroid: m.centroid,
            normal: m.normal,
            variance_ratio: m.variance_ratio,
        }
    }
}

// ── CategorySummary ───────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "CategorySummaryInfo", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCategorySummary {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub member_count: usize,
    #[pyo3(get)]
    pub centroid_theta: f64,
    #[pyo3(get)]
    pub centroid_phi: f64,
    #[pyo3(get)]
    pub angular_spread: f64,
    #[pyo3(get)]
    pub cohesion: f64,
    /// Mean territorial-adjusted bridge strength across outgoing edges.
    /// 0.0 if this category has no bridged neighbors.
    #[pyo3(get)]
    pub bridge_quality: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCategorySummary {
    fn __repr__(&self) -> String {
        format!(
            "CategorySummaryInfo(name={:?}, members={}, cohesion={:.4}, bridge_quality={:.4})",
            self.name, self.member_count, self.cohesion, self.bridge_quality
        )
    }
}

impl From<&CategorySummary> for PyCategorySummary {
    fn from(s: &CategorySummary) -> Self {
        Self {
            name: s.name.clone(),
            member_count: s.member_count,
            centroid_theta: s.centroid_position.theta,
            centroid_phi: s.centroid_position.phi,
            angular_spread: s.angular_spread,
            cohesion: s.cohesion,
            bridge_quality: s.bridge_quality,
        }
    }
}

// ── BridgeItem ────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "BridgeItemInfo", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyBridgeItem {
    #[pyo3(get)]
    pub item_index: usize,
    #[pyo3(get)]
    pub source_category: usize,
    #[pyo3(get)]
    pub target_category: usize,
    #[pyo3(get)]
    pub affinity_to_source: f64,
    #[pyo3(get)]
    pub affinity_to_target: f64,
    #[pyo3(get)]
    pub bridge_strength: f64,
    /// One of "Genuine", "OverlapArtifact", or "Weak".
    #[pyo3(get)]
    pub classification: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBridgeItem {
    fn __repr__(&self) -> String {
        format!(
            "BridgeItemInfo(item={}, src={}, tgt={}, strength={:.4}, class={})",
            self.item_index,
            self.source_category,
            self.target_category,
            self.bridge_strength,
            self.classification
        )
    }
}

impl From<&BridgeItem> for PyBridgeItem {
    fn from(b: &BridgeItem) -> Self {
        Self {
            item_index: b.item_index,
            source_category: b.source_category,
            target_category: b.target_category,
            affinity_to_source: b.affinity_to_source,
            affinity_to_target: b.affinity_to_target,
            bridge_strength: b.bridge_strength,
            classification: classification_name(b.classification).to_string(),
        }
    }
}

// ── CategoryPathStep ──────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "CategoryPathStep", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCategoryPathStep {
    #[pyo3(get)]
    pub category_index: usize,
    #[pyo3(get)]
    pub category_name: String,
    #[pyo3(get)]
    pub cumulative_distance: f64,
    #[pyo3(get)]
    pub bridges_to_next: Vec<PyBridgeItem>,
    /// Spatial confidence of this hop: bridge_strength × territorial_factor.
    /// 0.0 for the last step.
    #[pyo3(get)]
    pub hop_confidence: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCategoryPathStep {
    fn __repr__(&self) -> String {
        format!(
            "CategoryPathStep(category={:?}, distance={:.4}, bridges={}, hop_conf={:.4})",
            self.category_name,
            self.cumulative_distance,
            self.bridges_to_next.len(),
            self.hop_confidence
        )
    }
}

impl From<&CategoryPathStep> for PyCategoryPathStep {
    fn from(s: &CategoryPathStep) -> Self {
        Self {
            category_index: s.category_index,
            category_name: s.category_name.clone(),
            cumulative_distance: s.cumulative_distance,
            bridges_to_next: s.bridges_to_next.iter().map(PyBridgeItem::from).collect(),
            hop_confidence: s.hop_confidence,
        }
    }
}

// ── CategoryPath ──────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "CategoryPathResult", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyCategoryPath {
    #[pyo3(get)]
    pub total_distance: f64,
    #[pyo3(get)]
    pub steps: Vec<PyCategoryPathStep>,
    /// Product of all hop confidences along the path. Low values indicate
    /// the path routes through shaky connections.
    #[pyo3(get)]
    pub path_confidence: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCategoryPath {
    fn __repr__(&self) -> String {
        format!(
            "CategoryPathResult(steps={}, total_distance={:.4}, path_conf={:.4})",
            self.steps.len(),
            self.total_distance,
            self.path_confidence
        )
    }
}

impl From<&CategoryPath> for PyCategoryPath {
    fn from(p: &CategoryPath) -> Self {
        Self {
            total_distance: p.total_distance,
            steps: p.steps.iter().map(PyCategoryPathStep::from).collect(),
            path_confidence: p.path_confidence,
        }
    }
}

// ── DrillDown ─────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "DrillDownHit", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyDrillDown {
    #[pyo3(get)]
    pub item_index: usize,
    #[pyo3(get)]
    pub distance: f64,
    #[pyo3(get)]
    pub used_inner_sphere: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDrillDown {
    fn __repr__(&self) -> String {
        format!(
            "DrillDownHit(item={}, distance={:.4}, inner={})",
            self.item_index, self.distance, self.used_inner_sphere
        )
    }
}

impl From<&DrillDownResult> for PyDrillDown {
    fn from(r: &DrillDownResult) -> Self {
        Self {
            item_index: r.item_index,
            distance: r.distance,
            used_inner_sphere: r.used_inner_sphere,
        }
    }
}

// ── InnerSphereReport ─────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "InnerSphereInfo", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyInnerSphereReport {
    #[pyo3(get)]
    pub category_name: String,
    #[pyo3(get)]
    pub category_index: usize,
    #[pyo3(get)]
    pub member_count: usize,
    #[pyo3(get)]
    pub projection_type: String,
    #[pyo3(get)]
    pub inner_evr: f64,
    #[pyo3(get)]
    pub global_subset_evr: f64,
    #[pyo3(get)]
    pub evr_improvement: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInnerSphereReport {
    fn __repr__(&self) -> String {
        format!(
            "InnerSphereInfo(category={:?}, members={}, evr={:.4}, improvement={:.4})",
            self.category_name, self.member_count, self.inner_evr, self.evr_improvement
        )
    }
}

impl From<&InnerSphereReport> for PyInnerSphereReport {
    fn from(r: &InnerSphereReport) -> Self {
        Self {
            category_name: r.category_name.clone(),
            category_index: r.category_index,
            member_count: r.member_count,
            projection_type: r.projection_type.to_string(),
            inner_evr: r.inner_evr,
            global_subset_evr: r.global_subset_evr,
            evr_improvement: r.evr_improvement,
        }
    }
}

// ── DomainGroup ───────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "DomainGroupInfo", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyDomainGroup {
    /// Indices of member categories in the category layer.
    #[pyo3(get)]
    pub member_categories: Vec<usize>,
    #[pyo3(get)]
    pub category_names: Vec<String>,
    #[pyo3(get)]
    pub centroid_theta: f64,
    #[pyo3(get)]
    pub centroid_phi: f64,
    #[pyo3(get)]
    pub angular_spread: f64,
    #[pyo3(get)]
    pub cohesion: f64,
    #[pyo3(get)]
    pub total_items: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDomainGroup {
    fn __repr__(&self) -> String {
        format!(
            "DomainGroupInfo(categories={}, items={}, cohesion={:.4})",
            self.category_names.len(),
            self.total_items,
            self.cohesion
        )
    }
}

impl From<&DomainGroup> for PyDomainGroup {
    fn from(g: &DomainGroup) -> Self {
        Self {
            member_categories: g.member_categories.clone(),
            category_names: g.category_names.clone(),
            centroid_theta: g.centroid.theta,
            centroid_phi: g.centroid.phi,
            angular_spread: g.angular_spread,
            cohesion: g.cohesion,
            total_items: g.total_items,
        }
    }
}

// ── ProjectionWarning ────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(name = "ProjectionWarningInfo", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyProjectionWarning {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub evr: f64,
    /// "Info", "Warning", or "Critical".
    #[pyo3(get)]
    pub severity: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProjectionWarning {
    fn __repr__(&self) -> String {
        format!(
            "ProjectionWarningInfo(severity={}, evr={:.4}, msg={:?})",
            self.severity, self.evr, self.message
        )
    }
}

impl From<&ProjectionWarning> for PyProjectionWarning {
    fn from(w: &ProjectionWarning) -> Self {
        Self {
            message: w.message.clone(),
            evr: w.evr,
            severity: severity_name(w.severity).to_string(),
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn parse_f64_3(v: &serde_json::Value) -> [f64; 3] {
    let arr = v.as_array();
    match arr {
        Some(a) if a.len() >= 3 => [
            a[0].as_f64().unwrap_or(0.0),
            a[1].as_f64().unwrap_or(0.0),
            a[2].as_f64().unwrap_or(0.0),
        ],
        _ => [0.0; 3],
    }
}
