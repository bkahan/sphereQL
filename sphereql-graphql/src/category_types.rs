//! GraphQL output types for the category-enrichment surface.
//!
//! Mirrors the Rust types in `sphereql_embed::category`, `pipeline`, and
//! `domain_groups`. Each output type implements `From<&RustType>` so
//! resolvers can `.into()` results without bespoke per-resolver
//! conversion code.

use sphereql_embed::category::{
    BridgeClassification, BridgeItem, CategoryPath, CategoryPathStep, CategorySummary,
    DrillDownResult, InnerSphereReport,
};
use sphereql_embed::domain_groups::DomainGroup;
use sphereql_embed::pipeline::{NearestResult, PathResult, PipelinePathStep};

// ── CategorySummary ────────────────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct CategorySummaryOutput {
    pub name: String,
    pub member_count: i32,
    pub centroid_theta: f64,
    pub centroid_phi: f64,
    pub angular_spread: f64,
    pub cohesion: f64,
    pub cap_area: f64,
    pub exclusivity: f64,
    pub voronoi_area: f64,
    pub territorial_efficiency: f64,
    pub bridge_quality: f64,
}

impl From<&CategorySummary> for CategorySummaryOutput {
    fn from(s: &CategorySummary) -> Self {
        Self {
            name: s.name.clone(),
            member_count: s.member_count as i32,
            centroid_theta: s.centroid_position.theta,
            centroid_phi: s.centroid_position.phi,
            angular_spread: s.angular_spread,
            cohesion: s.cohesion,
            cap_area: s.cap_area,
            exclusivity: s.exclusivity,
            voronoi_area: s.voronoi_area,
            territorial_efficiency: s.territorial_efficiency,
            bridge_quality: s.bridge_quality,
        }
    }
}

// ── BridgeItem ─────────────────────────────────────────────────────────

#[derive(async_graphql::Enum, Copy, Clone, Eq, PartialEq, Debug)]
pub enum BridgeClassificationOutput {
    Genuine,
    OverlapArtifact,
    Weak,
}

impl From<BridgeClassification> for BridgeClassificationOutput {
    fn from(c: BridgeClassification) -> Self {
        match c {
            BridgeClassification::Genuine => Self::Genuine,
            BridgeClassification::OverlapArtifact => Self::OverlapArtifact,
            BridgeClassification::Weak => Self::Weak,
        }
    }
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct BridgeItemOutput {
    pub item_index: i32,
    pub source_category: i32,
    pub target_category: i32,
    pub affinity_to_source: f64,
    pub affinity_to_target: f64,
    pub bridge_strength: f64,
    pub classification: BridgeClassificationOutput,
}

impl From<&BridgeItem> for BridgeItemOutput {
    fn from(b: &BridgeItem) -> Self {
        Self {
            item_index: b.item_index as i32,
            source_category: b.source_category as i32,
            target_category: b.target_category as i32,
            affinity_to_source: b.affinity_to_source,
            affinity_to_target: b.affinity_to_target,
            bridge_strength: b.bridge_strength,
            classification: b.classification.into(),
        }
    }
}

// ── CategoryPath ───────────────────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct CategoryPathStepOutput {
    pub category_index: i32,
    pub category_name: String,
    pub cumulative_distance: f64,
    pub hop_confidence: f64,
    pub bridges_to_next: Vec<BridgeItemOutput>,
}

impl From<&CategoryPathStep> for CategoryPathStepOutput {
    fn from(s: &CategoryPathStep) -> Self {
        Self {
            category_index: s.category_index as i32,
            category_name: s.category_name.clone(),
            cumulative_distance: s.cumulative_distance,
            hop_confidence: s.hop_confidence,
            bridges_to_next: s
                .bridges_to_next
                .iter()
                .map(BridgeItemOutput::from)
                .collect(),
        }
    }
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct CategoryPathOutput {
    pub total_distance: f64,
    pub path_confidence: f64,
    pub steps: Vec<CategoryPathStepOutput>,
}

impl From<&CategoryPath> for CategoryPathOutput {
    fn from(p: &CategoryPath) -> Self {
        Self {
            total_distance: p.total_distance,
            path_confidence: p.path_confidence,
            steps: p.steps.iter().map(CategoryPathStepOutput::from).collect(),
        }
    }
}

// ── Item-level concept path ────────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct ConceptPathStepOutput {
    pub id: String,
    pub category: String,
    pub cumulative_distance: f64,
    pub hop_distance: f64,
    pub bridge_strength: Option<f64>,
}

impl From<&PipelinePathStep> for ConceptPathStepOutput {
    fn from(s: &PipelinePathStep) -> Self {
        Self {
            id: s.id.clone(),
            category: s.category.clone(),
            cumulative_distance: s.cumulative_distance,
            hop_distance: s.hop_distance,
            bridge_strength: s.bridge_strength,
        }
    }
}

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct ConceptPathOutput {
    pub total_distance: f64,
    pub steps: Vec<ConceptPathStepOutput>,
}

impl From<&PathResult> for ConceptPathOutput {
    fn from(p: &PathResult) -> Self {
        Self {
            total_distance: p.total_distance,
            steps: p.steps.iter().map(ConceptPathStepOutput::from).collect(),
        }
    }
}

// ── DrillDown ──────────────────────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct DrillDownOutput {
    pub item_index: i32,
    pub distance: f64,
    pub used_inner_sphere: bool,
}

impl From<&DrillDownResult> for DrillDownOutput {
    fn from(r: &DrillDownResult) -> Self {
        Self {
            item_index: r.item_index as i32,
            distance: r.distance,
            used_inner_sphere: r.used_inner_sphere,
        }
    }
}

// ── InnerSphereReport ──────────────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct InnerSphereReportOutput {
    pub category_name: String,
    pub category_index: i32,
    pub member_count: i32,
    pub projection_type: String,
    pub inner_evr: f64,
    pub global_subset_evr: f64,
    pub evr_improvement: f64,
}

impl From<&InnerSphereReport> for InnerSphereReportOutput {
    fn from(r: &InnerSphereReport) -> Self {
        Self {
            category_name: r.category_name.clone(),
            category_index: r.category_index as i32,
            member_count: r.member_count as i32,
            projection_type: r.projection_type.to_string(),
            inner_evr: r.inner_evr,
            global_subset_evr: r.global_subset_evr,
            evr_improvement: r.evr_improvement,
        }
    }
}

// ── CategoryStats (composite) ──────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct CategoryStatsOutput {
    pub summaries: Vec<CategorySummaryOutput>,
    pub inner_sphere_reports: Vec<InnerSphereReportOutput>,
}

// ── DomainGroup ────────────────────────────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct DomainGroupOutput {
    pub member_categories: Vec<i32>,
    pub category_names: Vec<String>,
    pub centroid_theta: f64,
    pub centroid_phi: f64,
    pub angular_spread: f64,
    pub cohesion: f64,
    pub total_items: i32,
}

impl From<&DomainGroup> for DomainGroupOutput {
    fn from(g: &DomainGroup) -> Self {
        Self {
            member_categories: g.member_categories.iter().map(|&i| i as i32).collect(),
            category_names: g.category_names.clone(),
            centroid_theta: g.centroid.theta,
            centroid_phi: g.centroid.phi,
            angular_spread: g.angular_spread,
            cohesion: g.cohesion,
            total_items: g.total_items as i32,
        }
    }
}

// ── Hierarchical / item-level Nearest ──────────────────────────────────

#[derive(async_graphql::SimpleObject, Debug, Clone)]
pub struct CategoryNearestResultOutput {
    pub id: String,
    pub category: String,
    pub distance: f64,
    pub certainty: f64,
    pub intensity: f64,
}

impl From<&NearestResult> for CategoryNearestResultOutput {
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
