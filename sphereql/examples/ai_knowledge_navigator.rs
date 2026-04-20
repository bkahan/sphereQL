//! AI Knowledge Navigator — Category Enrichment + Spatial Analysis Demo
//!
//! Demonstrates SphereQL's Category Enrichment Layer and all 7 spherical
//! spatial analysis research areas. The corpus simulates an AI's knowledge
//! across 23 academic domains with deliberately placed "bridge concepts".
//!
//! Analyses 1–7: Category enrichment (cohesion, adjacency, bridges, paths,
//!   globs, drill-down, reasoning chains)
//! Analyses §1–§7: Spherical spatial queries (antipodal, coverage, geodesic
//!   sweeps, Voronoi, overlap, curvature, lunes)
//!
//! Run with:
//!   cargo run --example ai_knowledge_navigator --features embed

use sphereql::embed::{
    PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
    NavigatorConfig, run_full_analysis, category_geodesic_sweep, category_path_deviation,
    gap_confidence,
};
use sphereql::core::SphericalPoint;

// ── Semantic axes ─────────────────────────────────────────────────────────

const DIM: usize = 32;
const ENERGY: usize = 0;
const FORCE: usize = 1;
const MATH: usize = 2;
const QUANTUM: usize = 3;
const SPACE: usize = 4;
const LIFE: usize = 5;
const EVOLUTION: usize = 6;
const CHEMISTRY: usize = 7;
const NATURE: usize = 8;
const GENETICS: usize = 9;
const COMPUTATION: usize = 10;
const LOGIC: usize = 11;
const INFORMATION: usize = 12;
const SYSTEMS: usize = 13;
const ETHICS: usize = 14;
const MIND: usize = 15;
const METAPHYSICS: usize = 16;
const LANGUAGE: usize = 17;
const MARKETS: usize = 18;
const OPTIMIZATION: usize = 19;
const BEHAVIOR: usize = 20;
const SOUND: usize = 21;
const EMOTION: usize = 22;
const PATTERN: usize = 23;
const PERFORMANCE: usize = 24;
const DIAGNOSTICS: usize = 25;
const STATISTICS: usize = 26;
const COGNITION: usize = 27;
const STRUCTURE: usize = 28;
const ENTROPY: usize = 29;
const WAVE: usize = 30;
const NETWORK: usize = 31;

fn embed(features: &[(usize, f64)], seed: u64) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for &(axis, val) in features {
        v[axis] = val;
    }
    let mut s = seed;
    for x in &mut v {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
    }
    v
}

struct Concept {
    label: &'static str,
    category: &'static str,
    features: Vec<(usize, f64)>,
}

// The build_corpus() function is unchanged — it returns the full corpus.
// Omitted here for brevity in the commit message; see the file for the
// complete 330-concept corpus across 23 categories.
