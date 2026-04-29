use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::concept::Concept;

/// A fixed reference domain with a known θ position on the atlas sphere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAnchor {
    pub name: String,
    pub theta: f64,
    pub angular_width: f64,
    pub keywords: Vec<String>,
}

/// The atlas sphere: a taxonomy of domain anchors partitioning θ ∈ [0, 2π).
///
/// Semantically related domains are placed adjacent on the circle:
/// Math → Formal → CS → AI → CogSci → Linguistics → Philosophy → ...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainTaxonomy {
    pub anchors: Vec<DomainAnchor>,
    /// keyword → Vec<(anchor_index, weight)>
    #[serde(skip)]
    keyword_index: HashMap<String, Vec<(usize, f64)>>,
}

impl Default for DomainTaxonomy {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainTaxonomy {
    pub fn new() -> Self {
        let anchors = Self::default_anchors();
        let mut tax = Self {
            anchors,
            keyword_index: HashMap::new(),
        };
        tax.rebuild_index();
        tax
    }

    fn rebuild_index(&mut self) {
        self.keyword_index.clear();
        for (i, anchor) in self.anchors.iter().enumerate() {
            for kw in &anchor.keywords {
                self.keyword_index
                    .entry(kw.to_lowercase())
                    .or_default()
                    .push((i, 1.0));
            }
        }
    }

    /// Assign θ to a concept based on domain affinity.
    pub fn assign_theta(&self, concept: &Concept) -> f64 {
        // Phase 1: direct routing from domain hint
        if let Some(ref hint) = concept.domain_hint
            && let Some(anchor) = self.anchors.iter().find(|a| &a.name == hint)
        {
            let perturbation = self.intra_domain_offset(concept, anchor);
            return (anchor.theta + perturbation).rem_euclid(TAU);
        }
        // Phase 2: keyword affinity → circular weighted mean
        let affinities = self.compute_affinities(concept);
        if affinities.is_empty() {
            return std::f64::consts::PI;
        }
        let angles: Vec<f64> = affinities
            .iter()
            .map(|&(idx, _)| self.anchors[idx].theta)
            .collect();
        let weights: Vec<f64> = affinities.iter().map(|&(_, w)| w).collect();
        circular_weighted_mean(&angles, &weights)
    }

    /// Name of the nearest domain to a given θ.
    pub fn domain_name(&self, theta: f64) -> &str {
        self.anchors
            .iter()
            .min_by(|a, b| {
                let da = theta_distance(theta, a.theta);
                let db = theta_distance(theta, b.theta);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.name.as_str())
            .unwrap_or("unknown")
    }

    fn compute_affinities(&self, concept: &Concept) -> Vec<(usize, f64)> {
        let mut scores: HashMap<usize, f64> = HashMap::new();
        for word in concept.normalized.split_whitespace() {
            if let Some(entries) = self.keyword_index.get(word) {
                for &(idx, weight) in entries {
                    *scores.entry(idx).or_default() += weight;
                }
            }
        }
        // substring matches for compound terms
        for (kw, entries) in &self.keyword_index {
            if kw.len() > 3 && concept.normalized.contains(kw.as_str()) {
                for &(idx, weight) in entries {
                    *scores.entry(idx).or_default() += weight * 0.5;
                }
            }
        }
        let total: f64 = scores.values().sum();
        if total > 0.0 {
            scores.iter().map(|(&k, &v)| (k, v / total)).collect()
        } else {
            Vec::new()
        }
    }

    fn intra_domain_offset(&self, concept: &Concept, anchor: &DomainAnchor) -> f64 {
        // Deterministic hash → small offset within domain region
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        concept.normalized.hash(&mut hasher);
        let h = hasher.finish() % 10000;
        let norm = (h as f64 / 10000.0) * 2.0 - 1.0; // ∈ [-1, 1]
        norm * anchor.angular_width / 3.0
    }

    fn default_anchors() -> Vec<DomainAnchor> {
        let raw: &[(&str, f64, &[&str])] = &[
            (
                "mathematics",
                0.00,
                &[
                    "math",
                    "mathematics",
                    "algebra",
                    "topology",
                    "geometry",
                    "calculus",
                    "theorem",
                    "equation",
                    "coordinates",
                    "spherical",
                    "angle",
                    "radius",
                    "projection",
                    "mapping",
                    "vector",
                    "theta",
                    "phi",
                    "manifold",
                    "geodesic",
                    "dimension",
                    "sphere",
                    "globe",
                    "longitude",
                    "latitude",
                    "pole",
                ],
            ),
            (
                "formal_system",
                0.52,
                &[
                    "formal",
                    "query",
                    "specification",
                    "type",
                    "structure",
                    "axiom",
                    "rule",
                    "grammar",
                    "sphereql",
                    "schema",
                    "model",
                    "representation",
                    "encoding",
                    "unit",
                ],
            ),
            (
                "computer_science",
                1.05,
                &[
                    "computer",
                    "algorithm",
                    "code",
                    "program",
                    "software",
                    "data",
                    "database",
                    "computation",
                    "recursive",
                    "binary",
                    "tree",
                    "search",
                    "sort",
                    "complexity",
                ],
            ),
            (
                "artificial_intelligence",
                1.57,
                &[
                    "ai",
                    "llm",
                    "neural",
                    "machine",
                    "learning",
                    "deep",
                    "embedding",
                    "attention",
                    "transformer",
                    "token",
                    "training",
                    "inference",
                    "generative",
                    "semantic",
                ],
            ),
            (
                "cognitive_science",
                2.09,
                &[
                    "cognition",
                    "perception",
                    "reasoning",
                    "understanding",
                    "comprehension",
                    "concept",
                    "category",
                    "abstraction",
                    "knowledge",
                    "belief",
                    "cognitive",
                    "thought",
                ],
            ),
            (
                "linguistics",
                2.62,
                &[
                    "language",
                    "linguistic",
                    "semantic",
                    "syntax",
                    "grammar",
                    "word",
                    "sentence",
                    "meaning",
                    "discourse",
                    "english",
                    "translation",
                    "communication",
                    "text",
                    "conversation",
                ],
            ),
            (
                "epistemology",
                std::f64::consts::PI,
                &[
                    "epistemology",
                    "ontology",
                    "philosophy",
                    "metaphysics",
                    "knowledge",
                    "truth",
                    "belief",
                    "abstraction",
                    "hierarchy",
                    "universality",
                    "salience",
                    "significance",
                    "weight",
                    "epistemic",
                    "ontological",
                    "conceptual",
                    "domain",
                ],
            ),
            (
                "social_sciences",
                3.67,
                &[
                    "society",
                    "social",
                    "psychology",
                    "economics",
                    "political",
                    "culture",
                    "behavior",
                    "institution",
                ],
            ),
            (
                "law",
                4.19,
                &[
                    "law",
                    "legal",
                    "contract",
                    "regulation",
                    "statute",
                    "court",
                    "rights",
                    "liability",
                ],
            ),
            (
                "medicine",
                4.71,
                &[
                    "medicine",
                    "medical",
                    "disease",
                    "diagnosis",
                    "treatment",
                    "oncology",
                    "cardiology",
                    "surgery",
                    "pathology",
                    "patient",
                    "symptom",
                    "therapy",
                    "health",
                    "cancer",
                    "tumor",
                    "adenocarcinoma",
                    "pancreatic",
                    "stage",
                ],
            ),
            (
                "natural_sciences",
                5.24,
                &[
                    "physics",
                    "chemistry",
                    "biology",
                    "quantum",
                    "molecular",
                    "atom",
                    "particle",
                    "energy",
                    "force",
                    "evolution",
                ],
            ),
            (
                "systems_theory",
                5.76,
                &[
                    "system",
                    "component",
                    "engineering",
                    "design",
                    "architecture",
                    "interface",
                    "module",
                    "integration",
                    "pipeline",
                ],
            ),
        ];
        raw.iter()
            .map(|(name, theta, kws)| DomainAnchor {
                name: name.to_string(),
                theta: *theta,
                angular_width: 0.40,
                keywords: kws.iter().map(|s| s.to_string()).collect(),
            })
            .collect()
    }
}

/// Circular weighted mean: MLE for von Mises mean direction.
pub fn circular_weighted_mean(angles: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total < 1e-10 || angles.is_empty() {
        return if angles.is_empty() { 0.0 } else { angles[0] };
    }
    let sin_sum: f64 = angles.iter().zip(weights).map(|(a, w)| w * a.sin()).sum();
    let cos_sum: f64 = angles.iter().zip(weights).map(|(a, w)| w * a.cos()).sum();
    (sin_sum / total).atan2(cos_sum / total).rem_euclid(TAU)
}

/// Shortest arc on S¹.
pub fn theta_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % TAU;
    d.min(TAU - d)
}
