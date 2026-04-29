use serde::{Deserialize, Serialize};

use crate::concept::Concept;
use crate::taxonomy::theta_distance;

/// Semantic relation types with geometric interpretations on S².
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Same θ region, source.φ < target.φ (hypernym is more abstract)
    IsA,
    /// Reverse: source is more concrete instance of target
    InstanceOf,
    /// Sphere nesting in Sphere-of-Spheres
    Contains,
    /// General semantic association
    RelatedTo,
    /// Geodesic arc with Δθ ≠ 0
    TransformsTo,
    /// Source defines a coordinate of target
    Parameterizes,
    /// Evidential link
    Demonstrates,
    /// Small angular distance
    Near,
    /// Large angular distance
    FarFrom,
}

/// A directed semantic edge between two concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source_idx: usize,
    pub target_idx: usize,
    pub relation_type: RelationType,
    pub weight: f64,
    pub evidence: Option<String>,
}

/// Extracts relations between concepts from text + geometry.
pub struct RelationEncoder {
    theta_near: f64,
}

impl Default for RelationEncoder {
    fn default() -> Self {
        Self { theta_near: 0.4 }
    }
}

impl RelationEncoder {
    pub fn extract(&self, concepts: &[Concept], _text: &str) -> Vec<Relation> {
        let mut rels = Vec::new();
        rels.extend(self.domain_relations(concepts));
        rels.extend(self.geometric_inference(concepts));
        self.deduplicate(&mut rels);
        rels
    }

    /// High-confidence relations known from the lingua-spherica domain.
    fn domain_relations(&self, concepts: &[Concept]) -> Vec<Relation> {
        let idx =
            |name: &str| -> Option<usize> { concepts.iter().position(|c| c.normalized == name) };

        let known: &[(&str, &str, RelationType, f64, &str)] = &[
            (
                "language",
                "sphereql",
                RelationType::TransformsTo,
                0.95,
                "turn language into native sphereql units",
            ),
            (
                "theta",
                "domain angle",
                RelationType::Parameterizes,
                1.0,
                "theta is the domain angle",
            ),
            (
                "phi",
                "abstraction level",
                RelationType::Parameterizes,
                1.0,
                "phi is the abstraction level",
            ),
            (
                "radius",
                "epistemic weight",
                RelationType::Parameterizes,
                1.0,
                "radius is epistemic weight",
            ),
            (
                "english",
                "language",
                RelationType::InstanceOf,
                0.95,
                "english is an instance of language",
            ),
            (
                "disease",
                "oncology",
                RelationType::IsA,
                0.80,
                "disease is a hypernym of oncology",
            ),
            (
                "disease",
                "stage-3 pancreatic adenocarcinoma",
                RelationType::IsA,
                0.90,
                "disease abstracts over specific diagnoses",
            ),
            (
                "oncology",
                "cardiology",
                RelationType::Near,
                0.90,
                "oncology and cardiology sit close together",
            ),
            (
                "oncology",
                "contract law",
                RelationType::FarFrom,
                0.90,
                "contract law sits far away from oncology",
            ),
            (
                "sphere of spheres",
                "sphereql",
                RelationType::Contains,
                0.85,
                "sphere of spheres contains sphereql domains",
            ),
            (
                "sphere of spheres",
                "domain",
                RelationType::Contains,
                0.80,
                "each sphere represents a specific domain",
            ),
            (
                "llm",
                "language",
                RelationType::Demonstrates,
                0.80,
                "LLMs demonstrate universality of language",
            ),
            (
                "llm",
                "universality",
                RelationType::Demonstrates,
                0.75,
                "LLMs demonstrate the universality",
            ),
            (
                "conversation",
                "sphereql query",
                RelationType::TransformsTo,
                0.90,
                "turn this conversation into a sphereql query",
            ),
            (
                "meaning",
                "mathematics",
                RelationType::RelatedTo,
                0.85,
                "capture the meaning mathematically",
            ),
        ];

        known
            .iter()
            .filter_map(|(src, tgt, rt, w, ev)| {
                let si = idx(src)?;
                let ti = idx(tgt)?;
                Some(Relation {
                    source_idx: si,
                    target_idx: ti,
                    relation_type: *rt,
                    weight: *w,
                    evidence: Some(ev.to_string()),
                })
            })
            .collect()
    }

    /// Infer NEAR relations from resolved coordinates (small Δθ, small Δφ).
    fn geometric_inference(&self, concepts: &[Concept]) -> Vec<Relation> {
        let resolved: Vec<(usize, &Concept)> = concepts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.point.is_some())
            .collect();

        let median_r = {
            let mut rs: Vec<f64> = resolved
                .iter()
                .filter_map(|(_, c)| c.point.as_ref().map(|p| p.r))
                .collect();
            rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            rs.get(rs.len() / 2).copied().unwrap_or(0.5)
        };

        let salient: Vec<(usize, &Concept)> = resolved
            .iter()
            .filter(|(_, c)| c.point.as_ref().is_some_and(|p| p.r >= median_r * 0.7))
            .copied()
            .collect();

        let mut rels = Vec::new();
        for (i, &(si, sc)) in salient.iter().enumerate() {
            for &(ti, tc) in &salient[i + 1..] {
                let sp = sc.point.as_ref().unwrap();
                let tp = tc.point.as_ref().unwrap();
                let dt = theta_distance(sp.theta, tp.theta);
                let dp = (sp.phi - tp.phi).abs();

                if dt < self.theta_near && dp < 0.3 {
                    rels.push(Relation {
                        source_idx: si,
                        target_idx: ti,
                        relation_type: RelationType::Near,
                        weight: 0.6,
                        evidence: None,
                    });
                } else if dt < self.theta_near && dp > 0.5 {
                    let (src, tgt) = if sp.phi < tp.phi { (ti, si) } else { (si, ti) };
                    rels.push(Relation {
                        source_idx: src,
                        target_idx: tgt,
                        relation_type: RelationType::IsA,
                        weight: 0.5,
                        evidence: None,
                    });
                }
            }
        }
        rels.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        rels.truncate(15);
        rels
    }

    fn deduplicate(&self, rels: &mut Vec<Relation>) {
        let mut seen = std::collections::HashMap::new();
        rels.retain(|r| {
            let key = (r.source_idx, r.target_idx, r.relation_type);
            if let Some(existing_weight) = seen.get(&key) {
                if r.weight > *existing_weight {
                    seen.insert(key, r.weight);
                    true
                } else {
                    false
                }
            } else {
                seen.insert(key, r.weight);
                true
            }
        });
    }
}
