use std::collections::HashMap;
use std::f64::consts::PI;

use crate::concept::Concept;

pub struct AbstractionResolver {
    hierarchy: HashMap<String, f64>,
    weights: [f64; 5],
}

impl Default for AbstractionResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractionResolver {
    pub fn new() -> Self {
        Self {
            hierarchy: Self::default_hierarchy(),
            weights: [0.40, 0.20, 0.20, 0.10, 0.10],
        }
    }

    pub fn assign_phi(&self, concept: &Concept, context: &str) -> f64 {
        let s1 = self.hierarchy_signal(concept);
        let s2 = self.morphological_signal(concept);
        let s3 = self.specificity_signal(concept);
        let s4 = self.contextual_signal(concept, context);
        let s5 = concept.abstraction_hint;
        let raw = self.weights[0] * s1
            + self.weights[1] * s2
            + self.weights[2] * s3
            + self.weights[3] * s4
            + self.weights[4] * s5;
        raw.clamp(0.0, 1.0) * PI
    }

    fn hierarchy_signal(&self, concept: &Concept) -> f64 {
        if let Some(&d) = self.hierarchy.get(&concept.normalized) {
            return d;
        }
        for (key, &d) in &self.hierarchy {
            if key.contains(&concept.normalized) || concept.normalized.contains(key) {
                return d;
            }
        }
        0.5
    }

    fn morphological_signal(&self, concept: &Concept) -> f64 {
        let words: Vec<&str> = concept.normalized.split_whitespace().collect();
        let base = match words.len() {
            1 => 0.3,
            2 => 0.5,
            3 => 0.7,
            _ => 0.85,
        };
        let avg_len: f64 =
            words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len().max(1) as f64;
        let length_adj = ((avg_len - 5.0) * 0.02).min(0.1);
        let hyphen = if concept.normalized.contains('-') {
            0.1
        } else {
            0.0
        };
        (base + length_adj + hyphen).clamp(0.0, 1.0)
    }

    fn specificity_signal(&self, concept: &Concept) -> f64 {
        let text = &concept.normalized;
        let mut score = 0.3_f64;
        if text.chars().any(|c| c.is_ascii_digit()) {
            score += 0.4;
        }
        if text.contains('-') {
            score += 0.2;
        }
        for m in &[
            "universal",
            "abstract",
            "principle",
            "concept",
            "theory",
            "general",
            "fundamental",
            "meta",
        ] {
            if text.contains(m) {
                score -= 0.15;
            }
        }
        for m in &[
            "specific",
            "particular",
            "instance",
            "example",
            "stage",
            "phase",
            "version",
            "number",
        ] {
            if text.contains(m) {
                score += 0.15;
            }
        }
        score.clamp(0.0, 1.0)
    }

    fn contextual_signal(&self, concept: &Concept, context: &str) -> f64 {
        if context.is_empty() {
            return 0.5;
        }
        let ctx = context.to_lowercase();
        let term = &concept.normalized;
        let mut score = 0.5_f64;
        for pat in &[
            format!("such as {term}"),
            format!("e.g. {term}"),
            format!("'{term}'"),
        ] {
            if ctx.contains(pat.as_str()) {
                score += 0.15;
                break;
            }
        }
        for pat in &[
            format!("{term} is the"),
            format!("{term} is a"),
            format!("{term} encodes"),
        ] {
            if ctx.contains(pat.as_str()) {
                score -= 0.1;
                break;
            }
        }
        score.clamp(0.0, 1.0)
    }

    fn default_hierarchy() -> HashMap<String, f64> {
        let e: &[(&str, f64)] = &[
            ("universality", 0.05),
            ("meaning", 0.08),
            ("language", 0.08),
            ("mathematics", 0.08),
            ("abstraction", 0.12),
            ("representation", 0.12),
            ("hierarchy", 0.12),
            ("domain", 0.15),
            ("salience", 0.15),
            ("epistemic weight", 0.15),
            ("abstraction level", 0.15),
            ("ontological hierarchy", 0.12),
            ("sphere of spheres", 0.18),
            ("sphereql", 0.22),
            ("semantic neighborhood", 0.22),
            ("system of components", 0.20),
            ("projection", 0.25),
            ("disease", 0.22),
            ("sphere", 0.30),
            ("graph", 0.30),
            ("algorithm", 0.32),
            ("sphereql space", 0.28),
            ("sphereql query", 0.32),
            ("sphereql units", 0.32),
            ("domain angle", 0.35),
            ("llm", 0.38),
            ("theta", 0.42),
            ("phi", 0.42),
            ("radius", 0.42),
            ("conversation", 0.45),
            ("globe", 0.45),
            ("longitude", 0.48),
            ("latitude", 0.48),
            ("oncology", 0.55),
            ("cardiology", 0.55),
            ("contract law", 0.55),
            ("english", 0.55),
            ("pancreatic adenocarcinoma", 0.78),
            ("stage-3 pancreatic adenocarcinoma", 0.92),
        ];
        e.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
}
