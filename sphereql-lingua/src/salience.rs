use crate::concept::Concept;
use std::collections::HashMap;

pub struct SalienceScorer {
    pub r_min: f64,
    pub r_max: f64,
    commonality: HashMap<String, f64>,
}
impl Default for SalienceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl SalienceScorer {
    pub fn new() -> Self {
        Self {
            r_min: 0.15,
            r_max: 1.0,
            commonality: Self::default_commonality(),
        }
    }
    pub fn score_all(&self, concepts: &mut [Concept], text: &str) {
        if concepts.is_empty() {
            return;
        }
        let raw: Vec<f64> = concepts
            .iter()
            .map(|c| self.raw_score(c, concepts, text))
            .collect();
        let mn = raw.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let rng = mx - mn;
        for (c, &r) in concepts.iter_mut().zip(raw.iter()) {
            c.salience_score = if rng > 0.0 { (r - mn) / rng } else { 0.5 };
        }
    }
    pub fn salience_to_r(&self, salience: f64) -> f64 {
        self.r_min + salience * (self.r_max - self.r_min)
    }
    fn raw_score(&self, concept: &Concept, all: &[Concept], text: &str) -> f64 {
        let w = [0.25, 0.20, 0.25, 0.20, 0.10];
        let s = [
            self.frequency_signal(concept),
            self.position_signal(concept, text),
            self.idf_signal(concept),
            self.centrality_signal(concept, all, text),
            self.discourse_signal(concept, text),
        ];
        w.iter().zip(s.iter()).map(|(wi, si)| wi * si).sum()
    }
    fn frequency_signal(&self, c: &Concept) -> f64 {
        (1.0 + c.frequency as f64).ln() / (1.0 + 10.0_f64).ln()
    }
    fn position_signal(&self, c: &Concept, text: &str) -> f64 {
        let tl = text.len().max(1) as f64;
        let fp = c.positions.first().copied().unwrap_or(text.len() / 2);
        let ps = 1.0 - (fp as f64 / tl);
        let lo = text.to_lowercase();
        let mut em = 0.0_f64;
        if lo.contains('?') {
            em = em.max(0.2);
        }
        if lo.contains(':') {
            em = em.max(0.15);
        }
        (ps * 0.7 + em + 0.1).min(1.0)
    }
    fn idf_signal(&self, c: &Concept) -> f64 {
        let cm = self.commonality.get(&c.normalized).copied().unwrap_or(0.3);
        ((1.0 / (cm + 0.01)).ln() / 4.6).min(1.0)
    }
    fn centrality_signal(&self, c: &Concept, all: &[Concept], text: &str) -> f64 {
        let lo = text.to_lowercase();
        let w = 200;
        let mut co = 0u32;
        for pos in &c.positions {
            let s = pos.saturating_sub(w);
            let e = (pos + c.normalized.len() + w).min(lo.len());
            let wt = &lo[s..e];
            for o in all {
                if o.normalized != c.normalized && wt.contains(&o.normalized) {
                    co += 1;
                    break;
                }
            }
        }
        co as f64 / (all.len() as u32).saturating_sub(1).max(1) as f64
    }
    fn discourse_signal(&self, c: &Concept, text: &str) -> f64 {
        let lo = text.to_lowercase();
        let t = &c.normalized;
        let mut sc = 0.3_f64;
        for m in &[
            "end goal",
            "goal:",
            "the point",
            "the question",
            "the method",
        ] {
            if let Some(mp) = lo.find(m) {
                let s = mp.saturating_sub(100);
                let e = (mp + m.len() + 100).min(lo.len());
                if lo[s..e].contains(t) {
                    sc += 0.3;
                    break;
                }
            }
        }
        for m in &["i think", "i want", "i suspect", "we need"] {
            if let Some(mp) = lo.find(m) {
                let e = (mp + m.len() + 150).min(lo.len());
                if lo[mp..e].contains(t) {
                    sc += 0.15;
                    break;
                }
            }
        }
        let pc = lo.split("\n\n").filter(|p| p.contains(t)).count();
        if pc > 1 {
            sc += 0.1 * (pc as f64).min(3.0);
        }
        sc.min(1.0)
    }
    fn default_commonality() -> HashMap<String, f64> {
        [
            ("language", 0.75),
            ("system", 0.80),
            ("meaning", 0.70),
            ("domain", 0.65),
            ("hierarchy", 0.50),
            ("graph", 0.55),
            ("conversation", 0.70),
            ("english", 0.65),
            ("mathematics", 0.55),
            ("algorithm", 0.45),
            ("sphere", 0.50),
            ("projection", 0.50),
            ("representation", 0.55),
            ("globe", 0.60),
            ("theta", 0.35),
            ("phi", 0.35),
            ("radius", 0.40),
            ("abstraction", 0.40),
            ("salience", 0.25),
            ("llm", 0.30),
            ("sphereql", 0.05),
            ("sphere of spheres", 0.05),
            ("sphereql space", 0.05),
            ("sphereql units", 0.05),
            ("sphereql query", 0.05),
            ("semantic neighborhood", 0.10),
            ("ontological hierarchy", 0.08),
            ("epistemic weight", 0.05),
            ("abstraction level", 0.15),
            ("domain angle", 0.08),
            ("system of components", 0.25),
            ("oncology", 0.20),
            ("cardiology", 0.20),
            ("disease", 0.50),
            ("pancreatic adenocarcinoma", 0.05),
            ("stage-3 pancreatic adenocarcinoma", 0.02),
            ("contract law", 0.15),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect()
    }
}
