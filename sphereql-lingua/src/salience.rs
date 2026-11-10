use crate::concept::{Concept, normalize_text};
use std::collections::HashMap;

/// Computes per-concept salience and maps it to a radial value r ∈ [r_min, r_max].
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

    /// Construct with a custom radial range. Returns `Err` if `r_min >= r_max`
    /// or if either bound is outside `[0.0, 1.0]`.
    pub fn with_range(r_min: f64, r_max: f64) -> Result<Self, String> {
        if r_min >= r_max {
            return Err(format!("r_min ({r_min}) must be less than r_max ({r_max})"));
        }
        if !(0.0..=1.0).contains(&r_min) || !(0.0..=1.0).contains(&r_max) {
            return Err(format!(
                "r_min ({r_min}) and r_max ({r_max}) must both be in [0.0, 1.0]"
            ));
        }
        Ok(Self {
            r_min,
            r_max,
            commonality: Self::default_commonality(),
        })
    }
    pub fn score_all(&self, concepts: &mut [Concept], text: &str) {
        if concepts.is_empty() {
            return;
        }
        // `Concept::positions` index the normalized text, so the signals
        // that use them read that text. The discourse signal counts
        // paragraphs, which normalizing collapses, so it keeps the
        // lowercased original.
        let normalized = normalize_text(text);
        let lowered = text.to_lowercase();
        let raw: Vec<f64> = concepts
            .iter()
            .map(|c| self.raw_score(c, concepts, &normalized, &lowered))
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
    fn raw_score(
        &self,
        concept: &Concept,
        all: &[Concept],
        normalized: &str,
        lowered: &str,
    ) -> f64 {
        let w = [0.25, 0.20, 0.25, 0.20, 0.10];
        let s = [
            self.frequency_signal(concept),
            self.position_signal(concept, normalized),
            self.idf_signal(concept),
            self.centrality_signal(concept, all, normalized),
            self.discourse_signal(concept, lowered),
        ];
        w.iter().zip(s.iter()).map(|(wi, si)| wi * si).sum()
    }
    fn frequency_signal(&self, c: &Concept) -> f64 {
        (1.0 + c.frequency as f64).ln() / (1.0 + 10.0_f64).ln()
    }
    fn position_signal(&self, c: &Concept, normalized: &str) -> f64 {
        let tl = normalized.len().max(1) as f64;
        let fp = c
            .positions
            .first()
            .copied()
            .unwrap_or(normalized.len() / 2)
            .min(normalized.len());
        let ps = 1.0 - (fp as f64 / tl);
        let mut em = 0.0_f64;
        if normalized.contains('?') {
            em = em.max(0.2);
        }
        if normalized.contains(':') {
            em = em.max(0.15);
        }
        (ps * 0.7 + em + 0.1).min(1.0)
    }
    fn idf_signal(&self, c: &Concept) -> f64 {
        let cm = self.commonality.get(&c.normalized).copied().unwrap_or(0.3);
        ((1.0 / (cm + 0.01)).ln() / 4.6).min(1.0)
    }
    fn centrality_signal(&self, c: &Concept, all: &[Concept], normalized: &str) -> f64 {
        let w = 200;
        let mut co = 0u32;
        for &pos in &c.positions {
            let wt = window(
                normalized,
                pos.saturating_sub(w),
                pos.saturating_add(c.normalized.len() + w),
            );
            for o in all {
                if o.normalized != c.normalized && wt.contains(&o.normalized) {
                    co += 1;
                    break;
                }
            }
        }
        co as f64 / (all.len() as u32).saturating_sub(1).max(1) as f64
    }
    fn discourse_signal(&self, c: &Concept, lo: &str) -> f64 {
        let t = &c.normalized;
        let mut sc = 0.3_f64;
        for m in &[
            "end goal",
            "goal:",
            "the point",
            "the question",
            "the method",
        ] {
            if let Some(mp) = lo.find(m)
                && window(lo, mp.saturating_sub(100), mp + m.len() + 100).contains(t)
            {
                sc += 0.3;
                break;
            }
        }
        for m in &["i think", "i want", "i suspect", "we need"] {
            if let Some(mp) = lo.find(m)
                && window(lo, mp, mp + m.len() + 150).contains(t)
            {
                sc += 0.15;
                break;
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

/// `text[start..end]`, with both ends clamped to the text and moved outward
/// to the nearest character boundary. A fixed-width window around a match
/// lands inside a multi-byte character as soon as the text isn't ASCII.
fn window(text: &str, start: usize, end: usize) -> &str {
    let mut s = start.min(text.len());
    while !text.is_char_boundary(s) {
        s -= 1;
    }
    let mut e = end.clamp(s, text.len());
    while !text.is_char_boundary(e) {
        e += 1;
    }
    &text[s..e]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concept::{ConceptExtractor, RegexExtractor};

    fn scored(text: &str) -> Vec<Concept> {
        let mut concepts = RegexExtractor::new().extract(text);
        SalienceScorer::new().score_all(&mut concepts, text);
        concepts
    }

    #[test]
    fn window_widens_to_character_boundaries() {
        let text = "aé—z"; // 1 + 2 + 3 + 1 bytes
        assert_eq!(window(text, 2, 4), "é—");
        assert_eq!(window(text, 0, 2), "aé");
        assert_eq!(window(text, 5, 99), "—z");
        assert_eq!(window(text, 99, 120), "");
        assert_eq!(window(text, 4, 1), "");
    }

    #[test]
    fn non_ascii_text_is_scored() {
        // Every window edge falls among multi-byte characters.
        let filler = "Über die Sprache — naïve café résumé ‘quoted’ “text” … ".repeat(12);
        let text = format!(
            "{filler}the sphere and the graph meet language. {filler}\n\n\
             I think the end goal: language again, sphere again. {filler}"
        );
        let concepts = scored(&text);
        assert!(concepts.iter().any(|c| c.normalized == "sphere"));
        for c in &concepts {
            assert!((0.0..=1.0).contains(&c.salience_score), "{c:?}");
        }
    }

    #[test]
    fn positions_past_the_lowercased_text_are_scored() {
        // "θ" is two bytes and normalizes to the five of "theta", so the
        // normalized text is longer than the original and late offsets
        // lie past the original's end.
        let text = format!("{} sphere graph", "θ ".repeat(200));
        let concepts = scored(&text);
        let sphere = concepts.iter().find(|c| c.normalized == "sphere").unwrap();
        assert!(sphere.positions[0] > text.len());
        assert!(sphere.salience_score.is_finite());
    }

    #[test]
    fn co_occurrence_is_read_at_the_recorded_positions() {
        // Each "θ" grows by three bytes when normalized. Read against the
        // original text, a window at sphere's offset lands 900 bytes late,
        // in the filler past both words.
        let text = format!("{} sphere graph. {}", "θ ".repeat(300), "x ".repeat(600));
        let concepts = scored(&text);
        let normalized = normalize_text(&text);
        let scorer = SalienceScorer::new();
        let sphere = concepts.iter().find(|c| c.normalized == "sphere").unwrap();
        assert!(scorer.centrality_signal(sphere, &concepts, &normalized) > 0.0);
    }
}
