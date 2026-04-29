use regex::Regex;
use serde::{Deserialize, Serialize};
use sphereql_core::SphericalPoint;
use std::collections::{HashMap, HashSet};

/// A semantic concept extracted from natural language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Original surface form from the text.
    pub text: String,
    /// Canonical lowercase form used for matching.
    pub normalized: String,
    /// Resolved spherical address. `None` until pipeline runs.
    pub point: Option<SphericalPoint>,
    /// Occurrence count in source text.
    pub frequency: u32,
    /// Character offsets of each occurrence.
    pub positions: Vec<usize>,
    /// Hint from vocabulary: which domain does this belong to?
    pub domain_hint: Option<String>,
    /// Raw abstraction score ∈ [0,1] before φ mapping.
    pub abstraction_hint: f64,
    /// Raw salience score ∈ [0,1] before r mapping.
    pub salience_score: f64,
}

impl Concept {
    pub fn new(text: &str, normalized: &str) -> Self {
        Self {
            text: text.to_string(),
            normalized: normalized.to_string(),
            point: None,
            frequency: 1,
            positions: Vec::new(),
            domain_hint: None,
            abstraction_hint: 0.5,
            salience_score: 0.5,
        }
    }
}

impl PartialEq for Concept {
    fn eq(&self, other: &Self) -> bool {
        self.normalized == other.normalized
    }
}
impl Eq for Concept {}

impl std::hash::Hash for Concept {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.normalized.hash(state);
    }
}

/// Trait for concept extraction backends.
pub trait ConceptExtractor: Send + Sync {
    fn extract(&self, text: &str) -> Vec<Concept>;
}

/// Known-term entry in the curated vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabEntry {
    pub domain_hint: Option<String>,
    pub abstraction_hint: f64,
}

/// Default extractor: regex matching against a curated vocabulary,
/// plus statistical detection of repeated content words.
pub struct RegexExtractor {
    /// term → metadata, sorted longest-first for greedy matching.
    vocab: Vec<(String, VocabEntry)>,
    stop_words: HashSet<String>,
}

impl Default for RegexExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl RegexExtractor {
    pub fn new() -> Self {
        let vocab = Self::default_vocab();
        let stop_words = Self::default_stop_words();
        let mut sorted: Vec<_> = vocab.into_iter().collect();
        sorted.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
        Self {
            vocab: sorted,
            stop_words,
        }
    }

    fn default_vocab() -> Vec<(String, VocabEntry)> {
        let entries: &[(&str, Option<&str>, f64)] = &[
            ("sphere of spheres", Some("formal_system"), 0.15),
            ("sphereql space", Some("formal_system"), 0.25),
            ("sphereql units", Some("formal_system"), 0.30),
            ("sphereql query", Some("formal_system"), 0.35),
            ("sphereql", Some("formal_system"), 0.20),
            ("domain angle", Some("mathematics"), 0.40),
            ("abstraction level", Some("epistemology"), 0.20),
            ("epistemic weight", Some("epistemology"), 0.20),
            ("semantic neighborhood", Some("linguistics_ai"), 0.30),
            ("ontological hierarchy", Some("epistemology"), 0.15),
            ("horizontal clustering dimension", Some("mathematics"), 0.35),
            ("system of components", Some("systems_theory"), 0.20),
            ("stage-3 pancreatic adenocarcinoma", Some("medicine"), 0.92),
            ("pancreatic adenocarcinoma", Some("medicine"), 0.85),
            ("contract law", Some("law"), 0.50),
            ("north pole", Some("mathematics"), 0.50),
            ("south pole", Some("mathematics"), 0.50),
            ("radial value", Some("mathematics"), 0.45),
            ("projection", Some("mathematics"), 0.35),
            ("longitude", Some("mathematics"), 0.50),
            ("latitude", Some("mathematics"), 0.50),
            ("globe", Some("mathematics"), 0.55),
            ("sphere", Some("mathematics"), 0.35),
            ("graph", Some("mathematics"), 0.35),
            ("llm", Some("artificial_intelligence"), 0.40),
            ("algorithm", Some("computer_science"), 0.30),
            ("language", Some("linguistics"), 0.12),
            ("english", Some("linguistics"), 0.55),
            ("meaning", Some("linguistics"), 0.15),
            ("conversation", Some("linguistics"), 0.45),
            ("universality", Some("epistemology"), 0.08),
            ("mathematics", Some("mathematics"), 0.10),
            ("salience", Some("epistemology"), 0.22),
            ("hierarchy", Some("epistemology"), 0.18),
            ("domain", Some("epistemology"), 0.20),
            ("representation", Some("epistemology"), 0.18),
            ("oncology", Some("medicine"), 0.50),
            ("cardiology", Some("medicine"), 0.50),
            ("disease", Some("medicine"), 0.22),
            ("theta", Some("mathematics"), 0.42),
            ("phi", Some("mathematics"), 0.42),
            ("radius", Some("mathematics"), 0.42),
        ];
        entries
            .iter()
            .map(|(term, dom, abs)| {
                (
                    term.to_string(),
                    VocabEntry {
                        domain_hint: dom.map(|s| s.to_string()),
                        abstraction_hint: *abs,
                    },
                )
            })
            .collect()
    }

    fn default_stop_words() -> HashSet<String> {
        [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "but",
            "and",
            "or",
            "if",
            "while",
            "about",
            "up",
            "that",
            "this",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "what",
            "which",
            "who",
            "now",
            "hey",
            "right",
            "think",
            "want",
            "come",
            "let",
            "get",
            "sit",
            "live",
            "turn",
            "sits",
            "lives",
            "near",
            "far",
            "away",
            "close",
            "together",
            "toward",
            "specific",
            "native",
            "distinct",
            "working",
            "using",
            "map",
            "mapped",
            "focus",
            "capture",
            "closer",
            "appears",
            "understands",
            "significant",
            "heavily",
            "high",
            "central",
            "concrete",
            "abstract",
            "cluster",
            "float",
            "sink",
            "encodes",
            "encoding",
            "measure",
            "work",
            "method",
            "question",
            "topic",
            "values",
            "demonstrate",
            "interacts",
            "referenced",
            "frequency",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn normalize_text(&self, text: &str) -> String {
        let mut s = text.to_lowercase();
        // Map Greek letters to ASCII names
        s = s.replace('θ', "theta").replace('φ', "phi");
        // Collapse whitespace
        let re = Regex::new(r"\s+").unwrap();
        re.replace_all(&s, " ").trim().to_string()
    }

    fn count_occurrences(term: &str, text: &str) -> (u32, Vec<usize>) {
        let mut count = 0u32;
        let mut positions = Vec::new();
        let mut start = 0;
        while let Some(pos) = text[start..].find(term) {
            let abs_pos = start + pos;
            positions.push(abs_pos);
            count += 1;
            start = abs_pos + term.len();
        }
        (count, positions)
    }
}

impl ConceptExtractor for RegexExtractor {
    fn extract(&self, text: &str) -> Vec<Concept> {
        let normalized = self.normalize_text(text);
        let mut seen: HashSet<String> = HashSet::new();
        let mut concepts = Vec::new();

        // Layer 1: vocab matching (longest first)
        for (term, entry) in &self.vocab {
            if normalized.contains(term.as_str()) && !seen.contains(term) {
                seen.insert(term.clone());
                let (freq, positions) = Self::count_occurrences(term, &normalized);
                let mut c = Concept::new(term, term);
                c.frequency = freq;
                c.positions = positions;
                c.domain_hint = entry.domain_hint.clone();
                c.abstraction_hint = entry.abstraction_hint;
                concepts.push(c);
            }
        }

        // Layer 2: repeated content words (freq >= 2, not already found)
        let mut word_freq: HashMap<String, u32> = HashMap::new();
        for word in normalized.split_whitespace() {
            let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
            if w.len() > 2 && !self.stop_words.contains(w) {
                *word_freq.entry(w.to_string()).or_default() += 1;
            }
        }
        for (word, freq) in &word_freq {
            if *freq >= 2 && !seen.contains(word) {
                let is_sub = seen.iter().any(|s| s.contains(word.as_str()));
                if !is_sub {
                    let (_, positions) = Self::count_occurrences(word, &normalized);
                    let mut c = Concept::new(word, word);
                    c.frequency = *freq;
                    c.positions = positions;
                    concepts.push(c);
                }
            }
        }

        concepts.sort_by_key(|c| std::cmp::Reverse(c.frequency));
        concepts
    }
}
