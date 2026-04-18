//! AI Knowledge Navigator — Category Enrichment Demo
//!
//! Demonstrates how sphereQL's Category Enrichment Layer could help an AI
//! model reason about cross-domain connections. The corpus simulates an AI's
//! knowledge across 8 academic domains with deliberately placed "bridge
//! concepts" that span multiple fields.
//!
//! The demo runs 7 analyses:
//!   1. Category landscape (cohesion, spread, relative positions)
//!   2. Inter-category adjacency graph
//!   3. Bridge concept detection
//!   4. Cross-domain concept path traversal
//!   5. Gap detection via glob analysis
//!   6. Inner-sphere drill-down for sub-topic precision
//!   7. Assembled reasoning chain from spatial structure
//!
//! Run with:
//!   cargo run --example ai_knowledge_navigator --features embed

use sphereql::embed::{
    PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};

// ── Semantic axes ─────────────────────────────────────────────────────────
// Each dimension represents an abstract conceptual feature.
// Real transformer models learn these implicitly; we define them by hand
// so the cross-domain relationships are transparent and verifiable.

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

/// Deterministic pseudo-random noise for realism.
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

// ─────────────────────────────────────────────────────────────────────────────
// SphereQL test corpus
// 330 concepts across 23 categories, engineered to stress-test spherical
// embedding: every one of the 32 semantic axes receives meaningful mass,
// and bridge concepts deliberately straddle category boundaries so that
// θ/φ clustering has non-trivial structure to recover.
// ─────────────────────────────────────────────────────────────────────────────
fn build_corpus() -> Vec<Concept> {
    vec![
        // ── Physics (15) ──────────────────────────────────────────────
        Concept { label: "Newtonian mechanics", category: "physics",
            features: vec![(FORCE, 1.0), (ENERGY, 0.7), (MATH, 0.6), (SPACE, 0.5)] },
        Concept { label: "Quantum field theory", category: "physics",
            features: vec![(QUANTUM, 1.0), (ENERGY, 0.8), (MATH, 0.9), (WAVE, 0.6)] },
        Concept { label: "General relativity", category: "physics",
            features: vec![(SPACE, 1.0), (ENERGY, 0.7), (MATH, 0.9), (FORCE, 0.6)] },
        Concept { label: "Thermodynamics", category: "physics",
            features: vec![(ENERGY, 1.0), (ENTROPY, 0.9), (CHEMISTRY, 0.3), (SYSTEMS, 0.4)] },
        Concept { label: "Electromagnetism", category: "physics",
            features: vec![(FORCE, 0.8), (ENERGY, 0.8), (WAVE, 0.9), (MATH, 0.5)] },
        Concept { label: "Statistical mechanics", category: "physics",
            features: vec![(STATISTICS, 0.8), (ENERGY, 0.7), (ENTROPY, 0.8), (MATH, 0.7)] },
        Concept { label: "Optics", category: "physics",
            features: vec![(WAVE, 1.0), (ENERGY, 0.5), (SPACE, 0.4), (PATTERN, 0.3)] },
        Concept { label: "Particle physics", category: "physics",
            features: vec![(QUANTUM, 0.9), (ENERGY, 0.9), (FORCE, 0.7), (MATH, 0.6)] },
        Concept { label: "Cosmology", category: "physics",
            features: vec![(SPACE, 0.9), (ENERGY, 0.6), (MATH, 0.5), (ENTROPY, 0.4)] },
        Concept { label: "Fluid dynamics", category: "physics",
            features: vec![(FORCE, 0.7), (ENERGY, 0.6), (MATH, 0.7), (WAVE, 0.5), (SYSTEMS, 0.3)] },
        Concept { label: "Acoustics", category: "physics",  // BRIDGE: physics <-> music
            features: vec![(WAVE, 0.9), (SOUND, 0.8), (ENERGY, 0.4), (PATTERN, 0.5), (MATH, 0.3)] },
        Concept { label: "Information theory (physics)", category: "physics",  // BRIDGE: physics <-> CS
            features: vec![(ENTROPY, 0.9), (INFORMATION, 0.8), (MATH, 0.7), (COMPUTATION, 0.4)] },
        Concept { label: "Biophysics", category: "physics",  // BRIDGE: physics <-> biology
            features: vec![(ENERGY, 0.6), (LIFE, 0.5), (CHEMISTRY, 0.5), (FORCE, 0.4), (SYSTEMS, 0.3)] },
        Concept { label: "Nuclear physics", category: "physics",
            features: vec![(QUANTUM, 0.8), (ENERGY, 1.0), (FORCE, 0.8)] },
        Concept { label: "Condensed matter", category: "physics",
            features: vec![(QUANTUM, 0.6), (STRUCTURE, 0.7), (ENERGY, 0.5), (MATH, 0.5)] },

        // ── Biology (15) ──────────────────────────────────────────────
        Concept { label: "Evolution by natural selection", category: "biology",
            features: vec![(EVOLUTION, 1.0), (LIFE, 0.9), (GENETICS, 0.7), (NATURE, 0.6)] },
        Concept { label: "Molecular biology", category: "biology",
            features: vec![(LIFE, 0.9), (CHEMISTRY, 0.8), (GENETICS, 0.7), (STRUCTURE, 0.5)] },
        Concept { label: "Ecology", category: "biology",
            features: vec![(LIFE, 0.8), (NATURE, 1.0), (SYSTEMS, 0.7), (EVOLUTION, 0.4)] },
        Concept { label: "Genetics", category: "biology",
            features: vec![(GENETICS, 1.0), (LIFE, 0.8), (INFORMATION, 0.5), (CHEMISTRY, 0.4)] },
        Concept { label: "Cell biology", category: "biology",
            features: vec![(LIFE, 1.0), (CHEMISTRY, 0.6), (SYSTEMS, 0.5), (STRUCTURE, 0.5)] },
        Concept { label: "Neuroscience", category: "biology",  // BRIDGE: biology <-> philosophy, medicine
            features: vec![(LIFE, 0.7), (MIND, 0.7), (COGNITION, 0.8), (CHEMISTRY, 0.4), (NETWORK, 0.5)] },
        Concept { label: "Bioinformatics", category: "biology",  // BRIDGE: biology <-> CS
            features: vec![(LIFE, 0.5), (COMPUTATION, 0.7), (GENETICS, 0.6), (INFORMATION, 0.6), (STATISTICS, 0.5)] },
        Concept { label: "Immunology", category: "biology",
            features: vec![(LIFE, 0.8), (CHEMISTRY, 0.5), (SYSTEMS, 0.6), (EVOLUTION, 0.3)] },
        Concept { label: "Botany", category: "biology",
            features: vec![(LIFE, 0.8), (NATURE, 0.9), (CHEMISTRY, 0.3), (EVOLUTION, 0.3)] },
        Concept { label: "Marine biology", category: "biology",
            features: vec![(LIFE, 0.8), (NATURE, 0.8), (EVOLUTION, 0.4), (SYSTEMS, 0.3)] },
        Concept { label: "Microbiology", category: "biology",
            features: vec![(LIFE, 0.9), (CHEMISTRY, 0.5), (EVOLUTION, 0.5), (GENETICS, 0.4)] },
        Concept { label: "Developmental biology", category: "biology",
            features: vec![(LIFE, 0.9), (GENETICS, 0.6), (SYSTEMS, 0.5), (STRUCTURE, 0.4)] },
        Concept { label: "Evolutionary psychology", category: "biology",  // BRIDGE: biology <-> philosophy
            features: vec![(EVOLUTION, 0.7), (MIND, 0.6), (BEHAVIOR, 0.6), (COGNITION, 0.5)] },
        Concept { label: "Biostatistics", category: "biology",
            features: vec![(LIFE, 0.5), (STATISTICS, 0.8), (MATH, 0.5), (GENETICS, 0.3)] },
        Concept { label: "Taxonomy", category: "biology",
            features: vec![(LIFE, 0.7), (EVOLUTION, 0.6), (STRUCTURE, 0.6), (NATURE, 0.5)] },

        // ── Computer Science (15) ─────────────────────────────────────
        Concept { label: "Algorithm design", category: "computer_science",
            features: vec![(COMPUTATION, 0.9), (LOGIC, 0.8), (MATH, 0.7), (OPTIMIZATION, 0.6)] },
        Concept { label: "Machine learning", category: "computer_science",
            features: vec![(COMPUTATION, 0.8), (STATISTICS, 0.7), (PATTERN, 0.8), (OPTIMIZATION, 0.7)] },
        Concept { label: "Database systems", category: "computer_science",
            features: vec![(COMPUTATION, 0.7), (INFORMATION, 0.8), (STRUCTURE, 0.7), (SYSTEMS, 0.6)] },
        Concept { label: "Networking", category: "computer_science",
            features: vec![(COMPUTATION, 0.6), (NETWORK, 0.9), (SYSTEMS, 0.7), (INFORMATION, 0.5)] },
        Concept { label: "Cryptography", category: "computer_science",
            features: vec![(COMPUTATION, 0.8), (MATH, 0.9), (INFORMATION, 0.7), (LOGIC, 0.5)] },
        Concept { label: "Operating systems", category: "computer_science",
            features: vec![(COMPUTATION, 0.8), (SYSTEMS, 0.9), (STRUCTURE, 0.5), (LOGIC, 0.4)] },
        Concept { label: "Artificial intelligence", category: "computer_science",
            features: vec![(COMPUTATION, 0.8), (COGNITION, 0.6), (LOGIC, 0.6), (PATTERN, 0.5), (MIND, 0.3)] },
        Concept { label: "Computational complexity", category: "computer_science",
            features: vec![(COMPUTATION, 0.9), (MATH, 0.9), (LOGIC, 0.8)] },
        Concept { label: "Computer graphics", category: "computer_science",
            features: vec![(COMPUTATION, 0.7), (MATH, 0.5), (SPACE, 0.5), (WAVE, 0.3), (PATTERN, 0.4)] },
        Concept { label: "Natural language processing", category: "computer_science",  // BRIDGE: CS <-> linguistics
            features: vec![(COMPUTATION, 0.7), (LANGUAGE, 0.8), (PATTERN, 0.6), (COGNITION, 0.4), (STATISTICS, 0.5)] },
        Concept { label: "Algorithmic trading", category: "computer_science",  // BRIDGE: CS <-> economics
            features: vec![(COMPUTATION, 0.6), (MARKETS, 0.7), (OPTIMIZATION, 0.7), (STATISTICS, 0.5)] },
        Concept { label: "Computational biology", category: "computer_science",  // BRIDGE: CS <-> biology
            features: vec![(COMPUTATION, 0.7), (LIFE, 0.4), (GENETICS, 0.4), (STATISTICS, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Formal verification", category: "computer_science",
            features: vec![(COMPUTATION, 0.7), (LOGIC, 1.0), (MATH, 0.8)] },
        Concept { label: "Information retrieval", category: "computer_science",
            features: vec![(COMPUTATION, 0.6), (INFORMATION, 0.9), (LANGUAGE, 0.4), (PATTERN, 0.5)] },
        Concept { label: "Distributed systems", category: "computer_science",
            features: vec![(COMPUTATION, 0.7), (SYSTEMS, 0.8), (NETWORK, 0.7), (LOGIC, 0.3)] },

        // ── Philosophy (12) ───────────────────────────────────────────
        Concept { label: "Formal logic", category: "philosophy",
            features: vec![(LOGIC, 1.0), (MATH, 0.7), (LANGUAGE, 0.4), (STRUCTURE, 0.4)] },
        Concept { label: "Ethics", category: "philosophy",
            features: vec![(ETHICS, 1.0), (BEHAVIOR, 0.6), (MIND, 0.4)] },
        Concept { label: "Philosophy of mind", category: "philosophy",
            features: vec![(MIND, 1.0), (COGNITION, 0.7), (METAPHYSICS, 0.6), (LANGUAGE, 0.3)] },
        Concept { label: "Epistemology", category: "philosophy",
            features: vec![(MIND, 0.7), (LOGIC, 0.6), (METAPHYSICS, 0.5), (COGNITION, 0.5)] },
        Concept { label: "Metaphysics", category: "philosophy",
            features: vec![(METAPHYSICS, 1.0), (MIND, 0.5), (SPACE, 0.3), (STRUCTURE, 0.3)] },
        Concept { label: "Philosophy of language", category: "philosophy",
            features: vec![(LANGUAGE, 0.8), (LOGIC, 0.6), (MIND, 0.5), (STRUCTURE, 0.4)] },
        Concept { label: "Political philosophy", category: "philosophy",
            features: vec![(ETHICS, 0.7), (BEHAVIOR, 0.6), (SYSTEMS, 0.5), (MARKETS, 0.2)] },
        Concept { label: "Bioethics", category: "philosophy",  // BRIDGE: philosophy <-> medicine
            features: vec![(ETHICS, 0.9), (LIFE, 0.5), (DIAGNOSTICS, 0.3), (MIND, 0.3)] },
        Concept { label: "Game theory (philosophy)", category: "philosophy",  // BRIDGE: philosophy <-> economics, CS
            features: vec![(LOGIC, 0.7), (BEHAVIOR, 0.7), (MATH, 0.6), (OPTIMIZATION, 0.5), (MARKETS, 0.3)] },
        Concept { label: "Aesthetics", category: "philosophy",
            features: vec![(MIND, 0.5), (EMOTION, 0.7), (PERFORMANCE, 0.3), (ETHICS, 0.2)] },
        Concept { label: "Philosophy of science", category: "philosophy",
            features: vec![(LOGIC, 0.6), (METAPHYSICS, 0.5), (MIND, 0.4), (STRUCTURE, 0.4), (SYSTEMS, 0.3)] },
        Concept { label: "Phenomenology", category: "philosophy",
            features: vec![(MIND, 0.8), (METAPHYSICS, 0.7), (COGNITION, 0.5)] },

        // ── Economics (12) ────────────────────────────────────────────
        Concept { label: "Microeconomics", category: "economics",
            features: vec![(MARKETS, 0.9), (OPTIMIZATION, 0.8), (BEHAVIOR, 0.6), (MATH, 0.5)] },
        Concept { label: "Macroeconomics", category: "economics",
            features: vec![(MARKETS, 0.8), (SYSTEMS, 0.7), (STATISTICS, 0.5), (BEHAVIOR, 0.4)] },
        Concept { label: "Behavioral economics", category: "economics",
            features: vec![(BEHAVIOR, 0.9), (MARKETS, 0.6), (COGNITION, 0.5), (MIND, 0.3)] },
        Concept { label: "Econometrics", category: "economics",
            features: vec![(STATISTICS, 0.9), (MATH, 0.7), (MARKETS, 0.6), (COMPUTATION, 0.3)] },
        Concept { label: "Game theory (economics)", category: "economics",  // BRIDGE: economics <-> philosophy, CS
            features: vec![(MATH, 0.7), (OPTIMIZATION, 0.8), (BEHAVIOR, 0.6), (LOGIC, 0.4), (MARKETS, 0.5)] },
        Concept { label: "Financial engineering", category: "economics",
            features: vec![(MARKETS, 0.8), (MATH, 0.7), (OPTIMIZATION, 0.6), (COMPUTATION, 0.4)] },
        Concept { label: "Development economics", category: "economics",
            features: vec![(MARKETS, 0.6), (SYSTEMS, 0.5), (BEHAVIOR, 0.5), (ETHICS, 0.3)] },
        Concept { label: "Labor economics", category: "economics",
            features: vec![(MARKETS, 0.7), (BEHAVIOR, 0.6), (SYSTEMS, 0.4), (STATISTICS, 0.3)] },
        Concept { label: "Network economics", category: "economics",  // BRIDGE: economics <-> CS
            features: vec![(MARKETS, 0.6), (NETWORK, 0.7), (SYSTEMS, 0.6), (OPTIMIZATION, 0.4)] },
        Concept { label: "Environmental economics", category: "economics",
            features: vec![(MARKETS, 0.5), (NATURE, 0.5), (SYSTEMS, 0.5), (ETHICS, 0.3)] },
        Concept { label: "Public choice theory", category: "economics",
            features: vec![(BEHAVIOR, 0.7), (MARKETS, 0.5), (LOGIC, 0.4), (SYSTEMS, 0.4)] },
        Concept { label: "Auction theory", category: "economics",
            features: vec![(MATH, 0.6), (OPTIMIZATION, 0.7), (MARKETS, 0.7), (BEHAVIOR, 0.4)] },

        // ── Music (12) ────────────────────────────────────────────────
        Concept { label: "Harmonic theory", category: "music",
            features: vec![(SOUND, 0.8), (MATH, 0.7), (PATTERN, 0.8), (WAVE, 0.5)] },
        Concept { label: "Orchestration", category: "music",
            features: vec![(SOUND, 0.9), (PERFORMANCE, 0.7), (EMOTION, 0.5), (SYSTEMS, 0.3)] },
        Concept { label: "Rhythm and meter", category: "music",
            features: vec![(SOUND, 0.7), (PATTERN, 1.0), (MATH, 0.4), (PERFORMANCE, 0.4)] },
        Concept { label: "Musical acoustics", category: "music",  // BRIDGE: music <-> physics
            features: vec![(SOUND, 0.9), (WAVE, 0.8), (ENERGY, 0.4), (MATH, 0.3), (PATTERN, 0.3)] },
        Concept { label: "Music cognition", category: "music",  // BRIDGE: music <-> linguistics, philosophy
            features: vec![(SOUND, 0.5), (COGNITION, 0.7), (EMOTION, 0.6), (PATTERN, 0.5), (MIND, 0.3)] },
        Concept { label: "Composition", category: "music",
            features: vec![(SOUND, 0.7), (EMOTION, 0.7), (PATTERN, 0.6), (STRUCTURE, 0.5)] },
        Concept { label: "Ethnomusicology", category: "music",  // BRIDGE: music <-> linguistics
            features: vec![(SOUND, 0.6), (LANGUAGE, 0.5), (BEHAVIOR, 0.4), (PATTERN, 0.4), (COGNITION, 0.3)] },
        Concept { label: "Music theory", category: "music",
            features: vec![(SOUND, 0.7), (MATH, 0.6), (PATTERN, 0.7), (STRUCTURE, 0.5)] },
        Concept { label: "Performance practice", category: "music",
            features: vec![(PERFORMANCE, 1.0), (SOUND, 0.7), (EMOTION, 0.6)] },
        Concept { label: "Digital audio", category: "music",  // BRIDGE: music <-> CS
            features: vec![(SOUND, 0.7), (COMPUTATION, 0.6), (WAVE, 0.5), (INFORMATION, 0.4)] },
        Concept { label: "Music therapy", category: "music",  // BRIDGE: music <-> medicine
            features: vec![(SOUND, 0.5), (EMOTION, 0.8), (MIND, 0.4), (DIAGNOSTICS, 0.2), (COGNITION, 0.3)] },
        Concept { label: "Counterpoint", category: "music",
            features: vec![(SOUND, 0.6), (PATTERN, 0.7), (MATH, 0.5), (STRUCTURE, 0.6)] },

        // ── Medicine (12) ─────────────────────────────────────────────
        Concept { label: "Clinical diagnostics", category: "medicine",
            features: vec![(DIAGNOSTICS, 1.0), (LIFE, 0.6), (STATISTICS, 0.5), (SYSTEMS, 0.3)] },
        Concept { label: "Pharmacology", category: "medicine",
            features: vec![(CHEMISTRY, 0.9), (LIFE, 0.7), (DIAGNOSTICS, 0.4), (SYSTEMS, 0.3)] },
        Concept { label: "Epidemiology", category: "medicine",
            features: vec![(STATISTICS, 0.9), (LIFE, 0.6), (SYSTEMS, 0.6), (NETWORK, 0.4)] },
        Concept { label: "Surgery", category: "medicine",
            features: vec![(LIFE, 0.7), (DIAGNOSTICS, 0.6), (FORCE, 0.3), (SYSTEMS, 0.3)] },
        Concept { label: "Psychiatry", category: "medicine",
            features: vec![(MIND, 0.7), (LIFE, 0.6), (DIAGNOSTICS, 0.5), (BEHAVIOR, 0.5), (CHEMISTRY, 0.3)] },
        Concept { label: "Radiology", category: "medicine",
            features: vec![(DIAGNOSTICS, 0.8), (WAVE, 0.5), (ENERGY, 0.3), (COMPUTATION, 0.3)] },
        Concept { label: "Pathology", category: "medicine",
            features: vec![(LIFE, 0.7), (CHEMISTRY, 0.5), (DIAGNOSTICS, 0.7), (STRUCTURE, 0.3)] },
        Concept { label: "Genomic medicine", category: "medicine",  // BRIDGE: medicine <-> biology
            features: vec![(GENETICS, 0.7), (LIFE, 0.6), (DIAGNOSTICS, 0.5), (INFORMATION, 0.4), (COMPUTATION, 0.3)] },
        Concept { label: "Medical imaging AI", category: "medicine",  // BRIDGE: medicine <-> CS
            features: vec![(DIAGNOSTICS, 0.6), (COMPUTATION, 0.7), (PATTERN, 0.6), (STATISTICS, 0.4)] },
        Concept { label: "Public health", category: "medicine",
            features: vec![(LIFE, 0.5), (STATISTICS, 0.6), (SYSTEMS, 0.6), (BEHAVIOR, 0.4), (ETHICS, 0.3)] },
        Concept { label: "Neurology", category: "medicine",
            features: vec![(LIFE, 0.6), (MIND, 0.5), (DIAGNOSTICS, 0.6), (COGNITION, 0.4), (NETWORK, 0.3)] },
        Concept { label: "Immunotherapy", category: "medicine",
            features: vec![(LIFE, 0.7), (CHEMISTRY, 0.5), (SYSTEMS, 0.5), (DIAGNOSTICS, 0.3)] },

        // ── Linguistics (12) ──────────────────────────────────────────
        Concept { label: "Syntax", category: "linguistics",
            features: vec![(LANGUAGE, 1.0), (STRUCTURE, 0.8), (LOGIC, 0.4), (PATTERN, 0.4)] },
        Concept { label: "Semantics", category: "linguistics",
            features: vec![(LANGUAGE, 0.9), (LOGIC, 0.6), (MIND, 0.4), (STRUCTURE, 0.4)] },
        Concept { label: "Phonology", category: "linguistics",
            features: vec![(LANGUAGE, 0.8), (SOUND, 0.7), (PATTERN, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Pragmatics", category: "linguistics",
            features: vec![(LANGUAGE, 0.8), (BEHAVIOR, 0.5), (COGNITION, 0.4), (MIND, 0.3)] },
        Concept { label: "Psycholinguistics", category: "linguistics",  // BRIDGE: linguistics <-> biology/medicine
            features: vec![(LANGUAGE, 0.7), (COGNITION, 0.8), (MIND, 0.5), (LIFE, 0.3), (BEHAVIOR, 0.3)] },
        Concept { label: "Computational linguistics", category: "linguistics",  // BRIDGE: linguistics <-> CS
            features: vec![(LANGUAGE, 0.7), (COMPUTATION, 0.7), (PATTERN, 0.5), (STATISTICS, 0.5), (STRUCTURE, 0.3)] },
        Concept { label: "Historical linguistics", category: "linguistics",
            features: vec![(LANGUAGE, 0.9), (EVOLUTION, 0.4), (PATTERN, 0.4), (STRUCTURE, 0.3)] },
        Concept { label: "Sociolinguistics", category: "linguistics",
            features: vec![(LANGUAGE, 0.8), (BEHAVIOR, 0.6), (SYSTEMS, 0.3), (COGNITION, 0.2)] },
        Concept { label: "Morphology", category: "linguistics",
            features: vec![(LANGUAGE, 0.8), (STRUCTURE, 0.7), (PATTERN, 0.5)] },
        Concept { label: "Typology", category: "linguistics",
            features: vec![(LANGUAGE, 0.7), (STRUCTURE, 0.6), (PATTERN, 0.5), (SYSTEMS, 0.3)] },
        Concept { label: "Corpus linguistics", category: "linguistics",
            features: vec![(LANGUAGE, 0.7), (STATISTICS, 0.6), (COMPUTATION, 0.4), (PATTERN, 0.5)] },
        Concept { label: "Discourse analysis", category: "linguistics",
            features: vec![(LANGUAGE, 0.8), (COGNITION, 0.4), (STRUCTURE, 0.4), (BEHAVIOR, 0.3)] },

        // ── Mathematics (15) ──────────────────────────────────────────
        Concept { label: "Number theory", category: "mathematics",
            features: vec![(MATH, 1.0), (LOGIC, 0.7), (PATTERN, 0.7), (STRUCTURE, 0.6)] },
        Concept { label: "Topology", category: "mathematics",
            features: vec![(MATH, 0.9), (STRUCTURE, 0.8), (SPACE, 0.7), (PATTERN, 0.4)] },
        Concept { label: "Real analysis", category: "mathematics",
            features: vec![(MATH, 1.0), (LOGIC, 0.8), (STRUCTURE, 0.6)] },
        Concept { label: "Complex analysis", category: "mathematics",
            features: vec![(MATH, 0.9), (STRUCTURE, 0.7), (WAVE, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Abstract algebra", category: "mathematics",
            features: vec![(MATH, 0.9), (STRUCTURE, 0.9), (LOGIC, 0.7), (PATTERN, 0.5)] },
        Concept { label: "Differential geometry", category: "mathematics",  // BRIDGE: math <-> physics
            features: vec![(MATH, 0.9), (SPACE, 0.8), (STRUCTURE, 0.6), (FORCE, 0.3)] },
        Concept { label: "Combinatorics", category: "mathematics",
            features: vec![(MATH, 0.9), (PATTERN, 0.7), (LOGIC, 0.6), (COMPUTATION, 0.4)] },
        Concept { label: "Category theory", category: "mathematics",
            features: vec![(MATH, 0.9), (LOGIC, 0.8), (STRUCTURE, 0.9), (PATTERN, 0.5)] },
        Concept { label: "Set theory", category: "mathematics",
            features: vec![(MATH, 0.9), (LOGIC, 0.9), (STRUCTURE, 0.7)] },
        Concept { label: "Graph theory", category: "mathematics",  // BRIDGE: math <-> CS
            features: vec![(MATH, 0.8), (NETWORK, 0.9), (STRUCTURE, 0.7), (PATTERN, 0.5)] },
        Concept { label: "Probability theory", category: "mathematics",  // BRIDGE: math <-> statistics
            features: vec![(MATH, 0.9), (STATISTICS, 0.9), (LOGIC, 0.6), (PATTERN, 0.4)] },
        Concept { label: "Mathematical logic", category: "mathematics",  // BRIDGE: math <-> philosophy
            features: vec![(MATH, 0.9), (LOGIC, 1.0), (LANGUAGE, 0.3), (STRUCTURE, 0.5)] },
        Concept { label: "Numerical analysis", category: "mathematics",  // BRIDGE: math <-> CS
            features: vec![(MATH, 0.8), (COMPUTATION, 0.7), (OPTIMIZATION, 0.5), (STATISTICS, 0.3)] },
        Concept { label: "Dynamical systems theory", category: "mathematics",
            features: vec![(MATH, 0.8), (SYSTEMS, 0.8), (PATTERN, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Measure theory", category: "mathematics",
            features: vec![(MATH, 1.0), (LOGIC, 0.6), (STRUCTURE, 0.7), (STATISTICS, 0.3)] },

        // ── Chemistry (15) ────────────────────────────────────────────
        Concept { label: "Organic chemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 1.0), (STRUCTURE, 0.7), (LIFE, 0.4), (PATTERN, 0.4)] },
        Concept { label: "Inorganic chemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.9), (STRUCTURE, 0.7), (ENERGY, 0.4)] },
        Concept { label: "Physical chemistry", category: "chemistry",  // BRIDGE: chem <-> physics
            features: vec![(CHEMISTRY, 0.8), (ENERGY, 0.7), (MATH, 0.6), (QUANTUM, 0.5)] },
        Concept { label: "Biochemistry", category: "chemistry",  // BRIDGE: chem <-> biology
            features: vec![(CHEMISTRY, 0.9), (LIFE, 0.9), (STRUCTURE, 0.5), (GENETICS, 0.3)] },
        Concept { label: "Analytical chemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.9), (DIAGNOSTICS, 0.7), (PATTERN, 0.5), (STATISTICS, 0.4)] },
        Concept { label: "Quantum chemistry", category: "chemistry",  // BRIDGE: chem <-> physics
            features: vec![(CHEMISTRY, 0.8), (QUANTUM, 0.9), (ENERGY, 0.7), (MATH, 0.6)] },
        Concept { label: "Thermochemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.8), (ENERGY, 0.9), (ENTROPY, 0.7)] },
        Concept { label: "Polymer chemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.8), (STRUCTURE, 0.8), (PATTERN, 0.5)] },
        Concept { label: "Electrochemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.9), (ENERGY, 0.7), (FORCE, 0.5)] },
        Concept { label: "Green chemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.8), (NATURE, 0.7), (ETHICS, 0.5), (SYSTEMS, 0.3)] },
        Concept { label: "Materials chemistry", category: "chemistry",
            features: vec![(CHEMISTRY, 0.8), (STRUCTURE, 0.9), (ENERGY, 0.4), (QUANTUM, 0.3)] },
        Concept { label: "Astrochemistry", category: "chemistry",  // BRIDGE: chem <-> astronomy
            features: vec![(CHEMISTRY, 0.7), (SPACE, 0.8), (QUANTUM, 0.3), (ENERGY, 0.4)] },
        Concept { label: "Nuclear chemistry", category: "chemistry",  // BRIDGE: chem <-> physics
            features: vec![(CHEMISTRY, 0.7), (QUANTUM, 0.7), (ENERGY, 0.9), (FORCE, 0.4)] },
        Concept { label: "Computational chemistry", category: "chemistry",  // BRIDGE: chem <-> CS
            features: vec![(CHEMISTRY, 0.7), (COMPUTATION, 0.8), (MATH, 0.6), (QUANTUM, 0.4)] },
        Concept { label: "Catalysis", category: "chemistry",
            features: vec![(CHEMISTRY, 0.9), (ENERGY, 0.6), (OPTIMIZATION, 0.5), (PATTERN, 0.3)] },

        // ── Psychology (15) ───────────────────────────────────────────
        Concept { label: "Cognitive psychology", category: "psychology",
            features: vec![(MIND, 0.8), (COGNITION, 1.0), (PATTERN, 0.5), (LOGIC, 0.4)] },
        Concept { label: "Developmental psychology", category: "psychology",
            features: vec![(MIND, 0.7), (BEHAVIOR, 0.6), (COGNITION, 0.6), (EVOLUTION, 0.3)] },
        Concept { label: "Social psychology", category: "psychology",
            features: vec![(BEHAVIOR, 0.9), (MIND, 0.7), (NETWORK, 0.6), (COGNITION, 0.4)] },
        Concept { label: "Clinical psychology", category: "psychology",  // BRIDGE: psychology <-> medicine
            features: vec![(MIND, 0.8), (DIAGNOSTICS, 0.7), (BEHAVIOR, 0.6), (EMOTION, 0.6)] },
        Concept { label: "Behavioral psychology", category: "psychology",
            features: vec![(BEHAVIOR, 1.0), (MIND, 0.6), (PATTERN, 0.5)] },
        Concept { label: "Personality psychology", category: "psychology",
            features: vec![(MIND, 0.7), (BEHAVIOR, 0.7), (PATTERN, 0.5), (EMOTION, 0.4)] },
        Concept { label: "Neuropsychology", category: "psychology",  // BRIDGE: psychology <-> neuroscience
            features: vec![(MIND, 0.8), (COGNITION, 0.7), (LIFE, 0.5), (DIAGNOSTICS, 0.4)] },
        Concept { label: "Educational psychology", category: "psychology",
            features: vec![(MIND, 0.6), (COGNITION, 0.7), (BEHAVIOR, 0.5), (LANGUAGE, 0.3)] },
        Concept { label: "Organizational psychology", category: "psychology",  // BRIDGE: psychology <-> economics
            features: vec![(BEHAVIOR, 0.8), (SYSTEMS, 0.6), (MARKETS, 0.4), (NETWORK, 0.4)] },
        Concept { label: "Positive psychology", category: "psychology",
            features: vec![(MIND, 0.7), (EMOTION, 0.9), (BEHAVIOR, 0.5), (ETHICS, 0.3)] },
        Concept { label: "Forensic psychology", category: "psychology",  // BRIDGE: psychology <-> law
            features: vec![(MIND, 0.6), (BEHAVIOR, 0.7), (ETHICS, 0.5), (LOGIC, 0.4)] },
        Concept { label: "Psychometrics", category: "psychology",
            features: vec![(MIND, 0.6), (STATISTICS, 0.9), (COGNITION, 0.5), (PATTERN, 0.5)] },
        Concept { label: "Humanistic psychology", category: "psychology",
            features: vec![(MIND, 0.8), (EMOTION, 0.8), (ETHICS, 0.5), (METAPHYSICS, 0.3)] },
        Concept { label: "Gestalt psychology", category: "psychology",
            features: vec![(MIND, 0.7), (PATTERN, 0.8), (COGNITION, 0.7), (STRUCTURE, 0.4)] },
        Concept { label: "Abnormal psychology", category: "psychology",
            features: vec![(MIND, 0.7), (BEHAVIOR, 0.7), (DIAGNOSTICS, 0.6), (EMOTION, 0.5)] },

        // ── Engineering (15) ──────────────────────────────────────────
        Concept { label: "Mechanical engineering", category: "engineering",
            features: vec![(STRUCTURE, 0.8), (FORCE, 0.9), (ENERGY, 0.6), (SYSTEMS, 0.5)] },
        Concept { label: "Electrical engineering", category: "engineering",
            features: vec![(ENERGY, 0.8), (WAVE, 0.7), (FORCE, 0.5), (SYSTEMS, 0.6)] },
        Concept { label: "Civil engineering", category: "engineering",
            features: vec![(STRUCTURE, 1.0), (FORCE, 0.7), (SYSTEMS, 0.5), (MATH, 0.3)] },
        Concept { label: "Chemical engineering", category: "engineering",  // BRIDGE: eng <-> chemistry
            features: vec![(CHEMISTRY, 0.9), (SYSTEMS, 0.7), (OPTIMIZATION, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Aerospace engineering", category: "engineering",
            features: vec![(FORCE, 0.7), (ENERGY, 0.7), (SPACE, 0.7), (STRUCTURE, 0.6)] },
        Concept { label: "Materials science", category: "engineering",  // BRIDGE: eng <-> chemistry/physics
            features: vec![(STRUCTURE, 0.9), (CHEMISTRY, 0.7), (ENERGY, 0.4), (QUANTUM, 0.3)] },
        Concept { label: "Control theory", category: "engineering",
            features: vec![(SYSTEMS, 0.9), (OPTIMIZATION, 0.8), (MATH, 0.7), (FORCE, 0.4)] },
        Concept { label: "Robotics", category: "engineering",  // BRIDGE: eng <-> CS
            features: vec![(COMPUTATION, 0.8), (FORCE, 0.6), (SYSTEMS, 0.7), (STRUCTURE, 0.6)] },
        Concept { label: "Biomedical engineering", category: "engineering",  // BRIDGE: eng <-> medicine
            features: vec![(STRUCTURE, 0.6), (LIFE, 0.7), (SYSTEMS, 0.5), (DIAGNOSTICS, 0.4)] },
        Concept { label: "Systems engineering", category: "engineering",
            features: vec![(SYSTEMS, 1.0), (OPTIMIZATION, 0.7), (STRUCTURE, 0.6), (LOGIC, 0.4)] },
        Concept { label: "Nuclear engineering", category: "engineering",  // BRIDGE: eng <-> physics
            features: vec![(ENERGY, 0.9), (QUANTUM, 0.6), (SYSTEMS, 0.5), (STRUCTURE, 0.4)] },
        Concept { label: "Environmental engineering", category: "engineering",
            features: vec![(NATURE, 0.7), (SYSTEMS, 0.7), (CHEMISTRY, 0.5), (ETHICS, 0.3)] },
        Concept { label: "Software engineering", category: "engineering",  // BRIDGE: eng <-> CS
            features: vec![(COMPUTATION, 0.9), (STRUCTURE, 0.8), (LOGIC, 0.7), (SYSTEMS, 0.6)] },
        Concept { label: "Industrial engineering", category: "engineering",  // BRIDGE: eng <-> economics
            features: vec![(OPTIMIZATION, 0.9), (SYSTEMS, 0.7), (MARKETS, 0.5), (STATISTICS, 0.4)] },
        Concept { label: "Structural engineering", category: "engineering",
            features: vec![(STRUCTURE, 1.0), (FORCE, 0.8), (MATH, 0.5), (OPTIMIZATION, 0.4)] },

        // ── Earth Science (15) ────────────────────────────────────────
        Concept { label: "Geology", category: "earth_science",
            features: vec![(NATURE, 0.9), (STRUCTURE, 0.8), (PATTERN, 0.5), (EVOLUTION, 0.4)] },
        Concept { label: "Meteorology", category: "earth_science",
            features: vec![(NATURE, 0.8), (SYSTEMS, 0.7), (PATTERN, 0.6), (ENERGY, 0.5)] },
        Concept { label: "Oceanography", category: "earth_science",
            features: vec![(NATURE, 0.9), (SYSTEMS, 0.7), (LIFE, 0.5), (FORCE, 0.4)] },
        Concept { label: "Climatology", category: "earth_science",
            features: vec![(NATURE, 0.8), (SYSTEMS, 0.8), (STATISTICS, 0.6), (PATTERN, 0.5)] },
        Concept { label: "Volcanology", category: "earth_science",
            features: vec![(NATURE, 0.8), (ENERGY, 0.7), (FORCE, 0.6), (STRUCTURE, 0.5)] },
        Concept { label: "Seismology", category: "earth_science",  // BRIDGE: earth_sci <-> physics
            features: vec![(WAVE, 0.9), (FORCE, 0.7), (NATURE, 0.7), (STRUCTURE, 0.4)] },
        Concept { label: "Paleontology", category: "earth_science",  // BRIDGE: earth_sci <-> biology
            features: vec![(LIFE, 0.8), (EVOLUTION, 0.9), (NATURE, 0.7), (STRUCTURE, 0.5)] },
        Concept { label: "Hydrology", category: "earth_science",
            features: vec![(NATURE, 0.8), (SYSTEMS, 0.6), (FORCE, 0.5), (STATISTICS, 0.3)] },
        Concept { label: "Geomorphology", category: "earth_science",
            features: vec![(NATURE, 0.8), (STRUCTURE, 0.7), (FORCE, 0.5), (PATTERN, 0.5)] },
        Concept { label: "Mineralogy", category: "earth_science",  // BRIDGE: earth_sci <-> chemistry
            features: vec![(NATURE, 0.7), (STRUCTURE, 0.8), (CHEMISTRY, 0.7), (PATTERN, 0.5)] },
        Concept { label: "Geophysics", category: "earth_science",  // BRIDGE: earth_sci <-> physics
            features: vec![(NATURE, 0.6), (ENERGY, 0.7), (FORCE, 0.7), (WAVE, 0.6)] },
        Concept { label: "Geochemistry", category: "earth_science",  // BRIDGE: earth_sci <-> chemistry
            features: vec![(NATURE, 0.7), (CHEMISTRY, 0.8), (STRUCTURE, 0.5)] },
        Concept { label: "Atmospheric science", category: "earth_science",
            features: vec![(NATURE, 0.7), (ENERGY, 0.6), (SYSTEMS, 0.6), (WAVE, 0.4)] },
        Concept { label: "Glaciology", category: "earth_science",
            features: vec![(NATURE, 0.9), (ENERGY, 0.5), (STRUCTURE, 0.6), (PATTERN, 0.4)] },
        Concept { label: "Geodesy", category: "earth_science",  // BRIDGE: earth_sci <-> math
            features: vec![(SPACE, 0.8), (MATH, 0.7), (NATURE, 0.6), (STRUCTURE, 0.4)] },

        // ── Astronomy (15) ────────────────────────────────────────────
        Concept { label: "Observational astronomy", category: "astronomy",
            features: vec![(SPACE, 0.9), (WAVE, 0.7), (PATTERN, 0.5), (DIAGNOSTICS, 0.4)] },
        Concept { label: "Astrophysics", category: "astronomy",  // BRIDGE: astronomy <-> physics
            features: vec![(SPACE, 1.0), (ENERGY, 0.9), (QUANTUM, 0.7), (MATH, 0.6)] },
        Concept { label: "Planetary science", category: "astronomy",
            features: vec![(SPACE, 0.9), (STRUCTURE, 0.6), (CHEMISTRY, 0.5), (NATURE, 0.4)] },
        Concept { label: "Exoplanet research", category: "astronomy",  // BRIDGE: astronomy <-> biology
            features: vec![(SPACE, 0.9), (LIFE, 0.4), (WAVE, 0.6), (DIAGNOSTICS, 0.5)] },
        Concept { label: "Stellar evolution", category: "astronomy",
            features: vec![(SPACE, 0.8), (EVOLUTION, 0.7), (ENERGY, 0.8), (QUANTUM, 0.4)] },
        Concept { label: "Radio astronomy", category: "astronomy",
            features: vec![(SPACE, 0.8), (WAVE, 0.9), (INFORMATION, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Astrobiology", category: "astronomy",  // BRIDGE: astronomy <-> biology
            features: vec![(SPACE, 0.8), (LIFE, 0.8), (CHEMISTRY, 0.6), (EVOLUTION, 0.5)] },
        Concept { label: "Galactic dynamics", category: "astronomy",
            features: vec![(SPACE, 0.9), (FORCE, 0.6), (NETWORK, 0.5), (MATH, 0.5)] },
        Concept { label: "Solar physics", category: "astronomy",
            features: vec![(SPACE, 0.7), (ENERGY, 0.9), (WAVE, 0.7), (QUANTUM, 0.5)] },
        Concept { label: "Gravitational wave astronomy", category: "astronomy",  // BRIDGE: astronomy <-> physics
            features: vec![(SPACE, 0.8), (WAVE, 0.9), (FORCE, 0.8), (MATH, 0.5)] },
        Concept { label: "Dark matter studies", category: "astronomy",
            features: vec![(SPACE, 0.8), (QUANTUM, 0.7), (ENERGY, 0.6), (MATH, 0.7)] },
        Concept { label: "High-energy astrophysics", category: "astronomy",
            features: vec![(SPACE, 0.8), (ENERGY, 1.0), (QUANTUM, 0.7), (FORCE, 0.5)] },
        Concept { label: "Astrometry", category: "astronomy",
            features: vec![(SPACE, 0.9), (MATH, 0.7), (DIAGNOSTICS, 0.5), (STATISTICS, 0.4)] },
        Concept { label: "Star formation", category: "astronomy",
            features: vec![(SPACE, 0.8), (ENERGY, 0.7), (STRUCTURE, 0.6), (ENTROPY, 0.4)] },
        Concept { label: "Cosmic microwave background", category: "astronomy",
            features: vec![(SPACE, 0.9), (WAVE, 0.8), (ENTROPY, 0.5), (ENERGY, 0.5)] },

        // ── Visual Arts (15) ──────────────────────────────────────────
        Concept { label: "Painting", category: "visual_arts",
            features: vec![(PATTERN, 0.8), (EMOTION, 0.8), (STRUCTURE, 0.4), (PERFORMANCE, 0.4)] },
        Concept { label: "Sculpture", category: "visual_arts",
            features: vec![(STRUCTURE, 0.8), (SPACE, 0.7), (EMOTION, 0.6), (FORCE, 0.3)] },
        Concept { label: "Drawing", category: "visual_arts",
            features: vec![(PATTERN, 0.7), (STRUCTURE, 0.5), (PERFORMANCE, 0.5), (EMOTION, 0.5)] },
        Concept { label: "Printmaking", category: "visual_arts",
            features: vec![(PATTERN, 0.7), (STRUCTURE, 0.6), (CHEMISTRY, 0.3), (PERFORMANCE, 0.4)] },
        Concept { label: "Photography", category: "visual_arts",  // BRIDGE: visual_arts <-> physics
            features: vec![(PATTERN, 0.7), (WAVE, 0.5), (EMOTION, 0.6), (PERFORMANCE, 0.4)] },
        Concept { label: "Digital art", category: "visual_arts",  // BRIDGE: visual_arts <-> CS
            features: vec![(COMPUTATION, 0.7), (PATTERN, 0.7), (EMOTION, 0.5), (STRUCTURE, 0.3)] },
        Concept { label: "Art history", category: "visual_arts",  // BRIDGE: visual_arts <-> history
            features: vec![(LANGUAGE, 0.6), (PATTERN, 0.6), (STRUCTURE, 0.5), (EMOTION, 0.4)] },
        Concept { label: "Art theory", category: "visual_arts",  // BRIDGE: visual_arts <-> philosophy
            features: vec![(METAPHYSICS, 0.6), (EMOTION, 0.6), (PATTERN, 0.4), (MIND, 0.5)] },
        Concept { label: "Illustration", category: "visual_arts",
            features: vec![(PATTERN, 0.7), (LANGUAGE, 0.5), (EMOTION, 0.6), (PERFORMANCE, 0.3)] },
        Concept { label: "Ceramics", category: "visual_arts",
            features: vec![(STRUCTURE, 0.7), (CHEMISTRY, 0.5), (PATTERN, 0.6), (PERFORMANCE, 0.5)] },
        Concept { label: "Textile arts", category: "visual_arts",
            features: vec![(PATTERN, 0.8), (STRUCTURE, 0.6), (EMOTION, 0.4), (PERFORMANCE, 0.4)] },
        Concept { label: "Calligraphy", category: "visual_arts",  // BRIDGE: visual_arts <-> linguistics
            features: vec![(LANGUAGE, 0.7), (PATTERN, 0.7), (PERFORMANCE, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Installation art", category: "visual_arts",
            features: vec![(SPACE, 0.7), (EMOTION, 0.7), (PERFORMANCE, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Conceptual art", category: "visual_arts",  // BRIDGE: visual_arts <-> philosophy
            features: vec![(MIND, 0.7), (METAPHYSICS, 0.6), (EMOTION, 0.5), (PATTERN, 0.3)] },
        Concept { label: "Glassblowing", category: "visual_arts",
            features: vec![(STRUCTURE, 0.7), (CHEMISTRY, 0.5), (ENERGY, 0.4), (PERFORMANCE, 0.7)] },

        // ── Literature (15) ───────────────────────────────────────────
        Concept { label: "Poetry", category: "literature",  // BRIDGE: literature <-> music
            features: vec![(LANGUAGE, 0.9), (EMOTION, 0.9), (PATTERN, 0.8), (SOUND, 0.6)] },
        Concept { label: "Fiction", category: "literature",
            features: vec![(LANGUAGE, 0.9), (EMOTION, 0.7), (STRUCTURE, 0.6), (COGNITION, 0.4)] },
        Concept { label: "Drama", category: "literature",  // BRIDGE: literature <-> visual_arts
            features: vec![(LANGUAGE, 0.8), (PERFORMANCE, 0.9), (EMOTION, 0.7), (BEHAVIOR, 0.5)] },
        Concept { label: "Literary criticism", category: "literature",
            features: vec![(LANGUAGE, 0.8), (LOGIC, 0.6), (STRUCTURE, 0.5), (MIND, 0.5)] },
        Concept { label: "Comparative literature", category: "literature",
            features: vec![(LANGUAGE, 0.9), (PATTERN, 0.6), (EVOLUTION, 0.4), (STRUCTURE, 0.3)] },
        Concept { label: "Narrative theory", category: "literature",
            features: vec![(LANGUAGE, 0.8), (STRUCTURE, 0.7), (COGNITION, 0.5), (PATTERN, 0.5)] },
        Concept { label: "Rhetoric", category: "literature",
            features: vec![(LANGUAGE, 0.8), (BEHAVIOR, 0.6), (LOGIC, 0.5), (COGNITION, 0.4)] },
        Concept { label: "Creative nonfiction", category: "literature",
            features: vec![(LANGUAGE, 0.8), (EMOTION, 0.6), (STRUCTURE, 0.5), (BEHAVIOR, 0.3)] },
        Concept { label: "Science fiction studies", category: "literature",  // BRIDGE: literature <-> philosophy
            features: vec![(LANGUAGE, 0.7), (METAPHYSICS, 0.5), (COGNITION, 0.5), (EMOTION, 0.4)] },
        Concept { label: "Children's literature", category: "literature",
            features: vec![(LANGUAGE, 0.7), (EMOTION, 0.7), (COGNITION, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Literary theory", category: "literature",
            features: vec![(LANGUAGE, 0.7), (METAPHYSICS, 0.5), (MIND, 0.5), (STRUCTURE, 0.5)] },
        Concept { label: "Digital humanities", category: "literature",  // BRIDGE: literature <-> CS
            features: vec![(LANGUAGE, 0.6), (COMPUTATION, 0.7), (PATTERN, 0.6), (STATISTICS, 0.5)] },
        Concept { label: "Translation studies", category: "literature",
            features: vec![(LANGUAGE, 0.9), (COGNITION, 0.5), (PATTERN, 0.4), (STRUCTURE, 0.3)] },
        Concept { label: "Oral tradition", category: "literature",  // BRIDGE: literature <-> music
            features: vec![(LANGUAGE, 0.8), (SOUND, 0.7), (EVOLUTION, 0.5), (PATTERN, 0.5)] },
        Concept { label: "Biographical writing", category: "literature",
            features: vec![(LANGUAGE, 0.8), (BEHAVIOR, 0.5), (EMOTION, 0.5), (COGNITION, 0.3)] },

        // ── History (15) ──────────────────────────────────────────────
        Concept { label: "Ancient history", category: "history",
            features: vec![(BEHAVIOR, 0.7), (SYSTEMS, 0.6), (EVOLUTION, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Medieval history", category: "history",
            features: vec![(BEHAVIOR, 0.6), (SYSTEMS, 0.7), (STRUCTURE, 0.4), (METAPHYSICS, 0.3)] },
        Concept { label: "Modern history", category: "history",
            features: vec![(BEHAVIOR, 0.7), (SYSTEMS, 0.7), (MARKETS, 0.5), (NETWORK, 0.3)] },
        Concept { label: "Military history", category: "history",
            features: vec![(FORCE, 0.7), (BEHAVIOR, 0.7), (SYSTEMS, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Economic history", category: "history",  // BRIDGE: history <-> economics
            features: vec![(MARKETS, 0.8), (BEHAVIOR, 0.6), (SYSTEMS, 0.6), (STATISTICS, 0.3)] },
        Concept { label: "Social history", category: "history",
            features: vec![(BEHAVIOR, 0.8), (NETWORK, 0.6), (SYSTEMS, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Intellectual history", category: "history",
            features: vec![(MIND, 0.6), (LANGUAGE, 0.6), (BEHAVIOR, 0.5), (EVOLUTION, 0.4)] },
        Concept { label: "Cultural history", category: "history",
            features: vec![(BEHAVIOR, 0.7), (EMOTION, 0.5), (PATTERN, 0.5), (LANGUAGE, 0.4)] },
        Concept { label: "Environmental history", category: "history",
            features: vec![(NATURE, 0.7), (SYSTEMS, 0.6), (BEHAVIOR, 0.5), (EVOLUTION, 0.4)] },
        Concept { label: "Historiography", category: "history",  // BRIDGE: history <-> philosophy
            features: vec![(LOGIC, 0.6), (LANGUAGE, 0.6), (STRUCTURE, 0.5), (METAPHYSICS, 0.4)] },
        Concept { label: "Archaeology", category: "history",  // BRIDGE: history <-> earth_science
            features: vec![(NATURE, 0.5), (STRUCTURE, 0.6), (EVOLUTION, 0.6), (PATTERN, 0.5)] },
        Concept { label: "Diplomatic history", category: "history",
            features: vec![(BEHAVIOR, 0.7), (NETWORK, 0.7), (LANGUAGE, 0.5), (SYSTEMS, 0.4)] },
        Concept { label: "History of technology", category: "history",  // BRIDGE: history <-> engineering
            features: vec![(BEHAVIOR, 0.5), (STRUCTURE, 0.5), (EVOLUTION, 0.6), (COMPUTATION, 0.3)] },
        Concept { label: "History of science", category: "history",  // BRIDGE: history <-> philosophy
            features: vec![(LOGIC, 0.5), (EVOLUTION, 0.6), (STRUCTURE, 0.4), (METAPHYSICS, 0.3)] },
        Concept { label: "Urban history", category: "history",
            features: vec![(BEHAVIOR, 0.6), (STRUCTURE, 0.6), (SYSTEMS, 0.5), (NETWORK, 0.4)] },

        // ── Sociology (15) ────────────────────────────────────────────
        Concept { label: "Social theory", category: "sociology",
            features: vec![(BEHAVIOR, 0.8), (SYSTEMS, 0.7), (METAPHYSICS, 0.4), (STRUCTURE, 0.5)] },
        Concept { label: "Urban sociology", category: "sociology",
            features: vec![(BEHAVIOR, 0.8), (STRUCTURE, 0.6), (SYSTEMS, 0.6), (NETWORK, 0.5)] },
        Concept { label: "Rural sociology", category: "sociology",
            features: vec![(BEHAVIOR, 0.7), (NATURE, 0.6), (SYSTEMS, 0.5), (NETWORK, 0.3)] },
        Concept { label: "Sociology of religion", category: "sociology",  // BRIDGE: sociology <-> religion
            features: vec![(BEHAVIOR, 0.7), (METAPHYSICS, 0.6), (SYSTEMS, 0.5), (NETWORK, 0.4)] },
        Concept { label: "Medical sociology", category: "sociology",  // BRIDGE: sociology <-> medicine
            features: vec![(BEHAVIOR, 0.7), (LIFE, 0.5), (DIAGNOSTICS, 0.4), (SYSTEMS, 0.5)] },
        Concept { label: "Gender studies", category: "sociology",
            features: vec![(BEHAVIOR, 0.8), (MIND, 0.5), (SYSTEMS, 0.5), (PATTERN, 0.4)] },
        Concept { label: "Criminology", category: "sociology",  // BRIDGE: sociology <-> law
            features: vec![(BEHAVIOR, 0.8), (LOGIC, 0.5), (STATISTICS, 0.5), (ETHICS, 0.5)] },
        Concept { label: "Social network analysis", category: "sociology",  // BRIDGE: sociology <-> CS
            features: vec![(NETWORK, 0.9), (BEHAVIOR, 0.7), (PATTERN, 0.6), (STATISTICS, 0.6)] },
        Concept { label: "Demography", category: "sociology",
            features: vec![(STATISTICS, 0.9), (BEHAVIOR, 0.6), (SYSTEMS, 0.5), (LIFE, 0.3)] },
        Concept { label: "Cultural sociology", category: "sociology",
            features: vec![(BEHAVIOR, 0.7), (EMOTION, 0.5), (LANGUAGE, 0.5), (PATTERN, 0.5)] },
        Concept { label: "Political sociology", category: "sociology",  // BRIDGE: sociology <-> political_science
            features: vec![(BEHAVIOR, 0.7), (SYSTEMS, 0.7), (MARKETS, 0.4), (NETWORK, 0.4)] },
        Concept { label: "Sociology of science", category: "sociology",  // BRIDGE: sociology <-> philosophy
            features: vec![(BEHAVIOR, 0.7), (LOGIC, 0.5), (STRUCTURE, 0.4), (SYSTEMS, 0.4)] },
        Concept { label: "Social movements research", category: "sociology",
            features: vec![(BEHAVIOR, 0.8), (NETWORK, 0.6), (EVOLUTION, 0.5), (EMOTION, 0.4)] },
        Concept { label: "Stratification research", category: "sociology",
            features: vec![(BEHAVIOR, 0.7), (MARKETS, 0.5), (SYSTEMS, 0.6), (STATISTICS, 0.5)] },
        Concept { label: "Ethnography", category: "sociology",
            features: vec![(BEHAVIOR, 0.8), (LANGUAGE, 0.5), (PATTERN, 0.5), (COGNITION, 0.3)] },

        // ── Political Science (15) ────────────────────────────────────
        Concept { label: "International relations", category: "political_science",
            features: vec![(BEHAVIOR, 0.8), (NETWORK, 0.7), (SYSTEMS, 0.6), (MARKETS, 0.3)] },
        Concept { label: "Comparative politics", category: "political_science",
            features: vec![(BEHAVIOR, 0.7), (SYSTEMS, 0.7), (PATTERN, 0.5), (STATISTICS, 0.4)] },
        Concept { label: "Political theory", category: "political_science",  // BRIDGE: political_science <-> philosophy
            features: vec![(ETHICS, 0.8), (METAPHYSICS, 0.5), (BEHAVIOR, 0.5), (SYSTEMS, 0.4)] },
        Concept { label: "Public administration", category: "political_science",
            features: vec![(SYSTEMS, 0.8), (BEHAVIOR, 0.6), (OPTIMIZATION, 0.5), (STRUCTURE, 0.4)] },
        Concept { label: "Public policy", category: "political_science",
            features: vec![(BEHAVIOR, 0.7), (SYSTEMS, 0.7), (ETHICS, 0.5), (OPTIMIZATION, 0.5)] },
        Concept { label: "Electoral studies", category: "political_science",
            features: vec![(BEHAVIOR, 0.8), (STATISTICS, 0.7), (NETWORK, 0.4), (PATTERN, 0.3)] },
        Concept { label: "Political economy", category: "political_science",  // BRIDGE: political_science <-> economics
            features: vec![(MARKETS, 0.8), (BEHAVIOR, 0.7), (SYSTEMS, 0.6), (OPTIMIZATION, 0.4)] },
        Concept { label: "Geopolitics", category: "political_science",
            features: vec![(BEHAVIOR, 0.7), (SPACE, 0.5), (SYSTEMS, 0.6), (NETWORK, 0.5)] },
        Concept { label: "Security studies", category: "political_science",
            features: vec![(BEHAVIOR, 0.7), (FORCE, 0.5), (SYSTEMS, 0.6), (NETWORK, 0.3)] },
        Concept { label: "Democratic theory", category: "political_science",
            features: vec![(ETHICS, 0.7), (BEHAVIOR, 0.6), (SYSTEMS, 0.6), (LOGIC, 0.4)] },
        Concept { label: "Nationalism studies", category: "political_science",
            features: vec![(BEHAVIOR, 0.7), (EMOTION, 0.6), (LANGUAGE, 0.4), (NETWORK, 0.3)] },
        Concept { label: "Peace studies", category: "political_science",
            features: vec![(BEHAVIOR, 0.7), (ETHICS, 0.7), (NETWORK, 0.4), (SYSTEMS, 0.3)] },
        Concept { label: "Political psychology", category: "political_science",  // BRIDGE: political_science <-> psychology
            features: vec![(MIND, 0.6), (BEHAVIOR, 0.8), (COGNITION, 0.5), (EMOTION, 0.4)] },
        Concept { label: "Constitutional studies", category: "political_science",  // BRIDGE: political_science <-> law
            features: vec![(LOGIC, 0.7), (ETHICS, 0.6), (STRUCTURE, 0.6), (LANGUAGE, 0.5)] },
        Concept { label: "Global governance", category: "political_science",
            features: vec![(SYSTEMS, 0.8), (NETWORK, 0.7), (ETHICS, 0.5), (BEHAVIOR, 0.5)] },

        // ── Law (15) ──────────────────────────────────────────────────
        Concept { label: "Constitutional law", category: "law",
            features: vec![(LOGIC, 0.8), (ETHICS, 0.6), (LANGUAGE, 0.7), (STRUCTURE, 0.6)] },
        Concept { label: "Criminal law", category: "law",
            features: vec![(LOGIC, 0.8), (ETHICS, 0.7), (BEHAVIOR, 0.6), (LANGUAGE, 0.4)] },
        Concept { label: "Contract law", category: "law",  // BRIDGE: law <-> economics
            features: vec![(LOGIC, 0.8), (LANGUAGE, 0.7), (MARKETS, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Tort law", category: "law",
            features: vec![(LOGIC, 0.8), (ETHICS, 0.6), (BEHAVIOR, 0.5)] },
        Concept { label: "International law", category: "law",
            features: vec![(LOGIC, 0.7), (ETHICS, 0.6), (SYSTEMS, 0.6), (NETWORK, 0.5)] },
        Concept { label: "Environmental law", category: "law",  // BRIDGE: law <-> earth_science
            features: vec![(LOGIC, 0.7), (NATURE, 0.7), (ETHICS, 0.6), (SYSTEMS, 0.4)] },
        Concept { label: "Intellectual property law", category: "law",  // BRIDGE: law <-> CS
            features: vec![(LOGIC, 0.7), (INFORMATION, 0.8), (MARKETS, 0.5), (ETHICS, 0.4)] },
        Concept { label: "Corporate law", category: "law",  // BRIDGE: law <-> economics
            features: vec![(LOGIC, 0.7), (MARKETS, 0.8), (SYSTEMS, 0.5), (STRUCTURE, 0.4)] },
        Concept { label: "Jurisprudence", category: "law",  // BRIDGE: law <-> philosophy
            features: vec![(LOGIC, 0.8), (ETHICS, 0.7), (METAPHYSICS, 0.6), (LANGUAGE, 0.5)] },
        Concept { label: "Family law", category: "law",
            features: vec![(LOGIC, 0.7), (BEHAVIOR, 0.6), (ETHICS, 0.6), (EMOTION, 0.4)] },
        Concept { label: "Tax law", category: "law",
            features: vec![(LOGIC, 0.8), (MARKETS, 0.7), (MATH, 0.5), (STRUCTURE, 0.5)] },
        Concept { label: "Administrative law", category: "law",
            features: vec![(LOGIC, 0.7), (SYSTEMS, 0.7), (BEHAVIOR, 0.4), (STRUCTURE, 0.4)] },
        Concept { label: "Human rights law", category: "law",
            features: vec![(ETHICS, 0.9), (BEHAVIOR, 0.5), (LANGUAGE, 0.5), (NETWORK, 0.4)] },
        Concept { label: "Antitrust law", category: "law",  // BRIDGE: law <-> economics
            features: vec![(MARKETS, 0.8), (LOGIC, 0.7), (BEHAVIOR, 0.5), (ETHICS, 0.4)] },
        Concept { label: "Cyberlaw", category: "law",  // BRIDGE: law <-> CS
            features: vec![(LOGIC, 0.7), (INFORMATION, 0.8), (COMPUTATION, 0.6), (ETHICS, 0.4)] },

        // ── Architecture (15) ─────────────────────────────────────────
        Concept { label: "Architectural design", category: "architecture",
            features: vec![(STRUCTURE, 0.9), (SPACE, 0.8), (PATTERN, 0.6), (EMOTION, 0.5)] },
        Concept { label: "Urban planning", category: "architecture",  // BRIDGE: architecture <-> sociology
            features: vec![(STRUCTURE, 0.7), (SPACE, 0.7), (SYSTEMS, 0.8), (BEHAVIOR, 0.5)] },
        Concept { label: "Landscape architecture", category: "architecture",
            features: vec![(NATURE, 0.7), (STRUCTURE, 0.6), (SPACE, 0.7), (PATTERN, 0.5)] },
        Concept { label: "Interior design", category: "architecture",
            features: vec![(STRUCTURE, 0.6), (SPACE, 0.7), (EMOTION, 0.7), (PATTERN, 0.5)] },
        Concept { label: "Sustainable architecture", category: "architecture",
            features: vec![(STRUCTURE, 0.7), (NATURE, 0.7), (ENERGY, 0.6), (ETHICS, 0.5)] },
        Concept { label: "Architectural history", category: "architecture",  // BRIDGE: architecture <-> history
            features: vec![(STRUCTURE, 0.6), (PATTERN, 0.5), (EVOLUTION, 0.5), (LANGUAGE, 0.4)] },
        Concept { label: "Architectural theory", category: "architecture",  // BRIDGE: architecture <-> philosophy
            features: vec![(STRUCTURE, 0.6), (METAPHYSICS, 0.5), (EMOTION, 0.5), (MIND, 0.4)] },
        Concept { label: "Parametric design", category: "architecture",  // BRIDGE: architecture <-> CS
            features: vec![(STRUCTURE, 0.8), (MATH, 0.7), (COMPUTATION, 0.6), (PATTERN, 0.7)] },
        Concept { label: "Building information modeling", category: "architecture",  // BRIDGE: architecture <-> CS
            features: vec![(STRUCTURE, 0.7), (COMPUTATION, 0.7), (INFORMATION, 0.7), (SYSTEMS, 0.4)] },
        Concept { label: "Historic preservation", category: "architecture",
            features: vec![(STRUCTURE, 0.7), (PATTERN, 0.5), (EVOLUTION, 0.6), (ETHICS, 0.3)] },
        Concept { label: "Vernacular architecture", category: "architecture",
            features: vec![(STRUCTURE, 0.7), (BEHAVIOR, 0.5), (NATURE, 0.5), (LANGUAGE, 0.3)] },
        Concept { label: "Biophilic design", category: "architecture",
            features: vec![(STRUCTURE, 0.7), (NATURE, 0.7), (EMOTION, 0.6), (MIND, 0.4)] },
        Concept { label: "Acoustic architecture", category: "architecture",  // BRIDGE: architecture <-> music
            features: vec![(STRUCTURE, 0.7), (SOUND, 0.8), (WAVE, 0.7), (MATH, 0.3)] },
        Concept { label: "Sacred architecture", category: "architecture",  // BRIDGE: architecture <-> religion
            features: vec![(STRUCTURE, 0.8), (METAPHYSICS, 0.7), (EMOTION, 0.7), (SPACE, 0.6)] },
        Concept { label: "Building physics", category: "architecture",  // BRIDGE: architecture <-> physics
            features: vec![(STRUCTURE, 0.6), (ENERGY, 0.7), (WAVE, 0.5), (FORCE, 0.4)] },

        // ── Culinary Arts (15) ────────────────────────────────────────
        Concept { label: "Cooking techniques", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.7), (ENERGY, 0.6), (PERFORMANCE, 0.7), (PATTERN, 0.4)] },
        Concept { label: "Baking", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.8), (PATTERN, 0.6), (STRUCTURE, 0.5), (PERFORMANCE, 0.6)] },
        Concept { label: "Pastry arts", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.7), (STRUCTURE, 0.6), (PATTERN, 0.7), (PERFORMANCE, 0.7)] },
        Concept { label: "Molecular gastronomy", category: "culinary_arts",  // BRIDGE: culinary_arts <-> chemistry
            features: vec![(CHEMISTRY, 0.9), (STRUCTURE, 0.6), (PATTERN, 0.5), (PERFORMANCE, 0.5)] },
        Concept { label: "Fermentation science", category: "culinary_arts",  // BRIDGE: culinary_arts <-> biology
            features: vec![(CHEMISTRY, 0.7), (LIFE, 0.7), (EVOLUTION, 0.4), (PATTERN, 0.4)] },
        Concept { label: "Food chemistry", category: "culinary_arts",  // BRIDGE: culinary_arts <-> chemistry
            features: vec![(CHEMISTRY, 0.9), (STRUCTURE, 0.5), (PATTERN, 0.5), (ENERGY, 0.3)] },
        Concept { label: "Nutrition science", category: "culinary_arts",  // BRIDGE: culinary_arts <-> medicine
            features: vec![(LIFE, 0.7), (CHEMISTRY, 0.6), (STATISTICS, 0.5), (DIAGNOSTICS, 0.4)] },
        Concept { label: "Oenology", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.6), (PATTERN, 0.6), (EMOTION, 0.7), (COGNITION, 0.4)] },
        Concept { label: "Pastry decoration", category: "culinary_arts",
            features: vec![(PATTERN, 0.8), (PERFORMANCE, 0.7), (EMOTION, 0.6), (STRUCTURE, 0.4)] },
        Concept { label: "Culinary history", category: "culinary_arts",  // BRIDGE: culinary_arts <-> history
            features: vec![(BEHAVIOR, 0.6), (PATTERN, 0.5), (EVOLUTION, 0.5), (LANGUAGE, 0.3)] },
        Concept { label: "Food science", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.7), (PATTERN, 0.5), (STATISTICS, 0.4), (STRUCTURE, 0.5)] },
        Concept { label: "Plating and presentation", category: "culinary_arts",
            features: vec![(PATTERN, 0.8), (PERFORMANCE, 0.7), (EMOTION, 0.5), (STRUCTURE, 0.4)] },
        Concept { label: "Brewing science", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.8), (LIFE, 0.6), (PATTERN, 0.5), (PERFORMANCE, 0.4)] },
        Concept { label: "Sommelier studies", category: "culinary_arts",
            features: vec![(CHEMISTRY, 0.5), (EMOTION, 0.7), (PATTERN, 0.6), (COGNITION, 0.5)] },
        Concept { label: "Traditional cuisine studies", category: "culinary_arts",
            features: vec![(LANGUAGE, 0.4), (BEHAVIOR, 0.6), (PATTERN, 0.5), (EVOLUTION, 0.4)] },

        // ── Religion / Theology (15) ──────────────────────────────────
        Concept { label: "Systematic theology", category: "religion",
            features: vec![(METAPHYSICS, 0.9), (LOGIC, 0.6), (ETHICS, 0.6), (LANGUAGE, 0.4)] },
        Concept { label: "Comparative religion", category: "religion",
            features: vec![(METAPHYSICS, 0.7), (BEHAVIOR, 0.6), (LANGUAGE, 0.5), (PATTERN, 0.5)] },
        Concept { label: "Biblical studies", category: "religion",
            features: vec![(LANGUAGE, 0.8), (METAPHYSICS, 0.7), (PATTERN, 0.4), (STRUCTURE, 0.3)] },
        Concept { label: "Islamic studies", category: "religion",
            features: vec![(METAPHYSICS, 0.7), (LANGUAGE, 0.7), (BEHAVIOR, 0.5), (ETHICS, 0.5)] },
        Concept { label: "Buddhist studies", category: "religion",
            features: vec![(METAPHYSICS, 0.8), (MIND, 0.7), (BEHAVIOR, 0.5), (COGNITION, 0.5)] },
        Concept { label: "Religious ethics", category: "religion",  // BRIDGE: religion <-> philosophy
            features: vec![(ETHICS, 0.9), (METAPHYSICS, 0.7), (BEHAVIOR, 0.5)] },
        Concept { label: "Philosophy of religion", category: "religion",  // BRIDGE: religion <-> philosophy
            features: vec![(METAPHYSICS, 0.9), (LOGIC, 0.7), (MIND, 0.5), (ETHICS, 0.4)] },
        Concept { label: "Church history", category: "religion",  // BRIDGE: religion <-> history
            features: vec![(BEHAVIOR, 0.6), (SYSTEMS, 0.5), (EVOLUTION, 0.6), (METAPHYSICS, 0.5)] },
        Concept { label: "Mysticism", category: "religion",
            features: vec![(METAPHYSICS, 0.9), (MIND, 0.8), (EMOTION, 0.7), (COGNITION, 0.5)] },
        Concept { label: "Theological anthropology", category: "religion",
            features: vec![(METAPHYSICS, 0.7), (LIFE, 0.4), (BEHAVIOR, 0.5), (MIND, 0.6)] },
        Concept { label: "Liturgical studies", category: "religion",  // BRIDGE: religion <-> music
            features: vec![(METAPHYSICS, 0.6), (PERFORMANCE, 0.8), (PATTERN, 0.6), (EMOTION, 0.5)] },
        Concept { label: "Eschatology", category: "religion",
            features: vec![(METAPHYSICS, 0.9), (MIND, 0.5), (EMOTION, 0.4), (LOGIC, 0.4)] },
        Concept { label: "Homiletics", category: "religion",
            features: vec![(LANGUAGE, 0.7), (PERFORMANCE, 0.7), (METAPHYSICS, 0.6), (EMOTION, 0.5)] },
        Concept { label: "Patristics", category: "religion",
            features: vec![(LANGUAGE, 0.7), (METAPHYSICS, 0.7), (PATTERN, 0.4), (EVOLUTION, 0.4)] },
        Concept { label: "Religious education", category: "religion",
            features: vec![(COGNITION, 0.5), (METAPHYSICS, 0.7), (BEHAVIOR, 0.5), (LANGUAGE, 0.4)] },
    ]
}
fn main() {
    println!("================================================================");
    println!("  SphereQL: AI Knowledge Navigator");
    println!("  Category Enrichment for Cross-Domain Reasoning");
    println!("================================================================\n");

    // ── Build corpus ──────────────────────────────────────────────────
    let corpus = build_corpus();
    let n = corpus.len();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();
    let labels: Vec<&str> = corpus.iter().map(|c| c.label).collect();

    println!("Corpus: {} concepts across 8 knowledge domains\n", n);

    let pipeline = SphereQLPipeline::new(PipelineInput {
        categories: categories.clone(),
        embeddings: embeddings.clone(),
    })
    .expect("pipeline build failed");

    let evr = pipeline.explained_variance_ratio();
    println!(
        "Projection quality: {:.1}% variance explained (EVR={:.4})\n",
        evr * 100.0,
        evr
    );

    // ==================================================================
    // ANALYSIS 1: Category Landscape
    // ==================================================================
    println!("────────────────────────────────────────────────────────────────");
    println!("  1. CATEGORY LANDSCAPE");
    println!("     How tightly clustered is each knowledge domain?");
    println!("────────────────────────────────────────────────────────────────\n");

    let layer = pipeline.category_layer();
    println!(
        "  {:<20} {:>5} {:>12} {:>10}",
        "Domain", "Items", "Spread (deg)", "Cohesion"
    );
    println!("  {}", "-".repeat(50));

    for summary in &layer.summaries {
        println!(
            "  {:<20} {:>5} {:>12.2} {:>10.4}",
            summary.name,
            summary.member_count,
            summary.angular_spread.to_degrees(),
            summary.cohesion,
        );
    }

    // Identify most and least cohesive
    let most_cohesive = layer
        .summaries
        .iter()
        .max_by(|a, b| a.cohesion.partial_cmp(&b.cohesion).unwrap())
        .unwrap();
    let least_cohesive = layer
        .summaries
        .iter()
        .min_by(|a, b| a.cohesion.partial_cmp(&b.cohesion).unwrap())
        .unwrap();
    println!(
        "\n  -> Most cohesive: {} ({:.4}) -- tightly defined domain",
        most_cohesive.name, most_cohesive.cohesion
    );
    println!(
        "  -> Least cohesive: {} ({:.4}) -- broad, diffuse domain",
        least_cohesive.name, least_cohesive.cohesion
    );
    println!("     (AI should express more uncertainty about diffuse domains)");

    // ==================================================================
    // ANALYSIS 2: Inter-Category Graph
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  2. KNOWLEDGE DOMAIN ADJACENCY GRAPH");
    println!("     Which domains are nearest neighbors on the sphere?");
    println!("────────────────────────────────────────────────────────────────\n");

    for summary in &layer.summaries {
        let neighbors = layer.category_neighbors(&summary.name, 3);
        let neighbor_strs: Vec<String> = neighbors
            .iter()
            .zip(layer.graph.adjacency[layer.name_to_index[&summary.name]].iter())
            .map(|(n, edge)| format!("{} ({:.2} rad)", n.name, edge.centroid_distance))
            .collect();
        println!("  {:<20} -> {}", summary.name, neighbor_strs.join(", "));
    }

    println!("\n  Key insight: these adjacencies reveal the STRUCTURE of knowledge.");
    println!("  An AI can use this graph to understand which domains share");
    println!("  conceptual foundations, even without being explicitly told.");

    // ==================================================================
    // ANALYSIS 3: Bridge Concept Detection
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  3. BRIDGE CONCEPTS -- Cross-Domain Connectors");
    println!("     Concepts that span two knowledge domains");
    println!("────────────────────────────────────────────────────────────────\n");

    // Check specific cross-domain bridges we expect to find
    let bridge_queries: Vec<(&str, &str)> = vec![
        ("physics", "computer_science"),
        ("physics", "music"),
        ("biology", "computer_science"),
        ("biology", "philosophy"),
        ("computer_science", "economics"),
        ("computer_science", "linguistics"),
        ("philosophy", "medicine"),
        ("philosophy", "economics"),
        ("music", "linguistics"),
        ("medicine", "biology"),
    ];

    for (src, tgt) in &bridge_queries {
        let bridges = pipeline.bridge_items(src, tgt, 3);
        if bridges.is_empty() {
            // Try reverse direction
            let rev_bridges = pipeline.bridge_items(tgt, src, 3);
            if rev_bridges.is_empty() {
                println!("  {} <-> {}: (no bridges -- conceptual gap)", src, tgt);
            } else {
                print!("  {} <-> {}:", src, tgt);
                for (i, b) in rev_bridges.iter().enumerate() {
                    let sep = if i == 0 { " " } else { "                        " };
                    println!(
                        "{}\"{}\" (strength={:.3})",
                        sep, labels[b.item_index], b.bridge_strength
                    );
                }
            }
        } else {
            print!("  {} <-> {}:", src, tgt);
            for (i, b) in bridges.iter().enumerate() {
                let sep = if i == 0 { " " } else { &" ".repeat(src.len() + tgt.len() + 7) };
                println!(
                    "{}\"{}\" (strength={:.3})",
                    sep, labels[b.item_index], b.bridge_strength
                );
            }
        }
    }

    println!("\n  AI use case: when asked \"What connects physics to music?\",");
    println!("  the model queries bridge_items and gets concrete concepts");
    println!("  like acoustics -- not a vague guess, but a precise connector.");

    // ==================================================================
    // ANALYSIS 4: Cross-Domain Concept Paths
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  4. CONCEPT PATH TRAVERSAL -- Reasoning Chains");
    println!("     How to get from one domain to another, step by step");
    println!("────────────────────────────────────────────────────────────────");

    let path_queries: Vec<(&str, &str, &str)> = vec![
        ("music", "economics", "How does music relate to economic systems?"),
        ("linguistics", "medicine", "How does language connect to health?"),
        ("philosophy", "computer_science", "From abstract thought to computation"),
        ("biology", "economics", "From living systems to market systems"),
    ];

    for (src, tgt, question) in &path_queries {
        println!("\n  Question: \"{question}\"");
        println!("  Path: {src} -> {tgt}\n");

        if let Some(path) = pipeline.category_path(src, tgt) {
            for (i, step) in path.steps.iter().enumerate() {
                let is_last = i + 1 >= path.steps.len();
                println!(
                    "    [{}] {} (cumulative distance: {:.3})",
                    i + 1,
                    step.category_name,
                    step.cumulative_distance,
                );

                if !is_last {
                    let bridge_descs: Vec<String> = step
                        .bridges_to_next
                        .iter()
                        .take(2)
                        .map(|b| format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength))
                        .collect();
                    if !bridge_descs.is_empty() {
                        println!("         bridged by: {}", bridge_descs.join(", "));
                    } else {
                        println!("         (direct adjacency, no bridge items)");
                    }
                    println!("         |");
                }
            }
            println!("    Total distance: {:.3}\n", path.total_distance);
        } else {
            println!("    (no path found)\n");
        }
    }

    println!("  AI use case: these paths are REASONING CHAINS. The model can");
    println!("  construct explanations by walking the path and using each");
    println!("  bridge concept as an explanatory link between domains.");

    // ==================================================================
    // ANALYSIS 5: Gap Detection via Glob Analysis
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  5. KNOWLEDGE DENSITY -- Glob Detection");
    println!("     Where is knowledge concentrated? Where are the gaps?");
    println!("────────────────────────────────────────────────────────────────\n");

    let dummy_q = PipelineQuery {
        embedding: vec![0.0; DIM],
    };
    let glob_result = pipeline.query(
        SphereQLQuery::DetectGlobs {
            k: None,
            max_k: 10,
        },
        &dummy_q,
    );

    if let SphereQLOutput::Globs(globs) = glob_result {
        println!("  Detected {} knowledge clusters on the sphere:\n", globs.len());
        for g in &globs {
            let cats: Vec<String> = g
                .top_categories
                .iter()
                .map(|(c, n)| format!("{} ({})", c, n))
                .collect();
            let density = if g.radius > 0.0 {
                g.member_count as f64 / (std::f64::consts::PI * g.radius * g.radius)
            } else {
                f64::INFINITY
            };
            println!(
                "  Glob {}: {} members, radius={:.3} rad, density={:.1}",
                g.id, g.member_count, g.radius, density
            );
            println!("    Domains: {}", cats.join(", "));
        }

        println!("\n  AI use case: sparse globs indicate thin coverage areas.");
        println!("  The model should be MORE CAUTIOUS in regions with low density");
        println!("  and can flag: \"I may have limited knowledge in this area.\"");
    }

    // ==================================================================
    // ANALYSIS 6: Drill-Down for Sub-Topic Precision
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  6. DRILL-DOWN -- Sub-Topic Precision");
    println!("     Zooming into a category for fine-grained retrieval");
    println!("────────────────────────────────────────────────────────────────\n");

    // Query: "quantum computing" sits between physics and CS
    let quantum_computing = embed(
        &[
            (QUANTUM, 0.8),
            (COMPUTATION, 0.7),
            (MATH, 0.6),
            (INFORMATION, 0.5),
            (LOGIC, 0.3),
        ],
        9999,
    );
    let qc_query = PipelineQuery {
        embedding: quantum_computing.clone(),
    };

    println!("  Query concept: \"quantum computing\" (novel, cross-domain)\n");

    // Which categories is this query nearest to?
    let near_cats = pipeline.query(SphereQLQuery::CategoryStats, &qc_query);
    if let SphereQLOutput::CategoryStats { summaries, .. } = &near_cats {
        // Compute distances to each category centroid manually via the layer
        let emb = sphereql::embed::Embedding::new(quantum_computing.clone());
        let nearby = layer.categories_near_embedding(&emb, pipeline.pca(), std::f64::consts::PI);
        println!("  Distance to each domain centroid:");
        for (ci, dist) in nearby.iter().take(8) {
            println!(
                "    {:<20} {:.3} rad ({:.1} deg)",
                summaries[*ci].name,
                dist,
                dist.to_degrees()
            );
        }
    }

    // Drill down into physics
    println!("\n  Drill-down into PHYSICS (top 5 nearest sub-topics):");
    let drill_physics = pipeline.query(
        SphereQLQuery::DrillDown {
            category: "physics",
            k: 5,
        },
        &qc_query,
    );
    if let SphereQLOutput::DrillDown(results) = drill_physics {
        for (i, r) in results.iter().enumerate() {
            let sphere_tag = if r.used_inner_sphere { "inner" } else { "outer" };
            println!(
                "    {}. \"{}\" (dist={:.4}, {})",
                i + 1,
                labels[r.item_index],
                r.distance,
                sphere_tag,
            );
        }
    }

    // Drill down into CS
    println!("\n  Drill-down into COMPUTER SCIENCE (top 5 nearest sub-topics):");
    let drill_cs = pipeline.query(
        SphereQLQuery::DrillDown {
            category: "computer_science",
            k: 5,
        },
        &qc_query,
    );
    if let SphereQLOutput::DrillDown(results) = drill_cs {
        for (i, r) in results.iter().enumerate() {
            let sphere_tag = if r.used_inner_sphere { "inner" } else { "outer" };
            println!(
                "    {}. \"{}\" (dist={:.4}, {})",
                i + 1,
                labels[r.item_index],
                r.distance,
                sphere_tag,
            );
        }
    }

    println!("\n  AI use case: for a cross-domain question like \"quantum computing\",");
    println!("  the model identifies WHICH domains are relevant, then drills into");
    println!("  each to find the most pertinent sub-topics for its answer.");

    // ==================================================================
    // ANALYSIS 7: Assembled Reasoning Chain
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  7. ASSEMBLED REASONING -- Putting It All Together");
    println!("     Simulating how an AI would answer a cross-domain question");
    println!("────────────────────────────────────────────────────────────────\n");

    let question = "How does music relate to economics?";
    println!("  USER QUESTION: \"{}\"\n", question);
    println!("  AI's spatial reasoning process:\n");

    // Step 1: Find the path
    println!("  Step 1: Query category path music -> economics");
    if let Some(path) = pipeline.category_path("music", "economics") {
        let domain_chain: Vec<&str> = path.steps.iter().map(|s| s.category_name.as_str()).collect();
        println!("    Path found: {}", domain_chain.join(" -> "));
        println!("    Total semantic distance: {:.3}\n", path.total_distance);

        // Step 2: Gather bridge concepts along each edge
        println!("  Step 2: Identify bridge concepts at each transition");
        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 3);
                let rev_bridges = pipeline.bridge_items(&next.category_name, &step.category_name, 3);
                let all_bridge_labels: Vec<&str> = bridges
                    .iter()
                    .chain(rev_bridges.iter())
                    .map(|b| labels[b.item_index])
                    .collect();
                if all_bridge_labels.is_empty() {
                    println!(
                        "    {} -> {}: (adjacent but no specific bridge)",
                        step.category_name, next.category_name
                    );
                } else {
                    println!(
                        "    {} -> {}: via {}",
                        step.category_name,
                        next.category_name,
                        all_bridge_labels.join(", ")
                    );
                }
            }
        }

        // Step 3: Check cohesion to calibrate confidence
        println!("\n  Step 3: Assess confidence via domain cohesion");
        for step in &path.steps {
            if let Some(summary) = layer.get_category(&step.category_name) {
                let confidence = if summary.cohesion > 0.8 {
                    "HIGH"
                } else if summary.cohesion > 0.6 {
                    "MODERATE"
                } else {
                    "LOW"
                };
                println!(
                    "    {} -- cohesion={:.3}, confidence={}",
                    step.category_name, summary.cohesion, confidence
                );
            }
        }

        // Step 4: Synthesize a narrative
        println!("\n  Step 4: Synthesized answer\n");
        println!("    \"Music and economics, while seemingly distant fields, are");
        println!("    connected through a chain of shared conceptual foundations.");

        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 1);
                let rev_bridges = pipeline.bridge_items(&next.category_name, &step.category_name, 1);
                let bridge_label = bridges
                    .first()
                    .or(rev_bridges.first())
                    .map(|b| labels[b.item_index])
                    .unwrap_or("shared foundations");
                println!(
                    "    {} connects to {} through concepts like {}.",
                    step.category_name, next.category_name, bridge_label
                );
            }
        }
        println!("    The total semantic distance of {:.3} (out of a maximum", path.total_distance);
        println!("    of pi = {:.3}) indicates these fields are {}.\"",
            std::f64::consts::PI,
            if path.total_distance < 1.0 { "surprisingly close" }
            else if path.total_distance < 2.0 { "moderately connected" }
            else { "quite distant" }
        );
    }

    // ── Inner sphere stats ────────────────────────────────────────────
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  APPENDIX: Inner Sphere Status");
    println!("────────────────────────────────────────────────────────────────\n");

    let stats = pipeline.inner_sphere_stats();
    if stats.is_empty() {
        println!("  No inner spheres materialized (categories have < 20 items).");
        println!("  With a real corpus (50+ items per domain), inner spheres would");
        println!("  automatically activate for domains where within-category PCA");
        println!("  captures significantly more variance than the global projection.");
    } else {
        for s in &stats {
            println!(
                "  {} -- {} ({} members, inner EVR={:.3}, improvement={:.3})",
                s.category_name,
                s.projection_type,
                s.member_count,
                s.inner_evr,
                s.evr_improvement,
            );
        }
    }

    println!("\n================================================================");
    println!("  Demo complete. {} concepts, {} categories, EVR={:.1}%",
        n,
        pipeline.num_categories(),
        pipeline.explained_variance_ratio() * 100.0,
    );
    println!("================================================================");
}
