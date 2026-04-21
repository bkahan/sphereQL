#![allow(clippy::uninlined_format_args)]
//! AI Knowledge Navigator — Category Enrichment Demo
//!
//! Demonstrates how sphereQL's Category Enrichment Layer could help an AI
//! model reason about cross-domain connections. The corpus simulates an AI's
//! knowledge across 31 academic domains with deliberately placed "bridge
//! concepts" that span multiple fields.
//!
//! The demo runs 13 analyses:
//!   1.  Category landscape (cohesion, spread, centroid positions)
//!   2.  Sphere geometry — centroid map with angular coordinates
//!   3.  Inter-category adjacency graph with edge weight decomposition
//!   4.  Bridge density analysis — most-connected category pairs
//!   5.  Bridge concept detection — specific cross-domain connectors
//!   6.  Category boundary analysis — "ambassador" items between domains
//!   7.  Cross-domain concept path traversal (category-level Dijkstra)
//!   8.  Item-level concept paths through the k-NN graph
//!   9.  Gap detection via glob analysis
//!  10.  Multi-query category routing — how queries get dispatched
//!  11.  Inner-sphere drill-down with precision comparison
//!  12.  Nearest-neighbor retrieval with projection quality metadata
//!  13.  Assembled reasoning chain from spatial structure
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

const DIM: usize = 128;

// ── Physics ──────────────────────────────────────────────────────────────
const ENERGY: usize = 0;
const FORCE: usize = 1;
const QUANTUM: usize = 2;
const WAVE: usize = 3;
const ENTROPY: usize = 4;
const RELATIVITY: usize = 5;
const PARTICLE: usize = 6;

// ── Mathematics ──────────────────────────────────────────────────────────
const MATH: usize = 7;
const PROOF: usize = 8;
const CALCULUS: usize = 9;
const GRAPH_THEORY: usize = 10;
const ALGEBRA: usize = 11;

// ── Biology ──────────────────────────────────────────────────────────────
const LIFE: usize = 12;
const EVOLUTION: usize = 13;
const GENETICS: usize = 14;
const CELLULAR: usize = 15;

// ── Chemistry ────────────────────────────────────────────────────────────
const CHEMISTRY: usize = 16;
const MOLECULAR: usize = 17;
const REACTION: usize = 18;

// ── Medicine ─────────────────────────────────────────────────────────────
const DIAGNOSTICS: usize = 19;
const THERAPY: usize = 20;
const ANATOMY: usize = 21;
const CLINICAL: usize = 22;

// ── Neuroscience ─────────────────────────────────────────────────────────
const NEURAL: usize = 23;
const BRAIN: usize = 24;
const CONSCIOUSNESS: usize = 25;

// ── Computer Science ─────────────────────────────────────────────────────
const COMPUTATION: usize = 26;
const LOGIC: usize = 27;
const SOFTWARE: usize = 28;
const ALGORITHM: usize = 29;

// ── AI / Data Science ────────────────────────────────────────────────────
const AI: usize = 30;
const LLM: usize = 31;
const DATA: usize = 32;
const MACHINE_LEARN: usize = 33;

// ── Engineering ──────────────────────────────────────────────────────────
const MECHANICAL: usize = 34;
const ELECTRICAL: usize = 35;
const MATERIAL: usize = 36;
const TRANSPORTATION: usize = 37;

// ── Nanotechnology ───────────────────────────────────────────────────────
const NANO: usize = 38;
const ATOMIC: usize = 39;
const SURFACE: usize = 40;

// ── Astronomy ────────────────────────────────────────────────────────────
const CELESTIAL: usize = 41;
const STELLAR: usize = 42;
const PLANETARY: usize = 43;
const ORBIT: usize = 44;

// ── Earth Science ────────────────────────────────────────────────────────
const GEOLOGY: usize = 45;
const CLIMATE: usize = 46;
const OCEAN: usize = 47;
const WATER: usize = 48;

// ── Environmental Science ────────────────────────────────────────────────
const ECOSYSTEM: usize = 49;
const CONSERVATION: usize = 50;
const NATURE: usize = 51;

// ── Psychology ───────────────────────────────────────────────────────────
const ATTACHMENT: usize = 52;
const TRAUMA: usize = 53;
const MENTAL_HEALTH: usize = 54;

// ── Philosophy ───────────────────────────────────────────────────────────
const ETHICS: usize = 55;
const METAPHYSICS: usize = 56;
const EPISTEMOLOGY: usize = 57;
const ONTOLOGY: usize = 58;

// ── Religion ─────────────────────────────────────────────────────────────
const SPIRITUAL: usize = 59;
const RITUAL: usize = 60;
const SACRED: usize = 61;
const DOCTRINE: usize = 62;

// ── Linguistics ──────────────────────────────────────────────────────────
const LANGUAGE: usize = 63;
const GRAMMAR: usize = 64;
const PHONETIC: usize = 65;
const SYNTAX: usize = 66;
const SEMANTICS_AX: usize = 67;

// ── Literature ───────────────────────────────────────────────────────────
const NARRATIVE: usize = 68;
const LITERARY: usize = 69;
const POETRY: usize = 70;

// ── History ──────────────────────────────────────────────────────────────
const HISTORICAL: usize = 71;
const ARCHIVAL: usize = 72;

// ── Sociology ────────────────────────────────────────────────────────────
const SOCIETY: usize = 73;
const COMMUNITY: usize = 74;
const SOCIAL_NETWORK: usize = 75;

// ── Anthropology ─────────────────────────────────────────────────────────
const CULTURE: usize = 76;
const TRADITION: usize = 77;
const KINSHIP: usize = 78;

// ── Political Science ────────────────────────────────────────────────────
const GOVERNANCE: usize = 79;
const POWER: usize = 80;
const POLICY: usize = 81;

// ── Law ──────────────────────────────────────────────────────────────────
const LEGAL: usize = 82;
const JUSTICE: usize = 83;
const RIGHTS: usize = 84;

// ── Economics ────────────────────────────────────────────────────────────
const MARKETS: usize = 85;
const FINANCE: usize = 86;
const LABOR: usize = 87;
const MONEY: usize = 88;

// ── Education ────────────────────────────────────────────────────────────
const PEDAGOGY: usize = 89;
const CURRICULUM: usize = 90;
const ASSESSMENT: usize = 91;

// ── Visual Arts ──────────────────────────────────────────────────────────
const VISUAL: usize = 92;
const COLOR: usize = 93;
const FORM: usize = 94;
const DESIGN: usize = 95;

// ── Music ────────────────────────────────────────────────────────────────
const SOUND: usize = 96;
const HARMONY: usize = 97;
const RHYTHM: usize = 98;
const TIMBRE: usize = 99;

// ── Film ─────────────────────────────────────────────────────────────────
const CINEMA: usize = 100;
const MONTAGE: usize = 101;

// ── Performing Arts ──────────────────────────────────────────────────────
const THEATRICAL: usize = 102;
const DANCE: usize = 103;

// ── Culinary Arts ────────────────────────────────────────────────────────
const TASTE: usize = 104;
const FLAVOR: usize = 105;
const COOKING: usize = 106;

// ── Cross-cutting axes ───────────────────────────────────────────────────
const INFORMATION: usize = 107;
const SYSTEMS: usize = 108;
const OPTIMIZATION: usize = 109;
const PATTERN: usize = 110;
const STRUCTURE: usize = 111;
const NETWORK: usize = 112;
const SPACE: usize = 113;
const PERFORMANCE: usize = 114;
const MEASUREMENT: usize = 115;
const MOTION: usize = 116;
const CYCLE: usize = 117;
const BEHAVIOR: usize = 118;
const EMOTION: usize = 119;
const CONCEPT: usize = 120;
const THEORY: usize = 121;
const LEARNING: usize = 122;
const STATISTICS: usize = 123;
const MORAL: usize = 124;
const DISCOURSE: usize = 125;
const COGNITION: usize = 126;
const MIND: usize = 127;

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
// 775 concepts across 31 categories, engineered to stress-test spherical
// embedding: every one of the 40 semantic axes receives meaningful mass,
// and bridge concepts deliberately straddle category boundaries so that
// θ/φ clustering has non-trivial structure to recover.
// ─────────────────────────────────────────────────────────────────────────────
fn build_corpus() -> Vec<Concept> {
    vec![
        // ── Physics (25) ──────────────────────────────────────────────
        Concept {
            label: "Newtonian mechanics",
            category: "physics",
            features: vec![
                (FORCE, 1.0),
                (ENERGY, 0.7),
                (MATH, 0.6),
                (SPACE, 0.5),
                (MOTION, 0.6),
                (CALCULUS, 0.7),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Quantum field theory",
            category: "physics",
            features: vec![
                (QUANTUM, 1.0),
                (ENERGY, 0.8),
                (MATH, 0.9),
                (WAVE, 0.6),
                (PARTICLE, 0.8),
                (THEORY, 0.7),
                (CALCULUS, 0.5),
            ],
        },
        Concept {
            label: "General relativity",
            category: "physics",
            features: vec![
                (SPACE, 1.0),
                (ENERGY, 0.7),
                (MATH, 0.9),
                (FORCE, 0.6),
                (RELATIVITY, 1.0),
                (CALCULUS, 0.7),
                (THEORY, 0.6),
            ],
        },
        Concept {
            label: "Thermodynamics",
            category: "physics",
            features: vec![
                (ENERGY, 1.0),
                (ENTROPY, 0.9),
                (CHEMISTRY, 0.3),
                (SYSTEMS, 0.4),
                (THEORY, 0.5),
                (CYCLE, 0.4),
            ],
        },
        Concept {
            label: "Electromagnetism",
            category: "physics",
            features: vec![
                (FORCE, 0.8),
                (ENERGY, 0.8),
                (WAVE, 0.9),
                (MATH, 0.5),
                (CALCULUS, 0.6),
                (ELECTRICAL, 0.5),
            ],
        },
        Concept {
            label: "Statistical mechanics",
            category: "physics",
            features: vec![
                (STATISTICS, 0.8),
                (ENERGY, 0.7),
                (ENTROPY, 0.8),
                (MATH, 0.7),
                (THEORY, 0.5),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Optics",
            category: "physics",
            features: vec![
                (WAVE, 1.0),
                (ENERGY, 0.5),
                (SPACE, 0.4),
                (PATTERN, 0.3),
                (VISUAL, 0.4),
                (PARTICLE, 0.3),
            ],
        },
        Concept {
            label: "Particle physics",
            category: "physics",
            features: vec![
                (QUANTUM, 0.9),
                (ENERGY, 0.9),
                (FORCE, 0.7),
                (MATH, 0.6),
                (PARTICLE, 1.0),
                (RELATIVITY, 0.4),
            ],
        },
        Concept {
            label: "Cosmology",
            category: "physics",
            features: vec![
                (SPACE, 0.9),
                (ENERGY, 0.6),
                (MATH, 0.5),
                (ENTROPY, 0.4),
                (RELATIVITY, 0.7),
                (CELESTIAL, 0.6),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Fluid dynamics",
            category: "physics",
            features: vec![
                (FORCE, 0.7),
                (ENERGY, 0.6),
                (MATH, 0.7),
                (WAVE, 0.5),
                (MOTION, 0.7),
                (CALCULUS, 0.6),
            ],
        },
        Concept {
            label: "Acoustics",
            category: "physics",
            features: vec![
                (WAVE, 0.9),
                (SOUND, 0.8),
                (ENERGY, 0.4),
                (PATTERN, 0.5),
                (MATH, 0.3),
                (TIMBRE, 0.3),
            ],
        },
        Concept {
            label: "Information theory (physics)",
            category: "physics",
            features: vec![
                (ENTROPY, 0.9),
                (INFORMATION, 0.8),
                (MATH, 0.7),
                (COMPUTATION, 0.4),
                (THEORY, 0.6),
                (CONCEPT, 0.4),
            ],
        },
        Concept {
            label: "Biophysics",
            category: "physics",
            features: vec![
                (ENERGY, 0.6),
                (LIFE, 0.5),
                (CHEMISTRY, 0.5),
                (FORCE, 0.4),
                (SYSTEMS, 0.3),
                (MOLECULAR, 0.5),
                (CELLULAR, 0.3),
            ],
        },
        Concept {
            label: "Nuclear physics",
            category: "physics",
            features: vec![
                (QUANTUM, 0.8),
                (ENERGY, 1.0),
                (FORCE, 0.8),
                (PARTICLE, 0.7),
                (ATOMIC, 0.6),
            ],
        },
        Concept {
            label: "Condensed matter",
            category: "physics",
            features: vec![
                (QUANTUM, 0.6),
                (STRUCTURE, 0.7),
                (ENERGY, 0.5),
                (MATH, 0.5),
                (MATERIAL, 0.5),
                (ATOMIC, 0.4),
            ],
        },
        Concept {
            label: "Plasma physics",
            category: "physics",
            features: vec![
                (ENERGY, 0.9),
                (FORCE, 0.6),
                (WAVE, 0.5),
                (SPACE, 0.3),
                (MOTION, 0.4),
                (PARTICLE, 0.5),
                (STELLAR, 0.3),
            ],
        },
        Concept {
            label: "Laser physics",
            category: "physics",
            features: vec![
                (WAVE, 0.9),
                (ENERGY, 0.7),
                (QUANTUM, 0.6),
                (MEASUREMENT, 0.5),
                (PARTICLE, 0.3),
            ],
        },
        Concept {
            label: "Nonlinear dynamics",
            category: "physics",
            features: vec![
                (MATH, 0.8),
                (SYSTEMS, 0.7),
                (PATTERN, 0.7),
                (MOTION, 0.6),
                (ENTROPY, 0.4),
                (CALCULUS, 0.5),
            ],
        },
        Concept {
            label: "Solid state physics",
            category: "physics",
            features: vec![
                (QUANTUM, 0.6),
                (STRUCTURE, 0.8),
                (ENERGY, 0.5),
                (MATERIAL, 0.7),
                (ATOMIC, 0.5),
            ],
        },
        Concept {
            label: "Cryogenics",
            category: "physics",
            features: vec![
                (ENERGY, 0.7),
                (ENTROPY, 0.6),
                (QUANTUM, 0.5),
                (MEASUREMENT, 0.5),
                (ATOMIC, 0.3),
            ],
        },
        Concept {
            label: "Photonics",
            category: "physics",
            features: vec![
                (WAVE, 0.9),
                (ENERGY, 0.6),
                (INFORMATION, 0.5),
                (QUANTUM, 0.4),
                (PARTICLE, 0.4),
                (ELECTRICAL, 0.3),
            ],
        },
        Concept {
            label: "Classical field theory",
            category: "physics",
            features: vec![
                (MATH, 0.9),
                (FORCE, 0.8),
                (SPACE, 0.6),
                (WAVE, 0.5),
                (CALCULUS, 0.7),
                (THEORY, 0.6),
            ],
        },
        Concept {
            label: "Atomic physics",
            category: "physics",
            features: vec![
                (QUANTUM, 0.8),
                (ENERGY, 0.7),
                (WAVE, 0.5),
                (MEASUREMENT, 0.4),
                (ATOMIC, 0.9),
                (PARTICLE, 0.5),
            ],
        },
        Concept {
            label: "Rheology",
            category: "physics",
            features: vec![
                (FORCE, 0.7),
                (MATERIAL, 0.8),
                (MOTION, 0.6),
                (MATH, 0.4),
                (MECHANICAL, 0.4),
            ],
        },
        Concept {
            label: "Quantum information",
            category: "physics",
            features: vec![
                (QUANTUM, 0.9),
                (INFORMATION, 0.9),
                (COMPUTATION, 0.6),
                (MATH, 0.7),
                (ENTROPY, 0.5),
                (THEORY, 0.4),
            ],
        },
        // ── BIOLOGY (25) ──
        Concept {
            label: "Evolution by natural selection",
            category: "biology",
            features: vec![
                (EVOLUTION, 1.0),
                (LIFE, 0.9),
                (GENETICS, 0.7),
                (NATURE, 0.6),
                (THEORY, 0.7),
                (CONCEPT, 0.5),
            ],
        },
        Concept {
            label: "Molecular biology",
            category: "biology",
            features: vec![
                (LIFE, 0.9),
                (CHEMISTRY, 0.8),
                (GENETICS, 0.7),
                (STRUCTURE, 0.5),
                (MOLECULAR, 0.8),
                (CELLULAR, 0.5),
            ],
        },
        Concept {
            label: "Ecology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (NATURE, 1.0),
                (SYSTEMS, 0.7),
                (ECOSYSTEM, 0.9),
                (EVOLUTION, 0.4),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Genetics",
            category: "biology",
            features: vec![
                (GENETICS, 1.0),
                (LIFE, 0.8),
                (INFORMATION, 0.5),
                (CHEMISTRY, 0.4),
                (MOLECULAR, 0.5),
                (CONCEPT, 0.4),
            ],
        },
        Concept {
            label: "Cell biology",
            category: "biology",
            features: vec![
                (LIFE, 1.0),
                (CHEMISTRY, 0.6),
                (SYSTEMS, 0.5),
                (STRUCTURE, 0.5),
                (CELLULAR, 0.9),
                (MOLECULAR, 0.5),
            ],
        },
        Concept {
            label: "Bioinformatics",
            category: "biology",
            features: vec![
                (LIFE, 0.5),
                (COMPUTATION, 0.7),
                (GENETICS, 0.6),
                (INFORMATION, 0.6),
                (STATISTICS, 0.5),
                (ALGORITHM, 0.4),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Immunology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (CHEMISTRY, 0.5),
                (SYSTEMS, 0.6),
                (EVOLUTION, 0.3),
                (CELLULAR, 0.6),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Botany",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (NATURE, 0.9),
                (CHEMISTRY, 0.3),
                (EVOLUTION, 0.3),
                (ECOSYSTEM, 0.5),
                (CELLULAR, 0.3),
            ],
        },
        Concept {
            label: "Marine biology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (NATURE, 0.8),
                (EVOLUTION, 0.4),
                (ECOSYSTEM, 0.7),
                (OCEAN, 0.7),
            ],
        },
        Concept {
            label: "Microbiology",
            category: "biology",
            features: vec![
                (LIFE, 0.9),
                (CHEMISTRY, 0.5),
                (EVOLUTION, 0.5),
                (GENETICS, 0.4),
                (CELLULAR, 0.7),
            ],
        },
        Concept {
            label: "Developmental biology",
            category: "biology",
            features: vec![
                (LIFE, 0.9),
                (GENETICS, 0.6),
                (SYSTEMS, 0.5),
                (STRUCTURE, 0.4),
                (CELLULAR, 0.6),
                (CYCLE, 0.4),
            ],
        },
        Concept {
            label: "Evolutionary psychology",
            category: "biology",
            features: vec![
                (EVOLUTION, 0.7),
                (MIND, 0.6),
                (BEHAVIOR, 0.6),
                (COGNITION, 0.5),
                (BRAIN, 0.4),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Biostatistics",
            category: "biology",
            features: vec![
                (LIFE, 0.5),
                (STATISTICS, 0.8),
                (MATH, 0.5),
                (GENETICS, 0.3),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Taxonomy",
            category: "biology",
            features: vec![
                (LIFE, 0.7),
                (EVOLUTION, 0.6),
                (STRUCTURE, 0.6),
                (NATURE, 0.5),
                (CONCEPT, 0.4),
            ],
        },
        Concept {
            label: "Zoology",
            category: "biology",
            features: vec![
                (LIFE, 0.9),
                (EVOLUTION, 0.6),
                (NATURE, 0.7),
                (BEHAVIOR, 0.4),
                (ANATOMY, 0.3),
            ],
        },
        Concept {
            label: "Parasitology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (EVOLUTION, 0.5),
                (ECOSYSTEM, 0.5),
                (SYSTEMS, 0.3),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Mycology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (CHEMISTRY, 0.5),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.4),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Virology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (GENETICS, 0.6),
                (EVOLUTION, 0.6),
                (CHEMISTRY, 0.4),
                (MOLECULAR, 0.5),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Conservation biology",
            category: "biology",
            features: vec![
                (LIFE, 0.7),
                (ECOSYSTEM, 0.9),
                (NATURE, 0.8),
                (ETHICS, 0.4),
                (CONSERVATION, 0.8),
            ],
        },
        Concept {
            label: "Population genetics",
            category: "biology",
            features: vec![
                (GENETICS, 0.9),
                (EVOLUTION, 0.8),
                (STATISTICS, 0.7),
                (MATH, 0.5),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Ethology",
            category: "biology",
            features: vec![
                (BEHAVIOR, 0.9),
                (LIFE, 0.7),
                (EVOLUTION, 0.5),
                (COGNITION, 0.4),
                (BRAIN, 0.3),
            ],
        },
        Concept {
            label: "Proteomics",
            category: "biology",
            features: vec![
                (LIFE, 0.7),
                (CHEMISTRY, 0.7),
                (INFORMATION, 0.6),
                (COMPUTATION, 0.4),
                (MOLECULAR, 0.7),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Systems biology",
            category: "biology",
            features: vec![
                (LIFE, 0.7),
                (SYSTEMS, 0.9),
                (COMPUTATION, 0.5),
                (NETWORK, 0.6),
                (CELLULAR, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Entomology",
            category: "biology",
            features: vec![
                (LIFE, 0.8),
                (NATURE, 0.7),
                (EVOLUTION, 0.5),
                (ECOSYSTEM, 0.5),
                (ANATOMY, 0.3),
            ],
        },
        Concept {
            label: "Epigenetics",
            category: "biology",
            features: vec![
                (GENETICS, 0.8),
                (LIFE, 0.7),
                (CHEMISTRY, 0.5),
                (EVOLUTION, 0.4),
                (INFORMATION, 0.4),
                (MOLECULAR, 0.5),
                (CELLULAR, 0.4),
            ],
        },
        // ── COMPUTER SCIENCE (25) ──
        Concept {
            label: "Algorithm design",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.9),
                (LOGIC, 0.8),
                (MATH, 0.7),
                (OPTIMIZATION, 0.6),
                (ALGORITHM, 0.9),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Machine learning",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (STATISTICS, 0.7),
                (PATTERN, 0.8),
                (OPTIMIZATION, 0.7),
                (MACHINE_LEARN, 0.9),
                (AI, 0.6),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Database systems",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (INFORMATION, 0.8),
                (STRUCTURE, 0.7),
                (SYSTEMS, 0.6),
                (SOFTWARE, 0.5),
                (DATA, 0.7),
            ],
        },
        Concept {
            label: "Networking",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.6),
                (NETWORK, 0.9),
                (SYSTEMS, 0.7),
                (INFORMATION, 0.5),
                (SOFTWARE, 0.4),
            ],
        },
        Concept {
            label: "Cryptography",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (MATH, 0.9),
                (INFORMATION, 0.7),
                (LOGIC, 0.5),
                (ALGORITHM, 0.6),
                (ALGEBRA, 0.4),
            ],
        },
        Concept {
            label: "Operating systems",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (SYSTEMS, 0.9),
                (STRUCTURE, 0.5),
                (LOGIC, 0.4),
                (SOFTWARE, 0.7),
            ],
        },
        Concept {
            label: "Artificial intelligence",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (COGNITION, 0.6),
                (LOGIC, 0.6),
                (PATTERN, 0.5),
                (MIND, 0.3),
                (AI, 0.9),
                (MACHINE_LEARN, 0.4),
            ],
        },
        Concept {
            label: "Computational complexity",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.9),
                (MATH, 0.9),
                (LOGIC, 0.8),
                (ALGORITHM, 0.7),
                (PROOF, 0.4),
            ],
        },
        Concept {
            label: "Computer graphics",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (MATH, 0.5),
                (SPACE, 0.5),
                (VISUAL, 0.7),
                (PATTERN, 0.4),
                (SOFTWARE, 0.4),
            ],
        },
        Concept {
            label: "Natural language processing",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (LANGUAGE, 0.8),
                (PATTERN, 0.6),
                (COGNITION, 0.4),
                (STATISTICS, 0.5),
                (AI, 0.5),
                (LLM, 0.4),
            ],
        },
        Concept {
            label: "Algorithmic trading",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.6),
                (MARKETS, 0.7),
                (OPTIMIZATION, 0.7),
                (STATISTICS, 0.5),
                (ALGORITHM, 0.6),
                (FINANCE, 0.5),
            ],
        },
        Concept {
            label: "Computational biology",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (LIFE, 0.4),
                (GENETICS, 0.4),
                (STATISTICS, 0.5),
                (PATTERN, 0.4),
                (ALGORITHM, 0.4),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Formal verification",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (LOGIC, 1.0),
                (MATH, 0.8),
                (PROOF, 0.7),
                (SOFTWARE, 0.4),
            ],
        },
        Concept {
            label: "Information retrieval",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.6),
                (INFORMATION, 0.9),
                (LANGUAGE, 0.4),
                (PATTERN, 0.5),
                (ALGORITHM, 0.4),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Distributed systems",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (SYSTEMS, 0.8),
                (NETWORK, 0.7),
                (LOGIC, 0.3),
                (SOFTWARE, 0.5),
            ],
        },
        Concept {
            label: "Human-computer interaction",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.5),
                (COGNITION, 0.7),
                (VISUAL, 0.5),
                (BEHAVIOR, 0.5),
                (PERFORMANCE, 0.3),
                (DESIGN, 0.4),
            ],
        },
        Concept {
            label: "Compiler design",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.9),
                (LANGUAGE, 0.6),
                (LOGIC, 0.7),
                (STRUCTURE, 0.6),
                (SOFTWARE, 0.6),
                (ALGORITHM, 0.4),
            ],
        },
        Concept {
            label: "Computer vision",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (VISUAL, 0.9),
                (PATTERN, 0.8),
                (MATH, 0.4),
                (AI, 0.5),
                (MACHINE_LEARN, 0.5),
            ],
        },
        Concept {
            label: "Parallel computing",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.9),
                (SYSTEMS, 0.7),
                (OPTIMIZATION, 0.6),
                (MATH, 0.3),
                (SOFTWARE, 0.4),
                (ALGORITHM, 0.4),
            ],
        },
        Concept {
            label: "Cybersecurity",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (INFORMATION, 0.7),
                (LOGIC, 0.6),
                (NETWORK, 0.6),
                (ETHICS, 0.3),
                (SOFTWARE, 0.5),
            ],
        },
        Concept {
            label: "Quantum computing",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (QUANTUM, 0.9),
                (MATH, 0.7),
                (INFORMATION, 0.6),
                (ALGORITHM, 0.5),
            ],
        },
        Concept {
            label: "Software engineering",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.8),
                (STRUCTURE, 0.8),
                (LOGIC, 0.6),
                (SYSTEMS, 0.6),
                (SOFTWARE, 0.9),
            ],
        },
        Concept {
            label: "Reinforcement learning",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (OPTIMIZATION, 0.8),
                (BEHAVIOR, 0.5),
                (PATTERN, 0.5),
                (MACHINE_LEARN, 0.7),
                (AI, 0.4),
                (LEARNING, 0.5),
            ],
        },
        Concept {
            label: "Computational geometry",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (MATH, 0.8),
                (SPACE, 0.7),
                (STRUCTURE, 0.5),
                (ALGORITHM, 0.5),
            ],
        },
        Concept {
            label: "Robotics (CS)",
            category: "computer_science",
            features: vec![
                (COMPUTATION, 0.7),
                (MOTION, 0.7),
                (SYSTEMS, 0.6),
                (FORCE, 0.4),
                (COGNITION, 0.3),
                (AI, 0.4),
                (MECHANICAL, 0.3),
            ],
        },
        // ── PHILOSOPHY (25) ──
        Concept {
            label: "Formal logic",
            category: "philosophy",
            features: vec![
                (LOGIC, 1.0),
                (MATH, 0.7),
                (LANGUAGE, 0.4),
                (STRUCTURE, 0.4),
                (PROOF, 0.6),
                (CONCEPT, 0.4),
            ],
        },
        Concept {
            label: "Ethics",
            category: "philosophy",
            features: vec![
                (ETHICS, 1.0),
                (BEHAVIOR, 0.6),
                (MIND, 0.4),
                (MORAL, 0.8),
                (CONCEPT, 0.5),
            ],
        },
        Concept {
            label: "Philosophy of mind",
            category: "philosophy",
            features: vec![
                (MIND, 1.0),
                (COGNITION, 0.7),
                (METAPHYSICS, 0.6),
                (LANGUAGE, 0.3),
                (CONSCIOUSNESS, 0.7),
                (ONTOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Epistemology",
            category: "philosophy",
            features: vec![
                (MIND, 0.7),
                (LOGIC, 0.6),
                (METAPHYSICS, 0.5),
                (COGNITION, 0.5),
                (EPISTEMOLOGY, 0.9),
                (CONCEPT, 0.5),
            ],
        },
        Concept {
            label: "Metaphysics",
            category: "philosophy",
            features: vec![
                (METAPHYSICS, 1.0),
                (MIND, 0.5),
                (SPACE, 0.3),
                (STRUCTURE, 0.3),
                (ONTOLOGY, 0.8),
                (CONCEPT, 0.5),
            ],
        },
        Concept {
            label: "Philosophy of language",
            category: "philosophy",
            features: vec![
                (LANGUAGE, 0.8),
                (LOGIC, 0.6),
                (MIND, 0.5),
                (STRUCTURE, 0.4),
                (SEMANTICS_AX, 0.5),
                (DISCOURSE, 0.4),
            ],
        },
        Concept {
            label: "Political philosophy",
            category: "philosophy",
            features: vec![
                (ETHICS, 0.7),
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.5),
                (GOVERNANCE, 0.5),
                (POWER, 0.5),
                (MORAL, 0.4),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Bioethics",
            category: "philosophy",
            features: vec![
                (ETHICS, 0.9),
                (LIFE, 0.5),
                (DIAGNOSTICS, 0.3),
                (MIND, 0.3),
                (MORAL, 0.7),
                (CLINICAL, 0.2),
            ],
        },
        Concept {
            label: "Game theory (philosophy)",
            category: "philosophy",
            features: vec![
                (LOGIC, 0.7),
                (BEHAVIOR, 0.7),
                (MATH, 0.6),
                (OPTIMIZATION, 0.5),
                (MARKETS, 0.3),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Aesthetics",
            category: "philosophy",
            features: vec![
                (MIND, 0.5),
                (EMOTION, 0.7),
                (VISUAL, 0.4),
                (ETHICS, 0.2),
                (FORM, 0.4),
                (CONCEPT, 0.3),
            ],
        },
        Concept {
            label: "Philosophy of science",
            category: "philosophy",
            features: vec![
                (LOGIC, 0.6),
                (METAPHYSICS, 0.5),
                (MIND, 0.4),
                (STRUCTURE, 0.4),
                (MEASUREMENT, 0.3),
                (EPISTEMOLOGY, 0.6),
                (THEORY, 0.6),
            ],
        },
        Concept {
            label: "Phenomenology",
            category: "philosophy",
            features: vec![
                (MIND, 0.8),
                (METAPHYSICS, 0.7),
                (COGNITION, 0.5),
                (CONSCIOUSNESS, 0.6),
                (ONTOLOGY, 0.4),
            ],
        },
        Concept {
            label: "Existentialism",
            category: "philosophy",
            features: vec![
                (MIND, 0.7),
                (EMOTION, 0.6),
                (METAPHYSICS, 0.7),
                (ETHICS, 0.4),
                (ONTOLOGY, 0.5),
                (CONSCIOUSNESS, 0.4),
            ],
        },
        Concept {
            label: "Philosophy of mathematics",
            category: "philosophy",
            features: vec![
                (LOGIC, 0.8),
                (MATH, 0.8),
                (METAPHYSICS, 0.6),
                (STRUCTURE, 0.5),
                (PROOF, 0.5),
                (EPISTEMOLOGY, 0.4),
            ],
        },
        Concept {
            label: "Virtue ethics",
            category: "philosophy",
            features: vec![
                (ETHICS, 0.9),
                (BEHAVIOR, 0.7),
                (EMOTION, 0.4),
                (MIND, 0.3),
                (MORAL, 0.8),
            ],
        },
        Concept {
            label: "Social contract theory",
            category: "philosophy",
            features: vec![
                (ETHICS, 0.7),
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.6),
                (LOGIC, 0.4),
                (THEORY, 0.5),
                (RIGHTS, 0.4),
                (SOCIETY, 0.3),
            ],
        },
        Concept {
            label: "Utilitarianism",
            category: "philosophy",
            features: vec![
                (ETHICS, 0.9),
                (OPTIMIZATION, 0.6),
                (BEHAVIOR, 0.5),
                (MATH, 0.3),
                (MORAL, 0.7),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Philosophy of technology",
            category: "philosophy",
            features: vec![
                (MIND, 0.5),
                (COMPUTATION, 0.4),
                (ETHICS, 0.5),
                (METAPHYSICS, 0.4),
                (ONTOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Free will debate",
            category: "philosophy",
            features: vec![
                (MIND, 0.8),
                (METAPHYSICS, 0.8),
                (BEHAVIOR, 0.5),
                (COGNITION, 0.4),
                (CONSCIOUSNESS, 0.5),
                (DISCOURSE, 0.3),
            ],
        },
        Concept {
            label: "Hermeneutics",
            category: "philosophy",
            features: vec![
                (LANGUAGE, 0.7),
                (MIND, 0.6),
                (NARRATIVE, 0.5),
                (METAPHYSICS, 0.5),
                (DISCOURSE, 0.5),
                (EPISTEMOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Modal logic",
            category: "philosophy",
            features: vec![
                (LOGIC, 0.9),
                (MATH, 0.7),
                (METAPHYSICS, 0.5),
                (STRUCTURE, 0.4),
                (PROOF, 0.4),
            ],
        },
        Concept {
            label: "Environmental ethics",
            category: "philosophy",
            features: vec![
                (ETHICS, 0.9),
                (NATURE, 0.7),
                (ECOSYSTEM, 0.5),
                (BEHAVIOR, 0.3),
                (MORAL, 0.7),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Philosophy of art",
            category: "philosophy",
            features: vec![
                (MIND, 0.5),
                (EMOTION, 0.7),
                (METAPHYSICS, 0.5),
                (VISUAL, 0.4),
                (PERFORMANCE, 0.3),
                (FORM, 0.4),
            ],
        },
        Concept {
            label: "Pragmatism",
            category: "philosophy",
            features: vec![
                (MIND, 0.6),
                (BEHAVIOR, 0.6),
                (LOGIC, 0.5),
                (COGNITION, 0.4),
                (EPISTEMOLOGY, 0.4),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Consciousness studies",
            category: "philosophy",
            features: vec![
                (MIND, 0.9),
                (COGNITION, 0.8),
                (METAPHYSICS, 0.6),
                (LIFE, 0.3),
                (CONSCIOUSNESS, 0.9),
                (NEURAL, 0.3),
            ],
        },
        // ── ECONOMICS (25) ──
        Concept {
            label: "Microeconomics",
            category: "economics",
            features: vec![
                (MARKETS, 0.9),
                (OPTIMIZATION, 0.8),
                (BEHAVIOR, 0.6),
                (MATH, 0.5),
                (THEORY, 0.5),
                (MONEY, 0.3),
            ],
        },
        Concept {
            label: "Macroeconomics",
            category: "economics",
            features: vec![
                (MARKETS, 0.8),
                (SYSTEMS, 0.7),
                (STATISTICS, 0.5),
                (BEHAVIOR, 0.4),
                (MONEY, 0.6),
                (POLICY, 0.4),
            ],
        },
        Concept {
            label: "Behavioral economics",
            category: "economics",
            features: vec![
                (BEHAVIOR, 0.9),
                (MARKETS, 0.6),
                (COGNITION, 0.5),
                (MIND, 0.3),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Econometrics",
            category: "economics",
            features: vec![
                (STATISTICS, 0.9),
                (MATH, 0.7),
                (MARKETS, 0.6),
                (COMPUTATION, 0.3),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Game theory (economics)",
            category: "economics",
            features: vec![
                (MATH, 0.7),
                (OPTIMIZATION, 0.8),
                (BEHAVIOR, 0.6),
                (LOGIC, 0.4),
                (MARKETS, 0.5),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Financial engineering",
            category: "economics",
            features: vec![
                (MARKETS, 0.8),
                (MATH, 0.7),
                (OPTIMIZATION, 0.6),
                (COMPUTATION, 0.4),
                (FINANCE, 0.8),
            ],
        },
        Concept {
            label: "Development economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (SYSTEMS, 0.5),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.3),
                (GOVERNANCE, 0.4),
                (POLICY, 0.4),
            ],
        },
        Concept {
            label: "Labor economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.7),
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.4),
                (STATISTICS, 0.3),
                (LABOR, 0.9),
            ],
        },
        Concept {
            label: "Network economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (NETWORK, 0.7),
                (SYSTEMS, 0.6),
                (OPTIMIZATION, 0.4),
                (SOCIAL_NETWORK, 0.3),
            ],
        },
        Concept {
            label: "Environmental economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.5),
                (NATURE, 0.5),
                (ECOSYSTEM, 0.5),
                (ETHICS, 0.3),
                (CONSERVATION, 0.4),
                (POLICY, 0.3),
            ],
        },
        Concept {
            label: "Public choice theory",
            category: "economics",
            features: vec![
                (BEHAVIOR, 0.7),
                (MARKETS, 0.5),
                (LOGIC, 0.4),
                (GOVERNANCE, 0.5),
                (THEORY, 0.5),
                (POLICY, 0.3),
            ],
        },
        Concept {
            label: "Auction theory",
            category: "economics",
            features: vec![
                (MATH, 0.6),
                (OPTIMIZATION, 0.7),
                (MARKETS, 0.7),
                (BEHAVIOR, 0.4),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Monetary economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.8),
                (SYSTEMS, 0.6),
                (GOVERNANCE, 0.5),
                (STATISTICS, 0.4),
                (MONEY, 0.9),
                (FINANCE, 0.4),
            ],
        },
        Concept {
            label: "International trade",
            category: "economics",
            features: vec![
                (MARKETS, 0.8),
                (NETWORK, 0.6),
                (SYSTEMS, 0.5),
                (GOVERNANCE, 0.3),
                (POLICY, 0.4),
            ],
        },
        Concept {
            label: "Health economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (LIFE, 0.5),
                (DIAGNOSTICS, 0.4),
                (STATISTICS, 0.5),
                (CLINICAL, 0.3),
                (POLICY, 0.3),
            ],
        },
        Concept {
            label: "Agricultural economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.7),
                (NATURE, 0.5),
                (ECOSYSTEM, 0.4),
                (SYSTEMS, 0.4),
                (LABOR, 0.3),
            ],
        },
        Concept {
            label: "Industrial organization",
            category: "economics",
            features: vec![
                (MARKETS, 0.8),
                (SYSTEMS, 0.6),
                (OPTIMIZATION, 0.5),
                (STRUCTURE, 0.4),
                (LABOR, 0.3),
            ],
        },
        Concept {
            label: "Welfare economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (ETHICS, 0.6),
                (OPTIMIZATION, 0.6),
                (GOVERNANCE, 0.4),
                (MORAL, 0.4),
                (POLICY, 0.4),
            ],
        },
        Concept {
            label: "Resource economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.5),
                (OPTIMIZATION, 0.5),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Urban economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (STRUCTURE, 0.5),
                (SYSTEMS, 0.5),
                (NETWORK, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Information economics",
            category: "economics",
            features: vec![
                (MARKETS, 0.6),
                (INFORMATION, 0.8),
                (BEHAVIOR, 0.5),
                (OPTIMIZATION, 0.4),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Risk management",
            category: "economics",
            features: vec![
                (MARKETS, 0.7),
                (STATISTICS, 0.7),
                (MATH, 0.5),
                (OPTIMIZATION, 0.5),
                (FINANCE, 0.6),
            ],
        },
        Concept {
            label: "Experimental economics",
            category: "economics",
            features: vec![
                (BEHAVIOR, 0.7),
                (MARKETS, 0.5),
                (STATISTICS, 0.6),
                (MEASUREMENT, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Mechanism design",
            category: "economics",
            features: vec![
                (MATH, 0.7),
                (OPTIMIZATION, 0.8),
                (LOGIC, 0.6),
                (MARKETS, 0.5),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Economic history",
            category: "economics",
            features: vec![
                (MARKETS, 0.7),
                (BEHAVIOR, 0.5),
                (SYSTEMS, 0.5),
                (EVOLUTION, 0.4),
                (NARRATIVE, 0.3),
                (HISTORICAL, 0.5),
            ],
        },
        // ── MUSIC (25) ──
        Concept {
            label: "Harmonic theory",
            category: "music",
            features: vec![
                (SOUND, 0.8),
                (MATH, 0.7),
                (PATTERN, 0.8),
                (WAVE, 0.5),
                (HARMONY, 0.9),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Orchestration",
            category: "music",
            features: vec![
                (SOUND, 0.9),
                (PERFORMANCE, 0.7),
                (EMOTION, 0.5),
                (SYSTEMS, 0.3),
                (TIMBRE, 0.8),
                (HARMONY, 0.4),
            ],
        },
        Concept {
            label: "Rhythm and meter",
            category: "music",
            features: vec![
                (SOUND, 0.7),
                (PATTERN, 1.0),
                (MATH, 0.4),
                (PERFORMANCE, 0.4),
                (RHYTHM, 0.9),
                (CYCLE, 0.4),
            ],
        },
        Concept {
            label: "Musical acoustics",
            category: "music",
            features: vec![
                (SOUND, 0.9),
                (WAVE, 0.8),
                (ENERGY, 0.4),
                (MATH, 0.3),
                (PATTERN, 0.3),
                (TIMBRE, 0.6),
            ],
        },
        Concept {
            label: "Music cognition",
            category: "music",
            features: vec![
                (SOUND, 0.5),
                (COGNITION, 0.7),
                (EMOTION, 0.6),
                (PATTERN, 0.5),
                (MIND, 0.3),
                (BRAIN, 0.3),
                (RHYTHM, 0.3),
            ],
        },
        Concept {
            label: "Composition",
            category: "music",
            features: vec![
                (SOUND, 0.7),
                (EMOTION, 0.7),
                (PATTERN, 0.6),
                (STRUCTURE, 0.5),
                (HARMONY, 0.6),
                (RHYTHM, 0.4),
            ],
        },
        Concept {
            label: "Ethnomusicology",
            category: "music",
            features: vec![
                (SOUND, 0.6),
                (LANGUAGE, 0.5),
                (BEHAVIOR, 0.4),
                (PATTERN, 0.4),
                (COGNITION, 0.3),
                (CULTURE, 0.5),
                (TRADITION, 0.4),
            ],
        },
        Concept {
            label: "Music theory",
            category: "music",
            features: vec![
                (SOUND, 0.7),
                (MATH, 0.6),
                (PATTERN, 0.7),
                (STRUCTURE, 0.5),
                (HARMONY, 0.7),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Performance practice",
            category: "music",
            features: vec![
                (PERFORMANCE, 1.0),
                (SOUND, 0.7),
                (EMOTION, 0.6),
                (MOTION, 0.3),
                (RHYTHM, 0.4),
            ],
        },
        Concept {
            label: "Digital audio",
            category: "music",
            features: vec![
                (SOUND, 0.7),
                (COMPUTATION, 0.6),
                (WAVE, 0.5),
                (INFORMATION, 0.4),
                (SOFTWARE, 0.3),
                (TIMBRE, 0.3),
            ],
        },
        Concept {
            label: "Music therapy",
            category: "music",
            features: vec![
                (SOUND, 0.5),
                (EMOTION, 0.8),
                (MIND, 0.4),
                (DIAGNOSTICS, 0.2),
                (COGNITION, 0.3),
                (THERAPY, 0.6),
                (MENTAL_HEALTH, 0.4),
            ],
        },
        Concept {
            label: "Counterpoint",
            category: "music",
            features: vec![
                (SOUND, 0.6),
                (PATTERN, 0.7),
                (MATH, 0.5),
                (STRUCTURE, 0.6),
                (HARMONY, 0.7),
            ],
        },
        Concept {
            label: "Choral conducting",
            category: "music",
            features: vec![
                (PERFORMANCE, 0.9),
                (SOUND, 0.7),
                (EMOTION, 0.6),
                (SYSTEMS, 0.3),
                (HARMONY, 0.5),
            ],
        },
        Concept {
            label: "Film scoring",
            category: "music",
            features: vec![
                (SOUND, 0.7),
                (EMOTION, 0.8),
                (NARRATIVE, 0.6),
                (VISUAL, 0.3),
                (CINEMA, 0.5),
                (TIMBRE, 0.4),
            ],
        },
        Concept {
            label: "Sound design",
            category: "music",
            features: vec![
                (SOUND, 0.9),
                (COMPUTATION, 0.5),
                (WAVE, 0.6),
                (EMOTION, 0.4),
                (TIMBRE, 0.7),
            ],
        },
        Concept {
            label: "Musical notation",
            category: "music",
            features: vec![
                (SOUND, 0.5),
                (LANGUAGE, 0.6),
                (PATTERN, 0.7),
                (STRUCTURE, 0.5),
                (INFORMATION, 0.3),
                (RHYTHM, 0.4),
            ],
        },
        Concept {
            label: "Tuning systems",
            category: "music",
            features: vec![
                (SOUND, 0.7),
                (MATH, 0.8),
                (WAVE, 0.6),
                (MEASUREMENT, 0.4),
                (HARMONY, 0.5),
            ],
        },
        Concept {
            label: "Music production",
            category: "music",
            features: vec![
                (SOUND, 0.8),
                (COMPUTATION, 0.5),
                (PERFORMANCE, 0.5),
                (SYSTEMS, 0.3),
                (TIMBRE, 0.4),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Improvisation",
            category: "music",
            features: vec![
                (PERFORMANCE, 0.8),
                (EMOTION, 0.7),
                (COGNITION, 0.6),
                (PATTERN, 0.5),
                (RHYTHM, 0.5),
                (HARMONY, 0.3),
            ],
        },
        Concept {
            label: "Opera studies",
            category: "music",
            features: vec![
                (PERFORMANCE, 0.8),
                (SOUND, 0.7),
                (LANGUAGE, 0.6),
                (EMOTION, 0.7),
                (NARRATIVE, 0.5),
                (THEATRICAL, 0.5),
            ],
        },
        Concept {
            label: "Musical form analysis",
            category: "music",
            features: vec![
                (PATTERN, 0.8),
                (STRUCTURE, 0.7),
                (MATH, 0.4),
                (SOUND, 0.5),
                (HARMONY, 0.4),
                (FORM, 0.4),
            ],
        },
        Concept {
            label: "Early music practice",
            category: "music",
            features: vec![
                (PERFORMANCE, 0.7),
                (SOUND, 0.6),
                (EVOLUTION, 0.4),
                (PATTERN, 0.4),
                (HISTORICAL, 0.4),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Music information retrieval",
            category: "music",
            features: vec![
                (SOUND, 0.6),
                (COMPUTATION, 0.7),
                (INFORMATION, 0.7),
                (PATTERN, 0.6),
                (ALGORITHM, 0.3),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Songwriting",
            category: "music",
            features: vec![
                (SOUND, 0.6),
                (LANGUAGE, 0.7),
                (EMOTION, 0.8),
                (NARRATIVE, 0.5),
                (POETRY, 0.3),
                (RHYTHM, 0.4),
            ],
        },
        Concept {
            label: "World music studies",
            category: "music",
            features: vec![
                (SOUND, 0.6),
                (BEHAVIOR, 0.5),
                (LANGUAGE, 0.4),
                (EVOLUTION, 0.4),
                (PATTERN, 0.5),
                (CULTURE, 0.5),
                (RHYTHM, 0.3),
            ],
        },
        // ── MEDICINE (25) ──
        Concept {
            label: "Clinical diagnostics",
            category: "medicine",
            features: vec![
                (DIAGNOSTICS, 1.0),
                (LIFE, 0.6),
                (STATISTICS, 0.5),
                (SYSTEMS, 0.3),
                (CLINICAL, 0.9),
                (ANATOMY, 0.3),
            ],
        },
        Concept {
            label: "Pharmacology",
            category: "medicine",
            features: vec![
                (CHEMISTRY, 0.9),
                (LIFE, 0.7),
                (DIAGNOSTICS, 0.4),
                (SYSTEMS, 0.3),
                (THERAPY, 0.5),
                (MOLECULAR, 0.4),
                (CLINICAL, 0.3),
            ],
        },
        Concept {
            label: "Epidemiology",
            category: "medicine",
            features: vec![
                (STATISTICS, 0.9),
                (LIFE, 0.6),
                (SYSTEMS, 0.6),
                (NETWORK, 0.4),
                (DATA, 0.4),
                (CLINICAL, 0.3),
            ],
        },
        Concept {
            label: "Surgery",
            category: "medicine",
            features: vec![
                (LIFE, 0.7),
                (DIAGNOSTICS, 0.6),
                (FORCE, 0.3),
                (PERFORMANCE, 0.4),
                (ANATOMY, 0.8),
                (CLINICAL, 0.5),
            ],
        },
        Concept {
            label: "Psychiatry",
            category: "medicine",
            features: vec![
                (MIND, 0.7),
                (LIFE, 0.6),
                (DIAGNOSTICS, 0.5),
                (BEHAVIOR, 0.5),
                (CHEMISTRY, 0.3),
                (MENTAL_HEALTH, 0.8),
                (THERAPY, 0.6),
            ],
        },
        Concept {
            label: "Radiology",
            category: "medicine",
            features: vec![
                (DIAGNOSTICS, 0.8),
                (WAVE, 0.5),
                (ENERGY, 0.3),
                (VISUAL, 0.5),
                (COMPUTATION, 0.3),
                (CLINICAL, 0.5),
                (ANATOMY, 0.4),
            ],
        },
        Concept {
            label: "Pathology",
            category: "medicine",
            features: vec![
                (LIFE, 0.7),
                (CHEMISTRY, 0.5),
                (DIAGNOSTICS, 0.7),
                (STRUCTURE, 0.3),
                (CELLULAR, 0.5),
                (CLINICAL, 0.4),
            ],
        },
        Concept {
            label: "Genomic medicine",
            category: "medicine",
            features: vec![
                (GENETICS, 0.7),
                (LIFE, 0.6),
                (DIAGNOSTICS, 0.5),
                (INFORMATION, 0.4),
                (COMPUTATION, 0.3),
                (MOLECULAR, 0.5),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Medical imaging AI",
            category: "medicine",
            features: vec![
                (DIAGNOSTICS, 0.6),
                (COMPUTATION, 0.7),
                (PATTERN, 0.6),
                (VISUAL, 0.5),
                (STATISTICS, 0.4),
                (AI, 0.5),
                (CLINICAL, 0.3),
            ],
        },
        Concept {
            label: "Public health",
            category: "medicine",
            features: vec![
                (LIFE, 0.5),
                (STATISTICS, 0.6),
                (SYSTEMS, 0.6),
                (BEHAVIOR, 0.4),
                (ETHICS, 0.3),
                (POLICY, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Neurology",
            category: "medicine",
            features: vec![
                (LIFE, 0.6),
                (MIND, 0.5),
                (DIAGNOSTICS, 0.6),
                (COGNITION, 0.4),
                (NETWORK, 0.3),
                (BRAIN, 0.7),
                (NEURAL, 0.5),
                (CLINICAL, 0.4),
            ],
        },
        Concept {
            label: "Immunotherapy",
            category: "medicine",
            features: vec![
                (LIFE, 0.7),
                (CHEMISTRY, 0.5),
                (SYSTEMS, 0.5),
                (DIAGNOSTICS, 0.3),
                (THERAPY, 0.7),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Cardiology",
            category: "medicine",
            features: vec![
                (LIFE, 0.7),
                (DIAGNOSTICS, 0.7),
                (SYSTEMS, 0.4),
                (MEASUREMENT, 0.4),
                (ANATOMY, 0.6),
                (CLINICAL, 0.5),
            ],
        },
        Concept {
            label: "Oncology",
            category: "medicine",
            features: vec![
                (LIFE, 0.8),
                (CHEMISTRY, 0.5),
                (GENETICS, 0.5),
                (DIAGNOSTICS, 0.6),
                (CELLULAR, 0.5),
                (THERAPY, 0.5),
                (CLINICAL, 0.4),
            ],
        },
        Concept {
            label: "Dermatology",
            category: "medicine",
            features: vec![
                (LIFE, 0.6),
                (DIAGNOSTICS, 0.7),
                (VISUAL, 0.5),
                (PATTERN, 0.4),
                (ANATOMY, 0.4),
                (CLINICAL, 0.4),
            ],
        },
        Concept {
            label: "Anesthesiology",
            category: "medicine",
            features: vec![
                (LIFE, 0.6),
                (CHEMISTRY, 0.6),
                (SYSTEMS, 0.4),
                (MEASUREMENT, 0.5),
                (CLINICAL, 0.5),
                (NEURAL, 0.3),
            ],
        },
        Concept {
            label: "Emergency medicine",
            category: "medicine",
            features: vec![
                (LIFE, 0.8),
                (DIAGNOSTICS, 0.7),
                (SYSTEMS, 0.5),
                (PERFORMANCE, 0.4),
                (CLINICAL, 0.7),
            ],
        },
        Concept {
            label: "Pediatrics",
            category: "medicine",
            features: vec![
                (LIFE, 0.8),
                (DIAGNOSTICS, 0.5),
                (BEHAVIOR, 0.3),
                (EVOLUTION, 0.2),
                (CLINICAL, 0.5),
                (ANATOMY, 0.3),
            ],
        },
        Concept {
            label: "Geriatrics",
            category: "medicine",
            features: vec![
                (LIFE, 0.7),
                (DIAGNOSTICS, 0.5),
                (SYSTEMS, 0.4),
                (EVOLUTION, 0.3),
                (CLINICAL, 0.5),
                (THERAPY, 0.3),
            ],
        },
        Concept {
            label: "Rehabilitation medicine",
            category: "medicine",
            features: vec![
                (LIFE, 0.6),
                (MOTION, 0.6),
                (DIAGNOSTICS, 0.5),
                (PERFORMANCE, 0.4),
                (THERAPY, 0.7),
                (ANATOMY, 0.4),
            ],
        },
        Concept {
            label: "Telemedicine",
            category: "medicine",
            features: vec![
                (DIAGNOSTICS, 0.5),
                (COMPUTATION, 0.6),
                (NETWORK, 0.6),
                (INFORMATION, 0.5),
                (CLINICAL, 0.4),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Evidence-based medicine",
            category: "medicine",
            features: vec![
                (STATISTICS, 0.8),
                (DIAGNOSTICS, 0.6),
                (LOGIC, 0.5),
                (MEASUREMENT, 0.5),
                (CLINICAL, 0.5),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Sports medicine",
            category: "medicine",
            features: vec![
                (LIFE, 0.6),
                (MOTION, 0.7),
                (FORCE, 0.4),
                (DIAGNOSTICS, 0.5),
                (ANATOMY, 0.5),
                (THERAPY, 0.4),
            ],
        },
        Concept {
            label: "Toxicology",
            category: "medicine",
            features: vec![
                (CHEMISTRY, 0.8),
                (LIFE, 0.6),
                (DIAGNOSTICS, 0.5),
                (MEASUREMENT, 0.4),
                (MOLECULAR, 0.4),
                (CLINICAL, 0.3),
            ],
        },
        Concept {
            label: "Medical ethics",
            category: "medicine",
            features: vec![
                (ETHICS, 0.9),
                (LIFE, 0.5),
                (MIND, 0.4),
                (GOVERNANCE, 0.3),
                (MORAL, 0.6),
                (CLINICAL, 0.3),
            ],
        },
        // ── LINGUISTICS (25) ──
        Concept {
            label: "Syntax",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 1.0),
                (STRUCTURE, 0.8),
                (LOGIC, 0.4),
                (PATTERN, 0.4),
                (SYNTAX, 0.9),
                (GRAMMAR, 0.6),
            ],
        },
        Concept {
            label: "Semantics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.9),
                (LOGIC, 0.6),
                (MIND, 0.4),
                (STRUCTURE, 0.4),
                (SEMANTICS_AX, 0.9),
                (CONCEPT, 0.4),
            ],
        },
        Concept {
            label: "Phonology",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (SOUND, 0.7),
                (PATTERN, 0.6),
                (STRUCTURE, 0.4),
                (PHONETIC, 0.7),
                (GRAMMAR, 0.3),
            ],
        },
        Concept {
            label: "Pragmatics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (BEHAVIOR, 0.5),
                (COGNITION, 0.4),
                (MIND, 0.3),
                (DISCOURSE, 0.5),
                (SEMANTICS_AX, 0.3),
            ],
        },
        Concept {
            label: "Psycholinguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (COGNITION, 0.8),
                (MIND, 0.5),
                (LIFE, 0.3),
                (BEHAVIOR, 0.3),
                (BRAIN, 0.4),
            ],
        },
        Concept {
            label: "Computational linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (COMPUTATION, 0.7),
                (PATTERN, 0.5),
                (STATISTICS, 0.5),
                (STRUCTURE, 0.3),
                (LLM, 0.3),
                (ALGORITHM, 0.3),
            ],
        },
        Concept {
            label: "Historical linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.9),
                (EVOLUTION, 0.4),
                (PATTERN, 0.4),
                (STRUCTURE, 0.3),
                (HISTORICAL, 0.5),
                (GRAMMAR, 0.3),
            ],
        },
        Concept {
            label: "Sociolinguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.3),
                (COGNITION, 0.2),
                (SOCIETY, 0.5),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Morphology",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (STRUCTURE, 0.7),
                (PATTERN, 0.5),
                (GRAMMAR, 0.6),
            ],
        },
        Concept {
            label: "Typology",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (STRUCTURE, 0.6),
                (PATTERN, 0.5),
                (SYSTEMS, 0.3),
                (GRAMMAR, 0.4),
            ],
        },
        Concept {
            label: "Corpus linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (STATISTICS, 0.6),
                (COMPUTATION, 0.4),
                (PATTERN, 0.5),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Discourse analysis",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (COGNITION, 0.4),
                (STRUCTURE, 0.4),
                (BEHAVIOR, 0.3),
                (NARRATIVE, 0.4),
                (DISCOURSE, 0.7),
            ],
        },
        Concept {
            label: "Phonetics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (SOUND, 0.9),
                (MEASUREMENT, 0.5),
                (WAVE, 0.3),
                (PHONETIC, 0.9),
                (ANATOMY, 0.2),
            ],
        },
        Concept {
            label: "Language acquisition",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (COGNITION, 0.7),
                (EVOLUTION, 0.3),
                (PEDAGOGY, 0.4),
                (LEARNING, 0.6),
                (BRAIN, 0.3),
            ],
        },
        Concept {
            label: "Neurolinguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (COGNITION, 0.7),
                (LIFE, 0.5),
                (NETWORK, 0.4),
                (MIND, 0.4),
                (BRAIN, 0.6),
                (NEURAL, 0.5),
            ],
        },
        Concept {
            label: "Forensic linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (LOGIC, 0.5),
                (PATTERN, 0.5),
                (BEHAVIOR, 0.3),
                (LEGAL, 0.3),
            ],
        },
        Concept {
            label: "Sign language linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (VISUAL, 0.5),
                (MOTION, 0.4),
                (COGNITION, 0.4),
                (GRAMMAR, 0.4),
            ],
        },
        Concept {
            label: "Translation studies",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.9),
                (COGNITION, 0.5),
                (PATTERN, 0.4),
                (STRUCTURE, 0.3),
                (SEMANTICS_AX, 0.4),
            ],
        },
        Concept {
            label: "Lexicography",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.9),
                (STRUCTURE, 0.6),
                (INFORMATION, 0.5),
                (PATTERN, 0.3),
                (SEMANTICS_AX, 0.4),
                (ARCHIVAL, 0.2),
            ],
        },
        Concept {
            label: "Applied linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (PEDAGOGY, 0.6),
                (COGNITION, 0.4),
                (BEHAVIOR, 0.3),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Dialectology",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (PATTERN, 0.5),
                (EVOLUTION, 0.4),
                (SOUND, 0.3),
                (PHONETIC, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Contact linguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.8),
                (EVOLUTION, 0.5),
                (BEHAVIOR, 0.4),
                (NETWORK, 0.3),
                (GRAMMAR, 0.3),
                (SOCIETY, 0.3),
            ],
        },
        Concept {
            label: "Semiotics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (PATTERN, 0.6),
                (COGNITION, 0.5),
                (VISUAL, 0.4),
                (STRUCTURE, 0.4),
                (SEMANTICS_AX, 0.5),
            ],
        },
        Concept {
            label: "Language documentation",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.9),
                (INFORMATION, 0.5),
                (EVOLUTION, 0.4),
                (MEASUREMENT, 0.3),
                (ARCHIVAL, 0.4),
            ],
        },
        Concept {
            label: "Ecolinguistics",
            category: "linguistics",
            features: vec![
                (LANGUAGE, 0.7),
                (ECOSYSTEM, 0.5),
                (NATURE, 0.4),
                (BEHAVIOR, 0.3),
                (DISCOURSE, 0.3),
            ],
        },
        // ── MATHEMATICS (25) ──
        Concept {
            label: "Number theory",
            category: "mathematics",
            features: vec![
                (MATH, 1.0),
                (LOGIC, 0.7),
                (PATTERN, 0.7),
                (STRUCTURE, 0.6),
                (PROOF, 0.7),
                (ALGEBRA, 0.4),
            ],
        },
        Concept {
            label: "Topology",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STRUCTURE, 0.8),
                (SPACE, 0.7),
                (PATTERN, 0.4),
                (PROOF, 0.5),
                (CONCEPT, 0.4),
            ],
        },
        Concept {
            label: "Real analysis",
            category: "mathematics",
            features: vec![
                (MATH, 1.0),
                (LOGIC, 0.8),
                (STRUCTURE, 0.6),
                (CALCULUS, 0.8),
                (PROOF, 0.7),
            ],
        },
        Concept {
            label: "Complex analysis",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STRUCTURE, 0.7),
                (WAVE, 0.5),
                (PATTERN, 0.4),
                (CALCULUS, 0.6),
            ],
        },
        Concept {
            label: "Abstract algebra",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STRUCTURE, 0.9),
                (LOGIC, 0.7),
                (PATTERN, 0.5),
                (ALGEBRA, 0.9),
                (PROOF, 0.5),
            ],
        },
        Concept {
            label: "Differential geometry",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (SPACE, 0.8),
                (STRUCTURE, 0.6),
                (FORCE, 0.3),
                (CALCULUS, 0.7),
                (RELATIVITY, 0.3),
            ],
        },
        Concept {
            label: "Combinatorics",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (PATTERN, 0.7),
                (LOGIC, 0.6),
                (COMPUTATION, 0.4),
                (GRAPH_THEORY, 0.4),
                (PROOF, 0.4),
            ],
        },
        Concept {
            label: "Category theory",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (LOGIC, 0.8),
                (STRUCTURE, 0.9),
                (PATTERN, 0.5),
                (ALGEBRA, 0.5),
                (CONCEPT, 0.5),
            ],
        },
        Concept {
            label: "Set theory",
            category: "mathematics",
            features: vec![(MATH, 0.9), (LOGIC, 0.9), (STRUCTURE, 0.7), (PROOF, 0.6)],
        },
        Concept {
            label: "Graph theory",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (NETWORK, 0.9),
                (STRUCTURE, 0.7),
                (PATTERN, 0.5),
                (GRAPH_THEORY, 0.9),
            ],
        },
        Concept {
            label: "Probability theory",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STATISTICS, 0.9),
                (LOGIC, 0.6),
                (PATTERN, 0.4),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Mathematical logic",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (LOGIC, 1.0),
                (LANGUAGE, 0.3),
                (STRUCTURE, 0.5),
                (PROOF, 0.8),
            ],
        },
        Concept {
            label: "Numerical analysis",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (COMPUTATION, 0.7),
                (OPTIMIZATION, 0.5),
                (MEASUREMENT, 0.4),
                (ALGORITHM, 0.5),
                (CALCULUS, 0.4),
            ],
        },
        Concept {
            label: "Dynamical systems theory",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (SYSTEMS, 0.8),
                (PATTERN, 0.6),
                (MOTION, 0.5),
                (CALCULUS, 0.5),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Measure theory",
            category: "mathematics",
            features: vec![
                (MATH, 1.0),
                (LOGIC, 0.6),
                (STRUCTURE, 0.7),
                (MEASUREMENT, 0.4),
                (PROOF, 0.5),
                (CALCULUS, 0.4),
            ],
        },
        Concept {
            label: "Algebraic geometry",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STRUCTURE, 0.8),
                (SPACE, 0.6),
                (PATTERN, 0.5),
                (ALGEBRA, 0.7),
            ],
        },
        Concept {
            label: "Functional analysis",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STRUCTURE, 0.7),
                (SPACE, 0.5),
                (LOGIC, 0.6),
                (CALCULUS, 0.5),
                (PROOF, 0.4),
            ],
        },
        Concept {
            label: "Stochastic processes",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (STATISTICS, 0.8),
                (PATTERN, 0.5),
                (ENTROPY, 0.4),
                (CALCULUS, 0.4),
            ],
        },
        Concept {
            label: "Optimization theory",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (OPTIMIZATION, 0.9),
                (COMPUTATION, 0.5),
                (LOGIC, 0.5),
                (CALCULUS, 0.4),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Information theory (math)",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (INFORMATION, 0.9),
                (ENTROPY, 0.7),
                (STATISTICS, 0.5),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Cryptographic mathematics",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (COMPUTATION, 0.6),
                (LOGIC, 0.6),
                (INFORMATION, 0.5),
                (ALGEBRA, 0.5),
            ],
        },
        Concept {
            label: "Representation theory",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (STRUCTURE, 0.8),
                (PATTERN, 0.6),
                (LOGIC, 0.5),
                (ALGEBRA, 0.7),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Differential equations",
            category: "mathematics",
            features: vec![
                (MATH, 0.9),
                (SYSTEMS, 0.6),
                (MOTION, 0.5),
                (FORCE, 0.3),
                (CALCULUS, 0.9),
            ],
        },
        Concept {
            label: "Mathematical statistics",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (STATISTICS, 0.9),
                (MEASUREMENT, 0.5),
                (LOGIC, 0.4),
                (PROOF, 0.3),
            ],
        },
        Concept {
            label: "Game theory (math)",
            category: "mathematics",
            features: vec![
                (MATH, 0.8),
                (OPTIMIZATION, 0.7),
                (LOGIC, 0.7),
                (BEHAVIOR, 0.4),
                (THEORY, 0.5),
            ],
        },
        // ── CHEMISTRY (25) ──
        Concept {
            label: "Organic chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 1.0),
                (STRUCTURE, 0.7),
                (LIFE, 0.4),
                (PATTERN, 0.4),
                (MOLECULAR, 0.8),
                (REACTION, 0.6),
            ],
        },
        Concept {
            label: "Inorganic chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.9),
                (STRUCTURE, 0.7),
                (ENERGY, 0.4),
                (MATERIAL, 0.4),
                (MOLECULAR, 0.5),
                (ATOMIC, 0.4),
            ],
        },
        Concept {
            label: "Physical chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (ENERGY, 0.7),
                (MATH, 0.6),
                (QUANTUM, 0.5),
                (REACTION, 0.5),
                (MOLECULAR, 0.4),
            ],
        },
        Concept {
            label: "Biochemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.9),
                (LIFE, 0.9),
                (STRUCTURE, 0.5),
                (GENETICS, 0.3),
                (MOLECULAR, 0.7),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Analytical chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.9),
                (DIAGNOSTICS, 0.7),
                (MEASUREMENT, 0.7),
                (STATISTICS, 0.4),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Quantum chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (QUANTUM, 0.9),
                (ENERGY, 0.7),
                (MATH, 0.6),
                (MOLECULAR, 0.6),
                (ATOMIC, 0.5),
            ],
        },
        Concept {
            label: "Thermochemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (ENERGY, 0.9),
                (ENTROPY, 0.7),
                (REACTION, 0.7),
                (CYCLE, 0.3),
            ],
        },
        Concept {
            label: "Polymer chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (STRUCTURE, 0.8),
                (MATERIAL, 0.7),
                (PATTERN, 0.5),
                (MOLECULAR, 0.6),
            ],
        },
        Concept {
            label: "Electrochemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.9),
                (ENERGY, 0.7),
                (FORCE, 0.5),
                (MEASUREMENT, 0.4),
                (REACTION, 0.6),
                (ELECTRICAL, 0.4),
            ],
        },
        Concept {
            label: "Green chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (NATURE, 0.7),
                (ETHICS, 0.5),
                (ECOSYSTEM, 0.5),
                (CONSERVATION, 0.4),
                (REACTION, 0.3),
            ],
        },
        Concept {
            label: "Materials chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (STRUCTURE, 0.9),
                (MATERIAL, 0.8),
                (QUANTUM, 0.3),
                (MOLECULAR, 0.5),
            ],
        },
        Concept {
            label: "Astrochemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.7),
                (SPACE, 0.8),
                (QUANTUM, 0.3),
                (ENERGY, 0.4),
                (STELLAR, 0.4),
                (MOLECULAR, 0.4),
            ],
        },
        Concept {
            label: "Nuclear chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.7),
                (QUANTUM, 0.7),
                (ENERGY, 0.9),
                (FORCE, 0.4),
                (ATOMIC, 0.6),
                (REACTION, 0.5),
            ],
        },
        Concept {
            label: "Computational chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.7),
                (COMPUTATION, 0.8),
                (MATH, 0.6),
                (QUANTUM, 0.4),
                (ALGORITHM, 0.3),
                (MOLECULAR, 0.4),
            ],
        },
        Concept {
            label: "Catalysis",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.9),
                (ENERGY, 0.6),
                (OPTIMIZATION, 0.5),
                (PATTERN, 0.3),
                (REACTION, 0.8),
                (MOLECULAR, 0.4),
            ],
        },
        Concept {
            label: "Surface chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (MATERIAL, 0.7),
                (STRUCTURE, 0.6),
                (FORCE, 0.3),
                (SURFACE, 0.7),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Medicinal chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (LIFE, 0.6),
                (DIAGNOSTICS, 0.4),
                (STRUCTURE, 0.5),
                (MOLECULAR, 0.6),
                (THERAPY, 0.3),
            ],
        },
        Concept {
            label: "Spectroscopy",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.7),
                (WAVE, 0.8),
                (MEASUREMENT, 0.8),
                (QUANTUM, 0.4),
                (MOLECULAR, 0.4),
                (ATOMIC, 0.3),
            ],
        },
        Concept {
            label: "Supramolecular chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (STRUCTURE, 0.9),
                (PATTERN, 0.6),
                (SYSTEMS, 0.3),
                (MOLECULAR, 0.7),
            ],
        },
        Concept {
            label: "Solid-state chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (STRUCTURE, 0.8),
                (MATERIAL, 0.7),
                (QUANTUM, 0.3),
                (ATOMIC, 0.4),
            ],
        },
        Concept {
            label: "Environmental chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (NATURE, 0.7),
                (ECOSYSTEM, 0.5),
                (MEASUREMENT, 0.4),
                (REACTION, 0.3),
                (WATER, 0.3),
            ],
        },
        Concept {
            label: "Photochemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (ENERGY, 0.7),
                (WAVE, 0.6),
                (QUANTUM, 0.4),
                (REACTION, 0.5),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Colloid chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.7),
                (MATERIAL, 0.6),
                (FORCE, 0.4),
                (STRUCTURE, 0.5),
                (SURFACE, 0.5),
            ],
        },
        Concept {
            label: "Food chemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (LIFE, 0.4),
                (STRUCTURE, 0.4),
                (MATERIAL, 0.3),
                (MOLECULAR, 0.3),
                (FLAVOR, 0.3),
            ],
        },
        Concept {
            label: "Petrochemistry",
            category: "chemistry",
            features: vec![
                (CHEMISTRY, 0.8),
                (ENERGY, 0.7),
                (MATERIAL, 0.5),
                (NATURE, 0.3),
                (MOLECULAR, 0.4),
            ],
        },
        // ── PSYCHOLOGY (25) ──
        Concept {
            label: "Cognitive psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.8),
                (COGNITION, 1.0),
                (PATTERN, 0.5),
                (LOGIC, 0.4),
                (BRAIN, 0.4),
                (CONCEPT, 0.3),
            ],
        },
        Concept {
            label: "Developmental psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.7),
                (BEHAVIOR, 0.6),
                (COGNITION, 0.6),
                (EVOLUTION, 0.3),
                (ATTACHMENT, 0.4),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Social psychology",
            category: "psychology",
            features: vec![
                (BEHAVIOR, 0.9),
                (MIND, 0.7),
                (NETWORK, 0.6),
                (COGNITION, 0.4),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Clinical psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.8),
                (DIAGNOSTICS, 0.7),
                (BEHAVIOR, 0.6),
                (EMOTION, 0.6),
                (MENTAL_HEALTH, 0.7),
                (THERAPY, 0.6),
            ],
        },
        Concept {
            label: "Behavioral psychology",
            category: "psychology",
            features: vec![
                (BEHAVIOR, 1.0),
                (MIND, 0.6),
                (PATTERN, 0.5),
                (LEARNING, 0.5),
            ],
        },
        Concept {
            label: "Personality psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.7),
                (BEHAVIOR, 0.7),
                (PATTERN, 0.5),
                (EMOTION, 0.4),
                (ATTACHMENT, 0.3),
            ],
        },
        Concept {
            label: "Neuropsychology",
            category: "psychology",
            features: vec![
                (MIND, 0.8),
                (COGNITION, 0.7),
                (LIFE, 0.5),
                (DIAGNOSTICS, 0.4),
                (BRAIN, 0.7),
                (NEURAL, 0.5),
            ],
        },
        Concept {
            label: "Educational psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.6),
                (COGNITION, 0.7),
                (PEDAGOGY, 0.7),
                (BEHAVIOR, 0.5),
                (LEARNING, 0.6),
            ],
        },
        Concept {
            label: "Organizational psychology",
            category: "psychology",
            features: vec![
                (BEHAVIOR, 0.8),
                (SYSTEMS, 0.6),
                (MARKETS, 0.4),
                (NETWORK, 0.4),
                (LABOR, 0.3),
            ],
        },
        Concept {
            label: "Positive psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.7),
                (EMOTION, 0.9),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.3),
                (MENTAL_HEALTH, 0.5),
            ],
        },
        Concept {
            label: "Forensic psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.6),
                (BEHAVIOR, 0.7),
                (ETHICS, 0.5),
                (LOGIC, 0.4),
                (LEGAL, 0.4),
            ],
        },
        Concept {
            label: "Psychometrics",
            category: "psychology",
            features: vec![
                (MIND, 0.6),
                (STATISTICS, 0.9),
                (COGNITION, 0.5),
                (MEASUREMENT, 0.6),
                (ASSESSMENT, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Humanistic psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.8),
                (EMOTION, 0.8),
                (ETHICS, 0.5),
                (METAPHYSICS, 0.3),
                (CONSCIOUSNESS, 0.3),
            ],
        },
        Concept {
            label: "Gestalt psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.7),
                (PATTERN, 0.8),
                (COGNITION, 0.7),
                (VISUAL, 0.5),
                (CONCEPT, 0.3),
            ],
        },
        Concept {
            label: "Abnormal psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.7),
                (BEHAVIOR, 0.7),
                (DIAGNOSTICS, 0.6),
                (EMOTION, 0.5),
                (MENTAL_HEALTH, 0.6),
                (TRAUMA, 0.3),
            ],
        },
        Concept {
            label: "Perception psychology",
            category: "psychology",
            features: vec![
                (COGNITION, 0.8),
                (VISUAL, 0.6),
                (MIND, 0.6),
                (PATTERN, 0.5),
                (SOUND, 0.3),
                (BRAIN, 0.3),
            ],
        },
        Concept {
            label: "Memory studies",
            category: "psychology",
            features: vec![
                (COGNITION, 0.9),
                (MIND, 0.7),
                (INFORMATION, 0.5),
                (PATTERN, 0.4),
                (BRAIN, 0.4),
                (NEURAL, 0.3),
            ],
        },
        Concept {
            label: "Psychopharmacology",
            category: "psychology",
            features: vec![
                (MIND, 0.6),
                (CHEMISTRY, 0.6),
                (LIFE, 0.5),
                (BEHAVIOR, 0.4),
                (THERAPY, 0.4),
                (NEURAL, 0.3),
            ],
        },
        Concept {
            label: "Environmental psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.5),
                (BEHAVIOR, 0.6),
                (NATURE, 0.5),
                (SPACE, 0.4),
                (EMOTION, 0.3),
            ],
        },
        Concept {
            label: "Sport psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.6),
                (BEHAVIOR, 0.6),
                (PERFORMANCE, 0.7),
                (MOTION, 0.4),
            ],
        },
        Concept {
            label: "Consumer psychology",
            category: "psychology",
            features: vec![
                (BEHAVIOR, 0.8),
                (MARKETS, 0.6),
                (COGNITION, 0.5),
                (EMOTION, 0.4),
                (FINANCE, 0.2),
            ],
        },
        Concept {
            label: "Cross-cultural psychology",
            category: "psychology",
            features: vec![
                (BEHAVIOR, 0.7),
                (MIND, 0.5),
                (LANGUAGE, 0.4),
                (PATTERN, 0.4),
                (CULTURE, 0.5),
            ],
        },
        Concept {
            label: "Health psychology",
            category: "psychology",
            features: vec![
                (MIND, 0.6),
                (LIFE, 0.5),
                (BEHAVIOR, 0.7),
                (DIAGNOSTICS, 0.3),
                (MENTAL_HEALTH, 0.5),
            ],
        },
        Concept {
            label: "Evolutionary psychology (psych)",
            category: "psychology",
            features: vec![
                (EVOLUTION, 0.7),
                (MIND, 0.6),
                (BEHAVIOR, 0.7),
                (COGNITION, 0.5),
                (BRAIN, 0.3),
            ],
        },
        Concept {
            label: "Attention and decision science",
            category: "psychology",
            features: vec![
                (COGNITION, 0.8),
                (MIND, 0.6),
                (BEHAVIOR, 0.5),
                (OPTIMIZATION, 0.4),
                (INFORMATION, 0.3),
                (BRAIN, 0.3),
            ],
        },
        // ── ENGINEERING (25) ──
        Concept {
            label: "Mechanical engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 0.8),
                (FORCE, 0.9),
                (ENERGY, 0.6),
                (SYSTEMS, 0.5),
                (MOTION, 0.5),
                (MECHANICAL, 0.9),
            ],
        },
        Concept {
            label: "Electrical engineering",
            category: "engineering",
            features: vec![
                (ENERGY, 0.8),
                (WAVE, 0.7),
                (FORCE, 0.5),
                (SYSTEMS, 0.6),
                (ELECTRICAL, 0.9),
            ],
        },
        Concept {
            label: "Civil engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 1.0),
                (FORCE, 0.7),
                (SYSTEMS, 0.5),
                (MATERIAL, 0.5),
                (MECHANICAL, 0.4),
            ],
        },
        Concept {
            label: "Chemical engineering",
            category: "engineering",
            features: vec![
                (CHEMISTRY, 0.9),
                (SYSTEMS, 0.7),
                (OPTIMIZATION, 0.6),
                (MATERIAL, 0.4),
                (REACTION, 0.5),
            ],
        },
        Concept {
            label: "Aerospace engineering",
            category: "engineering",
            features: vec![
                (FORCE, 0.7),
                (ENERGY, 0.7),
                (SPACE, 0.7),
                (STRUCTURE, 0.6),
                (MOTION, 0.5),
                (MECHANICAL, 0.5),
                (TRANSPORTATION, 0.4),
            ],
        },
        Concept {
            label: "Materials science",
            category: "engineering",
            features: vec![
                (STRUCTURE, 0.9),
                (MATERIAL, 0.9),
                (CHEMISTRY, 0.7),
                (QUANTUM, 0.3),
                (ATOMIC, 0.3),
            ],
        },
        Concept {
            label: "Control theory",
            category: "engineering",
            features: vec![
                (SYSTEMS, 0.9),
                (OPTIMIZATION, 0.8),
                (MATH, 0.7),
                (FORCE, 0.4),
                (THEORY, 0.4),
                (CALCULUS, 0.3),
            ],
        },
        Concept {
            label: "Robotics",
            category: "engineering",
            features: vec![
                (COMPUTATION, 0.8),
                (FORCE, 0.6),
                (SYSTEMS, 0.7),
                (MOTION, 0.7),
                (MECHANICAL, 0.5),
                (AI, 0.3),
            ],
        },
        Concept {
            label: "Biomedical engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 0.6),
                (LIFE, 0.7),
                (SYSTEMS, 0.5),
                (DIAGNOSTICS, 0.4),
                (CLINICAL, 0.3),
                (ANATOMY, 0.3),
            ],
        },
        Concept {
            label: "Systems engineering",
            category: "engineering",
            features: vec![
                (SYSTEMS, 1.0),
                (OPTIMIZATION, 0.7),
                (STRUCTURE, 0.6),
                (LOGIC, 0.4),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Nuclear engineering",
            category: "engineering",
            features: vec![
                (ENERGY, 0.9),
                (QUANTUM, 0.6),
                (SYSTEMS, 0.5),
                (STRUCTURE, 0.4),
                (ATOMIC, 0.5),
            ],
        },
        Concept {
            label: "Environmental engineering",
            category: "engineering",
            features: vec![
                (NATURE, 0.7),
                (ECOSYSTEM, 0.5),
                (SYSTEMS, 0.7),
                (CHEMISTRY, 0.5),
                (CONSERVATION, 0.3),
                (WATER, 0.3),
            ],
        },
        Concept {
            label: "Software engineering",
            category: "engineering",
            features: vec![
                (COMPUTATION, 0.9),
                (STRUCTURE, 0.8),
                (LOGIC, 0.7),
                (SYSTEMS, 0.6),
                (SOFTWARE, 0.9),
            ],
        },
        Concept {
            label: "Industrial engineering",
            category: "engineering",
            features: vec![
                (OPTIMIZATION, 0.9),
                (SYSTEMS, 0.7),
                (MARKETS, 0.5),
                (STATISTICS, 0.4),
                (LABOR, 0.3),
                (MECHANICAL, 0.3),
            ],
        },
        Concept {
            label: "Structural engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 1.0),
                (FORCE, 0.8),
                (MATH, 0.5),
                (MATERIAL, 0.6),
                (MECHANICAL, 0.5),
            ],
        },
        Concept {
            label: "Mechatronics",
            category: "engineering",
            features: vec![
                (SYSTEMS, 0.7),
                (COMPUTATION, 0.6),
                (FORCE, 0.6),
                (MOTION, 0.6),
                (MECHANICAL, 0.6),
                (ELECTRICAL, 0.5),
            ],
        },
        Concept {
            label: "Nanotechnology engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 0.7),
                (QUANTUM, 0.5),
                (MATERIAL, 0.8),
                (CHEMISTRY, 0.5),
                (NANO, 0.7),
            ],
        },
        Concept {
            label: "Renewable energy engineering",
            category: "engineering",
            features: vec![
                (ENERGY, 0.9),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.4),
                (SYSTEMS, 0.5),
                (ELECTRICAL, 0.4),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Telecommunications engineering",
            category: "engineering",
            features: vec![
                (WAVE, 0.7),
                (NETWORK, 0.8),
                (INFORMATION, 0.6),
                (SYSTEMS, 0.5),
                (ELECTRICAL, 0.5),
            ],
        },
        Concept {
            label: "Geotechnical engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 0.7),
                (FORCE, 0.6),
                (NATURE, 0.5),
                (MATERIAL, 0.5),
                (GEOLOGY, 0.4),
            ],
        },
        Concept {
            label: "Hydraulic engineering",
            category: "engineering",
            features: vec![
                (FORCE, 0.7),
                (ENERGY, 0.5),
                (MOTION, 0.6),
                (NATURE, 0.4),
                (WATER, 0.6),
                (MECHANICAL, 0.3),
            ],
        },
        Concept {
            label: "Manufacturing engineering",
            category: "engineering",
            features: vec![
                (STRUCTURE, 0.6),
                (MATERIAL, 0.7),
                (OPTIMIZATION, 0.6),
                (SYSTEMS, 0.6),
                (MECHANICAL, 0.5),
            ],
        },
        Concept {
            label: "Reliability engineering",
            category: "engineering",
            features: vec![
                (SYSTEMS, 0.8),
                (STATISTICS, 0.6),
                (OPTIMIZATION, 0.5),
                (MEASUREMENT, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Acoustical engineering",
            category: "engineering",
            features: vec![
                (WAVE, 0.8),
                (SOUND, 0.7),
                (STRUCTURE, 0.5),
                (MEASUREMENT, 0.4),
                (TIMBRE, 0.3),
            ],
        },
        Concept {
            label: "Optical engineering",
            category: "engineering",
            features: vec![
                (WAVE, 0.8),
                (VISUAL, 0.5),
                (MEASUREMENT, 0.6),
                (ENERGY, 0.4),
                (ELECTRICAL, 0.3),
            ],
        },
        // ── EARTH SCIENCE (25) ──
        Concept {
            label: "Geology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.9),
                (STRUCTURE, 0.8),
                (PATTERN, 0.5),
                (EVOLUTION, 0.4),
                (MATERIAL, 0.5),
                (GEOLOGY, 0.9),
            ],
        },
        Concept {
            label: "Meteorology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (SYSTEMS, 0.7),
                (PATTERN, 0.6),
                (ENERGY, 0.5),
                (CLIMATE, 0.6),
            ],
        },
        Concept {
            label: "Oceanography",
            category: "earth_science",
            features: vec![
                (NATURE, 0.9),
                (SYSTEMS, 0.7),
                (LIFE, 0.5),
                (ECOSYSTEM, 0.5),
                (OCEAN, 0.9),
                (WATER, 0.5),
            ],
        },
        Concept {
            label: "Climatology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (SYSTEMS, 0.8),
                (STATISTICS, 0.6),
                (ECOSYSTEM, 0.6),
                (CLIMATE, 0.9),
                (CYCLE, 0.3),
            ],
        },
        Concept {
            label: "Volcanology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (ENERGY, 0.7),
                (FORCE, 0.6),
                (MATERIAL, 0.5),
                (GEOLOGY, 0.6),
            ],
        },
        Concept {
            label: "Seismology",
            category: "earth_science",
            features: vec![
                (WAVE, 0.9),
                (FORCE, 0.7),
                (NATURE, 0.7),
                (MEASUREMENT, 0.5),
                (GEOLOGY, 0.5),
            ],
        },
        Concept {
            label: "Paleontology",
            category: "earth_science",
            features: vec![
                (LIFE, 0.8),
                (EVOLUTION, 0.9),
                (NATURE, 0.7),
                (STRUCTURE, 0.5),
                (GEOLOGY, 0.4),
                (HISTORICAL, 0.3),
            ],
        },
        Concept {
            label: "Hydrology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (SYSTEMS, 0.6),
                (FORCE, 0.5),
                (ECOSYSTEM, 0.4),
                (WATER, 0.8),
                (CYCLE, 0.5),
            ],
        },
        Concept {
            label: "Geomorphology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (STRUCTURE, 0.7),
                (FORCE, 0.5),
                (PATTERN, 0.5),
                (GEOLOGY, 0.5),
            ],
        },
        Concept {
            label: "Mineralogy",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (STRUCTURE, 0.8),
                (CHEMISTRY, 0.7),
                (MATERIAL, 0.6),
                (GEOLOGY, 0.5),
                (ATOMIC, 0.3),
            ],
        },
        Concept {
            label: "Geophysics",
            category: "earth_science",
            features: vec![
                (NATURE, 0.6),
                (ENERGY, 0.7),
                (FORCE, 0.7),
                (WAVE, 0.6),
                (GEOLOGY, 0.5),
                (PLANETARY, 0.3),
            ],
        },
        Concept {
            label: "Geochemistry",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (CHEMISTRY, 0.8),
                (STRUCTURE, 0.5),
                (MEASUREMENT, 0.4),
                (GEOLOGY, 0.5),
            ],
        },
        Concept {
            label: "Atmospheric science",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (ENERGY, 0.6),
                (SYSTEMS, 0.6),
                (WAVE, 0.4),
                (CLIMATE, 0.6),
                (PLANETARY, 0.3),
            ],
        },
        Concept {
            label: "Glaciology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.9),
                (ENERGY, 0.5),
                (STRUCTURE, 0.6),
                (ECOSYSTEM, 0.4),
                (CLIMATE, 0.5),
                (WATER, 0.4),
            ],
        },
        Concept {
            label: "Geodesy",
            category: "earth_science",
            features: vec![
                (SPACE, 0.8),
                (MATH, 0.7),
                (NATURE, 0.6),
                (MEASUREMENT, 0.7),
                (PLANETARY, 0.3),
            ],
        },
        Concept {
            label: "Soil science",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (CHEMISTRY, 0.5),
                (LIFE, 0.4),
                (ECOSYSTEM, 0.5),
                (MATERIAL, 0.4),
                (GEOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Stratigraphy",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (STRUCTURE, 0.7),
                (EVOLUTION, 0.6),
                (PATTERN, 0.5),
                (GEOLOGY, 0.6),
                (HISTORICAL, 0.3),
            ],
        },
        Concept {
            label: "Petrology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (CHEMISTRY, 0.6),
                (MATERIAL, 0.7),
                (STRUCTURE, 0.6),
                (GEOLOGY, 0.6),
            ],
        },
        Concept {
            label: "Speleology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.8),
                (STRUCTURE, 0.6),
                (CHEMISTRY, 0.3),
                (SPACE, 0.3),
                (GEOLOGY, 0.5),
                (WATER, 0.3),
            ],
        },
        Concept {
            label: "Paleoecology",
            category: "earth_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (EVOLUTION, 0.7),
                (LIFE, 0.6),
                (NATURE, 0.7),
                (CLIMATE, 0.4),
            ],
        },
        Concept {
            label: "Remote sensing",
            category: "earth_science",
            features: vec![
                (NATURE, 0.5),
                (WAVE, 0.6),
                (MEASUREMENT, 0.8),
                (COMPUTATION, 0.5),
                (VISUAL, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Geochronology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.6),
                (MEASUREMENT, 0.8),
                (EVOLUTION, 0.6),
                (CHEMISTRY, 0.4),
                (GEOLOGY, 0.4),
            ],
        },
        Concept {
            label: "Paleoclimatology",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (ECOSYSTEM, 0.6),
                (EVOLUTION, 0.5),
                (STATISTICS, 0.4),
                (CLIMATE, 0.7),
                (HISTORICAL, 0.3),
            ],
        },
        Concept {
            label: "Tectonics",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (FORCE, 0.8),
                (STRUCTURE, 0.7),
                (MOTION, 0.5),
                (GEOLOGY, 0.6),
            ],
        },
        Concept {
            label: "Natural hazards science",
            category: "earth_science",
            features: vec![
                (NATURE, 0.7),
                (FORCE, 0.6),
                (SYSTEMS, 0.5),
                (STATISTICS, 0.4),
                (MEASUREMENT, 0.3),
                (CLIMATE, 0.3),
            ],
        },
        // ── ASTRONOMY (25) ──
        Concept {
            label: "Observational astronomy",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (WAVE, 0.7),
                (PATTERN, 0.5),
                (MEASUREMENT, 0.6),
                (CELESTIAL, 0.7),
            ],
        },
        Concept {
            label: "Astrophysics",
            category: "astronomy",
            features: vec![
                (SPACE, 1.0),
                (ENERGY, 0.9),
                (QUANTUM, 0.7),
                (MATH, 0.6),
                (STELLAR, 0.6),
                (RELATIVITY, 0.5),
            ],
        },
        Concept {
            label: "Planetary science",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (STRUCTURE, 0.6),
                (CHEMISTRY, 0.5),
                (NATURE, 0.4),
                (PLANETARY, 0.9),
                (ORBIT, 0.5),
            ],
        },
        Concept {
            label: "Exoplanet research",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (LIFE, 0.4),
                (WAVE, 0.6),
                (MEASUREMENT, 0.5),
                (PLANETARY, 0.8),
                (ORBIT, 0.5),
            ],
        },
        Concept {
            label: "Stellar evolution",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (EVOLUTION, 0.7),
                (ENERGY, 0.8),
                (QUANTUM, 0.4),
                (STELLAR, 0.9),
                (CYCLE, 0.4),
            ],
        },
        Concept {
            label: "Radio astronomy",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (WAVE, 0.9),
                (INFORMATION, 0.5),
                (MEASUREMENT, 0.5),
                (CELESTIAL, 0.4),
            ],
        },
        Concept {
            label: "Astrobiology",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (LIFE, 0.8),
                (CHEMISTRY, 0.6),
                (EVOLUTION, 0.5),
                (PLANETARY, 0.5),
            ],
        },
        Concept {
            label: "Galactic dynamics",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (FORCE, 0.6),
                (NETWORK, 0.5),
                (MATH, 0.5),
                (MOTION, 0.4),
                (CELESTIAL, 0.5),
                (ORBIT, 0.4),
            ],
        },
        Concept {
            label: "Solar physics",
            category: "astronomy",
            features: vec![
                (SPACE, 0.7),
                (ENERGY, 0.9),
                (WAVE, 0.7),
                (QUANTUM, 0.5),
                (STELLAR, 0.8),
            ],
        },
        Concept {
            label: "Gravitational wave astronomy",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (WAVE, 0.9),
                (FORCE, 0.8),
                (MATH, 0.5),
                (RELATIVITY, 0.8),
            ],
        },
        Concept {
            label: "Dark matter studies",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (QUANTUM, 0.7),
                (ENERGY, 0.6),
                (MATH, 0.7),
                (PARTICLE, 0.5),
                (CELESTIAL, 0.4),
            ],
        },
        Concept {
            label: "High-energy astrophysics",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (ENERGY, 1.0),
                (QUANTUM, 0.7),
                (FORCE, 0.5),
                (PARTICLE, 0.5),
                (STELLAR, 0.4),
            ],
        },
        Concept {
            label: "Astrometry",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (MATH, 0.7),
                (MEASUREMENT, 0.7),
                (STATISTICS, 0.4),
                (CELESTIAL, 0.5),
            ],
        },
        Concept {
            label: "Star formation",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (ENERGY, 0.7),
                (STRUCTURE, 0.6),
                (ENTROPY, 0.4),
                (STELLAR, 0.7),
            ],
        },
        Concept {
            label: "Cosmic microwave background",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (WAVE, 0.8),
                (ENTROPY, 0.5),
                (ENERGY, 0.5),
                (RELATIVITY, 0.4),
                (CELESTIAL, 0.5),
            ],
        },
        Concept {
            label: "Binary star systems",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (FORCE, 0.6),
                (ENERGY, 0.5),
                (MOTION, 0.5),
                (STELLAR, 0.7),
                (ORBIT, 0.7),
            ],
        },
        Concept {
            label: "Infrared astronomy",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (WAVE, 0.8),
                (MEASUREMENT, 0.5),
                (ENERGY, 0.4),
                (CELESTIAL, 0.4),
            ],
        },
        Concept {
            label: "X-ray astronomy",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (WAVE, 0.7),
                (ENERGY, 0.7),
                (MEASUREMENT, 0.5),
                (STELLAR, 0.4),
            ],
        },
        Concept {
            label: "Asteroid science",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (STRUCTURE, 0.5),
                (NATURE, 0.4),
                (FORCE, 0.4),
                (PLANETARY, 0.6),
                (ORBIT, 0.5),
            ],
        },
        Concept {
            label: "Nebula studies",
            category: "astronomy",
            features: vec![
                (SPACE, 0.8),
                (CHEMISTRY, 0.5),
                (ENERGY, 0.5),
                (STRUCTURE, 0.4),
                (STELLAR, 0.5),
                (CELESTIAL, 0.4),
            ],
        },
        Concept {
            label: "Black hole physics",
            category: "astronomy",
            features: vec![
                (SPACE, 0.9),
                (ENERGY, 0.8),
                (ENTROPY, 0.7),
                (MATH, 0.6),
                (QUANTUM, 0.5),
                (RELATIVITY, 0.8),
            ],
        },
        Concept {
            label: "Cosmochemistry",
            category: "astronomy",
            features: vec![
                (SPACE, 0.7),
                (CHEMISTRY, 0.7),
                (EVOLUTION, 0.4),
                (MEASUREMENT, 0.4),
                (STELLAR, 0.3),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Pulsar timing",
            category: "astronomy",
            features: vec![
                (SPACE, 0.7),
                (WAVE, 0.7),
                (MEASUREMENT, 0.7),
                (MATH, 0.5),
                (STELLAR, 0.5),
                (CYCLE, 0.4),
            ],
        },
        Concept {
            label: "Heliophysics",
            category: "astronomy",
            features: vec![
                (SPACE, 0.7),
                (ENERGY, 0.8),
                (WAVE, 0.6),
                (FORCE, 0.4),
                (STELLAR, 0.6),
                (PLANETARY, 0.3),
            ],
        },
        Concept {
            label: "Telescope instrumentation",
            category: "astronomy",
            features: vec![
                (SPACE, 0.6),
                (MEASUREMENT, 0.9),
                (WAVE, 0.6),
                (COMPUTATION, 0.4),
                (CELESTIAL, 0.3),
            ],
        },
        // ── VISUAL ARTS (25) ──
        Concept {
            label: "Painting",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.9),
                (PATTERN, 0.8),
                (EMOTION, 0.8),
                (MATERIAL, 0.4),
                (COLOR, 0.7),
                (FORM, 0.5),
            ],
        },
        Concept {
            label: "Sculpture",
            category: "visual_arts",
            features: vec![
                (STRUCTURE, 0.8),
                (SPACE, 0.7),
                (EMOTION, 0.6),
                (MATERIAL, 0.7),
                (FORM, 0.8),
            ],
        },
        Concept {
            label: "Drawing",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.8),
                (PATTERN, 0.7),
                (PERFORMANCE, 0.5),
                (EMOTION, 0.5),
                (FORM, 0.5),
            ],
        },
        Concept {
            label: "Printmaking",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.7),
                (PATTERN, 0.7),
                (MATERIAL, 0.5),
                (PERFORMANCE, 0.4),
                (DESIGN, 0.3),
            ],
        },
        Concept {
            label: "Photography",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.9),
                (WAVE, 0.5),
                (EMOTION, 0.6),
                (PATTERN, 0.5),
                (COLOR, 0.4),
            ],
        },
        Concept {
            label: "Digital art",
            category: "visual_arts",
            features: vec![
                (COMPUTATION, 0.7),
                (VISUAL, 0.8),
                (PATTERN, 0.7),
                (EMOTION, 0.5),
                (DESIGN, 0.4),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Art history",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.6),
                (NARRATIVE, 0.5),
                (PATTERN, 0.6),
                (EVOLUTION, 0.4),
                (HISTORICAL, 0.5),
            ],
        },
        Concept {
            label: "Art theory",
            category: "visual_arts",
            features: vec![
                (METAPHYSICS, 0.6),
                (EMOTION, 0.6),
                (VISUAL, 0.5),
                (MIND, 0.5),
                (THEORY, 0.5),
                (FORM, 0.3),
            ],
        },
        Concept {
            label: "Illustration",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.8),
                (NARRATIVE, 0.5),
                (EMOTION, 0.6),
                (PERFORMANCE, 0.3),
                (DESIGN, 0.4),
            ],
        },
        Concept {
            label: "Ceramics",
            category: "visual_arts",
            features: vec![
                (MATERIAL, 0.7),
                (STRUCTURE, 0.7),
                (CHEMISTRY, 0.5),
                (PERFORMANCE, 0.5),
                (FORM, 0.5),
            ],
        },
        Concept {
            label: "Textile arts",
            category: "visual_arts",
            features: vec![
                (PATTERN, 0.8),
                (MATERIAL, 0.7),
                (STRUCTURE, 0.6),
                (VISUAL, 0.5),
                (DESIGN, 0.4),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Calligraphy",
            category: "visual_arts",
            features: vec![
                (LANGUAGE, 0.7),
                (VISUAL, 0.7),
                (PATTERN, 0.7),
                (PERFORMANCE, 0.6),
                (FORM, 0.4),
            ],
        },
        Concept {
            label: "Installation art",
            category: "visual_arts",
            features: vec![
                (SPACE, 0.7),
                (EMOTION, 0.7),
                (VISUAL, 0.6),
                (STRUCTURE, 0.4),
                (FORM, 0.4),
            ],
        },
        Concept {
            label: "Conceptual art",
            category: "visual_arts",
            features: vec![
                (MIND, 0.7),
                (METAPHYSICS, 0.6),
                (EMOTION, 0.5),
                (VISUAL, 0.3),
                (CONCEPT, 0.5),
            ],
        },
        Concept {
            label: "Glassblowing",
            category: "visual_arts",
            features: vec![
                (MATERIAL, 0.7),
                (CHEMISTRY, 0.5),
                (ENERGY, 0.4),
                (PERFORMANCE, 0.7),
                (FORM, 0.4),
            ],
        },
        Concept {
            label: "Mosaic art",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.7),
                (PATTERN, 0.8),
                (MATERIAL, 0.6),
                (STRUCTURE, 0.5),
                (COLOR, 0.4),
            ],
        },
        Concept {
            label: "Street art",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.8),
                (EMOTION, 0.7),
                (BEHAVIOR, 0.4),
                (SPACE, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Color theory",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.9),
                (WAVE, 0.4),
                (PATTERN, 0.6),
                (COGNITION, 0.3),
                (COLOR, 0.9),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Mixed media",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.7),
                (MATERIAL, 0.7),
                (PATTERN, 0.5),
                (EMOTION, 0.5),
                (FORM, 0.3),
            ],
        },
        Concept {
            label: "Graphic design",
            category: "visual_arts",
            features: vec![
                (VISUAL, 0.8),
                (PATTERN, 0.7),
                (LANGUAGE, 0.4),
                (COMPUTATION, 0.3),
                (DESIGN, 0.8),
                (COLOR, 0.4),
            ],
        },
        Concept {
            label: "Land art",
            category: "visual_arts",
            features: vec![
                (NATURE, 0.7),
                (SPACE, 0.7),
                (EMOTION, 0.5),
                (MATERIAL, 0.5),
                (FORM, 0.4),
            ],
        },
        Concept {
            label: "Performance art (visual)",
            category: "visual_arts",
            features: vec![
                (PERFORMANCE, 0.8),
                (EMOTION, 0.7),
                (VISUAL, 0.5),
                (BEHAVIOR, 0.5),
                (MOTION, 0.4),
                (THEATRICAL, 0.3),
            ],
        },
        Concept {
            label: "Bookbinding",
            category: "visual_arts",
            features: vec![
                (MATERIAL, 0.7),
                (STRUCTURE, 0.6),
                (PATTERN, 0.5),
                (LANGUAGE, 0.3),
                (DESIGN, 0.3),
            ],
        },
        Concept {
            label: "Jewelry making",
            category: "visual_arts",
            features: vec![
                (MATERIAL, 0.8),
                (STRUCTURE, 0.6),
                (VISUAL, 0.6),
                (CHEMISTRY, 0.3),
                (DESIGN, 0.4),
                (FORM, 0.3),
            ],
        },
        Concept {
            label: "Restoration and conservation",
            category: "visual_arts",
            features: vec![
                (MATERIAL, 0.7),
                (CHEMISTRY, 0.6),
                (VISUAL, 0.5),
                (EVOLUTION, 0.4),
                (ARCHIVAL, 0.3),
            ],
        },
        // ── LITERATURE (25) ──
        Concept {
            label: "Poetry",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.9),
                (EMOTION, 0.9),
                (PATTERN, 0.8),
                (SOUND, 0.6),
                (POETRY, 0.9),
                (RHYTHM, 0.4),
            ],
        },
        Concept {
            label: "Fiction",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.9),
                (EMOTION, 0.7),
                (NARRATIVE, 0.8),
                (COGNITION, 0.4),
                (LITERARY, 0.6),
            ],
        },
        Concept {
            label: "Drama",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (PERFORMANCE, 0.9),
                (EMOTION, 0.7),
                (NARRATIVE, 0.6),
                (THEATRICAL, 0.7),
                (LITERARY, 0.4),
            ],
        },
        Concept {
            label: "Literary criticism",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (LOGIC, 0.6),
                (STRUCTURE, 0.5),
                (MIND, 0.5),
                (LITERARY, 0.7),
                (DISCOURSE, 0.4),
            ],
        },
        Concept {
            label: "Comparative literature",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.9),
                (PATTERN, 0.6),
                (EVOLUTION, 0.4),
                (STRUCTURE, 0.3),
                (LITERARY, 0.6),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Narrative theory",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (NARRATIVE, 0.9),
                (STRUCTURE, 0.7),
                (COGNITION, 0.5),
                (THEORY, 0.5),
                (LITERARY, 0.4),
            ],
        },
        Concept {
            label: "Rhetoric",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (BEHAVIOR, 0.6),
                (LOGIC, 0.5),
                (COGNITION, 0.4),
                (DISCOURSE, 0.6),
            ],
        },
        Concept {
            label: "Creative nonfiction",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (NARRATIVE, 0.7),
                (EMOTION, 0.6),
                (BEHAVIOR, 0.3),
                (LITERARY, 0.4),
            ],
        },
        Concept {
            label: "Science fiction studies",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (NARRATIVE, 0.6),
                (METAPHYSICS, 0.5),
                (COGNITION, 0.5),
                (LITERARY, 0.4),
            ],
        },
        Concept {
            label: "Children's literature",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (EMOTION, 0.7),
                (NARRATIVE, 0.6),
                (PEDAGOGY, 0.4),
                (LEARNING, 0.3),
            ],
        },
        Concept {
            label: "Literary theory",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (METAPHYSICS, 0.5),
                (MIND, 0.5),
                (STRUCTURE, 0.5),
                (LITERARY, 0.8),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Digital humanities",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.6),
                (COMPUTATION, 0.7),
                (PATTERN, 0.6),
                (STATISTICS, 0.5),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Translation studies",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.9),
                (COGNITION, 0.5),
                (PATTERN, 0.4),
                (STRUCTURE, 0.3),
                (SEMANTICS_AX, 0.3),
            ],
        },
        Concept {
            label: "Oral tradition",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (SOUND, 0.7),
                (EVOLUTION, 0.5),
                (NARRATIVE, 0.6),
                (TRADITION, 0.5),
            ],
        },
        Concept {
            label: "Biographical writing",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (NARRATIVE, 0.7),
                (BEHAVIOR, 0.5),
                (EMOTION, 0.5),
                (HISTORICAL, 0.3),
            ],
        },
        Concept {
            label: "Postcolonial literature",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (BEHAVIOR, 0.5),
                (EMOTION, 0.5),
                (GOVERNANCE, 0.3),
                (POWER, 0.3),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Gothic literature",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (EMOTION, 0.8),
                (NARRATIVE, 0.6),
                (METAPHYSICS, 0.3),
                (LITERARY, 0.5),
            ],
        },
        Concept {
            label: "Ecocriticism",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.6),
                (NATURE, 0.7),
                (ECOSYSTEM, 0.5),
                (ETHICS, 0.3),
                (LITERARY, 0.4),
                (DISCOURSE, 0.3),
            ],
        },
        Concept {
            label: "Autobiography studies",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (NARRATIVE, 0.8),
                (MIND, 0.5),
                (EMOTION, 0.5),
                (LITERARY, 0.4),
            ],
        },
        Concept {
            label: "Screenwriting",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (NARRATIVE, 0.8),
                (VISUAL, 0.4),
                (PERFORMANCE, 0.4),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Graphic novel studies",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.6),
                (VISUAL, 0.7),
                (NARRATIVE, 0.7),
                (PATTERN, 0.4),
                (LITERARY, 0.4),
            ],
        },
        Concept {
            label: "Satire studies",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.8),
                (BEHAVIOR, 0.6),
                (EMOTION, 0.5),
                (ETHICS, 0.3),
                (LITERARY, 0.5),
                (MORAL, 0.3),
            ],
        },
        Concept {
            label: "Myth studies",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (NARRATIVE, 0.7),
                (METAPHYSICS, 0.6),
                (EVOLUTION, 0.4),
                (SACRED, 0.3),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Travel writing",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (NARRATIVE, 0.6),
                (SPACE, 0.4),
                (NATURE, 0.3),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Literary magazines and publishing",
            category: "literature",
            features: vec![
                (LANGUAGE, 0.7),
                (INFORMATION, 0.5),
                (MARKETS, 0.3),
                (SYSTEMS, 0.3),
                (LITERARY, 0.3),
            ],
        },
        // ── HISTORY (25) ──
        Concept {
            label: "Ancient history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.7),
                (SYSTEMS, 0.6),
                (EVOLUTION, 0.5),
                (NARRATIVE, 0.5),
                (HISTORICAL, 0.8),
                (ARCHIVAL, 0.3),
            ],
        },
        Concept {
            label: "Medieval history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.7),
                (STRUCTURE, 0.4),
                (METAPHYSICS, 0.3),
                (HISTORICAL, 0.8),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Modern history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.7),
                (SYSTEMS, 0.7),
                (MARKETS, 0.5),
                (GOVERNANCE, 0.4),
                (HISTORICAL, 0.8),
            ],
        },
        Concept {
            label: "Military history",
            category: "history",
            features: vec![
                (FORCE, 0.7),
                (BEHAVIOR, 0.7),
                (SYSTEMS, 0.6),
                (STRUCTURE, 0.4),
                (HISTORICAL, 0.7),
                (POWER, 0.4),
            ],
        },
        Concept {
            label: "Economic history",
            category: "history",
            features: vec![
                (MARKETS, 0.8),
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.6),
                (EVOLUTION, 0.4),
                (HISTORICAL, 0.7),
                (MONEY, 0.3),
            ],
        },
        Concept {
            label: "Social history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.8),
                (NETWORK, 0.6),
                (SYSTEMS, 0.5),
                (NARRATIVE, 0.4),
                (HISTORICAL, 0.7),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Intellectual history",
            category: "history",
            features: vec![
                (MIND, 0.6),
                (LANGUAGE, 0.6),
                (BEHAVIOR, 0.5),
                (EVOLUTION, 0.4),
                (HISTORICAL, 0.7),
                (DISCOURSE, 0.3),
            ],
        },
        Concept {
            label: "Cultural history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.7),
                (EMOTION, 0.5),
                (PATTERN, 0.5),
                (LANGUAGE, 0.4),
                (HISTORICAL, 0.6),
                (CULTURE, 0.5),
            ],
        },
        Concept {
            label: "Environmental history",
            category: "history",
            features: vec![
                (NATURE, 0.7),
                (ECOSYSTEM, 0.5),
                (SYSTEMS, 0.6),
                (EVOLUTION, 0.5),
                (HISTORICAL, 0.6),
                (CLIMATE, 0.3),
            ],
        },
        Concept {
            label: "Historiography",
            category: "history",
            features: vec![
                (LOGIC, 0.6),
                (LANGUAGE, 0.6),
                (NARRATIVE, 0.6),
                (METAPHYSICS, 0.4),
                (HISTORICAL, 0.6),
                (EPISTEMOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Archaeology",
            category: "history",
            features: vec![
                (NATURE, 0.5),
                (STRUCTURE, 0.6),
                (EVOLUTION, 0.6),
                (MATERIAL, 0.5),
                (HISTORICAL, 0.6),
                (ARCHIVAL, 0.4),
            ],
        },
        Concept {
            label: "Diplomatic history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.7),
                (NETWORK, 0.7),
                (GOVERNANCE, 0.5),
                (LANGUAGE, 0.4),
                (HISTORICAL, 0.6),
                (POWER, 0.3),
            ],
        },
        Concept {
            label: "History of technology",
            category: "history",
            features: vec![
                (EVOLUTION, 0.6),
                (STRUCTURE, 0.5),
                (COMPUTATION, 0.3),
                (SYSTEMS, 0.4),
                (HISTORICAL, 0.6),
            ],
        },
        Concept {
            label: "History of science",
            category: "history",
            features: vec![
                (LOGIC, 0.5),
                (EVOLUTION, 0.6),
                (METAPHYSICS, 0.3),
                (MEASUREMENT, 0.3),
                (HISTORICAL, 0.6),
                (EPISTEMOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Urban history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.6),
                (STRUCTURE, 0.6),
                (SYSTEMS, 0.5),
                (NETWORK, 0.4),
                (HISTORICAL, 0.6),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Maritime history",
            category: "history",
            features: vec![
                (NATURE, 0.5),
                (FORCE, 0.4),
                (NETWORK, 0.5),
                (BEHAVIOR, 0.5),
                (HISTORICAL, 0.6),
                (OCEAN, 0.4),
            ],
        },
        Concept {
            label: "History of art",
            category: "history",
            features: vec![
                (VISUAL, 0.6),
                (EMOTION, 0.4),
                (EVOLUTION, 0.5),
                (PATTERN, 0.4),
                (HISTORICAL, 0.6),
            ],
        },
        Concept {
            label: "History of medicine",
            category: "history",
            features: vec![
                (LIFE, 0.5),
                (DIAGNOSTICS, 0.4),
                (EVOLUTION, 0.5),
                (BEHAVIOR, 0.4),
                (HISTORICAL, 0.6),
                (CLINICAL, 0.2),
            ],
        },
        Concept {
            label: "Labor history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.7),
                (MARKETS, 0.6),
                (SYSTEMS, 0.5),
                (ETHICS, 0.3),
                (HISTORICAL, 0.6),
                (LABOR, 0.6),
            ],
        },
        Concept {
            label: "Constitutional history",
            category: "history",
            features: vec![
                (GOVERNANCE, 0.7),
                (LOGIC, 0.5),
                (LANGUAGE, 0.5),
                (BEHAVIOR, 0.4),
                (HISTORICAL, 0.6),
                (LEGAL, 0.4),
                (RIGHTS, 0.3),
            ],
        },
        Concept {
            label: "History of education",
            category: "history",
            features: vec![
                (PEDAGOGY, 0.6),
                (BEHAVIOR, 0.5),
                (EVOLUTION, 0.4),
                (SYSTEMS, 0.4),
                (HISTORICAL, 0.6),
            ],
        },
        Concept {
            label: "Colonial history",
            category: "history",
            features: vec![
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.6),
                (NETWORK, 0.5),
                (MARKETS, 0.4),
                (HISTORICAL, 0.6),
                (POWER, 0.5),
            ],
        },
        Concept {
            label: "Digital history",
            category: "history",
            features: vec![
                (COMPUTATION, 0.5),
                (INFORMATION, 0.5),
                (NARRATIVE, 0.4),
                (STATISTICS, 0.3),
                (HISTORICAL, 0.5),
                (ARCHIVAL, 0.4),
            ],
        },
        Concept {
            label: "Oral history",
            category: "history",
            features: vec![
                (LANGUAGE, 0.6),
                (NARRATIVE, 0.7),
                (SOUND, 0.3),
                (BEHAVIOR, 0.4),
                (HISTORICAL, 0.6),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Religious history",
            category: "history",
            features: vec![
                (METAPHYSICS, 0.6),
                (BEHAVIOR, 0.5),
                (SYSTEMS, 0.4),
                (EVOLUTION, 0.4),
                (HISTORICAL, 0.6),
                (DOCTRINE, 0.3),
            ],
        },
        // ── SOCIOLOGY (25) ──
        Concept {
            label: "Social theory",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (SYSTEMS, 0.7),
                (METAPHYSICS, 0.4),
                (STRUCTURE, 0.5),
                (SOCIETY, 0.7),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Urban sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (STRUCTURE, 0.6),
                (SYSTEMS, 0.6),
                (NETWORK, 0.5),
                (SOCIETY, 0.5),
                (COMMUNITY, 0.5),
            ],
        },
        Concept {
            label: "Rural sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.3),
                (NETWORK, 0.3),
                (COMMUNITY, 0.5),
                (SOCIETY, 0.3),
            ],
        },
        Concept {
            label: "Sociology of religion",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (METAPHYSICS, 0.6),
                (SYSTEMS, 0.5),
                (NETWORK, 0.4),
                (SPIRITUAL, 0.3),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Medical sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (LIFE, 0.5),
                (DIAGNOSTICS, 0.4),
                (SYSTEMS, 0.5),
                (SOCIETY, 0.4),
                (CLINICAL, 0.2),
            ],
        },
        Concept {
            label: "Gender studies",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (MIND, 0.5),
                (SYSTEMS, 0.5),
                (PATTERN, 0.4),
                (SOCIETY, 0.5),
                (POWER, 0.3),
            ],
        },
        Concept {
            label: "Criminology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (LOGIC, 0.5),
                (STATISTICS, 0.5),
                (ETHICS, 0.5),
                (LEGAL, 0.4),
                (JUSTICE, 0.4),
            ],
        },
        Concept {
            label: "Social network analysis",
            category: "sociology",
            features: vec![
                (NETWORK, 0.9),
                (BEHAVIOR, 0.7),
                (PATTERN, 0.6),
                (STATISTICS, 0.6),
                (COMPUTATION, 0.3),
                (SOCIAL_NETWORK, 0.8),
            ],
        },
        Concept {
            label: "Demography",
            category: "sociology",
            features: vec![
                (STATISTICS, 0.9),
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.5),
                (LIFE, 0.3),
                (SOCIETY, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Cultural sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (EMOTION, 0.5),
                (LANGUAGE, 0.5),
                (PATTERN, 0.5),
                (CULTURE, 0.6),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Political sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.6),
                (SYSTEMS, 0.7),
                (NETWORK, 0.4),
                (POWER, 0.5),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Sociology of science",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (LOGIC, 0.5),
                (STRUCTURE, 0.4),
                (SYSTEMS, 0.4),
                (SOCIETY, 0.4),
                (EPISTEMOLOGY, 0.2),
            ],
        },
        Concept {
            label: "Social movements research",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (NETWORK, 0.6),
                (EVOLUTION, 0.5),
                (EMOTION, 0.4),
                (SOCIETY, 0.5),
                (POWER, 0.3),
            ],
        },
        Concept {
            label: "Stratification research",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (MARKETS, 0.5),
                (SYSTEMS, 0.6),
                (STATISTICS, 0.5),
                (SOCIETY, 0.5),
            ],
        },
        Concept {
            label: "Ethnography",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (LANGUAGE, 0.5),
                (PATTERN, 0.5),
                (COGNITION, 0.3),
                (CULTURE, 0.5),
            ],
        },
        Concept {
            label: "Economic sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (MARKETS, 0.7),
                (NETWORK, 0.5),
                (SYSTEMS, 0.5),
                (SOCIETY, 0.4),
                (MONEY, 0.3),
            ],
        },
        Concept {
            label: "Sociology of education",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (PEDAGOGY, 0.6),
                (SYSTEMS, 0.5),
                (NETWORK, 0.3),
                (SOCIETY, 0.4),
                (LEARNING, 0.3),
            ],
        },
        Concept {
            label: "Environmental sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.6),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.5),
                (SYSTEMS, 0.5),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Sociology of technology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.6),
                (COMPUTATION, 0.4),
                (NETWORK, 0.5),
                (SYSTEMS, 0.5),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Migration studies",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (MOTION, 0.4),
                (NETWORK, 0.5),
                (SYSTEMS, 0.4),
                (SOCIETY, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Sociology of health",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.7),
                (LIFE, 0.5),
                (SYSTEMS, 0.5),
                (STATISTICS, 0.3),
                (SOCIETY, 0.4),
            ],
        },
        Concept {
            label: "Deviance studies",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.8),
                (ETHICS, 0.4),
                (PATTERN, 0.4),
                (LOGIC, 0.3),
                (SOCIETY, 0.4),
                (JUSTICE, 0.3),
            ],
        },
        Concept {
            label: "Visual sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.6),
                (VISUAL, 0.6),
                (PATTERN, 0.5),
                (LANGUAGE, 0.3),
                (SOCIETY, 0.3),
            ],
        },
        Concept {
            label: "Sociology of knowledge",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.6),
                (INFORMATION, 0.5),
                (MIND, 0.4),
                (SYSTEMS, 0.4),
                (SOCIETY, 0.4),
                (EPISTEMOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Computational sociology",
            category: "sociology",
            features: vec![
                (BEHAVIOR, 0.6),
                (COMPUTATION, 0.6),
                (NETWORK, 0.6),
                (STATISTICS, 0.5),
                (SOCIAL_NETWORK, 0.4),
                (DATA, 0.3),
            ],
        },
        // ── POLITICAL SCIENCE (25) ──
        Concept {
            label: "International relations",
            category: "political_science",
            features: vec![
                (BEHAVIOR, 0.8),
                (NETWORK, 0.7),
                (GOVERNANCE, 0.6),
                (SYSTEMS, 0.5),
                (POWER, 0.5),
            ],
        },
        Concept {
            label: "Comparative politics",
            category: "political_science",
            features: vec![
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.6),
                (SYSTEMS, 0.7),
                (PATTERN, 0.5),
                (POLICY, 0.3),
            ],
        },
        Concept {
            label: "Political theory",
            category: "political_science",
            features: vec![
                (ETHICS, 0.8),
                (METAPHYSICS, 0.5),
                (GOVERNANCE, 0.6),
                (BEHAVIOR, 0.5),
                (POWER, 0.5),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Public administration",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.8),
                (SYSTEMS, 0.8),
                (OPTIMIZATION, 0.5),
                (BEHAVIOR, 0.4),
                (POLICY, 0.5),
            ],
        },
        Concept {
            label: "Public policy",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.8),
                (BEHAVIOR, 0.7),
                (SYSTEMS, 0.7),
                (ETHICS, 0.5),
                (POLICY, 0.9),
            ],
        },
        Concept {
            label: "Electoral studies",
            category: "political_science",
            features: vec![
                (BEHAVIOR, 0.8),
                (STATISTICS, 0.7),
                (GOVERNANCE, 0.5),
                (PATTERN, 0.3),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Political economy",
            category: "political_science",
            features: vec![
                (MARKETS, 0.8),
                (GOVERNANCE, 0.6),
                (BEHAVIOR, 0.7),
                (SYSTEMS, 0.5),
                (POWER, 0.4),
                (MONEY, 0.3),
            ],
        },
        Concept {
            label: "Geopolitics",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.6),
                (SPACE, 0.5),
                (BEHAVIOR, 0.6),
                (NETWORK, 0.5),
                (POWER, 0.6),
            ],
        },
        Concept {
            label: "Security studies",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.5),
                (FORCE, 0.5),
                (BEHAVIOR, 0.6),
                (SYSTEMS, 0.5),
                (POWER, 0.4),
            ],
        },
        Concept {
            label: "Democratic theory",
            category: "political_science",
            features: vec![
                (ETHICS, 0.7),
                (GOVERNANCE, 0.7),
                (BEHAVIOR, 0.6),
                (LOGIC, 0.4),
                (RIGHTS, 0.5),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Nationalism studies",
            category: "political_science",
            features: vec![
                (BEHAVIOR, 0.7),
                (EMOTION, 0.6),
                (LANGUAGE, 0.4),
                (GOVERNANCE, 0.4),
                (CULTURE, 0.3),
                (POWER, 0.3),
            ],
        },
        Concept {
            label: "Peace studies",
            category: "political_science",
            features: vec![
                (ETHICS, 0.7),
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.4),
                (NETWORK, 0.4),
                (JUSTICE, 0.4),
                (MORAL, 0.3),
            ],
        },
        Concept {
            label: "Political psychology",
            category: "political_science",
            features: vec![
                (MIND, 0.6),
                (BEHAVIOR, 0.8),
                (COGNITION, 0.5),
                (EMOTION, 0.4),
                (POWER, 0.3),
            ],
        },
        Concept {
            label: "Constitutional studies",
            category: "political_science",
            features: vec![
                (LOGIC, 0.7),
                (GOVERNANCE, 0.7),
                (ETHICS, 0.6),
                (LANGUAGE, 0.5),
                (LEGAL, 0.5),
                (RIGHTS, 0.5),
            ],
        },
        Concept {
            label: "Global governance",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.9),
                (NETWORK, 0.7),
                (SYSTEMS, 0.6),
                (ETHICS, 0.5),
                (POLICY, 0.5),
            ],
        },
        Concept {
            label: "Federalism studies",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.8),
                (STRUCTURE, 0.5),
                (SYSTEMS, 0.6),
                (LOGIC, 0.3),
                (POWER, 0.3),
            ],
        },
        Concept {
            label: "Political communication",
            category: "political_science",
            features: vec![
                (LANGUAGE, 0.6),
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.5),
                (NETWORK, 0.4),
                (DISCOURSE, 0.4),
            ],
        },
        Concept {
            label: "Foreign policy analysis",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.7),
                (BEHAVIOR, 0.7),
                (NETWORK, 0.5),
                (SYSTEMS, 0.4),
                (POWER, 0.4),
                (POLICY, 0.4),
            ],
        },
        Concept {
            label: "Political philosophy (pol sci)",
            category: "political_science",
            features: vec![
                (ETHICS, 0.7),
                (GOVERNANCE, 0.6),
                (METAPHYSICS, 0.4),
                (LOGIC, 0.5),
                (MORAL, 0.4),
                (THEORY, 0.4),
            ],
        },
        Concept {
            label: "Comparative institutional analysis",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.7),
                (SYSTEMS, 0.7),
                (STRUCTURE, 0.5),
                (PATTERN, 0.4),
                (POLICY, 0.3),
            ],
        },
        Concept {
            label: "Quantitative political science",
            category: "political_science",
            features: vec![
                (STATISTICS, 0.8),
                (BEHAVIOR, 0.5),
                (GOVERNANCE, 0.5),
                (COMPUTATION, 0.4),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Social policy",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.7),
                (BEHAVIOR, 0.6),
                (ETHICS, 0.5),
                (SYSTEMS, 0.5),
                (POLICY, 0.7),
                (SOCIETY, 0.3),
            ],
        },
        Concept {
            label: "Conflict studies",
            category: "political_science",
            features: vec![
                (BEHAVIOR, 0.7),
                (FORCE, 0.5),
                (GOVERNANCE, 0.4),
                (EMOTION, 0.4),
                (POWER, 0.4),
            ],
        },
        Concept {
            label: "Environmental politics",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.6),
                (NATURE, 0.5),
                (ECOSYSTEM, 0.5),
                (ETHICS, 0.4),
                (POLICY, 0.5),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Regulatory studies",
            category: "political_science",
            features: vec![
                (GOVERNANCE, 0.9),
                (SYSTEMS, 0.6),
                (LOGIC, 0.5),
                (MARKETS, 0.4),
                (POLICY, 0.5),
                (LEGAL, 0.3),
            ],
        },
        // ── LAW (25) ──
        Concept {
            label: "Constitutional law",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (GOVERNANCE, 0.7),
                (ETHICS, 0.6),
                (LANGUAGE, 0.7),
                (LEGAL, 0.8),
                (RIGHTS, 0.6),
            ],
        },
        Concept {
            label: "Criminal law",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (ETHICS, 0.7),
                (BEHAVIOR, 0.6),
                (LANGUAGE, 0.4),
                (LEGAL, 0.8),
                (JUSTICE, 0.6),
            ],
        },
        Concept {
            label: "Contract law",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (LANGUAGE, 0.7),
                (MARKETS, 0.6),
                (STRUCTURE, 0.4),
                (LEGAL, 0.8),
            ],
        },
        Concept {
            label: "Tort law",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (ETHICS, 0.6),
                (BEHAVIOR, 0.5),
                (LEGAL, 0.8),
                (JUSTICE, 0.5),
            ],
        },
        Concept {
            label: "International law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (ETHICS, 0.6),
                (GOVERNANCE, 0.6),
                (NETWORK, 0.5),
                (LEGAL, 0.7),
                (RIGHTS, 0.4),
            ],
        },
        Concept {
            label: "Environmental law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (NATURE, 0.7),
                (ECOSYSTEM, 0.5),
                (ETHICS, 0.6),
                (LEGAL, 0.7),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Intellectual property law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (INFORMATION, 0.8),
                (MARKETS, 0.5),
                (ETHICS, 0.4),
                (LEGAL, 0.7),
                (RIGHTS, 0.5),
            ],
        },
        Concept {
            label: "Corporate law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (MARKETS, 0.8),
                (GOVERNANCE, 0.5),
                (STRUCTURE, 0.4),
                (LEGAL, 0.7),
                (FINANCE, 0.3),
            ],
        },
        Concept {
            label: "Jurisprudence",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (ETHICS, 0.7),
                (METAPHYSICS, 0.6),
                (LANGUAGE, 0.5),
                (LEGAL, 0.7),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Family law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (BEHAVIOR, 0.6),
                (ETHICS, 0.6),
                (EMOTION, 0.4),
                (LEGAL, 0.7),
                (KINSHIP, 0.4),
            ],
        },
        Concept {
            label: "Tax law",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (MARKETS, 0.7),
                (MATH, 0.5),
                (GOVERNANCE, 0.5),
                (LEGAL, 0.7),
                (MONEY, 0.4),
            ],
        },
        Concept {
            label: "Administrative law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (GOVERNANCE, 0.8),
                (SYSTEMS, 0.6),
                (STRUCTURE, 0.4),
                (LEGAL, 0.7),
                (POLICY, 0.4),
            ],
        },
        Concept {
            label: "Human rights law",
            category: "law",
            features: vec![
                (ETHICS, 0.9),
                (GOVERNANCE, 0.5),
                (BEHAVIOR, 0.5),
                (LANGUAGE, 0.5),
                (LEGAL, 0.6),
                (RIGHTS, 0.9),
                (JUSTICE, 0.5),
            ],
        },
        Concept {
            label: "Antitrust law",
            category: "law",
            features: vec![
                (MARKETS, 0.8),
                (LOGIC, 0.7),
                (GOVERNANCE, 0.5),
                (BEHAVIOR, 0.4),
                (LEGAL, 0.7),
            ],
        },
        Concept {
            label: "Cyberlaw",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (INFORMATION, 0.8),
                (COMPUTATION, 0.6),
                (GOVERNANCE, 0.4),
                (LEGAL, 0.7),
                (RIGHTS, 0.3),
            ],
        },
        Concept {
            label: "Labor law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (MARKETS, 0.6),
                (ETHICS, 0.5),
                (GOVERNANCE, 0.5),
                (LEGAL, 0.7),
                (LABOR, 0.6),
                (RIGHTS, 0.4),
            ],
        },
        Concept {
            label: "Immigration law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (GOVERNANCE, 0.7),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.5),
                (LEGAL, 0.7),
                (RIGHTS, 0.4),
            ],
        },
        Concept {
            label: "Health law",
            category: "law",
            features: vec![
                (LOGIC, 0.6),
                (LIFE, 0.5),
                (GOVERNANCE, 0.6),
                (ETHICS, 0.5),
                (LEGAL, 0.7),
                (CLINICAL, 0.2),
            ],
        },
        Concept {
            label: "Property law",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (MARKETS, 0.6),
                (STRUCTURE, 0.5),
                (GOVERNANCE, 0.4),
                (LEGAL, 0.7),
                (RIGHTS, 0.4),
            ],
        },
        Concept {
            label: "International humanitarian law",
            category: "law",
            features: vec![
                (ETHICS, 0.8),
                (GOVERNANCE, 0.6),
                (FORCE, 0.4),
                (BEHAVIOR, 0.4),
                (LEGAL, 0.6),
                (JUSTICE, 0.5),
                (RIGHTS, 0.5),
            ],
        },
        Concept {
            label: "Legal theory",
            category: "law",
            features: vec![
                (LOGIC, 0.8),
                (METAPHYSICS, 0.5),
                (LANGUAGE, 0.5),
                (ETHICS, 0.5),
                (LEGAL, 0.7),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Maritime law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (GOVERNANCE, 0.5),
                (NATURE, 0.4),
                (NETWORK, 0.4),
                (LEGAL, 0.6),
                (OCEAN, 0.3),
            ],
        },
        Concept {
            label: "Competition law",
            category: "law",
            features: vec![
                (MARKETS, 0.7),
                (LOGIC, 0.7),
                (GOVERNANCE, 0.5),
                (OPTIMIZATION, 0.3),
                (LEGAL, 0.6),
            ],
        },
        Concept {
            label: "Space law",
            category: "law",
            features: vec![
                (LOGIC, 0.6),
                (SPACE, 0.6),
                (GOVERNANCE, 0.5),
                (ETHICS, 0.4),
                (LEGAL, 0.6),
                (CELESTIAL, 0.2),
            ],
        },
        Concept {
            label: "Data protection law",
            category: "law",
            features: vec![
                (LOGIC, 0.7),
                (INFORMATION, 0.8),
                (ETHICS, 0.6),
                (GOVERNANCE, 0.5),
                (LEGAL, 0.7),
                (RIGHTS, 0.5),
            ],
        },
        // ── ARCHITECTURE (25) ──
        Concept {
            label: "Architectural design",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.9),
                (SPACE, 0.8),
                (VISUAL, 0.6),
                (PATTERN, 0.6),
                (DESIGN, 0.7),
                (FORM, 0.5),
            ],
        },
        Concept {
            label: "Urban planning",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.7),
                (SPACE, 0.7),
                (SYSTEMS, 0.8),
                (BEHAVIOR, 0.5),
                (COMMUNITY, 0.4),
                (TRANSPORTATION, 0.3),
            ],
        },
        Concept {
            label: "Landscape architecture",
            category: "architecture",
            features: vec![
                (NATURE, 0.7),
                (STRUCTURE, 0.6),
                (SPACE, 0.7),
                (ECOSYSTEM, 0.4),
                (DESIGN, 0.5),
            ],
        },
        Concept {
            label: "Interior design",
            category: "architecture",
            features: vec![
                (SPACE, 0.7),
                (VISUAL, 0.7),
                (EMOTION, 0.7),
                (MATERIAL, 0.4),
                (DESIGN, 0.7),
                (COLOR, 0.3),
            ],
        },
        Concept {
            label: "Sustainable architecture",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.7),
                (NATURE, 0.7),
                (ENERGY, 0.6),
                (ECOSYSTEM, 0.5),
                (CONSERVATION, 0.4),
                (DESIGN, 0.3),
            ],
        },
        Concept {
            label: "Architectural history",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.6),
                (EVOLUTION, 0.5),
                (VISUAL, 0.4),
                (NARRATIVE, 0.3),
                (HISTORICAL, 0.5),
            ],
        },
        Concept {
            label: "Architectural theory",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.6),
                (METAPHYSICS, 0.5),
                (MIND, 0.4),
                (VISUAL, 0.4),
                (THEORY, 0.5),
                (FORM, 0.3),
            ],
        },
        Concept {
            label: "Parametric design",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.8),
                (MATH, 0.7),
                (COMPUTATION, 0.6),
                (PATTERN, 0.7),
                (DESIGN, 0.5),
                (ALGORITHM, 0.3),
            ],
        },
        Concept {
            label: "Building information modeling",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.7),
                (COMPUTATION, 0.7),
                (INFORMATION, 0.7),
                (SYSTEMS, 0.4),
                (SOFTWARE, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Historic preservation",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.7),
                (MATERIAL, 0.5),
                (EVOLUTION, 0.6),
                (ETHICS, 0.3),
                (HISTORICAL, 0.5),
                (ARCHIVAL, 0.3),
            ],
        },
        Concept {
            label: "Vernacular architecture",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.7),
                (BEHAVIOR, 0.5),
                (MATERIAL, 0.5),
                (NATURE, 0.4),
                (TRADITION, 0.5),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Biophilic design",
            category: "architecture",
            features: vec![
                (NATURE, 0.7),
                (STRUCTURE, 0.6),
                (EMOTION, 0.6),
                (ECOSYSTEM, 0.4),
                (DESIGN, 0.5),
            ],
        },
        Concept {
            label: "Acoustic architecture",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.7),
                (SOUND, 0.8),
                (WAVE, 0.7),
                (SPACE, 0.4),
                (TIMBRE, 0.3),
            ],
        },
        Concept {
            label: "Sacred architecture",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.8),
                (METAPHYSICS, 0.7),
                (EMOTION, 0.7),
                (SPACE, 0.6),
                (SACRED, 0.6),
                (SPIRITUAL, 0.3),
            ],
        },
        Concept {
            label: "Building physics",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.6),
                (ENERGY, 0.7),
                (WAVE, 0.5),
                (MATERIAL, 0.5),
                (MECHANICAL, 0.3),
            ],
        },
        Concept {
            label: "Housing studies",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.6),
                (BEHAVIOR, 0.5),
                (SYSTEMS, 0.5),
                (MARKETS, 0.3),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Structural analysis",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.9),
                (FORCE, 0.7),
                (MATH, 0.6),
                (MATERIAL, 0.5),
                (MECHANICAL, 0.4),
            ],
        },
        Concept {
            label: "Urban morphology",
            category: "architecture",
            features: vec![
                (SPACE, 0.7),
                (STRUCTURE, 0.6),
                (PATTERN, 0.6),
                (EVOLUTION, 0.4),
                (FORM, 0.3),
            ],
        },
        Concept {
            label: "Lighting design",
            category: "architecture",
            features: vec![
                (WAVE, 0.5),
                (VISUAL, 0.7),
                (ENERGY, 0.5),
                (EMOTION, 0.4),
                (DESIGN, 0.4),
                (COLOR, 0.3),
            ],
        },
        Concept {
            label: "Landscape ecology",
            category: "architecture",
            features: vec![
                (NATURE, 0.7),
                (ECOSYSTEM, 0.7),
                (STRUCTURE, 0.4),
                (SYSTEMS, 0.4),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Transportation planning",
            category: "architecture",
            features: vec![
                (SYSTEMS, 0.7),
                (MOTION, 0.5),
                (STRUCTURE, 0.5),
                (NETWORK, 0.5),
                (TRANSPORTATION, 0.6),
            ],
        },
        Concept {
            label: "Climate-responsive design",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.6),
                (ENERGY, 0.6),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.5),
                (DESIGN, 0.4),
                (CLIMATE, 0.4),
            ],
        },
        Concept {
            label: "Architectural materials",
            category: "architecture",
            features: vec![
                (MATERIAL, 0.9),
                (STRUCTURE, 0.7),
                (CHEMISTRY, 0.4),
                (FORCE, 0.3),
                (SURFACE, 0.3),
            ],
        },
        Concept {
            label: "Universal design",
            category: "architecture",
            features: vec![
                (STRUCTURE, 0.5),
                (BEHAVIOR, 0.6),
                (ETHICS, 0.5),
                (COGNITION, 0.3),
                (DESIGN, 0.5),
                (RIGHTS, 0.2),
            ],
        },
        Concept {
            label: "Building services engineering",
            category: "architecture",
            features: vec![
                (SYSTEMS, 0.7),
                (ENERGY, 0.6),
                (STRUCTURE, 0.5),
                (MEASUREMENT, 0.4),
                (MECHANICAL, 0.3),
                (ELECTRICAL, 0.3),
            ],
        },
        // ── CULINARY ARTS (25) ──
        Concept {
            label: "Cooking techniques",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.7),
                (ENERGY, 0.6),
                (PERFORMANCE, 0.7),
                (PATTERN, 0.4),
                (COOKING, 0.9),
                (TASTE, 0.4),
            ],
        },
        Concept {
            label: "Baking",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.8),
                (PATTERN, 0.6),
                (STRUCTURE, 0.5),
                (MEASUREMENT, 0.4),
                (COOKING, 0.7),
                (REACTION, 0.3),
            ],
        },
        Concept {
            label: "Pastry arts",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.7),
                (VISUAL, 0.6),
                (STRUCTURE, 0.6),
                (PERFORMANCE, 0.7),
                (COOKING, 0.6),
                (DESIGN, 0.3),
            ],
        },
        Concept {
            label: "Molecular gastronomy",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.9),
                (STRUCTURE, 0.6),
                (PATTERN, 0.5),
                (MEASUREMENT, 0.4),
                (COOKING, 0.5),
                (MOLECULAR, 0.4),
                (FLAVOR, 0.4),
            ],
        },
        Concept {
            label: "Fermentation science",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.7),
                (LIFE, 0.7),
                (EVOLUTION, 0.4),
                (PATTERN, 0.4),
                (COOKING, 0.4),
                (REACTION, 0.5),
                (CYCLE, 0.3),
            ],
        },
        Concept {
            label: "Food chemistry",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.9),
                (MATERIAL, 0.4),
                (STRUCTURE, 0.5),
                (MEASUREMENT, 0.3),
                (FLAVOR, 0.5),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Nutrition science",
            category: "culinary_arts",
            features: vec![
                (LIFE, 0.7),
                (CHEMISTRY, 0.6),
                (STATISTICS, 0.5),
                (DIAGNOSTICS, 0.4),
                (COOKING, 0.3),
            ],
        },
        Concept {
            label: "Oenology",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.6),
                (PATTERN, 0.6),
                (EMOTION, 0.7),
                (COGNITION, 0.4),
                (TASTE, 0.7),
                (FLAVOR, 0.6),
            ],
        },
        Concept {
            label: "Pastry decoration",
            category: "culinary_arts",
            features: vec![
                (VISUAL, 0.7),
                (PATTERN, 0.8),
                (PERFORMANCE, 0.7),
                (EMOTION, 0.5),
                (DESIGN, 0.4),
                (COLOR, 0.3),
            ],
        },
        Concept {
            label: "Culinary history",
            category: "culinary_arts",
            features: vec![
                (BEHAVIOR, 0.6),
                (EVOLUTION, 0.5),
                (NARRATIVE, 0.4),
                (LANGUAGE, 0.3),
                (HISTORICAL, 0.4),
                (TRADITION, 0.4),
                (COOKING, 0.4),
            ],
        },
        Concept {
            label: "Food science",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.7),
                (MEASUREMENT, 0.5),
                (STATISTICS, 0.4),
                (MATERIAL, 0.4),
                (FLAVOR, 0.3),
            ],
        },
        Concept {
            label: "Plating and presentation",
            category: "culinary_arts",
            features: vec![
                (VISUAL, 0.8),
                (PATTERN, 0.7),
                (PERFORMANCE, 0.7),
                (EMOTION, 0.5),
                (DESIGN, 0.4),
                (COLOR, 0.3),
            ],
        },
        Concept {
            label: "Brewing science",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.8),
                (LIFE, 0.6),
                (PATTERN, 0.5),
                (MEASUREMENT, 0.4),
                (COOKING, 0.4),
                (REACTION, 0.3),
            ],
        },
        Concept {
            label: "Sommelier studies",
            category: "culinary_arts",
            features: vec![
                (COGNITION, 0.5),
                (EMOTION, 0.7),
                (PATTERN, 0.6),
                (CHEMISTRY, 0.4),
                (TASTE, 0.7),
                (FLAVOR, 0.6),
            ],
        },
        Concept {
            label: "Traditional cuisine studies",
            category: "culinary_arts",
            features: vec![
                (BEHAVIOR, 0.6),
                (LANGUAGE, 0.4),
                (PATTERN, 0.5),
                (EVOLUTION, 0.4),
                (TRADITION, 0.5),
                (CULTURE, 0.4),
                (COOKING, 0.4),
            ],
        },
        Concept {
            label: "Food safety",
            category: "culinary_arts",
            features: vec![
                (LIFE, 0.5),
                (CHEMISTRY, 0.6),
                (DIAGNOSTICS, 0.5),
                (SYSTEMS, 0.4),
                (COOKING, 0.3),
            ],
        },
        Concept {
            label: "Charcuterie",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.6),
                (MATERIAL, 0.5),
                (PATTERN, 0.5),
                (PERFORMANCE, 0.4),
                (COOKING, 0.6),
                (FLAVOR, 0.4),
            ],
        },
        Concept {
            label: "Confectionery science",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.8),
                (STRUCTURE, 0.5),
                (PATTERN, 0.5),
                (VISUAL, 0.4),
                (COOKING, 0.4),
                (TASTE, 0.3),
            ],
        },
        Concept {
            label: "Culinary anthropology",
            category: "culinary_arts",
            features: vec![
                (BEHAVIOR, 0.7),
                (EVOLUTION, 0.5),
                (LANGUAGE, 0.4),
                (PATTERN, 0.4),
                (CULTURE, 0.5),
                (COOKING, 0.3),
            ],
        },
        Concept {
            label: "Food preservation",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.7),
                (LIFE, 0.5),
                (ENTROPY, 0.4),
                (MEASUREMENT, 0.3),
                (COOKING, 0.4),
                (REACTION, 0.3),
            ],
        },
        Concept {
            label: "Sous vide technique",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.6),
                (ENERGY, 0.6),
                (MEASUREMENT, 0.6),
                (PERFORMANCE, 0.4),
                (COOKING, 0.6),
            ],
        },
        Concept {
            label: "Flavor pairing theory",
            category: "culinary_arts",
            features: vec![
                (CHEMISTRY, 0.6),
                (PATTERN, 0.7),
                (COGNITION, 0.5),
                (COMPUTATION, 0.3),
                (FLAVOR, 0.8),
                (TASTE, 0.5),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Kitchen management",
            category: "culinary_arts",
            features: vec![
                (SYSTEMS, 0.6),
                (PERFORMANCE, 0.5),
                (OPTIMIZATION, 0.4),
                (BEHAVIOR, 0.4),
                (COOKING, 0.4),
            ],
        },
        Concept {
            label: "Gastronomy philosophy",
            category: "culinary_arts",
            features: vec![
                (MIND, 0.4),
                (EMOTION, 0.6),
                (ETHICS, 0.4),
                (BEHAVIOR, 0.4),
                (TASTE, 0.4),
            ],
        },
        Concept {
            label: "Food photography",
            category: "culinary_arts",
            features: vec![
                (VISUAL, 0.8),
                (WAVE, 0.3),
                (EMOTION, 0.5),
                (PATTERN, 0.5),
                (COLOR, 0.4),
            ],
        },
        // ── RELIGION (25) ──
        Concept {
            label: "Systematic theology",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.9),
                (LOGIC, 0.6),
                (ETHICS, 0.6),
                (STRUCTURE, 0.4),
                (DOCTRINE, 0.7),
                (SPIRITUAL, 0.5),
            ],
        },
        Concept {
            label: "Comparative religion",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.7),
                (BEHAVIOR, 0.6),
                (LANGUAGE, 0.5),
                (PATTERN, 0.5),
                (SPIRITUAL, 0.5),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Biblical studies",
            category: "religion",
            features: vec![
                (LANGUAGE, 0.8),
                (METAPHYSICS, 0.7),
                (NARRATIVE, 0.5),
                (PATTERN, 0.4),
                (SACRED, 0.6),
                (ARCHIVAL, 0.3),
            ],
        },
        Concept {
            label: "Islamic studies",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.7),
                (LANGUAGE, 0.7),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.5),
                (DOCTRINE, 0.5),
                (SACRED, 0.4),
            ],
        },
        Concept {
            label: "Buddhist studies",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.8),
                (MIND, 0.7),
                (BEHAVIOR, 0.5),
                (COGNITION, 0.5),
                (CONSCIOUSNESS, 0.4),
                (SPIRITUAL, 0.5),
            ],
        },
        Concept {
            label: "Religious ethics",
            category: "religion",
            features: vec![
                (ETHICS, 0.9),
                (METAPHYSICS, 0.7),
                (BEHAVIOR, 0.5),
                (MORAL, 0.7),
                (DOCTRINE, 0.4),
            ],
        },
        Concept {
            label: "Philosophy of religion",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.9),
                (LOGIC, 0.7),
                (MIND, 0.5),
                (ETHICS, 0.4),
                (EPISTEMOLOGY, 0.4),
                (ONTOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Church history",
            category: "religion",
            features: vec![
                (BEHAVIOR, 0.6),
                (GOVERNANCE, 0.4),
                (SYSTEMS, 0.5),
                (EVOLUTION, 0.6),
                (HISTORICAL, 0.5),
                (DOCTRINE, 0.3),
            ],
        },
        Concept {
            label: "Mysticism",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.9),
                (MIND, 0.8),
                (EMOTION, 0.7),
                (COGNITION, 0.5),
                (SPIRITUAL, 0.8),
                (CONSCIOUSNESS, 0.5),
            ],
        },
        Concept {
            label: "Theological anthropology",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.7),
                (LIFE, 0.4),
                (BEHAVIOR, 0.5),
                (MIND, 0.6),
                (SPIRITUAL, 0.4),
            ],
        },
        Concept {
            label: "Liturgical studies",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.6),
                (PERFORMANCE, 0.8),
                (PATTERN, 0.6),
                (SOUND, 0.4),
                (RITUAL, 0.7),
                (SACRED, 0.5),
            ],
        },
        Concept {
            label: "Eschatology",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.9),
                (MIND, 0.5),
                (EMOTION, 0.4),
                (NARRATIVE, 0.4),
                (DOCTRINE, 0.5),
            ],
        },
        Concept {
            label: "Homiletics",
            category: "religion",
            features: vec![
                (LANGUAGE, 0.7),
                (PERFORMANCE, 0.7),
                (METAPHYSICS, 0.6),
                (EMOTION, 0.5),
                (DISCOURSE, 0.4),
            ],
        },
        Concept {
            label: "Patristics",
            category: "religion",
            features: vec![
                (LANGUAGE, 0.7),
                (METAPHYSICS, 0.7),
                (EVOLUTION, 0.4),
                (NARRATIVE, 0.3),
                (HISTORICAL, 0.4),
                (DOCTRINE, 0.4),
            ],
        },
        Concept {
            label: "Religious education",
            category: "religion",
            features: vec![
                (PEDAGOGY, 0.6),
                (COGNITION, 0.5),
                (METAPHYSICS, 0.7),
                (BEHAVIOR, 0.4),
                (LEARNING, 0.3),
                (DOCTRINE, 0.3),
            ],
        },
        Concept {
            label: "Hindu studies",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.8),
                (BEHAVIOR, 0.5),
                (LANGUAGE, 0.5),
                (PATTERN, 0.4),
                (SPIRITUAL, 0.5),
                (SACRED, 0.4),
            ],
        },
        Concept {
            label: "Jewish studies",
            category: "religion",
            features: vec![
                (LANGUAGE, 0.7),
                (METAPHYSICS, 0.7),
                (ETHICS, 0.6),
                (BEHAVIOR, 0.5),
                (SACRED, 0.4),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Religious art",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.6),
                (VISUAL, 0.6),
                (EMOTION, 0.7),
                (PERFORMANCE, 0.3),
                (SACRED, 0.5),
            ],
        },
        Concept {
            label: "Sociology of religion (religion)",
            category: "religion",
            features: vec![
                (BEHAVIOR, 0.7),
                (METAPHYSICS, 0.5),
                (SYSTEMS, 0.5),
                (NETWORK, 0.3),
                (SOCIETY, 0.3),
                (RITUAL, 0.3),
            ],
        },
        Concept {
            label: "Religious epistemology",
            category: "religion",
            features: vec![
                (METAPHYSICS, 0.8),
                (LOGIC, 0.6),
                (MIND, 0.6),
                (COGNITION, 0.4),
                (EPISTEMOLOGY, 0.6),
            ],
        },
        Concept {
            label: "Liberation theology",
            category: "religion",
            features: vec![
                (ETHICS, 0.8),
                (METAPHYSICS, 0.6),
                (BEHAVIOR, 0.5),
                (GOVERNANCE, 0.3),
                (JUSTICE, 0.4),
                (MORAL, 0.4),
            ],
        },
        Concept {
            label: "Interreligious dialogue",
            category: "religion",
            features: vec![
                (LANGUAGE, 0.6),
                (METAPHYSICS, 0.5),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.5),
                (DISCOURSE, 0.4),
            ],
        },
        Concept {
            label: "Sacred texts studies",
            category: "religion",
            features: vec![
                (LANGUAGE, 0.8),
                (METAPHYSICS, 0.7),
                (NARRATIVE, 0.5),
                (STRUCTURE, 0.3),
                (SACRED, 0.7),
                (ARCHIVAL, 0.3),
            ],
        },
        Concept {
            label: "Ritual studies",
            category: "religion",
            features: vec![
                (BEHAVIOR, 0.7),
                (PERFORMANCE, 0.7),
                (PATTERN, 0.6),
                (METAPHYSICS, 0.5),
                (RITUAL, 0.8),
                (TRADITION, 0.4),
            ],
        },
        Concept {
            label: "New religious movements",
            category: "religion",
            features: vec![
                (BEHAVIOR, 0.6),
                (METAPHYSICS, 0.5),
                (EVOLUTION, 0.5),
                (NETWORK, 0.3),
                (SPIRITUAL, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        // ── ENVIRONMENTAL SCIENCE (25) ──
        Concept {
            label: "Climate change science",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.9),
                (NATURE, 0.8),
                (SYSTEMS, 0.7),
                (STATISTICS, 0.5),
                (CLIMATE, 0.9),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Conservation ecology",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.9),
                (LIFE, 0.7),
                (NATURE, 0.8),
                (ETHICS, 0.5),
                (CONSERVATION, 0.9),
            ],
        },
        Concept {
            label: "Pollution science",
            category: "environmental_science",
            features: vec![
                (CHEMISTRY, 0.7),
                (NATURE, 0.7),
                (ECOSYSTEM, 0.7),
                (MEASUREMENT, 0.5),
                (WATER, 0.3),
            ],
        },
        Concept {
            label: "Ecosystem services",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (MARKETS, 0.4),
                (SYSTEMS, 0.6),
                (NATURE, 0.6),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Biodiversity science",
            category: "environmental_science",
            features: vec![
                (LIFE, 0.8),
                (ECOSYSTEM, 0.9),
                (EVOLUTION, 0.6),
                (PATTERN, 0.4),
                (CONSERVATION, 0.5),
            ],
        },
        Concept {
            label: "Environmental toxicology",
            category: "environmental_science",
            features: vec![
                (CHEMISTRY, 0.7),
                (LIFE, 0.6),
                (ECOSYSTEM, 0.6),
                (DIAGNOSTICS, 0.4),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Renewable energy science",
            category: "environmental_science",
            features: vec![
                (ENERGY, 0.9),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.5),
                (SYSTEMS, 0.5),
                (CONSERVATION, 0.4),
                (ELECTRICAL, 0.3),
            ],
        },
        Concept {
            label: "Carbon cycle science",
            category: "environmental_science",
            features: vec![
                (CHEMISTRY, 0.6),
                (ECOSYSTEM, 0.8),
                (SYSTEMS, 0.6),
                (NATURE, 0.5),
                (CYCLE, 0.7),
                (CLIMATE, 0.5),
            ],
        },
        Concept {
            label: "Water resource management",
            category: "environmental_science",
            features: vec![
                (NATURE, 0.7),
                (ECOSYSTEM, 0.6),
                (GOVERNANCE, 0.4),
                (SYSTEMS, 0.6),
                (WATER, 0.8),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Waste management science",
            category: "environmental_science",
            features: vec![
                (CHEMISTRY, 0.5),
                (ECOSYSTEM, 0.6),
                (SYSTEMS, 0.6),
                (ETHICS, 0.3),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Environmental remote sensing",
            category: "environmental_science",
            features: vec![
                (MEASUREMENT, 0.8),
                (COMPUTATION, 0.5),
                (NATURE, 0.5),
                (ECOSYSTEM, 0.5),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Sustainability science",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (ETHICS, 0.6),
                (SYSTEMS, 0.7),
                (BEHAVIOR, 0.4),
                (CONSERVATION, 0.6),
                (MORAL, 0.3),
            ],
        },
        Concept {
            label: "Ecological modeling",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.7),
                (COMPUTATION, 0.6),
                (MATH, 0.5),
                (SYSTEMS, 0.6),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Environmental policy",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.6),
                (GOVERNANCE, 0.7),
                (ETHICS, 0.5),
                (BEHAVIOR, 0.4),
                (POLICY, 0.6),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Agroecology",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (LIFE, 0.6),
                (NATURE, 0.6),
                (SYSTEMS, 0.4),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Environmental impact assessment",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.7),
                (MEASUREMENT, 0.6),
                (STATISTICS, 0.5),
                (GOVERNANCE, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Restoration ecology",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.9),
                (NATURE, 0.7),
                (LIFE, 0.6),
                (SYSTEMS, 0.4),
                (CONSERVATION, 0.7),
            ],
        },
        Concept {
            label: "Environmental microbiology",
            category: "environmental_science",
            features: vec![
                (LIFE, 0.7),
                (ECOSYSTEM, 0.7),
                (CHEMISTRY, 0.5),
                (NATURE, 0.5),
                (CELLULAR, 0.3),
            ],
        },
        Concept {
            label: "Landscape ecology (env)",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (NATURE, 0.7),
                (PATTERN, 0.5),
                (SPACE, 0.4),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Environmental geochemistry",
            category: "environmental_science",
            features: vec![
                (CHEMISTRY, 0.7),
                (NATURE, 0.6),
                (ECOSYSTEM, 0.5),
                (MEASUREMENT, 0.4),
                (GEOLOGY, 0.3),
            ],
        },
        Concept {
            label: "Marine ecology",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (LIFE, 0.7),
                (NATURE, 0.7),
                (SYSTEMS, 0.4),
                (OCEAN, 0.6),
            ],
        },
        Concept {
            label: "Urban ecology",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.7),
                (BEHAVIOR, 0.4),
                (SYSTEMS, 0.5),
                (STRUCTURE, 0.4),
                (COMMUNITY, 0.3),
            ],
        },
        Concept {
            label: "Environmental data science",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.5),
                (COMPUTATION, 0.6),
                (STATISTICS, 0.7),
                (MEASUREMENT, 0.5),
                (DATA, 0.6),
                (CLIMATE, 0.3),
            ],
        },
        Concept {
            label: "Invasive species biology",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.8),
                (LIFE, 0.7),
                (EVOLUTION, 0.5),
                (NATURE, 0.5),
                (CONSERVATION, 0.4),
            ],
        },
        Concept {
            label: "Circular economy",
            category: "environmental_science",
            features: vec![
                (ECOSYSTEM, 0.6),
                (MARKETS, 0.5),
                (SYSTEMS, 0.6),
                (ETHICS, 0.4),
                (CYCLE, 0.5),
                (CONSERVATION, 0.4),
            ],
        },
        // ── NEUROSCIENCE (25) ──
        Concept {
            label: "Cognitive neuroscience",
            category: "neuroscience",
            features: vec![
                (COGNITION, 0.9),
                (LIFE, 0.6),
                (MIND, 0.7),
                (NETWORK, 0.5),
                (BRAIN, 0.8),
                (NEURAL, 0.6),
            ],
        },
        Concept {
            label: "Computational neuroscience",
            category: "neuroscience",
            features: vec![
                (COMPUTATION, 0.7),
                (LIFE, 0.5),
                (NETWORK, 0.7),
                (MATH, 0.5),
                (COGNITION, 0.4),
                (NEURAL, 0.6),
                (ALGORITHM, 0.3),
            ],
        },
        Concept {
            label: "Cellular neuroscience",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.9),
                (CHEMISTRY, 0.6),
                (NETWORK, 0.5),
                (STRUCTURE, 0.4),
                (NEURAL, 0.7),
                (CELLULAR, 0.6),
            ],
        },
        Concept {
            label: "Behavioral neuroscience",
            category: "neuroscience",
            features: vec![
                (BEHAVIOR, 0.8),
                (LIFE, 0.6),
                (COGNITION, 0.5),
                (MIND, 0.4),
                (BRAIN, 0.6),
                (NEURAL, 0.4),
            ],
        },
        Concept {
            label: "Neuroimaging",
            category: "neuroscience",
            features: vec![
                (MEASUREMENT, 0.8),
                (LIFE, 0.5),
                (VISUAL, 0.5),
                (COMPUTATION, 0.5),
                (BRAIN, 0.7),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Synaptic neuroscience",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.8),
                (CHEMISTRY, 0.7),
                (NETWORK, 0.6),
                (STRUCTURE, 0.4),
                (NEURAL, 0.8),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Motor neuroscience",
            category: "neuroscience",
            features: vec![
                (MOTION, 0.8),
                (LIFE, 0.6),
                (COGNITION, 0.5),
                (FORCE, 0.3),
                (BRAIN, 0.5),
                (NEURAL, 0.5),
            ],
        },
        Concept {
            label: "Sensory neuroscience",
            category: "neuroscience",
            features: vec![
                (WAVE, 0.5),
                (LIFE, 0.6),
                (COGNITION, 0.6),
                (PATTERN, 0.5),
                (BRAIN, 0.5),
                (NEURAL, 0.5),
            ],
        },
        Concept {
            label: "Developmental neuroscience",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.8),
                (EVOLUTION, 0.4),
                (GENETICS, 0.5),
                (COGNITION, 0.4),
                (BRAIN, 0.5),
                (CELLULAR, 0.4),
            ],
        },
        Concept {
            label: "Neuroendocrinology",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.7),
                (CHEMISTRY, 0.7),
                (BEHAVIOR, 0.5),
                (SYSTEMS, 0.3),
                (BRAIN, 0.4),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Social neuroscience",
            category: "neuroscience",
            features: vec![
                (COGNITION, 0.6),
                (BEHAVIOR, 0.7),
                (NETWORK, 0.5),
                (EMOTION, 0.5),
                (BRAIN, 0.5),
            ],
        },
        Concept {
            label: "Affective neuroscience",
            category: "neuroscience",
            features: vec![
                (EMOTION, 0.9),
                (COGNITION, 0.6),
                (LIFE, 0.5),
                (MIND, 0.5),
                (BRAIN, 0.6),
                (NEURAL, 0.4),
            ],
        },
        Concept {
            label: "Neural network theory",
            category: "neuroscience",
            features: vec![
                (NETWORK, 0.9),
                (COMPUTATION, 0.7),
                (PATTERN, 0.6),
                (MATH, 0.4),
                (NEURAL, 0.8),
                (AI, 0.3),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Neurogenetics",
            category: "neuroscience",
            features: vec![
                (GENETICS, 0.8),
                (LIFE, 0.7),
                (COGNITION, 0.4),
                (INFORMATION, 0.3),
                (BRAIN, 0.4),
                (MOLECULAR, 0.3),
            ],
        },
        Concept {
            label: "Neuroplasticity",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.7),
                (COGNITION, 0.6),
                (EVOLUTION, 0.4),
                (NETWORK, 0.5),
                (BRAIN, 0.6),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Pain neuroscience",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.6),
                (EMOTION, 0.5),
                (DIAGNOSTICS, 0.4),
                (CHEMISTRY, 0.4),
                (NEURAL, 0.5),
                (BRAIN, 0.4),
            ],
        },
        Concept {
            label: "Neuroethics",
            category: "neuroscience",
            features: vec![
                (ETHICS, 0.7),
                (MIND, 0.6),
                (LIFE, 0.4),
                (COGNITION, 0.4),
                (BRAIN, 0.4),
                (MORAL, 0.4),
            ],
        },
        Concept {
            label: "Connectomics",
            category: "neuroscience",
            features: vec![
                (NETWORK, 0.9),
                (LIFE, 0.5),
                (STRUCTURE, 0.6),
                (COMPUTATION, 0.5),
                (BRAIN, 0.6),
                (NEURAL, 0.6),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Visual neuroscience",
            category: "neuroscience",
            features: vec![
                (VISUAL, 0.7),
                (COGNITION, 0.6),
                (LIFE, 0.5),
                (WAVE, 0.3),
                (BRAIN, 0.5),
                (NEURAL, 0.4),
            ],
        },
        Concept {
            label: "Auditory neuroscience",
            category: "neuroscience",
            features: vec![
                (SOUND, 0.7),
                (COGNITION, 0.6),
                (LIFE, 0.5),
                (WAVE, 0.4),
                (BRAIN, 0.5),
                (NEURAL, 0.4),
            ],
        },
        Concept {
            label: "Neuropharmacology",
            category: "neuroscience",
            features: vec![
                (CHEMISTRY, 0.8),
                (LIFE, 0.6),
                (DIAGNOSTICS, 0.4),
                (BEHAVIOR, 0.3),
                (NEURAL, 0.5),
                (THERAPY, 0.3),
            ],
        },
        Concept {
            label: "Neuroinformatics",
            category: "neuroscience",
            features: vec![
                (COMPUTATION, 0.7),
                (INFORMATION, 0.7),
                (LIFE, 0.4),
                (STATISTICS, 0.4),
                (DATA, 0.5),
                (BRAIN, 0.3),
            ],
        },
        Concept {
            label: "Sleep neuroscience",
            category: "neuroscience",
            features: vec![
                (LIFE, 0.7),
                (COGNITION, 0.5),
                (BEHAVIOR, 0.4),
                (PATTERN, 0.4),
                (BRAIN, 0.5),
                (CYCLE, 0.4),
            ],
        },
        Concept {
            label: "Language neuroscience",
            category: "neuroscience",
            features: vec![
                (LANGUAGE, 0.6),
                (COGNITION, 0.7),
                (LIFE, 0.5),
                (NETWORK, 0.4),
                (BRAIN, 0.6),
                (NEURAL, 0.4),
            ],
        },
        Concept {
            label: "Decision neuroscience",
            category: "neuroscience",
            features: vec![
                (COGNITION, 0.7),
                (BEHAVIOR, 0.6),
                (OPTIMIZATION, 0.4),
                (MIND, 0.5),
                (BRAIN, 0.5),
            ],
        },
        // ── DATA SCIENCE (25) ──
        Concept {
            label: "Statistical learning",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.9),
                (COMPUTATION, 0.7),
                (PATTERN, 0.7),
                (MATH, 0.5),
                (MACHINE_LEARN, 0.6),
                (DATA, 0.5),
            ],
        },
        Concept {
            label: "Data visualization",
            category: "data_science",
            features: vec![
                (VISUAL, 0.8),
                (INFORMATION, 0.7),
                (PATTERN, 0.6),
                (COGNITION, 0.4),
                (DATA, 0.6),
                (DESIGN, 0.3),
            ],
        },
        Concept {
            label: "Feature engineering",
            category: "data_science",
            features: vec![
                (PATTERN, 0.7),
                (COMPUTATION, 0.6),
                (OPTIMIZATION, 0.5),
                (STATISTICS, 0.5),
                (DATA, 0.5),
                (MACHINE_LEARN, 0.4),
            ],
        },
        Concept {
            label: "Deep learning",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.9),
                (PATTERN, 0.8),
                (MATH, 0.5),
                (NETWORK, 0.5),
                (MACHINE_LEARN, 0.8),
                (AI, 0.5),
                (NEURAL, 0.4),
            ],
        },
        Concept {
            label: "Natural language processing (DS)",
            category: "data_science",
            features: vec![
                (LANGUAGE, 0.7),
                (COMPUTATION, 0.7),
                (STATISTICS, 0.5),
                (PATTERN, 0.6),
                (LLM, 0.5),
                (AI, 0.4),
                (DATA, 0.3),
            ],
        },
        Concept {
            label: "Time series analysis",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.8),
                (PATTERN, 0.7),
                (MATH, 0.6),
                (MEASUREMENT, 0.4),
                (DATA, 0.5),
                (CYCLE, 0.3),
            ],
        },
        Concept {
            label: "Bayesian methods",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.9),
                (MATH, 0.7),
                (LOGIC, 0.5),
                (INFORMATION, 0.4),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Causal inference",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.8),
                (LOGIC, 0.7),
                (MATH, 0.5),
                (BEHAVIOR, 0.3),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Recommender systems",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.7),
                (PATTERN, 0.7),
                (BEHAVIOR, 0.5),
                (OPTIMIZATION, 0.5),
                (MACHINE_LEARN, 0.5),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Anomaly detection",
            category: "data_science",
            features: vec![
                (PATTERN, 0.8),
                (STATISTICS, 0.7),
                (COMPUTATION, 0.5),
                (DIAGNOSTICS, 0.3),
                (DATA, 0.4),
                (ALGORITHM, 0.3),
            ],
        },
        Concept {
            label: "Dimensionality reduction",
            category: "data_science",
            features: vec![
                (MATH, 0.7),
                (COMPUTATION, 0.6),
                (PATTERN, 0.6),
                (STRUCTURE, 0.5),
                (DATA, 0.4),
                (ALGORITHM, 0.3),
            ],
        },
        Concept {
            label: "A/B testing",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.8),
                (BEHAVIOR, 0.5),
                (MEASUREMENT, 0.6),
                (OPTIMIZATION, 0.4),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Data engineering",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.8),
                (SYSTEMS, 0.7),
                (INFORMATION, 0.6),
                (STRUCTURE, 0.4),
                (DATA, 0.8),
                (SOFTWARE, 0.4),
            ],
        },
        Concept {
            label: "Geospatial analytics",
            category: "data_science",
            features: vec![
                (SPACE, 0.6),
                (COMPUTATION, 0.5),
                (STATISTICS, 0.5),
                (VISUAL, 0.5),
                (DATA, 0.5),
                (PLANETARY, 0.2),
            ],
        },
        Concept {
            label: "Graph analytics",
            category: "data_science",
            features: vec![
                (NETWORK, 0.8),
                (COMPUTATION, 0.6),
                (PATTERN, 0.6),
                (MATH, 0.4),
                (GRAPH_THEORY, 0.5),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Survival analysis",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.8),
                (LIFE, 0.4),
                (MATH, 0.5),
                (MEASUREMENT, 0.4),
                (DATA, 0.4),
                (CLINICAL, 0.2),
            ],
        },
        Concept {
            label: "Ensemble methods",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.7),
                (STATISTICS, 0.6),
                (OPTIMIZATION, 0.6),
                (PATTERN, 0.5),
                (MACHINE_LEARN, 0.5),
            ],
        },
        Concept {
            label: "Interpretable ML",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.6),
                (COGNITION, 0.5),
                (PATTERN, 0.5),
                (LOGIC, 0.5),
                (MACHINE_LEARN, 0.5),
                (AI, 0.3),
            ],
        },
        Concept {
            label: "Data ethics",
            category: "data_science",
            features: vec![
                (ETHICS, 0.7),
                (INFORMATION, 0.6),
                (BEHAVIOR, 0.4),
                (GOVERNANCE, 0.4),
                (DATA, 0.5),
                (MORAL, 0.4),
            ],
        },
        Concept {
            label: "Transfer learning",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.7),
                (PATTERN, 0.6),
                (OPTIMIZATION, 0.5),
                (INFORMATION, 0.4),
                (MACHINE_LEARN, 0.6),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Text mining",
            category: "data_science",
            features: vec![
                (LANGUAGE, 0.6),
                (COMPUTATION, 0.6),
                (PATTERN, 0.6),
                (STATISTICS, 0.5),
                (DATA, 0.5),
                (LLM, 0.2),
            ],
        },
        Concept {
            label: "Computer vision (DS)",
            category: "data_science",
            features: vec![
                (VISUAL, 0.8),
                (COMPUTATION, 0.7),
                (PATTERN, 0.7),
                (MATH, 0.3),
                (MACHINE_LEARN, 0.5),
                (AI, 0.3),
            ],
        },
        Concept {
            label: "Streaming analytics",
            category: "data_science",
            features: vec![
                (COMPUTATION, 0.7),
                (SYSTEMS, 0.6),
                (INFORMATION, 0.5),
                (STATISTICS, 0.4),
                (DATA, 0.6),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Experiment design",
            category: "data_science",
            features: vec![
                (STATISTICS, 0.8),
                (LOGIC, 0.5),
                (MEASUREMENT, 0.6),
                (OPTIMIZATION, 0.4),
                (DATA, 0.4),
            ],
        },
        Concept {
            label: "Fairness in ML",
            category: "data_science",
            features: vec![
                (ETHICS, 0.7),
                (COMPUTATION, 0.5),
                (STATISTICS, 0.5),
                (BEHAVIOR, 0.4),
                (MACHINE_LEARN, 0.4),
                (MORAL, 0.3),
                (JUSTICE, 0.3),
            ],
        },
        // ── ANTHROPOLOGY (25) ──
        Concept {
            label: "Cultural anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.9),
                (LANGUAGE, 0.5),
                (PATTERN, 0.5),
                (EVOLUTION, 0.4),
                (CULTURE, 0.8),
                (TRADITION, 0.4),
            ],
        },
        Concept {
            label: "Physical anthropology",
            category: "anthropology",
            features: vec![
                (LIFE, 0.7),
                (EVOLUTION, 0.8),
                (STRUCTURE, 0.5),
                (GENETICS, 0.4),
                (ANATOMY, 0.4),
            ],
        },
        Concept {
            label: "Linguistic anthropology",
            category: "anthropology",
            features: vec![
                (LANGUAGE, 0.8),
                (BEHAVIOR, 0.7),
                (COGNITION, 0.4),
                (EVOLUTION, 0.3),
                (CULTURE, 0.5),
            ],
        },
        Concept {
            label: "Archaeological anthropology",
            category: "anthropology",
            features: vec![
                (EVOLUTION, 0.6),
                (MATERIAL, 0.6),
                (BEHAVIOR, 0.6),
                (STRUCTURE, 0.5),
                (HISTORICAL, 0.4),
                (ARCHIVAL, 0.3),
            ],
        },
        Concept {
            label: "Medical anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (LIFE, 0.5),
                (DIAGNOSTICS, 0.3),
                (ETHICS, 0.3),
                (CULTURE, 0.4),
                (THERAPY, 0.2),
            ],
        },
        Concept {
            label: "Economic anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (MARKETS, 0.5),
                (SYSTEMS, 0.4),
                (PATTERN, 0.4),
                (CULTURE, 0.4),
                (MONEY, 0.2),
            ],
        },
        Concept {
            label: "Political anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (GOVERNANCE, 0.5),
                (SYSTEMS, 0.5),
                (NETWORK, 0.3),
                (POWER, 0.4),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Visual anthropology",
            category: "anthropology",
            features: vec![
                (VISUAL, 0.6),
                (BEHAVIOR, 0.7),
                (PATTERN, 0.4),
                (LANGUAGE, 0.3),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Kinship studies",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (NETWORK, 0.6),
                (GENETICS, 0.3),
                (SYSTEMS, 0.4),
                (KINSHIP, 0.9),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Ritual anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.8),
                (PERFORMANCE, 0.6),
                (PATTERN, 0.5),
                (METAPHYSICS, 0.4),
                (RITUAL, 0.7),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Urban anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (STRUCTURE, 0.4),
                (NETWORK, 0.5),
                (SYSTEMS, 0.4),
                (COMMUNITY, 0.4),
            ],
        },
        Concept {
            label: "Ethnobotany",
            category: "anthropology",
            features: vec![
                (LIFE, 0.6),
                (NATURE, 0.6),
                (BEHAVIOR, 0.6),
                (CHEMISTRY, 0.3),
                (CULTURE, 0.4),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Digital anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.6),
                (COMPUTATION, 0.5),
                (NETWORK, 0.5),
                (INFORMATION, 0.4),
                (SOCIAL_NETWORK, 0.3),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Food anthropology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (CHEMISTRY, 0.3),
                (EVOLUTION, 0.4),
                (PATTERN, 0.5),
                (CULTURE, 0.5),
                (COOKING, 0.3),
            ],
        },
        Concept {
            label: "Anthropology of religion",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.7),
                (METAPHYSICS, 0.6),
                (COGNITION, 0.3),
                (PATTERN, 0.4),
                (RITUAL, 0.4),
                (SPIRITUAL, 0.3),
            ],
        },
        Concept {
            label: "Anthropology of art",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.6),
                (VISUAL, 0.5),
                (EMOTION, 0.4),
                (PATTERN, 0.5),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Primatology",
            category: "anthropology",
            features: vec![
                (LIFE, 0.7),
                (EVOLUTION, 0.7),
                (BEHAVIOR, 0.7),
                (COGNITION, 0.4),
                (BRAIN, 0.2),
            ],
        },
        Concept {
            label: "Forensic anthropology",
            category: "anthropology",
            features: vec![
                (LIFE, 0.5),
                (STRUCTURE, 0.6),
                (DIAGNOSTICS, 0.5),
                (EVOLUTION, 0.3),
                (ANATOMY, 0.5),
                (LEGAL, 0.2),
            ],
        },
        Concept {
            label: "Ecological anthropology",
            category: "anthropology",
            features: vec![
                (ECOSYSTEM, 0.6),
                (BEHAVIOR, 0.7),
                (NATURE, 0.6),
                (SYSTEMS, 0.3),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Anthropology of development",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.6),
                (MARKETS, 0.4),
                (GOVERNANCE, 0.4),
                (SYSTEMS, 0.4),
                (CULTURE, 0.3),
                (POLICY, 0.2),
            ],
        },
        Concept {
            label: "Ethnoarchaeology",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.6),
                (MATERIAL, 0.6),
                (EVOLUTION, 0.5),
                (PATTERN, 0.4),
                (CULTURE, 0.4),
                (HISTORICAL, 0.3),
            ],
        },
        Concept {
            label: "Cognitive anthropology",
            category: "anthropology",
            features: vec![
                (COGNITION, 0.7),
                (BEHAVIOR, 0.6),
                (LANGUAGE, 0.5),
                (MIND, 0.4),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Anthropology of space",
            category: "anthropology",
            features: vec![
                (BEHAVIOR, 0.6),
                (SPACE, 0.5),
                (STRUCTURE, 0.4),
                (PATTERN, 0.4),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Ethnomedicine",
            category: "anthropology",
            features: vec![
                (LIFE, 0.5),
                (BEHAVIOR, 0.6),
                (CHEMISTRY, 0.3),
                (DIAGNOSTICS, 0.3),
                (CULTURE, 0.4),
                (TRADITION, 0.3),
            ],
        },
        Concept {
            label: "Folklore studies",
            category: "anthropology",
            features: vec![
                (NARRATIVE, 0.6),
                (LANGUAGE, 0.5),
                (BEHAVIOR, 0.6),
                (PATTERN, 0.5),
                (TRADITION, 0.5),
                (CULTURE, 0.4),
            ],
        },
        // ── EDUCATION (25) ──
        Concept {
            label: "Curriculum design",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.9),
                (STRUCTURE, 0.6),
                (COGNITION, 0.5),
                (SYSTEMS, 0.4),
                (CURRICULUM, 0.9),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Assessment theory",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (MEASUREMENT, 0.8),
                (STATISTICS, 0.6),
                (COGNITION, 0.4),
                (ASSESSMENT, 0.9),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Learning theory",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.8),
                (COGNITION, 0.8),
                (MIND, 0.5),
                (BEHAVIOR, 0.4),
                (LEARNING, 0.8),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Educational technology",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (COMPUTATION, 0.7),
                (INFORMATION, 0.5),
                (SYSTEMS, 0.3),
                (SOFTWARE, 0.3),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Special education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.8),
                (COGNITION, 0.5),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.4),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Higher education studies",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (SYSTEMS, 0.6),
                (GOVERNANCE, 0.4),
                (BEHAVIOR, 0.4),
                (CURRICULUM, 0.3),
            ],
        },
        Concept {
            label: "Early childhood education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.8),
                (COGNITION, 0.6),
                (BEHAVIOR, 0.5),
                (EVOLUTION, 0.3),
                (LEARNING, 0.6),
                (ATTACHMENT, 0.3),
            ],
        },
        Concept {
            label: "STEM education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (MATH, 0.4),
                (COMPUTATION, 0.3),
                (COGNITION, 0.5),
                (CURRICULUM, 0.4),
                (LEARNING, 0.3),
            ],
        },
        Concept {
            label: "Literacy studies",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (LANGUAGE, 0.8),
                (COGNITION, 0.5),
                (PATTERN, 0.3),
                (LEARNING, 0.5),
            ],
        },
        Concept {
            label: "Multicultural education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (BEHAVIOR, 0.6),
                (LANGUAGE, 0.4),
                (ETHICS, 0.4),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Philosophy of education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.6),
                (MIND, 0.6),
                (ETHICS, 0.5),
                (METAPHYSICS, 0.4),
                (EPISTEMOLOGY, 0.3),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Distance education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (COMPUTATION, 0.5),
                (NETWORK, 0.5),
                (INFORMATION, 0.4),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Instructional design",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.8),
                (COGNITION, 0.6),
                (STRUCTURE, 0.5),
                (OPTIMIZATION, 0.3),
                (CURRICULUM, 0.5),
                (DESIGN, 0.3),
            ],
        },
        Concept {
            label: "Music education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (SOUND, 0.6),
                (COGNITION, 0.5),
                (PERFORMANCE, 0.4),
                (HARMONY, 0.2),
                (LEARNING, 0.3),
            ],
        },
        Concept {
            label: "Arts education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (VISUAL, 0.5),
                (EMOTION, 0.5),
                (COGNITION, 0.4),
                (LEARNING, 0.3),
            ],
        },
        Concept {
            label: "Language education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.8),
                (LANGUAGE, 0.8),
                (COGNITION, 0.5),
                (BEHAVIOR, 0.3),
                (LEARNING, 0.5),
                (GRAMMAR, 0.2),
            ],
        },
        Concept {
            label: "Science education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (LOGIC, 0.4),
                (MEASUREMENT, 0.3),
                (COGNITION, 0.5),
                (CURRICULUM, 0.3),
                (LEARNING, 0.3),
            ],
        },
        Concept {
            label: "Mathematics education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (MATH, 0.6),
                (COGNITION, 0.5),
                (LOGIC, 0.3),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Physical education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.6),
                (MOTION, 0.7),
                (LIFE, 0.4),
                (BEHAVIOR, 0.4),
                (ANATOMY, 0.2),
            ],
        },
        Concept {
            label: "Environmental education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.7),
                (ECOSYSTEM, 0.5),
                (NATURE, 0.5),
                (ETHICS, 0.3),
                (LEARNING, 0.3),
                (CONSERVATION, 0.3),
            ],
        },
        Concept {
            label: "Education policy",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.6),
                (GOVERNANCE, 0.7),
                (SYSTEMS, 0.5),
                (STATISTICS, 0.3),
                (POLICY, 0.5),
                (CURRICULUM, 0.3),
            ],
        },
        Concept {
            label: "Teacher education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.9),
                (BEHAVIOR, 0.5),
                (COGNITION, 0.4),
                (SYSTEMS, 0.3),
                (LEARNING, 0.4),
            ],
        },
        Concept {
            label: "Game-based learning",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.6),
                (COMPUTATION, 0.5),
                (COGNITION, 0.6),
                (BEHAVIOR, 0.4),
                (LEARNING, 0.5),
            ],
        },
        Concept {
            label: "Comparative education",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.6),
                (SYSTEMS, 0.5),
                (BEHAVIOR, 0.4),
                (PATTERN, 0.4),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Educational measurement",
            category: "education",
            features: vec![
                (PEDAGOGY, 0.6),
                (MEASUREMENT, 0.8),
                (STATISTICS, 0.7),
                (COGNITION, 0.3),
                (ASSESSMENT, 0.7),
                (DATA, 0.3),
            ],
        },
        // ── FILM STUDIES (25) ──
        Concept {
            label: "Film theory",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (NARRATIVE, 0.6),
                (MIND, 0.5),
                (METAPHYSICS, 0.4),
                (CINEMA, 0.7),
                (THEORY, 0.5),
            ],
        },
        Concept {
            label: "Cinematography",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.9),
                (WAVE, 0.4),
                (MOTION, 0.6),
                (PATTERN, 0.5),
                (CINEMA, 0.8),
                (COLOR, 0.3),
            ],
        },
        Concept {
            label: "Film editing",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (NARRATIVE, 0.8),
                (PATTERN, 0.6),
                (COGNITION, 0.3),
                (CINEMA, 0.6),
                (MONTAGE, 0.8),
            ],
        },
        Concept {
            label: "Sound design (film)",
            category: "film_studies",
            features: vec![
                (SOUND, 0.8),
                (EMOTION, 0.6),
                (NARRATIVE, 0.4),
                (WAVE, 0.4),
                (CINEMA, 0.5),
                (TIMBRE, 0.3),
            ],
        },
        Concept {
            label: "Screenwriting (film)",
            category: "film_studies",
            features: vec![
                (NARRATIVE, 0.9),
                (LANGUAGE, 0.7),
                (EMOTION, 0.6),
                (STRUCTURE, 0.5),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Documentary studies",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (NARRATIVE, 0.7),
                (BEHAVIOR, 0.5),
                (ETHICS, 0.3),
                (CINEMA, 0.6),
            ],
        },
        Concept {
            label: "Animation studies",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.8),
                (MOTION, 0.7),
                (COMPUTATION, 0.4),
                (NARRATIVE, 0.5),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Film history",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.5),
                (NARRATIVE, 0.5),
                (EVOLUTION, 0.5),
                (BEHAVIOR, 0.3),
                (CINEMA, 0.6),
                (HISTORICAL, 0.4),
            ],
        },
        Concept {
            label: "Genre studies",
            category: "film_studies",
            features: vec![
                (PATTERN, 0.7),
                (NARRATIVE, 0.6),
                (EMOTION, 0.5),
                (STRUCTURE, 0.4),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Film semiotics",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.6),
                (LANGUAGE, 0.5),
                (PATTERN, 0.5),
                (COGNITION, 0.4),
                (CINEMA, 0.5),
                (SEMANTICS_AX, 0.3),
            ],
        },
        Concept {
            label: "National cinema studies",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.5),
                (BEHAVIOR, 0.5),
                (LANGUAGE, 0.4),
                (PATTERN, 0.4),
                (CINEMA, 0.6),
                (CULTURE, 0.4),
            ],
        },
        Concept {
            label: "Film and philosophy",
            category: "film_studies",
            features: vec![
                (MIND, 0.6),
                (VISUAL, 0.5),
                (METAPHYSICS, 0.5),
                (NARRATIVE, 0.4),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Visual effects",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.8),
                (COMPUTATION, 0.7),
                (WAVE, 0.3),
                (PATTERN, 0.4),
                (CINEMA, 0.5),
                (SOFTWARE, 0.3),
            ],
        },
        Concept {
            label: "Production design",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (SPACE, 0.6),
                (MATERIAL, 0.4),
                (EMOTION, 0.4),
                (CINEMA, 0.5),
                (DESIGN, 0.4),
            ],
        },
        Concept {
            label: "Film music",
            category: "film_studies",
            features: vec![
                (SOUND, 0.7),
                (EMOTION, 0.8),
                (NARRATIVE, 0.5),
                (VISUAL, 0.3),
                (CINEMA, 0.5),
                (HARMONY, 0.3),
            ],
        },
        Concept {
            label: "Experimental film",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (PATTERN, 0.5),
                (METAPHYSICS, 0.4),
                (EMOTION, 0.4),
                (CINEMA, 0.6),
                (MONTAGE, 0.3),
            ],
        },
        Concept {
            label: "Film criticism",
            category: "film_studies",
            features: vec![
                (LANGUAGE, 0.6),
                (VISUAL, 0.5),
                (MIND, 0.4),
                (NARRATIVE, 0.5),
                (CINEMA, 0.5),
                (DISCOURSE, 0.3),
            ],
        },
        Concept {
            label: "Color grading",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.8),
                (WAVE, 0.4),
                (EMOTION, 0.5),
                (PATTERN, 0.4),
                (CINEMA, 0.5),
                (COLOR, 0.7),
            ],
        },
        Concept {
            label: "Film and technology",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.5),
                (COMPUTATION, 0.5),
                (EVOLUTION, 0.3),
                (SYSTEMS, 0.3),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Adaptation studies",
            category: "film_studies",
            features: vec![
                (NARRATIVE, 0.7),
                (LANGUAGE, 0.5),
                (VISUAL, 0.5),
                (PATTERN, 0.4),
                (CINEMA, 0.4),
                (LITERARY, 0.3),
            ],
        },
        Concept {
            label: "Spectatorship theory",
            category: "film_studies",
            features: vec![
                (COGNITION, 0.6),
                (VISUAL, 0.5),
                (MIND, 0.5),
                (BEHAVIOR, 0.4),
                (CINEMA, 0.5),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Stop-motion animation",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (MOTION, 0.6),
                (MATERIAL, 0.5),
                (PERFORMANCE, 0.4),
                (CINEMA, 0.4),
            ],
        },
        Concept {
            label: "Film restoration",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.5),
                (MATERIAL, 0.5),
                (CHEMISTRY, 0.3),
                (EVOLUTION, 0.3),
                (CINEMA, 0.4),
                (ARCHIVAL, 0.4),
            ],
        },
        Concept {
            label: "Virtual reality cinema",
            category: "film_studies",
            features: vec![
                (VISUAL, 0.7),
                (COMPUTATION, 0.6),
                (SPACE, 0.6),
                (COGNITION, 0.4),
                (CINEMA, 0.5),
            ],
        },
        Concept {
            label: "Film distribution",
            category: "film_studies",
            features: vec![
                (MARKETS, 0.5),
                (NETWORK, 0.5),
                (INFORMATION, 0.4),
                (SYSTEMS, 0.3),
                (CINEMA, 0.4),
            ],
        },
        // ── PERFORMING ARTS (25) ──
        Concept {
            label: "Dance technique",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.9),
                (PERFORMANCE, 0.9),
                (FORCE, 0.4),
                (PATTERN, 0.5),
                (DANCE, 0.9),
                (RHYTHM, 0.4),
            ],
        },
        Concept {
            label: "Choreography",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.8),
                (PATTERN, 0.8),
                (PERFORMANCE, 0.7),
                (SPACE, 0.5),
                (DANCE, 0.7),
            ],
        },
        Concept {
            label: "Theater directing",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.8),
                (NARRATIVE, 0.6),
                (BEHAVIOR, 0.5),
                (LANGUAGE, 0.4),
                (THEATRICAL, 0.7),
            ],
        },
        Concept {
            label: "Acting method",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.9),
                (EMOTION, 0.8),
                (BEHAVIOR, 0.6),
                (COGNITION, 0.4),
                (THEATRICAL, 0.6),
            ],
        },
        Concept {
            label: "Stage design",
            category: "performing_arts",
            features: vec![
                (SPACE, 0.7),
                (VISUAL, 0.6),
                (STRUCTURE, 0.5),
                (PERFORMANCE, 0.5),
                (THEATRICAL, 0.5),
                (DESIGN, 0.4),
            ],
        },
        Concept {
            label: "Costume design",
            category: "performing_arts",
            features: vec![
                (VISUAL, 0.6),
                (MATERIAL, 0.6),
                (PERFORMANCE, 0.5),
                (EMOTION, 0.4),
                (THEATRICAL, 0.4),
                (DESIGN, 0.4),
            ],
        },
        Concept {
            label: "Stage lighting",
            category: "performing_arts",
            features: vec![
                (WAVE, 0.5),
                (VISUAL, 0.7),
                (EMOTION, 0.5),
                (PERFORMANCE, 0.5),
                (THEATRICAL, 0.4),
                (COLOR, 0.3),
            ],
        },
        Concept {
            label: "Mime and physical theater",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.8),
                (PERFORMANCE, 0.8),
                (EMOTION, 0.5),
                (COGNITION, 0.3),
                (THEATRICAL, 0.6),
            ],
        },
        Concept {
            label: "Circus arts",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.9),
                (MOTION, 0.8),
                (FORCE, 0.5),
                (EMOTION, 0.5),
                (THEATRICAL, 0.3),
            ],
        },
        Concept {
            label: "Dance therapy",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.7),
                (EMOTION, 0.8),
                (MIND, 0.5),
                (LIFE, 0.3),
                (DANCE, 0.5),
                (THERAPY, 0.5),
            ],
        },
        Concept {
            label: "Puppetry",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.7),
                (VISUAL, 0.5),
                (NARRATIVE, 0.5),
                (MOTION, 0.5),
                (THEATRICAL, 0.5),
            ],
        },
        Concept {
            label: "Musical theater",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.8),
                (SOUND, 0.7),
                (NARRATIVE, 0.6),
                (EMOTION, 0.7),
                (THEATRICAL, 0.7),
                (HARMONY, 0.2),
            ],
        },
        Concept {
            label: "Improvised theater",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.8),
                (COGNITION, 0.6),
                (BEHAVIOR, 0.5),
                (EMOTION, 0.5),
                (THEATRICAL, 0.6),
            ],
        },
        Concept {
            label: "Ballet",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.9),
                (PERFORMANCE, 0.8),
                (PATTERN, 0.7),
                (EMOTION, 0.5),
                (DANCE, 0.8),
                (RHYTHM, 0.3),
            ],
        },
        Concept {
            label: "Contemporary dance",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.8),
                (PERFORMANCE, 0.7),
                (EMOTION, 0.6),
                (SPACE, 0.4),
                (DANCE, 0.7),
            ],
        },
        Concept {
            label: "Theater history",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.5),
                (NARRATIVE, 0.5),
                (EVOLUTION, 0.5),
                (LANGUAGE, 0.4),
                (THEATRICAL, 0.5),
                (HISTORICAL, 0.4),
            ],
        },
        Concept {
            label: "Performance studies",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.9),
                (BEHAVIOR, 0.6),
                (MIND, 0.4),
                (COGNITION, 0.3),
                (THEATRICAL, 0.4),
                (THEORY, 0.3),
            ],
        },
        Concept {
            label: "Stage combat",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.7),
                (FORCE, 0.6),
                (MOTION, 0.7),
                (BEHAVIOR, 0.3),
                (THEATRICAL, 0.4),
            ],
        },
        Concept {
            label: "Voice and speech",
            category: "performing_arts",
            features: vec![
                (SOUND, 0.7),
                (PERFORMANCE, 0.7),
                (LANGUAGE, 0.5),
                (WAVE, 0.3),
                (PHONETIC, 0.3),
            ],
        },
        Concept {
            label: "Dance notation",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.6),
                (LANGUAGE, 0.5),
                (PATTERN, 0.6),
                (STRUCTURE, 0.5),
                (DANCE, 0.5),
            ],
        },
        Concept {
            label: "Devised theater",
            category: "performing_arts",
            features: vec![
                (PERFORMANCE, 0.7),
                (COGNITION, 0.5),
                (BEHAVIOR, 0.5),
                (NARRATIVE, 0.4),
                (THEATRICAL, 0.5),
            ],
        },
        Concept {
            label: "World dance traditions",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.7),
                (BEHAVIOR, 0.5),
                (PERFORMANCE, 0.6),
                (EVOLUTION, 0.4),
                (DANCE, 0.6),
                (TRADITION, 0.4),
                (CULTURE, 0.3),
            ],
        },
        Concept {
            label: "Theater criticism",
            category: "performing_arts",
            features: vec![
                (LANGUAGE, 0.6),
                (PERFORMANCE, 0.5),
                (MIND, 0.4),
                (NARRATIVE, 0.4),
                (THEATRICAL, 0.4),
                (DISCOURSE, 0.3),
            ],
        },
        Concept {
            label: "Somatics",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.7),
                (MIND, 0.6),
                (LIFE, 0.4),
                (COGNITION, 0.5),
                (ANATOMY, 0.3),
            ],
        },
        Concept {
            label: "Kinesiology for performers",
            category: "performing_arts",
            features: vec![
                (MOTION, 0.8),
                (LIFE, 0.6),
                (FORCE, 0.5),
                (PERFORMANCE, 0.5),
                (ANATOMY, 0.5),
            ],
        },
        // ── NANOTECHNOLOGY (25) ──
        Concept {
            label: "Nanomaterials",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.9),
                (STRUCTURE, 0.8),
                (CHEMISTRY, 0.6),
                (QUANTUM, 0.4),
                (NANO, 0.9),
                (ATOMIC, 0.4),
            ],
        },
        Concept {
            label: "Nanomedicine",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.7),
                (LIFE, 0.7),
                (DIAGNOSTICS, 0.5),
                (CHEMISTRY, 0.5),
                (NANO, 0.7),
                (THERAPY, 0.3),
            ],
        },
        Concept {
            label: "Quantum dots",
            category: "nanotechnology",
            features: vec![
                (QUANTUM, 0.8),
                (MATERIAL, 0.7),
                (WAVE, 0.5),
                (ENERGY, 0.4),
                (NANO, 0.6),
                (ATOMIC, 0.4),
            ],
        },
        Concept {
            label: "Nanoelectronics",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.7),
                (ENERGY, 0.6),
                (COMPUTATION, 0.5),
                (QUANTUM, 0.5),
                (NANO, 0.7),
                (ELECTRICAL, 0.5),
            ],
        },
        Concept {
            label: "Carbon nanotubes",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.9),
                (STRUCTURE, 0.8),
                (CHEMISTRY, 0.5),
                (ENERGY, 0.4),
                (NANO, 0.8),
                (ATOMIC, 0.3),
            ],
        },
        Concept {
            label: "Self-assembly",
            category: "nanotechnology",
            features: vec![
                (STRUCTURE, 0.8),
                (PATTERN, 0.7),
                (CHEMISTRY, 0.5),
                (SYSTEMS, 0.3),
                (NANO, 0.5),
                (MOLECULAR, 0.5),
            ],
        },
        Concept {
            label: "Molecular machines",
            category: "nanotechnology",
            features: vec![
                (STRUCTURE, 0.7),
                (MOTION, 0.6),
                (CHEMISTRY, 0.6),
                (ENERGY, 0.4),
                (NANO, 0.5),
                (MOLECULAR, 0.7),
            ],
        },
        Concept {
            label: "Nanooptics",
            category: "nanotechnology",
            features: vec![
                (WAVE, 0.7),
                (MATERIAL, 0.6),
                (QUANTUM, 0.5),
                (VISUAL, 0.3),
                (NANO, 0.6),
            ],
        },
        Concept {
            label: "Nanofabrication",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.8),
                (STRUCTURE, 0.7),
                (MEASUREMENT, 0.5),
                (CHEMISTRY, 0.4),
                (NANO, 0.7),
                (SURFACE, 0.3),
            ],
        },
        Concept {
            label: "Nanotoxicology",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.6),
                (LIFE, 0.6),
                (CHEMISTRY, 0.5),
                (DIAGNOSTICS, 0.4),
                (NANO, 0.5),
            ],
        },
        Concept {
            label: "Nanocomposites",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.9),
                (STRUCTURE, 0.7),
                (CHEMISTRY, 0.4),
                (FORCE, 0.3),
                (NANO, 0.6),
            ],
        },
        Concept {
            label: "DNA nanotechnology",
            category: "nanotechnology",
            features: vec![
                (GENETICS, 0.6),
                (MATERIAL, 0.6),
                (STRUCTURE, 0.7),
                (INFORMATION, 0.4),
                (NANO, 0.6),
                (MOLECULAR, 0.5),
            ],
        },
        Concept {
            label: "Nanophotonics",
            category: "nanotechnology",
            features: vec![
                (WAVE, 0.8),
                (MATERIAL, 0.6),
                (ENERGY, 0.5),
                (QUANTUM, 0.5),
                (NANO, 0.6),
            ],
        },
        Concept {
            label: "Nanosensors",
            category: "nanotechnology",
            features: vec![
                (MEASUREMENT, 0.8),
                (MATERIAL, 0.6),
                (DIAGNOSTICS, 0.5),
                (CHEMISTRY, 0.4),
                (NANO, 0.6),
                (SURFACE, 0.3),
            ],
        },
        Concept {
            label: "Nano-catalysis",
            category: "nanotechnology",
            features: vec![
                (CHEMISTRY, 0.7),
                (MATERIAL, 0.6),
                (ENERGY, 0.5),
                (OPTIMIZATION, 0.3),
                (NANO, 0.5),
                (REACTION, 0.4),
                (SURFACE, 0.4),
            ],
        },
        Concept {
            label: "Graphene research",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.9),
                (STRUCTURE, 0.7),
                (ENERGY, 0.5),
                (QUANTUM, 0.4),
                (NANO, 0.6),
                (ATOMIC, 0.5),
            ],
        },
        Concept {
            label: "Nanolithography",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.7),
                (PATTERN, 0.6),
                (MEASUREMENT, 0.6),
                (STRUCTURE, 0.5),
                (NANO, 0.6),
                (SURFACE, 0.3),
            ],
        },
        Concept {
            label: "Nanofluidics",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.6),
                (MOTION, 0.5),
                (FORCE, 0.4),
                (MEASUREMENT, 0.4),
                (NANO, 0.5),
                (SURFACE, 0.3),
            ],
        },
        Concept {
            label: "Nano-biointerfaces",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.6),
                (LIFE, 0.6),
                (CHEMISTRY, 0.5),
                (STRUCTURE, 0.4),
                (NANO, 0.5),
                (SURFACE, 0.4),
                (CELLULAR, 0.3),
            ],
        },
        Concept {
            label: "Nanoparticle synthesis",
            category: "nanotechnology",
            features: vec![
                (CHEMISTRY, 0.7),
                (MATERIAL, 0.7),
                (STRUCTURE, 0.5),
                (MEASUREMENT, 0.3),
                (NANO, 0.6),
                (REACTION, 0.3),
            ],
        },
        Concept {
            label: "Scanning probe microscopy",
            category: "nanotechnology",
            features: vec![
                (MEASUREMENT, 0.9),
                (MATERIAL, 0.5),
                (STRUCTURE, 0.5),
                (FORCE, 0.3),
                (NANO, 0.5),
                (SURFACE, 0.5),
                (ATOMIC, 0.4),
            ],
        },
        Concept {
            label: "Nanoethics",
            category: "nanotechnology",
            features: vec![
                (ETHICS, 0.7),
                (MATERIAL, 0.4),
                (LIFE, 0.3),
                (GOVERNANCE, 0.3),
                (NANO, 0.4),
                (MORAL, 0.3),
            ],
        },
        Concept {
            label: "Thin film technology",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.8),
                (STRUCTURE, 0.6),
                (CHEMISTRY, 0.5),
                (ENERGY, 0.3),
                (NANO, 0.4),
                (SURFACE, 0.5),
            ],
        },
        Concept {
            label: "Polymer nanostructures",
            category: "nanotechnology",
            features: vec![
                (MATERIAL, 0.8),
                (STRUCTURE, 0.7),
                (CHEMISTRY, 0.6),
                (PATTERN, 0.4),
                (NANO, 0.6),
                (MOLECULAR, 0.4),
            ],
        },
        Concept {
            label: "Nanoscale energy harvesting",
            category: "nanotechnology",
            features: vec![
                (ENERGY, 0.7),
                (MATERIAL, 0.6),
                (QUANTUM, 0.4),
                (MEASUREMENT, 0.4),
                (NANO, 0.5),
                (ELECTRICAL, 0.3),
            ],
        },
    ]
}
fn main() {
    println!("================================================================");
    println!("  SphereQL: AI Knowledge Navigator");
    println!("  Category Enrichment Layer — Full Capability Demo");
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

    let pipeline = SphereQLPipeline::new(PipelineInput {
        categories: categories.clone(),
        embeddings: embeddings.clone(),
    })
    .expect("pipeline build failed");

    let evr = pipeline.explained_variance_ratio();
    println!(
        "Corpus: {} concepts across {} knowledge domains",
        n,
        pipeline.num_categories()
    );
    println!(
        "Projection quality: {:.1}% variance explained (EVR={:.4})\n",
        evr * 100.0,
        evr
    );

    let layer = pipeline.category_layer();

    // ==================================================================
    // ANALYSIS 1: Category Landscape
    // ==================================================================
    println!("────────────────────────────────────────────────────────────────");
    println!("  1. CATEGORY LANDSCAPE");
    println!("     Cohesion, spread, and centroid positions for every domain");
    println!("────────────────────────────────────────────────────────────────\n");

    println!(
        "  {:<22} {:>5} {:>10} {:>8} {:>9} {:>9}",
        "Domain", "Items", "Spread(°)", "Cohesion", "θ (°)", "φ (°)"
    );
    println!("  {}", "-".repeat(66));

    let mut sorted_summaries: Vec<&_> = layer.summaries.iter().collect();
    sorted_summaries.sort_by(|a, b| b.cohesion.partial_cmp(&a.cohesion).unwrap());

    for summary in &sorted_summaries {
        println!(
            "  {:<22} {:>5} {:>10.2} {:>8.4} {:>9.2} {:>9.2}",
            summary.name,
            summary.member_count,
            summary.angular_spread.to_degrees(),
            summary.cohesion,
            summary.centroid_position.theta.to_degrees(),
            summary.centroid_position.phi.to_degrees(),
        );
    }

    let most_cohesive = sorted_summaries[0];
    let least_cohesive = sorted_summaries[sorted_summaries.len() - 1];
    println!(
        "\n  -> Tightest cluster:  {} (cohesion {:.4}, spread {:.1}°)",
        most_cohesive.name,
        most_cohesive.cohesion,
        most_cohesive.angular_spread.to_degrees()
    );
    println!(
        "  -> Most diffuse:     {} (cohesion {:.4}, spread {:.1}°)",
        least_cohesive.name,
        least_cohesive.cohesion,
        least_cohesive.angular_spread.to_degrees()
    );
    println!("     An AI should express more uncertainty about diffuse domains —");
    println!("     their concepts are spread across the sphere, not tightly clustered.");

    // ==================================================================
    // ANALYSIS 2: Sphere Geometry — Centroid Map
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  2. SPHERE GEOMETRY — Centroid Distance Map");
    println!("     Angular distances between domain centroids reveal the");
    println!("     topology of knowledge on the sphere");
    println!("────────────────────────────────────────────────────────────────\n");

    // Show pairwise distances for a curated set of domains to keep output manageable
    let focus_domains = [
        "physics",
        "biology",
        "computer_science",
        "philosophy",
        "economics",
        "music",
        "medicine",
        "linguistics",
        "nanotechnology",
        "law",
    ];

    // Header
    print!("  {:>15}", "");
    for &d in &focus_domains {
        let short = &d[..d.len().min(6)];
        print!(" {:>6}", short);
    }
    println!();
    print!("  {:>15}", "");
    println!(" {}", "-".repeat(focus_domains.len() * 7));

    for &row in &focus_domains {
        let ri = match layer.name_to_index.get(row) {
            Some(&i) => i,
            None => continue,
        };
        print!("  {:>15} ", row);
        for &col in &focus_domains {
            let ci = match layer.name_to_index.get(col) {
                Some(&i) => i,
                None => {
                    print!("      -");
                    continue;
                }
            };
            if ri == ci {
                print!("     --");
            } else {
                let dist = sphereql::core::angular_distance(
                    &layer.summaries[ri].centroid_position,
                    &layer.summaries[ci].centroid_position,
                );
                print!(" {:>6.2}", dist.to_degrees());
            }
        }
        println!();
    }

    // Find the closest and most distant category pairs overall
    let num_cats = layer.summaries.len();
    let mut closest_pair = ("", "", f64::INFINITY);
    let mut farthest_pair = ("", "", 0.0f64);
    for i in 0..num_cats {
        for j in (i + 1)..num_cats {
            let d = sphereql::core::angular_distance(
                &layer.summaries[i].centroid_position,
                &layer.summaries[j].centroid_position,
            );
            if d < closest_pair.2 {
                closest_pair = (&layer.summaries[i].name, &layer.summaries[j].name, d);
            }
            if d > farthest_pair.2 {
                farthest_pair = (&layer.summaries[i].name, &layer.summaries[j].name, d);
            }
        }
    }
    println!(
        "\n  Closest pair:   {} <-> {} ({:.2}°)",
        closest_pair.0,
        closest_pair.1,
        closest_pair.2.to_degrees()
    );
    println!(
        "  Farthest pair:  {} <-> {} ({:.2}°)",
        farthest_pair.0,
        farthest_pair.1,
        farthest_pair.2.to_degrees()
    );
    println!(
        "  Sphere utilization: {:.1}° spread across {:.1}° max",
        farthest_pair.2.to_degrees(),
        180.0
    );

    // ==================================================================
    // ANALYSIS 3: Adjacency Graph with Edge Weight Decomposition
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  3. ADJACENCY GRAPH — Edge Weight Decomposition");
    println!("     Shows how bridges reduce effective distance between domains");
    println!("────────────────────────────────────────────────────────────────\n");

    println!(
        "  {:<16} -> {:<16} {:>8} {:>7} {:>8} {:>8} {:>8}",
        "Source", "Target", "Raw(°)", "Bridges", "MaxStr", "Weight", "Savings"
    );
    println!("  {}", "-".repeat(78));

    // Show the top 3 neighbors for each focus domain, with full edge decomposition
    for &domain in &focus_domains[..6] {
        let ci = match layer.name_to_index.get(domain) {
            Some(&i) => i,
            None => continue,
        };
        for edge in layer.graph.adjacency[ci].iter().take(3) {
            let target_name = &layer.summaries[edge.target].name;
            let raw_deg = edge.centroid_distance.to_degrees();
            let weight_deg = edge.weight.to_degrees();
            let savings_pct = if edge.centroid_distance > 0.0 {
                (1.0 - edge.weight / edge.centroid_distance) * 100.0
            } else {
                0.0
            };
            println!(
                "  {:<16} -> {:<16} {:>7.2}° {:>7} {:>8.3} {:>7.2}° {:>7.1}%",
                domain,
                target_name,
                raw_deg,
                edge.bridge_count,
                edge.max_bridge_strength,
                weight_deg,
                savings_pct,
            );
        }
        if domain != focus_domains[5] {
            println!("  {}", "·".repeat(78));
        }
    }

    println!("\n  The \"Savings\" column shows how much shorter the effective path");
    println!("  becomes when bridge concepts pull two domains together.");
    println!("  More bridges with higher strength = cheaper traversal for the AI.");

    // ==================================================================
    // ANALYSIS 4: Bridge Density Analysis
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  4. BRIDGE DENSITY ANALYSIS");
    println!("     Which domain pairs are most interconnected?");
    println!("────────────────────────────────────────────────────────────────\n");

    // Collect all bridge pair stats
    let mut bridge_pairs: Vec<(&str, &str, usize, f64)> = Vec::new();
    for (&(si, ti), bridges) in &layer.graph.bridges {
        if !bridges.is_empty() {
            let mean_str: f64 =
                bridges.iter().map(|b| b.bridge_strength).sum::<f64>() / bridges.len() as f64;
            bridge_pairs.push((
                &layer.summaries[si].name,
                &layer.summaries[ti].name,
                bridges.len(),
                mean_str,
            ));
        }
    }
    bridge_pairs.sort_by(|a, b| b.2.cmp(&a.2).then(b.3.partial_cmp(&a.3).unwrap()));

    println!("  Top 20 most-bridged domain pairs:\n");
    println!(
        "  {:<22} <-> {:<22} {:>7} {:>10}",
        "Domain A", "Domain B", "Bridges", "Mean Str."
    );
    println!("  {}", "-".repeat(65));
    for (a, b, count, mean) in bridge_pairs.iter().take(20) {
        println!("  {:<22} <-> {:<22} {:>7} {:>10.4}", a, b, count, mean);
    }

    // Per-category total bridge counts (outbound)
    println!("\n  Bridge counts per domain (total outbound):\n");
    let mut cat_bridge_totals: Vec<(&str, usize)> = layer
        .summaries
        .iter()
        .enumerate()
        .map(|(ci, s)| {
            let total: usize = layer.graph.adjacency[ci]
                .iter()
                .map(|e| e.bridge_count)
                .sum();
            (s.name.as_str(), total)
        })
        .collect();
    cat_bridge_totals.sort_by_key(|x| std::cmp::Reverse(x.1));

    for (name, total) in &cat_bridge_totals {
        let bar = "█".repeat((*total / 10).min(40));
        println!("  {:<22} {:>5} {}", name, total, bar);
    }

    println!("\n  Domains with many bridges are conceptual hubs — they share");
    println!("  vocabulary and ideas with many other fields. An AI should");
    println!("  route cross-domain queries through these hubs.");

    // ==================================================================
    // ANALYSIS 5: Bridge Concept Detection
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  5. BRIDGE CONCEPTS — Cross-Domain Connectors");
    println!("     Concepts that span two knowledge domains, with affinities");
    println!("────────────────────────────────────────────────────────────────\n");

    let bridge_queries: Vec<(&str, &str)> = vec![
        ("physics", "computer_science"),
        ("physics", "music"),
        ("physics", "chemistry"),
        ("biology", "computer_science"),
        ("biology", "philosophy"),
        ("biology", "medicine"),
        ("computer_science", "economics"),
        ("computer_science", "linguistics"),
        ("philosophy", "economics"),
        ("philosophy", "neuroscience"),
        ("music", "psychology"),
        ("medicine", "engineering"),
        ("nanotechnology", "medicine"),
        ("law", "philosophy"),
        ("data_science", "biology"),
    ];

    for (src, tgt) in &bridge_queries {
        let bridges = pipeline.bridge_items(src, tgt, 3);
        let rev_bridges = pipeline.bridge_items(tgt, src, 3);

        let all: Vec<_> = bridges.iter().chain(rev_bridges.iter()).take(3).collect();

        if all.is_empty() {
            println!("  {} <-> {}: (no bridges — conceptual gap)", src, tgt);
        } else {
            println!("  {} <-> {}:", src, tgt);
            for b in &all {
                let direction = if b.source_category == layer.name_to_index[*src] {
                    format!("{} -> {}", src, tgt)
                } else {
                    format!("{} -> {}", tgt, src)
                };
                println!(
                    "    \"{}\" [{}]  src_affinity={:.3}  tgt_affinity={:.3}  strength={:.3}",
                    labels[b.item_index],
                    direction,
                    b.affinity_to_source,
                    b.affinity_to_target,
                    b.bridge_strength
                );
            }
        }
    }

    println!("\n  Bridge strength is the harmonic mean of source/target affinities.");
    println!("  High strength means the concept is equally relevant to both domains —");
    println!("  it's a genuine conceptual connector, not just noise.");

    // ==================================================================
    // ANALYSIS 6: Category Boundary Analysis
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  6. CATEGORY BOUNDARY ANALYSIS — Domain Ambassadors");
    println!("     Items that live near the border between two domains");
    println!("────────────────────────────────────────────────────────────────\n");

    // Find the top bridge items globally (strongest bridges across all pairs)
    let mut all_bridges: Vec<_> = layer
        .graph
        .bridges
        .values()
        .flat_map(|v| v.iter())
        .collect();
    all_bridges.sort_by(|a, b| b.bridge_strength.partial_cmp(&a.bridge_strength).unwrap());

    println!("  Top 15 boundary-straddling concepts (strongest bridges globally):\n");
    println!(
        "  {:<30} {:<18} {:<18} {:>8}",
        "Concept", "Home Domain", "Foreign Domain", "Strength"
    );
    println!("  {}", "-".repeat(78));
    let mut seen_items = std::collections::HashSet::new();
    let mut shown = 0;
    for b in &all_bridges {
        if shown >= 15 {
            break;
        }
        if !seen_items.insert(b.item_index) {
            continue;
        }
        println!(
            "  {:<30} {:<18} {:<18} {:>8.4}",
            labels[b.item_index],
            layer.summaries[b.source_category].name,
            layer.summaries[b.target_category].name,
            b.bridge_strength,
        );
        shown += 1;
    }

    println!("\n  These \"ambassador\" concepts are the most valuable for cross-domain");
    println!("  reasoning. When an AI needs to connect two distant fields, it should");
    println!("  look for items with high bridge strength as natural transition points.");

    // ==================================================================
    // ANALYSIS 7: Cross-Domain Concept Paths (Category-Level)
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  7. CONCEPT PATH TRAVERSAL — Category-Level Reasoning Chains");
    println!("     Shortest paths through the category graph (Dijkstra)");
    println!("────────────────────────────────────────────────────────────────");

    let path_queries: Vec<(&str, &str, &str)> = vec![
        (
            "nanotechnology",
            "economics",
            "How does nanotechnology impact economic systems?",
        ),
        (
            "linguistics",
            "medicine",
            "How does language connect to health?",
        ),
        (
            "philosophy",
            "computer_science",
            "From abstract thought to computation",
        ),
        (
            "music",
            "biology",
            "What connects musical patterns to living systems?",
        ),
        (
            "culinary_arts",
            "physics",
            "From the kitchen to the laws of nature",
        ),
        (
            "religion",
            "data_science",
            "From spiritual traditions to data analysis",
        ),
    ];

    for (src, tgt, question) in &path_queries {
        println!("\n  Q: \"{}\"", question);
        println!("  Path: {} -> {}\n", src, tgt);

        if let Some(path) = pipeline.category_path(src, tgt) {
            for (i, step) in path.steps.iter().enumerate() {
                let is_last = i + 1 >= path.steps.len();
                println!(
                    "    [{}] {} (cumulative: {:.3})",
                    i + 1,
                    step.category_name,
                    step.cumulative_distance,
                );

                if !is_last {
                    let bridge_descs: Vec<String> = step
                        .bridges_to_next
                        .iter()
                        .take(2)
                        .map(|b| {
                            format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength)
                        })
                        .collect();
                    if !bridge_descs.is_empty() {
                        println!("         bridged by: {}", bridge_descs.join(", "));
                    } else {
                        println!("         (direct adjacency)");
                    }
                    println!("         |");
                }
            }
            println!(
                "    Total distance: {:.3} ({:.1}° — {})\n",
                path.total_distance,
                path.total_distance.to_degrees(),
                if path.total_distance < 0.5 {
                    "very close"
                } else if path.total_distance < 1.0 {
                    "close"
                } else if path.total_distance < 2.0 {
                    "moderate"
                } else {
                    "distant"
                }
            );
        } else {
            println!("    (no path found)\n");
        }
    }

    // ==================================================================
    // ANALYSIS 8: Item-Level Concept Paths
    // ==================================================================
    println!("────────────────────────────────────────────────────────────────");
    println!("  8. ITEM-LEVEL CONCEPT PATHS — k-NN Graph Traversal");
    println!("     Tracing concept-to-concept paths through semantic space");
    println!("────────────────────────────────────────────────────────────────");

    let item_paths: Vec<(usize, usize, &str)> = vec![
        (10, 125, "Acoustics (physics) -> Harmonic theory (music)"),
        (
            24,
            70,
            "Quantum information (physics) -> Quantum computing (CS)",
        ),
        (
            30,
            100,
            "Bioinformatics (biology) -> Microeconomics (economics)",
        ),
    ];

    let dummy_q = PipelineQuery {
        embedding: vec![0.0; DIM],
    };

    for (src_idx, tgt_idx, desc) in &item_paths {
        let src_id = format!("s-{:04}", src_idx);
        let tgt_id = format!("s-{:04}", tgt_idx);
        println!("\n  {}", desc);
        println!("  {} -> {}\n", src_id, tgt_id);

        let result = pipeline.query(
            SphereQLQuery::ConceptPath {
                source_id: &src_id,
                target_id: &tgt_id,
                graph_k: 8,
            },
            &dummy_q,
        );

        if let SphereQLOutput::ConceptPath(Some(path)) = result {
            for (i, step) in path.steps.iter().enumerate() {
                let item_idx: usize = step.id.strip_prefix("s-").unwrap().parse().unwrap();
                let hop_str = if step.hop_distance > 0.0 {
                    format!(" hop={:.4}", step.hop_distance)
                } else {
                    String::new()
                };
                println!(
                    "    [{}] \"{}\" [{}]{} (cum={:.4})",
                    i + 1,
                    labels[item_idx],
                    step.category,
                    hop_str,
                    step.cumulative_distance,
                );
            }
            println!(
                "    Total: {:.4} ({} hops)",
                path.total_distance,
                path.steps.len() - 1
            );
        } else {
            println!("    (no path found)");
        }
    }

    println!("\n  Item-level paths show the SPECIFIC chain of intermediate concepts");
    println!("  that connect two ideas through semantic space. Each hop is a");
    println!("  k-nearest-neighbor link in the projected sphere.");

    // ==================================================================
    // ANALYSIS 9: Knowledge Density — Glob Detection
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  9. KNOWLEDGE DENSITY — Glob Detection");
    println!("     Where is knowledge concentrated? Where are the gaps?");
    println!("────────────────────────────────────────────────────────────────\n");

    let glob_result = pipeline.query(SphereQLQuery::DetectGlobs { k: None, max_k: 10 }, &dummy_q);

    if let SphereQLOutput::Globs(globs) = glob_result {
        println!(
            "  Detected {} knowledge clusters on the sphere:\n",
            globs.len()
        );
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
                "  Glob {}: {} members, radius={:.3} rad ({:.1}°), density={:.1}",
                g.id,
                g.member_count,
                g.radius,
                g.radius.to_degrees(),
                density,
            );
            println!("    Domains: {}", cats.join(", "));
        }

        // Identify pure vs mixed globs
        let pure_globs = globs
            .iter()
            .filter(|g| {
                g.top_categories.len() == 1
                    || g.top_categories[0].1 as f64 / g.member_count as f64 > 0.8
            })
            .count();
        let mixed_globs = globs.len() - pure_globs;
        println!(
            "\n  {} pure-domain clusters, {} mixed-domain clusters",
            pure_globs, mixed_globs
        );
        println!("  Mixed clusters are where interdisciplinary research lives.");
        println!("  Pure clusters indicate well-separated fields.");
    }

    // ==================================================================
    // ANALYSIS 10: Multi-Query Category Routing
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  10. MULTI-QUERY CATEGORY ROUTING");
    println!("      How the category layer dispatches diverse queries");
    println!("────────────────────────────────────────────────────────────────\n");

    let test_queries: Vec<(&str, Vec<(usize, f64)>)> = vec![
        (
            "neural networks and consciousness",
            vec![
                (NEURAL, 0.8),
                (CONSCIOUSNESS, 0.7),
                (COMPUTATION, 0.5),
                (MIND, 0.6),
                (AI, 0.4),
            ],
        ),
        (
            "climate change policy",
            vec![
                (CLIMATE, 0.9),
                (POLICY, 0.7),
                (ECOSYSTEM, 0.5),
                (GOVERNANCE, 0.4),
                (CONSERVATION, 0.5),
            ],
        ),
        (
            "music and mathematical patterns",
            vec![
                (SOUND, 0.6),
                (MATH, 0.7),
                (PATTERN, 0.8),
                (HARMONY, 0.5),
                (WAVE, 0.4),
            ],
        ),
        (
            "legal ethics of AI",
            vec![
                (LEGAL, 0.7),
                (ETHICS, 0.8),
                (AI, 0.6),
                (RIGHTS, 0.5),
                (COMPUTATION, 0.3),
                (MORAL, 0.4),
            ],
        ),
        (
            "genetic engineering in agriculture",
            vec![
                (GENETICS, 0.8),
                (LIFE, 0.6),
                (CHEMISTRY, 0.4),
                (ECOSYSTEM, 0.5),
                (NATURE, 0.3),
                (MOLECULAR, 0.4),
            ],
        ),
        (
            "theatrical storytelling in film",
            vec![
                (THEATRICAL, 0.7),
                (NARRATIVE, 0.8),
                (CINEMA, 0.7),
                (EMOTION, 0.5),
                (VISUAL, 0.4),
            ],
        ),
    ];

    for (query_desc, features) in &test_queries {
        let qvec = embed(features, 42 + query_desc.len() as u64);
        let emb = sphereql::embed::Embedding::new(qvec.clone());
        let nearby = layer.categories_near_embedding(&emb, pipeline.pca(), std::f64::consts::PI);

        println!("  Query: \"{}\"", query_desc);
        print!("    Top categories: ");
        let top: Vec<String> = nearby
            .iter()
            .take(5)
            .map(|(ci, dist)| format!("{} ({:.1}°)", layer.summaries[*ci].name, dist.to_degrees()))
            .collect();
        println!("{}", top.join(", "));

        // Also show which category the query is CLOSEST to and how confident we are
        if let Some(&(best_ci, best_dist)) = nearby.first() {
            let best_cat = &layer.summaries[best_ci];
            let in_spread = best_dist <= best_cat.angular_spread;
            println!(
                "    -> Primary: {} ({}within category spread)",
                best_cat.name,
                if in_spread { "" } else { "outside " }
            );
        }
        println!();
    }

    println!("  An AI uses category routing to decide which domain's knowledge");
    println!("  to activate. Queries near multiple centroids trigger cross-domain");
    println!("  reasoning; queries far from all centroids signal knowledge gaps.");

    // ==================================================================
    // ANALYSIS 11: Drill-Down with Inner Sphere
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  11. DRILL-DOWN — Inner Sphere Precision");
    println!("      Zooming into categories for fine-grained retrieval");
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

    println!("  Query: \"quantum computing\" (cross-domain concept)\n");

    // Show distances to relevant category centroids
    let emb = sphereql::embed::Embedding::new(quantum_computing.clone());
    let nearby = layer.categories_near_embedding(&emb, pipeline.pca(), std::f64::consts::PI);
    println!("  Nearest domain centroids:");
    for (ci, dist) in nearby.iter().take(6) {
        let has_inner = if layer.inner_spheres.contains_key(ci) {
            " [inner sphere]"
        } else {
            ""
        };
        println!(
            "    {:<22} {:.3} rad ({:.1}°){}",
            layer.summaries[*ci].name,
            dist,
            dist.to_degrees(),
            has_inner,
        );
    }

    // Drill down into several categories
    let drill_targets = ["physics", "computer_science", "mathematics", "philosophy"];
    for &cat in &drill_targets {
        println!("\n  Drill-down into {} (top 5):", cat.to_uppercase());
        let result = pipeline.query(
            SphereQLQuery::DrillDown {
                category: cat,
                k: 5,
            },
            &qc_query,
        );
        if let SphereQLOutput::DrillDown(results) = result {
            if results.is_empty() {
                println!("    (category not found or empty)");
                continue;
            }
            for (i, r) in results.iter().enumerate() {
                let sphere_tag = if r.used_inner_sphere {
                    "inner"
                } else {
                    "outer"
                };
                println!(
                    "    {}. \"{}\" (dist={:.4}, {} sphere)",
                    i + 1,
                    labels[r.item_index],
                    r.distance,
                    sphere_tag,
                );
            }
        }
    }

    println!("\n  When an inner sphere exists, drill-down uses a category-specific");
    println!("  projection that captures more within-category variance than the");
    println!("  global projection. This gives finer angular discrimination.");

    // ==================================================================
    // ANALYSIS 12: Nearest Neighbor with Projection Metadata
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  12. NEAREST NEIGHBOR — Projection Quality Signals");
    println!("      Certainty and intensity metadata on search results");
    println!("────────────────────────────────────────────────────────────────\n");

    // Run a nearest-neighbor query and show certainty/intensity
    println!("  Query: \"quantum computing\" — 10 nearest neighbors:\n");
    let nn_result = pipeline.query(SphereQLQuery::Nearest { k: 10 }, &qc_query);
    if let SphereQLOutput::Nearest(results) = nn_result {
        println!(
            "  {:<4} {:<30} {:<18} {:>8} {:>9} {:>9}",
            "#", "Concept", "Domain", "Dist(°)", "Certainty", "Intensity"
        );
        println!("  {}", "-".repeat(82));
        for (i, r) in results.iter().enumerate() {
            let idx: usize = r.id.strip_prefix("s-").unwrap().parse().unwrap();
            println!(
                "  {:<4} {:<30} {:<18} {:>8.2} {:>9.4} {:>9.4}",
                i + 1,
                labels[idx],
                r.category,
                r.distance.to_degrees(),
                r.certainty,
                r.intensity,
            );
        }
        println!("\n  Certainty: how faithfully the 3D projection represents the high-D");
        println!("  embedding. Low certainty = the concept lost structure in projection.");
        println!("  Intensity: pre-normalization magnitude — strong signals vs. weak ones.");
    }

    // Also run a cosine similarity threshold query
    println!("\n  Cosine similarity threshold query (min_cosine=0.85):\n");
    let sim_result = pipeline.query(SphereQLQuery::SimilarAbove { min_cosine: 0.85 }, &qc_query);
    if let SphereQLOutput::KNearest(results) = sim_result {
        println!("  Found {} concepts above threshold:", results.len());
        for r in results.iter().take(8) {
            let idx: usize = r.id.strip_prefix("s-").unwrap().parse().unwrap();
            println!(
                "    \"{}\" [{}] dist={:.4} certainty={:.3}",
                labels[idx], r.category, r.distance, r.certainty
            );
        }
        if results.len() > 8 {
            println!("    ... and {} more", results.len() - 8);
        }
    }

    // ==================================================================
    // ANALYSIS 13: Assembled Reasoning Chain
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  13. ASSEMBLED REASONING — Full AI Workflow");
    println!("      Simulating how an AI answers a cross-domain question");
    println!("      using every layer of the category enrichment system");
    println!("────────────────────────────────────────────────────────────────\n");

    let question = "How does music relate to economics?";
    println!("  USER QUESTION: \"{}\"\n", question);
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  AI's spatial reasoning process                        │");
    println!("  └─────────────────────────────────────────────────────────┘\n");

    // Step 1: Route the query to relevant categories
    println!("  STEP 1: Category routing — which domains are relevant?\n");
    let music_econ_q = embed(
        &[
            (SOUND, 0.5),
            (MARKETS, 0.5),
            (PATTERN, 0.4),
            (BEHAVIOR, 0.3),
            (PERFORMANCE, 0.3),
        ],
        7777,
    );
    let me_emb = sphereql::embed::Embedding::new(music_econ_q.clone());
    let me_nearby = layer.categories_near_embedding(&me_emb, pipeline.pca(), std::f64::consts::PI);
    for (ci, dist) in me_nearby.iter().take(5) {
        println!(
            "    {:<22} {:.2}°",
            layer.summaries[*ci].name,
            dist.to_degrees()
        );
    }

    // Step 2: Find the path
    println!("\n  STEP 2: Category path — music -> economics\n");
    if let Some(path) = pipeline.category_path("music", "economics") {
        let domain_chain: Vec<&str> = path
            .steps
            .iter()
            .map(|s| s.category_name.as_str())
            .collect();
        println!("    Route: {}", domain_chain.join(" → "));
        println!(
            "    Semantic distance: {:.3} ({:.1}°)\n",
            path.total_distance,
            path.total_distance.to_degrees()
        );

        // Step 3: Gather bridge concepts along each edge
        println!("  STEP 3: Bridge concepts at each transition\n");
        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 3);
                let rev_bridges =
                    pipeline.bridge_items(&next.category_name, &step.category_name, 3);
                let all_labels: Vec<String> = bridges
                    .iter()
                    .chain(rev_bridges.iter())
                    .take(3)
                    .map(|b| format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength))
                    .collect();
                if all_labels.is_empty() {
                    println!(
                        "    {} → {}: (direct adjacency)",
                        step.category_name, next.category_name
                    );
                } else {
                    println!(
                        "    {} → {}: {}",
                        step.category_name,
                        next.category_name,
                        all_labels.join(", ")
                    );
                }
            }
        }

        // Step 4: Check cohesion to calibrate confidence
        println!("\n  STEP 4: Confidence calibration via domain cohesion\n");
        for step in &path.steps {
            if let Some(summary) = layer.get_category(&step.category_name) {
                let confidence = if summary.cohesion > 0.8 {
                    "HIGH  ██████████"
                } else if summary.cohesion > 0.7 {
                    "MED   ██████░░░░"
                } else if summary.cohesion > 0.6 {
                    "LOW   ████░░░░░░"
                } else {
                    "VLOW  ██░░░░░░░░"
                };
                println!(
                    "    {:<22} cohesion={:.3}  {}",
                    step.category_name, summary.cohesion, confidence
                );
            }
        }

        // Step 5: Drill into endpoints for supporting concepts
        println!("\n  STEP 5: Drill-down into endpoint domains for evidence\n");
        let me_query = PipelineQuery {
            embedding: music_econ_q.clone(),
        };
        for &domain in &["music", "economics"] {
            println!("    {} — top 3:", domain.to_uppercase());
            let drill = pipeline.query(
                SphereQLQuery::DrillDown {
                    category: domain,
                    k: 3,
                },
                &me_query,
            );
            if let SphereQLOutput::DrillDown(results) = drill {
                for (i, r) in results.iter().enumerate() {
                    println!(
                        "      {}. \"{}\" (dist={:.4})",
                        i + 1,
                        labels[r.item_index],
                        r.distance
                    );
                }
            }
            println!();
        }

        // Step 6: Synthesize a narrative
        println!("  STEP 6: Synthesized answer\n");
        println!("    ┌──────────────────────────────────────────────────────┐");
        println!("    │ \"Music and economics, while seemingly distant,     │");
        println!("    │  connect through shared conceptual foundations:     │");
        println!("    │                                                    │");

        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 1);
                let rev_bridges =
                    pipeline.bridge_items(&next.category_name, &step.category_name, 1);
                let bridge_label = bridges
                    .first()
                    .or(rev_bridges.first())
                    .map(|b| labels[b.item_index])
                    .unwrap_or("shared foundations");
                println!(
                    "    │  • {} → {} via {}",
                    step.category_name, next.category_name, bridge_label
                );
                println!("    │    {:>52}│", "");
            }
        }

        let closeness = if path.total_distance < 1.0 {
            "surprisingly close"
        } else if path.total_distance < 2.0 {
            "moderately connected"
        } else {
            "quite distant"
        };
        println!(
            "    │  Distance {:.3} / π — fields are {}.  │",
            path.total_distance, closeness
        );
        println!("    └──────────────────────────────────────────────────────┘");
    }

    // ── Inner sphere stats ────────────────────────────────────────────
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  APPENDIX: Inner Sphere Status");
    println!("────────────────────────────────────────────────────────────────\n");

    let stats = pipeline.inner_sphere_stats();
    if stats.is_empty() {
        println!("  No inner spheres materialized.");
        println!(
            "  (Categories need ≥{} members AND ≥{:.0}% EVR improvement",
            20, 10.0
        );
        println!("  over the global projection to qualify.)\n");
        println!("  Current category sizes:");
        let mut sizes: Vec<_> = layer
            .summaries
            .iter()
            .map(|s| (&s.name, s.member_count))
            .collect();
        sizes.sort_by_key(|x| std::cmp::Reverse(x.1));
        for (name, count) in sizes.iter().take(10) {
            let bar = "█".repeat(*count);
            let threshold_marker = if *count >= 20 { " ✓ (eligible)" } else { "" };
            println!("    {:<22} {:>3} {}{}", name, count, bar, threshold_marker);
        }
        if sizes.len() > 10 {
            println!("    ... and {} more categories", sizes.len() - 10);
        }
        println!("\n  With a real corpus (50+ items per domain), inner spheres would");
        println!("  automatically activate and provide finer within-category angular");
        println!("  discrimination than the global projection.");
    } else {
        println!(
            "  {} of {} categories have inner spheres:\n",
            stats.len(),
            layer.num_categories()
        );
        println!(
            "  {:<22} {:>6} {:>10} {:>10} {:>10} {:>12}",
            "Domain", "Items", "Projection", "Inner EVR", "Global EVR", "Improvement"
        );
        println!("  {}", "-".repeat(74));
        for s in &stats {
            println!(
                "  {:<22} {:>6} {:>10} {:>10.4} {:>10.4} {:>11.4}",
                s.category_name,
                s.member_count,
                s.projection_type,
                s.inner_evr,
                s.global_subset_evr,
                s.evr_improvement,
            );
        }
    }

    println!("\n================================================================");
    println!(
        "  Demo complete. {} concepts, {} categories, EVR={:.1}%",
        n,
        pipeline.num_categories(),
        pipeline.explained_variance_ratio() * 100.0,
    );
    println!(
        "  Category layer: {} summaries, {} bridge pairs, {} inner spheres",
        layer.num_categories(),
        layer.graph.bridges.len(),
        layer.inner_spheres.len(),
    );
    println!("================================================================");
}
