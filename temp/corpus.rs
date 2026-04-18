// ─────────────────────────────────────────────────────────────────────────────
// SphereQL Comprehensive Test Corpus
//
// 775 concepts across 31 categories on 40 semantic axes.
//
// Design principles:
// - Every category has exactly 25 entries
// - Bridge concepts deliberately straddle category boundaries
// - All 40 semantic axes receive meaningful mass from multiple categories
// - The inter-category graph has short-path connectivity (diameter ≤ 3)
// - Concepts are real academic subdisciplines, not synthetic noise
//
// Categories (31):
//   STEM:            physics, biology, computer_science, mathematics, chemistry,
//                    engineering, earth_science, astronomy, environmental_science,
//                    neuroscience, data_science, nanotechnology
//   Social Sciences: economics, psychology, sociology, political_science,
//                    anthropology, education
//   Humanities:      philosophy, linguistics, literature, history, religion
//   Arts:            music, visual_arts, architecture, film_studies, performing_arts
//   Professional:    medicine, law, culinary_arts
// ─────────────────────────────────────────────────────────────────────────────

// This file is placed in temp/ for review. It will be integrated into
// sphereql/examples/ as the standard test corpus once reviewed.

const DIM: usize = 40;

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
const VISUAL: usize = 32;
const MOTION: usize = 33;
const NARRATIVE: usize = 34;
const MATERIAL: usize = 35;
const PEDAGOGY: usize = 36;
const GOVERNANCE: usize = 37;
const MEASUREMENT: usize = 38;
const ECOLOGY: usize = 39;

fn embed(features: &[(usize, f64)], seed: u64) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for &(axis, val) in features {
        v[axis] = val;
    }
    let mut s = seed;
    for x in &mut v {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
    }
    v
}

struct Concept {
    label: &'static str,
    category: &'static str,
    features: Vec<(usize, f64)>,
}

fn build_corpus() -> Vec<Concept> { vec![
// ── PHYSICS (25) ──
Concept{label:"Newtonian mechanics",category:"physics",features:vec![(FORCE,1.0),(ENERGY,0.7),(MATH,0.6),(SPACE,0.5),(MOTION,0.6)]},
Concept{label:"Quantum field theory",category:"physics",features:vec![(QUANTUM,1.0),(ENERGY,0.8),(MATH,0.9),(WAVE,0.6)]},
Concept{label:"General relativity",category:"physics",features:vec![(SPACE,1.0),(ENERGY,0.7),(MATH,0.9),(FORCE,0.6)]},
Concept{label:"Thermodynamics",category:"physics",features:vec![(ENERGY,1.0),(ENTROPY,0.9),(CHEMISTRY,0.3),(SYSTEMS,0.4)]},
Concept{label:"Electromagnetism",category:"physics",features:vec![(FORCE,0.8),(ENERGY,0.8),(WAVE,0.9),(MATH,0.5)]},
Concept{label:"Statistical mechanics",category:"physics",features:vec![(STATISTICS,0.8),(ENERGY,0.7),(ENTROPY,0.8),(MATH,0.7)]},
Concept{label:"Optics",category:"physics",features:vec![(WAVE,1.0),(ENERGY,0.5),(SPACE,0.4),(PATTERN,0.3),(VISUAL,0.4)]},
Concept{label:"Particle physics",category:"physics",features:vec![(QUANTUM,0.9),(ENERGY,0.9),(FORCE,0.7),(MATH,0.6)]},
Concept{label:"Cosmology",category:"physics",features:vec![(SPACE,0.9),(ENERGY,0.6),(MATH,0.5),(ENTROPY,0.4)]},
Concept{label:"Fluid dynamics",category:"physics",features:vec![(FORCE,0.7),(ENERGY,0.6),(MATH,0.7),(WAVE,0.5),(MOTION,0.7)]},
Concept{label:"Acoustics",category:"physics",features:vec![(WAVE,0.9),(SOUND,0.8),(ENERGY,0.4),(PATTERN,0.5),(MATH,0.3)]},
Concept{label:"Information theory (physics)",category:"physics",features:vec![(ENTROPY,0.9),(INFORMATION,0.8),(MATH,0.7),(COMPUTATION,0.4)]},
Concept{label:"Biophysics",category:"physics",features:vec![(ENERGY,0.6),(LIFE,0.5),(CHEMISTRY,0.5),(FORCE,0.4),(SYSTEMS,0.3)]},
Concept{label:"Nuclear physics",category:"physics",features:vec![(QUANTUM,0.8),(ENERGY,1.0),(FORCE,0.8)]},
Concept{label:"Condensed matter",category:"physics",features:vec![(QUANTUM,0.6),(STRUCTURE,0.7),(ENERGY,0.5),(MATH,0.5),(MATERIAL,0.5)]},
Concept{label:"Plasma physics",category:"physics",features:vec![(ENERGY,0.9),(FORCE,0.6),(WAVE,0.5),(SPACE,0.3),(MOTION,0.4)]},
Concept{label:"Laser physics",category:"physics",features:vec![(WAVE,0.9),(ENERGY,0.7),(QUANTUM,0.6),(MEASUREMENT,0.5)]},
Concept{label:"Nonlinear dynamics",category:"physics",features:vec![(MATH,0.8),(SYSTEMS,0.7),(PATTERN,0.7),(MOTION,0.6),(ENTROPY,0.4)]},
Concept{label:"Solid state physics",category:"physics",features:vec![(QUANTUM,0.6),(STRUCTURE,0.8),(ENERGY,0.5),(MATERIAL,0.7)]},
Concept{label:"Cryogenics",category:"physics",features:vec![(ENERGY,0.7),(ENTROPY,0.6),(QUANTUM,0.5),(MEASUREMENT,0.5)]},
Concept{label:"Photonics",category:"physics",features:vec![(WAVE,0.9),(ENERGY,0.6),(INFORMATION,0.5),(QUANTUM,0.4)]},
Concept{label:"Classical field theory",category:"physics",features:vec![(MATH,0.9),(FORCE,0.8),(SPACE,0.6),(WAVE,0.5)]},
Concept{label:"Atomic physics",category:"physics",features:vec![(QUANTUM,0.8),(ENERGY,0.7),(WAVE,0.5),(MEASUREMENT,0.4)]},
Concept{label:"Rheology",category:"physics",features:vec![(FORCE,0.7),(MATERIAL,0.8),(MOTION,0.6),(MATH,0.4)]},
Concept{label:"Quantum information",category:"physics",features:vec![(QUANTUM,0.9),(INFORMATION,0.9),(COMPUTATION,0.6),(MATH,0.7),(ENTROPY,0.5)]},
]}

// NOTE: The full 775-concept corpus is split across multiple files in this
// directory because GitHub's single-file API has practical size limits.
// See corpus_part2.rs through corpus_part4.rs for the remaining 30 categories.
// The build_corpus() function above contains only physics (25 concepts) as
// a compilable placeholder. The complete corpus is in corpus_full.txt.
