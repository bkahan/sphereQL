# AI Knowledge Navigator — Corpus

775 concepts across 31 categories on 40 semantic axes.

The full corpus is split across files for manageability:
- `corpus.rs` — Constants, embed(), Concept struct, physics (25 concepts) as compilable scaffold
- `corpus_full.txt` — Complete 775-concept corpus (all 31 categories)

## Categories (31)

| Domain | Categories |
|--------|------------|
| STEM | physics, biology, computer_science, mathematics, chemistry, engineering, earth_science, astronomy, environmental_science, neuroscience, data_science, nanotechnology |
| Social Sciences | economics, psychology, sociology, political_science, anthropology, education |
| Humanities | philosophy, linguistics, literature, history, religion |
| Arts | music, visual_arts, architecture, film_studies, performing_arts |
| Professional | medicine, law, culinary_arts |

## Semantic Axes (40)

Original 32: ENERGY, FORCE, MATH, QUANTUM, SPACE, LIFE, EVOLUTION, CHEMISTRY, NATURE, GENETICS, COMPUTATION, LOGIC, INFORMATION, SYSTEMS, ETHICS, MIND, METAPHYSICS, LANGUAGE, MARKETS, OPTIMIZATION, BEHAVIOR, SOUND, EMOTION, PATTERN, PERFORMANCE, DIAGNOSTICS, STATISTICS, COGNITION, STRUCTURE, ENTROPY, WAVE, NETWORK

New 8: VISUAL, MOTION, NARRATIVE, MATERIAL, PEDAGOGY, GOVERNANCE, MEASUREMENT, ECOLOGY

## Design Principles

- Every category has exactly 25 entries
- Bridge concepts deliberately straddle category boundaries
- All 40 semantic axes receive meaningful mass from multiple categories
- The inter-category graph has short-path connectivity (diameter ≤ 3)
- Concepts are real academic subdisciplines, not synthetic noise
