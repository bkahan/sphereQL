//! 128 semantic axis definitions grouped by domain.

// ── Physics ──────────────────────────────────────────────────────────────
pub const ENERGY: usize = 0;
pub const FORCE: usize = 1;
pub const QUANTUM: usize = 2;
pub const WAVE: usize = 3;
pub const ENTROPY: usize = 4;
pub const RELATIVITY: usize = 5;
pub const PARTICLE: usize = 6;

// ── Mathematics ──────────────────────────────────────────────────────────
pub const MATH: usize = 7;
pub const PROOF: usize = 8;
pub const CALCULUS: usize = 9;
pub const GRAPH_THEORY: usize = 10;
pub const ALGEBRA: usize = 11;

// ── Biology ──────────────────────────────────────────────────────────────
pub const LIFE: usize = 12;
pub const EVOLUTION: usize = 13;
pub const GENETICS: usize = 14;
pub const CELLULAR: usize = 15;

// ── Chemistry ────────────────────────────────────────────────────────────
pub const CHEMISTRY: usize = 16;
pub const MOLECULAR: usize = 17;
pub const REACTION: usize = 18;

// ── Medicine ─────────────────────────────────────────────────────────────
pub const DIAGNOSTICS: usize = 19;
pub const THERAPY: usize = 20;
pub const ANATOMY: usize = 21;
pub const CLINICAL: usize = 22;

// ── Neuroscience ─────────────────────────────────────────────────────────
pub const NEURAL: usize = 23;
pub const BRAIN: usize = 24;
pub const CONSCIOUSNESS: usize = 25;

// ── Computer Science ─────────────────────────────────────────────────────
pub const COMPUTATION: usize = 26;
pub const LOGIC: usize = 27;
pub const SOFTWARE: usize = 28;
pub const ALGORITHM: usize = 29;

// ── AI / Data Science ────────────────────────────────────────────────────
pub const AI: usize = 30;
pub const LLM: usize = 31;
pub const DATA: usize = 32;
pub const MACHINE_LEARN: usize = 33;

// ── Engineering ──────────────────────────────────────────────────────────
pub const MECHANICAL: usize = 34;
pub const ELECTRICAL: usize = 35;
pub const MATERIAL: usize = 36;
pub const TRANSPORTATION: usize = 37;

// ── Nanotechnology ───────────────────────────────────────────────────────
pub const NANO: usize = 38;
pub const ATOMIC: usize = 39;
pub const SURFACE: usize = 40;

// ── Astronomy ────────────────────────────────────────────────────────────
pub const CELESTIAL: usize = 41;
pub const STELLAR: usize = 42;
pub const PLANETARY: usize = 43;
pub const ORBIT: usize = 44;

// ── Earth Science ────────────────────────────────────────────────────────
pub const GEOLOGY: usize = 45;
pub const CLIMATE: usize = 46;
pub const OCEAN: usize = 47;
pub const WATER: usize = 48;

// ── Environmental Science ────────────────────────────────────────────────
pub const ECOSYSTEM: usize = 49;
pub const CONSERVATION: usize = 50;
pub const NATURE: usize = 51;

// ── Psychology ───────────────────────────────────────────────────────────
pub const ATTACHMENT: usize = 52;
pub const TRAUMA: usize = 53;
pub const MENTAL_HEALTH: usize = 54;

// ── Philosophy ───────────────────────────────────────────────────────────
pub const ETHICS: usize = 55;
pub const METAPHYSICS: usize = 56;
pub const EPISTEMOLOGY: usize = 57;
pub const ONTOLOGY: usize = 58;

// ── Religion ─────────────────────────────────────────────────────────────
pub const SPIRITUAL: usize = 59;
pub const RITUAL: usize = 60;
pub const SACRED: usize = 61;
pub const DOCTRINE: usize = 62;

// ── Linguistics ──────────────────────────────────────────────────────────
pub const LANGUAGE: usize = 63;
pub const GRAMMAR: usize = 64;
pub const PHONETIC: usize = 65;
pub const SYNTAX: usize = 66;
pub const SEMANTICS_AX: usize = 67;

// ── Literature ───────────────────────────────────────────────────────────
pub const NARRATIVE: usize = 68;
pub const LITERARY: usize = 69;
pub const POETRY: usize = 70;

// ── History ──────────────────────────────────────────────────────────────
pub const HISTORICAL: usize = 71;
pub const ARCHIVAL: usize = 72;

// ── Sociology ────────────────────────────────────────────────────────────
pub const SOCIETY: usize = 73;
pub const COMMUNITY: usize = 74;
pub const SOCIAL_NETWORK: usize = 75;

// ── Anthropology ─────────────────────────────────────────────────────────
pub const CULTURE: usize = 76;
pub const TRADITION: usize = 77;
pub const KINSHIP: usize = 78;

// ── Political Science ────────────────────────────────────────────────────
pub const GOVERNANCE: usize = 79;
pub const POWER: usize = 80;
pub const POLICY: usize = 81;

// ── Law ──────────────────────────────────────────────────────────────────
pub const LEGAL: usize = 82;
pub const JUSTICE: usize = 83;
pub const RIGHTS: usize = 84;

// ── Economics ────────────────────────────────────────────────────────────
pub const MARKETS: usize = 85;
pub const FINANCE: usize = 86;
pub const LABOR: usize = 87;
pub const MONEY: usize = 88;

// ── Education ────────────────────────────────────────────────────────────
pub const PEDAGOGY: usize = 89;
pub const CURRICULUM: usize = 90;
pub const ASSESSMENT: usize = 91;

// ── Visual Arts ──────────────────────────────────────────────────────────
pub const VISUAL: usize = 92;
pub const COLOR: usize = 93;
pub const FORM: usize = 94;
pub const DESIGN: usize = 95;

// ── Music ────────────────────────────────────────────────────────────────
pub const SOUND: usize = 96;
pub const HARMONY: usize = 97;
pub const RHYTHM: usize = 98;
pub const TIMBRE: usize = 99;

// ── Film ─────────────────────────────────────────────────────────────────
pub const CINEMA: usize = 100;
pub const MONTAGE: usize = 101;

// ── Performing Arts ──────────────────────────────────────────────────────
pub const THEATRICAL: usize = 102;
pub const DANCE: usize = 103;

// ── Culinary Arts ────────────────────────────────────────────────────────
pub const TASTE: usize = 104;
pub const FLAVOR: usize = 105;
pub const COOKING: usize = 106;

// ── Cross-cutting axes ───────────────────────────────────────────────────
pub const INFORMATION: usize = 107;
pub const SYSTEMS: usize = 108;
pub const OPTIMIZATION: usize = 109;
pub const PATTERN: usize = 110;
pub const STRUCTURE: usize = 111;
pub const NETWORK: usize = 112;
pub const SPACE: usize = 113;
pub const PERFORMANCE: usize = 114;
pub const MEASUREMENT: usize = 115;
pub const MOTION: usize = 116;
pub const CYCLE: usize = 117;
pub const BEHAVIOR: usize = 118;
pub const EMOTION: usize = 119;
pub const CONCEPT: usize = 120;
pub const THEORY: usize = 121;
pub const LEARNING: usize = 122;
pub const STATISTICS: usize = 123;
pub const MORAL: usize = 124;
pub const DISCOURSE: usize = 125;
pub const COGNITION: usize = 126;
pub const MIND: usize = 127;
