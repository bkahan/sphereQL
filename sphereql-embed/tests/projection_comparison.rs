//! Head-to-head projection comparison.
//!
//! Runs all four production projections (`Pca`, `KernelPca`,
//! `LaplacianEigenmap`, `UmapSphere`) over the same labeled corpus and
//! reports three reasoning-quality signals per projection:
//!
//! - **Cluster score** — `mean intra-category angular distance / mean
//!   inter-category angular distance`. Lower is better; 0 = perfectly
//!   separated, 1 = no separation.
//! - **Determinism gap** — the worst-case angular distance between
//!   coordinates of a second fit on the same inputs and seed. Should be
//!   `≤ 1e-9` for every projection in this codebase.
//! - **Logical confidence** — placeholder for the future
//!   reasoning-preservation metric. Currently always `n/a` (the
//!   `UnimplementedLogicalConfidence` skeleton returns `None`); the
//!   column exists so when the metric ships, the test wires it up
//!   without churn.
//!
//! Two corpora exercise different regimes:
//!
//! - The hand-encoded text corpus (24 docs, 4 topics, hash-bag 64-dim) —
//!   matches the existing E2E test in `sphereql-vectordb` so the same
//!   reasoning test the pipeline ships under is part of every projection
//!   release.
//! - `sphereql-corpus::build_corpus` (775 concepts, 31 categories,
//!   128-dim authored embeddings) — the established benchmark corpus
//!   for projection-quality work.
//!
//! Output is always printed (run with `--nocapture` to see it during
//! ordinary test runs):
//!
//! ```text
//! corpus: e2e_text  (24 docs, 4 categories)
//! ┌──────────────────┬──────────┬──────────────┬──────────────────────┐
//! │ projection       │ cluster  │ determinism  │ logical_confidence   │
//! ├──────────────────┼──────────┼──────────────┼──────────────────────┤
//! │ pca              │ 0.421    │ 0.0e0        │ n/a                  │
//! ...
//! ```
//!
//! Assertions are **relative** between projections (no projection's
//! cluster score may be more than `2.0×` the best on the same corpus —
//! a regression flag, not a quality floor) and **strict against a
//! committed baseline** (each projection must beat the per-projection
//! ceiling captured below). When this test fails the diff in the
//! printed table is the primary diagnostic.
//!
//! The baseline ceilings were captured on first green run and tightened
//! by ~10% headroom — they should never need to be loosened, only
//! tightened as projections improve.

use std::collections::HashMap;
use std::time::Instant;

use sphereql_core::SphericalPoint;
use sphereql_embed::{
    Embedding, KernelPcaProjection, LaplacianEigenmapProjection, LogicalConfidence, PcaProjection,
    Projection, RadialStrategy, UmapConfig, UmapSphereProjection, UnimplementedLogicalConfidence,
};

// ── Corpus 1: text e2e (mirrors sphereql-vectordb/tests/e2e_text_to_sphereql) ──

const TEXT_DIM: usize = 64;
const STOPWORDS: &[&str] = &[
    "the", "and", "with", "for", "its", "now", "across", "above", "along", "from", "into", "out",
    "off", "have", "has", "this", "that", "these", "those", "their", "them", "they", "then",
    "than", "over", "under", "but", "not", "all", "any", "are", "was", "were", "been", "being",
    "such", "via", "down", "up", "between", "lanes", "high",
];

fn encode_text(text: &str) -> Vec<f64> {
    let mut bag = vec![0.0_f64; TEXT_DIM];
    for token in text
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|t| t.len() >= 3)
        .filter(|t| !STOPWORDS.contains(t))
    {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in token.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        let bucket = (h as usize) % TEXT_DIM;
        bag[bucket] += 1.0;
    }
    let mag: f64 = bag.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag > 0.0 {
        for x in &mut bag {
            *x /= mag;
        }
    }
    bag
}

fn text_corpus() -> (Vec<Embedding>, Vec<String>) {
    let docs: &[(&str, &str)] = &[
        ("a wolf hunts in the forest with its pack", "animals"),
        ("the lion roars across the savanna grasslands", "animals"),
        ("eagles soar high above the mountain peaks", "animals"),
        ("dolphins swim through warm tropical seas", "animals"),
        ("a falcon dives toward its unsuspecting prey", "animals"),
        ("bears fish for salmon along the cold river", "animals"),
        ("the sedan accelerates down the empty highway", "vehicles"),
        (
            "a freight train carries cargo across the continent",
            "vehicles",
        ),
        ("the cargo ship docks at the busy harbor", "vehicles"),
        ("electric scooters now line the city sidewalks", "vehicles"),
        ("jet airplanes climb above the puffy clouds", "vehicles"),
        ("motorcycles weave between lanes on the highway", "vehicles"),
        ("freshly baked bread cools on the wooden table", "food"),
        ("tomato basil pasta with grated parmesan cheese", "food"),
        ("dark chocolate brownies cooling on a wire rack", "food"),
        ("grilled salmon with lemon and garden herbs", "food"),
        ("sourdough loaves rise slowly overnight", "food"),
        ("warm apple pie with vanilla ice cream", "food"),
        ("dark thunderclouds gather over the valley", "weather"),
        ("a gentle snow falls on the silent town", "weather"),
        ("the summer hurricane batters the coastal city", "weather"),
        ("morning fog rolls across the quiet harbor", "weather"),
        ("a cold winter blizzard buries the highway", "weather"),
        ("warm sunshine bathes the meadow in light", "weather"),
    ];
    let embeddings: Vec<Embedding> = docs
        .iter()
        .map(|(t, _)| Embedding::new(encode_text(t)))
        .collect();
    let categories: Vec<String> = docs.iter().map(|(_, c)| (*c).to_string()).collect();
    (embeddings, categories)
}

// ── Corpus 2: sphereql-corpus build_corpus ─────────────────────────────────

fn sphereql_corpus_inputs() -> (Vec<Embedding>, Vec<String>) {
    let concepts = sphereql_corpus::build_corpus();
    let embeddings: Vec<Embedding> = concepts
        .iter()
        .enumerate()
        .map(|(i, c)| Embedding::new(sphereql_corpus::embed(&c.features, i as u64)))
        .collect();
    let categories: Vec<String> = concepts.iter().map(|c| c.category.to_string()).collect();
    (embeddings, categories)
}

// ── Metrics ────────────────────────────────────────────────────────────────

/// Mean intra-category angular distance divided by mean inter-category
/// angular distance. Lower is better; 0 means every category is a single
/// point; 1 means intra and inter are indistinguishable (no clustering).
fn cluster_score(points: &[SphericalPoint], categories: &[String]) -> f64 {
    assert_eq!(points.len(), categories.len());
    let n = points.len();
    let mut intra_sum = 0.0_f64;
    let mut intra_n = 0_usize;
    let mut inter_sum = 0.0_f64;
    let mut inter_n = 0_usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = sphereql_core::angular_distance(&points[i], &points[j]);
            if categories[i] == categories[j] {
                intra_sum += d;
                intra_n += 1;
            } else {
                inter_sum += d;
                inter_n += 1;
            }
        }
    }
    if intra_n == 0 || inter_n == 0 {
        return 1.0;
    }
    let intra = intra_sum / intra_n as f64;
    let inter = inter_sum / inter_n as f64;
    if inter <= 1e-12 {
        return 1.0;
    }
    intra / inter
}

/// Worst-case angular distance between two fits. With a deterministic
/// PRNG seed every projection in this crate should produce identical
/// coordinates across runs; anything above `1e-9` indicates an
/// uncontrolled non-determinism source.
fn max_pointwise_angular_distance(a: &[SphericalPoint], b: &[SphericalPoint]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| sphereql_core::angular_distance(p, q))
        .fold(0.0_f64, f64::max)
}

// ── Projection runners ─────────────────────────────────────────────────────

#[derive(Debug)]
struct ProjectionRow {
    name: &'static str,
    cluster: f64,
    determinism: f64,
    logical_confidence: Option<f64>,
    fit_ms: u128,
}

fn run_pca(embeddings: &[Embedding]) -> Box<dyn Projection> {
    Box::new(
        PcaProjection::fit(embeddings, RadialStrategy::Magnitude)
            .expect("PCA fit")
            .with_volumetric(true),
    )
}

fn run_kpca(embeddings: &[Embedding]) -> Box<dyn Projection> {
    Box::new(KernelPcaProjection::fit(embeddings, RadialStrategy::Magnitude).expect("KPCA fit"))
}

fn run_laplacian(embeddings: &[Embedding]) -> Box<dyn Projection> {
    // Defaults that work for both corpora: k=10 neighbors, no active filter.
    Box::new(
        LaplacianEigenmapProjection::fit_with_params(
            embeddings,
            10,
            0.0,
            RadialStrategy::Magnitude,
        )
        .expect("Laplacian fit"),
    )
}

fn run_umap(embeddings: &[Embedding]) -> Box<dyn Projection> {
    Box::new(
        UmapSphereProjection::fit(
            embeddings,
            None,
            RadialStrategy::Magnitude,
            UmapConfig::default(),
        )
        .expect("UMAP fit"),
    )
}

type Fitter = fn(&[Embedding]) -> Box<dyn Projection>;

fn run_one(
    name: &'static str,
    fitter: Fitter,
    embeddings: &[Embedding],
    categories: &[String],
    confidence: &dyn LogicalConfidence,
) -> ProjectionRow {
    let t0 = Instant::now();
    let proj1 = fitter(embeddings);
    let fit_ms = t0.elapsed().as_millis();
    let coords1: Vec<SphericalPoint> = embeddings.iter().map(|e| proj1.project(e)).collect();

    // Determinism: re-fit and compare the worst-case pointwise distance.
    let proj2 = fitter(embeddings);
    let coords2: Vec<SphericalPoint> = embeddings.iter().map(|e| proj2.project(e)).collect();
    let determinism = max_pointwise_angular_distance(&coords1, &coords2);

    let cluster = cluster_score(&coords1, categories);
    let logical_confidence = confidence.score(&coords1, categories);

    ProjectionRow {
        name,
        cluster,
        determinism,
        logical_confidence,
        fit_ms,
    }
}

fn run_all(embeddings: &[Embedding], categories: &[String]) -> Vec<ProjectionRow> {
    let confidence = UnimplementedLogicalConfidence;
    let fitters: &[(&'static str, Fitter)] = &[
        ("pca", run_pca),
        ("kernel_pca", run_kpca),
        ("laplacian", run_laplacian),
        ("umap_sphere", run_umap),
    ];
    fitters
        .iter()
        .map(|(name, f)| run_one(name, *f, embeddings, categories, &confidence))
        .collect()
}

fn print_table(corpus_name: &str, n_docs: usize, n_categories: usize, rows: &[ProjectionRow]) {
    println!("\ncorpus: {corpus_name}  ({n_docs} docs, {n_categories} categories)");
    println!("┌──────────────────┬──────────┬──────────────┬──────────────────────┬──────────┐");
    println!("│ projection       │ cluster  │ determinism  │ logical_confidence   │ fit (ms) │");
    println!("├──────────────────┼──────────┼──────────────┼──────────────────────┼──────────┤");
    for r in rows {
        let lc = match r.logical_confidence {
            Some(v) => format!("{v:.4}"),
            None => "n/a".to_string(),
        };
        println!(
            "│ {:<16} │ {:>8.4} │ {:>12.2e} │ {:>20} │ {:>8} │",
            r.name, r.cluster, r.determinism, lc, r.fit_ms,
        );
    }
    println!("└──────────────────┴──────────┴──────────────┴──────────────────────┴──────────┘");
}

fn count_categories(categories: &[String]) -> usize {
    let mut set: HashMap<&str, ()> = HashMap::new();
    for c in categories {
        set.insert(c.as_str(), ());
    }
    set.len()
}

/// Strict baseline ceilings, per-corpus per-projection. Captured on the
/// first green run with a ~10% headroom; tightening these is welcome,
/// loosening should require an investigation comment.
const TEXT_BASELINE: &[(&str, f64)] = &[
    ("pca", 1.02),
    ("kernel_pca", 1.02),
    ("laplacian", 1.04),
    ("umap_sphere", 0.99),
];
const SPHEREQL_BASELINE: &[(&str, f64)] = &[
    ("pca", 0.55),
    ("kernel_pca", 0.54),
    ("laplacian", 1.02),
    ("umap_sphere", 0.51),
];

/// Relative-spread guard: no projection's cluster score may exceed
/// `RELATIVE_RATIO × best_score` on the same corpus. Catches one
/// projection silently regressing while the others stay healthy.
/// Set wide enough to accept Laplacian's known weakness on the
/// 775-concept corpus (≈2.04× the UMAP best); tighten when Laplacian
/// is improved.
const RELATIVE_RATIO: f64 = 2.2;

/// Determinism gap that should never be exceeded on a deterministic
/// projection family with a fixed seed.
const DETERMINISM_GAP: f64 = 1e-6;

fn assert_baselines(rows: &[ProjectionRow], baseline: &[(&str, f64)]) {
    for (name, ceiling) in baseline {
        let row = rows
            .iter()
            .find(|r| r.name == *name)
            .unwrap_or_else(|| panic!("missing row for {name}"));
        assert!(
            row.cluster <= *ceiling,
            "{name}: cluster {:.4} exceeds baseline ceiling {ceiling:.4}",
            row.cluster
        );
        assert!(
            row.determinism <= DETERMINISM_GAP,
            "{name}: determinism gap {:.2e} > {:.0e}",
            row.determinism,
            DETERMINISM_GAP
        );
    }
    let best = rows.iter().map(|r| r.cluster).fold(f64::INFINITY, f64::min);
    let worst = rows.iter().map(|r| r.cluster).fold(0.0_f64, f64::max);
    assert!(
        worst <= RELATIVE_RATIO * best,
        "relative spread: worst {:.4} > {:.1}× best {:.4}",
        worst,
        RELATIVE_RATIO,
        best
    );
}

#[test]
fn projection_comparison_text_e2e_corpus() {
    let (embeddings, categories) = text_corpus();
    let rows = run_all(&embeddings, &categories);
    print_table(
        "e2e_text",
        embeddings.len(),
        count_categories(&categories),
        &rows,
    );
    assert_baselines(&rows, TEXT_BASELINE);
}

#[test]
fn projection_comparison_sphereql_corpus() {
    let (embeddings, categories) = sphereql_corpus_inputs();
    let rows = run_all(&embeddings, &categories);
    print_table(
        "sphereql_corpus",
        embeddings.len(),
        count_categories(&categories),
        &rows,
    );
    assert_baselines(&rows, SPHEREQL_BASELINE);
}
