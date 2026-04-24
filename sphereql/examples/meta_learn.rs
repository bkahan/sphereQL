#![allow(clippy::uninlined_format_args)]
//! End-to-end validation of the L2 metalearning layer.
//!
//! Tunes on both the 775-concept built-in corpus (favors PCA) and the
//! 300-concept stress corpus (favors Laplacian), accumulates both
//! winning configs into a MetaTrainingRecord store, fits two meta-models
//! (NearestNeighbor + DistanceWeighted) on those records, and verifies
//! that querying the store with a "built-in-like" feature profile
//! predicts PCA while a "stress-like" profile predicts Laplacian.
//!
//! This exercises:
//!   - auto_tune across two genuinely different corpus regimes
//!   - MetaTrainingRecord::from_tune_result
//!   - MetaTrainingRecord::save_list / load_list (local scratch path)
//!   - NearestNeighborMetaModel::fit/predict
//!   - DistanceWeightedMetaModel::fit/predict
//!   - CorpusFeatures::extract and to_vec
//!
//! Writes to `./target/meta_records.json` (not the user's default
//! `~/.sphereql/` store) so running the example is self-contained.
//!
//! Run with:
//!   cargo run --example meta_learn --features embed --release

use sphereql::embed::{
    CompositeMetric, CorpusFeatures, DistanceWeightedMetaModel, MetaModel, MetaTrainingRecord,
    NearestNeighborMetaModel, PipelineConfig, PipelineInput, ProjectionKind, QualityMetric,
    SearchSpace, SearchStrategy, auto_tune,
};
use sphereql_corpus::{
    Concept, STRESS_NOISE_AMPLITUDE, build_corpus, build_stress_corpus, embed, embed_with_noise,
};

const BUDGET: usize = 12;
const BASE_SEED: u64 = 0x0A17_CABE_CAFE;

fn main() {
    println!("================================================================");
    println!("  SphereQL meta-learning — cross-corpus populate + verify");
    println!("================================================================\n");

    let metric = CompositeMetric::default_composite();
    let space = SearchSpace::default();

    // ── 1. Tune on BOTH corpora, emit one training record each ─────────
    let (built_in_record, built_in_features) =
        tune_and_record("built_in_775", build_corpus(), false, &space, &metric);
    let (stress_record, stress_features) =
        tune_and_record("stress_300", build_stress_corpus(), true, &space, &metric);

    println!("\nTraining store: 2 records");
    println!(
        "  {:<16}  projection={:<20}  score={:.4}",
        built_in_record.corpus_id,
        built_in_record.best_config.projection_kind.name(),
        built_in_record.best_score,
    );
    println!(
        "  {:<16}  projection={:<20}  score={:.4}",
        stress_record.corpus_id,
        stress_record.best_config.projection_kind.name(),
        stress_record.best_score,
    );

    // ── 2. Save to scratch, reload ────────────────────────────────────
    let store = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("target")
        .join("meta_records.json");
    MetaTrainingRecord::save_list(&[built_in_record.clone(), stress_record.clone()], &store)
        .expect("save failed");
    let loaded = MetaTrainingRecord::load_list(&store).expect("load failed");
    assert_eq!(loaded.len(), 2);
    println!("\nStore round-tripped to {}", store.display());

    // ── 3. Fit both meta-models ───────────────────────────────────────
    let mut nn = NearestNeighborMetaModel::new();
    nn.fit(&loaded);
    let mut dw = DistanceWeightedMetaModel::new();
    dw.fit(&loaded);
    println!("\nFitted models: {} + {}", nn.name(), dw.name());

    // ── 4. Verify: feed each corpus's actual features back through ────
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  Verification: do the models predict the right kind?");
    println!("────────────────────────────────────────────────────────────────");
    verify(
        &nn,
        &dw,
        "built_in_775 profile",
        &built_in_features,
        built_in_record.best_config.projection_kind,
    );
    verify(
        &nn,
        &dw,
        "stress_300 profile",
        &stress_features,
        stress_record.best_config.projection_kind,
    );

    // ── 5. Verify: perturbed profiles still route correctly ───────────
    // A profile halfway between the two regimes: should still lean
    // toward whichever it's closer to. Test both directions.
    let near_built_in = interp(&built_in_features, &stress_features, 0.2);
    let near_stress = interp(&built_in_features, &stress_features, 0.8);
    verify(
        &nn,
        &dw,
        "built-in-like (20% toward stress)",
        &near_built_in,
        ProjectionKind::Pca,
    );
    verify(
        &nn,
        &dw,
        "stress-like (80% toward stress)",
        &near_stress,
        ProjectionKind::LaplacianEigenmap,
    );

    println!();
}

fn tune_and_record(
    corpus_id: &str,
    corpus: Vec<Concept>,
    stress: bool,
    space: &SearchSpace,
    metric: &impl QualityMetric,
) -> (MetaTrainingRecord, CorpusFeatures) {
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = if stress {
        corpus
            .iter()
            .enumerate()
            .map(|(i, c)| embed_with_noise(&c.features, 9000 + i as u64, STRESS_NOISE_AMPLITUDE))
            .collect()
    } else {
        corpus
            .iter()
            .enumerate()
            .map(|(i, c)| embed(&c.features, 1000 + i as u64))
            .collect()
    };

    println!(
        "\n[{}] tuning {} concepts / {} categories...",
        corpus_id,
        embeddings.len(),
        categories
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    let start = std::time::Instant::now();
    let (_pipeline, report) = auto_tune(
        PipelineInput {
            categories: categories.clone(),
            embeddings: embeddings.clone(),
        },
        space,
        metric,
        SearchStrategy::Random {
            budget: BUDGET,
            seed: BASE_SEED.wrapping_add(corpus_id.len() as u64),
        },
        &PipelineConfig::default(),
    )
    .expect("auto_tune failed");
    let elapsed = start.elapsed();

    let features = CorpusFeatures::extract(&categories, &embeddings);
    let record = MetaTrainingRecord::from_tune_result(
        corpus_id,
        features.clone(),
        &report,
        format!("random{{budget={}}}", BUDGET),
    );

    println!(
        "  {} → best {:.4} under {} in {:.2}s (projection {})",
        corpus_id,
        report.best_score,
        report.metric_name,
        elapsed.as_secs_f64(),
        record.best_config.projection_kind.name(),
    );

    (record, features)
}

fn verify(
    nn: &NearestNeighborMetaModel,
    dw: &DistanceWeightedMetaModel,
    label: &str,
    features: &CorpusFeatures,
    expected_kind: ProjectionKind,
) {
    let nn_pred = nn.predict(features).projection_kind;
    let dw_pred = dw.predict(features).projection_kind;
    let mark = |k: ProjectionKind| {
        if k == expected_kind { "OK " } else { "MISS" }
    };
    println!(
        "  {:<42}  expected={:<20}  nn={} {}  dw={} {}",
        label,
        expected_kind.name(),
        nn_pred.name(),
        mark(nn_pred),
        dw_pred.name(),
        mark(dw_pred),
    );
}

/// Interpolate between two CorpusFeatures profiles. Used to probe how
/// the models behave on intermediate points that aren't exactly either
/// stored record's profile.
fn interp(a: &CorpusFeatures, b: &CorpusFeatures, t: f64) -> CorpusFeatures {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: f64, y: f64| (1.0 - t) * x + t * y;
    CorpusFeatures {
        // Discrete fields: round the mix; these usually don't move the
        // normalized distance much compared with the scalar features.
        n_items: mix(a.n_items as f64, b.n_items as f64).round() as usize,
        n_categories: mix(a.n_categories as f64, b.n_categories as f64).round() as usize,
        dim: a.dim, // dim is fixed across corpora (128 here); no need to interp
        mean_members_per_category: mix(a.mean_members_per_category, b.mean_members_per_category),
        category_size_entropy: mix(a.category_size_entropy, b.category_size_entropy),
        mean_sparsity: mix(a.mean_sparsity, b.mean_sparsity),
        axis_utilization_entropy: mix(a.axis_utilization_entropy, b.axis_utilization_entropy),
        noise_estimate: mix(a.noise_estimate, b.noise_estimate),
        mean_intra_category_similarity: mix(
            a.mean_intra_category_similarity,
            b.mean_intra_category_similarity,
        ),
        mean_inter_category_similarity: mix(
            a.mean_inter_category_similarity,
            b.mean_inter_category_similarity,
        ),
        category_separation_ratio: mix(a.category_separation_ratio, b.category_separation_ratio),
    }
}
