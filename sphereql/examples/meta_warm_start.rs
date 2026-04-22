#![allow(clippy::uninlined_format_args)]
//! Demonstrates `SphereQLPipeline::new_from_metamodel_tuned` — the
//! warm-started hybrid between recall and tuning.
//!
//! Flow:
//!   1. Prime a meta-model with one training record that pins an
//!      unusual knob value the default SearchSpace doesn't explore.
//!   2. Build a pipeline via `new_from_metamodel_tuned` on a fresh
//!      corpus, varying only a subset of knobs.
//!   3. Verify the unusual pinned knob survived the tuner pass (it's
//!      outside the search space) while the searched knobs were
//!      actually refined.
//!
//! Takeaway: the meta-model's prediction becomes the tuner's base_config,
//! so every knob NOT in SearchSpace stays at the predicted value. This
//! is the sweet spot between "skip search entirely" and "search from
//! scratch every time".
//!
//! Run with:
//!   cargo run --example meta_warm_start --features embed --release

use sphereql::embed::{
    CompositeMetric, CorpusFeatures, MetaModel, MetaTrainingRecord, NearestNeighborMetaModel,
    PipelineConfig, PipelineInput, ProjectionKind, SearchSpace, SearchStrategy, SphereQLPipeline,
};
use sphereql_corpus::{build_corpus, embed};

const BUDGET: usize = 6;

fn main() {
    println!("================================================================");
    println!("  SphereQL warm-started hybrid: metamodel prediction + tuning");
    println!("================================================================\n");

    let corpus = build_corpus();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();
    println!("Corpus: {} concepts\n", embeddings.len());

    // ── 1. Seed a meta-model with one "expert-provided" record that
    //     pins an unusual threshold_evr_penalty value. The default
    //     SearchSpace uses {0.2, 0.4, 0.6} for this knob — we'll pin
    //     0.75 via the prediction and verify it survives tuning.
    let features = CorpusFeatures::extract(&categories, &embeddings);

    let mut expert_cfg = PipelineConfig::default();
    expert_cfg.bridges.threshold_evr_penalty = 0.75;
    expert_cfg.routing.num_domain_groups = 4; // also not in default space

    let record = MetaTrainingRecord::from_tune_result(
        "expert_seed",
        features.clone(),
        &mock_report(expert_cfg.clone(), 0.6, "default_composite"),
        "expert_seed".to_string(),
    );

    let mut model = NearestNeighborMetaModel::new();
    model.fit(&[record]);

    println!("Expert seed record:");
    println!(
        "  threshold_evr_penalty = {} (NOT in default SearchSpace)",
        expert_cfg.bridges.threshold_evr_penalty
    );
    println!(
        "  num_domain_groups     = {} (NOT in default SearchSpace)",
        expert_cfg.routing.num_domain_groups
    );
    println!(
        "  overlap_artifact_territorial = {} (IS in default SearchSpace — will be tuned)",
        expert_cfg.bridges.overlap_artifact_territorial
    );

    // ── 2. Constrain the SearchSpace to vary only overlap_artifact.
    //     Everything else stays pinned by the expert prediction.
    let space = SearchSpace {
        projection_kinds: vec![ProjectionKind::Pca],
        laplacian_k_neighbors: vec![15],
        laplacian_active_threshold: vec![0.05],
        num_domain_groups: vec![expert_cfg.routing.num_domain_groups], // pin
        low_evr_threshold: vec![expert_cfg.routing.low_evr_threshold], // pin
        overlap_artifact_territorial: vec![0.2, 0.3, 0.4],             // the only tuned axis
        threshold_base: vec![expert_cfg.bridges.threshold_base],       // pin
        threshold_evr_penalty: vec![expert_cfg.bridges.threshold_evr_penalty], // pin at 0.75
        min_evr_improvement: vec![expert_cfg.inner_sphere.min_evr_improvement], // pin
    };

    let metric = CompositeMetric::default_composite();
    println!("\nTuning — only `overlap_artifact_territorial` varies:");
    let start = std::time::Instant::now();
    let (pipeline, _f, report) = SphereQLPipeline::new_from_metamodel_tuned(
        PipelineInput {
            categories,
            embeddings,
        },
        &model,
        &space,
        &metric,
        SearchStrategy::Random {
            budget: BUDGET,
            seed: 0xC0FF_EE,
        },
    )
    .expect("warm-start tune failed");
    println!(
        "  {} trials in {:.2}s",
        report.trials.len(),
        start.elapsed().as_secs_f64()
    );
    println!("  best score: {:.4}", report.best_score);

    // ── 3. Verify pinned knobs survived ───────────────────────────────
    let winner = &pipeline.config();
    println!("\nVerification (pinned knobs from expert seed survived tuning):");
    check_eq(
        "threshold_evr_penalty",
        winner.bridges.threshold_evr_penalty,
        expert_cfg.bridges.threshold_evr_penalty,
    );
    check_eq_usize(
        "num_domain_groups",
        winner.routing.num_domain_groups,
        expert_cfg.routing.num_domain_groups,
    );

    println!("\nSearched knob (overlap_artifact_territorial):");
    println!(
        "  tuner picked: {:.2} (from candidates {:?})",
        winner.bridges.overlap_artifact_territorial, space.overlap_artifact_territorial
    );
}

fn mock_report(cfg: PipelineConfig, score: f64, metric_name: &str) -> sphereql::embed::TuneReport {
    sphereql::embed::TuneReport {
        metric_name: metric_name.to_string(),
        best_score: score,
        best_config: cfg,
        trials: Vec::new(),
        failures: Vec::new(),
    }
}

fn check_eq(label: &str, actual: f64, expected: f64) {
    let ok = (actual - expected).abs() < 1e-9;
    let mark = if ok { "OK" } else { "FAIL" };
    println!(
        "  [{}] {:<28} actual={:.4}  expected={:.4}",
        mark, label, actual, expected
    );
}

fn check_eq_usize(label: &str, actual: usize, expected: usize) {
    let ok = actual == expected;
    let mark = if ok { "OK" } else { "FAIL" };
    println!(
        "  [{}] {:<28} actual={}  expected={}",
        mark, label, actual, expected
    );
}
