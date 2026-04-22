#![allow(clippy::uninlined_format_args)]
//! Demonstrates the Phase-4 meta-learning loop end-to-end:
//!
//! 1. Tune a pipeline on the built-in corpus (auto_tune).
//! 2. Extract a CorpusFeatures profile and build a MetaTrainingRecord.
//! 3. Save the record list to a scratch file.
//! 4. Reload records, fit a NearestNeighborMetaModel, predict a config.
//! 5. Build a "recalled" pipeline via SphereQLPipeline::new_from_metamodel.
//! 6. Verify the recalled pipeline's config matches the tuner's winner.
//!
//! This example writes to `./target/meta_records.json`, NOT the default
//! `~/.sphereql/` store — keeping example runs scoped to the workspace.
//! Production callers should use MetaTrainingRecord::append_to_default_store
//! to accumulate records across sessions.
//!
//! Run with:
//!   cargo run --example meta_learn --features embed --release

use sphereql::embed::{
    auto_tune, CompositeMetric, CorpusFeatures, MetaModel, MetaTrainingRecord,
    NearestNeighborMetaModel, PipelineConfig, PipelineInput, QualityMetric, SearchSpace,
    SearchStrategy, SphereQLPipeline,
};
use sphereql_corpus::{build_corpus, embed};

const BUDGET: usize = 12;
const SEED: u64 = 0xA17C_ABE_CAFE;

fn main() {
    println!("================================================================");
    println!("  SphereQL meta-learning: tune → record → save → recall");
    println!("================================================================\n");

    let corpus = build_corpus();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();

    println!("Corpus: {} concepts\n", embeddings.len());

    // ── 1. Tune ────────────────────────────────────────────────────────
    let space = SearchSpace::default();
    let metric = CompositeMetric::default_composite();

    println!("Step 1: tuning ({} trials)...", BUDGET);
    let (tuned_pipeline, report) = auto_tune(
        PipelineInput {
            categories: categories.clone(),
            embeddings: embeddings.clone(),
        },
        &space,
        &metric,
        SearchStrategy::Random {
            budget: BUDGET,
            seed: SEED,
        },
        &PipelineConfig::default(),
    )
    .expect("auto_tune failed");

    println!(
        "  best {} score = {:.4}  ({})",
        report.metric_name,
        report.best_score,
        tuned_pipeline.projection_kind().name()
    );

    // ── 2. Extract features + build training record ───────────────────
    println!("\nStep 2: extracting corpus features...");
    let features = CorpusFeatures::extract(&categories, &embeddings);
    print_features(&features);

    let record = MetaTrainingRecord::from_tune_result(
        "built_in_775",
        features,
        &report,
        format!("random{{budget={},seed={:#X}}}", BUDGET, SEED),
    );

    // ── 3. Save to a local scratch file ────────────────────────────────
    let scratch = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("target")
        .join("meta_records.json");
    println!("\nStep 3: saving record to {}", scratch.display());

    // Accumulate: load whatever exists, append, save back.
    let mut records = MetaTrainingRecord::load_list(&scratch).expect("load failed");
    records.push(record.clone());
    MetaTrainingRecord::save_list(&records, &scratch).expect("save failed");
    println!("  store now holds {} record(s)", records.len());

    // ── 4. Reload + fit meta-model ────────────────────────────────────
    println!("\nStep 4: fitting NearestNeighborMetaModel on the store...");
    let loaded = MetaTrainingRecord::load_list(&scratch).expect("load failed");
    let mut model = NearestNeighborMetaModel::new();
    model.fit(&loaded);
    println!("  model fitted on {} record(s)", loaded.len());

    // ── 5. Recall: build a pipeline via the model ─────────────────────
    println!("\nStep 5: building a pipeline via new_from_metamodel...");
    let (recalled_pipeline, _f, predicted_config) = SphereQLPipeline::new_from_metamodel(
        PipelineInput {
            categories: categories.clone(),
            embeddings: embeddings.clone(),
        },
        &model,
    )
    .expect("metamodel recall failed");

    println!(
        "  predicted projection: {}",
        predicted_config.projection_kind.name()
    );
    println!(
        "  recalled pipeline projection_kind: {}",
        recalled_pipeline.projection_kind().name()
    );

    // ── 6. Verify ─────────────────────────────────────────────────────
    let recall_score = metric.score(&recalled_pipeline);
    println!("\nStep 6: scoring both pipelines under {}:", metric.name());
    println!("  tuned    = {:.4}", report.best_score);
    println!("  recalled = {:.4}", recall_score);

    if (recall_score - report.best_score).abs() < 1e-6 {
        println!("  → recall exactly matches tuner's winner (expected with 1 record)");
    } else {
        println!(
            "  → recall differs from tuner ({:+.4}) — likely because the stored record's config",
            recall_score - report.best_score
        );
        println!("    is applied to a newly-built pipeline with the same inputs.");
    }

    // ── Operational note: where the "real" store lives ────────────────
    match MetaTrainingRecord::default_store_path() {
        Ok(p) => println!(
            "\nProduction default store would live at: {}\n(not written by this example)",
            p.display()
        ),
        Err(e) => println!("\nDefault store path unavailable: {}", e),
    }
}

fn print_features(f: &CorpusFeatures) {
    let labels = CorpusFeatures::feature_names();
    let values = f.to_vec();
    println!(
        "  {:<35} {:>10}",
        "feature", "value"
    );
    println!("  {}", "─".repeat(48));
    for (name, v) in labels.iter().zip(values.iter()) {
        println!("  {:<35} {:>10.4}", name, v);
    }
    println!(
        "  {:<35} {:>10.4}",
        "category_separation_ratio (derived)", f.category_separation_ratio
    );
}
