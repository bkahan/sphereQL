//! End-to-end demo: tune a pipeline with `CorpusQuality`.
//!
//! Builds a small synthetic corpus, hands it to `auto_tune` with the
//! Phase-5 composite metric, and prints the resulting `TuneReport`
//! summary plus the per-axis sub-score breakdown captured on the
//! metric's `last_breakdown()` after the best trial.
//!
//! Run:
//! ```bash
//! cargo run -p sphereql-embed --example cq_e2e --release
//! ```

use sphereql_embed::{
    CorpusQuality, PipelineConfig, PipelineInput, SearchSpace, SearchStrategy, auto_tune,
};

fn synthetic_corpus(n_per: usize, n_cats: usize, dim: usize) -> PipelineInput {
    let mut categories = Vec::with_capacity(n_per * n_cats);
    let mut embeddings = Vec::with_capacity(n_per * n_cats);
    let mut state: u64 = 0xABCDEF0123456789;
    for c in 0..n_cats {
        for _ in 0..n_per {
            categories.push(format!("cat_{c}"));
            let mut v = vec![0.0_f64; dim];
            v[c % dim] = 1.0;
            for x in v.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 33) as f64 / (1u64 << 31) as f64;
                *x += (u - 0.5) * 0.05;
            }
            embeddings.push(v);
        }
    }
    PipelineInput {
        categories,
        embeddings,
    }
}

fn main() {
    let input = synthetic_corpus(20, 6, 16);
    let space = SearchSpace::default();
    let metric = CorpusQuality::default();

    let (_pipeline, report) = auto_tune(
        input,
        &space,
        &metric,
        SearchStrategy::Grid,
        &PipelineConfig::default(),
    )
    .expect("auto_tune");

    assert_eq!(report.metric_name, "corpus_quality");
    println!("metric_name = {}", report.metric_name);
    println!("best_score  = {:.4}", report.best_score);
    println!("trials      = {}", report.trials.len());

    if let Some(bd) = metric.last_breakdown() {
        println!(
            "breakdown   = evr={:.3} bridge={:.3} curv={:.3} balance={:.3} composite={:.3}",
            bd.evr, bd.bridge_coherence, bd.curvature_health, bd.category_balance, bd.composite
        );
    }
}
