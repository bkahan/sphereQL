#![allow(clippy::uninlined_format_args)]
//! Auto-tune the SphereQL pipeline against the built-in 775-concept corpus.
//!
//! Runs a random sweep over the [`SearchSpace`] using the default composite
//! quality metric (bridge coherence + territorial health + cluster
//! silhouette). Prints the winning config, top-5 trial ranking, and the
//! per-component score breakdown for the winner.
//!
//! Run with:
//!   cargo run --example auto_tune --features embed --release
//!
//! Use --release; each trial rebuilds the pipeline (spatial quality Monte
//! Carlo + graph + bridge classification) so debug mode is ~5× slower.

use sphereql::embed::{
    auto_tune, CompositeMetric, PipelineConfig, PipelineInput, QualityMetric, SearchSpace,
    SearchStrategy,
};
use sphereql_corpus::{build_corpus, embed};

const RANDOM_BUDGET: usize = 24;
const RANDOM_SEED: u64 = 0xA17C_ABE_CAFE;

fn main() {
    println!("================================================================");
    println!("  SphereQL AutoTuner: random sweep on the 775-concept corpus");
    println!("================================================================\n");

    let corpus = build_corpus();
    let n = corpus.len();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();
    let unique_cats: std::collections::HashSet<&str> =
        categories.iter().map(|s| s.as_str()).collect();

    println!("Corpus: {} concepts across {} categories", n, unique_cats.len());
    println!("Budget: {} random trials (seed = 0x{:X})\n", RANDOM_BUDGET, RANDOM_SEED);

    let space = SearchSpace::default();
    let kinds_str: Vec<&str> = space.projection_kinds.iter().map(|k| k.name()).collect();
    println!("Search space (discrete):");
    println!("  projection_kinds .............. {:?}", kinds_str);
    println!("  num_domain_groups ............. {:?}", space.num_domain_groups);
    println!("  low_evr_threshold ............. {:?}", space.low_evr_threshold);
    println!("  overlap_artifact_territorial .. {:?}", space.overlap_artifact_territorial);
    println!("  threshold_base ................ {:?}", space.threshold_base);
    println!("  threshold_evr_penalty ......... {:?}", space.threshold_evr_penalty);
    println!("  min_evr_improvement ........... {:?}", space.min_evr_improvement);
    println!("  grid cardinality .............. {}\n", space.grid_cardinality());

    let metric = CompositeMetric::default_composite();
    let start = std::time::Instant::now();
    let (best_pipeline, report) = auto_tune(
        PipelineInput {
            categories,
            embeddings,
        },
        &space,
        &metric,
        SearchStrategy::Random {
            budget: RANDOM_BUDGET,
            seed: RANDOM_SEED,
        },
        &PipelineConfig::default(),
    )
    .expect("auto_tune failed");
    let elapsed = start.elapsed();

    println!("Tuning complete in {:.2}s", elapsed.as_secs_f64());
    println!("Metric: {}\n", report.metric_name);

    // Baseline for comparison.
    let default_pipeline = sphereql::embed::SphereQLPipeline::new(PipelineInput {
        categories: corpus.iter().map(|c| c.category.to_string()).collect(),
        embeddings: corpus
            .iter()
            .enumerate()
            .map(|(i, c)| embed(&c.features, 1000 + i as u64))
            .collect(),
    })
    .unwrap();
    let default_score = metric.score(&default_pipeline);

    println!("Score summary:");
    println!("  default config ......... {:.4}", default_score);
    println!("  mean across trials ..... {:.4}", report.mean_score());
    println!("  best (found) ........... {:.4}", report.best_score);
    let lift = report.best_score - default_score;
    println!(
        "  lift over default ...... {:+.4}  ({:+.1}%)\n",
        lift,
        100.0 * lift / default_score.max(1e-12)
    );

    println!(
        "Winning projection kind: {}",
        best_pipeline.projection_kind().name()
    );

    println!("\nTop 5 trials (by score):");
    println!(
        "  {:>4}  {:>6}  {:>6}  {:>18}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "rank", "score", "build", "projection", "groups", "low_evr", "overlap", "base", "evr_pen"
    );
    println!("  {}", "─".repeat(102));
    for (rank, t) in report.ranked_trials().iter().take(5).enumerate() {
        println!(
            "  {:>4}  {:>6.4}  {:>4}ms  {:>18}  {:>8}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
            rank + 1,
            t.score,
            t.build_ms,
            t.config.projection_kind.name(),
            t.config.routing.num_domain_groups,
            t.config.routing.low_evr_threshold,
            t.config.bridges.overlap_artifact_territorial,
            t.config.bridges.threshold_base,
            t.config.bridges.threshold_evr_penalty,
        );
    }

    // Per-projection-kind score summary so you can see which family won
    // overall, not just the single best trial.
    use std::collections::BTreeMap;
    let mut per_kind: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
    for t in &report.trials {
        let entry = per_kind
            .entry(t.config.projection_kind.name())
            .or_insert((f64::NEG_INFINITY, 0.0, 0));
        entry.0 = entry.0.max(t.score);
        entry.1 += t.score;
        entry.2 += 1;
    }
    println!("\nPer-projection summary:");
    println!(
        "  {:<20} {:>8} {:>10} {:>10}",
        "kind", "trials", "best", "mean"
    );
    println!("  {}", "─".repeat(52));
    for (kind, (best, sum, count)) in per_kind {
        println!(
            "  {:<20} {:>8} {:>10.4} {:>10.4}",
            kind,
            count,
            best,
            sum / count as f64
        );
    }

    println!("\nWinning config component scores:");
    for (name, weight, score) in metric.score_components(&best_pipeline) {
        let bar_len = (score * 30.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(30 - bar_len);
        println!(
            "  {:<24} w={:.2}  score={:.4}  [{}]",
            name, weight, score, bar
        );
    }

    if !report.failures.is_empty() {
        println!(
            "\n{} trial(s) failed to build:",
            report.failures.len()
        );
        for (_, err) in report.failures.iter().take(3) {
            println!("  - {}", err);
        }
    }

    // Bridge classification distribution on the tuned pipeline — the most
    // actionable signal for whether the tuner actually moved the needle
    // on the low-EVR pathology.
    let layer = best_pipeline.category_layer();
    let (mut genuine, mut artifact, mut weak) = (0usize, 0usize, 0usize);
    for bridges in layer.graph.bridges.values() {
        for b in bridges {
            use sphereql::embed::BridgeClassification::*;
            match b.classification {
                Genuine => genuine += 1,
                OverlapArtifact => artifact += 1,
                Weak => weak += 1,
            }
        }
    }
    let total = genuine + artifact + weak;
    println!("\nBridge classification on winning pipeline ({} bridges):", total);
    if total > 0 {
        println!(
            "  Genuine         : {:>5}  ({:>5.1}%)",
            genuine,
            100.0 * genuine as f64 / total as f64
        );
        println!(
            "  Weak            : {:>5}  ({:>5.1}%)",
            weak,
            100.0 * weak as f64 / total as f64
        );
        println!(
            "  OverlapArtifact : {:>5}  ({:>5.1}%)",
            artifact,
            100.0 * artifact as f64 / total as f64
        );
    }

    println!("\nDone.");
}
