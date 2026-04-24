#![allow(clippy::uninlined_format_args)]
//! Auto-tune the SphereQL pipeline against the built-in 775-concept corpus,
//! under two contrasting quality metrics:
//!
//! 1. `default_composite` — 40% bridge coherence / 35% territorial health /
//!    25% cluster silhouette. Silhouette is variance-centric and typically
//!    rewards PCA's spread.
//! 2. `connectivity_composite` — 50% graph modularity / 30% bridge
//!    coherence / 20% territorial health. Modularity evaluates the k-NN
//!    graph of projected positions; a projection that preserves
//!    same-category adjacency scores well regardless of total variance.
//!
//! Running both head-to-head tests the hypothesis that "PCA wins" is a
//! metric-choice artifact rather than a property of the corpus.
//!
//! Run with:
//!   cargo run --example auto_tune --features embed --release
//!
//! Switch corpora via the `SPHEREQL_CORPUS` env var:
//!   SPHEREQL_CORPUS=stress  → 300-concept extreme-sparsity stress corpus
//!   (anything else or unset → 775-concept built-in corpus)

use sphereql::embed::{
    BridgeClassification, CompositeMetric, PipelineConfig, PipelineInput, ProjectionKind,
    QualityMetric, SearchSpace, SearchStrategy, SphereQLPipeline, TuneReport, auto_tune,
};
use sphereql_corpus::{
    Concept, STRESS_NOISE_AMPLITUDE, build_corpus, build_stress_corpus, embed, embed_with_noise,
};

const RANDOM_BUDGET: usize = 24;
const RANDOM_SEED: u64 = 0x0A17_CABE_CAFE;

fn load_corpus_from_env() -> (Vec<Concept>, Vec<String>, Vec<Vec<f64>>, &'static str) {
    let use_stress = std::env::var("SPHEREQL_CORPUS")
        .map(|v| v == "stress")
        .unwrap_or(false);

    if use_stress {
        let corpus = build_stress_corpus();
        let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
        let embeddings: Vec<Vec<f64>> = corpus
            .iter()
            .enumerate()
            .map(|(i, c)| embed_with_noise(&c.features, 9000 + i as u64, STRESS_NOISE_AMPLITUDE))
            .collect();
        (corpus, categories, embeddings, "stress")
    } else {
        let corpus = build_corpus();
        let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
        let embeddings: Vec<Vec<f64>> = corpus
            .iter()
            .enumerate()
            .map(|(i, c)| embed(&c.features, 1000 + i as u64))
            .collect();
        (corpus, categories, embeddings, "built_in")
    }
}

fn main() {
    println!("================================================================");
    println!("  SphereQL AutoTuner: metric-choice comparison");
    println!("================================================================\n");

    let (corpus, categories, embeddings, corpus_label) = load_corpus_from_env();
    let n = corpus.len();
    let unique_cats: std::collections::HashSet<&str> =
        categories.iter().map(|s| s.as_str()).collect();

    println!(
        "Corpus: {} — {} concepts across {} categories",
        corpus_label,
        n,
        unique_cats.len()
    );
    println!(
        "Budget: {} random trials (seed = 0x{:X})\n",
        RANDOM_BUDGET, RANDOM_SEED
    );
    let _ = corpus; // keep for future per-concept reporting; not used past this point

    let space = SearchSpace::default();
    // Small budget chosen deliberately so Bayesian's sample-efficiency
    // edge over Random has a chance to show up. At 24 random trials both
    // strategies tend to hit the same ceiling.
    let small_budget = 12;
    let warmup = 4;

    let kinds_str: Vec<&str> = space.projection_kinds.iter().map(|k| k.name()).collect();
    println!("Search space (discrete):");
    println!("  projection_kinds .............. {:?}", kinds_str);
    println!(
        "  laplacian_k_neighbors ......... {:?}",
        space.laplacian_k_neighbors
    );
    println!(
        "  laplacian_active_threshold .... {:?}",
        space.laplacian_active_threshold
    );
    println!(
        "  num_domain_groups ............. {:?}",
        space.num_domain_groups
    );
    println!(
        "  low_evr_threshold ............. {:?}",
        space.low_evr_threshold
    );
    println!(
        "  overlap_artifact_territorial .. {:?}",
        space.overlap_artifact_territorial
    );
    println!(
        "  threshold_base ................ {:?}",
        space.threshold_base
    );
    println!(
        "  threshold_evr_penalty ......... {:?}",
        space.threshold_evr_penalty
    );
    println!(
        "  min_evr_improvement ........... {:?}",
        space.min_evr_improvement
    );
    println!(
        "  grid cardinality .............. {}\n",
        space.grid_cardinality()
    );

    // Baseline: the default-config pipeline, built once and scored under
    // each metric below for the "lift over default" comparison.
    let baseline_pipeline = SphereQLPipeline::new(PipelineInput {
        categories: categories.clone(),
        embeddings: embeddings.clone(),
    })
    .expect("baseline pipeline build failed");

    // Default (silhouette-weighted) composite.
    let default_metric = CompositeMetric::default_composite();
    let default_baseline_score = default_metric.score(&baseline_pipeline);
    let (default_pipeline, default_report) =
        run_tune(&categories, &embeddings, &space, &default_metric);

    println!("\n================================================================");
    println!("  Metric 1: default_composite (silhouette 25%)");
    println!("================================================================");
    print_report(
        &default_pipeline,
        &default_report,
        &default_metric,
        default_baseline_score,
    );

    // Connectivity (modularity-weighted) composite.
    let conn_metric = CompositeMetric::connectivity_composite();
    let conn_baseline_score = conn_metric.score(&baseline_pipeline);
    let (conn_pipeline, conn_report) = run_tune(&categories, &embeddings, &space, &conn_metric);

    println!("\n================================================================");
    println!("  Metric 2: connectivity_composite (graph modularity 50%)");
    println!("================================================================");
    print_report(
        &conn_pipeline,
        &conn_report,
        &conn_metric,
        conn_baseline_score,
    );

    // Strategy comparison: at a small budget, does Bayesian beat Random
    // under the default composite? This tests the sample-efficiency claim.
    println!("\n================================================================");
    println!(
        "  Strategy comparison: Random vs Bayesian (budget = {})",
        small_budget
    );
    println!("================================================================\n");

    let (random_best, random_trials_to_best) = run_strategy(
        &categories,
        &embeddings,
        &space,
        &default_metric,
        SearchStrategy::Random {
            budget: small_budget,
            seed: RANDOM_SEED,
        },
    );
    let (bayes_best, bayes_trials_to_best) = run_strategy(
        &categories,
        &embeddings,
        &space,
        &default_metric,
        SearchStrategy::Bayesian {
            budget: small_budget,
            warmup,
            gamma: 0.25,
            seed: RANDOM_SEED,
        },
    );

    println!(
        "  {:<12}  {:<10}  {:<22}",
        "strategy", "best", "trials to best"
    );
    println!("  {}", "─".repeat(48));
    println!(
        "  {:<12}  {:<10.4}  {:<22}",
        "random", random_best, random_trials_to_best
    );
    println!(
        "  {:<12}  {:<10.4}  {:<22}",
        "bayesian", bayes_best, bayes_trials_to_best
    );
    if bayes_best > random_best {
        println!(
            "\n  → bayesian beat random by {:+.4} ({:+.1}%) at budget {}",
            bayes_best - random_best,
            100.0 * (bayes_best - random_best) / random_best.max(1e-12),
            small_budget
        );
    } else if (bayes_best - random_best).abs() < 1e-9 {
        println!("\n  → bayesian tied with random at this budget");
    } else {
        println!(
            "\n  → random beat bayesian by {:+.4} at budget {} (warmup may be too low)",
            random_best - bayes_best,
            small_budget
        );
    }

    // Side-by-side verdict.
    println!("\n================================================================");
    println!("  Head-to-head verdict");
    println!("================================================================\n");
    println!(
        "  {:<24}  {:<20}  {:<20}",
        "metric", "winning projection", "best score"
    );
    println!("  {}", "─".repeat(70));
    println!(
        "  {:<24}  {:<20}  {:<20.4}",
        default_metric.name(),
        default_pipeline.projection_kind().name(),
        default_report.best_score,
    );
    println!(
        "  {:<24}  {:<20}  {:<20.4}",
        conn_metric.name(),
        conn_pipeline.projection_kind().name(),
        conn_report.best_score,
    );

    let metric_flips_winner = default_pipeline.projection_kind() != conn_pipeline.projection_kind();
    println!();
    if metric_flips_winner {
        println!(
            "  → metric choice FLIPS the winner: {} under silhouette-weighted,",
            default_pipeline.projection_kind().name()
        );
        println!(
            "    {} under connectivity-weighted. The 'best projection'",
            conn_pipeline.projection_kind().name()
        );
        println!("    conclusion is metric-dependent on this corpus.");
    } else {
        println!(
            "  → {} wins under BOTH metrics. The ranking is robust to",
            default_pipeline.projection_kind().name()
        );
        println!("    the metric choice on this corpus.");
    }

    println!();
}

/// Run the tuner under a given strategy and return (best_score, trial
/// index at which that best was first seen). 1-based index: "reached in
/// 5 trials" means the best-seen score first appeared on trial #5.
fn run_strategy(
    categories: &[String],
    embeddings: &[Vec<f64>],
    space: &SearchSpace,
    metric: &impl QualityMetric,
    strategy: SearchStrategy,
) -> (f64, usize) {
    let input = PipelineInput {
        categories: categories.to_vec(),
        embeddings: embeddings.to_vec(),
    };
    let (_p, report) = auto_tune(input, space, metric, strategy, &PipelineConfig::default())
        .expect("auto_tune failed");
    let mut best = f64::NEG_INFINITY;
    let mut best_trial = 0usize;
    for (i, t) in report.trials.iter().enumerate() {
        if t.score > best {
            best = t.score;
            best_trial = i + 1;
        }
    }
    (best, best_trial)
}

fn run_tune(
    categories: &[String],
    embeddings: &[Vec<f64>],
    space: &SearchSpace,
    metric: &impl QualityMetric,
) -> (SphereQLPipeline, TuneReport) {
    let input = PipelineInput {
        categories: categories.to_vec(),
        embeddings: embeddings.to_vec(),
    };
    let start = std::time::Instant::now();
    let result = auto_tune(
        input,
        space,
        metric,
        SearchStrategy::Random {
            budget: RANDOM_BUDGET,
            seed: RANDOM_SEED,
        },
        &PipelineConfig::default(),
    )
    .expect("auto_tune failed");
    println!(
        "\n[{}] tuning complete in {:.2}s",
        metric.name(),
        start.elapsed().as_secs_f64()
    );
    result
}

fn print_report(
    best_pipeline: &SphereQLPipeline,
    report: &TuneReport,
    composite: &CompositeMetric,
    baseline_score: f64,
) {
    println!("\nScore summary (under {}):", report.metric_name);
    println!("  default config ......... {:.4}", baseline_score);
    println!("  mean across trials ..... {:.4}", report.mean_score());
    println!("  best (found) ........... {:.4}", report.best_score);
    let lift = report.best_score - baseline_score;
    println!(
        "  lift over default ...... {:+.4}  ({:+.1}%)\n",
        lift,
        100.0 * lift / baseline_score.max(1e-12)
    );

    println!(
        "Winning projection: {}",
        best_pipeline.projection_kind().name()
    );
    if best_pipeline.projection_kind() == ProjectionKind::LaplacianEigenmap {
        let lc = &best_pipeline.config().laplacian;
        println!("  laplacian_k_neighbors ....... {}", lc.k_neighbors);
        println!("  laplacian_active_threshold .. {:.3}", lc.active_threshold);
    }

    println!("\nTop 5 trials (by score):");
    println!(
        "  {:>4}  {:>6}  {:>6}  {:>18}  {:>8}  {:>10}  {:>10}",
        "rank", "score", "build", "projection", "groups", "low_evr", "overlap"
    );
    println!("  {}", "─".repeat(82));
    for (rank, t) in report.ranked_trials().iter().take(5).enumerate() {
        println!(
            "  {:>4}  {:>6.4}  {:>4}ms  {:>18}  {:>8}  {:>10.2}  {:>10.2}",
            rank + 1,
            t.score,
            t.build_ms,
            t.config.projection_kind.name(),
            t.config.routing.num_domain_groups,
            t.config.routing.low_evr_threshold,
            t.config.bridges.overlap_artifact_territorial,
        );
    }

    use std::collections::BTreeMap;
    let mut per_kind: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
    for t in &report.trials {
        let entry =
            per_kind
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
    for (name, weight, score) in composite.score_components(best_pipeline) {
        let bar_len = (score * 30.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(30 - bar_len);
        println!(
            "  {:<24} w={:.2}  score={:.4}  [{}]",
            name, weight, score, bar
        );
    }

    let layer = best_pipeline.category_layer();
    let (mut genuine, mut artifact, mut weak) = (0usize, 0usize, 0usize);
    for bridges in layer.graph.bridges.values() {
        for b in bridges {
            match b.classification {
                BridgeClassification::Genuine => genuine += 1,
                BridgeClassification::OverlapArtifact => artifact += 1,
                BridgeClassification::Weak => weak += 1,
            }
        }
    }
    let total = genuine + artifact + weak;
    println!(
        "\nBridge classification on winning pipeline ({} bridges):",
        total
    );
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
}
