#![allow(clippy::uninlined_format_args)]
//! Demonstrates the L3 feedback primitives end-to-end.
//!
//! Flow:
//!   1. Build a tuned pipeline from a mock training record.
//!   2. Simulate user query outcomes as FeedbackEvent rows.
//!   3. Aggregate events in a FeedbackAggregator, save/reload from disk.
//!   4. Summarize feedback per corpus and blend it into the record's
//!      best_score via MetaTrainingRecord::adjust_score_with_feedback
//!      at three different alpha weights.
//!
//! Shows how the feedback primitives compose — the meta-model isn't
//! retrained here; users call `adjust_score_with_feedback` when
//! ranking candidate records to decide which stored configuration to
//! trust on a new corpus.
//!
//! Writes to `./target/feedback_events.json` (scratch, not the user's
//! default `~/.sphereql/` store).
//!
//! Run with:
//!   cargo run --example meta_feedback --features embed --release

use sphereql::embed::{
    CorpusFeatures, FeedbackAggregator, FeedbackEvent, MetaTrainingRecord, PipelineConfig,
    ProjectionKind,
};

fn main() {
    println!("================================================================");
    println!("  SphereQL L3 feedback primitives");
    println!("================================================================\n");

    let corpus_id = "demo_corpus_v1";

    // ── 1. Mock training record (normally produced by auto_tune) ─────
    let mut cfg = PipelineConfig::default();
    cfg.projection_kind = ProjectionKind::Pca;
    let record = MetaTrainingRecord {
        corpus_id: corpus_id.to_string(),
        features: mock_features(),
        best_config: cfg,
        best_score: 0.72, // automated quality score from the tuner
        metric_name: "default_composite".to_string(),
        strategy: "random{budget=24}".to_string(),
        timestamp: "0".to_string(),
    };
    println!(
        "Stored record for '{}': automated best_score = {:.3}",
        record.corpus_id, record.best_score
    );

    // ── 2. Simulate 8 user feedback events ───────────────────────────
    // User satisfaction trends low — feedback mean < automated score,
    // so blending will pull the adjusted score downward. In a real
    // system these come from CTR, star ratings, etc.
    let scores = [0.8, 0.4, 0.3, 0.5, 0.9, 0.2, 0.4, 0.6];
    let mut agg = FeedbackAggregator::new();
    for (i, &s) in scores.iter().enumerate() {
        agg.record(FeedbackEvent {
            corpus_id: corpus_id.to_string(),
            query_id: format!("q_{:03}", i),
            score: s,
            timestamp: i.to_string(),
        });
    }
    println!(
        "\nRecorded {} feedback events (raw scores {:?})",
        agg.len(),
        scores
    );

    // ── 3. Save + reload from disk to exercise persistence ───────────
    let store = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("target")
        .join("feedback_events.json");
    agg.save(&store).expect("save failed");
    let reloaded = FeedbackAggregator::load(&store).expect("load failed");
    assert_eq!(reloaded.len(), agg.len());
    println!("Round-tripped event log to {}", store.display());
    println!(
        "  (production: FeedbackEvent::append_to_default_store() writes to {})",
        FeedbackAggregator::default_store_path().unwrap().display()
    );

    // ── 4. Summarize + blend at three alphas ─────────────────────────
    let summary = reloaded.summarize(corpus_id).expect("no feedback");
    println!(
        "\nFeedback summary for '{}': n={}  mean={:.3}  min={:.2}  max={:.2}",
        summary.corpus_id,
        summary.n_events,
        summary.mean_score,
        summary.min_score,
        summary.max_score,
    );

    println!(
        "\nBlending adjusted score (best_score=0.720, feedback mean={:.3}):",
        summary.mean_score
    );
    println!("  {:<8} {:<10} {}", "alpha", "adjusted", "interpretation");
    println!("  {}", "-".repeat(60));
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let adjusted = record.adjust_score_with_feedback(&summary, alpha);
        let interp = match alpha {
            x if x < 0.05 => "ignore feedback (pure tuner score)",
            x if x < 0.4 => "trust tuner > users",
            x if x < 0.6 => "equal weight",
            x if x < 0.95 => "trust users > tuner",
            _ => "replace tuner with feedback",
        };
        println!("  {:<8.2} {:<10.3} {}", alpha, adjusted, interp);
    }

    println!("\nTakeaway: the feedback-adjusted score is a caller-controlled");
    println!("blend. Ranking candidate records by adjusted score (rather than");
    println!("raw best_score) lets user satisfaction override automated quality");
    println!("signals when the tuner's metric diverges from real-world use.");
}

fn mock_features() -> CorpusFeatures {
    // Plausible values for a ~500-concept, 20-category corpus. The
    // specific values don't affect the feedback demo — we just need a
    // valid CorpusFeatures for the MetaTrainingRecord.
    CorpusFeatures {
        n_items: 500,
        n_categories: 20,
        dim: 128,
        mean_members_per_category: 25.0,
        category_size_entropy: 0.95,
        mean_sparsity: 0.05,
        axis_utilization_entropy: 0.88,
        noise_estimate: 0.015,
        mean_intra_category_similarity: 0.45,
        mean_inter_category_similarity: 0.10,
        category_separation_ratio: 4.5,
    }
}
