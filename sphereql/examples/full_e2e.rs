#![allow(clippy::uninlined_format_args)]
//! Full End-to-End SphereQL Demo
//!
//! A single example that exercises every major capability of the system:
//!
//!   Phase 1 — Auto-tuning: optimize the pipeline config against a
//!             quality metric using random search.
//!   Phase 2 — Meta-learning & feedback: extract corpus features, build
//!             a meta-model from tuning results, simulate user feedback,
//!             and blend it into the automated score.
//!   Phase 3 — Embedding & projection: inspect how high-D embeddings
//!             land on the sphere, with certainty and intensity metadata.
//!   Phase 4 — Spatial analysis: full navigator report — antipodal
//!             discovery, coverage, geodesic sweeps, Voronoi tessellation,
//!             overlap, curvature, lune decomposition.
//!   Phase 5 — Category analysis: landscape, paths, bridges, domain
//!             groups, inner spheres.
//!   Phase 6 — Queries: nearest neighbor, drill-down, concept paths,
//!             glob detection, hierarchical routing.
//!   Phase 7 — AI-enhanced divergence: knowledge gap cartography,
//!             bridge classification, geodesic deviation, gap confidence,
//!             and a synthesized cross-domain reasoning chain.
//!
//! Run with:
//!   cargo run --example full_e2e --features embed --release

use sphereql::core::spatial::*;
use sphereql::core::SphericalPoint;
use sphereql::embed::{
    BridgeClassification, CompositeMetric, CorpusFeatures, DistanceWeightedMetaModel, Embedding,
    FeedbackAggregator, FeedbackEvent, MetaModel, MetaTrainingRecord, NavigatorConfig,
    NearestNeighborMetaModel, PipelineConfig, PipelineInput, PipelineQuery, Projection,
    QualityMetric, SearchSpace, SearchStrategy, SphereQLOutput, SphereQLQuery, auto_tune,
    category_geodesic_sweep, category_path_deviation, gap_confidence, run_full_analysis,
};
use sphereql_corpus::axes::*;
use sphereql_corpus::{DIM, build_corpus, embed};

fn parse_item_index(id: &str) -> usize {
    id.strip_prefix("s-")
        .and_then(|s| s.parse().ok())
        .expect("invalid item ID format")
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      SphereQL — Full End-to-End Demo                       ║");
    println!("║      Auto-tune → Meta-learn → Embed → Analyze → Query     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BUILD CORPUS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let corpus = build_corpus();
    let n = corpus.len();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();
    let labels: Vec<&str> = corpus.iter().map(|c| c.label).collect();

    let unique_cats: std::collections::HashSet<&str> =
        categories.iter().map(|s| s.as_str()).collect();
    println!(
        "Corpus: {} concepts across {} categories (dim={})\n",
        n,
        unique_cats.len(),
        DIM
    );

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 1: AUTO-TUNING
    // ════════════════════════════════════════════════════════════════════════

    println!("================================================================");
    println!("  PHASE 1: AUTO-TUNING");
    println!("  Optimize pipeline parameters against a quality metric");
    println!("================================================================\n");

    let space = SearchSpace::default();
    let metric = CompositeMetric::default_composite();
    let budget = 16;
    let seed = 0x0A17_CABE_CAFE_u64;

    let kinds_str: Vec<&str> = space.projection_kinds.iter().map(|k| k.name()).collect();
    println!("Search space:");
    println!("  projection_kinds ........ {:?}", kinds_str);
    println!(
        "  grid cardinality ........ {} configs",
        space.grid_cardinality()
    );
    println!("  strategy ................ Random(budget={}, seed=0x{:X})", budget, seed);
    println!("  metric .................. {}\n", metric.name());

    let start = std::time::Instant::now();
    let (pipeline, tune_report) = auto_tune(
        PipelineInput {
            categories: categories.clone(),
            embeddings: embeddings.clone(),
        },
        &space,
        &metric,
        SearchStrategy::Random { budget, seed },
        &PipelineConfig::default(),
    )
    .expect("auto_tune failed");
    let tune_elapsed = start.elapsed();

    let evr = pipeline.explained_variance_ratio();
    println!(
        "Tuning complete in {:.2}s — {} trials evaluated",
        tune_elapsed.as_secs_f64(),
        tune_report.trials.len()
    );
    println!("  Best score:   {:.4}", tune_report.best_score);
    println!("  Mean score:   {:.4}", tune_report.mean_score());
    println!(
        "  Projection:   {}",
        pipeline.projection_kind().name()
    );
    println!(
        "  EVR:          {:.1}% variance explained\n",
        evr * 100.0
    );

    println!("Top 5 trials:");
    println!(
        "  {:>4}  {:>7}  {:>6}  {:>18}  {:>7}  {:>8}",
        "rank", "score", "ms", "projection", "groups", "low_evr"
    );
    println!("  {}", "─".repeat(56));
    for (rank, t) in tune_report.ranked_trials().iter().take(5).enumerate() {
        println!(
            "  {:>4}  {:>7.4}  {:>4}ms  {:>18}  {:>7}  {:>8.2}",
            rank + 1,
            t.score,
            t.build_ms,
            t.config.projection_kind.name(),
            t.config.routing.num_domain_groups,
            t.config.routing.low_evr_threshold,
        );
    }

    println!("\nComponent scores on winning config:");
    for (name, weight, score) in metric.score_components(&pipeline) {
        let bar_len = ((score * 25.0) as usize).min(25);
        let bar = "█".repeat(bar_len) + &"░".repeat(25 - bar_len);
        println!(
            "  {:<24} w={:.2}  score={:.4}  [{}]",
            name, weight, score, bar
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 2: META-LEARNING & FEEDBACK
    // ════════════════════════════════════════════════════════════════════════

    println!("\n================================================================");
    println!("  PHASE 2: META-LEARNING & FEEDBACK");
    println!("  Extract corpus profile → build meta-model → simulate feedback");
    println!("================================================================\n");

    // 2a. Extract corpus features and create a training record
    let features = CorpusFeatures::extract(&categories, &embeddings);
    println!("Corpus feature profile (10-D):");
    println!("  n_items ..................... {}", features.n_items);
    println!("  n_categories ................ {}", features.n_categories);
    println!("  dim ......................... {}", features.dim);
    println!(
        "  mean_members_per_category ... {:.1}",
        features.mean_members_per_category
    );
    println!(
        "  category_size_entropy ....... {:.4}",
        features.category_size_entropy
    );
    println!(
        "  mean_sparsity ............... {:.4}",
        features.mean_sparsity
    );
    println!(
        "  noise_estimate .............. {:.4}",
        features.noise_estimate
    );
    println!(
        "  intra-category similarity ... {:.4}",
        features.mean_intra_category_similarity
    );
    println!(
        "  inter-category similarity ... {:.4}",
        features.mean_inter_category_similarity
    );
    println!(
        "  category_separation_ratio ... {:.2}",
        features.category_separation_ratio
    );

    let record = MetaTrainingRecord::from_tune_result(
        "demo_e2e",
        features.clone(),
        &tune_report,
        format!("random{{budget={}}}", budget),
    );

    // 2b. Fit meta-models on this single record (in practice you'd have many)
    let records = vec![record.clone()];
    let mut nn_model = NearestNeighborMetaModel::new();
    nn_model.fit(&records);
    let mut dw_model = DistanceWeightedMetaModel::new();
    dw_model.fit(&records);

    let nn_pred = nn_model.predict(&features);
    let dw_pred = dw_model.predict(&features);
    println!(
        "\nMeta-model predictions (self-query, should match tuner):"
    );
    println!(
        "  {} → {}",
        nn_model.name(),
        nn_pred.projection_kind.name(),
    );
    println!(
        "  {} → {}",
        dw_model.name(),
        dw_pred.projection_kind.name(),
    );
    let matches = nn_pred.projection_kind == pipeline.projection_kind();
    println!(
        "  Match tuner winner ({}): {}",
        pipeline.projection_kind().name(),
        if matches { "YES" } else { "no" }
    );

    // 2c. Simulate user feedback and blend
    let feedback_scores = [0.85, 0.72, 0.90, 0.60, 0.78, 0.95, 0.55, 0.80];
    let mut agg = FeedbackAggregator::new();
    for (i, &s) in feedback_scores.iter().enumerate() {
        agg.record(FeedbackEvent {
            corpus_id: "demo_e2e".to_string(),
            query_id: format!("q_{:03}", i),
            score: s,
            timestamp: i.to_string(),
        });
    }
    let summary = agg.summarize("demo_e2e").expect("no feedback");
    println!(
        "\nUser feedback: {} events, mean={:.3}, min={:.2}, max={:.2}",
        summary.n_events, summary.mean_score, summary.min_score, summary.max_score,
    );

    println!("\nScore blending (automated={:.3}, user_mean={:.3}):", record.best_score, summary.mean_score);
    println!("  {:<8} {:<10} interpretation", "alpha", "blended");
    println!("  {}", "─".repeat(50));
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let blended = record.adjust_score_with_feedback(&summary, alpha);
        let interp = match alpha {
            x if x < 0.05 => "pure tuner",
            x if x < 0.4 => "tuner-heavy",
            x if x < 0.6 => "equal weight",
            x if x < 0.95 => "user-heavy",
            _ => "pure user feedback",
        };
        println!("  {:<8.2} {:<10.4} {}", alpha, blended, interp);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 3: EMBEDDING & PROJECTION
    // ════════════════════════════════════════════════════════════════════════

    println!("\n================================================================");
    println!("  PHASE 3: EMBEDDING & PROJECTION");
    println!("  How high-D vectors land on the sphere");
    println!("================================================================\n");

    let exported = pipeline.exported_points();
    let positions: Vec<SphericalPoint> = exported
        .iter()
        .map(|p| SphericalPoint::new_unchecked(p.r, p.theta, p.phi))
        .collect();

    // Show a sample of projected points across different categories
    let sample_cats = ["physics", "music", "computer_science", "biology", "philosophy"];
    println!(
        "  {:<30} {:<18} {:>7} {:>7} {:>9} {:>9}",
        "Concept", "Category", "θ (°)", "φ (°)", "Certainty", "Intensity"
    );
    println!("  {}", "─".repeat(85));

    let layer = pipeline.category_layer();
    for &cat in &sample_cats {
        if let Some(s) = layer.summaries.iter().find(|s| s.name == cat) {
            if let Some(&idx) = s.member_indices.first() {
                let p = &exported[idx];
                println!(
                    "  {:<30} {:<18} {:>7.1} {:>7.1} {:>9.4} {:>9.4}",
                    labels[idx],
                    cat,
                    p.theta.to_degrees(),
                    p.phi.to_degrees(),
                    p.certainty,
                    p.intensity,
                );
            }
        }
    }

    // Project a fresh query embedding to show the Projection trait in action
    let fresh_query_features = vec![
        (QUANTUM, 0.7),
        (COMPUTATION, 0.6),
        (INFORMATION, 0.5),
        (MATH, 0.4),
        (LOGIC, 0.3),
    ];
    let fresh_vec = embed(&fresh_query_features, 42);
    let fresh_emb = Embedding::new(fresh_vec.clone());
    let projected = pipeline.projection().project_rich(&fresh_emb);
    println!(
        "\n  Fresh query (quantum computing):"
    );
    println!(
        "    Projected to: θ={:.1}°, φ={:.1}°, r={:.4}",
        projected.position.theta.to_degrees(),
        projected.position.phi.to_degrees(),
        projected.position.r,
    );
    println!("    Certainty: {:.4}", projected.certainty);
    println!("    Intensity: {:.4}", projected.intensity);
    println!(
        "    Projection magnitude: {:.4}",
        projected.projection_magnitude
    );

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 4: SPATIAL ANALYSIS
    // ════════════════════════════════════════════════════════════════════════

    println!("\n================================================================");
    println!("  PHASE 4: SPATIAL ANALYSIS ON S²");
    println!("  Geometry that only works on a sphere");
    println!("================================================================");

    let config = NavigatorConfig::default();
    let report = run_full_analysis(layer, &positions, &categories, evr, &config);

    // §4a: Antipodal discovery
    println!("\n  ── §4a. ANTIPODAL DISCOVERY ─────────────────────────────────");
    println!("  What is semantically OPPOSITE to each domain?\n");

    println!(
        "  {:<20} {:>10} {:>12} {:>20}",
        "Category", "Coherence", "Items@Anti", "Dominant@Anti"
    );
    println!("  {}", "─".repeat(65));
    for ar in report.antipodal.iter().take(10) {
        let dominant = ar
            .dominant_antipodal_category
            .as_deref()
            .unwrap_or("(void)");
        println!(
            "  {:<20} {:>10.3} {:>12} {:>20}",
            ar.category_name,
            ar.antipodal_coherence,
            ar.antipodal_items.len(),
            dominant,
        );
    }

    // §4b: Coverage & voids
    println!("\n  ── §4b. KNOWLEDGE COVERAGE ──────────────────────────────────");
    let cov = &report.coverage;
    println!(
        "  Coverage: {:.1}% of S² is claimed by at least one category",
        cov.coverage_fraction * 100.0
    );
    println!(
        "  Overlap area: {:.3} sr (multi-claimed territory)",
        cov.overlap_area
    );
    println!(
        "  Void: {:.1}% of the sphere is unmapped knowledge space\n",
        (1.0 - cov.coverage_fraction) * 100.0
    );

    // §4c: Geodesic sweeps
    println!("  ── §4c. GEODESIC SWEEPS ────────────────────────────────────");
    println!("  What lies between two ideas on the great circle?\n");

    let sweep_pairs = [
        ("physics", "music"),
        ("computer_science", "philosophy"),
        ("biology", "economics"),
    ];
    for (src, tgt) in &sweep_pairs {
        if let Some(sweep) = category_geodesic_sweep(
            layer,
            src,
            tgt,
            &positions,
            &categories,
            config.geodesic_epsilon,
            config.density_bins,
        ) {
            let max_d = *sweep.density_profile.iter().max().unwrap_or(&1).max(&1);
            let spark: String = sweep
                .density_profile
                .iter()
                .map(|&d| {
                    let level = (d as f64 / max_d as f64 * 7.0) as usize;
                    ['░', '░', '▒', '▒', '▓', '▓', '█', '█'][level.min(7)]
                })
                .collect();

            println!(
                "  {} → {} : {} items on arc, gap={:.0}%  [{}]",
                src,
                tgt,
                sweep.items.len(),
                sweep.gap_fraction * 100.0,
                spark,
            );
        }
    }

    // §4d: Voronoi tessellation (top 5 by area)
    println!("\n  ── §4d. VORONOI TESSELLATION ────────────────────────────────");
    println!("  How much spherical real-estate does each domain own?\n");

    let vor = &report.voronoi;
    let mut cells = vor.cells.clone();
    cells.sort_by(|a, b| b.cell_area.total_cmp(&a.cell_area));
    println!(
        "  {:<20} {:>8} {:>6} {:>10}",
        "Category", "Area sr", "Items", "Efficiency"
    );
    println!("  {}", "─".repeat(48));
    for cell in cells.iter().take(8) {
        println!(
            "  {:<20} {:>8.3} {:>6} {:>10.2}",
            cell.category_name, cell.cell_area, cell.item_count, cell.territorial_efficiency
        );
    }
    if cells.len() > 8 {
        println!("  ... and {} more categories", cells.len() - 8);
    }

    // §4e: Curvature
    println!("\n  ── §4e. CURVATURE SIGNATURES ───────────────────────────────");
    println!("  Spherical excess = 0 in flat space; nonzero = genuine curvature\n");

    let curv = &report.curvature;
    for triple in curv.top_triples.iter().take(5) {
        let name = format!(
            "{} × {} × {}",
            triple.categories[0], triple.categories[1], triple.categories[2]
        );
        println!("  {:<50} excess = {:.6} sr", name, triple.excess);
    }

    // §4f: Lune decomposition
    println!("\n  ── §4f. LUNE DECOMPOSITION ─────────────────────────────────");
    println!("  Do bridge concepts lean toward one domain or the other?\n");

    let mut lunes = report.lunes.clone();
    lunes.sort_by(|a, b| b.asymmetry.total_cmp(&a.asymmetry));
    println!(
        "  {:<35} {:>4} {:>4} {:>4} {:>8}",
        "Pair", "→A", "→B", "=", "Asym"
    );
    println!("  {}", "─".repeat(58));
    for lune in lunes.iter().take(8) {
        let pair = format!("{} ↔ {}", lune.category_a, lune.category_b);
        println!(
            "  {:<35} {:>4} {:>4} {:>4} {:>8.3}",
            pair,
            lune.a_leaning_count,
            lune.b_leaning_count,
            lune.on_bisector_count,
            lune.asymmetry,
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 5: CATEGORY ANALYSIS
    // ════════════════════════════════════════════════════════════════════════

    println!("\n================================================================");
    println!("  PHASE 5: CATEGORY ANALYSIS");
    println!("  Landscape, paths, bridges, domain groups");
    println!("================================================================");

    // 5a: Category landscape
    println!("\n  ── §5a. CATEGORY LANDSCAPE ─────────────────────────────────\n");
    let mut sorted_summaries: Vec<&_> = layer.summaries.iter().collect();
    sorted_summaries.sort_by(|a, b| b.cohesion.total_cmp(&a.cohesion));

    println!(
        "  {:<22} {:>5} {:>10} {:>8} {:>9}",
        "Domain", "Items", "Spread(°)", "Cohesion", "BrQual"
    );
    println!("  {}", "─".repeat(58));
    for s in sorted_summaries.iter().take(15) {
        println!(
            "  {:<22} {:>5} {:>10.2} {:>8.4} {:>9.4}",
            s.name, s.member_count, s.angular_spread.to_degrees(), s.cohesion, s.bridge_quality,
        );
    }
    if sorted_summaries.len() > 15 {
        println!("  ... and {} more", sorted_summaries.len() - 15);
    }

    // 5b: Cross-domain concept paths
    println!("\n  ── §5b. CROSS-DOMAIN CONCEPT PATHS ────────────────────────");

    let path_queries = [
        ("nanotechnology", "economics", "How does nanotech impact the economy?"),
        ("music", "biology", "Musical patterns in living systems?"),
        ("philosophy", "computer_science", "From abstract thought to computation"),
    ];

    for (src, tgt, question) in &path_queries {
        println!("\n  Q: \"{}\"", question);
        if let Some(path) = pipeline.category_path(src, tgt) {
            let chain: Vec<&str> = path
                .steps
                .iter()
                .map(|s| s.category_name.as_str())
                .collect();
            println!("  Route: {}", chain.join(" → "));
            println!(
                "  Distance: {:.3} ({:.1}°)",
                path.total_distance,
                path.total_distance.to_degrees()
            );

            for (i, step) in path.steps.iter().enumerate() {
                if i + 1 < path.steps.len() {
                    let bridge_labels: Vec<String> = step
                        .bridges_to_next
                        .iter()
                        .take(2)
                        .map(|b| format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength))
                        .collect();
                    if !bridge_labels.is_empty() {
                        println!("    {} bridged by: {}", step.category_name, bridge_labels.join(", "));
                    }
                }
            }
        } else {
            println!("  (no path found)");
        }
    }

    // 5c: Bridge items between select pairs
    println!("\n  ── §5c. BRIDGE CONCEPTS ───────────────────────────────────\n");
    let bridge_pairs = [
        ("physics", "computer_science"),
        ("biology", "philosophy"),
        ("music", "psychology"),
        ("nanotechnology", "medicine"),
    ];
    for (src, tgt) in &bridge_pairs {
        let bridges = pipeline.bridge_items(src, tgt, 3);
        let rev = pipeline.bridge_items(tgt, src, 3);
        let all: Vec<_> = bridges.iter().chain(rev.iter()).take(3).collect();
        if all.is_empty() {
            println!("  {} ↔ {}: (no bridges — conceptual gap)", src, tgt);
        } else {
            print!("  {} ↔ {}: ", src, tgt);
            let desc: Vec<String> = all
                .iter()
                .map(|b| format!("\"{}\" (s={:.3}, {:?})", labels[b.item_index], b.bridge_strength, b.classification))
                .collect();
            println!("{}", desc.join(", "));
        }
    }

    // 5d: Domain groups
    println!("\n  ── §5d. HIERARCHICAL DOMAIN GROUPS ────────────────────────\n");
    let groups = pipeline.domain_groups();
    println!(
        "  {:<4} {:>5} {:>6} {:>10} {:>9}  members",
        "Grp", "#cat", "items", "cohesion", "spread°"
    );
    println!("  {}", "─".repeat(70));
    for (gi, g) in groups.iter().enumerate() {
        println!(
            "  {:<4} {:>5} {:>6} {:>10.3} {:>9.2}  {}",
            gi,
            g.member_categories.len(),
            g.total_items,
            g.cohesion,
            g.angular_spread.to_degrees(),
            g.category_names.join(", ")
        );
    }

    // 5e: Inner sphere stats
    let stats = pipeline.inner_sphere_stats();
    if !stats.is_empty() {
        println!("\n  ── §5e. INNER SPHERE STATUS ──────────────────────────────\n");
        println!(
            "  {:<22} {:>6} {:>10} {:>10} {:>10}",
            "Domain", "Items", "Inner EVR", "Global EVR", "Improve"
        );
        println!("  {}", "─".repeat(62));
        for s in &stats {
            println!(
                "  {:<22} {:>6} {:>10.4} {:>10.4} {:>10.4}",
                s.category_name, s.member_count, s.inner_evr, s.global_subset_evr, s.evr_improvement,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 6: QUERIES
    // ════════════════════════════════════════════════════════════════════════

    println!("\n================================================================");
    println!("  PHASE 6: QUERIES");
    println!("  Nearest neighbor, drill-down, concept paths, globs, routing");
    println!("================================================================");

    let qvec = embed(
        &[
            (QUANTUM, 0.8),
            (COMPUTATION, 0.7),
            (MATH, 0.5),
            (INFORMATION, 0.4),
        ],
        9999,
    );
    let pq = PipelineQuery {
        embedding: qvec.clone(),
    };

    // 6a: Nearest neighbor
    println!("\n  ── §6a. NEAREST NEIGHBOR ──────────────────────────────────");
    println!("  Query: \"quantum computation\" — top 8\n");

    let nn_result = pipeline
        .query(SphereQLQuery::Nearest { k: 8 }, &pq)
        .expect("nearest query");
    if let SphereQLOutput::Nearest(results) = nn_result {
        println!(
            "  {:<4} {:<30} {:<18} {:>7} {:>9}",
            "#", "Concept", "Category", "Dist°", "Certainty"
        );
        println!("  {}", "─".repeat(72));
        for (i, r) in results.iter().enumerate() {
            let idx = parse_item_index(&r.id);
            println!(
                "  {:<4} {:<30} {:<18} {:>7.2} {:>9.4}",
                i + 1,
                labels[idx],
                r.category,
                r.distance.to_degrees(),
                r.certainty,
            );
        }
    }

    // 6b: Drill-down
    println!("\n  ── §6b. DRILL-DOWN ───────────────────────────────────────");
    for cat in ["physics", "computer_science", "mathematics"] {
        let drill = pipeline
            .query(SphereQLQuery::DrillDown { category: cat, k: 3 }, &pq)
            .ok();
        if let Some(SphereQLOutput::DrillDown(results)) = drill {
            let items: Vec<String> = results
                .iter()
                .map(|r| {
                    let sphere = if r.used_inner_sphere { "inner" } else { "outer" };
                    format!("\"{}\" ({}, d={:.4})", labels[r.item_index], sphere, r.distance)
                })
                .collect();
            println!("  {:<20} → {}", cat, items.join(", "));
        }
    }

    // 6c: Item-level concept path
    println!("\n  ── §6c. ITEM-LEVEL CONCEPT PATH ──────────────────────────");
    let dummy_q = PipelineQuery {
        embedding: vec![0.0; DIM],
    };

    let src_id = "s-0010";
    let tgt_id = "s-0125";
    println!("  Path: {} → {}", src_id, tgt_id);
    let cp_result = pipeline
        .query(
            SphereQLQuery::ConceptPath {
                source_id: src_id,
                target_id: tgt_id,
                graph_k: 8,
            },
            &dummy_q,
        )
        .expect("concept_path query");
    if let SphereQLOutput::ConceptPath(Some(path)) = cp_result {
        for (i, step) in path.steps.iter().enumerate() {
            let item_idx = parse_item_index(&step.id);
            println!(
                "    [{}] \"{}\" [{}] (cum={:.4})",
                i + 1,
                labels[item_idx],
                step.category,
                step.cumulative_distance,
            );
        }
        println!(
            "  Total: {:.4} ({} hops)\n",
            path.total_distance,
            path.steps.len() - 1
        );
    }

    // 6d: Glob detection
    println!("  ── §6d. KNOWLEDGE DENSITY — Glob Detection ────────────────\n");
    let glob_result = pipeline
        .query(SphereQLQuery::DetectGlobs { k: None, max_k: 10 }, &dummy_q)
        .expect("detect_globs query");
    if let SphereQLOutput::Globs(globs) = glob_result {
        println!("  {} knowledge clusters detected:", globs.len());
        for g in globs.iter().take(6) {
            let cats: Vec<String> = g
                .top_categories
                .iter()
                .take(3)
                .map(|(c, n)| format!("{} ({})", c, n))
                .collect();
            println!(
                "    Glob {}: {} items, radius={:.3} rad — {}",
                g.id,
                g.member_count,
                g.radius,
                cats.join(", ")
            );
        }
    }

    // 6e: Hierarchical routing
    println!("\n  ── §6e. HIERARCHICAL ROUTING ─────────────────────────────");
    println!(
        "  EVR={:.3} (hierarchical activates below 0.35)\n",
        evr
    );

    let route_queries = [
        ("quantum computing", vec![(QUANTUM, 0.8), (COMPUTATION, 0.7), (MATH, 0.5)]),
        ("musical cognition", vec![(SOUND, 0.7), (CONSCIOUSNESS, 0.6), (EMOTION, 0.5)]),
        ("legal ethics of AI", vec![(LEGAL, 0.7), (ETHICS, 0.8), (AI, 0.5)]),
    ];

    for (desc, feats) in &route_queries {
        let qvec = embed(feats, 42 + desc.len() as u64);
        let emb = Embedding::new(qvec);

        if let Some(group) = pipeline.route_to_group(&emb) {
            println!(
                "  \"{}\": group [{}]",
                desc,
                group.category_names.join(", ")
            );
        }

        let hier = pipeline.hierarchical_nearest(&emb, 3);
        let items: Vec<String> = hier
            .iter()
            .map(|r| {
                let idx = parse_item_index(&r.id);
                format!("\"{}\" [{}]", labels[idx], r.category)
            })
            .collect();
        println!("    hierarchical top 3: {}", items.join(", "));
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PHASE 7: AI-ENHANCED DIVERGENCE & DISJUNCTION
    // ════════════════════════════════════════════════════════════════════════

    println!("\n================================================================");
    println!("  PHASE 7: AI-ENHANCED DIVERGENCE");
    println!("  Knowledge gaps, bridge classification, geodesic deviation,");
    println!("  gap confidence, and a synthesized reasoning chain");
    println!("================================================================");

    // 7a: Knowledge gap cartography — gap_confidence at strategic points
    println!("\n  ── §7a. KNOWLEDGE GAP CARTOGRAPHY ─────────────────────────");
    println!("  Where on the sphere does knowledge run out?\n");

    let test_points = [
        (
            "At physics centroid",
            layer.summaries.iter().find(|s| s.name == "physics").unwrap().centroid_position,
        ),
        (
            "At music centroid",
            layer.summaries.iter().find(|s| s.name == "music").unwrap().centroid_position,
        ),
        (
            "North pole (void?)",
            SphericalPoint::new_unchecked(1.0, 0.0, 0.01),
        ),
        (
            "South pole (void?)",
            SphericalPoint::new_unchecked(1.0, 0.0, std::f64::consts::PI - 0.01),
        ),
        (
            "Physics antipode",
            antipode(&layer.summaries.iter().find(|s| s.name == "physics").unwrap().centroid_position),
        ),
    ];

    println!(
        "  {:<25} {:>12} interpretation",
        "Location", "Confidence"
    );
    println!("  {}", "─".repeat(55));
    for (name, pt) in &test_points {
        let conf = gap_confidence(pt, layer, config.gap_sharpness);
        let interp = if conf > 0.8 {
            "well-mapped territory"
        } else if conf > 0.4 {
            "boundary zone"
        } else if conf > 0.1 {
            "sparse frontier"
        } else {
            "KNOWLEDGE VOID"
        };
        println!("  {:<25} {:>12.4} {}", name, conf, interp);
    }

    // 7b: Bridge classification census
    println!("\n  ── §7b. BRIDGE CLASSIFICATION CENSUS ─────────────────────\n");
    let mut genuine = 0usize;
    let mut artifact = 0usize;
    let mut weak = 0usize;
    for bridges in layer.graph.bridges.values() {
        for b in bridges {
            match b.classification {
                BridgeClassification::Genuine => genuine += 1,
                BridgeClassification::OverlapArtifact => artifact += 1,
                BridgeClassification::Weak => weak += 1,
            }
        }
    }
    let total_bridges = genuine + artifact + weak;
    println!("  Total bridges: {}", total_bridges);
    if total_bridges > 0 {
        println!(
            "    Genuine         : {:>5}  ({:>5.1}%) — real cross-domain connectors",
            genuine,
            100.0 * genuine as f64 / total_bridges as f64
        );
        println!(
            "    Weak            : {:>5}  ({:>5.1}%) — low affinity, noisy",
            weak,
            100.0 * weak as f64 / total_bridges as f64
        );
        println!(
            "    OverlapArtifact : {:>5}  ({:>5.1}%) — shared territory, not real links",
            artifact,
            100.0 * artifact as f64 / total_bridges as f64
        );
    }

    // 7c: Geodesic deviation — how much do graph paths stray from direct arcs?
    println!("\n  ── §7c. GEODESIC DEVIATION ───────────────────────────────");
    println!("  Graph path vs. direct great-circle arc (0 = perfect alignment)\n");

    let dev_pairs = [
        ("physics", "music"),
        ("computer_science", "religion"),
        ("biology", "law"),
        ("nanotechnology", "culinary_arts"),
    ];
    for (src, tgt) in &dev_pairs {
        if let Some(dev) = category_path_deviation(layer, src, tgt) {
            let quality = if dev < 0.1 {
                "tight"
            } else if dev < 0.5 {
                "moderate detour"
            } else {
                "major detour"
            };
            println!(
                "  {} → {}: deviation = {:.4} rad ({:.1}°) — {}",
                src,
                tgt,
                dev,
                dev.to_degrees(),
                quality,
            );
        }
    }

    // 7d: Bridge quality matrix — strongest off-diagonal cell
    println!("\n  ── §7d. BRIDGE QUALITY MATRIX ────────────────────────────\n");
    let matrix = &layer.spatial_quality.bridge_quality_matrix;
    let cat_names: Vec<&str> = layer.summaries.iter().map(|s| s.name.as_str()).collect();
    let nc = cat_names.len();

    let mut best_val = 0.0_f64;
    let mut best_i = 0;
    let mut best_j = 0;
    for (i, row) in matrix.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if i != j && v > best_val {
                best_val = v;
                best_i = i;
                best_j = j;
            }
        }
    }
    let nonzero: usize = matrix
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v > 0.001)
        .count();
    println!(
        "  {}×{} matrix, {} nonzero cells (of {} off-diagonal)",
        nc,
        nc,
        nonzero,
        nc * nc.saturating_sub(1)
    );
    if best_val > 0.0 {
        println!(
            "  Strongest link: {} → {} = {:.4}",
            cat_names[best_i], cat_names[best_j], best_val
        );
    }

    // 7e: Synthesized AI reasoning — cross-domain question
    println!("\n  ── §7e. SYNTHESIZED REASONING CHAIN ─────────────────────");
    println!("  Simulating how an AI answers: \"How does music relate to physics?\"\n");

    let question_vec = embed(
        &[
            (SOUND, 0.6),
            (WAVE, 0.7),
            (MATH, 0.4),
            (ENERGY, 0.3),
            (HARMONY, 0.5),
        ],
        7777,
    );
    let question_emb = Embedding::new(question_vec.clone());

    // Step 1: Route to relevant categories
    let nearby =
        layer.categories_near_embedding(&question_emb, pipeline.projection(), std::f64::consts::PI);
    println!("  Step 1 — Category routing:");
    for (ci, dist) in nearby.iter().take(5) {
        println!(
            "    {:<22} {:.2}°",
            layer.summaries[*ci].name,
            dist.to_degrees()
        );
    }

    // Step 2: Find the category path
    println!("\n  Step 2 — Category path:");
    if let Some(path) = pipeline.category_path("music", "physics") {
        let chain: Vec<&str> = path
            .steps
            .iter()
            .map(|s| s.category_name.as_str())
            .collect();
        println!("    Route: {}", chain.join(" → "));
        println!(
            "    Distance: {:.3} ({:.1}°)",
            path.total_distance,
            path.total_distance.to_degrees()
        );

        // Step 3: Gather bridge concepts at each transition
        println!("\n  Step 3 — Bridge concepts at each hop:");
        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 2);
                let rev = pipeline.bridge_items(&next.category_name, &step.category_name, 2);
                let all_labels: Vec<String> = bridges
                    .iter()
                    .chain(rev.iter())
                    .take(2)
                    .map(|b| format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength))
                    .collect();
                if all_labels.is_empty() {
                    println!("    {} → {}: (adjacency)", step.category_name, next.category_name);
                } else {
                    println!(
                        "    {} → {}: {}",
                        step.category_name,
                        next.category_name,
                        all_labels.join(", ")
                    );
                }
            }
        }

        // Step 4: Confidence calibration
        println!("\n  Step 4 — Confidence per hop:");
        for step in &path.steps {
            if let Some(s) = layer.get_category(&step.category_name) {
                let bar_len = ((s.cohesion * 20.0) as usize).min(20);
                let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
                println!(
                    "    {:<22} cohesion={:.3}  [{}]",
                    step.category_name, s.cohesion, bar
                );
            }
        }

        // Step 5: Geodesic deviation for this pair
        if let Some(dev) = category_path_deviation(layer, "music", "physics") {
            println!(
                "\n  Step 5 — Path deviation from geodesic: {:.4} rad ({:.1}°)",
                dev,
                dev.to_degrees()
            );
            if dev > 0.3 {
                println!("    The graph path detours significantly — the connection is indirect.");
            } else {
                println!("    The graph path tracks the geodesic closely — domains are well-linked.");
            }
        }

        // Step 6: Gap confidence along the path
        println!("\n  Step 6 — Gap confidence along path midpoints:");
        for step in &path.steps {
            if let Some(s) = layer.get_category(&step.category_name) {
                let conf = gap_confidence(&s.centroid_position, layer, config.gap_sharpness);
                println!(
                    "    {:<22} gap_conf={:.4} {}",
                    step.category_name,
                    conf,
                    if conf > 0.7 { "— solid ground" } else { "— thin coverage" }
                );
            }
        }

        // Step 7: Synthesize
        let closeness = if path.total_distance < 0.5 {
            "surprisingly close"
        } else if path.total_distance < 1.0 {
            "well-connected"
        } else if path.total_distance < 2.0 {
            "moderately connected"
        } else {
            "quite distant"
        };

        println!("\n  ┌──────────────────────────────────────────────────────────┐");
        println!("  │  SYNTHESIZED ANSWER                                      │");
        println!("  │                                                          │");
        println!(
            "  │  Music and physics are {} on the semantic  │",
            closeness
        );
        println!("  │  sphere (distance {:.3} / π). The connection runs     │", path.total_distance);
        println!("  │  through shared mathematical structure — waves,        │");
        println!("  │  harmonics, and resonance are native to both fields.   │");
        println!("  │                                                          │");
        println!("  │  Bridge concepts at each transition provide specific    │");
        println!("  │  jumping-off points for cross-domain reasoning. The    │");
        println!("  │  projection preserves {:.1}% of the original semantic    │", evr * 100.0);
        println!("  │  structure, giving confidence that these spatial        │");
        println!("  │  relationships reflect genuine conceptual affinity.     │");
        println!("  └──────────────────────────────────────────────────────────┘");
    }

    // ════════════════════════════════════════════════════════════════════════
    //  SUMMARY
    // ════════════════════════════════════════════════════════════════════════

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  Corpus:       {} concepts across {} categories", n, pipeline.num_categories());
    println!("  Projection:   {} (EVR={:.1}%)", pipeline.projection_kind().name(), evr * 100.0);
    println!("  Tuning:       best={:.4} over {} trials in {:.2}s", tune_report.best_score, tune_report.trials.len(), tune_elapsed.as_secs_f64());
    println!("  Meta-model:   {} + {} fitted", nn_model.name(), dw_model.name());
    println!("  Feedback:     {} events, blended score at α=0.5: {:.4}", summary.n_events, record.adjust_score_with_feedback(&summary, 0.5));
    println!("  Coverage:     {:.1}% of S²", cov.coverage_fraction * 100.0);
    println!("  Voronoi:      {} cells, total area {:.3} sr", vor.cells.len(), vor.total_area);
    println!("  Bridges:      {} total — {} genuine / {} weak / {} artifact", total_bridges, genuine, weak, artifact);
    println!("  Groups:       {} hierarchical domain groups", groups.len());
    println!("  Curvature:    {} triples analyzed", curv.top_triples.len());

    println!("\n  All 7 phases demonstrated: auto-tune → meta-learn → embed →");
    println!("  spatial analysis → category analysis → queries → AI divergence.");
    println!("\n================================================================");
}
