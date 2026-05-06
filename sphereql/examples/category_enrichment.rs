//! Category Enrichment Layer — Example
//!
//! Demonstrates the hierarchical category system in SphereQL:
//! - Category summaries (centroid, spread, cohesion)
//! - Inter-category graph and shortest paths
//! - Bridge item detection (cross-domain connectors)
//! - Automatic inner spheres for large categories
//! - Drill-down queries within a single category
//!
//! Run with:
//!   cargo run --example category_enrichment --features embed

use sphereql::embed::{
    BridgeClassification, BridgeItem, CategorySummary, PipelineInput, PipelineQuery,
    SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};

// ---------------------------------------------------------------------------
// Toy embedding dimensions — a real system uses a sentence transformer.
// ---------------------------------------------------------------------------
const DIM: usize = 16;
const PHYSICS: usize = 0;
const BIOLOGY: usize = 1;
const CHEMISTRY: usize = 2;
const MATH: usize = 3;
const COOKING: usize = 4;
const FLAVOR: usize = 5;
const HEAT: usize = 6;
const MUSIC: usize = 7;
const PERFORM: usize = 8;
const RHYTHM: usize = 9;
const HISTORY: usize = 10;
const CIVILIZE: usize = 11;
const ENERGY: usize = 12;
const NATURE: usize = 13;
const TECH: usize = 14;
const EMOTION: usize = 15;

/// Create a toy embedding with small noise for realism.
fn embed(features: &[(usize, f64)], seed: u64) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for &(axis, val) in features {
        v[axis] = val;
    }
    let mut s = seed;
    for x in &mut v {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.03;
    }
    v
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        SphereQL: Category Enrichment Layer Example          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Build corpus ───────────────────────────────────────────────────

    let mut categories = Vec::new();
    let mut embeddings = Vec::new();
    let mut labels = Vec::new();

    // Science (6 items)
    let science_docs: Vec<(&str, Vec<(usize, f64)>)> = vec![
        (
            "Speed of light",
            vec![(PHYSICS, 1.0), (ENERGY, 0.6), (MATH, 0.3)],
        ),
        (
            "DNA structure",
            vec![(BIOLOGY, 1.0), (CHEMISTRY, 0.5), (NATURE, 0.4)],
        ),
        (
            "Quantum entanglement",
            vec![(PHYSICS, 0.9), (MATH, 0.5), (ENERGY, 0.4)],
        ),
        (
            "Photosynthesis",
            vec![
                (BIOLOGY, 0.8),
                (CHEMISTRY, 0.7),
                (ENERGY, 0.8),
                (NATURE, 0.6),
            ],
        ),
        (
            "General relativity",
            vec![(PHYSICS, 0.9), (MATH, 0.8), (ENERGY, 0.3)],
        ),
        (
            "Cell division",
            vec![(BIOLOGY, 0.9), (CHEMISTRY, 0.3), (NATURE, 0.5)],
        ),
    ];
    for (i, (label, feats)) in science_docs.iter().enumerate() {
        categories.push("science".to_string());
        embeddings.push(embed(feats, 100 + i as u64));
        labels.push(*label);
    }

    // Cooking (6 items)
    let cooking_docs: Vec<(&str, Vec<(usize, f64)>)> = vec![
        (
            "Preheat the oven",
            vec![(COOKING, 1.0), (HEAT, 0.9), (FLAVOR, 0.2)],
        ),
        (
            "Simmer the sauce",
            vec![(COOKING, 0.9), (HEAT, 0.7), (FLAVOR, 0.6)],
        ),
        (
            "Season with paprika",
            vec![(COOKING, 0.8), (FLAVOR, 1.0), (CHEMISTRY, 0.2)],
        ),
        ("Fold egg whites", vec![(COOKING, 0.9), (FLAVOR, 0.4)]),
        (
            "Caramelize onions",
            vec![(COOKING, 0.8), (HEAT, 0.8), (FLAVOR, 0.7), (CHEMISTRY, 0.3)],
        ),
        (
            "Fermentation process",
            vec![
                (COOKING, 0.6),
                (CHEMISTRY, 0.7),
                (BIOLOGY, 0.4),
                (NATURE, 0.3),
            ],
        ),
    ];
    for (i, (label, feats)) in cooking_docs.iter().enumerate() {
        categories.push("cooking".to_string());
        embeddings.push(embed(feats, 200 + i as u64));
        labels.push(*label);
    }

    // Music (5 items)
    let music_docs: Vec<(&str, Vec<(usize, f64)>)> = vec![
        (
            "Symphony climax",
            vec![(MUSIC, 1.0), (PERFORM, 0.7), (EMOTION, 0.8), (ENERGY, 0.5)],
        ),
        (
            "Concert hall voice",
            vec![(MUSIC, 0.9), (PERFORM, 0.9), (EMOTION, 0.6)],
        ),
        (
            "Polyrhythm drumming",
            vec![(MUSIC, 0.8), (PERFORM, 0.6), (RHYTHM, 1.0)],
        ),
        (
            "Jazz improvisation",
            vec![(MUSIC, 0.7), (PERFORM, 0.8), (RHYTHM, 0.6), (EMOTION, 0.5)],
        ),
        (
            "Electronic synthesis",
            vec![(MUSIC, 0.6), (TECH, 0.7), (RHYTHM, 0.5), (ENERGY, 0.4)],
        ),
    ];
    for (i, (label, feats)) in music_docs.iter().enumerate() {
        categories.push("music".to_string());
        embeddings.push(embed(feats, 300 + i as u64));
        labels.push(*label);
    }

    // History (5 items)
    let history_docs: Vec<(&str, Vec<(usize, f64)>)> = vec![
        ("Fall of Rome", vec![(HISTORY, 1.0), (CIVILIZE, 0.8)]),
        (
            "Industrial Revolution",
            vec![(HISTORY, 0.8), (TECH, 0.7), (ENERGY, 0.6), (CIVILIZE, 0.6)],
        ),
        ("Ancient Egypt", vec![(HISTORY, 0.9), (CIVILIZE, 1.0)]),
        (
            "Renaissance art",
            vec![
                (HISTORY, 0.7),
                (CIVILIZE, 0.5),
                (EMOTION, 0.6),
                (PERFORM, 0.3),
            ],
        ),
        (
            "Space race",
            vec![(HISTORY, 0.6), (TECH, 0.8), (PHYSICS, 0.4), (ENERGY, 0.5)],
        ),
    ];
    for (i, (label, feats)) in history_docs.iter().enumerate() {
        categories.push("history".to_string());
        embeddings.push(embed(feats, 400 + i as u64));
        labels.push(*label);
    }

    let n = categories.len();
    println!("Corpus: {n} items across 4 categories\n");

    // ── Build pipeline ─────────────────────────────────────────────────

    let pipeline = SphereQLPipeline::new(PipelineInput {
        categories: categories.clone(),
        embeddings: embeddings.clone(),
    })
    .expect("pipeline build failed");

    println!(
        "Pipeline built — EVR: {:.2}%\n",
        pipeline.explained_variance_ratio() * 100.0
    );

    // ── 1. Category summaries ──────────────────────────────────────────

    println!("━━━ Category Summaries ━━━\n");
    println!(
        "{:<12} {:>6} {:>12} {:>10}",
        "Category", "Items", "Spread (°)", "Cohesion"
    );
    println!("{}", "─".repeat(44));

    let layer = pipeline.category_layer();
    for summary in &layer.summaries {
        println!(
            "{:<12} {:>6} {:>12.2} {:>10.4}",
            summary.name,
            summary.member_count,
            summary.angular_spread.to_degrees(),
            summary.cohesion,
        );
    }

    // ── 2. Category graph — nearest neighbors ──────────────────────────

    println!("\n━━━ Category Neighbors (by graph weight) ━━━\n");
    for summary in &layer.summaries {
        let neighbors = layer.category_neighbors(&summary.name, 3);
        let neighbor_names: Vec<&str> = neighbors.iter().map(|n| n.name.as_str()).collect();
        println!("  {} → {}", summary.name, neighbor_names.join(", "));
    }

    // ── 3. Category concept path ───────────────────────────────────────

    println!("\n━━━ Category Concept Path: cooking → history ━━━\n");
    if let Some(path) = pipeline.category_path("cooking", "history") {
        for (i, step) in path.steps.iter().enumerate() {
            let arrow = if i + 1 < path.steps.len() {
                let bridge_desc: Vec<String> = step
                    .bridges_to_next
                    .iter()
                    .map(|b| {
                        let idx = b.item_index;
                        format!("\"{}\" (strength={:.3})", labels[idx], b.bridge_strength)
                    })
                    .collect();
                if bridge_desc.is_empty() {
                    "  │".to_string()
                } else {
                    format!("  │ via: {}", bridge_desc.join(", "))
                }
            } else {
                String::new()
            };

            println!(
                "  ● {} (d={:.4})",
                step.category_name, step.cumulative_distance
            );
            if !arrow.is_empty() {
                println!("{arrow}");
            }
        }
        println!("  Total distance: {:.4}", path.total_distance);
    } else {
        println!("  (no path found)");
    }

    // ── 4. Bridge items ────────────────────────────────────────────────

    println!("\n━━━ Bridge Items: science ↔ cooking ━━━\n");
    let bridges = pipeline.bridge_items("science", "cooking", 5);
    if bridges.is_empty() {
        println!("  (no bridges detected — categories may be too well-separated)");
    }
    for b in &bridges {
        println!(
            "  \"{}\" — strength={:.3}, sim_to_science={:.3}, sim_to_cooking={:.3}",
            labels[b.item_index], b.bridge_strength, b.affinity_to_source, b.affinity_to_target,
        );
    }

    let bridges_cook_sci = pipeline.bridge_items("cooking", "science", 5);
    if !bridges_cook_sci.is_empty() {
        println!();
        for b in &bridges_cook_sci {
            println!(
                "  \"{}\" — strength={:.3}, sim_to_cooking={:.3}, sim_to_science={:.3}",
                labels[b.item_index], b.bridge_strength, b.affinity_to_source, b.affinity_to_target,
            );
        }
    }

    // ── 5. Drill-down within a category ────────────────────────────────

    println!("\n━━━ Drill-Down: nearest in \"science\" to a physics query ━━━\n");
    let query_emb = embed(&[(PHYSICS, 0.8), (ENERGY, 0.7), (MATH, 0.3)], 999);
    let pq = PipelineQuery {
        embedding: query_emb,
    };
    let result = pipeline
        .query(
            SphereQLQuery::DrillDown {
                category: "science",
                k: 4,
            },
            &pq,
        )
        .expect("drill_down query");
    if let SphereQLOutput::DrillDown(items) = result {
        for (i, r) in items.iter().enumerate() {
            let inner_tag = if r.used_inner_sphere {
                " [inner sphere]"
            } else {
                " [outer sphere]"
            };
            println!(
                "  {}. \"{}\" — dist={:.4}{inner_tag}",
                i + 1,
                labels[r.item_index],
                r.distance,
            );
        }
    }

    // ── 6. Inner sphere stats ──────────────────────────────────────────

    println!("\n━━━ Inner Sphere Status ━━━\n");
    let stats = pipeline.inner_sphere_stats();
    if stats.is_empty() {
        println!("  No inner spheres materialized (all categories have < 20 items,",);
        println!("  or EVR improvement was below the 0.10 threshold).");
        println!("  In a real corpus with 50+ items per category, inner spheres");
        println!("  would automatically activate for categories where they help.");
    } else {
        for s in &stats {
            println!(
                "  {} — {} (inner EVR={:.3}, global EVR={:.3}, improvement={:.3})",
                s.category_name,
                s.projection_type,
                s.inner_evr,
                s.global_subset_evr,
                s.evr_improvement,
            );
        }
    }

    // ── 7. Full category stats via query interface ───────────────────

    println!("\n━━━ Category Stats (via SphereQLQuery::CategoryStats) ━━━\n");
    let dummy_q = PipelineQuery {
        embedding: vec![0.0; DIM],
    };
    if let SphereQLOutput::CategoryStats {
        summaries,
        inner_sphere_reports,
    } = pipeline
        .query(SphereQLQuery::CategoryStats, &dummy_q)
        .expect("category_stats query")
    {
        println!(
            "  {} categories, {} inner spheres",
            summaries.len(),
            inner_sphere_reports.len()
        );
    }

    // ── 8. Bridge classification breakdown ─────────────────────────────

    println!("\n━━━ Bridge Classification Breakdown ━━━\n");

    let mut all_bridges: Vec<&BridgeItem> = Vec::new();
    for items in layer.graph.bridges.values() {
        for b in items {
            all_bridges.push(b);
        }
    }

    let total_bridges = all_bridges.len();
    let n_genuine = all_bridges
        .iter()
        .filter(|b| b.classification == BridgeClassification::Genuine)
        .count();
    let n_overlap = all_bridges
        .iter()
        .filter(|b| b.classification == BridgeClassification::OverlapArtifact)
        .count();
    let n_weak = all_bridges
        .iter()
        .filter(|b| b.classification == BridgeClassification::Weak)
        .count();

    let pct = |n: usize| {
        if total_bridges == 0 {
            0.0
        } else {
            100.0 * n as f64 / total_bridges as f64
        }
    };

    println!("  Total bridges: {total_bridges}");
    println!(
        "    Genuine         : {:>3}  ({:>5.1}%)",
        n_genuine,
        pct(n_genuine)
    );
    println!(
        "    OverlapArtifact : {:>3}  ({:>5.1}%)",
        n_overlap,
        pct(n_overlap)
    );
    println!(
        "    Weak            : {:>3}  ({:>5.1}%)",
        n_weak,
        pct(n_weak)
    );

    // Per-pair: pick strongest bridge for each (src, tgt) that has any.
    let mut pair_rows: Vec<(usize, usize, usize, &BridgeItem)> = Vec::new();
    for (&(src, tgt), items) in &layer.graph.bridges {
        if let Some(strongest) = items
            .iter()
            .max_by(|a, b| a.bridge_strength.total_cmp(&b.bridge_strength))
        {
            pair_rows.push((src, tgt, items.len(), strongest));
        }
    }
    pair_rows.sort_by(|a, b| b.3.bridge_strength.total_cmp(&a.3.bridge_strength));

    println!();
    println!(
        "  {:<12} {:<12} {:>6} {:>10} {:<16}",
        "Source", "Target", "Count", "Strongest", "Classification"
    );
    println!("  {}", "─".repeat(60));
    for (src, tgt, count, strongest) in &pair_rows {
        let src_name = &layer.summaries[*src].name;
        let tgt_name = &layer.summaries[*tgt].name;
        let cls = match strongest.classification {
            BridgeClassification::Genuine => "Genuine",
            BridgeClassification::OverlapArtifact => "OverlapArtifact",
            BridgeClassification::Weak => "Weak",
        };
        println!(
            "  {:<12} {:<12} {:>6} {:>10.3} {:<16}",
            src_name, tgt_name, count, strongest.bridge_strength, cls
        );
    }

    // Call out overlap artifacts — these look like strong bridges by
    // raw strength but are really shared-territory noise.
    let overlap_pairs: Vec<(&str, &str, f64)> = pair_rows
        .iter()
        .filter(|(_, _, _, b)| b.classification == BridgeClassification::OverlapArtifact)
        .map(|(s, t, _, b)| {
            (
                layer.summaries[*s].name.as_str(),
                layer.summaries[*t].name.as_str(),
                b.bridge_strength,
            )
        })
        .collect();

    println!();
    if overlap_pairs.is_empty() {
        println!("  (no OverlapArtifact pairs in this corpus — every pair has enough");
        println!("   territorial separation for bridges to count as real connectors)");
    } else {
        println!("  OverlapArtifact pairs (shared territory, not real connectors):");
        for (s, t, strength) in &overlap_pairs {
            println!("    {s} ↔ {t}  (strongest={strength:.3})");
        }
    }

    println!();
    println!("  Insight: a high-strength bridge between two cap-overlapping categories");
    println!("  is just shared territory, not a genuine cross-domain connector. The");
    println!("  classification lifts this signal out of the raw strength number.");

    // ── 9. Per-category bridge_quality ranking ─────────────────────────

    println!("\n━━━ Category bridge_quality Ranking ━━━\n");

    let mut ranked: Vec<&CategorySummary> = layer.summaries.iter().collect();
    ranked.sort_by(|a, b| b.bridge_quality.total_cmp(&a.bridge_quality));

    println!(
        "  {:>4} {:<12} {:>7} {:>14} {:>12} {:>14}",
        "Rank", "Category", "Items", "bridge_qual", "exclusivity", "territ_effic"
    );
    println!("  {}", "─".repeat(70));
    for (i, s) in ranked.iter().enumerate() {
        println!(
            "  {:>4} {:<12} {:>7} {:>14.4} {:>12.3} {:>14.2}",
            i + 1,
            s.name,
            s.member_count,
            s.bridge_quality,
            s.exclusivity,
            s.territorial_efficiency,
        );
    }

    println!();
    println!("  Insight: high bridge_quality marks a hub category — it connects into");
    println!("  the broader graph through strong, territorially clean edges. Low");
    println!("  bridge_quality means either isolation (few bridges) or overlap-");
    println!("  dominated neighbors (strength discounted by the territorial factor).");

    println!("\n✓ Category Enrichment Layer example complete.");
}
