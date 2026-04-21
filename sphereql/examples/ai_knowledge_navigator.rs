#![allow(clippy::uninlined_format_args)]
//! AI Knowledge Navigator — Category Enrichment Demo
//!
//! Demonstrates how sphereQL's Category Enrichment Layer could help an AI
//! model reason about cross-domain connections. The corpus simulates an AI's
//! knowledge across 31 academic domains with deliberately placed "bridge
//! concepts" that span multiple fields.
//!
//! The corpus is provided by the `sphereql-corpus` crate, which contains
//! 775 concepts with 128-dimensional hand-crafted embeddings.
//!
//! The demo runs 13 analyses:
//!   1.  Category landscape (cohesion, spread, centroid positions)
//!   2.  Sphere geometry — centroid map with angular coordinates
//!   3.  Inter-category adjacency graph with edge weight decomposition
//!   4.  Bridge density analysis — most-connected category pairs
//!   5.  Bridge concept detection — specific cross-domain connectors
//!   6.  Category boundary analysis — "ambassador" items between domains
//!   7.  Cross-domain concept path traversal (category-level Dijkstra)
//!   8.  Item-level concept paths through the k-NN graph
//!   9.  Gap detection via glob analysis
//!  10.  Multi-query category routing — how queries get dispatched
//!  11.  Inner-sphere drill-down with precision comparison
//!  12.  Nearest-neighbor retrieval with projection quality metadata
//!  13.  Assembled reasoning chain from spatial structure
//!
//! Run with:
//!   cargo run --example ai_knowledge_navigator --features embed

use sphereql::embed::{
    PipelineInput, PipelineQuery, SphereQLOutput, SphereQLPipeline, SphereQLQuery,
};
use sphereql_corpus::axes::*;
use sphereql_corpus::{DIM, build_corpus, embed};

fn main() {
    println!("================================================================");
    println!("  SphereQL: AI Knowledge Navigator");
    println!("  Category Enrichment Layer — Full Capability Demo");
    println!("================================================================\n");

    // ── Build corpus ──────────────────────────────────────────────────
    let corpus = build_corpus();
    let n = corpus.len();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();
    let labels: Vec<&str> = corpus.iter().map(|c| c.label).collect();

    let pipeline = SphereQLPipeline::new(PipelineInput {
        categories: categories.clone(),
        embeddings: embeddings.clone(),
    })
    .expect("pipeline build failed");

    let evr = pipeline.explained_variance_ratio();
    println!(
        "Corpus: {} concepts across {} knowledge domains",
        n,
        pipeline.num_categories()
    );
    println!(
        "Projection quality: {:.1}% variance explained (EVR={:.4})\n",
        evr * 100.0,
        evr
    );

    let layer = pipeline.category_layer();

    // ==================================================================
    // ANALYSIS 1: Category Landscape
    // ==================================================================
    println!("────────────────────────────────────────────────────────────────");
    println!("  1. CATEGORY LANDSCAPE");
    println!("     Cohesion, spread, and centroid positions for every domain");
    println!("────────────────────────────────────────────────────────────────\n");

    println!(
        "  {:<22} {:>5} {:>10} {:>8} {:>9} {:>9}",
        "Domain", "Items", "Spread(°)", "Cohesion", "θ (°)", "φ (°)"
    );
    println!("  {}", "-".repeat(66));

    let mut sorted_summaries: Vec<&_> = layer.summaries.iter().collect();
    sorted_summaries.sort_by(|a, b| b.cohesion.partial_cmp(&a.cohesion).unwrap());

    for summary in &sorted_summaries {
        println!(
            "  {:<22} {:>5} {:>10.2} {:>8.4} {:>9.2} {:>9.2}",
            summary.name,
            summary.member_count,
            summary.angular_spread.to_degrees(),
            summary.cohesion,
            summary.centroid_position.theta.to_degrees(),
            summary.centroid_position.phi.to_degrees(),
        );
    }

    let most_cohesive = sorted_summaries[0];
    let least_cohesive = sorted_summaries[sorted_summaries.len() - 1];
    println!(
        "\n  -> Tightest cluster:  {} (cohesion {:.4}, spread {:.1}°)",
        most_cohesive.name,
        most_cohesive.cohesion,
        most_cohesive.angular_spread.to_degrees()
    );
    println!(
        "  -> Most diffuse:     {} (cohesion {:.4}, spread {:.1}°)",
        least_cohesive.name,
        least_cohesive.cohesion,
        least_cohesive.angular_spread.to_degrees()
    );
    println!("     An AI should express more uncertainty about diffuse domains —");
    println!("     their concepts are spread across the sphere, not tightly clustered.");

    // ==================================================================
    // ANALYSIS 2: Sphere Geometry — Centroid Map
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  2. SPHERE GEOMETRY — Centroid Distance Map");
    println!("     Angular distances between domain centroids reveal the");
    println!("     topology of knowledge on the sphere");
    println!("────────────────────────────────────────────────────────────────\n");

    // Show pairwise distances for a curated set of domains to keep output manageable
    let focus_domains = [
        "physics",
        "biology",
        "computer_science",
        "philosophy",
        "economics",
        "music",
        "medicine",
        "linguistics",
        "nanotechnology",
        "law",
    ];

    // Header
    print!("  {:>15}", "");
    for &d in &focus_domains {
        let short = &d[..d.len().min(6)];
        print!(" {:>6}", short);
    }
    println!();
    print!("  {:>15}", "");
    println!(" {}", "-".repeat(focus_domains.len() * 7));

    for &row in &focus_domains {
        let ri = match layer.name_to_index.get(row) {
            Some(&i) => i,
            None => continue,
        };
        print!("  {:>15} ", row);
        for &col in &focus_domains {
            let ci = match layer.name_to_index.get(col) {
                Some(&i) => i,
                None => {
                    print!("      -");
                    continue;
                }
            };
            if ri == ci {
                print!("     --");
            } else {
                let dist = sphereql::core::angular_distance(
                    &layer.summaries[ri].centroid_position,
                    &layer.summaries[ci].centroid_position,
                );
                print!(" {:>6.2}", dist.to_degrees());
            }
        }
        println!();
    }

    // Find the closest and most distant category pairs overall
    let num_cats = layer.summaries.len();
    let mut closest_pair = ("", "", f64::INFINITY);
    let mut farthest_pair = ("", "", 0.0f64);
    for i in 0..num_cats {
        for j in (i + 1)..num_cats {
            let d = sphereql::core::angular_distance(
                &layer.summaries[i].centroid_position,
                &layer.summaries[j].centroid_position,
            );
            if d < closest_pair.2 {
                closest_pair = (&layer.summaries[i].name, &layer.summaries[j].name, d);
            }
            if d > farthest_pair.2 {
                farthest_pair = (&layer.summaries[i].name, &layer.summaries[j].name, d);
            }
        }
    }
    println!(
        "\n  Closest pair:   {} <-> {} ({:.2}°)",
        closest_pair.0,
        closest_pair.1,
        closest_pair.2.to_degrees()
    );
    println!(
        "  Farthest pair:  {} <-> {} ({:.2}°)",
        farthest_pair.0,
        farthest_pair.1,
        farthest_pair.2.to_degrees()
    );
    println!(
        "  Sphere utilization: {:.1}° spread across {:.1}° max",
        farthest_pair.2.to_degrees(),
        180.0
    );

    // ==================================================================
    // ANALYSIS 3: Adjacency Graph with Edge Weight Decomposition
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  3. ADJACENCY GRAPH — Edge Weight Decomposition");
    println!("     Shows how bridges reduce effective distance between domains");
    println!("────────────────────────────────────────────────────────────────\n");

    println!(
        "  {:<16} -> {:<16} {:>8} {:>7} {:>8} {:>8} {:>8}",
        "Source", "Target", "Raw(°)", "Bridges", "MaxStr", "Weight", "Savings"
    );
    println!("  {}", "-".repeat(78));

    // Show the top 3 neighbors for each focus domain, with full edge decomposition
    for &domain in &focus_domains[..6] {
        let ci = match layer.name_to_index.get(domain) {
            Some(&i) => i,
            None => continue,
        };
        for edge in layer.graph.adjacency[ci].iter().take(3) {
            let target_name = &layer.summaries[edge.target].name;
            let raw_deg = edge.centroid_distance.to_degrees();
            let weight_deg = edge.weight.to_degrees();
            let savings_pct = if edge.centroid_distance > 0.0 {
                (1.0 - edge.weight / edge.centroid_distance) * 100.0
            } else {
                0.0
            };
            println!(
                "  {:<16} -> {:<16} {:>7.2}° {:>7} {:>8.3} {:>7.2}° {:>7.1}%",
                domain,
                target_name,
                raw_deg,
                edge.bridge_count,
                edge.max_bridge_strength,
                weight_deg,
                savings_pct,
            );
        }
        if domain != focus_domains[5] {
            println!("  {}", "·".repeat(78));
        }
    }

    println!("\n  The \"Savings\" column shows how much shorter the effective path");
    println!("  becomes when bridge concepts pull two domains together.");
    println!("  More bridges with higher strength = cheaper traversal for the AI.");

    // ==================================================================
    // ANALYSIS 4: Bridge Density Analysis
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  4. BRIDGE DENSITY ANALYSIS");
    println!("     Which domain pairs are most interconnected?");
    println!("────────────────────────────────────────────────────────────────\n");

    // Collect all bridge pair stats
    let mut bridge_pairs: Vec<(&str, &str, usize, f64)> = Vec::new();
    for (&(si, ti), bridges) in &layer.graph.bridges {
        if !bridges.is_empty() {
            let mean_str: f64 =
                bridges.iter().map(|b| b.bridge_strength).sum::<f64>() / bridges.len() as f64;
            bridge_pairs.push((
                &layer.summaries[si].name,
                &layer.summaries[ti].name,
                bridges.len(),
                mean_str,
            ));
        }
    }
    bridge_pairs.sort_by(|a, b| b.2.cmp(&a.2).then(b.3.partial_cmp(&a.3).unwrap()));

    println!("  Top 20 most-bridged domain pairs:\n");
    println!(
        "  {:<22} <-> {:<22} {:>7} {:>10}",
        "Domain A", "Domain B", "Bridges", "Mean Str."
    );
    println!("  {}", "-".repeat(65));
    for (a, b, count, mean) in bridge_pairs.iter().take(20) {
        println!("  {:<22} <-> {:<22} {:>7} {:>10.4}", a, b, count, mean);
    }

    // Per-category total bridge counts (outbound)
    println!("\n  Bridge counts per domain (total outbound):\n");
    let mut cat_bridge_totals: Vec<(&str, usize)> = layer
        .summaries
        .iter()
        .enumerate()
        .map(|(ci, s)| {
            let total: usize = layer.graph.adjacency[ci]
                .iter()
                .map(|e| e.bridge_count)
                .sum();
            (s.name.as_str(), total)
        })
        .collect();
    cat_bridge_totals.sort_by_key(|x| std::cmp::Reverse(x.1));

    for (name, total) in &cat_bridge_totals {
        let bar = "█".repeat((*total / 10).min(40));
        println!("  {:<22} {:>5} {}", name, total, bar);
    }

    println!("\n  Domains with many bridges are conceptual hubs — they share");
    println!("  vocabulary and ideas with many other fields. An AI should");
    println!("  route cross-domain queries through these hubs.");

    // ==================================================================
    // ANALYSIS 5: Bridge Concept Detection
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  5. BRIDGE CONCEPTS — Cross-Domain Connectors");
    println!("     Concepts that span two knowledge domains, with affinities");
    println!("────────────────────────────────────────────────────────────────\n");

    let bridge_queries: Vec<(&str, &str)> = vec![
        ("physics", "computer_science"),
        ("physics", "music"),
        ("physics", "chemistry"),
        ("biology", "computer_science"),
        ("biology", "philosophy"),
        ("biology", "medicine"),
        ("computer_science", "economics"),
        ("computer_science", "linguistics"),
        ("philosophy", "economics"),
        ("philosophy", "neuroscience"),
        ("music", "psychology"),
        ("medicine", "engineering"),
        ("nanotechnology", "medicine"),
        ("law", "philosophy"),
        ("data_science", "biology"),
    ];

    for (src, tgt) in &bridge_queries {
        let bridges = pipeline.bridge_items(src, tgt, 3);
        let rev_bridges = pipeline.bridge_items(tgt, src, 3);

        let all: Vec<_> = bridges.iter().chain(rev_bridges.iter()).take(3).collect();

        if all.is_empty() {
            println!("  {} <-> {}: (no bridges — conceptual gap)", src, tgt);
        } else {
            println!("  {} <-> {}:", src, tgt);
            for b in &all {
                let direction = if b.source_category == layer.name_to_index[*src] {
                    format!("{} -> {}", src, tgt)
                } else {
                    format!("{} -> {}", tgt, src)
                };
                println!(
                    "    \"{}\" [{}]  src_affinity={:.3}  tgt_affinity={:.3}  strength={:.3}",
                    labels[b.item_index],
                    direction,
                    b.affinity_to_source,
                    b.affinity_to_target,
                    b.bridge_strength
                );
            }
        }
    }

    println!("\n  Bridge strength is the harmonic mean of source/target affinities.");
    println!("  High strength means the concept is equally relevant to both domains —");
    println!("  it's a genuine conceptual connector, not just noise.");

    // ==================================================================
    // ANALYSIS 6: Category Boundary Analysis
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  6. CATEGORY BOUNDARY ANALYSIS — Domain Ambassadors");
    println!("     Items that live near the border between two domains");
    println!("────────────────────────────────────────────────────────────────\n");

    // Find the top bridge items globally (strongest bridges across all pairs)
    let mut all_bridges: Vec<_> = layer
        .graph
        .bridges
        .values()
        .flat_map(|v| v.iter())
        .collect();
    all_bridges.sort_by(|a, b| b.bridge_strength.partial_cmp(&a.bridge_strength).unwrap());

    println!("  Top 15 boundary-straddling concepts (strongest bridges globally):\n");
    println!(
        "  {:<30} {:<18} {:<18} {:>8}",
        "Concept", "Home Domain", "Foreign Domain", "Strength"
    );
    println!("  {}", "-".repeat(78));
    let mut seen_items = std::collections::HashSet::new();
    let mut shown = 0;
    for b in &all_bridges {
        if shown >= 15 {
            break;
        }
        if !seen_items.insert(b.item_index) {
            continue;
        }
        println!(
            "  {:<30} {:<18} {:<18} {:>8.4}",
            labels[b.item_index],
            layer.summaries[b.source_category].name,
            layer.summaries[b.target_category].name,
            b.bridge_strength,
        );
        shown += 1;
    }

    println!("\n  These \"ambassador\" concepts are the most valuable for cross-domain");
    println!("  reasoning. When an AI needs to connect two distant fields, it should");
    println!("  look for items with high bridge strength as natural transition points.");

    // ==================================================================
    // ANALYSIS 7: Cross-Domain Concept Paths (Category-Level)
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  7. CONCEPT PATH TRAVERSAL — Category-Level Reasoning Chains");
    println!("     Shortest paths through the category graph (Dijkstra)");
    println!("────────────────────────────────────────────────────────────────");

    let path_queries: Vec<(&str, &str, &str)> = vec![
        (
            "nanotechnology",
            "economics",
            "How does nanotechnology impact economic systems?",
        ),
        (
            "linguistics",
            "medicine",
            "How does language connect to health?",
        ),
        (
            "philosophy",
            "computer_science",
            "From abstract thought to computation",
        ),
        (
            "music",
            "biology",
            "What connects musical patterns to living systems?",
        ),
        (
            "culinary_arts",
            "physics",
            "From the kitchen to the laws of nature",
        ),
        (
            "religion",
            "data_science",
            "From spiritual traditions to data analysis",
        ),
    ];

    for (src, tgt, question) in &path_queries {
        println!("\n  Q: \"{}\"", question);
        println!("  Path: {} -> {}\n", src, tgt);

        if let Some(path) = pipeline.category_path(src, tgt) {
            for (i, step) in path.steps.iter().enumerate() {
                let is_last = i + 1 >= path.steps.len();
                println!(
                    "    [{}] {} (cumulative: {:.3})",
                    i + 1,
                    step.category_name,
                    step.cumulative_distance,
                );

                if !is_last {
                    let bridge_descs: Vec<String> = step
                        .bridges_to_next
                        .iter()
                        .take(2)
                        .map(|b| {
                            format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength)
                        })
                        .collect();
                    if !bridge_descs.is_empty() {
                        println!("         bridged by: {}", bridge_descs.join(", "));
                    } else {
                        println!("         (direct adjacency)");
                    }
                    println!("         |");
                }
            }
            println!(
                "    Total distance: {:.3} ({:.1}° — {})\n",
                path.total_distance,
                path.total_distance.to_degrees(),
                if path.total_distance < 0.5 {
                    "very close"
                } else if path.total_distance < 1.0 {
                    "close"
                } else if path.total_distance < 2.0 {
                    "moderate"
                } else {
                    "distant"
                }
            );
        } else {
            println!("    (no path found)\n");
        }
    }

    // ==================================================================
    // ANALYSIS 8: Item-Level Concept Paths
    // ==================================================================
    println!("────────────────────────────────────────────────────────────────");
    println!("  8. ITEM-LEVEL CONCEPT PATHS — k-NN Graph Traversal");
    println!("     Tracing concept-to-concept paths through semantic space");
    println!("────────────────────────────────────────────────────────────────");

    let item_paths: Vec<(usize, usize, &str)> = vec![
        (10, 125, "Acoustics (physics) -> Harmonic theory (music)"),
        (
            24,
            70,
            "Quantum information (physics) -> Quantum computing (CS)",
        ),
        (
            30,
            100,
            "Bioinformatics (biology) -> Microeconomics (economics)",
        ),
    ];

    let dummy_q = PipelineQuery {
        embedding: vec![0.0; DIM],
    };

    for (src_idx, tgt_idx, desc) in &item_paths {
        let src_id = format!("s-{:04}", src_idx);
        let tgt_id = format!("s-{:04}", tgt_idx);
        println!("\n  {}", desc);
        println!("  {} -> {}\n", src_id, tgt_id);

        let result = pipeline.query(
            SphereQLQuery::ConceptPath {
                source_id: &src_id,
                target_id: &tgt_id,
                graph_k: 8,
            },
            &dummy_q,
        );

        if let SphereQLOutput::ConceptPath(Some(path)) = result {
            for (i, step) in path.steps.iter().enumerate() {
                let item_idx: usize = step.id.strip_prefix("s-").unwrap().parse().unwrap();
                let hop_str = if step.hop_distance > 0.0 {
                    format!(" hop={:.4}", step.hop_distance)
                } else {
                    String::new()
                };
                println!(
                    "    [{}] \"{}\" [{}]{} (cum={:.4})",
                    i + 1,
                    labels[item_idx],
                    step.category,
                    hop_str,
                    step.cumulative_distance,
                );
            }
            println!(
                "    Total: {:.4} ({} hops)",
                path.total_distance,
                path.steps.len() - 1
            );
        } else {
            println!("    (no path found)");
        }
    }

    println!("\n  Item-level paths show the SPECIFIC chain of intermediate concepts");
    println!("  that connect two ideas through semantic space. Each hop is a");
    println!("  k-nearest-neighbor link in the projected sphere.");

    // ==================================================================
    // ANALYSIS 9: Knowledge Density — Glob Detection
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  9. KNOWLEDGE DENSITY — Glob Detection");
    println!("     Where is knowledge concentrated? Where are the gaps?");
    println!("────────────────────────────────────────────────────────────────\n");

    let glob_result = pipeline.query(SphereQLQuery::DetectGlobs { k: None, max_k: 10 }, &dummy_q);

    if let SphereQLOutput::Globs(globs) = glob_result {
        println!(
            "  Detected {} knowledge clusters on the sphere:\n",
            globs.len()
        );
        for g in &globs {
            let cats: Vec<String> = g
                .top_categories
                .iter()
                .map(|(c, n)| format!("{} ({})", c, n))
                .collect();
            let density = if g.radius > 0.0 {
                g.member_count as f64 / (std::f64::consts::PI * g.radius * g.radius)
            } else {
                f64::INFINITY
            };
            println!(
                "  Glob {}: {} members, radius={:.3} rad ({:.1}°), density={:.1}",
                g.id,
                g.member_count,
                g.radius,
                g.radius.to_degrees(),
                density,
            );
            println!("    Domains: {}", cats.join(", "));
        }

        // Identify pure vs mixed globs
        let pure_globs = globs
            .iter()
            .filter(|g| {
                g.top_categories.len() == 1
                    || g.top_categories[0].1 as f64 / g.member_count as f64 > 0.8
            })
            .count();
        let mixed_globs = globs.len() - pure_globs;
        println!(
            "\n  {} pure-domain clusters, {} mixed-domain clusters",
            pure_globs, mixed_globs
        );
        println!("  Mixed clusters are where interdisciplinary research lives.");
        println!("  Pure clusters indicate well-separated fields.");
    }

    // ==================================================================
    // ANALYSIS 10: Multi-Query Category Routing
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  10. MULTI-QUERY CATEGORY ROUTING");
    println!("      How the category layer dispatches diverse queries");
    println!("────────────────────────────────────────────────────────────────\n");

    let test_queries: Vec<(&str, Vec<(usize, f64)>)> = vec![
        (
            "neural networks and consciousness",
            vec![
                (NEURAL, 0.8),
                (CONSCIOUSNESS, 0.7),
                (COMPUTATION, 0.5),
                (MIND, 0.6),
                (AI, 0.4),
            ],
        ),
        (
            "climate change policy",
            vec![
                (CLIMATE, 0.9),
                (POLICY, 0.7),
                (ECOSYSTEM, 0.5),
                (GOVERNANCE, 0.4),
                (CONSERVATION, 0.5),
            ],
        ),
        (
            "music and mathematical patterns",
            vec![
                (SOUND, 0.6),
                (MATH, 0.7),
                (PATTERN, 0.8),
                (HARMONY, 0.5),
                (WAVE, 0.4),
            ],
        ),
        (
            "legal ethics of AI",
            vec![
                (LEGAL, 0.7),
                (ETHICS, 0.8),
                (AI, 0.6),
                (RIGHTS, 0.5),
                (COMPUTATION, 0.3),
                (MORAL, 0.4),
            ],
        ),
        (
            "genetic engineering in agriculture",
            vec![
                (GENETICS, 0.8),
                (LIFE, 0.6),
                (CHEMISTRY, 0.4),
                (ECOSYSTEM, 0.5),
                (NATURE, 0.3),
                (MOLECULAR, 0.4),
            ],
        ),
        (
            "theatrical storytelling in film",
            vec![
                (THEATRICAL, 0.7),
                (NARRATIVE, 0.8),
                (CINEMA, 0.7),
                (EMOTION, 0.5),
                (VISUAL, 0.4),
            ],
        ),
    ];

    for (query_desc, features) in &test_queries {
        let qvec = embed(features, 42 + query_desc.len() as u64);
        let emb = sphereql::embed::Embedding::new(qvec.clone());
        let nearby = layer.categories_near_embedding(&emb, pipeline.pca(), std::f64::consts::PI);

        println!("  Query: \"{}\"", query_desc);
        print!("    Top categories: ");
        let top: Vec<String> = nearby
            .iter()
            .take(5)
            .map(|(ci, dist)| format!("{} ({:.1}°)", layer.summaries[*ci].name, dist.to_degrees()))
            .collect();
        println!("{}", top.join(", "));

        // Also show which category the query is CLOSEST to and how confident we are
        if let Some(&(best_ci, best_dist)) = nearby.first() {
            let best_cat = &layer.summaries[best_ci];
            let in_spread = best_dist <= best_cat.angular_spread;
            println!(
                "    -> Primary: {} ({}within category spread)",
                best_cat.name,
                if in_spread { "" } else { "outside " }
            );
        }
        println!();
    }

    println!("  An AI uses category routing to decide which domain's knowledge");
    println!("  to activate. Queries near multiple centroids trigger cross-domain");
    println!("  reasoning; queries far from all centroids signal knowledge gaps.");

    // ==================================================================
    // ANALYSIS 11: Drill-Down with Inner Sphere
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  11. DRILL-DOWN — Inner Sphere Precision");
    println!("      Zooming into categories for fine-grained retrieval");
    println!("────────────────────────────────────────────────────────────────\n");

    // Query: "quantum computing" sits between physics and CS
    let quantum_computing = embed(
        &[
            (QUANTUM, 0.8),
            (COMPUTATION, 0.7),
            (MATH, 0.6),
            (INFORMATION, 0.5),
            (LOGIC, 0.3),
        ],
        9999,
    );
    let qc_query = PipelineQuery {
        embedding: quantum_computing.clone(),
    };

    println!("  Query: \"quantum computing\" (cross-domain concept)\n");

    // Show distances to relevant category centroids
    let emb = sphereql::embed::Embedding::new(quantum_computing.clone());
    let nearby = layer.categories_near_embedding(&emb, pipeline.pca(), std::f64::consts::PI);
    println!("  Nearest domain centroids:");
    for (ci, dist) in nearby.iter().take(6) {
        let has_inner = if layer.inner_spheres.contains_key(ci) {
            " [inner sphere]"
        } else {
            ""
        };
        println!(
            "    {:<22} {:.3} rad ({:.1}°){}",
            layer.summaries[*ci].name,
            dist,
            dist.to_degrees(),
            has_inner,
        );
    }

    // Drill down into several categories
    let drill_targets = ["physics", "computer_science", "mathematics", "philosophy"];
    for &cat in &drill_targets {
        println!("\n  Drill-down into {} (top 5):", cat.to_uppercase());
        let result = pipeline.query(
            SphereQLQuery::DrillDown {
                category: cat,
                k: 5,
            },
            &qc_query,
        );
        if let SphereQLOutput::DrillDown(results) = result {
            if results.is_empty() {
                println!("    (category not found or empty)");
                continue;
            }
            for (i, r) in results.iter().enumerate() {
                let sphere_tag = if r.used_inner_sphere {
                    "inner"
                } else {
                    "outer"
                };
                println!(
                    "    {}. \"{}\" (dist={:.4}, {} sphere)",
                    i + 1,
                    labels[r.item_index],
                    r.distance,
                    sphere_tag,
                );
            }
        }
    }

    println!("\n  When an inner sphere exists, drill-down uses a category-specific");
    println!("  projection that captures more within-category variance than the");
    println!("  global projection. This gives finer angular discrimination.");

    // ==================================================================
    // ANALYSIS 12: Nearest Neighbor with Projection Metadata
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  12. NEAREST NEIGHBOR — Projection Quality Signals");
    println!("      Certainty and intensity metadata on search results");
    println!("────────────────────────────────────────────────────────────────\n");

    // Run a nearest-neighbor query and show certainty/intensity
    println!("  Query: \"quantum computing\" — 10 nearest neighbors:\n");
    let nn_result = pipeline.query(SphereQLQuery::Nearest { k: 10 }, &qc_query);
    if let SphereQLOutput::Nearest(results) = nn_result {
        println!(
            "  {:<4} {:<30} {:<18} {:>8} {:>9} {:>9}",
            "#", "Concept", "Domain", "Dist(°)", "Certainty", "Intensity"
        );
        println!("  {}", "-".repeat(82));
        for (i, r) in results.iter().enumerate() {
            let idx: usize = r.id.strip_prefix("s-").unwrap().parse().unwrap();
            println!(
                "  {:<4} {:<30} {:<18} {:>8.2} {:>9.4} {:>9.4}",
                i + 1,
                labels[idx],
                r.category,
                r.distance.to_degrees(),
                r.certainty,
                r.intensity,
            );
        }
        println!("\n  Certainty: how faithfully the 3D projection represents the high-D");
        println!("  embedding. Low certainty = the concept lost structure in projection.");
        println!("  Intensity: pre-normalization magnitude — strong signals vs. weak ones.");
    }

    // Also run a cosine similarity threshold query
    println!("\n  Cosine similarity threshold query (min_cosine=0.85):\n");
    let sim_result = pipeline.query(SphereQLQuery::SimilarAbove { min_cosine: 0.85 }, &qc_query);
    if let SphereQLOutput::KNearest(results) = sim_result {
        println!("  Found {} concepts above threshold:", results.len());
        for r in results.iter().take(8) {
            let idx: usize = r.id.strip_prefix("s-").unwrap().parse().unwrap();
            println!(
                "    \"{}\" [{}] dist={:.4} certainty={:.3}",
                labels[idx], r.category, r.distance, r.certainty
            );
        }
        if results.len() > 8 {
            println!("    ... and {} more", results.len() - 8);
        }
    }

    // ==================================================================
    // ANALYSIS 13: Assembled Reasoning Chain
    // ==================================================================
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  13. ASSEMBLED REASONING — Full AI Workflow");
    println!("      Simulating how an AI answers a cross-domain question");
    println!("      using every layer of the category enrichment system");
    println!("────────────────────────────────────────────────────────────────\n");

    let question = "How does music relate to economics?";
    println!("  USER QUESTION: \"{}\"\n", question);
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  AI's spatial reasoning process                        │");
    println!("  └─────────────────────────────────────────────────────────┘\n");

    // Step 1: Route the query to relevant categories
    println!("  STEP 1: Category routing — which domains are relevant?\n");
    let music_econ_q = embed(
        &[
            (SOUND, 0.5),
            (MARKETS, 0.5),
            (PATTERN, 0.4),
            (BEHAVIOR, 0.3),
            (PERFORMANCE, 0.3),
        ],
        7777,
    );
    let me_emb = sphereql::embed::Embedding::new(music_econ_q.clone());
    let me_nearby = layer.categories_near_embedding(&me_emb, pipeline.pca(), std::f64::consts::PI);
    for (ci, dist) in me_nearby.iter().take(5) {
        println!(
            "    {:<22} {:.2}°",
            layer.summaries[*ci].name,
            dist.to_degrees()
        );
    }

    // Step 2: Find the path
    println!("\n  STEP 2: Category path — music -> economics\n");
    if let Some(path) = pipeline.category_path("music", "economics") {
        let domain_chain: Vec<&str> = path
            .steps
            .iter()
            .map(|s| s.category_name.as_str())
            .collect();
        println!("    Route: {}", domain_chain.join(" → "));
        println!(
            "    Semantic distance: {:.3} ({:.1}°)\n",
            path.total_distance,
            path.total_distance.to_degrees()
        );

        // Step 3: Gather bridge concepts along each edge
        println!("  STEP 3: Bridge concepts at each transition\n");
        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 3);
                let rev_bridges =
                    pipeline.bridge_items(&next.category_name, &step.category_name, 3);
                let all_labels: Vec<String> = bridges
                    .iter()
                    .chain(rev_bridges.iter())
                    .take(3)
                    .map(|b| format!("\"{}\" (s={:.3})", labels[b.item_index], b.bridge_strength))
                    .collect();
                if all_labels.is_empty() {
                    println!(
                        "    {} → {}: (direct adjacency)",
                        step.category_name, next.category_name
                    );
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

        // Step 4: Check cohesion to calibrate confidence
        println!("\n  STEP 4: Confidence calibration via domain cohesion\n");
        for step in &path.steps {
            if let Some(summary) = layer.get_category(&step.category_name) {
                let confidence = if summary.cohesion > 0.8 {
                    "HIGH  ██████████"
                } else if summary.cohesion > 0.7 {
                    "MED   ██████░░░░"
                } else if summary.cohesion > 0.6 {
                    "LOW   ████░░░░░░"
                } else {
                    "VLOW  ██░░░░░░░░"
                };
                println!(
                    "    {:<22} cohesion={:.3}  {}",
                    step.category_name, summary.cohesion, confidence
                );
            }
        }

        // Step 5: Drill into endpoints for supporting concepts
        println!("\n  STEP 5: Drill-down into endpoint domains for evidence\n");
        let me_query = PipelineQuery {
            embedding: music_econ_q.clone(),
        };
        for &domain in &["music", "economics"] {
            println!("    {} — top 3:", domain.to_uppercase());
            let drill = pipeline.query(
                SphereQLQuery::DrillDown {
                    category: domain,
                    k: 3,
                },
                &me_query,
            );
            if let SphereQLOutput::DrillDown(results) = drill {
                for (i, r) in results.iter().enumerate() {
                    println!(
                        "      {}. \"{}\" (dist={:.4})",
                        i + 1,
                        labels[r.item_index],
                        r.distance
                    );
                }
            }
            println!();
        }

        // Step 6: Synthesize a narrative
        println!("  STEP 6: Synthesized answer\n");
        println!("    ┌──────────────────────────────────────────────────────┐");
        println!("    │ \"Music and economics, while seemingly distant,     │");
        println!("    │  connect through shared conceptual foundations:     │");
        println!("    │                                                    │");

        for (i, step) in path.steps.iter().enumerate() {
            if i + 1 < path.steps.len() {
                let next = &path.steps[i + 1];
                let bridges = pipeline.bridge_items(&step.category_name, &next.category_name, 1);
                let rev_bridges =
                    pipeline.bridge_items(&next.category_name, &step.category_name, 1);
                let bridge_label = bridges
                    .first()
                    .or(rev_bridges.first())
                    .map(|b| labels[b.item_index])
                    .unwrap_or("shared foundations");
                println!(
                    "    │  • {} → {} via {}",
                    step.category_name, next.category_name, bridge_label
                );
                println!("    │    {:>52}│", "");
            }
        }

        let closeness = if path.total_distance < 1.0 {
            "surprisingly close"
        } else if path.total_distance < 2.0 {
            "moderately connected"
        } else {
            "quite distant"
        };
        println!(
            "    │  Distance {:.3} / π — fields are {}.  │",
            path.total_distance, closeness
        );
        println!("    └──────────────────────────────────────────────────────┘");
    }

    // ── Inner sphere stats ────────────────────────────────────────────
    println!("\n────────────────────────────────────────────────────────────────");
    println!("  APPENDIX: Inner Sphere Status");
    println!("────────────────────────────────────────────────────────────────\n");

    let stats = pipeline.inner_sphere_stats();
    if stats.is_empty() {
        println!("  No inner spheres materialized.");
        println!(
            "  (Categories need ≥{} members AND ≥{:.0}% EVR improvement",
            20, 10.0
        );
        println!("  over the global projection to qualify.)\n");
        println!("  Current category sizes:");
        let mut sizes: Vec<_> = layer
            .summaries
            .iter()
            .map(|s| (&s.name, s.member_count))
            .collect();
        sizes.sort_by_key(|x| std::cmp::Reverse(x.1));
        for (name, count) in sizes.iter().take(10) {
            let bar = "█".repeat(*count);
            let threshold_marker = if *count >= 20 { " ✓ (eligible)" } else { "" };
            println!("    {:<22} {:>3} {}{}", name, count, bar, threshold_marker);
        }
        if sizes.len() > 10 {
            println!("    ... and {} more categories", sizes.len() - 10);
        }
        println!("\n  With a real corpus (50+ items per domain), inner spheres would");
        println!("  automatically activate and provide finer within-category angular");
        println!("  discrimination than the global projection.");
    } else {
        println!(
            "  {} of {} categories have inner spheres:\n",
            stats.len(),
            layer.num_categories()
        );
        println!(
            "  {:<22} {:>6} {:>10} {:>10} {:>10} {:>12}",
            "Domain", "Items", "Projection", "Inner EVR", "Global EVR", "Improvement"
        );
        println!("  {}", "-".repeat(74));
        for s in &stats {
            println!(
                "  {:<22} {:>6} {:>10} {:>10.4} {:>10.4} {:>11.4}",
                s.category_name,
                s.member_count,
                s.projection_type,
                s.inner_evr,
                s.global_subset_evr,
                s.evr_improvement,
            );
        }
    }

    println!("\n================================================================");
    println!(
        "  Demo complete. {} concepts, {} categories, EVR={:.1}%",
        n,
        pipeline.num_categories(),
        pipeline.explained_variance_ratio() * 100.0,
    );
    println!(
        "  Category layer: {} summaries, {} bridge pairs, {} inner spheres",
        layer.num_categories(),
        layer.graph.bridges.len(),
        layer.inner_spheres.len(),
    );
    println!("================================================================");
}
