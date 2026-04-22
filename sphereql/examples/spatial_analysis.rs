#![allow(clippy::uninlined_format_args)]
//! Spatial Analysis on S² — Demonstrating Every Geometric Primitive
//!
//! SphereQL projects high-dimensional embeddings onto a 2-sphere (S²). Unlike
//! flat vector spaces, S² has finite area (4π sr), unique antipodes, geodesics
//! as shortest paths, and intrinsic curvature. This example demonstrates how
//! an AI can exploit each of these properties for semantic reasoning.
//!
//! Covered capabilities:
//!   1. Antipodal discovery — what is maximally opposite to a concept?
//!   2. Coverage & void detection — what fraction of knowledge space is mapped?
//!   3. Geodesic sweep — what lies between two ideas?
//!   4. Voronoi tessellation — how much territory does each domain own?
//!   5. Overlap & exclusivity — where do domains contest shared territory?
//!   6. Curvature signatures — is the knowledge landscape flat or curved?
//!   7. Lune classification — which side of a boundary does a bridge concept fall?
//!   8. Region coherence — how dense is a region compared to random?
//!   9. Bridge quality matrix — spatially-adjusted cross-domain connectivity.
//!  10. Hierarchical domain groups — coarse routing when EVR is low.
//!
//! Run with:
//!   cargo run --example spatial_analysis --features embed

use std::collections::HashMap;

use sphereql::core::spatial::*;
use sphereql::core::{SphericalPoint, angular_distance};
use sphereql::embed::{
    BridgeClassification, Embedding, NavigatorConfig, PipelineInput, PipelineQuery, Projection,
    SphereQLOutput, SphereQLPipeline, SphereQLQuery, category_geodesic_sweep,
    category_path_deviation, gap_confidence, run_full_analysis,
};
use sphereql_corpus::{build_corpus, embed};

fn main() {
    println!("================================================================");
    println!("  SphereQL: Spatial Analysis on S²");
    println!("  Every geometric primitive — raw and navigator-wrapped");
    println!("================================================================\n");

    // ── Build corpus and pipeline ────────────────────────────────────────
    let corpus = build_corpus();
    let n = corpus.len();
    let categories: Vec<String> = corpus.iter().map(|c| c.category.to_string()).collect();
    let embeddings: Vec<Vec<f64>> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| embed(&c.features, 1000 + i as u64))
        .collect();

    let pipeline = SphereQLPipeline::new(PipelineInput {
        categories: categories.clone(),
        embeddings,
    })
    .expect("pipeline build failed");

    let evr = pipeline.explained_variance_ratio();
    println!(
        "Corpus: {} concepts across {} categories",
        n,
        pipeline.num_categories()
    );
    println!(
        "Projection quality: {:.1}% variance explained (EVR={:.4})\n",
        evr * 100.0,
        evr
    );

    // Extract positions as SphericalPoints from the exported data
    let exported = pipeline.exported_points();
    let positions: Vec<SphericalPoint> = exported
        .iter()
        .map(|p| SphericalPoint::new_unchecked(p.r, p.theta, p.phi))
        .collect();

    let layer = pipeline.category_layer();
    let config = NavigatorConfig::default();

    // Collect centroids and half-angles for raw primitive calls
    let centroids: Vec<SphericalPoint> = layer
        .summaries
        .iter()
        .map(|s| s.centroid_position)
        .collect();
    let half_angles: Vec<f64> = layer.summaries.iter().map(|s| s.angular_spread).collect();

    // Run the full navigator analysis
    let report = run_full_analysis(layer, &positions, &categories, evr, &config);

    // §1: ANTIPODAL DISCOVERY — "What is maximally OPPOSITE to this concept?"
    println!("════════════════════════════════════════════════════════════════");
    println!("  §1. ANTIPODAL DISCOVERY");
    println!("  Every point on S² has a unique antipode at distance π.");
    println!("  Question: Is 'the opposite of physics' semantically meaningful?");
    println!("════════════════════════════════════════════════════════════════\n");

    let physics_idx = layer
        .summaries
        .iter()
        .position(|s| s.name == "physics")
        .unwrap_or(0);
    let physics_centroid = centroids[physics_idx];
    let physics_anti = antipode(&physics_centroid);
    let dist_to_anti = angular_distance(&physics_centroid, &physics_anti);

    println!("  [Raw] antipode(physics_centroid):");
    println!(
        "    Physics centroid: θ={:.2}°, φ={:.2}°",
        physics_centroid.theta.to_degrees(),
        physics_centroid.phi.to_degrees()
    );
    println!(
        "    Antipode:         θ={:.2}°, φ={:.2}°",
        physics_anti.theta.to_degrees(),
        physics_anti.phi.to_degrees()
    );
    println!(
        "    Distance = {:.6} rad (expected π = {:.6})",
        dist_to_anti,
        std::f64::consts::PI
    );

    // Navigator wrapper: full antipodal report
    println!("\n  [Navigator] Antipodal analysis across all categories:\n");
    println!(
        "  {:<20} {:>10} {:>12} {:>20}",
        "Category", "Coherence", "Items@Anti", "Dominant@Anti"
    );
    println!("  {}", "─".repeat(65));

    for ar in &report.antipodal {
        let dominant = ar
            .dominant_antipodal_category
            .as_deref()
            .unwrap_or("(empty)");
        println!(
            "  {:<20} {:>10.3} {:>12} {:>20}",
            ar.category_name,
            ar.antipodal_coherence,
            ar.antipodal_items.len(),
            dominant,
        );
    }

    if let Some(best) = report.antipodal.iter().max_by(|a, b| {
        a.antipodal_coherence
            .partial_cmp(&b.antipodal_coherence)
            .unwrap()
    }) {
        println!(
            "\n  Highest antipodal coherence: {} ({:.3})",
            best.category_name, best.antipodal_coherence
        );
        if best.antipodal_coherence > 1.5 {
            println!("  → The antipodal region has MORE structure than chance.");
            println!("    This suggests a genuine 'semantic opposite' exists.");
        } else if best.antipodal_coherence > 0.5 {
            println!("  → Moderate structure — some signal, some noise.");
        } else {
            println!("  → Low coherence — the antipode is mostly projection noise.");
        }
    }

    println!("\n  Insight: Antipodes are unique on S² and undefined in R^n.");

    // §2: COVERAGE & VOID DETECTION — "What fraction of knowledge space is covered?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §2. COVERAGE & KNOWLEDGE GAP CARTOGRAPHY");
    println!("  S² has finite area 4π sr. How much is 'claimed' by categories?");
    println!("════════════════════════════════════════════════════════════════\n");

    // Raw primitive: estimate_coverage directly
    let raw_coverage = estimate_coverage(&centroids, &half_angles, 100_000);
    println!(
        "  [Raw] estimate_coverage({} caps, 100k samples):",
        centroids.len()
    );
    println!(
        "    Coverage fraction: {:.2}%",
        raw_coverage.coverage_fraction * 100.0
    );
    println!(
        "    Covered area:     {:.3} sr (of {:.3} sr total)",
        raw_coverage.covered_area,
        4.0 * std::f64::consts::PI
    );
    println!(
        "    Void samples:     {} / {}",
        raw_coverage.void_count, raw_coverage.total_samples
    );

    // Navigator wrapper
    let cov = &report.coverage;
    println!("\n  [Navigator] Knowledge coverage report:");
    println!(
        "    Coverage fraction:  {:.2}% of S²",
        cov.coverage_fraction * 100.0
    );
    println!(
        "    Overlap area:       {:.3} sr (multi-claimed territory)",
        cov.overlap_area
    );

    println!("\n  Per-category cap sizes:");
    println!(
        "  {:<20} {:>12} {:>12}",
        "Category", "Half-angle°", "Solid angle"
    );
    println!("  {}", "─".repeat(47));
    for cap in &cov.category_caps {
        println!(
            "  {:<20} {:>12.2} {:>12.4}",
            cap.name,
            cap.half_angle.to_degrees(),
            cap.solid_angle
        );
    }

    // Gap confidence test
    println!(
        "\n  Gap-aware confidence (sigmoid, sharpness={}):",
        config.gap_sharpness
    );
    let test_points = [
        (
            "At physics centroid",
            layer
                .summaries
                .iter()
                .find(|s| s.name == "physics")
                .map(|s| s.centroid_position)
                .unwrap_or(positions[0]),
        ),
        (
            "At music centroid",
            layer
                .summaries
                .iter()
                .find(|s| s.name == "music")
                .map(|s| s.centroid_position)
                .unwrap_or(positions[0]),
        ),
        (
            "North pole (likely void)",
            SphericalPoint::new_unchecked(1.0, 0.0, 0.01),
        ),
        (
            "South pole (likely void)",
            SphericalPoint::new_unchecked(1.0, 0.0, std::f64::consts::PI - 0.01),
        ),
    ];
    for (name, pt) in &test_points {
        let conf = gap_confidence(pt, layer, config.gap_sharpness);
        println!("    {:<30} confidence = {:.4}", name, conf);
    }

    println!("\n  Insight: Coverage fractions are well-defined only on bounded manifolds.");

    // §3: GEODESIC SWEEP — "What concepts lie along the path between two ideas?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §3. GEODESIC SWEEP — Great Circle Interpolation");
    println!("  What concepts lie 'between' two domains on S²?");
    println!("════════════════════════════════════════════════════════════════\n");

    let cs_idx = layer
        .summaries
        .iter()
        .position(|s| s.name == "computer_science")
        .unwrap_or(1);
    let philo_idx = layer
        .summaries
        .iter()
        .position(|s| s.name == "philosophy")
        .unwrap_or(2);

    let raw_hits = geodesic_sweep(
        &centroids[cs_idx],
        &centroids[philo_idx],
        &positions,
        config.geodesic_epsilon,
    );
    let raw_density = geodesic_density_profile(
        &centroids[cs_idx],
        &centroids[philo_idx],
        &positions,
        config.geodesic_epsilon,
        config.density_bins,
    );

    println!(
        "  [Raw] geodesic_sweep(computer_science → philosophy, ε={:.1}°):",
        config.geodesic_epsilon.to_degrees()
    );
    println!("    Items within ε of arc: {}", raw_hits.len());
    println!(
        "    Density profile ({} bins): {:?}",
        config.density_bins, raw_density
    );

    // Navigator wrapper: category_geodesic_sweep
    println!("\n  [Navigator] Category geodesic sweeps:\n");
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
            println!(
                "  {} → {} (arc = {:.3} rad, {:.1}°)",
                src,
                tgt,
                sweep.arc_length,
                sweep.arc_length.to_degrees()
            );
            println!(
                "    Items within ε={:.1}° of arc: {}",
                config.geodesic_epsilon.to_degrees(),
                sweep.items.len()
            );
            println!(
                "    Gap fraction: {:.1}% of density bins empty",
                sweep.gap_fraction * 100.0
            );

            let max_d = *sweep.density_profile.iter().max().unwrap_or(&1).max(&1);
            let spark: String = sweep
                .density_profile
                .iter()
                .map(|&d| {
                    let level = (d as f64 / max_d as f64 * 7.0) as usize;
                    ['░', '░', '▒', '▒', '▓', '▓', '█', '█'][level.min(7)]
                })
                .collect();
            println!("    Density profile: [{}]", spark);

            let mut cat_counts: HashMap<&str, usize> = HashMap::new();
            for item in &sweep.items {
                *cat_counts.entry(item.category.as_str()).or_default() += 1;
            }
            let mut sorted_cats: Vec<_> = cat_counts.iter().collect();
            sorted_cats.sort_by_key(|&(_, c)| std::cmp::Reverse(*c));
            let top_3: Vec<String> = sorted_cats
                .iter()
                .take(3)
                .map(|(cat, count)| format!("{} ({})", cat, count))
                .collect();
            if !top_3.is_empty() {
                println!("    Top categories along arc: {}", top_3.join(", "));
            }
            println!();
        }
    }

    // Geodesic deviation
    println!("  Geodesic deviation of category paths:");
    println!("  (How far does the graph path stray from the direct arc?)\n");
    let dev_pairs = [
        ("physics", "music"),
        ("computer_science", "religion"),
        ("biology", "law"),
    ];
    for (src, tgt) in &dev_pairs {
        if let Some(dev) = category_path_deviation(layer, src, tgt) {
            println!(
                "    {} → {}: deviation = {:.4} rad ({:.2}°)",
                src,
                tgt,
                dev,
                dev.to_degrees()
            );
        }
    }

    println!("\n  Insight: Great circles are geodesics on S² — the unique shortest path.");
    println!("  R^n has no natural 'between' operator with these properties.");

    // §4: VORONOI TESSELLATION — "How much territory does each domain own?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §4. VORONOI TESSELLATION — Territory and Efficiency");
    println!("  Which domains claim the most sphere real-estate?");
    println!("════════════════════════════════════════════════════════════════\n");

    // Raw primitive: spherical_voronoi
    let raw_voronoi = spherical_voronoi(&centroids, 50_000);
    println!(
        "  [Raw] spherical_voronoi({} generators, 50k samples):",
        centroids.len()
    );
    let raw_total: f64 = raw_voronoi.iter().map(|c| c.area).sum();
    println!(
        "    Total area: {:.3} sr (expected {:.3} = 4π)",
        raw_total,
        4.0 * std::f64::consts::PI
    );
    println!("    Cells: {}", raw_voronoi.len());

    // Navigator wrapper
    let vor = &report.voronoi;
    println!("\n  [Navigator] Voronoi tessellation report:");
    println!(
        "    Total area: {:.3} sr (expected {:.3} = 4π)",
        vor.total_area,
        4.0 * std::f64::consts::PI
    );

    println!(
        "\n  {:<20} {:>8} {:>6} {:>10} {:>10}",
        "Category", "Area sr", "Items", "Efficiency", "GrphOvlp"
    );
    println!("  {}", "─".repeat(58));

    let mut cells = vor.cells.clone();
    cells.sort_by(|a, b| b.cell_area.partial_cmp(&a.cell_area).unwrap());
    for cell in &cells {
        println!(
            "  {:<20} {:>8.3} {:>6} {:>10.2} {:>10.2}",
            cell.category_name,
            cell.cell_area,
            cell.item_count,
            cell.territorial_efficiency,
            cell.graph_neighbor_overlap
        );
    }

    if let Some(hog) = cells.first() {
        println!(
            "\n  Largest Voronoi cell: {} ({:.3} sr)",
            hog.category_name, hog.cell_area
        );
    }
    if let Some(eff) = cells.iter().max_by(|a, b| {
        a.territorial_efficiency
            .partial_cmp(&b.territorial_efficiency)
            .unwrap()
    }) {
        println!(
            "  Most efficient: {} ({:.2} items/sr)",
            eff.category_name, eff.territorial_efficiency
        );
    }

    println!("\n  Insight: Voronoi cells on S² have finite area by construction.");
    println!("  In R^n, Voronoi cells extend to infinity — areas are undefined.");

    // §5: OVERLAP & EXCLUSIVITY — "How much do domains overlap?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §5. SOLID ANGLE BUDGETS — Overlap & Exclusivity");
    println!("  Which domains contest the same territory?");
    println!("════════════════════════════════════════════════════════════════\n");

    // Raw primitive: cap_intersection_area for one pair
    if centroids.len() >= 2 {
        let area_01 =
            cap_intersection_area(&centroids[0], half_angles[0], &centroids[1], half_angles[1]);
        println!(
            "  [Raw] cap_intersection_area({} ∩ {}) = {:.5} sr",
            layer.summaries[0].name, layer.summaries[1].name, area_01
        );

        let excl_0 = cap_exclusivity(0, &centroids, &half_angles, 10_000);
        println!(
            "  [Raw] cap_exclusivity({}) = {:.3}",
            layer.summaries[0].name, excl_0
        );
    }

    // Navigator wrapper
    let ov = &report.overlap;
    println!("\n  [Navigator] Overlap analysis:\n");
    if ov.pairs.is_empty() {
        println!("  No overlapping cap pairs detected.");
    } else {
        println!(
            "  {:<20} {:<20} {:>12} {:>8}",
            "Category A", "Category B", "Overlap sr", "Bridges"
        );
        println!("  {}", "─".repeat(64));
        for pair in ov.pairs.iter().take(10) {
            println!(
                "  {:<20} {:<20} {:>12.5} {:>8}",
                pair.category_a, pair.category_b, pair.intersection_area, pair.bridge_count
            );
        }
    }

    println!("\n  Per-category exclusivity (1.0 = no overlap with anyone):");
    println!(
        "  {:<20} {:>10} {:>12}",
        "Category", "Cap area", "Exclusivity"
    );
    println!("  {}", "─".repeat(45));
    let mut excls = ov.exclusivities.clone();
    excls.sort_by(|a, b| a.exclusivity.partial_cmp(&b.exclusivity).unwrap());
    for e in &excls {
        let bar_len = (e.exclusivity * 20.0) as usize;
        let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
        println!(
            "  {:<20} {:>10.4} {:>12.3} [{}]",
            e.category_name, e.cap_area, e.exclusivity, bar
        );
    }

    println!("\n  Insight: Solid angles are finite on S². Overlap fractions are");
    println!("  geometrically meaningful and predict bridge concept density.");

    // §6: CURVATURE SIGNATURES — "Is the knowledge landscape flat or curved?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §6. SPHERICAL EXCESS — Curvature Signatures");
    println!("  Which category triples have the most non-Euclidean geometry?");
    println!("════════════════════════════════════════════════════════════════\n");

    // Raw primitive: spherical_excess for one triple
    if centroids.len() >= 3 {
        let excess_012 = spherical_excess(&centroids[0], &centroids[1], &centroids[2]);
        println!(
            "  [Raw] spherical_excess({}, {}, {}) = {:.6} sr",
            layer.summaries[0].name, layer.summaries[1].name, layer.summaries[2].name, excess_012
        );
        println!("    (In Euclidean space this would be exactly 0. Non-zero = curvature.)\n");
    }

    // Navigator wrapper
    let curv = &report.curvature;
    println!(
        "  [Navigator] Top {} triples by spherical excess:",
        curv.top_triples.len()
    );
    println!("  {:<50} {:>12}", "Triple", "Excess (sr)");
    println!("  {}", "─".repeat(65));
    for triple in curv.top_triples.iter().take(10) {
        let name = format!(
            "{} × {} × {}",
            triple.categories[0], triple.categories[1], triple.categories[2]
        );
        println!("  {:<50} {:>12.6}", name, triple.excess);
    }

    println!("\n  Per-category curvature signatures:");
    println!(
        "  {:<20} {:>10} {:>10} {:>10}",
        "Category", "Mean E", "Min E", "Max E"
    );
    println!("  {}", "─".repeat(53));
    let mut sigs = curv.signatures.clone();
    sigs.sort_by(|a, b| b.mean_excess.partial_cmp(&a.mean_excess).unwrap());
    for sig in sigs.iter().take(15) {
        println!(
            "  {:<20} {:>10.6} {:>10.6} {:>10.6}",
            sig.category_name, sig.mean_excess, sig.min_excess, sig.max_excess
        );
    }

    if let Some(high) = sigs.first() {
        println!(
            "\n  Highest mean curvature: {} — deeply embedded in S² geometry.",
            high.category_name
        );
    }

    println!("\n  Insight: Spherical excess = 0 in Euclidean space. This metric is");
    println!("  literally the 'curvature tax' — unique to curved manifolds.");

    // §7: LUNE CLASSIFICATION — "Which domain's territory does a bridge concept belong to?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §7. LUNE DECOMPOSITION — Bridge Asymmetry");
    println!("  Do bridge concepts lean toward one domain or the other?");
    println!("════════════════════════════════════════════════════════════════\n");

    // Raw primitive: lune_classify on a few points
    if centroids.len() >= 2 {
        println!("  [Raw] lune_classify() on first 5 items between categories 0 and 1:\n");
        let ca = &centroids[0];
        let cb = &centroids[1];
        for (i, pos) in positions.iter().take(5).enumerate() {
            let side = lune_classify(ca, cb, pos);
            println!(
                "    Item {} → {:?} (between {} and {})",
                i, side, layer.summaries[0].name, layer.summaries[1].name
            );
        }
    }

    // Navigator wrapper
    println!("\n  [Navigator] Lune decomposition:\n");
    if report.lunes.is_empty() {
        println!("  No category pairs with bridges to decompose.");
    } else {
        println!(
            "  {:<35} {:>4} {:>4} {:>4} {:>8} {:>8}",
            "Pair", "→A", "→B", "=", "Asym", "V-div"
        );
        println!("  {}", "─".repeat(68));
        let mut lunes = report.lunes.clone();
        lunes.sort_by(|a, b| b.asymmetry.partial_cmp(&a.asymmetry).unwrap());
        for lune in lunes.iter().take(15) {
            let pair = format!("{} ↔ {}", lune.category_a, lune.category_b);
            println!(
                "  {:<35} {:>4} {:>4} {:>4} {:>8.3} {:>8.5}",
                pair,
                lune.a_leaning_count,
                lune.b_leaning_count,
                lune.on_bisector_count,
                lune.asymmetry,
                lune.bisector_voronoi_divergence
            );
        }

        let mean_asym: f64 = report.lunes.iter().map(|l| l.asymmetry).sum::<f64>()
            / report.lunes.len().max(1) as f64;
        let max_div = report
            .lunes
            .iter()
            .map(|l| l.bisector_voronoi_divergence)
            .fold(0.0_f64, f64::max);

        println!(
            "\n  Mean asymmetry: {:.3} (0=symmetric, 1=fully one-sided)",
            mean_asym
        );
        println!("  Max bisector-Voronoi divergence: {:.5} rad", max_div);
        if max_div > 0.01 {
            println!("  → Divergence > 0 means curvature displaces domain boundaries");
            println!("    from where flat-space geometry would predict them.");
        }
    }

    println!("\n  Insight: Lunes are the region between two great circles —");
    println!("  a construct unique to S². The bisector-Voronoi divergence");
    println!("  directly measures curvature's effect on domain boundaries.");

    // §8: REGION COHERENCE — "How dense is a region vs random?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §8. REGION COHERENCE — Density vs. Uniform Expectation");
    println!("  Is each category's region denser than what chance predicts?");
    println!("════════════════════════════════════════════════════════════════\n");

    println!(
        "  {:<20} {:>12} {:>12} {:>12}",
        "Category", "Radius (°)", "Coherence", "Interpretation"
    );
    println!("  {}", "─".repeat(60));

    for (i, summary) in layer.summaries.iter().enumerate() {
        let coherence = region_coherence(&centroids[i], summary.angular_spread, &positions);
        let interpretation = if coherence > 2.0 {
            "highly clustered"
        } else if coherence > 1.0 {
            "denser than chance"
        } else if coherence > 0.5 {
            "near uniform"
        } else {
            "sparse"
        };
        println!(
            "  {:<20} {:>12.2} {:>12.3} {:>12}",
            summary.name,
            summary.angular_spread.to_degrees(),
            coherence,
            interpretation
        );
    }

    println!("\n  Insight: region_coherence normalizes by the cap's expected fraction");
    println!("  of S². Values > 1 mean the category is genuinely clustered, not just");
    println!("  an artifact of cap size. This normalization requires finite total area.");

    // §9: BRIDGE QUALITY MATRIX — "Which cross-domain hops are trustworthy?"
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §9. BRIDGE QUALITY MATRIX");
    println!("  max_bridge_strength(i,j) × territorial_factor(i,j) on S².");
    println!("════════════════════════════════════════════════════════════════\n");

    let matrix = &pipeline.category_layer().spatial_quality.bridge_quality_matrix;
    let cat_names: Vec<&str> = layer.summaries.iter().map(|s| s.name.as_str()).collect();
    let nc = cat_names.len();

    let total_cells = nc * nc.saturating_sub(1);
    let nonzero: usize = matrix
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v > 0.001)
        .count();

    if nonzero == 0 {
        println!(
            "  [Raw] All {} off-diagonal cells are below 0.001 — the corpus",
            total_cells
        );
        println!("  has no bridges strong enough to register after territorial discount.");
    } else {
        println!(
            "  [Raw] {}/{} off-diagonal cells above 0.001:\n",
            nonzero, total_cells
        );

        // Header row: 4-char truncated names as column labels, indices below for lookup.
        let label_width = 5usize;
        print!("  {:<16}", "row \\ col");
        for j in 0..nc {
            print!(" {:>width$}", j, width = label_width);
        }
        println!();
        print!("  {:<16}", "");
        for _ in 0..nc {
            print!(" {:>width$}", "─────", width = label_width);
        }
        println!();

        for i in 0..nc {
            let row_label = {
                let n = cat_names[i];
                if n.len() > 14 { format!("{}..", &n[..12]) } else { n.to_string() }
            };
            print!("  {:<14} {:>1}", row_label, i);
            for j in 0..nc {
                let v = matrix[i][j];
                if v < 0.001 {
                    print!(" {:>width$}", "·", width = label_width);
                } else {
                    print!(" {:>width$.3}", v, width = label_width);
                }
            }
            println!();
        }

        println!("\n  Index legend:");
        for (i, name) in cat_names.iter().enumerate() {
            println!("    [{:>2}] {}", i, name);
        }
    }

    // Strongest off-diagonal cell and the bridge behind it.
    let mut best_i = 0usize;
    let mut best_j = 0usize;
    let mut best_val = 0.0f64;
    for i in 0..nc {
        for j in 0..nc {
            if i != j && matrix[i][j] > best_val {
                best_val = matrix[i][j];
                best_i = i;
                best_j = j;
            }
        }
    }
    if best_val > 0.0 {
        println!(
            "\n  Strongest off-diagonal: [{}] {} → [{}] {}  =  {:.3}",
            best_i, cat_names[best_i], best_j, cat_names[best_j], best_val
        );
        if let Some(bridges) = layer.graph.bridges.get(&(best_i, best_j))
            && let Some(top) = bridges.first()
        {
            println!(
                "    Top bridge item: raw strength={:.3}, affinity→src={:.3}, affinity→tgt={:.3}",
                top.bridge_strength, top.affinity_to_source, top.affinity_to_target
            );
            println!("    Classification: {:?}", top.classification);
        }
    }

    // Corpus-wide bridge classification counts.
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
    println!("\n  Bridge classification (across {} bridges):", total_bridges);
    println!("    Genuine         : {:>5}", genuine);
    println!("    OverlapArtifact : {:>5}", artifact);
    println!("    Weak            : {:>5}", weak);

    println!(
        "\n  Insight: Bridge quality fuses raw affinity with S² territorial separation —"
    );
    println!("  pairs that 'share cap' produce apparent bridges that aren't real connectors.");
    println!("  This matrix surfaces which cross-domain hops are trustworthy.");

    // §10: HIERARCHICAL DOMAIN GROUPS — "Coarse routing when EVR is low."
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  §10. HIERARCHICAL DOMAIN GROUPS");
    println!("  Super-groups of related categories + low-EVR coarse routing.");
    println!("════════════════════════════════════════════════════════════════\n");

    let groups = pipeline.domain_groups();
    println!("  [Raw] {} domain groups detected:\n", groups.len());
    println!(
        "  {:<4} {:>6} {:>6} {:>10} {:>10}  {}",
        "Grp", "#cats", "items", "cohesion", "spread°", "members"
    );
    println!("  {}", "─".repeat(70));
    for (gi, g) in groups.iter().enumerate() {
        println!(
            "  {:<4} {:>6} {:>6} {:>10.3} {:>10.2}  {}",
            gi,
            g.member_categories.len(),
            g.total_items,
            g.cohesion,
            g.angular_spread.to_degrees(),
            g.category_names.join(", ")
        );
    }

    // Pick 2-3 real queries: a physics item's embedding, a music item's embedding,
    // and a computer_science centroid embedding.
    let queries: Vec<(String, Embedding)> = {
        let mut qs: Vec<(String, Embedding)> = Vec::new();
        for cat_name in ["physics", "music", "computer_science"] {
            if let Some(summary) = layer.summaries.iter().find(|s| s.name == cat_name) {
                // Use an actual member item's embedding — not a synthetic average.
                if let Some(&mi) = summary.member_indices.first() {
                    let features = &corpus[mi].features;
                    let vec = embed(features, 1000 + mi as u64);
                    qs.push((
                        format!("{} item #{} ({})", cat_name, mi, corpus[mi].label),
                        Embedding::new(vec),
                    ));
                }
            }
        }
        qs
    };

    println!(
        "\n  [Navigator] Side-by-side: hierarchical_nearest vs. standard Nearest.\n"
    );
    println!(
        "  EVR = {:.3} (hierarchical path activates below 0.35).",
        evr
    );

    for (label, emb) in &queries {
        println!("\n  ── Query: {} ──", label);
        if let Some(group) = pipeline.route_to_group(emb) {
            let projected = pipeline.pca().project(emb);
            let d = angular_distance(&projected, &group.centroid);
            let gi = groups
                .iter()
                .position(|g| std::ptr::eq(g, group))
                .unwrap_or(usize::MAX);
            println!(
                "    Routed to group {}: [{}]  centroid_dist = {:.3} rad ({:.1}°)",
                gi,
                group.category_names.join(", "),
                d,
                d.to_degrees()
            );
        } else {
            println!("    No group routing available.");
        }

        let hier = pipeline.hierarchical_nearest(emb, 5);
        let standard = match pipeline.query(
            SphereQLQuery::Nearest { k: 5 },
            &PipelineQuery {
                embedding: emb.values.clone(),
            },
        ) {
            SphereQLOutput::Nearest(v) => v,
            _ => Vec::new(),
        };

        println!(
            "\n    {:<20}   {:<20}",
            "hierarchical_nearest", "SphereQLQuery::Nearest"
        );
        println!(
            "    {:<8} {:<12} {:>6}   {:<8} {:<12} {:>6}",
            "id", "category", "dist", "id", "category", "dist"
        );
        println!("    {}", "─".repeat(66));
        let rows = hier.len().max(standard.len());
        for r in 0..rows {
            let (hid, hcat, hdist) = hier
                .get(r)
                .map(|n| (n.id.as_str(), n.category.as_str(), n.distance))
                .unwrap_or(("—", "—", f64::NAN));
            let (sid, scat, sdist) = standard
                .get(r)
                .map(|n| (n.id.as_str(), n.category.as_str(), n.distance))
                .unwrap_or(("—", "—", f64::NAN));
            let hdist_str = if hdist.is_nan() { "—".into() } else { format!("{:.3}", hdist) };
            let sdist_str = if sdist.is_nan() { "—".into() } else { format!("{:.3}", sdist) };
            let hcat_t = if hcat.len() > 12 { &hcat[..12] } else { hcat };
            let scat_t = if scat.len() > 12 { &scat[..12] } else { scat };
            println!(
                "    {:<8} {:<12} {:>6}   {:<8} {:<12} {:>6}",
                hid, hcat_t, hdist_str, sid, scat_t, sdist_str
            );
        }

        let hier_ids: Vec<&str> = hier.iter().map(|n| n.id.as_str()).collect();
        let std_ids: Vec<&str> = standard.iter().map(|n| n.id.as_str()).collect();
        let agree = hier_ids == std_ids;
        println!(
            "    → Result sets {} (EVR {} hierarchical threshold 0.35)",
            if agree { "AGREE" } else { "DIVERGE" },
            if evr >= 0.35 { "≥" } else { "<" }
        );
    }

    println!(
        "\n  Insight: When EVR < 0.35, fine-grained category routing on the outer"
    );
    println!("  sphere is noise-dominated. Hierarchical routing drops the routing");
    println!("  problem's cardinality from N categories to ~5 groups, then uses each");
    println!("  category's inner sphere (where available) for drill-down.");

    // SUMMARY
    println!("\n\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!(
        "  Corpus:     {} concepts across {} categories",
        n,
        pipeline.num_categories()
    );
    println!("  EVR:        {:.1}% variance explained", evr * 100.0);
    println!("  Coverage:   {:.1}% of S²", cov.coverage_fraction * 100.0);
    println!(
        "  Voronoi:    {} cells, total area {:.3} sr",
        vor.cells.len(),
        vor.total_area
    );
    println!(
        "  Overlaps:   {} category pairs with shared territory",
        ov.pairs.len()
    );
    println!("  Curvature:  {} triples analyzed", curv.top_triples.len());
    println!(
        "  Lunes:      {} pairs with bridge decomposition",
        report.lunes.len()
    );
    println!(
        "  Bridges:    {} total — {} genuine / {} overlap-artifact / {} weak",
        total_bridges, genuine, artifact, weak
    );
    println!(
        "  Groups:     {} hierarchical domain groups detected",
        pipeline.domain_groups().len()
    );

    println!("\n  All 10 spatial analyses exploit properties unique to S²:");
    println!("  finite area, antipodality, great circles, solid angles, and curvature.");
    println!("  These queries are ill-defined or degenerate in unbounded R^n.");
    println!("\n================================================================");
}
