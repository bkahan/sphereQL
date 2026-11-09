use std::collections::HashMap;

use sphereql::core::{
    CartesianPoint, SphericalPoint, cartesian_to_spherical, spherical_to_cartesian,
};
use sphereql::embed::{
    ConceptPath, Embedding, EmbeddingIndex, GlobResult, PcaProjection, Projection, RadialStrategy,
    SlicingManifold,
};
use sphereql::vis::{Overlay, Scene, ScenePoint, SceneStats};

struct Sentence {
    id: String,
    text: String,
    category: String,
    embedding: Embedding,
}

struct ProjectedPoint {
    id: String,
    text: String,
    category: String,
    x: f64,
    y: f64,
    z: f64,
}

fn category_color(cat: &str) -> &'static str {
    match cat {
        "science" => "#2196F3",
        "technology" => "#9C27B0",
        "sports" => "#FF5722",
        "cooking" => "#FF9800",
        "arts" => "#4CAF50",
        "nature" => "#009688",
        "history" => "#795548",
        "health" => "#E91E63",
        "philosophy" => "#607D8B",
        "business" => "#FFC107",
        _ => "#FFFFFF",
    }
}

fn main() {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sphereql-embed/tools/embeddings.json".into());
    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "sphere_viz.html".into());

    // ── 1. Load embeddings ──────────────────────────────────────────────
    eprintln!("Loading embeddings from {json_path}...");
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("Cannot read {json_path}: {e}"));
    let data: serde_json::Value = serde_json::from_str(&raw).expect("Invalid JSON");

    let dim = data["dimension"].as_u64().unwrap() as usize;
    let model_name = data["model"].as_str().unwrap_or("unknown");
    let arr = data["sentences"].as_array().expect("missing sentences");

    let sentences: Vec<Sentence> = arr
        .iter()
        .map(|s| {
            let values: Vec<f64> = s["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            Sentence {
                id: s["id"].as_str().unwrap().into(),
                text: s["text"].as_str().unwrap().into(),
                category: s["category"].as_str().unwrap().into(),
                embedding: Embedding::new(values),
            }
        })
        .collect();

    eprintln!(
        "Loaded {} sentences ({dim}-d, model: {model_name})",
        sentences.len()
    );

    // ── 2. Fit volumetric PCA projection ────────────────────────────────
    eprintln!("Fitting volumetric PCA projection ({dim}-d → 3D volume)...");
    let all_emb: Vec<Embedding> = sentences.iter().map(|s| s.embedding.clone()).collect();
    let pca = PcaProjection::fit(&all_emb, RadialStrategy::Magnitude)
        .expect("PCA fit")
        .with_volumetric(true);

    let mut index = EmbeddingIndex::builder(pca.clone())
        .uniform_shells(10, 1.0)
        .theta_divisions(12)
        .phi_divisions(6)
        .build();

    for s in &sentences {
        index.insert(&s.id, &s.embedding);
    }

    // ── 3. Project all points → Cartesian for manifold analysis ─────────
    let mut cart_points: Vec<[f64; 3]> = Vec::with_capacity(sentences.len());
    let mut projected: Vec<ProjectedPoint> = Vec::with_capacity(sentences.len());

    for s in &sentences {
        let sp = pca.project(&s.embedding);
        let c = spherical_to_cartesian(&sp);
        cart_points.push([c.x, c.y, c.z]);

        projected.push(ProjectedPoint {
            id: s.id.clone(),
            text: s.text.clone(),
            category: s.category.clone(),
            x: c.x,
            y: c.y,
            z: c.z,
        });
    }

    // ── 4. Fit slicing manifold ─────────────────────────────────────────
    eprintln!("Fitting optimal slicing manifold...");
    let manifold = SlicingManifold::fit(&cart_points);

    // ── 5. Concept paths ────────────────────────────────────────────────
    // One path per category pair — 10 paths covering all categories
    eprintln!("Computing concept paths...");

    let path_pairs = [
        ("sci-04", "cook-15"),     // Einstein → Maillard reaction
        ("phil-06", "tech-03"),    // Mind-body problem → AI text generation
        ("sport-02", "health-01"), // Marathon → cardiovascular exercise
        ("nat-01", "biz-12"),      // Amazon rainforest → corporate social responsibility
        ("art-01", "hist-01"),     // Beethoven → Renaissance
        ("health-12", "phil-02"),  // Cognitive behavioral therapy → trolley problem
        ("tech-07", "sci-10"),     // Quantum computers → Higgs boson
        ("cook-11", "nat-08"),     // Sourdough bread → bee pollination
        ("biz-04", "art-09"),      // Marketing → film editing
        ("sport-12", "phil-11"),   // Rock climbing → Stoic philosophy
    ];

    let mut paths: Vec<(String, String, ConceptPath)> = Vec::new();
    for &(src, tgt) in &path_pairs {
        if let Some(path) = index.concept_path(src, tgt, 8) {
            paths.push((src.into(), tgt.into(), path));
        }
    }

    // ── 6. Queries ──────────────────────────────────────────────────────
    eprintln!("Running queries...");

    let json_queries = data["queries"].as_array();
    let queries: Vec<(&str, Embedding)> = if let Some(jq) = json_queries {
        jq.iter()
            .map(|q| {
                let text = q["text"].as_str().unwrap();
                let values: Vec<f64> = q["embedding"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                (text, Embedding::new(values))
            })
            .collect()
    } else {
        Vec::new()
    };

    struct QueryResult {
        description: String,
        x: f64,
        y: f64,
        z: f64,
        hits: Vec<(String, f64, String)>,
        local_manifold: SlicingManifold,
    }

    let mut query_results: Vec<QueryResult> = Vec::new();
    for (desc, emb) in &queries {
        let results = index.search_nearest(emb, 5);
        let sp = pca.project(emb);
        let c = spherical_to_cartesian(&sp);
        let qpt = [c.x, c.y, c.z];
        let local_manifold = SlicingManifold::fit_local(&qpt, &cart_points, 20);
        query_results.push(QueryResult {
            description: desc.to_string(),
            x: c.x,
            y: c.y,
            z: c.z,
            hits: results
                .iter()
                .map(|r| {
                    let cat = sentences
                        .iter()
                        .find(|s| s.id == r.item.id)
                        .map(|s| s.category.clone())
                        .unwrap_or_default();
                    (r.item.id.clone(), r.distance, cat)
                })
                .collect(),
            local_manifold,
        });
    }

    // ── 6b. Concept Globs ───────────────────────────────────────────────
    // Use category count as k so there's one glob per category
    eprintln!("Detecting concept globs...");
    let all_ids: Vec<String> = sentences.iter().map(|s| s.id.clone()).collect();
    let num_categories = {
        let mut cats: Vec<&str> = sentences.iter().map(|s| s.category.as_str()).collect();
        cats.sort();
        cats.dedup();
        cats.len()
    };
    let glob_result = GlobResult::detect(
        &cart_points,
        &all_ids,
        Some(num_categories),
        num_categories + 5,
    );

    // ── 7. Terminal output ──────────────────────────────────────────────
    println!("=== SphereQL: End-to-End Transformer Pipeline ===\n");
    println!(
        "Model: {model_name}  |  Corpus: {} sentences  |  Dim: {dim} → 3D (volumetric)\n",
        sentences.len()
    );

    let mut cat_counts: HashMap<&str, usize> = HashMap::new();
    for s in &sentences {
        *cat_counts.entry(&s.category).or_default() += 1;
    }
    println!("Categories:");
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (cat, count) in &cats {
        println!("  {:<14} {} docs  {}", cat, count, category_color(cat));
    }

    println!(
        "\nSlicing manifold: {:.1}% variance captured in 2D plane",
        manifold.variance_ratio * 100.0
    );

    println!(
        "\nConcept Globs: auto-detected k={} (silhouette={:.3})",
        glob_result.k, glob_result.silhouette,
    );
    for g in &glob_result.globs {
        let mut glob_cats: HashMap<&str, usize> = HashMap::new();
        for mid in &g.member_ids {
            if let Some(s) = sentences.iter().find(|s| s.id == *mid) {
                *glob_cats.entry(&s.category).or_default() += 1;
            }
        }
        let mut gc: Vec<_> = glob_cats.iter().collect();
        gc.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        let top: String = gc
            .iter()
            .take(3)
            .map(|(c, n)| format!("{c}({n})"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  Glob {:>2}: {:>3} members, radius={:.4}, top: {}",
            g.id,
            g.member_ids.len(),
            g.radius,
            top
        );
    }

    for (src, tgt, path) in &paths {
        let src_text = sentences
            .iter()
            .find(|s| s.id == *src)
            .map(|s| truncate(&s.text, 40))
            .unwrap_or_default();
        let tgt_text = sentences
            .iter()
            .find(|s| s.id == *tgt)
            .map(|s| truncate(&s.text, 40))
            .unwrap_or_default();
        println!(
            "\n--- Concept Path: \"{}\" → \"{}\" ({} hops, {:.4} rad total) ---",
            src_text,
            tgt_text,
            path.steps.len() - 1,
            path.total_distance
        );
        for (i, step) in path.steps.iter().enumerate() {
            let cat = sentences
                .iter()
                .find(|s| s.id == step.id)
                .map(|s| s.category.as_str())
                .unwrap_or("?");
            let text = sentences
                .iter()
                .find(|s| s.id == step.id)
                .map(|s| truncate(&s.text, 50))
                .unwrap_or_default();
            println!(
                "  {:>2}. [{:<12}] cum={:.4}  \"{}\"",
                i, cat, step.cumulative_distance, text
            );
        }
    }

    for qr in &query_results {
        println!("\n--- Query (Nearest): \"{}\" ---", qr.description);
        for (i, (id, dist, cat)) in qr.hits.iter().enumerate() {
            let text = sentences
                .iter()
                .find(|s| s.id == *id)
                .map(|s| truncate(&s.text, 55))
                .unwrap_or_default();
            println!(
                "  {}. [{:<12}] {:.4} rad ({:>6.2}°)  \"{}\"",
                i + 1,
                cat,
                dist,
                dist.to_degrees(),
                text,
            );
        }
    }

    // ── Similarity threshold demo (SimilarAbove) ────────────────────────
    if let Some(first_q) = queries.first() {
        let sim_result = index.search_similar(&first_q.1, 0.85);
        println!(
            "\n--- SimilarAbove (cosine ≥ 0.85): \"{}\" → {} hits ---",
            first_q.0,
            sim_result.items.len(),
        );
        for item in sim_result.items.iter().take(8) {
            let cat = sentences
                .iter()
                .find(|s| s.id == item.id)
                .map(|s| s.category.as_str())
                .unwrap_or("?");
            let text = sentences
                .iter()
                .find(|s| s.id == item.id)
                .map(|s| truncate(&s.text, 55))
                .unwrap_or_default();
            println!("  [{cat:<12}] \"{text}\"");
        }
    }

    // ── Local manifold demo ─────────────────────────────────────────────
    if let Some(first_qr) = query_results.first() {
        println!("\n--- Local Manifold: \"{}\" ---", first_qr.description,);
        println!(
            "  Variance captured: {:.1}% (1.0 = flat plane, 0.67 = spherical)",
            first_qr.local_manifold.variance_ratio * 100.0,
        );
    }

    // ── 8. Generate visualization via the shared sphereql-vis crate ─────
    // The bespoke per-query / per-path *buttons* of the old inline template
    // are consolidated into the shared viewer: the click-to-inspect nearest
    // neighbors cover query exploration, and paths / manifolds / globs are
    // toggleable overlays.
    eprintln!("\nGenerating visualization...");

    let points: Vec<ScenePoint> = projected
        .iter()
        .map(|p| ScenePoint::from_cartesian(p.category.clone(), p.text.clone(), [p.x, p.y, p.z]))
        .collect();
    let sr = Scene::surface_radius_for(&points);
    let pos_by_id: HashMap<&str, [f64; 3]> = projected
        .iter()
        .map(|p| (p.id.as_str(), [p.x, p.y, p.z]))
        .collect();

    let mut overlays: Vec<Overlay> = Vec::new();

    // Global slicing manifold.
    overlays.push(Overlay::manifold_slice(
        manifold.centroid,
        manifold.normal,
        format!("slice plane (vr={:.2})", manifold.variance_ratio),
    ));

    // Concept globs as translucent spheres.
    for g in &glob_result.globs {
        overlays.push(Overlay::glob(
            g.centroid,
            g.radius,
            "#4dd0e1",
            format!("glob {} ({} members)", g.id, g.member_ids.len()),
        ));
    }

    // Concept paths as geodesic arcs through their item positions.
    for (src, tgt, path) in &paths {
        let arc: Vec<SphericalPoint> = path
            .steps
            .iter()
            .filter_map(|s| pos_by_id.get(s.id.as_str()))
            .map(|xyz| cartesian_to_spherical(&CartesianPoint::new(xyz[0], xyz[1], xyz[2])))
            .collect();
        if arc.len() >= 2 {
            overlays.push(Overlay::geodesic_path(
                &arc,
                sr,
                8,
                "#ffffff",
                format!("{src} → {tgt}"),
            ));
        }
    }

    // Query points + their fitted local manifolds.
    for qr in &query_results {
        overlays.push(Overlay::glob(
            [qr.x, qr.y, qr.z],
            sr * 0.04,
            "#fff176",
            format!("query: {}", qr.description),
        ));
        overlays.push(Overlay::manifold_slice(
            qr.local_manifold.centroid,
            qr.local_manifold.normal,
            format!("manifold: {}", qr.description),
        ));
    }

    let scene = Scene::builder()
        .title(format!("SphereQL — Transformer Embeddings ({model_name})"))
        .points(points)
        .overlays(overlays)
        .stats(SceneStats::new("pca", pca.explained_variance_ratio()).with_label("PCA variance"))
        .surface_radius(sr)
        .show_axes(true)
        .build();

    std::fs::write(&output_path, scene.to_html())
        .unwrap_or_else(|e| panic!("Cannot write {output_path}: {e}"));
    eprintln!("Wrote {output_path}");
    println!("\n✓ Visualization: {output_path}");
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}
