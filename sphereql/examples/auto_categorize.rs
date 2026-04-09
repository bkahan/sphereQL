//! Auto-categorized embedding pipeline.
//!
//! Loads embeddings classified by a zero-shot NLI model (facebook/bart-large-mnli)
//! with NO pre-assigned categories. Demonstrates:
//!   JSON (auto-classified) → PCA projection → spatial index → queries
//!
//! Run:
//!   python3 sphereql-embed/tools/auto_classify.py > sphereql-embed/tools/embeddings_auto.json
//!   cargo run --example auto_categorize -p sphereql --features embed

use std::collections::HashMap;

use sphereql::core::spherical_to_cartesian;
use sphereql::embed::{
    Embedding, EmbeddingIndex, PcaProjection, Projection, RadialStrategy, SlicingManifold,
};

struct Sent {
    id: String,
    text: String,
    cat: String,
    emb: Embedding,
}

fn main() {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sphereql-embed/tools/embeddings_auto.json".into());

    // ── Load ────────────────────────────────────────────────────────────
    eprintln!("Loading {json_path}...");
    let raw = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("Cannot read {json_path}: {e}"));
    let data: serde_json::Value = serde_json::from_str(&raw).expect("Invalid JSON");

    let dim = data["dimension"].as_u64().unwrap() as usize;
    let model = data["model"].as_str().unwrap_or("?");
    let classifier = data["classifier"].as_str().unwrap_or("?");
    let arr = data["sentences"].as_array().expect("missing sentences");

    let sentences: Vec<Sent> = arr
        .iter()
        .map(|s| Sent {
            id: s["id"].as_str().unwrap().into(),
            text: s["text"].as_str().unwrap().into(),
            cat: s["category"].as_str().unwrap().into(),
            emb: Embedding::new(
                s["embedding"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            ),
        })
        .collect();

    let mut cat_counts: HashMap<&str, usize> = HashMap::new();
    for s in &sentences {
        *cat_counts.entry(&s.cat).or_default() += 1;
    }

    println!("=== SphereQL: Auto-Categorized Pipeline ===\n");
    println!(
        "Embedder: {model}  |  Classifier: {classifier}  |  Dim: {dim}  |  Sentences: {}",
        sentences.len()
    );
    println!("\nAuto-detected categories (zero-shot, no training data):");
    let mut cats: Vec<_> = cat_counts.iter().collect();
    cats.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (cat, count) in &cats {
        println!("  {:<14} {count}", cat);
    }

    // ── Project (volumetric) ────────────────────────────────────────────
    eprintln!("Fitting volumetric PCA...");
    let all_emb: Vec<Embedding> = sentences.iter().map(|s| s.emb.clone()).collect();
    let pca = PcaProjection::fit(&all_emb, RadialStrategy::Magnitude).with_volumetric(true);

    let mut index = EmbeddingIndex::builder(pca.clone())
        .uniform_shells(10, 1.0)
        .theta_divisions(12)
        .phi_divisions(6)
        .build();

    for s in &sentences {
        index.insert(&s.id, &s.emb);
    }

    // ── Concept paths across auto-detected categories ───────────────────
    let path_pairs = cross_category_pairs(&sentences);
    for (src_id, tgt_id) in &path_pairs {
        if let Some(path) = index.concept_path(src_id, tgt_id, 8) {
            let src_t = find_text(&sentences, src_id, 45);
            let tgt_t = find_text(&sentences, tgt_id, 45);
            println!(
                "\n--- Path: \"{}\" → \"{}\" ({} hops, {:.3} rad) ---",
                src_t,
                tgt_t,
                path.steps.len() - 1,
                path.total_distance,
            );
            for (i, step) in path.steps.iter().enumerate() {
                let cat = sentences
                    .iter()
                    .find(|s| s.id == step.id)
                    .map(|s| s.cat.as_str())
                    .unwrap_or("?");
                let text = find_text(&sentences, &step.id, 55);
                println!(
                    "  {:>2}. [{cat:<12}] cum={:.4}  \"{text}\"",
                    i, step.cumulative_distance,
                );
            }
        }
    }

    // ── Queries ─────────────────────────────────────────────────────────
    if let Some(jq) = data["queries"].as_array() {
        for q in jq {
            let text = q["text"].as_str().unwrap();
            let values: Vec<f64> = q["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let emb = Embedding::new(values);
            let results = index.search_nearest(&emb, 5);
            println!("\n--- Query: \"{text}\" ---");
            for (i, r) in results.iter().enumerate() {
                let cat = sentences
                    .iter()
                    .find(|s| s.id == r.item.id)
                    .map(|s| s.cat.as_str())
                    .unwrap_or("?");
                let t = find_text(&sentences, &r.item.id, 55);
                println!(
                    "  {}. [{cat:<12}] {:.4} rad ({:>6.2}°)  \"{t}\"",
                    i + 1,
                    r.distance,
                    r.distance.to_degrees(),
                );
            }
        }
    }

    // ── Manifold ────────────────────────────────────────────────────────
    let cart_points: Vec<[f64; 3]> = sentences
        .iter()
        .map(|s| {
            let sp = pca.project(&s.emb);
            let c = spherical_to_cartesian(&sp);
            [c.x, c.y, c.z]
        })
        .collect();

    let manifold = SlicingManifold::fit(&cart_points);
    println!(
        "\nSlicing manifold: {:.1}% variance captured in 2D plane",
        manifold.variance_ratio * 100.0
    );

    println!(
        "\n✓ {} sentences auto-categorized into {} categories by {classifier}",
        sentences.len(),
        cat_counts.len(),
    );
    println!("  No human-assigned labels were used.");
}

fn find_text(sentences: &[Sent], id: &str, max: usize) -> String {
    let text = sentences
        .iter()
        .find(|s| s.id == id)
        .map(|s| s.text.as_str())
        .unwrap_or("");
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}…", &text[..max])
    }
}

fn cross_category_pairs(sentences: &[Sent]) -> Vec<(String, String)> {
    let mut seen: Vec<(String, String)> = Vec::new(); // (id, cat)
    for s in sentences {
        if !seen.iter().any(|(_, c)| c == &s.cat) && seen.len() < 4 {
            seen.push((s.id.clone(), s.cat.clone()));
        }
    }
    let mut pairs = Vec::new();
    if seen.len() >= 2 {
        pairs.push((seen[0].0.clone(), seen[1].0.clone()));
    }
    if seen.len() >= 4 {
        pairs.push((seen[2].0.clone(), seen[3].0.clone()));
    }
    pairs
}
