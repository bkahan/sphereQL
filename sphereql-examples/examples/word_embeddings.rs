use sphereql::core::angular_distance;
use sphereql::embed::{
    Embedding, EmbeddingIndex, EmbeddingMapper, PcaProjection, Projection, RadialStrategy,
    SemanticQuery,
};
use sphereql::index::SpatialItem;
use sphereql::layout::{ClusteredLayout, LayoutStrategy};

// ---------------------------------------------------------------------------
// Semantic axes — each dimension of our toy embedding space encodes one
// abstract concept. Real models (BERT, OpenAI) learn these axes implicitly;
// here we define them by hand so the relationships are transparent.
// ---------------------------------------------------------------------------
const DIM: usize = 32;
const LIVING: usize = 0;
const ANIMAL: usize = 1;
const HUMAN: usize = 2;
const ROYALTY: usize = 3;
const MASCULINE: usize = 4;
const SIZE: usize = 5;
const DOMESTIC: usize = 6;
const AQUATIC: usize = 7;
const AERIAL: usize = 8;
const PREDATORY: usize = 9;
const INTELLECT: usize = 10;
const WARMTH: usize = 11;
const SPEED: usize = 12;
const MATURE: usize = 13;
const FOOD: usize = 14;
const TECH: usize = 15;
const NATURE: usize = 16;
const SOCIAL: usize = 17;
const BEAUTY: usize = 18;
const POWER: usize = 19;
const SOFT: usize = 20;
const DARK: usize = 21;
const BRIGHT: usize = 22;
const LOUD: usize = 23;
const COLD: usize = 24;
const WET: usize = 25;
const HEIGHT: usize = 26;
const WEALTH: usize = 27;
const KNOWLEDGE: usize = 28;
const CREATIVE: usize = 29;
const MOTION: usize = 30;
const SWEET: usize = 31;

/// Build a 32-d embedding from sparse (axis, value) pairs.
/// A deterministic per-word noise floor makes them feel more like real vectors.
fn embed(features: &[(usize, f64)], seed: u64) -> Embedding {
    let mut v = vec![0.0; DIM];
    for &(axis, val) in features {
        v[axis] = val;
    }
    let mut s = seed;
    for x in &mut v {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.05;
    }
    Embedding::new(v)
}

fn word_table() -> Vec<(&'static str, &'static str, Embedding)> {
    // (word, category, embedding)
    vec![
        // --- Royalty ---
        (
            "king",
            "royalty",
            embed(
                &[
                    (HUMAN, 0.9),
                    (ROYALTY, 1.0),
                    (MASCULINE, 0.9),
                    (POWER, 0.9),
                    (WEALTH, 0.9),
                    (MATURE, 0.7),
                    (INTELLECT, 0.6),
                ],
                1,
            ),
        ),
        (
            "queen",
            "royalty",
            embed(
                &[
                    (HUMAN, 0.9),
                    (ROYALTY, 1.0),
                    (MASCULINE, -0.8),
                    (POWER, 0.8),
                    (WEALTH, 0.9),
                    (BEAUTY, 0.6),
                    (INTELLECT, 0.6),
                ],
                2,
            ),
        ),
        (
            "prince",
            "royalty",
            embed(
                &[
                    (HUMAN, 0.9),
                    (ROYALTY, 0.8),
                    (MASCULINE, 0.8),
                    (POWER, 0.5),
                    (WEALTH, 0.8),
                    (MATURE, -0.3),
                ],
                3,
            ),
        ),
        (
            "princess",
            "royalty",
            embed(
                &[
                    (HUMAN, 0.9),
                    (ROYALTY, 0.8),
                    (MASCULINE, -0.7),
                    (BEAUTY, 0.7),
                    (WEALTH, 0.8),
                    (MATURE, -0.3),
                ],
                4,
            ),
        ),
        (
            "throne",
            "royalty",
            embed(
                &[(ROYALTY, 0.9), (POWER, 0.8), (WEALTH, 0.7), (HEIGHT, 0.3)],
                5,
            ),
        ),
        (
            "crown",
            "royalty",
            embed(
                &[
                    (ROYALTY, 0.9),
                    (WEALTH, 0.8),
                    (BRIGHT, 0.5),
                    (BEAUTY, 0.5),
                    (POWER, 0.6),
                ],
                6,
            ),
        ),
        // --- Animals ---
        (
            "cat",
            "animal",
            embed(
                &[
                    (LIVING, 0.9),
                    (ANIMAL, 1.0),
                    (DOMESTIC, 0.8),
                    (PREDATORY, 0.4),
                    (SOFT, 0.7),
                    (SPEED, 0.5),
                    (SIZE, -0.3),
                ],
                7,
            ),
        ),
        (
            "dog",
            "animal",
            embed(
                &[
                    (LIVING, 0.9),
                    (ANIMAL, 1.0),
                    (DOMESTIC, 0.9),
                    (SOCIAL, 0.7),
                    (LOUD, 0.5),
                    (SPEED, 0.5),
                ],
                8,
            ),
        ),
        (
            "lion",
            "animal",
            embed(
                &[
                    (LIVING, 0.9),
                    (ANIMAL, 1.0),
                    (PREDATORY, 0.9),
                    (POWER, 0.8),
                    (SIZE, 0.6),
                    (LOUD, 0.7),
                    (SPEED, 0.6),
                ],
                9,
            ),
        ),
        (
            "eagle",
            "animal",
            embed(
                &[
                    (LIVING, 0.8),
                    (ANIMAL, 0.9),
                    (AERIAL, 0.9),
                    (PREDATORY, 0.7),
                    (SPEED, 0.7),
                    (HEIGHT, 0.8),
                    (BRIGHT, 0.3),
                ],
                10,
            ),
        ),
        (
            "fish",
            "animal",
            embed(
                &[
                    (LIVING, 0.8),
                    (ANIMAL, 0.9),
                    (AQUATIC, 1.0),
                    (WET, 0.8),
                    (COLD, 0.3),
                    (SPEED, 0.4),
                    (FOOD, 0.5),
                ],
                11,
            ),
        ),
        (
            "whale",
            "animal",
            embed(
                &[
                    (LIVING, 0.8),
                    (ANIMAL, 0.9),
                    (AQUATIC, 0.9),
                    (SIZE, 1.0),
                    (WET, 0.7),
                    (INTELLECT, 0.5),
                    (SOCIAL, 0.4),
                ],
                12,
            ),
        ),
        // --- Technology ---
        (
            "computer",
            "tech",
            embed(
                &[
                    (TECH, 1.0),
                    (INTELLECT, 0.6),
                    (KNOWLEDGE, 0.5),
                    (BRIGHT, 0.3),
                    (POWER, 0.2),
                ],
                13,
            ),
        ),
        (
            "robot",
            "tech",
            embed(
                &[
                    (TECH, 0.9),
                    (INTELLECT, 0.5),
                    (MOTION, 0.5),
                    (POWER, 0.3),
                    (HUMAN, 0.2),
                ],
                14,
            ),
        ),
        (
            "software",
            "tech",
            embed(
                &[
                    (TECH, 0.9),
                    (INTELLECT, 0.5),
                    (KNOWLEDGE, 0.4),
                    (CREATIVE, 0.3),
                ],
                15,
            ),
        ),
        (
            "algorithm",
            "tech",
            embed(
                &[
                    (TECH, 0.8),
                    (INTELLECT, 0.7),
                    (KNOWLEDGE, 0.6),
                    (POWER, 0.2),
                ],
                16,
            ),
        ),
        // --- Nature ---
        (
            "tree",
            "nature",
            embed(
                &[
                    (LIVING, 0.8),
                    (NATURE, 1.0),
                    (HEIGHT, 0.6),
                    (SIZE, 0.5),
                    (MATURE, 0.5),
                ],
                17,
            ),
        ),
        (
            "flower",
            "nature",
            embed(
                &[
                    (LIVING, 0.8),
                    (NATURE, 0.9),
                    (BEAUTY, 0.8),
                    (BRIGHT, 0.6),
                    (SWEET, 0.5),
                    (SOFT, 0.5),
                ],
                18,
            ),
        ),
        (
            "mountain",
            "nature",
            embed(
                &[
                    (NATURE, 0.9),
                    (HEIGHT, 1.0),
                    (SIZE, 1.0),
                    (COLD, 0.5),
                    (POWER, 0.4),
                ],
                19,
            ),
        ),
        (
            "river",
            "nature",
            embed(
                &[
                    (NATURE, 0.9),
                    (WET, 0.9),
                    (MOTION, 0.7),
                    (COLD, 0.2),
                    (AQUATIC, 0.5),
                ],
                20,
            ),
        ),
        (
            "ocean",
            "nature",
            embed(
                &[
                    (NATURE, 0.8),
                    (AQUATIC, 0.9),
                    (WET, 1.0),
                    (SIZE, 1.0),
                    (DARK, 0.3),
                    (POWER, 0.5),
                ],
                21,
            ),
        ),
        // --- Food ---
        (
            "bread",
            "food",
            embed(&[(FOOD, 1.0), (WARMTH, 0.5), (SOFT, 0.5), (SWEET, 0.2)], 22),
        ),
        (
            "apple",
            "food",
            embed(
                &[
                    (FOOD, 0.9),
                    (LIVING, 0.4),
                    (SWEET, 0.6),
                    (NATURE, 0.3),
                    (BRIGHT, 0.3),
                ],
                23,
            ),
        ),
        (
            "cake",
            "food",
            embed(
                &[
                    (FOOD, 0.9),
                    (SWEET, 0.9),
                    (WARMTH, 0.3),
                    (BEAUTY, 0.3),
                    (CREATIVE, 0.3),
                ],
                24,
            ),
        ),
        (
            "pizza",
            "food",
            embed(
                &[(FOOD, 0.9), (WARMTH, 0.6), (SOCIAL, 0.4), (CREATIVE, 0.2)],
                25,
            ),
        ),
    ]
}

fn main() {
    println!("=== SphereQL: Word Embeddings on the Sphere ===\n");

    let words = word_table();
    let embeddings: Vec<Embedding> = words.iter().map(|(_, _, e)| e.clone()).collect();

    // -----------------------------------------------------------------------
    // Fit PCA projection: 32-d → S²
    // -----------------------------------------------------------------------
    println!(
        "Fitting PCA projection from {} word embeddings ({DIM} dimensions → 3D sphere)...\n",
        words.len()
    );
    let pca = PcaProjection::fit(&embeddings, RadialStrategy::Magnitude).expect("PCA fit");

    // -----------------------------------------------------------------------
    // Build spatial index
    // -----------------------------------------------------------------------
    let mut index = EmbeddingIndex::builder(pca.clone())
        .uniform_shells(5, 5.0)
        .theta_divisions(8)
        .phi_divisions(4)
        .build();

    for (word, _, emb) in &words {
        index.insert(*word, emb);
    }

    // -----------------------------------------------------------------------
    // Show projected coordinates grouped by category
    // -----------------------------------------------------------------------
    println!("--- Projected Coordinates ---");
    println!(
        "{:<12} {:<10} {:>8} {:>10} {:>10}",
        "word", "category", "r", "θ (deg)", "φ (deg)"
    );
    println!("{}", "-".repeat(54));
    for (word, cat, emb) in &words {
        let sp = pca.project(emb);
        println!(
            "{:<12} {:<10} {:>8.3} {:>10.2} {:>10.2}",
            word,
            cat,
            sp.r,
            sp.theta.to_degrees(),
            sp.phi.to_degrees()
        );
    }

    // -----------------------------------------------------------------------
    // Nearest neighbors for representative words
    // -----------------------------------------------------------------------
    for query_word in ["king", "cat", "computer", "tree", "bread"] {
        println!("\n--- 5 Nearest Neighbors: \"{query_word}\" ---");
        let query_emb = words
            .iter()
            .find(|(w, _, _)| *w == query_word)
            .map(|(_, _, e)| e)
            .unwrap();
        let results = index.search_nearest(query_emb, 6); // 6 because the word itself is #1

        for (i, res) in results.iter().enumerate() {
            let tag = words
                .iter()
                .find(|(w, _, _)| *w == res.item.id)
                .map(|(_, c, _)| *c)
                .unwrap_or("?");
            let deg = res.distance.to_degrees();
            let marker = if res.item.id == query_word {
                " ← (self)"
            } else {
                ""
            };
            println!(
                "  {}. {:<12} [{:<7}]  dist = {:.4} rad ({:>6.2}°){}",
                i + 1,
                res.item.id,
                tag,
                res.distance,
                deg,
                marker,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Similarity search: find all words within a cosine-similarity threshold
    // -----------------------------------------------------------------------
    println!("\n--- Similarity Search: words similar to \"lion\" (projected cosine ≥ 0.85) ---");
    let lion_emb = words
        .iter()
        .find(|(w, _, _)| *w == "lion")
        .map(|(_, _, e)| e)
        .unwrap();
    let similar = index.search_similar(lion_emb, 0.85);
    let lion_proj = index.projection().project(lion_emb);
    for item in &similar.items {
        let d = angular_distance(&lion_proj, item.position());
        let tag = words
            .iter()
            .find(|(w, _, _)| *w == item.id)
            .map(|(_, c, _)| *c)
            .unwrap_or("?");
        println!(
            "  {:<12} [{:<7}]  angular dist = {:.4} rad ({:.2}°)",
            item.id,
            tag,
            d,
            d.to_degrees()
        );
    }
    println!(
        "  ({} found, {} scanned)",
        similar.items.len(),
        similar.total_scanned
    );

    // -----------------------------------------------------------------------
    // Radial dimension: shell query for high-magnitude embeddings
    // -----------------------------------------------------------------------
    println!("\n--- Shell Query: high-magnitude words (r > 2.0) ---");
    println!("(Magnitude ≈ number/strength of semantic associations)");
    let region = SemanticQuery::in_shell(2.0, 100.0);
    let shell_result = index.search_region(&region);
    let mut shell_items: Vec<_> = shell_result.items.iter().collect();
    shell_items.sort_by(|a, b| b.position.r.total_cmp(&a.position.r));
    for item in &shell_items {
        let tag = words
            .iter()
            .find(|(w, _, _)| *w == item.id)
            .map(|(_, c, _)| *c)
            .unwrap_or("?");
        println!(
            "  {:<12} [{:<7}]  r = {:.3}  (magnitude = {:.3})",
            item.id, tag, item.position.r, item.original_magnitude
        );
    }

    // -----------------------------------------------------------------------
    // Compound query: similar to "queen" AND in a magnitude shell
    // -----------------------------------------------------------------------
    println!("\n--- Compound Query: similar to \"queen\" AND magnitude in [1.5, 3.0] ---");
    let queen_emb = words
        .iter()
        .find(|(w, _, _)| *w == "queen")
        .map(|(_, _, e)| e)
        .unwrap();
    let compound = SemanticQuery::similar_in_shell(queen_emb, index.projection(), 0.7, 1.5, 3.0);
    let compound_result = index.search_region(&compound);
    for item in &compound_result.items {
        let tag = words
            .iter()
            .find(|(w, _, _)| *w == item.id)
            .map(|(_, c, _)| *c)
            .unwrap_or("?");
        println!("  {:<12} [{:<7}]  r = {:.3}", item.id, tag, item.position.r);
    }

    // -----------------------------------------------------------------------
    // Layout integration: clustered layout of word embeddings
    // -----------------------------------------------------------------------
    println!("\n--- Clustered Layout (k=5) ---");
    let mapper = EmbeddingMapper::new(pca);
    let layout = ClusteredLayout::new().with_clusters(5).with_spread(0.2);
    let result = layout.layout(&embeddings, &mapper);

    println!(
        "Quality: dispersion={:.3}, overlap={:.3}, silhouette={:.3}",
        result.quality.dispersion_score,
        result.quality.overlap_score,
        result.quality.silhouette_score,
    );
    for (i, entry) in result.entries.iter().enumerate() {
        let (word, cat, _) = &words[i];
        let p = &entry.position;
        println!(
            "  {:<12} [{:<7}]  layout pos: (θ={:>7.2}°, φ={:>7.2}°)",
            word,
            cat,
            p.theta.to_degrees(),
            p.phi.to_degrees(),
        );
    }
}
