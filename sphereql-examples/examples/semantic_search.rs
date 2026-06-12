use sphereql::core::{Region, Shell, angular_distance};
use sphereql::embed::{
    Embedding, EmbeddingIndex, PcaProjection, Projection, RadialStrategy, SemanticQuery,
};
use sphereql::index::SpatialItem;

// ---------------------------------------------------------------------------
// Semantic axes for document-level features.
// Each dimension represents an abstract topic or attribute.
// A real system would use a sentence transformer (e.g. all-MiniLM-L6-v2);
// here we encode topics by hand to keep the example self-contained.
// ---------------------------------------------------------------------------
const DIM: usize = 24;
const PHYSICS: usize = 0;
const BIOLOGY: usize = 1;
const CHEMISTRY: usize = 2;
const MATH: usize = 3;
const COOKING: usize = 4;
const FLAVOR: usize = 5;
const HEAT: usize = 6;
const SPORT: usize = 7;
const COMPETE: usize = 8;
const TEAM: usize = 9;
const MUSIC: usize = 10;
const PERFORM: usize = 11;
const RHYTHM: usize = 12;
const HISTORY: usize = 13;
const CIVILIZE: usize = 14;
const WAR: usize = 15;
const NATURE: usize = 16;
const WATER: usize = 17;
const SPACE: usize = 18;
const TECH: usize = 19;
const ENERGY: usize = 20;
const MOTION: usize = 21;
const EMOTION: usize = 22;
const TIME: usize = 23;

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
        *x += ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04;
    }
    Embedding::new(v)
}

struct Doc {
    id: &'static str,
    text: &'static str,
    topic: &'static str,
    importance: f64, // 0.0 – 1.0, used as metadata radius
    embedding: Embedding,
}

fn corpus() -> Vec<Doc> {
    vec![
        // --- Science ---
        Doc {
            id: "sci-1",
            topic: "science",
            text: "The speed of light in a vacuum is approximately 299,792 km/s.",
            importance: 0.95,
            embedding: embed(
                &[
                    (PHYSICS, 1.0),
                    (SPACE, 0.5),
                    (ENERGY, 0.6),
                    (MOTION, 0.7),
                    (MATH, 0.3),
                ],
                1,
            ),
        },
        Doc {
            id: "sci-2",
            topic: "science",
            text: "DNA carries the genetic instructions for all living organisms.",
            importance: 0.92,
            embedding: embed(
                &[(BIOLOGY, 1.0), (CHEMISTRY, 0.5), (NATURE, 0.4), (TIME, 0.2)],
                2,
            ),
        },
        Doc {
            id: "sci-3",
            topic: "science",
            text: "Quantum entanglement allows particles to be correlated across vast distances.",
            importance: 0.88,
            embedding: embed(
                &[(PHYSICS, 0.9), (SPACE, 0.6), (MATH, 0.5), (ENERGY, 0.4)],
                3,
            ),
        },
        Doc {
            id: "sci-4",
            topic: "science",
            text: "Photosynthesis converts sunlight into chemical energy in plants.",
            importance: 0.85,
            embedding: embed(
                &[
                    (BIOLOGY, 0.8),
                    (CHEMISTRY, 0.7),
                    (ENERGY, 0.8),
                    (NATURE, 0.6),
                ],
                4,
            ),
        },
        // --- Cooking ---
        Doc {
            id: "cook-1",
            topic: "cooking",
            text: "Preheat the oven to 350°F and line a baking sheet with parchment paper.",
            importance: 0.60,
            embedding: embed(&[(COOKING, 1.0), (HEAT, 0.9), (FLAVOR, 0.2)], 5),
        },
        Doc {
            id: "cook-2",
            topic: "cooking",
            text: "Simmer the tomato sauce on low heat until it thickens, stirring occasionally.",
            importance: 0.65,
            embedding: embed(
                &[(COOKING, 0.9), (HEAT, 0.7), (FLAVOR, 0.6), (TIME, 0.3)],
                6,
            ),
        },
        Doc {
            id: "cook-3",
            topic: "cooking",
            text: "Season generously with salt, pepper, and a pinch of smoked paprika.",
            importance: 0.55,
            embedding: embed(&[(COOKING, 0.8), (FLAVOR, 1.0), (CHEMISTRY, 0.2)], 7),
        },
        Doc {
            id: "cook-4",
            topic: "cooking",
            text: "Fold the egg whites gently into the batter to keep it light and airy.",
            importance: 0.50,
            embedding: embed(&[(COOKING, 0.9), (FLAVOR, 0.4), (MOTION, 0.3)], 8),
        },
        // --- Sports ---
        Doc {
            id: "sport-1",
            topic: "sports",
            text: "The team scored a last-minute goal to win the championship.",
            importance: 0.80,
            embedding: embed(
                &[
                    (SPORT, 1.0),
                    (COMPETE, 0.9),
                    (TEAM, 0.8),
                    (EMOTION, 0.6),
                    (TIME, 0.3),
                ],
                9,
            ),
        },
        Doc {
            id: "sport-2",
            topic: "sports",
            text: "She ran the marathon in under three hours, setting a personal record.",
            importance: 0.75,
            embedding: embed(
                &[
                    (SPORT, 0.9),
                    (COMPETE, 0.7),
                    (MOTION, 0.8),
                    (ENERGY, 0.5),
                    (TIME, 0.4),
                ],
                10,
            ),
        },
        Doc {
            id: "sport-3",
            topic: "sports",
            text: "The goalkeeper made a spectacular diving save in extra time.",
            importance: 0.70,
            embedding: embed(
                &[
                    (SPORT, 0.9),
                    (COMPETE, 0.6),
                    (TEAM, 0.5),
                    (MOTION, 0.7),
                    (EMOTION, 0.4),
                ],
                11,
            ),
        },
        // --- Music ---
        Doc {
            id: "mus-1",
            topic: "music",
            text: "The symphony's third movement builds to a powerful fortissimo climax.",
            importance: 0.82,
            embedding: embed(
                &[
                    (MUSIC, 1.0),
                    (PERFORM, 0.7),
                    (RHYTHM, 0.5),
                    (EMOTION, 0.8),
                    (ENERGY, 0.5),
                ],
                12,
            ),
        },
        Doc {
            id: "mus-2",
            topic: "music",
            text: "Her voice resonated through the concert hall with crystalline clarity.",
            importance: 0.78,
            embedding: embed(
                &[(MUSIC, 0.9), (PERFORM, 0.9), (EMOTION, 0.6), (ENERGY, 0.3)],
                13,
            ),
        },
        Doc {
            id: "mus-3",
            topic: "music",
            text: "The drummer maintained a complex polyrhythm throughout the entire piece.",
            importance: 0.68,
            embedding: embed(
                &[(MUSIC, 0.8), (PERFORM, 0.6), (RHYTHM, 1.0), (TIME, 0.5)],
                14,
            ),
        },
        // --- History ---
        Doc {
            id: "hist-1",
            topic: "history",
            text: "The fall of the Roman Empire in 476 AD reshaped the political landscape of Europe.",
            importance: 0.90,
            embedding: embed(
                &[(HISTORY, 1.0), (CIVILIZE, 0.8), (WAR, 0.5), (TIME, 0.7)],
                15,
            ),
        },
        Doc {
            id: "hist-2",
            topic: "history",
            text: "The Industrial Revolution transformed manufacturing through steam-powered machinery.",
            importance: 0.88,
            embedding: embed(
                &[
                    (HISTORY, 0.8),
                    (CIVILIZE, 0.6),
                    (TECH, 0.7),
                    (ENERGY, 0.6),
                    (TIME, 0.5),
                ],
                16,
            ),
        },
        Doc {
            id: "hist-3",
            topic: "history",
            text: "Ancient Egyptians built the pyramids as monumental tombs for their pharaohs.",
            importance: 0.85,
            embedding: embed(&[(HISTORY, 0.9), (CIVILIZE, 1.0), (TIME, 0.6)], 17),
        },
    ]
}

fn main() {
    println!("=== SphereQL: Semantic Document Search ===\n");

    let docs = corpus();
    let all_embeddings: Vec<Embedding> = docs.iter().map(|d| d.embedding.clone()).collect();

    // -----------------------------------------------------------------------
    // Fit projection with magnitude-based radius
    // -----------------------------------------------------------------------
    println!(
        "Fitting PCA from {} documents ({DIM}-d embeddings → 3D sphere)...",
        docs.len()
    );
    let pca = PcaProjection::fit(&all_embeddings, RadialStrategy::Magnitude).expect("PCA fit");

    // -----------------------------------------------------------------------
    // Build index — insert with metadata-driven radius (importance score)
    // -----------------------------------------------------------------------
    let mut index = EmbeddingIndex::builder(pca.clone())
        .uniform_shells(5, 5.0)
        .theta_divisions(8)
        .phi_divisions(4)
        .build();

    println!("Inserting documents with importance-weighted radius...\n");
    for doc in &docs {
        // Scale importance to a useful radial range [1.0, 3.0]
        let r = 1.0 + doc.importance * 2.0;
        index.insert_with_radius(doc.id, &doc.embedding, r);
    }

    // -----------------------------------------------------------------------
    // Show the projected document map
    // -----------------------------------------------------------------------
    println!("--- Document Map ---");
    println!(
        "{:<10} {:<10} {:>6} {:>10} {:>10}  text (truncated)",
        "id", "topic", "r", "θ (deg)", "φ (deg)"
    );
    println!("{}", "-".repeat(90));
    for doc in &docs {
        let item = index.get(doc.id).unwrap();
        let p = item.position;
        let short: String = doc.text.chars().take(50).collect();
        println!(
            "{:<10} {:<10} {:>6.2} {:>10.2} {:>10.2}  {}…",
            doc.id,
            doc.topic,
            p.r,
            p.theta.to_degrees(),
            p.phi.to_degrees(),
            short,
        );
    }

    // -----------------------------------------------------------------------
    // Query 1: "Tell me about physics and energy"
    // -----------------------------------------------------------------------
    let q1 = embed(
        &[(PHYSICS, 0.8), (ENERGY, 0.7), (SPACE, 0.3), (MATH, 0.2)],
        100,
    );
    run_nearest_query(&index, &docs, &q1, "Tell me about physics and energy", 5);

    // -----------------------------------------------------------------------
    // Query 2: "How do I cook something?"
    // -----------------------------------------------------------------------
    let q2 = embed(&[(COOKING, 0.9), (HEAT, 0.5), (FLAVOR, 0.4)], 101);
    run_nearest_query(&index, &docs, &q2, "How do I cook something?", 5);

    // -----------------------------------------------------------------------
    // Query 3: "What happened in ancient history?"
    // -----------------------------------------------------------------------
    let q3 = embed(&[(HISTORY, 0.9), (CIVILIZE, 0.7), (TIME, 0.5)], 102);
    run_nearest_query(&index, &docs, &q3, "What happened in ancient history?", 5);

    // -----------------------------------------------------------------------
    // Similarity search: find all docs semantically close to a music query
    // -----------------------------------------------------------------------
    println!("\n--- Similarity Search: \"orchestral performance\" (cosine ≥ 0.75) ---");
    let q_music = embed(&[(MUSIC, 0.9), (PERFORM, 0.8), (EMOTION, 0.4)], 103);
    let sim_results = index.search_similar(&q_music, 0.75);
    let q_proj = index.projection().project(&q_music);
    for item in &sim_results.items {
        let d = angular_distance(&q_proj, item.position());
        let doc = docs.iter().find(|d| d.id == item.id).unwrap();
        println!(
            "  [{:<10}] {:.4} rad ({:>6.2}°)  \"{}\"",
            doc.topic,
            d,
            d.to_degrees(),
            truncate(doc.text, 60),
        );
    }
    println!(
        "  ({} found, {} scanned)",
        sim_results.items.len(),
        sim_results.total_scanned,
    );

    // -----------------------------------------------------------------------
    // Compound query: science docs with high importance (r > 2.5)
    // -----------------------------------------------------------------------
    println!("\n--- Compound Query: science-like AND high importance (r > 2.5) ---");
    let q_sci = embed(
        &[
            (PHYSICS, 0.5),
            (BIOLOGY, 0.5),
            (CHEMISTRY, 0.4),
            (ENERGY, 0.3),
        ],
        104,
    );
    let region = SemanticQuery::similar_in_shell(&q_sci, index.projection(), 0.6, 2.5, 5.0);
    let compound = index.search_region(&region);
    if compound.items.is_empty() {
        println!("  (no results — all science docs may fall outside the shell bounds)");
    }
    for item in &compound.items {
        let doc = docs.iter().find(|d| d.id == item.id).unwrap();
        println!(
            "  [{:<10}] r={:.2} importance={:.2}  \"{}\"",
            doc.topic,
            item.position.r,
            doc.importance,
            truncate(doc.text, 55),
        );
    }

    // -----------------------------------------------------------------------
    // Shell query: only high-importance documents (r > 2.7)
    // -----------------------------------------------------------------------
    println!("\n--- High-Importance Documents (r > 2.7, i.e. importance > 0.85) ---");
    let shell = Region::Shell(Shell::new(2.7, 5.0).unwrap());
    let important = index.search_region(&shell);
    let mut items: Vec<_> = important.items.iter().collect();
    items.sort_by(|a, b| b.position.r.partial_cmp(&a.position.r).unwrap());
    for item in items {
        let doc = docs.iter().find(|d| d.id == item.id).unwrap();
        println!(
            "  [{:<10}] r={:.2} importance={:.2}  \"{}\"",
            doc.topic,
            item.position.r,
            doc.importance,
            truncate(doc.text, 55),
        );
    }

    // -----------------------------------------------------------------------
    // Cross-domain discovery: what's near the boundary between topics?
    // -----------------------------------------------------------------------
    println!("\n--- Cross-Domain: midpoint between \"science\" and \"cooking\" ---");
    let q_cross = embed(
        &[
            (BIOLOGY, 0.4),
            (CHEMISTRY, 0.5),
            (COOKING, 0.4),
            (ENERGY, 0.3),
            (HEAT, 0.3),
            (WATER, 0.2),
        ],
        105,
    );
    let cross_results = index.search_nearest(&q_cross, 5);
    for (i, res) in cross_results.iter().enumerate() {
        let doc = docs.iter().find(|d| d.id == res.item.id).unwrap();
        println!(
            "  {}. [{:<10}] dist={:.4} rad ({:>6.2}°)  \"{}\"",
            i + 1,
            doc.topic,
            res.distance,
            res.distance.to_degrees(),
            truncate(doc.text, 50),
        );
    }
    println!("  (Cross-domain queries find docs from multiple topics!)");
}

fn run_nearest_query(
    index: &EmbeddingIndex<PcaProjection>,
    docs: &[Doc],
    query: &Embedding,
    description: &str,
    k: usize,
) {
    println!("\n--- Query: \"{description}\" (top {k}) ---");
    let results = index.search_nearest(query, k);
    for (i, res) in results.iter().enumerate() {
        let doc = docs.iter().find(|d| d.id == res.item.id).unwrap();
        println!(
            "  {}. [{:<10}] dist={:.4} rad ({:>6.2}°)  \"{}\"",
            i + 1,
            doc.topic,
            res.distance,
            res.distance.to_degrees(),
            truncate(doc.text, 55),
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}
