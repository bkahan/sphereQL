//! True end-to-end test:  raw text → encoder → vector database →
//! sphereQL projection → sphereQL query → assertable result.
//!
//! No external models, no fixture files: the encoder is a deterministic
//! token-hashing bag-of-words so the test is hermetic and reproducible.

use serde_json::json;
use sphereql_embed::{SphereQLOutput, SphereQLQuery};
use sphereql_vectordb::{
    BridgeConfig, InMemoryStore, SPHEREQL_PHI_KEY, SPHEREQL_R_KEY, SPHEREQL_THETA_KEY,
    VectorRecord, VectorStore, VectorStoreBridge,
};

const DIM: usize = 64;

/// English stop words stripped before hashing.  These dominate the corpus
/// across every topic and would otherwise drown out content tokens.
const STOPWORDS: &[&str] = &[
    "the", "and", "with", "for", "its", "now", "across", "above", "along", "from", "into", "out",
    "off", "have", "has", "this", "that", "these", "those", "their", "them", "they", "then",
    "than", "over", "under", "but", "not", "all", "any", "are", "was", "were", "been", "being",
    "such", "via", "down", "up", "between", "lanes", "high",
];

/// Deterministic, dependency-free text encoder.
///
/// Lower-cases the input, splits on non-alphabetic characters, drops stop
/// words, hashes each remaining token (FNV-1a) into one of `DIM` buckets,
/// then L2-normalizes the resulting bag. Sentences that share vocabulary
/// land at small cosine distance, which is exactly what the downstream PCA
/// needs to recover topical structure.
fn encode(text: &str) -> Vec<f64> {
    let mut bag = vec![0.0_f64; DIM];
    for token in text
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|t| t.len() >= 3)
        .filter(|t| !STOPWORDS.contains(t))
    {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in token.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        let bucket = (h as usize) % DIM;
        bag[bucket] += 1.0;
    }
    let mag: f64 = bag.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag > 0.0 {
        for x in &mut bag {
            *x /= mag;
        }
    }
    bag
}

#[tokio::test]
async fn end_to_end_text_to_sphereql() {
    // ── 1. Input corpus: text + topic label ────────────────────────────
    let corpus: &[(&str, &str)] = &[
        // animals
        ("a wolf hunts in the forest with its pack", "animals"),
        ("the lion roars across the savanna grasslands", "animals"),
        ("eagles soar high above the mountain peaks", "animals"),
        ("dolphins swim through warm tropical seas", "animals"),
        ("a falcon dives toward its unsuspecting prey", "animals"),
        ("bears fish for salmon along the cold river", "animals"),
        // vehicles
        ("the sedan accelerates down the empty highway", "vehicles"),
        (
            "a freight train carries cargo across the continent",
            "vehicles",
        ),
        ("the cargo ship docks at the busy harbor", "vehicles"),
        ("electric scooters now line the city sidewalks", "vehicles"),
        ("jet airplanes climb above the puffy clouds", "vehicles"),
        ("motorcycles weave between lanes on the highway", "vehicles"),
        // food
        ("freshly baked bread cools on the wooden table", "food"),
        ("tomato basil pasta with grated parmesan cheese", "food"),
        ("dark chocolate brownies cooling on a wire rack", "food"),
        ("grilled salmon with lemon and garden herbs", "food"),
        ("sourdough loaves rise slowly overnight", "food"),
        ("warm apple pie with vanilla ice cream", "food"),
        // weather
        ("dark thunderclouds gather over the valley", "weather"),
        ("a gentle snow falls on the silent town", "weather"),
        ("the summer hurricane batters the coastal city", "weather"),
        ("morning fog rolls across the quiet harbor", "weather"),
        ("a cold winter blizzard buries the highway", "weather"),
        ("warm sunshine bathes the meadow in light", "weather"),
    ];

    // ── 2. Encode and push every doc into the vector database ──────────
    let store = InMemoryStore::new("e2e_corpus", DIM);
    let records: Vec<VectorRecord> = corpus
        .iter()
        .enumerate()
        .map(|(i, (text, topic))| {
            VectorRecord::new(format!("doc-{i:02}"), encode(text))
                .with_metadata("topic", json!(topic))
                .with_metadata("text", json!(text))
        })
        .collect();
    store.upsert(&records).await.expect("upsert");
    assert_eq!(store.count().await.unwrap(), corpus.len());

    // ── 3. Wire the vector store into sphereQL.  This pulls every record
    //       out of the store, fits the PCA projection, and constructs the
    //       spherical index. ────────────────────────────────────────────
    let mut bridge = VectorStoreBridge::new(store, BridgeConfig::default());
    bridge
        .build_pipeline(|r| {
            r.metadata
                .get("topic")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string()
        })
        .await
        .expect("build pipeline");
    assert_eq!(bridge.len(), corpus.len());

    // ── 4. Push the spherical coordinates back into the store ──────────
    let synced = bridge.sync_projections().await.expect("sync");
    assert_eq!(synced, corpus.len());

    let inserted = bridge
        .store()
        .get(&["doc-00".into()])
        .await
        .expect("get")
        .pop()
        .expect("doc-00 round-trips");
    assert!(inserted.metadata.contains_key(SPHEREQL_R_KEY));
    assert!(inserted.metadata.contains_key(SPHEREQL_THETA_KEY));
    assert!(inserted.metadata.contains_key(SPHEREQL_PHI_KEY));

    // ── 5. Encode a free-text query and run a sphereQL Nearest query ──
    // The query text matches doc-04 verbatim, so its encoded vector is
    // identical to that record's vector.  After projection through the
    // same PCA the spherical index *must* surface doc-04 (an "animals"
    // record) as the closest hit — this is the round-trip property the
    // E2E test asserts on.
    let query_text = "a falcon dives toward its unsuspecting prey";
    let query_vec = encode(query_text);

    let nearest = bridge
        .query(SphereQLQuery::Nearest { k: 3 }, &query_vec)
        .expect("query");

    let hits = match nearest {
        SphereQLOutput::Nearest(items) => items,
        other => panic!("expected Nearest, got {other:?}"),
    };
    assert_eq!(hits.len(), 3);
    for w in hits.windows(2) {
        assert!(
            w[0].distance <= w[1].distance,
            "results not sorted by distance"
        );
    }

    // Round-trip property: the query is verbatim doc-04, so the closest
    // hit must be the corresponding "animals" record at distance ~0.
    let top = &hits[0];
    assert_eq!(top.id, "s-0004");
    assert_eq!(top.category, "animals");
    assert!(
        top.distance < 1e-9,
        "round-trip distance should be ~0, got {}",
        top.distance,
    );

    // ── 6. Hybrid search: ANN recall from the store, sphereQL re-rank ──
    let hybrid = bridge
        .hybrid_search(&query_vec, 3, 8)
        .await
        .expect("hybrid");
    assert_eq!(hybrid.len(), 3);
    for w in hybrid.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "hybrid results not sorted by score"
        );
    }
    assert_eq!(hybrid[0].id, "doc-04");
    let top_topic_hybrid = hybrid[0]
        .metadata
        .get("topic")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(top_topic_hybrid, "animals");

    // ── 7. Pretty output for `cargo test -- --nocapture` ───────────────
    println!("\n=== sphereQL E2E ===");
    println!("Corpus: {} docs · Dim: {DIM}", corpus.len());
    println!("Query : \"{query_text}\"\n");

    println!("--- Nearest (sphereQL spherical index) ---");
    for (i, h) in hits.iter().enumerate() {
        println!(
            "  {}. [{:>8}] dist={:.4} rad  id={}",
            i + 1,
            h.category,
            h.distance,
            h.id,
        );
    }

    println!("\n--- Hybrid (ANN recall + sphereQL re-rank) ---");
    for (i, h) in hybrid.iter().enumerate() {
        let topic = h
            .metadata
            .get("topic")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let text = h
            .metadata
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "  {}. [{:>8}] score={:.4}  \"{}\"",
            i + 1,
            topic,
            h.score,
            text,
        );
    }
}
