//! Retrieval benchmark: Vanilla ANN vs SphereQL Hybrid vs SphereQL-only.
//!
//! Run with:
//!   cargo run --example benchmark -p sphereql --features full --release

use std::collections::HashMap;
use std::time::Instant;

use sphereql::embed::pipeline::{SphereQLOutput, SphereQLQuery};
use sphereql::vectordb::{
    BridgeConfig, InMemoryStore, VectorRecord, VectorStoreBridge,
    store::VectorStore,
};

const NUM_CLUSTERS: usize = 20;
const POINTS_PER_CLUSTER: usize = 500;
const TOTAL: usize = NUM_CLUSTERS * POINTS_PER_CLUSTER;
const DIM: usize = 384;
const NUM_QUERIES: usize = 200;
const WARMUP: usize = 50;
const SEED: u64 = 42;

// ── PRNG ─────────────────────────────────────────────────────────────────

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Dataset generation ───────────────────────────────────────────────────

fn normalize(v: &mut [f64]) {
    let mag: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag > f64::EPSILON {
        v.iter_mut().for_each(|x| *x /= mag);
    }
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ma: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if ma < f64::EPSILON || mb < f64::EPSILON {
        return 0.0;
    }
    dot / (ma * mb)
}

struct Dataset {
    records: Vec<VectorRecord>,
    queries: Vec<QueryData>,
}

struct QueryData {
    vector: Vec<f64>,
    _cluster: usize,
}

fn generate_dataset() -> Dataset {
    let mut rng = SplitMix64::new(SEED);
    let mut records = Vec::with_capacity(TOTAL);
    let mut centroids = Vec::with_capacity(NUM_CLUSTERS);

    for c in 0..NUM_CLUSTERS {
        let mut centroid: Vec<f64> = (0..DIM).map(|_| rng.normal()).collect();
        normalize(&mut centroid);
        centroids.push(centroid.clone());

        for j in 0..POINTS_PER_CLUSTER {
            let mut point: Vec<f64> = centroid
                .iter()
                .map(|&x| x + rng.normal() * 0.15)
                .collect();
            normalize(&mut point);

            let id = format!("p-{}", c * POINTS_PER_CLUSTER + j);
            let record = VectorRecord::new(id, point)
                .with_metadata("category", serde_json::json!(format!("cluster-{c}")));
            records.push(record);
        }
    }

    // Pick query points: evenly sample from clusters
    let mut queries = Vec::with_capacity(NUM_QUERIES);
    let per_cluster = NUM_QUERIES / NUM_CLUSTERS;
    for c in 0..NUM_CLUSTERS {
        for j in 0..per_cluster {
            let idx = c * POINTS_PER_CLUSTER + (j * 7 + 3) % POINTS_PER_CLUSTER;
            queries.push(QueryData {
                vector: records[idx].vector.clone(),
                _cluster: c,
            });
        }
    }

    Dataset { records, queries }
}

/// Compute ground truth: for each query, rank ALL points by cosine similarity.
fn compute_ground_truth(dataset: &Dataset) -> Vec<Vec<(usize, f64)>> {
    dataset
        .queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(usize, f64)> = dataset
                .records
                .iter()
                .enumerate()
                .map(|(i, r)| (i, cosine_sim(&q.vector, &r.vector)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored
        })
        .collect()
}

// ── Metrics ──────────────────────────────────────────────────────────────

fn precision_at_k(retrieved_ids: &[String], truth_ids: &[String], k: usize) -> f64 {
    let truth_set: std::collections::HashSet<&str> =
        truth_ids.iter().take(k).map(|s| s.as_str()).collect();
    let hits = retrieved_ids
        .iter()
        .take(k)
        .filter(|id| truth_set.contains(id.as_str()))
        .count();
    hits as f64 / k as f64
}

fn recall_at_k(retrieved_ids: &[String], truth_ids: &[String], k: usize) -> f64 {
    let truth_set: std::collections::HashSet<&str> =
        truth_ids.iter().take(k).map(|s| s.as_str()).collect();
    if truth_set.is_empty() {
        return 1.0;
    }
    let hits = retrieved_ids
        .iter()
        .take(k)
        .filter(|id| truth_set.contains(id.as_str()))
        .count();
    hits as f64 / truth_set.len() as f64
}

fn ndcg_at_k(
    retrieved_ids: &[String],
    truth_relevance: &HashMap<String, f64>,
    k: usize,
) -> f64 {
    let dcg: f64 = retrieved_ids
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = truth_relevance.get(id).copied().unwrap_or(0.0);
            rel / (i as f64 + 2.0).log2()
        })
        .sum();

    let mut ideal_rels: Vec<f64> = truth_relevance.values().copied().collect();
    ideal_rels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f64 = ideal_rels
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| rel / (i as f64 + 2.0).log2())
        .sum();

    if idcg < f64::EPSILON {
        return 1.0;
    }
    dcg / idcg
}

fn p99(mut latencies: Vec<f64>) -> f64 {
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((latencies.len() as f64 * 0.99) as usize).min(latencies.len() - 1);
    latencies[idx]
}

// ── Result types ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct MethodResult {
    method: String,
    k: usize,
    precision: f64,
    recall: f64,
    ndcg: f64,
    mean_us: f64,
    p99_us: f64,
}

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    eprintln!("Generating dataset: {TOTAL} points, {DIM}-d, {NUM_CLUSTERS} clusters...");
    let dataset = generate_dataset();

    eprintln!("Computing ground truth for {NUM_QUERIES} queries...");
    let ground_truth = compute_ground_truth(&dataset);

    // Build ground truth ID lists and relevance maps
    let truth_ids: Vec<Vec<String>> = ground_truth
        .iter()
        .map(|gt| {
            gt.iter()
                .map(|(idx, _)| dataset.records[*idx].id.clone())
                .collect()
        })
        .collect();

    let truth_relevance: Vec<HashMap<String, f64>> = ground_truth
        .iter()
        .map(|gt| {
            gt.iter()
                .map(|(idx, sim)| (dataset.records[*idx].id.clone(), *sim))
                .collect()
        })
        .collect();

    // Set up store and bridge
    eprintln!("Building index...");
    let build_start = Instant::now();

    let store = InMemoryStore::new("benchmark", DIM);
    store.upsert(&dataset.records).await.unwrap();

    let config = BridgeConfig {
        batch_size: 1000,
        max_records: TOTAL + 1,
        ..Default::default()
    };
    let mut bridge = VectorStoreBridge::new(store, config);

    let category_fn = |r: &VectorRecord| {
        r.metadata
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    };
    bridge.build_pipeline(category_fn).await.unwrap();

    let build_time_ms = build_start.elapsed().as_millis();
    let evr = bridge.projection().unwrap().explained_variance_ratio();

    eprintln!("Index built in {} ms, PCA EVR: {:.1}%", build_time_ms, evr * 100.0);

    let ks = [1, 5, 10, 20];
    let recall_multipliers = [2, 4, 8];
    let mut all_results: Vec<MethodResult> = Vec::new();

    // --- Warm-up ---
    eprintln!("Warming up ({WARMUP} queries)...");
    for q in dataset.queries.iter().take(WARMUP) {
        let _ = bridge.store().search(&q.vector, 20).await;
        let _ = bridge.hybrid_search(&q.vector, 20, 80).await;
        let _ = bridge.query(SphereQLQuery::Nearest { k: 20 }, &q.vector);
    }

    // --- Vanilla ANN ---
    eprintln!("Running Vanilla ANN...");
    for &k in &ks {
        let mut latencies = Vec::with_capacity(NUM_QUERIES);
        let mut precisions = Vec::with_capacity(NUM_QUERIES);
        let mut recalls = Vec::with_capacity(NUM_QUERIES);
        let mut ndcgs = Vec::with_capacity(NUM_QUERIES);

        for (qi, q) in dataset.queries.iter().enumerate() {
            let start = Instant::now();
            let results = bridge.store().search(&q.vector, k).await.unwrap();
            let elapsed = start.elapsed().as_micros() as f64;
            latencies.push(elapsed);

            let ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
            precisions.push(precision_at_k(&ids, &truth_ids[qi], k));
            recalls.push(recall_at_k(&ids, &truth_ids[qi], k));
            ndcgs.push(ndcg_at_k(&ids, &truth_relevance[qi], k));
        }

        all_results.push(MethodResult {
            method: "Vanilla ANN".into(),
            k,
            precision: precisions.iter().sum::<f64>() / NUM_QUERIES as f64,
            recall: recalls.iter().sum::<f64>() / NUM_QUERIES as f64,
            ndcg: ndcgs.iter().sum::<f64>() / NUM_QUERIES as f64,
            mean_us: latencies.iter().sum::<f64>() / NUM_QUERIES as f64,
            p99_us: p99(latencies),
        });
    }

    // --- SphereQL-only ---
    eprintln!("Running SphereQL-only...");
    for &k in &ks {
        let mut latencies = Vec::with_capacity(NUM_QUERIES);
        let mut precisions = Vec::with_capacity(NUM_QUERIES);
        let mut recalls = Vec::with_capacity(NUM_QUERIES);
        let mut ndcgs = Vec::with_capacity(NUM_QUERIES);

        for (qi, q) in dataset.queries.iter().enumerate() {
            let start = Instant::now();
            let output = bridge
                .query(SphereQLQuery::Nearest { k }, &q.vector)
                .unwrap();
            let elapsed = start.elapsed().as_micros() as f64;
            latencies.push(elapsed);

            let ids: Vec<String> = match output {
                SphereQLOutput::Nearest(items) => items.iter().map(|r| r.id.clone()).collect(),
                _ => Vec::new(),
            };
            // Map pipeline IDs (s-NNNN) back to record IDs (p-NNNN)
            let mapped_ids: Vec<String> = ids
                .iter()
                .filter_map(|sid| {
                    let idx: usize = sid.strip_prefix("s-")?.parse().ok()?;
                    Some(dataset.records.get(idx)?.id.clone())
                })
                .collect();

            precisions.push(precision_at_k(&mapped_ids, &truth_ids[qi], k));
            recalls.push(recall_at_k(&mapped_ids, &truth_ids[qi], k));
            ndcgs.push(ndcg_at_k(&mapped_ids, &truth_relevance[qi], k));
        }

        all_results.push(MethodResult {
            method: "SphereQL-only".into(),
            k,
            precision: precisions.iter().sum::<f64>() / NUM_QUERIES as f64,
            recall: recalls.iter().sum::<f64>() / NUM_QUERIES as f64,
            ndcg: ndcgs.iter().sum::<f64>() / NUM_QUERIES as f64,
            mean_us: latencies.iter().sum::<f64>() / NUM_QUERIES as f64,
            p99_us: p99(latencies),
        });
    }

    // --- SphereQL Hybrid (varying recall_k) ---
    for &mult in &recall_multipliers {
        eprintln!("Running SphereQL Hybrid (recall=k*{mult})...");
        for &k in &ks {
            let recall_k = k * mult;
            let mut latencies = Vec::with_capacity(NUM_QUERIES);
            let mut precisions = Vec::with_capacity(NUM_QUERIES);
            let mut recalls = Vec::with_capacity(NUM_QUERIES);
            let mut ndcgs = Vec::with_capacity(NUM_QUERIES);

            for (qi, q) in dataset.queries.iter().enumerate() {
                let start = Instant::now();
                let results = bridge
                    .hybrid_search(&q.vector, k, recall_k)
                    .await
                    .unwrap();
                let elapsed = start.elapsed().as_micros() as f64;
                latencies.push(elapsed);

                let ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
                precisions.push(precision_at_k(&ids, &truth_ids[qi], k));
                recalls.push(recall_at_k(&ids, &truth_ids[qi], k));
                ndcgs.push(ndcg_at_k(&ids, &truth_relevance[qi], k));
            }

            all_results.push(MethodResult {
                method: format!("Hybrid (r=k*{mult})"),
                k,
                precision: precisions.iter().sum::<f64>() / NUM_QUERIES as f64,
                recall: recalls.iter().sum::<f64>() / NUM_QUERIES as f64,
                ndcg: ndcgs.iter().sum::<f64>() / NUM_QUERIES as f64,
                mean_us: latencies.iter().sum::<f64>() / NUM_QUERIES as f64,
                p99_us: p99(latencies),
            });
        }
    }

    // --- Output ---
    println!("## SphereQL Hybrid Search Benchmark");
    println!("Dataset: {TOTAL} points, {DIM}-d, {NUM_CLUSTERS} clusters");
    println!("Queries: {NUM_QUERIES}");
    println!("PCA explained variance: {:.1}%", evr * 100.0);
    println!("Index build time: {build_time_ms} ms");
    println!();
    println!(
        "| {:<20} | {:>2} | {:>11} | {:>8} | {:>6} | {:>9} | {:>8} |",
        "Method", "k", "Precision@k", "Recall@k", "nDCG@k", "Mean μs", "p99 μs"
    );
    println!(
        "|{:-<22}|{:-<4}|{:-<13}|{:-<10}|{:-<8}|{:-<11}|{:-<10}|",
        "", "", "", "", "", "", ""
    );

    for r in &all_results {
        println!(
            "| {:<20} | {:>2} | {:>11.4} | {:>8.4} | {:>6.4} | {:>9.1} | {:>8.1} |",
            r.method, r.k, r.precision, r.recall, r.ndcg, r.mean_us, r.p99_us
        );
    }

    // JSON output
    let json_results: Vec<serde_json::Value> = all_results
        .iter()
        .map(|r| {
            serde_json::json!({
                "method": r.method,
                "k": r.k,
                "precision_at_k": r.precision,
                "recall_at_k": r.recall,
                "ndcg_at_k": r.ndcg,
                "mean_latency_us": r.mean_us,
                "p99_latency_us": r.p99_us,
            })
        })
        .collect();

    let json_output = serde_json::json!({
        "dataset": {
            "num_points": TOTAL,
            "dimension": DIM,
            "num_clusters": NUM_CLUSTERS,
            "num_queries": NUM_QUERIES,
        },
        "pca_explained_variance_ratio": evr,
        "index_build_time_ms": build_time_ms,
        "results": json_results,
    });

    std::fs::write(
        "benchmark_results.json",
        serde_json::to_string_pretty(&json_output).unwrap(),
    )
    .unwrap();
    eprintln!("Results written to benchmark_results.json");
}
