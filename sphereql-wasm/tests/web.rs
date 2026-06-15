//! `wasm-bindgen-test` smoke suite for `sphereql-wasm`.
//!
//! These run under Node (`wasm-pack test --node sphereql-wasm`) — the
//! default `wasm-bindgen-test` runner, so no `run_in_browser` configure
//! call is needed. They exercise the core WASM surface end-to-end through
//! the *real* JSON-string boundary that JS callers cross, on the actual
//! `wasm32-unknown-unknown` target rather than a host build.
//!
//! Inputs are deterministic and tiny: 6 items, 4-dim embeddings across two
//! categories. The JSON shapes match exactly what `lib.rs`'s `parse_input`
//! / `parse_query` accept.

use sphereql_wasm::{Pipeline, run_self_tune};
use wasm_bindgen_test::*;

/// 6 items, two categories, 4-dim embeddings. Two loose clusters: the
/// "alpha" rows load axis 0, the "beta" rows load axis 1. Matches
/// `parse_input`: `categories.len() == embeddings.len()`, every row the
/// same length (>= 3), all values finite.
const CORPUS_JSON: &str = r#"{
    "categories": ["alpha", "alpha", "alpha", "beta", "beta", "beta"],
    "embeddings": [
        [1.0, 0.1, 0.0, 0.2],
        [0.9, 0.2, 0.1, 0.1],
        [0.95, 0.05, 0.05, 0.15],
        [0.1, 1.0, 0.2, 0.0],
        [0.2, 0.9, 0.1, 0.1],
        [0.05, 0.95, 0.15, 0.05]
    ]
}"#;

/// A bare JSON array of numbers — the shape `parse_query` expects. Its
/// length (4) must equal the corpus embedding dim, or `require_matching_dim`
/// rejects it before the query runs.
const QUERY_JSON: &str = "[0.9, 0.1, 0.0, 0.2]";

#[wasm_bindgen_test]
fn pipeline_constructs_from_valid_input() {
    let pipeline = Pipeline::new(CORPUS_JSON).expect("valid input must construct a pipeline");
    assert_eq!(pipeline.len(), 6, "all six items should be indexed");
    assert!(!pipeline.is_empty());
}

#[wasm_bindgen_test]
fn projection_kind_defaults_to_pca() {
    let pipeline = Pipeline::new(CORPUS_JSON).expect("pipeline must construct");
    assert_eq!(
        pipeline.projection_kind(),
        "pca",
        "default projection family is PCA"
    );
}

#[wasm_bindgen_test]
fn nearest_returns_k_results_with_fields() {
    let pipeline = Pipeline::new(CORPUS_JSON).expect("pipeline must construct");
    let results = pipeline
        .nearest(QUERY_JSON, 3)
        .expect("nearest on a matching-dim query must succeed");

    assert_eq!(results.len(), 3, "k=3 must return exactly three neighbors");
    for r in &results {
        assert!(!r.id.is_empty(), "every result carries a non-empty id");
        assert!(!r.category.is_empty(), "every result carries a category");
        assert!(r.distance.is_finite(), "distances must be finite");
    }
    // Results come back distance-sorted ascending.
    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);
}

#[wasm_bindgen_test]
fn config_round_trips_as_json() {
    let pipeline = Pipeline::new(CORPUS_JSON).expect("pipeline must construct");
    let config_json = pipeline.config().expect("config() must serialize");

    // The string `config()` hands back must be valid JSON, and parsing it
    // must surface the PCA default we asserted above.
    let parsed: serde_json::Value =
        serde_json::from_str(&config_json).expect("config() must return parseable JSON");
    assert_eq!(
        parsed["projection_kind"], "Pca",
        "serialized config carries the PCA projection kind"
    );
}

#[wasm_bindgen_test]
fn mismatched_lengths_are_rejected() {
    // Two categories but three embeddings — `parse_input` must reject this
    // before any pipeline geometry is built.
    let bad = r#"{
        "categories": ["alpha", "beta"],
        "embeddings": [
            [1.0, 0.1, 0.0, 0.2],
            [0.1, 1.0, 0.2, 0.0],
            [0.5, 0.5, 0.1, 0.1]
        ]
    }"#;
    assert!(
        Pipeline::new(bad).is_err(),
        "categories.len != embeddings.len must be an Err"
    );
}

#[wasm_bindgen_test]
fn query_dim_mismatch_is_rejected() {
    // The corpus is 4-dim; a 3-dim query must be rejected by
    // `require_matching_dim` rather than panicking downstream.
    let pipeline = Pipeline::new(CORPUS_JSON).expect("pipeline must construct");
    let short_query = "[0.9, 0.1, 0.0]";
    assert!(
        pipeline.nearest(short_query, 2).is_err(),
        "a query whose dim differs from the corpus must be an Err"
    );
}

#[wasm_bindgen_test]
fn run_self_tune_smoke() {
    // `runSelfTune` takes a JSON *array* of TunableConcept objects (not the
    // categories/embeddings shape). `features` are `[axis, weight]` pairs;
    // axes must be < 128 (the synthetic embedder's DIM). Six concepts over
    // two categories is enough for the pipeline (>= 3) to fit.
    let concepts = r#"[
        {"label":"a0","category":"alpha","features":[[0,1.0]],"quality":0.8,"axis_coherence":0.7,"bridge_degree":1,"source_confidence":0.6,"home_affinity":0.8,"source":"synthetic","openalex_id":null},
        {"label":"a1","category":"alpha","features":[[1,1.0]],"quality":0.8,"axis_coherence":0.7,"bridge_degree":1,"source_confidence":0.6,"home_affinity":0.8,"source":"synthetic","openalex_id":null},
        {"label":"a2","category":"alpha","features":[[2,1.0]],"quality":0.8,"axis_coherence":0.7,"bridge_degree":1,"source_confidence":0.6,"home_affinity":0.8,"source":"synthetic","openalex_id":null},
        {"label":"b0","category":"beta","features":[[3,1.0]],"quality":0.8,"axis_coherence":0.7,"bridge_degree":1,"source_confidence":0.6,"home_affinity":0.8,"source":"synthetic","openalex_id":null},
        {"label":"b1","category":"beta","features":[[4,1.0]],"quality":0.8,"axis_coherence":0.7,"bridge_degree":1,"source_confidence":0.6,"home_affinity":0.8,"source":"synthetic","openalex_id":null},
        {"label":"b2","category":"beta","features":[[5,1.0]],"quality":0.8,"axis_coherence":0.7,"bridge_degree":1,"source_confidence":0.6,"home_affinity":0.8,"source":"synthetic","openalex_id":null}
    ]"#;
    // Defaults set `min_concepts_per_category: 50`, which would prune this
    // tiny corpus to nothing; drop the floor and cap iterations so the
    // smoke run stays fast and deterministic.
    let opts =
        r#"{"cfg":{"max_iterations":2,"min_concepts_per_category":1,"min_quality_to_keep":0.0}}"#;

    let report = run_self_tune(concepts, opts).expect("self-tune on a valid corpus must succeed");
    assert!(
        !report.iterations.is_empty(),
        "at least one iteration must be recorded"
    );
    assert_eq!(
        report.tuned_concepts.len(),
        6,
        "no concepts pruned with the floor disabled"
    );
    assert!(
        matches!(
            report.stopped_reason.as_str(),
            "plateau" | "max_iterations" | "prune_floor_hit"
        ),
        "stopped_reason must be one of the known StopReason names, got {:?}",
        report.stopped_reason
    );
}
