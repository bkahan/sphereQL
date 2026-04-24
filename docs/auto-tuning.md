# Auto-tuning and meta-learning

sphereQL's pipeline has many tunable constants — projection family,
bridge thresholds, inner-sphere gates, routing EVR cutoffs, Laplacian
hyperparameters. They all live in a single `PipelineConfig` hierarchy
and every one is reachable by the auto-tuner.

## The three layers

- **L1 — per-corpus search.** `auto_tune` sweeps a discrete
  `SearchSpace` under one of three strategies (Grid / Random / Bayesian
  TPE-lite) and returns the best pipeline plus a `TuneReport`.
- **L2 — cross-corpus generalization.** A fitted `MetaModel` maps
  `CorpusFeatures` → `PipelineConfig` so a new corpus can skip search
  (or warm-start it).
- **L3 — online refinement.** `FeedbackEvent`s record per-query user
  satisfaction; `MetaTrainingRecord::adjust_score_with_feedback` blends
  observed satisfaction into the stored record's score.

## L1: `auto_tune`

Every tunable constant lives in `PipelineConfig`. Projection family is
a first-class field, so the tuner can compare families on equal
footing with the rest of the knobs.

```rust
use sphereql::embed::*;

let mut base = PipelineConfig::default();
base.projection_kind = ProjectionKind::LaplacianEigenmap;
base.laplacian.k_neighbors = 20;

// Build a pipeline directly with a custom config
let pipeline =
    SphereQLPipeline::new_with_config(input.clone(), base.clone()).unwrap();
```

`auto_tune` sweeps a `SearchSpace` under
`SearchStrategy::{Grid, Random, Bayesian}` and returns the best pipeline
plus a `TuneReport`:

```rust
let space = SearchSpace::default();       // sweeps PCA + Laplacian by default
let metric = CompositeMetric::default_composite();
let strategy = SearchStrategy::Random { budget: 24, seed: 0xCAFE };

let (tuned, report) =
    auto_tune(input.clone(), &space, &metric, strategy, &base).unwrap();

println!(
    "best: {} score={:.4}",
    report.best_config.projection_kind.name(),
    report.best_score,
);
```

Metrics implement the `QualityMetric` trait:

- `TerritorialHealth` — mean territorial_factor across category pairs.
- `BridgeCoherence` — fraction of bridges classified `Genuine` versus
  `OverlapArtifact` / `Weak`.
- `ClusterSilhouette` — silhouette score of the category assignment on
  S², remapped to `[0, 1]`.
- `GraphModularity` — modularity of the category assignment on a
  k-NN graph over projected positions.
- `CompositeMetric` — weight-normalized linear combination.
  `default_composite()` and `connectivity_composite()` cover the common
  cases.

## L2: `MetaModel`

The tuner result can be persisted as a `MetaTrainingRecord`, keyed on a
10-feature `CorpusFeatures` profile. The default store lives at
`~/.sphereql/meta_records.json` and accumulates across runs.

```rust
let features = CorpusFeatures::extract(&input.categories, &input.embeddings);
let record = MetaTrainingRecord::from_tune_result(
    "my_corpus_v1",
    features,
    &report,
    "random_24",
);
record.append_to_default_store().unwrap();
```

On a new corpus, a `MetaModel` predicts the config without running the
tuner:

```rust
let records = MetaTrainingRecord::load_default_store().unwrap();
let mut model = NearestNeighborMetaModel::default();
model.fit(&records);

// Recall only — fast, zero tuner trials.
let (pipeline, _features, _cfg) =
    SphereQLPipeline::new_from_metamodel(input.clone(), &model).unwrap();
```

`new_from_metamodel_tuned` takes the same inputs plus a `SearchSpace`
and runs a small tuner pass warm-started from the model's prediction
— useful when you want the recall to be a *starting point* rather
than a final answer.

Two concrete `MetaModel` impls ship:

- `NearestNeighborMetaModel` — picks the training record closest in
  z-score-normalized Euclidean distance. Zero hyperparameters, works
  with N ≥ 1 records.
- `DistanceWeightedMetaModel` — picks the record that maximizes
  `best_score / (distance + ε)`. Folds the training record's score into
  the selection, so a nearby but poorly-tuned outlier doesn't dominate.

## L3: feedback

For an online-refinement loop, record `FeedbackEvent`s against the
pipeline's `corpus_id` and blend the aggregated satisfaction score back
into the stored record via
`MetaTrainingRecord::adjust_score_with_feedback(&summary, alpha)`.
`alpha` is the weight of feedback in the blended score — `0.0` ignores
feedback, `1.0` replaces the tuner's score entirely.

```rust
let events = vec![
    FeedbackEvent::new("my_corpus_v1", 1.0),
    FeedbackEvent::new("my_corpus_v1", 0.3),
];
let aggregator = FeedbackAggregator::from_events(events);
let summary = aggregator.summary();
let blended = record.adjust_score_with_feedback(&summary, 0.5);
```

The meta-model is deliberately not retrained inside this crate — L3 is
a recording + blending surface, not an online-learning framework. Users
who want to retrain pull the blended scores out and fit whatever they
want.

## Design notes

- Projections are fit **once per distinct fit-affecting hyperparameter
  tuple** inside `auto_tune` and reused across trials, so projection
  fitting contributes only a one-time prefit cost per unique
  (`ProjectionKind`, Laplacian params) combination.
- `SearchSpace` is kind-conditional: trials for `ProjectionKind::Pca`
  don't iterate over Laplacian hyperparameters, and vice versa. The
  grid cardinality reflects the union, not the product.
- `CorpusFeatures::to_vec()` returns a fixed-order feature vector; the
  `category_separation_ratio` field is deliberately excluded because
  it's a derived ratio of two other features already in the vector.

## See also

- [`examples/auto_tune.rs`](../sphereql/examples/auto_tune.rs) — a full
  sweep on either corpus.
- [`examples/meta_learn.rs`](../sphereql/examples/meta_learn.rs) —
  cross-corpus tune → record → verify MetaModel prediction.
- [`examples/meta_warm_start.rs`](../sphereql/examples/meta_warm_start.rs)
  — recall a config, refine from it.
- [`examples/meta_feedback.rs`](../sphereql/examples/meta_feedback.rs)
  — L3 feedback blending in action.
- [Empirical findings](empirical-findings.md) — PCA wins the built-in
  corpus, Laplacian wins the stress corpus. The metalearning framework
  exists because neither is right on its own.
