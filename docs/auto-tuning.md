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

```rust,ignore
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

```rust,ignore
let space = SearchSpace::default();       // sweeps PCA + Laplacian by default
let metric = CompositeMetric::default_composite();
let strategy = SearchStrategy::Random { budget: 24, seed: 0xCAFE, max_wall_secs: None };

let (tuned, report) =
    auto_tune(input.clone(), &space, &metric, strategy, &base).unwrap();

println!(
    "best: {} score={:.4}",
    report.best_config.projection_kind.name(),
    report.best_score,
);
```

Under `Random` and `Bayesian`, `base_config` itself is evaluated as
trial 0 — counted against the budget, and for Bayesian seeding the TPE
history — so a warm-start prediction competes directly with sampled
candidates. `Grid` skips the seed trial: its trial set is defined as
the exact Cartesian enumeration of the space.

Metrics implement the `QualityMetric` trait:

- `TerritorialHealth` — mean territorial_factor across category pairs.
- `BridgeDiversity` — fraction of distinct category pairs connected by at
  least one `Genuine` bridge. Used by both default composites because it
  varies meaningfully across projections.
- `BridgeCoherence` — fraction of bridges classified `Genuine` versus
  `OverlapArtifact` / `Weak`. Available standalone; excluded from the
  default composites because it converges to ~0.50 under the
  quantile-based classification floor.
- `ClusterSilhouette` — silhouette score of the category assignment on
  S², remapped to `[0, 1]`.
- `GraphModularity` — modularity of the category assignment on a
  k-NN graph over projected positions.
- `CompositeMetric` — weight-normalized linear combination.
  `default_composite()` (30% bridge_diversity / 25% territorial_health /
  25% cluster_silhouette / 20% graph_modularity) and
  `connectivity_composite()` (40% graph_modularity / 35% bridge_diversity
  / 25% territorial_health) cover the common cases.

## L2: `MetaModel`

The tuner result can be persisted as a `MetaTrainingRecord`, keyed on a
10-feature `CorpusFeatures` profile. The default store lives at
`~/.sphereql/meta_records.json` and accumulates across runs.

```rust,ignore
let features = CorpusFeatures::extract(&input.categories, &input.embeddings).unwrap();
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

```rust,ignore
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
than a final answer. The prediction supplies values only for knobs
**not** enumerated by the space; any knob the space lists is searched
cold across its axes and the predicted value for it is ignored. Under
`Random`/`Bayesian` the predicted config is additionally evaluated as
trial 0 (see above).

Both constructors check `MetaModel::is_fitted` and return
`PipelineError::InvalidInput` for an unfitted model rather than
panicking.

Two concrete `MetaModel` impls ship:

- `NearestNeighborMetaModel` — picks the training record closest in
  z-score-normalized Euclidean distance (scale features are
  `ln(1+x)`-compressed first). Zero hyperparameters, works with
  N ≥ 1 records. `predict_blended(features, k)` aggregates per-knob
  medians + majority projection kind over the k nearest records
  (`k = 1` reproduces `predict`).
- `DistanceWeightedMetaModel` — picks the record that maximizes
  `evidence / (distance + ε)`, where evidence is the record's
  `score_lift` (`(best − mean) / (1 − mean)` over the run's trial
  distribution — cross-corpus comparable) with `best_score` as the
  fallback for legacy records. Folds demonstrated tuner signal into
  the selection, so a nearby but poorly-tuned outlier doesn't
  dominate.

Both models stratify mixed-metric training sets to the dominant
`metric_name` at fit time, since scores under different metrics aren't
comparable.

## L3: feedback

For an online-refinement loop, record `FeedbackEvent`s against the
pipeline's `corpus_id` and blend the aggregated satisfaction score back
into the stored record via
`MetaTrainingRecord::adjust_score_with_feedback(&summary, alpha)`.
`alpha` is the weight of feedback in the blended score — `0.0` ignores
feedback, `1.0` replaces the tuner's score entirely.

```rust,ignore
let mut aggregator = FeedbackAggregator::new();
aggregator.record(FeedbackEvent::now("my_corpus_v1", "q-001", 1.0));
aggregator.record(FeedbackEvent::now("my_corpus_v1", "q-002", 0.3));
let summary = aggregator.summarize("my_corpus_v1").unwrap();
let blended = record.adjust_score_with_feedback(&summary, 0.5);
```

The blend operates on the raw `best_score` scale, which is not
comparable across corpora of different difficulty — don't substitute
it for `score_lift` in cross-corpus comparisons.

The meta-model is deliberately not retrained inside this crate — L3 is
a recording + blending surface, not an online-learning framework. Users
who want to retrain pull the blended scores out and fit whatever they
want.

## Corpus self-tuning: `run_self_tune`

Orthogonal to config search: `run_self_tune`
(`sphereql-embed/src/self_tune.rs`) iteratively reweights and prunes
the *corpus itself* against a `CorpusQuality` composite under a fixed
`PipelineConfig`, stopping on plateau or an iteration cap. Contracts
worth knowing:

- The function validates its `SelfTuneConfig` up front and returns
  `Result` — smoothings and penalties must be in `[0, 1]`, boosts ≥ 1,
  `plateau_epsilon` finite and non-negative. `SelfTuneConfig` is
  serde-serializable, so runs reproduce from config files.
- Plateau detection fires **before** the iteration's reweight + prune:
  a plateaued corpus is never mutated one extra unmeasured time.
- Per-iteration `composite_score`s are *entry* scores;
  `SelfTuneReport::final_composite` is the only measurement of the
  corpus actually returned to the caller.
- The loop's reweighting is idempotent — quality is recomputed from
  the run-entry base each iteration. The standalone `reweight_in_place`
  helper is **not**: it snapshots the current qualities as its base on
  every call, so repeated calls compound the multipliers.

## Design notes

- Projections are fit **once per distinct fit-affecting hyperparameter
  tuple** inside `auto_tune` and reused across trials. PCA and Kernel
  PCA key per kind; Laplacian keys on `(k_neighbors, active_threshold)`;
  UMAP keys on `(n_neighbors, n_epochs, category_weight, min_dist)`,
  and its kNN graph + PCA warm-start are additionally cached per
  `n_neighbors` so epoch/weight/min_dist sweeps don't rebuild the graph
  (`TuneReport::umap_graph_builds` reports the cache's effectiveness).
- `SearchSpace` is kind-conditional: trials for `ProjectionKind::Pca`
  don't iterate over Laplacian or UMAP hyperparameters, and vice versa.
  The grid cardinality reflects the union, not the product.
- `CorpusFeatures::to_vec()` returns a fixed-order feature vector; the
  `category_separation_ratio` field is deliberately excluded because
  it's a derived ratio of two other features already in the vector.

## See also

- [`examples/auto_tune.rs`](../sphereql-examples/examples/auto_tune.rs) — a full
  sweep on either corpus.
- [`examples/meta_learn.rs`](../sphereql-examples/examples/meta_learn.rs) —
  cross-corpus tune → record → verify MetaModel prediction.
- [`examples/meta_warm_start.rs`](../sphereql-examples/examples/meta_warm_start.rs)
  — recall a config, refine from it.
- [`examples/meta_feedback.rs`](../sphereql-examples/examples/meta_feedback.rs)
  — L3 feedback blending in action.
- [Empirical findings](empirical-findings.md) — three-way head-to-head:
  UMAP-on-sphere wins both corpora; PCA still edges Laplacian on the
  built-in corpus while Laplacian collapses on the stress corpus. The
  metalearning framework exists to predict the winner per corpus.
