# Empirical: when does each projection win?

The right projection is corpus-dependent. Two sanity checks, same
pipeline, same tuner, same metrics, two different corpora, opposite
winners:

| Corpus | Metric | PCA score | Laplacian score | Winner |
|---|---|---|---|---|
| Built-in 775-concept (31 academic domains, hand-crafted 128-d) | `default_composite` | best | lower | **PCA** |
| Built-in 775-concept | `connectivity_composite` | best | lower | **PCA** |
| Stress 300-concept (10 categories, 2-axis signatures, high noise) | `default_composite` | 0.9606 | 1.0000 | **Laplacian** |
| Stress 300-concept | `connectivity_composite` | 0.9265 | 0.9500 | **Laplacian** |

Dense, low-noise embeddings where variance tracks meaning: PCA wins.
Sparse, noise-heavy regimes where variance is dominated by noise and the
real signal is in the co-activation graph: Laplacian wins. This is the
whole motivation for the auto-tuner + meta-model layer — no single
projection is right, so the pipeline picks one per corpus.

## Reproduce

```bash
# Built-in corpus (default)
cargo run --example auto_tune -p sphereql --features embed --release

# Stress corpus
SPHEREQL_CORPUS=stress \
    cargo run --example auto_tune -p sphereql --features embed --release

# Both corpora at once, with MetaModel verification
cargo run --example meta_learn -p sphereql --features embed --release
```

[`examples/meta_learn.rs`](../sphereql/examples/meta_learn.rs) also
verifies that a `NearestNeighborMetaModel` fitted on both records can
predict the winning projection family from each corpus's feature
profile without re-running the tuner.

## Metric details

- `default_composite` — 40% `BridgeCoherence` + 35% `TerritorialHealth`
  + 25% `ClusterSilhouette`.
- `connectivity_composite` — 50% `GraphModularity` + 30%
  `BridgeCoherence` + 20% `TerritorialHealth`.

See the [`QualityMetric`
docs](https://docs.rs/sphereql-embed/latest/sphereql_embed/quality_metric/)
or [auto-tuning guide](auto-tuning.md) for what each metric measures.
