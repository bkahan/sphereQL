# Examples

Runnable examples live in [`sphereql-examples/examples/`](../sphereql-examples/examples/)
(Rust) and [`sphereql-python-examples/`](../sphereql-python-examples/)
(Python).

The Rust examples crate pulls in `sphereql` with the `full` feature set
plus each sub-crate directly, so individual invocations no longer need
`--features` flags.

## Rust — basics

```bash
# Basic spherical math
cargo run -p sphereql-examples --example basic_positioning

# Spatial indexing and geospatial queries
cargo run -p sphereql-examples --example geospatial

# GraphQL server
cargo run -p sphereql-examples --example graphql_server

# Embedding projection
cargo run -p sphereql-examples --example word_embeddings
cargo run -p sphereql-examples --example semantic_search
cargo run -p sphereql-examples --example auto_categorize
```

## Rust — category enrichment & spatial analysis

```bash
# Category Enrichment Layer — inter-category graph, bridges, inner spheres
cargo run -p sphereql-examples --example category_enrichment

# AI Knowledge Navigator — 13 analyses on the 775-concept corpus
cargo run -p sphereql-examples --example ai_knowledge_navigator

# Spatial Analysis on S² — every geometric primitive (antipode, Voronoi,
# geodesic sweep, lunes, curvature) raw and navigator-wrapped
cargo run -p sphereql-examples --example spatial_analysis

# End-to-end transformer embedding pipeline
cargo run -p sphereql-examples --example e2e_transformer

# Benchmarks
cargo run -p sphereql-examples --example benchmark
```

## Rust — comprehensive end-to-end demo

```bash
# Full 7-phase demo: auto-tune → meta-learn → embed → spatial analysis →
# category analysis → queries → AI-enhanced divergence cartography.
# Exercises every major API in a single run.
cargo run -p sphereql-examples --example full_e2e --release
```

## Rust — auto-tuning and meta-learning

```bash
# Sweep PCA vs Laplacian on either corpus (flip with SPHEREQL_CORPUS=stress)
cargo run -p sphereql-examples --example auto_tune --release

# Tune both corpora, accumulate MetaTrainingRecords, verify the MetaModel
# predicts the right projection family from a corpus feature profile
cargo run -p sphereql-examples --example meta_learn --release

# Warm-started hybrid: recall a config from the meta-store, then run a
# few refinement trials from that starting point
cargo run -p sphereql-examples --example meta_warm_start --release

# L3 feedback loop: blend per-query FeedbackEvents into stored records
cargo run -p sphereql-examples --example meta_feedback --release
```

## Python

```bash
cd sphereql-python
pip install maturin numpy
maturin develop
cd ..

python sphereql-python-examples/quickstart.py
python sphereql-python-examples/kernel_pca.py
python sphereql-python-examples/dataset.py
python sphereql-python-examples/category_enrichment.py
```
