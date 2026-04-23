# Examples

Runnable examples live in [`sphereql/examples/`](../sphereql/examples/)
(Rust) and [`sphereql-python/examples/`](../sphereql-python/examples/)
plus [`examples/`](../examples/) (Python).

## Rust — basics

```bash
# Basic spherical math
cargo run --example basic_positioning -p sphereql --features core

# Spatial indexing and geospatial queries
cargo run --example geospatial -p sphereql --features index

# GraphQL server
cargo run --example graphql_server -p sphereql --features full

# Embedding projection
cargo run --example word_embeddings   -p sphereql --features embed
cargo run --example semantic_search   -p sphereql --features embed
cargo run --example auto_categorize   -p sphereql --features embed
```

## Rust — category enrichment & spatial analysis

```bash
# Category Enrichment Layer — inter-category graph, bridges, inner spheres
cargo run --example category_enrichment -p sphereql --features embed

# AI Knowledge Navigator — 13 analyses on the 775-concept corpus
cargo run --example ai_knowledge_navigator -p sphereql --features embed

# Spatial Analysis on S² — every geometric primitive (antipode, Voronoi,
# geodesic sweep, lunes, curvature) raw and navigator-wrapped
cargo run --example spatial_analysis -p sphereql --features embed

# End-to-end transformer embedding pipeline
cargo run --example e2e_transformer -p sphereql --features embed

# Benchmarks
cargo run --example benchmark -p sphereql --features full
```

## Rust — auto-tuning and meta-learning

```bash
# Sweep PCA vs Laplacian on either corpus (flip with SPHEREQL_CORPUS=stress)
cargo run --example auto_tune -p sphereql --features embed --release

# Tune both corpora, accumulate MetaTrainingRecords, verify the MetaModel
# predicts the right projection family from a corpus feature profile
cargo run --example meta_learn -p sphereql --features embed --release

# Warm-started hybrid: recall a config from the meta-store, then run a
# few refinement trials from that starting point
cargo run --example meta_warm_start -p sphereql --features embed --release

# L3 feedback loop: blend per-query FeedbackEvents into stored records
cargo run --example meta_feedback -p sphereql --features embed --release
```

## Python

```bash
cd sphereql-python
pip install maturin numpy
maturin develop

python examples/quickstart.py
python examples/kernel_pca.py
python examples/dataset.py

# Category enrichment (from repo root)
python examples/category_enrichment.py
```
