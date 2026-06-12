# Examples

Runnable examples live in [`sphereql-examples/examples/`](../sphereql-examples/examples/)
(Rust), [`sphereql-python-examples/`](../sphereql-python-examples/)
(Python), and [`sphereql-wasm/examples/`](../sphereql-wasm/examples/)
(browser).

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
# Full 8-phase demo: auto-tune → meta-learn → embed → spatial analysis →
# category analysis → queries → AI-enhanced divergence cartography →
# self-tune controller. Exercises every major API in a single run; an
# interactive prompt picks the corpus (or corpora) to run.
cargo run -p sphereql-examples --example full_e2e --release
```

## Rust — auto-tuning and meta-learning

```bash
# Sweep all four projection families under two contrasting quality
# metrics on either corpus (flip with SPHEREQL_CORPUS=stress)
cargo run -p sphereql-examples --example auto_tune --release

# Tune three corpora (775 built-in, 5k extended, 300 stress), accumulate
# MetaTrainingRecords, verify the MetaModel predicts the right projection
# family from a corpus feature profile
cargo run -p sphereql-examples --example meta_learn --release

# Warm-started hybrid: recall a config from the meta-store, then run a
# few refinement trials from that starting point
cargo run -p sphereql-examples --example meta_warm_start --release

# L3 feedback loop: blend per-query FeedbackEvents into stored records
cargo run -p sphereql-examples --example meta_feedback --release

# Tune with the CorpusQuality composite metric and print the per-axis
# sub-score breakdown from the TuneReport
cargo run -p sphereql-examples --example cq_e2e --release
```

## Rust — corpus tooling

```bash
# Phase-6 self-tune loop over the extended parquet corpus (dry run by
# default; --commit-confirm writes in place)
cargo run -p sphereql-examples --example corpus_self_tune --release

# Streaming bulk-corpus ingest (Wikidata SPARQL / OpenAlex shards /
# Wikidata dump; the dump source needs --features bulk-dump) — see the
# example's header for full invocations
cargo run -p sphereql-examples --example bulk_ingest --release -- --help

# 500K-corpus load smoke test (build the parquet first with
# sphereql-corpus/tools/synthesize_500k.py)
cargo run -p sphereql-examples --example load_500k_smoke --release

# Laplacian diagnostics on the built-in and extended corpora
cargo run -p sphereql-examples --example lap_diag --release
```

## Rust — lingua

```bash
# Text → ConceptGraph → SphereQL positions, with validation checks
cargo run -p sphereql-examples --example lingua_e2e
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
python sphereql-python-examples/category_enrichment_basic.py

# L1/L2/L3 metalearning walkthrough: auto_tune → MetaModel → feedback
python sphereql-python-examples/metalearning.py

# VectorStoreBridge with the full category layer
python sphereql-python-examples/vectordb_advanced.py

# Live Qdrant cluster end-to-end (needs the qdrant feature + a server)
python sphereql-python-examples/qdrant_e2e.py

# Lingua pipeline (build with: maturin develop --features lingua)
python sphereql-python-examples/lingua_e2e.py
```

See [`sphereql-python-examples/README.md`](../sphereql-python-examples/README.md)
for what each script covers.

## WASM

A self-contained browser demo lives in
[`sphereql-wasm/examples/`](../sphereql-wasm/examples/): pipeline
construction, `newWithConfig`, `autoTune`, `hierarchicalNearest`,
`MetaModel.predict`, and `FeedbackAggregator`.

```bash
cd sphereql-wasm
wasm-pack build --target web
cp -r pkg examples/pkg
cd examples && python3 -m http.server 8000
```
