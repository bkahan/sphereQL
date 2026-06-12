# sphereQL Python Examples

Runnable demos for the [sphereQL](https://github.com/bkahan/sphereQL)
Python bindings (`pip install sphereql`).

Most examples use the 100-sentence [`dataset.py`](./dataset.py) — 10
topics, deterministic 64-d FNV-1a hash embeddings — and need no API
keys or external services. The two exceptions are called out below:
`lingua_e2e.py` needs a feature-gated build, and `qdrant_e2e.py` needs
a live Qdrant cluster.

## [`quickstart.py`](./quickstart.py)

5-minute walkthrough of sphereQL's core surface: 3D visualization,
nearest-neighbor search, glob detection, concept paths, local
manifolds, data export, and a vector-DB bridge demo.

```bash
python quickstart.py
```

## [`kernel_pca.py`](./kernel_pca.py)

Compares linear PCA against Gaussian kernel PCA as projection
families: fit, `project`, `project_rich`, batch projection, per-category
coherence, coordinate conversions, sigma tuning, volumetric mode,
out-of-sample projection.

```bash
python kernel_pca.py
```

## [`category_enrichment_basic.py`](./category_enrichment_basic.py)

Gentler introduction to the category layer on a small hand-built
corpus: categorized pipeline construction, category concept paths,
drill-down within one category, category stats, and the inner-sphere
report.

```bash
python category_enrichment_basic.py
```

## [`category_enrichment.py`](./category_enrichment.py)

The category layer — now with classification, confidence, and
hierarchical routing. Covers:

- `PipelineConfig` as a dict (projection kind, routing thresholds,
  inner-sphere gates).
- `projection_warnings()` — structured health signals when EVR is low.
- `category_stats()` with the `bridge_quality` field.
- `category_concept_path` showing `Genuine` / `OverlapArtifact` / `Weak`
  bridge classification and per-hop + end-to-end confidence.
- `drill_down` with inner-sphere fallback.
- `domain_groups()` + `hierarchical_nearest()` — coarse routing for
  low-EVR corpora.

```bash
python category_enrichment.py
```

## [`metalearning.py`](./metalearning.py)

End-to-end walkthrough of sphereQL's three-level self-optimization
hierarchy:

- **L1** `corpus_features` + `auto_tune` over a custom `search_space`.
- **L2** `NearestNeighborMetaModel` + `DistanceWeightedMetaModel` fit on
  past tuner runs, predicting a config for a new corpus (no tuner
  re-run required).
- **L3** `FeedbackEvent` + `FeedbackAggregator` to blend observed user
  satisfaction into scored records.

```bash
python metalearning.py
```

## [`vectordb_advanced.py`](./vectordb_advanced.py)

`VectorStoreBridge` with the full category layer — the same query
surface that `Pipeline` exposes, now on the bridge:

- `build_pipeline_with_config(config, ...)` for non-PCA projections and
  custom thresholds.
- Category enrichment: `category_stats`, `category_neighbors`,
  `category_concept_path`, `drill_down`.
- Hierarchical routing: `domain_groups`, `hierarchical_nearest`,
  `projection_warnings`.
- `hybrid_search` + `sync_projections` for production flows.

Uses `InMemoryStore` for reproducibility. `QdrantBridge` and
`PineconeBridge` accept identical method signatures.

```bash
python vectordb_advanced.py
```

## [`lingua_e2e.py`](./lingua_e2e.py)

The Rust lingua pipeline from Python: free-form text →
`LinguaPipeline` → `ConceptGraph` with native sphereQL coordinates.
Requires the `lingua` feature, which the default wheel doesn't
include:

```bash
cd sphereql-python && maturin develop --features lingua
python ../sphereql-python-examples/lingua_e2e.py
```

## [`qdrant_e2e.py`](./qdrant_e2e.py)

End-to-end run against a live Qdrant cluster: embed → upsert →
`QdrantBridge` → hybrid search. Needs a build with the `qdrant`
feature plus `QDRANT_API_KEY` / `QDRANT_CLUSTER_ENDPOINT` in
`.env.local` — see the docstring at the top of the file for full
setup.

## Dataset

[`dataset.py`](./dataset.py) contains 100 factual sentences across 10
categories (science, technology, cooking, sports, music, history,
nature, health, philosophy, business) with deterministic 64-d
embeddings generated via FNV-1a hashing. Import it for your own
experiments:

```python
from dataset import SENTENCES, encode
```
