# SphereQL Python Examples

All examples use the 100-sentence [`dataset.py`](./dataset.py) — 10
topics, deterministic 64-d FNV-1a hash embeddings. No API keys, no
external services. Run any of them after `pip install sphereql`.

## [`quickstart.py`](./quickstart.py)

5-minute walkthrough of SphereQL's core surface: 3D visualization,
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

End-to-end walkthrough of SphereQL's three-level self-optimization
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

## Dataset

[`dataset.py`](./dataset.py) contains 100 factual sentences across 10
categories (science, technology, cooking, sports, music, history,
nature, health, philosophy, business) with deterministic 64-d
embeddings generated via FNV-1a hashing. Import it for your own
experiments:

```python
from dataset import SENTENCES, encode
```
