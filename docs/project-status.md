# Project status

sphereQL is at **v0.1.0-alpha**. The core API is functional and covered
by 400+ tests, but may change before 1.0.

## Known limitations

- **Search precision degrades at higher k.** The 384-d to 3-d projection
  is inherently lossy (~2–5% explained variance). Precision at k=1 is
  perfect; at k=5 it drops to ~20%. Mitigations available today:
    - Hybrid search (angular recall + cosine re-ranking) for production
      precision.
    - `pipeline.hierarchical_nearest(embedding, k)` falls back to
      domain-group routing when the fitted EVR is below
      `RoutingConfig::low_evr_threshold`.
    - `ProjectionKind::LaplacianEigenmap` via `auto_tune` often
      preserves more neighbor structure than PCA on sparse/noisy
      corpora (see [empirical findings](empirical-findings.md)).

  See [search-precision-roadmap.md](search-precision-roadmap.md) for
  tracked improvements.

- **GraphQL does not expose category enrichment.** The GraphQL crate
  operates on the raw spatial index, not the embedding pipeline.
  Category queries (concept paths, neighbors, drill-down, stats) are
  available in Rust, Python, and WASM but not yet through GraphQL.

- **Inner spheres require 20+ items per category.** Categories below
  this threshold fall back to the outer sphere for drill-down queries.
  This is by design — small categories don't benefit from a separate
  projection, and the threshold is configurable via
  `InnerSphereConfig::min_size`.

- **Python and WASM bindings lag behind Rust.** The Laplacian projection,
  `auto_tune`, `MetaModel`, and `FeedbackAggregator` layers are Rust-only
  in 0.1.x.

## Roadmap

- Improve search precision at higher k values.
- HNSW or VP-tree indexing for better recall without brute-force
  fallback.
- Streaming/incremental PCA for large-scale datasets.
- GraphQL integration for category enrichment queries.
- Python/WASM bindings for Laplacian projection, auto-tuner, and
  meta-model.
