# Use cases

What sphereQL is designed for:

- **Semantic search** — project embeddings to S² and query nearest neighbors
  in microseconds, with optional cosine-similarity re-ranking in the original
  space for precision.
- **Knowledge visualization** — render your entire embedding corpus as an
  interactive 3D sphere, colored by category, explorable in the browser.
- **Concept path tracing** — find the shortest semantic path between two
  concepts through projected space.
- **Cluster detection** — automatically discover concept "globs" (dense
  regions) on the sphere surface.
- **Category enrichment** — automatic inter-category graph with bridge
  detection, category-level concept paths, cohesion metrics, and
  hierarchical inner spheres for high-resolution within-category drill-down.
- **Auto-tuned projection selection** — `auto_tune` sweeps projection family
  and pipeline knobs against a `QualityMetric` (Grid / Random / Bayesian
  search); `MetaTrainingRecord` + `NearestNeighborMetaModel` /
  `DistanceWeightedMetaModel` recall the winning config on a new corpus from
  a feature profile.
- **Geospatial indexing** — use the core library for pure spherical
  geometry: coordinate conversions, great-circle distances, region queries
  (cone, cap, shell, band, wedge).
- **Vector database bridge** — connect Qdrant or Pinecone collections to
  sphereQL's pipeline for hybrid search and spherical coordinate enrichment.
