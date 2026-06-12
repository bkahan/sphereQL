# Search Precision Roadmap

Algorithmic improvements to SphereQL's nearest-neighbor search precision,
ordered by expected impact and implementation effort. Items are marked
**(shipped)** / **(partially shipped)** where the code has since landed.

See [benchmark-analysis.md](benchmark-analysis.md) for the baseline
numbers (2026-04-10 run, pre-fixes) and root cause analysis.

---

## 1. Fix hybrid re-ranking (shipped)

**Problem:** Hybrid mode retrieved candidates via ANN, then re-ranked by
spherical distance. With 2.8% EVR, that re-ranking added noise and
demoted correct results.

**What shipped** (commit `e169f59`):
`VectorStoreBridge::hybrid_search` now re-ranks ANN candidates by
**original cosine similarity** in the full embedding space
(`sphereql-vectordb/src/bridge.rs`), so the lossy projection is never
responsible for final ordering. The same commit accelerated the
spherical query path itself (unit-Cartesian cosine proxy + sector-pruned
search in `SpatialIndex::nearest`).

**Not implemented:** the spherical-pre-filter inversion (sphere prunes
candidates, cosine scores survivors) and the EVR-tuned
`alpha * cosine + (1 - alpha) * angular` blend remain open ideas. The
pipeline's `default_nearest` does use an EVR gate
(`HIGH_EVR_ROUTING_BYPASS = 0.90`) for routing decisions, which is the
same spirit applied to group routing rather than scoring.

**Still pending:** re-running the retrieval benchmark — the published
hybrid numbers predate this fix.

## 2. Higher-dimensional search projection (high impact, low effort)

**Problem:** The 3D constraint exists for visualization. Search doesn't
need to be 3D.

**Fix:** Maintain two projections:
- 3D PCA for visualization, glob detection, concept paths
- 8-16D PCA for search queries

An 8D projection on 384D embeddings would capture significantly more
variance (likely 20-40% EVR vs 2.8%), dramatically improving neighbor
preservation.

**Implementation:** Add a `SearchProjection` that projects to a
configurable number of dimensions. The spatial index already supports
arbitrary `SphericalPoint`-like coordinates; it would need generalization
to N-dimensional points, or a separate simpler index for the search path.

## 3. Graph-based index with spherical entry points (high impact, medium effort)

**Problem:** Brute-force ANN is O(n) per query. The spherical index is
fast but imprecise.

**Groundwork shipped:** an RP-forest ANN index now exists
(`sphereql-embed/src/ann.rs`). It currently serves UMAP's kNN-graph
build / out-of-sample transform and `GraphModularity`'s edge
construction at ≥ 2000 items — it is *not* yet wired into the
user-facing query path.

**Fix:** Build an HNSW-style navigable small world graph over the
original 384D vectors. Use the spherical projection to select entry
points for graph traversal -- the sphere tells you *where to start
looking*, the graph does precise navigation in the original space.

**Expected outcome:** Sub-linear query time with high precision. The
sphere contributes fast approximate locality without being responsible
for final distance computation.

**Trade-off:** HNSW adds memory overhead (graph edges per point) and
complicates the index build. But this is the standard approach in
production vector search for a reason.

## 4. Residual re-projection (medium impact, low effort)

**Problem:** A single PCA-3 projection discards 97.2% of variance
irrecoverably.

**Fix:** After the initial 3D projection, compute the residual
(original embedding minus its reconstruction from 3 components). Project
the residual onto another 3 components. Chain 2-3 stages.

```
stage_1 = PCA_3(embedding)           // captures top 3 components
residual_1 = embedding - reconstruct(stage_1)
stage_2 = PCA_3(residual_1)          // captures next 3 components
residual_2 = residual_1 - reconstruct(stage_2)
stage_3 = PCA_3(residual_2)          // captures next 3 components
```

Approximate original distance from the chain of projections. This is
conceptually similar to approach #2 but preserves the 3D-per-stage
structure, which may compose better with the existing spatial index.

**Trade-off:** Each stage requires its own index or lookup structure.
Distance approximation across chained projections adds complexity.

## 5. Product quantization (shipped)

**Problem:** Compressing 384D to 3D is too aggressive.

**What shipped:** native PQ as a sidecar for the full embedding —
`sphereql-vectordb/src/pq.rs` (`PqConfig`, `PqCodebook`, `PqIndex`,
`PqStore`, with an LMDB-backed store behind the `pq-lmdb` feature in
`pq_lmdb.rs`). Standard layout: M subspaces, k-means codebook per
subspace (default 8 bits → 256 centroids, one byte per subspace per
item), asymmetric distance via an M × K lookup table. Usable as a
re-ranker over spherical candidates or as a standalone search path when
EVR is poor.

**Still pending:** no precision/latency numbers for the PQ path have
been recorded in the retrieval benchmark yet.

**Trade-off:** Codebook training adds to build time. Overlaps with
vector DB backends (Qdrant, Pinecone) that do PQ internally — the
native path matters mostly for the in-memory store.

## 6. Locality-sensitive hashing (medium impact, medium effort)

**Problem:** PCA optimizes for global variance, not local neighbor
preservation.

**Fix:** Use random hyperplane LSH or cross-polytope LSH with multiple
hash tables. Each table independently hashes points; candidates are
the union of bucket collisions across tables. Nearby points in the
original space collide with high probability.

**Expected outcome:** Probabilistic guarantees on recall (tunable via
number of tables and hash functions). Sub-linear query time.

**Trade-off:** Memory scales linearly with number of tables. No single
"projected position" to visualize -- this is a search-only structure.
Doesn't compose with the spatial analysis features (globs, paths,
manifolds).

## 7. Learned projection (partially shipped)

**Problem:** PCA is an unsupervised linear projection that maximizes
variance. It doesn't optimize for neighbor preservation.

**What shipped since:**

- **UMAP-on-sphere** (`ProjectionKind::UmapSphere`,
  `sphereql-embed/src/umap.rs`) — a projection whose objective *is*
  neighborhood preservation, optimized with Adam directly on S², with an
  optional supervised category term. This delivers most of what this
  item wanted, in-pipeline and tuner-sweepable.
- **Offline contrastive fine-tuning**
  (`scripts/contrastive_finetune.py`) — supervised NT-Xent fine-tuning
  of the upstream sentence-transformer embeddings so same-category
  concepts cluster before projection.

**Still open:** a dedicated learned 3D map (linear or shallow MLP)
trained with a triplet/contrastive loss on k-NN preservation:

```
loss = sum over triplets (anchor, positive, negative):
    max(0, d_proj(anchor, positive) - d_proj(anchor, negative) + margin)
```

A learned 3D projection optimized for neighbor preservation could
dramatically outperform PCA even at the same dimensionality.

**Trade-off:** Requires a training step (offline, but adds complexity).
The projection is dataset-specific -- a model trained on one embedding
distribution may not transfer. Adds a dependency on a training framework
(though a linear map can be trained with just gradient descent on
matrices, no deep learning framework needed).

---

## 8. Meta-learned projection selection (shipped)

**Problem:** PCA is one of several projection families, and no single choice
wins on every corpus regime — variance-based projections degrade on sparse,
noise-heavy embeddings where connectivity-based ones preserve the signal.

**Fix (shipped):** The `sphereql-embed` crate now carries a
`ProjectionKind` enum with four families (PCA / Kernel PCA / Laplacian
eigenmap / UMAP-on-sphere; `SearchSpace::default()` sweeps PCA + UMAP), a
`PipelineConfig` hierarchy for every tunable constant, a `QualityMetric`
trait with concrete metrics + composite presets, a discrete
`SearchSpace` sweep via `auto_tune` (Grid / Random / Bayesian TPE-lite),
a 10-feature `CorpusFeatures` profile, and a `MetaModel` layer
(`NearestNeighborMetaModel`, `DistanceWeightedMetaModel`) with an on-disk
store at `~/.sphereql/meta_records.json`. Workflow:

1. `auto_tune` on a new corpus, emit a `MetaTrainingRecord`.
2. Store accumulates across sessions.
3. On the next new corpus, `SphereQLPipeline::new_from_metamodel` predicts
   the winning config without rerunning the tuner, or
   `new_from_metamodel_tuned` does a short warm-started tuner pass.
4. Per-query `FeedbackEvent`s blend user satisfaction back into the
   stored records for L3 online refinement.

This is adjacent to #1–#3 rather than a substitute. It doesn't improve
search precision at the ANN level; it addresses the "the sphere is too
lossy for _this_ corpus" failure mode that motivates #2 and #7.

See [`benchmark-analysis.md`](benchmark-analysis.md) for the empirical
finding that motivated the framework (PCA wins the built-in corpus,
Laplacian wins the stress corpus — same pipeline, same tuner).

---

## Priority recommendation

**Done:** hybrid scoring fix (#1), product quantization (#5),
meta-learned selection (#8), and the UMAP half of #7.

**Next (measurement before more code):** re-run the retrieval benchmark
on current code — every published number predates the #1 fix, the query
acceleration, and PQ. Add UMAP and the PQ re-rank path to the harness
while at it.

**Short term (surgical):**
1. Higher-dimensional search projection (#2) -- keep 3D for viz, use 8-16D for search

**Medium term (architectural):**
2. Graph-based index (#3) -- the RP-forest in `sphereql-embed/src/ann.rs`
   is partial groundwork; the query path still needs it

**Exploratory:**
3. Learned 3D map (#7, remaining half) -- highest theoretical ceiling, most R&D risk
