# Search Precision Roadmap

Algorithmic improvements to SphereQL's nearest-neighbor search precision,
ordered by expected impact and implementation effort.

See [benchmark-analysis.md](benchmark-analysis.md) for the current
baseline numbers and root cause analysis.

---

## 1. Fix hybrid re-ranking (high impact, low effort)

**Problem:** Hybrid mode retrieves candidates via ANN, then re-ranks by
spherical distance. With 2.8% EVR, this re-ranking adds noise and
demotes correct results.

**Fix:** Reverse the flow. Use spherical projection as a cheap pre-filter
to eliminate obvious non-candidates, then score survivors with original
cosine similarity. Alternatively, use a weighted combination:

```
score = alpha * cosine_sim(original) + (1 - alpha) * angular_proximity(sphere)
```

where `alpha` is tuned based on EVR (high EVR = trust the sphere more).

**Expected outcome:** Hybrid precision should match or exceed vanilla ANN
at lower latency, since the pre-filter reduces the candidate set before
expensive high-dimensional distance computation.

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

## 5. Product quantization (medium impact, medium effort)

**Problem:** Compressing 384D to 3D is too aggressive.

**Fix:** Split the 384D space into M subspaces (e.g. 8 groups of 48D).
Quantize each subspace independently (k-means codebook per subspace).
Approximate distances via lookup table over codebook indices.

This is the approach used by FAISS IVF-PQ and ScaNN. SphereQL could
either implement PQ natively or integrate as a backend.

**Expected outcome:** Asymmetric distance computation (exact query vs
quantized database) gives high recall at very low memory and compute
cost.

**Trade-off:** Significant implementation effort. Codebook training adds
to build time. May overlap with existing vector DB backends (Qdrant,
Pinecone) that already do PQ internally.

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

## 7. Learned projection (high impact, high effort)

**Problem:** PCA is an unsupervised linear projection that maximizes
variance. It doesn't optimize for neighbor preservation.

**Fix:** Train a small model (linear map or shallow MLP) with a triplet
or contrastive loss that directly optimizes k-NN preservation in the
projected space:

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
`ConfiguredProjection` enum (PCA / Kernel PCA / Laplacian eigenmap), a
`PipelineConfig` hierarchy for every tunable constant, a `QualityMetric`
trait with four concrete metrics + composite presets, a discrete
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

**Short term (surgical fixes):**
1. Fix hybrid scoring (#1) -- biggest precision gain for least code change
2. Add higher-dimensional search projection (#2) -- keep 3D for viz, use 8-16D for search

**Medium term (architectural):**
3. Graph-based index (#3) -- the proven production approach for ANN

**Exploratory:**
4. Learned projection (#7) -- highest theoretical ceiling, but most R&D risk
