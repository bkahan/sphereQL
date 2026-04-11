# Benchmark Analysis

## Setup

- **Dataset:** 10,000 points, 384 dimensions, 20 synthetic clusters
- **Queries:** 200 random queries with pre-computed ground truth
- **Metrics:** Precision@k, Recall@k, nDCG@k, latency (mean and p99)
- **Methods tested:**
  - Vanilla ANN (brute-force cosine similarity)
  - SphereQL-only (search in 3D projected space)
  - Hybrid (ANN recall + spherical re-ranking, at recall multipliers 2x/4x/8x)

## Results

```
PCA explained variance ratio: 2.8%
Index build time: 77,910 ms

| Method               |  k | Precision@k | Recall@k | nDCG@k |   Mean us |   p99 us |
|----------------------|----|-------------|----------|--------|-----------|----------|
| Vanilla ANN          |  1 |      1.0000 |   1.0000 | 1.0000 |  177550.4 | 409086.0 |
| Vanilla ANN          |  5 |      1.0000 |   1.0000 | 1.0000 |  172679.5 | 256517.0 |
| Vanilla ANN          | 10 |      1.0000 |   1.0000 | 1.0000 |  163682.0 | 305402.0 |
| Vanilla ANN          | 20 |      1.0000 |   1.0000 | 1.0000 |  175358.2 | 343595.0 |
| SphereQL-only        |  1 |      1.0000 |   1.0000 | 1.0000 |    1903.5 |   2381.0 |
| SphereQL-only        |  5 |      0.2050 |   0.2050 | 0.7453 |    2111.0 |   4439.0 |
| SphereQL-only        | 10 |      0.1090 |   0.1090 | 0.6495 |    2045.8 |   3059.0 |
| SphereQL-only        | 20 |      0.0660 |   0.0660 | 0.5629 |    2060.9 |   2659.0 |
| Hybrid (r=k*2)       |  1 |      1.0000 |   1.0000 | 1.0000 |  168757.8 | 280784.0 |
| Hybrid (r=k*2)       |  5 |      0.5740 |   0.5740 | 0.9824 |  158985.3 | 236723.0 |
| Hybrid (r=k*2)       | 10 |      0.5385 |   0.5385 | 0.9699 |  169847.1 | 299427.0 |
| Hybrid (r=k*2)       | 20 |      0.5397 |   0.5397 | 0.9547 |  167400.4 | 252811.0 |
| Hybrid (r=k*4)       |  1 |      1.0000 |   1.0000 | 1.0000 |  168828.5 | 273534.0 |
| Hybrid (r=k*4)       |  5 |      0.3780 |   0.3780 | 0.9653 |  174477.5 | 283212.0 |
| Hybrid (r=k*4)       | 10 |      0.3280 |   0.3280 | 0.9435 |  160207.4 | 232902.0 |
| Hybrid (r=k*4)       | 20 |      0.3045 |   0.3045 | 0.9168 |  162350.8 | 269068.0 |
| Hybrid (r=k*8)       |  1 |      1.0000 |   1.0000 | 1.0000 |  170615.3 | 305655.0 |
| Hybrid (r=k*8)       |  5 |      0.2940 |   0.2940 | 0.9474 |  160601.3 | 223958.0 |
| Hybrid (r=k*8)       | 10 |      0.2205 |   0.2205 | 0.9161 |  161750.5 | 279010.0 |
| Hybrid (r=k*8)       | 20 |      0.1892 |   0.1892 | 0.8770 |  167206.6 | 306234.0 |
```

## Key Findings

### The EVR problem

PCA projects 384 dimensions down to 3, retaining only 2.8% of the
original variance. This is the dominant factor limiting search precision.
The 3D sphere simply cannot represent enough of the embedding structure
to distinguish neighbors beyond the single closest point.

- **k=1 works** because the absolute nearest neighbor tends to survive
  even a very lossy projection.
- **k=5+ collapses** (20% precision at k=5, 6.6% at k=20) because the
  projection scrambles the ordering of points that are close-but-not-closest
  in the original space.

### Speed vs. accuracy tradeoff

SphereQL-only search is **80-90x faster** than vanilla ANN (~2ms vs
~170ms). The speed advantage is real and comes from operating in 3
dimensions instead of 384. But at current precision levels, this speed
is only useful for approximate/exploratory queries, not as an ANN
replacement.

### Hybrid re-ranking is counterproductive

The hybrid approach pays the full ANN cost (~170ms) and then re-ranks
by spherical distance, which actively *demotes* correct results. Higher
recall multipliers make it worse, not better, because more candidates
means more opportunities for the lossy re-ranking to reorder incorrectly.

The core issue: re-ranking by 3D spherical distance when the 3D
projection only captures 2.8% of variance is equivalent to adding noise
to a correct ranking.

### Build time

77 seconds for 10k points is dominated by the spatial index construction
(shell partitioning, theta/phi divisions). This needs profiling to
identify whether PCA fitting, projection, or index insertion is the
bottleneck.

## What SphereQL is good at

These results confirm that SphereQL's value is in **spatial analysis**,
not search replacement:

- **Visualization** -- 3D projection is purpose-built for this
- **Glob detection** -- coarse cluster structure survives low EVR
- **Concept paths** -- topological connectivity doesn't need metric precision
- **Local manifold analysis** -- directional variance is meaningful even
  in a lossy projection

Search precision improvements are tracked in
[search-precision-roadmap.md](search-precision-roadmap.md).
