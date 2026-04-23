# Python quickstart

The Python bindings are built via PyO3/maturin.

In 0.1.x the Python bindings expose PCA and Kernel PCA only. The
Laplacian eigenmap projection, `auto_tune`, and the `MetaModel` layer
are Rust-only for now; bindings will follow.

## Install

```bash
pip install sphereql
```

For Qdrant vector database support:

```bash
pip install sphereql[qdrant]
```

## Semantic search

```python
import sphereql

categories = ["science", "science", "cooking", "cooking", "sports"]
embeddings = [
    [0.1, 0.9, 0.3, 0.0],
    [0.2, 0.8, 0.4, 0.1],
    [0.9, 0.1, 0.0, 0.5],
    [0.8, 0.2, 0.1, 0.4],
    [0.4, 0.4, 0.8, 0.2],
]

pipeline = sphereql.Pipeline(categories, embeddings)

# k-nearest neighbors
query = [0.15, 0.85, 0.35, 0.05]
results = pipeline.nearest(query, k=3)
for r in results:
    print(f"{r.id}  {r.category}  distance={r.distance:.4f}")

# Similarity threshold search
similar = pipeline.similar_above(query, min_cosine=0.8)

# Concept path between items
path = pipeline.concept_path("s-0000", "s-0003", graph_k=10)

# Cluster detection
globs = pipeline.detect_globs(max_k=10)

# Local manifold fitting
manifold = pipeline.local_manifold(query, neighborhood_k=10)

# --- Category Enrichment ---

# Category-level concept path
cat_path = pipeline.category_concept_path("science", "cooking")
if cat_path:
    for step in cat_path.steps:
        print(f"  {step.category_name} (d={step.cumulative_distance:.4f})")

# Nearest neighbor categories
neighbors = pipeline.category_neighbors("science", k=3)
for n in neighbors:
    print(f"  {n.name}: cohesion={n.cohesion:.4f}, members={n.member_count}")

# Drill down within a category (uses inner sphere if available)
hits = pipeline.drill_down("science", query, k=5)
for h in hits:
    print(f"  item={h.item_index} distance={h.distance:.4f} inner={h.used_inner_sphere}")

# Category stats (summaries + inner sphere reports)
summaries, inner_reports = pipeline.category_stats()

# Export projected coordinates
points = pipeline.exported_points()
print(f"Explained variance ratio: {pipeline.explained_variance_ratio:.4f}")
```

## Interactive 3D visualization

```python
import sphereql

# Opens an interactive WebGL sphere in your browser
sphereql.visualize(categories, embeddings, title="My Embeddings")

# Or visualize from an existing pipeline
sphereql.visualize_pipeline(pipeline, title="Pipeline View")
```

## Vector database bridge

```python
import sphereql

# In-memory store (for testing and small datasets)
store = sphereql.InMemoryStore("my-collection", dimension=384)
store.upsert([
    {"id": "doc-1", "vector": embedding_1, "metadata": {"category": "science"}},
    {"id": "doc-2", "vector": embedding_2, "metadata": {"category": "cooking"}},
])

bridge = sphereql.VectorStoreBridge(store)
bridge.build_pipeline(category_key="category")

# Hybrid search: angular candidates + cosine re-ranking
results = bridge.hybrid_search(query_vec, final_k=5, recall_k=20)
```

## Core types in Python

```python
import sphereql

# Spherical/Cartesian/Geo point types
p = sphereql.SphericalPoint(1.0, 0.5, 0.8)
c = sphereql.spherical_to_cartesian(p)
g = sphereql.spherical_to_geo(p)

# Distance functions
d = sphereql.angular_distance(p1, p2)
gc = sphereql.great_circle_distance(p1, p2, radius=6371.0)

# Projection classes
pca = sphereql.PcaProjection.fit(embeddings, radial="magnitude")
kpca = sphereql.KernelPcaProjection.fit(embeddings, radial="magnitude")
rp = sphereql.RandomProjection(dimension=384, radial=1.0, seed=42)
```
