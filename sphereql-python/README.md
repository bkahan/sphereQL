# SphereQL

Project high-dimensional embeddings onto a 3D sphere for fast semantic search,
interactive visualization, and knowledge structure analysis. Built in Rust,
exposed to Python via PyO3.

## Install

```bash
pip install sphereql
```

For Qdrant vector database support:

```bash
pip install sphereql[qdrant]
```

## Quick Start: Semantic Search

```python
import sphereql

categories = ["science", "science", "cooking", "cooking"]
embeddings = [
    [0.1, 0.9, 0.3, 0.0],
    [0.2, 0.8, 0.4, 0.1],
    [0.9, 0.1, 0.0, 0.5],
    [0.8, 0.2, 0.1, 0.4],
]

pipeline = sphereql.Pipeline(categories, embeddings)

query = [0.15, 0.85, 0.35, 0.05]
results = pipeline.nearest(query, k=3)

for hit in results:
    print(f"{hit.id}  {hit.category}  distance={hit.distance:.4f}")
```

## Quick Start: Visualization

```python
import sphereql

sphereql.visualize(categories, embeddings, title="My Embeddings")
# Opens an interactive 3D sphere in your browser
```

## Quick Start: Vector DB Bridge

```python
import sphereql

store = sphereql.InMemoryStore("my-collection", dimension=384)
store.upsert([
    {"id": "doc-1", "vector": embedding_1, "metadata": {"category": "science"}},
    {"id": "doc-2", "vector": embedding_2, "metadata": {"category": "cooking"}},
    # ...
])

bridge = sphereql.VectorStoreBridge(store)
bridge.build_pipeline(category_key="category")

results = bridge.hybrid_search(query_vec, final_k=5, recall_k=20)
```

## How It Works

SphereQL fits a PCA projection to reduce embeddings to 3 dimensions, then maps
them onto spherical coordinates (r, theta, phi). The radial component encodes
magnitude/confidence, while angular position preserves semantic similarity.
This enables angular-distance queries, cluster detection, concept paths, and
interactive 3D visualization — all in projected space.

## API Reference

See the [type stubs](https://github.com/bkahan/sphereQL/blob/main/sphereql-python/python/sphereql/sphereql.pyi)
for complete API signatures and docstrings.

## License

MIT
