# sphereQL

Project high-dimensional embeddings onto a 3D sphere for fast semantic search,
interactive visualization, and knowledge structure analysis. Built in Rust,
exposed to Python via PyO3. These are the official Python bindings for the
[sphereQL](https://github.com/bkahan/sphereQL) workspace.

## Install

```bash
pip install sphereql
```

The default wheel ships the core pipeline, projections, metalearning,
visualization, and the in-memory vector-store bridge. The Qdrant and
Pinecone bridges are compile-time features — build from source to get
them:

```bash
cd sphereql-python
python -m maturin develop --release --features qdrant   # adds QdrantBridge
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

sphereql.visualize(categories, embeddings, title="My Embeddings", open_browser=True)
# Opens an interactive 3D sphere in your browser; set open_browser=False to only write the file
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

sphereQL fits a projection to reduce embeddings to 3 dimensions, then maps
them onto spherical coordinates (r, theta, phi). The radial component encodes
magnitude/confidence, while angular position preserves semantic similarity.
This enables angular-distance queries, cluster detection, concept paths, and
interactive 3D visualization — all in projected space.

The pipeline supports four projection families, selected via the config
dict: `"Pca"` (default), `"KernelPca"`, `"LaplacianEigenmap"`
(connectivity-preserving spectral projection over a k-NN similarity
graph), and `"UmapSphere"` (UMAP optimized directly on S², new in this
release — no standalone class; configure it through
`config={"projection_kind": "UmapSphere", "umap": {...}}`). Standalone
projection classes are also exposed for direct use: `PcaProjection`,
`KernelPcaProjection`, `LaplacianEigenmap`, and `RandomProjection`.

The auto-tuning and metalearning surface is bound too — `corpus_features`,
`auto_tune`, `NearestNeighborMetaModel`, `DistanceWeightedMetaModel`,
`FeedbackEvent`, and `FeedbackAggregator`.

```python
# Non-default projection via config dict
pipeline = sphereql.Pipeline(
    categories, embeddings,
    config={"projection_kind": "LaplacianEigenmap"},
)

# Auto-tune over the search space
tuned, report = sphereql.auto_tune(categories, embeddings, budget=16)
```

Most of the Rust query surface has a 1:1 Python binding; the gaps are
tracked explicitly in the workspace's
[`.bindings-ignore.toml`](https://github.com/bkahan/sphereQL/blob/main/.bindings-ignore.toml)
allowlist and enforced by a drift checker in CI. Notably *not* bound
yet: the corpus self-tune controller (`run_self_tune`) and bridge
relation-type annotations.

## API Reference

Type stubs (`python/sphereql/__init__.pyi`) are auto-generated via
`pyo3-stub-gen` and ship with the wheel — IDEs, `mypy`, and `pyright`
pick them up automatically. They are generated at the `vectordb` feature
level, so the in-memory vector-store classes (`InMemoryStore`,
`VectorStoreBridge`) are covered; `QdrantBridge` / `PineconeBridge`
(behind their own features, not in the default wheel) are not.
Regenerate after binding changes with:

```bash
cd sphereql-python && cargo run --bin gen-stubs --features vectordb
```

## Status

Pre-1.0 (`0.3.0`). Expect
breaking changes between minor versions. Source, issues, and full
documentation live in the
[sphereQL repository](https://github.com/bkahan/sphereQL).

## License

MIT
