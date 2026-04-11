# SphereQL Examples

## Quickstart

A 5-minute walkthrough of every major SphereQL feature using a built-in
dataset of 100 sentences across 10 topics. No external services or API
keys required.

```bash
pip install sphereql
cd examples/
python quickstart.py
```

The script demonstrates:
- 3D sphere visualization (opens in browser)
- Nearest neighbor search in spherical space
- Automatic cluster (glob) detection
- Concept paths between items
- Local manifold analysis
- Vector DB bridge with hybrid search

All output goes to stdout. The only file created is an HTML visualization.

## Dataset

`dataset.py` contains 100 factual sentences across 10 categories
(science, technology, cooking, sports, music, history, nature, health,
philosophy, business) with deterministic 64-d embeddings generated via
FNV-1a hashing. Import it for your own experiments:

```python
from dataset import SENTENCES, encode
```
