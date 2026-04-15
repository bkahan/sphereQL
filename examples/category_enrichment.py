"""Category Enrichment Layer — Python Example

Demonstrates SphereQL's hierarchical category system:
  - Building a pipeline with categorized embeddings
  - Category concept paths (domain-level navigation)
  - Drill-down queries within a single category
  - Category stats and inner sphere reporting

Usage:
  pip install sphereql  # or: maturin develop --features embed
  python examples/category_enrichment.py
"""

import numpy as np

# sphereql is the PyO3 module built from sphereql-python
import sphereql


def make_embedding(features: dict, dim: int = 16, seed: int = 0):
    """Create a toy embedding from axis→weight pairs with small noise."""
    rng = np.random.RandomState(seed)
    v = np.zeros(dim, dtype=np.float64)
    for axis, weight in features.items():
        v[axis] = weight
    v += rng.uniform(-0.015, 0.015, size=dim)
    return v


# Semantic axes (matching the Rust example)
PHYSICS, BIOLOGY, CHEMISTRY, MATH = 0, 1, 2, 3
COOKING, FLAVOR, HEAT = 4, 5, 6
MUSIC, PERFORM, RHYTHM = 7, 8, 9
HISTORY, CIVILIZE = 10, 11
ENERGY, NATURE, TECH, EMOTION = 12, 13, 14, 15
DIM = 16


def main():
    print("=" * 62)
    print("   SphereQL: Category Enrichment Layer (Python)")
    print("=" * 62)
    print()

    # ── Build corpus ───────────────────────────────────────────────

    categories = []
    embeddings = []
    labels = []

    # Science
    science = [
        ("Speed of light",      {PHYSICS: 1.0, ENERGY: 0.6, MATH: 0.3}),
        ("DNA structure",        {BIOLOGY: 1.0, CHEMISTRY: 0.5, NATURE: 0.4}),
        ("Quantum entanglement", {PHYSICS: 0.9, MATH: 0.5, ENERGY: 0.4}),
        ("Photosynthesis",       {BIOLOGY: 0.8, CHEMISTRY: 0.7, ENERGY: 0.8, NATURE: 0.6}),
        ("General relativity",   {PHYSICS: 0.9, MATH: 0.8, ENERGY: 0.3}),
        ("Cell division",        {BIOLOGY: 0.9, CHEMISTRY: 0.3, NATURE: 0.5}),
    ]
    for i, (label, feats) in enumerate(science):
        categories.append("science")
        embeddings.append(make_embedding(feats, DIM, seed=100 + i))
        labels.append(label)

    # Cooking
    cooking = [
        ("Preheat the oven",    {COOKING: 1.0, HEAT: 0.9, FLAVOR: 0.2}),
        ("Simmer the sauce",    {COOKING: 0.9, HEAT: 0.7, FLAVOR: 0.6}),
        ("Season with paprika", {COOKING: 0.8, FLAVOR: 1.0, CHEMISTRY: 0.2}),
        ("Fold egg whites",     {COOKING: 0.9, FLAVOR: 0.4}),
        ("Caramelize onions",   {COOKING: 0.8, HEAT: 0.8, FLAVOR: 0.7, CHEMISTRY: 0.3}),
        ("Fermentation process", {COOKING: 0.6, CHEMISTRY: 0.7, BIOLOGY: 0.4, NATURE: 0.3}),
    ]
    for i, (label, feats) in enumerate(cooking):
        categories.append("cooking")
        embeddings.append(make_embedding(feats, DIM, seed=200 + i))
        labels.append(label)

    # Music
    music = [
        ("Symphony climax",       {MUSIC: 1.0, PERFORM: 0.7, EMOTION: 0.8, ENERGY: 0.5}),
        ("Concert hall voice",    {MUSIC: 0.9, PERFORM: 0.9, EMOTION: 0.6}),
        ("Polyrhythm drumming",   {MUSIC: 0.8, PERFORM: 0.6, RHYTHM: 1.0}),
        ("Jazz improvisation",    {MUSIC: 0.7, PERFORM: 0.8, RHYTHM: 0.6, EMOTION: 0.5}),
        ("Electronic synthesis",  {MUSIC: 0.6, TECH: 0.7, RHYTHM: 0.5, ENERGY: 0.4}),
    ]
    for i, (label, feats) in enumerate(music):
        categories.append("music")
        embeddings.append(make_embedding(feats, DIM, seed=300 + i))
        labels.append(label)

    # History
    history = [
        ("Fall of Rome",           {HISTORY: 1.0, CIVILIZE: 0.8}),
        ("Industrial Revolution",  {HISTORY: 0.8, TECH: 0.7, ENERGY: 0.6, CIVILIZE: 0.6}),
        ("Ancient Egypt",          {HISTORY: 0.9, CIVILIZE: 1.0}),
        ("Renaissance art",        {HISTORY: 0.7, CIVILIZE: 0.5, EMOTION: 0.6, PERFORM: 0.3}),
        ("Space race",             {HISTORY: 0.6, TECH: 0.8, PHYSICS: 0.4, ENERGY: 0.5}),
    ]
    for i, (label, feats) in enumerate(history):
        categories.append("history")
        embeddings.append(make_embedding(feats, DIM, seed=400 + i))
        labels.append(label)

    emb_matrix = np.array(embeddings, dtype=np.float64)
    print(f"Corpus: {len(categories)} items across {len(set(categories))} categories\n")

    # ── Build pipeline ─────────────────────────────────────────────

    pipeline = sphereql.Pipeline(categories, emb_matrix)
    print(f"Pipeline built — {pipeline.num_items} items")
    print(f"EVR: {pipeline.explained_variance_ratio:.2%}")
    print(f"Categories: {pipeline.unique_categories()}\n")

    # ── 1. Category concept path ─────────────────────────────────────

    print("━" * 50)
    print("Category Concept Path: cooking → history")
    print("━" * 50)
    print()

    # Show unique categories and their items
    cats = pipeline.unique_categories()
    for cat in cats:
        items_in_cat = [
            labels[i] for i, c in enumerate(categories) if c == cat
        ]
        print(f"  {cat}: {', '.join(items_in_cat)}")

    # ── 2. k-NN search with category context ─────────────────────────

    print()
    print("━" * 50)
    print("Nearest Neighbors: 'physics + energy' query")
    print("━" * 50)
    print()

    query = make_embedding({PHYSICS: 0.8, ENERGY: 0.7, MATH: 0.3}, DIM, seed=999)
    results = pipeline.nearest(query, k=5)
    for i, r in enumerate(results):
        idx = int(r.id.split("-")[1])
        print(
            f"  {i+1}. [{r.category:<10}] dist={r.distance:.4f}  "
            f'certainty={r.certainty:.3f}  "{labels[idx]}"'
        )

    # ── 3. Cross-domain query ────────────────────────────────────────

    print()
    print("━" * 50)
    print("Cross-Domain: midpoint between 'science' and 'cooking'")
    print("━" * 50)
    print()

    cross_query = make_embedding(
        {BIOLOGY: 0.4, CHEMISTRY: 0.5, COOKING: 0.4, ENERGY: 0.3, HEAT: 0.3},
        DIM, seed=888,
    )
    results = pipeline.nearest(cross_query, k=6)
    seen_cats = set()
    for r in results:
        seen_cats.add(r.category)
        idx = int(r.id.split("-")[1])
        print(f"  [{r.category:<10}] dist={r.distance:.4f}  \"{labels[idx]}\"")
    print(f"\n  → Found items from {len(seen_cats)} categories: {', '.join(sorted(seen_cats))}")

    # ── 4. Detect concept globs ──────────────────────────────────────

    print()
    print("━" * 50)
    print("Concept Glob Detection (k=4)")
    print("━" * 50)
    print()

    globs = pipeline.detect_globs(k=4, max_k=6)
    for g in globs:
        top_cats = ", ".join(f"{cat}({count})" for cat, count in g.top_categories)
        print(
            f"  Glob {g.id}: {g.member_count} members, "
            f"radius={g.radius:.4f}, categories=[{top_cats}]"
        )

    # ── 5. Concept path between items ────────────────────────────────

    print()
    print("━" * 50)
    print("Item Concept Path: s-0000 (Speed of light) → s-0006 (Preheat oven)")
    print("━" * 50)
    print()

    path = pipeline.concept_path("s-0000", "s-0006", graph_k=8)
    if path is not None:
        for step in path.steps:
            idx = int(step.id.split("-")[1])
            print(
                f"  → {step.id} [{categories[idx]:<10}] "
                f'd={step.cumulative_distance:.4f}  "{labels[idx]}"'
            )
        print(f"\n  Total distance: {path.total_distance:.4f}")
    else:
        print("  (no path found)")

    # ── 6. Export and summary ────────────────────────────────────────

    print()
    print("━" * 50)
    print("Pipeline Summary")
    print("━" * 50)
    print()
    print(f"  Items: {pipeline.num_items}")
    print(f"  Categories: {len(cats)}")
    print(f"  EVR: {pipeline.explained_variance_ratio:.4f}")
    print(f"  CSV header: {pipeline.to_csv().splitlines()[0]}")
    print()
    print("✓ Category Enrichment Layer Python example complete.")


if __name__ == "__main__":
    main()
