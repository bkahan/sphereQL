"""
SphereQL Quickstart — run this script to see SphereQL in action.

No external services or API keys required. Uses a built-in dataset
of 100 sentences across 10 topics with pre-computed embeddings.

Usage:
    pip install sphereql
    python quickstart.py
"""

import sphereql
from dataset import SENTENCES


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    categories = [s["category"] for s in SENTENCES]
    embeddings = [s["embedding"] for s in SENTENCES]
    texts = [s["text"] for s in SENTENCES]

    # ── Act 1: Visualization ─────────────────────────────────────────
    header("Act 1: What does my knowledge look like?")
    print("SphereQL projects high-dimensional embeddings onto a 3D sphere.")
    print("Each point is a sentence. Color = topic. Proximity = similarity.\n")

    path = sphereql.visualize(
        categories, embeddings,
        labels=texts,
        title="SphereQL Quickstart — 100 Sentences",
        open_browser=True,
    )
    print(f"Visualization saved to: {path}")
    print("Open it in your browser to explore the sphere interactively.")

    # ── Build the pipeline ───────────────────────────────────────────
    pipeline = sphereql.Pipeline(categories, embeddings)
    print(f"\nPipeline built: {pipeline.num_items} items, "
          f"{len(set(categories))} categories")

    # ── Act 2: Nearest Neighbor Search ───────────────────────────────
    header("Act 2: Find me something similar")
    print("Nearest-neighbor search in projected spherical space finds")
    print("semantically related items using angular distance.\n")

    query_idx = 0  # "The double-slit experiment..."
    query_text = texts[query_idx]
    query_emb = embeddings[query_idx]
    print(f"Query: \"{query_text}\"\n")

    results = pipeline.nearest(query_emb, k=5)
    for i, hit in enumerate(results, 1):
        idx = int(hit.id.split("-")[1])
        print(f"  {i}. [{hit.category}] distance={hit.distance:.4f}")
        print(f"     \"{texts[idx]}\"")
    print()

    # ── Act 3: Glob Detection ────────────────────────────────────────
    header("Act 3: What are the natural groupings?")
    print("Glob detection finds clusters of related items on the sphere.")
    print("These emerge from the data — no labels needed.\n")

    globs = pipeline.detect_globs(max_k=10)
    for g in globs:
        top = ", ".join(f"{cat} ({n})" for cat, n in g.top_categories[:3])
        print(f"  Glob {g.id}: {g.member_count} members, "
              f"radius={g.radius:.3f}")
        print(f"    Top categories: {top}")
    print()

    # ── Act 4: Concept Path ──────────────────────────────────────────
    header("Act 4: How do I get from A to B?")
    print("Concept paths trace the shortest route between two items on")
    print("the sphere, revealing how topics connect.\n")

    source_idx = 0   # science
    target_idx = 20  # cooking
    source_id = f"s-{source_idx:04d}"
    target_id = f"s-{target_idx:04d}"
    print(f"From: \"{texts[source_idx]}\"")
    print(f"  To: \"{texts[target_idx]}\"\n")

    path_result = pipeline.concept_path(
        source_id, target_id, graph_k=10, query=query_emb,
    )
    if path_result:
        print(f"  Path found ({len(path_result.steps)} steps, "
              f"total distance={path_result.total_distance:.4f}):")
        for step in path_result.steps:
            idx = int(step.id.split("-")[1])
            print(f"    -> [{step.category}] \"{texts[idx]}\"")
    else:
        print("  No path found (items may be too distant).")
    print()

    # ── Act 5: Local Manifold ────────────────────────────────────────
    header("Act 5: What's the local shape here?")
    print("The local manifold describes the geometry around a query point.")
    print("High variance ratio = items spread across directions (diverse).")
    print("Low variance ratio = items clustered in one direction (focused).\n")

    manifold = pipeline.local_manifold(query_emb, neighborhood_k=15)
    print(f"  Variance ratio: {manifold.variance_ratio:.4f}")
    print(f"  Centroid: [{', '.join(f'{x:.3f}' for x in manifold.centroid)}]")
    print(f"  Normal:   [{', '.join(f'{x:.3f}' for x in manifold.normal)}]")
    print()

    # ── Act 6: Vector DB Bridge ──────────────────────────────────────
    header("Act 6: Plug into my existing vector DB")
    print("VectorStoreBridge connects any vector store to SphereQL's")
    print("analysis pipeline. Here we use InMemoryStore as a demo.\n")

    store = sphereql.InMemoryStore("quickstart", dimension=64)
    records = [
        {"id": f"doc-{i}", "vector": s["embedding"],
         "metadata": {"category": s["category"], "text": s["text"]}}
        for i, s in enumerate(SENTENCES)
    ]
    store.upsert(records)
    print(f"  Upserted {len(store)} records into InMemoryStore")

    bridge = sphereql.VectorStoreBridge(store)
    bridge.build_pipeline(category_key="category")
    print(f"  Built pipeline: {len(bridge)} records")

    hybrid_results = bridge.hybrid_search(
        query_emb, final_k=5, recall_k=20,
    )
    print(f"\n  Hybrid search (ANN recall + angular re-ranking):")
    for i, r in enumerate(hybrid_results, 1):
        print(f"    {i}. [{r['metadata']['category']}] "
              f"score={r['score']:.4f} — {r['id']}")

    updated = bridge.sync_projections()
    print(f"\n  Synced spherical coordinates to {updated} records")

    print("\n  In production, replace InMemoryStore with QdrantStore or")
    print("  PineconeStore to add SphereQL to your existing stack.")

    # ── Done ─────────────────────────────────────────────────────────
    header("Done!")
    print("You've seen SphereQL's core capabilities:")
    print("  - 3D sphere visualization")
    print("  - Nearest neighbor search in spherical space")
    print("  - Automatic cluster (glob) detection")
    print("  - Concept paths between items")
    print("  - Local manifold analysis")
    print("  - Vector DB bridge with hybrid search")
    print("\nNext steps:")
    print("  - Try with your own embeddings (e.g. from sentence-transformers)")
    print("  - Connect to Qdrant or Pinecone for production use")
    print("  - See https://github.com/bkahan/sphereQL for full docs\n")


if __name__ == "__main__":
    main()
