"""
VectorStoreBridge with the full category layer.

VectorStoreBridge now mirrors the stand-alone Pipeline's query surface.
The same bridge object gives you:

  - PipelineConfig-driven builds (pick projection family + tune thresholds).
  - Category enrichment: neighbors, concept paths with classified bridges,
    drill-down, per-category stats.
  - Hierarchical routing + domain groups for low-EVR corpora.
  - Projection warnings.
  - Hybrid search: ANN recall + exact-cosine re-rank on the store side.

This example uses InMemoryStore for reproducibility. Swap in QdrantBridge
or PineconeBridge with identical call signatures for production use.

Usage:
    pip install sphereql
    python vectordb_advanced.py
"""

import sphereql
from dataset import SENTENCES


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    texts = [s["text"] for s in SENTENCES]

    # ── 1. Populate an InMemoryStore ─────────────────────────────────
    header("1. Populate an InMemoryStore")

    store = sphereql.InMemoryStore("advanced_demo", dimension=64)
    records = [
        {
            "id": f"doc-{i:04d}",
            "vector": s["embedding"],
            "metadata": {"category": s["category"], "text": s["text"]},
        }
        for i, s in enumerate(SENTENCES)
    ]
    store.upsert(records)
    print(f"  upserted {len(store)} records")

    # ── 2. Build with an explicit PipelineConfig ─────────────────────
    header("2. build_pipeline_with_config")
    print("The legacy build_pipeline(category_key=...) still works as a")
    print("PCA-only path. build_pipeline_with_config lets you pick any")
    print("projection family and override bridge / routing thresholds.\n")

    config = {
        "projection_kind": "Pca",
        "inner_sphere": {
            "min_size": 8,
            "min_evr_improvement": 0.01,
            "kernel_pca_min_size": 20,
            "min_kernel_improvement": 0.05,
        },
        "routing": {
            "low_evr_threshold": 0.10,
            "num_domain_groups": 3,
        },
    }

    bridge = sphereql.VectorStoreBridge(store)
    bridge.build_pipeline_with_config(config, category_key="category")

    print(f"  records in bridge:  {len(bridge)}")
    print(f"  projection_kind:    {bridge.projection_kind}")

    back = bridge.config()
    print(f"  low_evr_threshold:  {back['routing']['low_evr_threshold']}")
    print(f"  num_domain_groups:  {back['routing']['num_domain_groups']}")

    # ── 3. Projection warnings on the bridge ─────────────────────────
    header("3. projection_warnings")
    warnings = bridge.projection_warnings()
    if not warnings:
        print("  (projection is healthy)")
    for w in warnings:
        print(f"  [{w.severity}] evr={w.evr:.4f}")
        print(f"    {w.message}")

    # ── 4. Standard query surface ────────────────────────────────────
    header("4. Standard query surface")

    query_vec = SENTENCES[0]["embedding"]
    print(f"  Query: \"{texts[0][:55]}...\"\n")

    print("  query_nearest (k=3):")
    for i, h in enumerate(bridge.query_nearest(query_vec, k=3), 1):
        print(f"    {i}. [{h.category}] d={h.distance:.4f} id={h.id}")

    print("\n  query_similar (min_cosine=0.6):")
    similar = bridge.query_similar(query_vec, min_cosine=0.6)
    print(f"    -> {len(similar)} above threshold")
    for h in similar[:3]:
        print(f"       [{h.category}] d={h.distance:.4f} id={h.id}")

    print("\n  query_local_manifold (neighborhood_k=10):")
    m = bridge.query_local_manifold(query_vec, neighborhood_k=10)
    print(f"    variance_ratio={m.variance_ratio:.4f}")

    # ── 5. Category layer through the bridge ─────────────────────────
    header("5. Category enrichment through the bridge")

    summaries, inner_reports = bridge.category_stats()
    print(f"  category_stats: {len(summaries)} summaries, "
          f"{len(inner_reports)} inner-sphere reports\n")

    summaries.sort(key=lambda s: -s.bridge_quality)
    print(f"  {'Category':<12} {'members':>7} {'cohesion':>9} "
          f"{'bridge_q':>9}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 9} {'-' * 9}")
    for s in summaries[:5]:
        print(f"  {s.name:<12} {s.member_count:>7} "
              f"{s.cohesion:>9.4f} {s.bridge_quality:>9.4f}")

    print(f"\n  category_neighbors('science', k=3):")
    for n in bridge.category_neighbors("science", k=3):
        print(f"    {n.name:<12} members={n.member_count}, "
              f"cohesion={n.cohesion:.4f}")

    print(f"\n  category_concept_path(science -> cooking):")
    path = bridge.category_concept_path("science", "cooking")
    if path is None:
        print("    (no path)")
    else:
        print(f"    path_confidence={path.path_confidence:.4f} "
              f"total_distance={path.total_distance:.4f}")
        for step in path.steps:
            genuine = sum(
                1 for b in step.bridges_to_next
                if b.classification == "Genuine"
            )
            print(f"      {step.category_name:<12} "
                  f"hop_conf={step.hop_confidence:.4f} "
                  f"bridges={len(step.bridges_to_next)} "
                  f"(genuine: {genuine})")

    print(f"\n  drill_down('science', query, k=3):")
    for h in bridge.drill_down("science", query_vec, k=3):
        tag = "(inner)" if h.used_inner_sphere else "(outer)"
        print(f"    item={h.item_index:<3} distance={h.distance:.4f} {tag}")

    # ── 6. Hierarchical routing ──────────────────────────────────────
    header("6. Domain groups + hierarchical_nearest")
    groups = bridge.domain_groups()
    for i, g in enumerate(groups):
        print(f"  Group {i}: {g.total_items} items, "
              f"cohesion={g.cohesion:.4f}")
        print(f"    [{', '.join(g.category_names)}]")

    print(f"\n  hierarchical_nearest (k=3):")
    for i, h in enumerate(bridge.hierarchical_nearest(query_vec, k=3), 1):
        print(f"    {i}. [{h.category}] d={h.distance:.4f} id={h.id}")

    # ── 7. Hybrid search + sync ──────────────────────────────────────
    header("7. hybrid_search + sync_projections")
    print("hybrid_search uses the store's ANN for recall and re-ranks by")
    print("exact cosine in the original high-d space — bypassing the lossy")
    print("3D projection for final scoring.\n")

    for i, r in enumerate(bridge.hybrid_search(query_vec, final_k=3), 1):
        print(f"  {i}. [{r['metadata']['category']}] "
              f"score={r['score']:.4f} id={r['id']}")

    # Write spherical coordinates back into the store's payload so
    # downstream services can read them without rebuilding the pipeline.
    updated = bridge.sync_projections()
    print(f"\n  sync_projections: wrote r/theta/phi to {updated} records")

    fetched = store.count()
    print(f"  store still holds {fetched} records (metadata enriched in place)")

    # ── Done ──────────────────────────────────────────────────────────
    header("Done!")
    print("Everything this example did works identically against Qdrant or")
    print("Pinecone — swap InMemoryStore for:")
    print()
    print("    store = sphereql.QdrantBridge(")
    print("        url='http://localhost:6334',")
    print("        collection='my-vectors',")
    print("        dimension=64,")
    print("    )")
    print()
    print("or sphereql.PineconeBridge(api_key, host, dimension). Every")
    print("method shown here (build_pipeline_with_config, category_stats,")
    print("hierarchical_nearest, hybrid_search, ...) has the same signature")
    print("across all three bridge types.\n")


if __name__ == "__main__":
    main()
