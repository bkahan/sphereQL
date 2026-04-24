"""
Category Enrichment & Hierarchical Routing.

Walks through SphereQL's category-aware surface:
  - PipelineConfig: selecting a projection family and tuning inner-sphere /
    bridge thresholds up front.
  - Category summaries with bridge_quality scoring.
  - Inter-category graph with Genuine / OverlapArtifact / Weak bridge
    classification.
  - Category concept paths with per-hop and end-to-end confidence.
  - Drill-down inside a single category.
  - Domain groups: coarse routing when EVR is low.
  - hierarchical_nearest: group → category → items fallback path.
  - projection_warnings: structured health signals.

Uses the 100-sentence built-in dataset — 64-d FNV-1a hash embeddings, so
EVR is deliberately low (~3–8%). That's realistic for the metalearning
framework: low-EVR is exactly when hierarchical routing earns its keep.

Usage:
    pip install sphereql
    python category_enrichment.py
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

    # ── 1. Build with an explicit PipelineConfig ──────────────────────
    header("1. Build with a PipelineConfig")
    print("Every tunable constant (projection family, inner-sphere gating,")
    print("bridge thresholds, routing) lives in a single PipelineConfig.")
    print("Pass a dict; any field you omit falls back to the default.\n")

    config = {
        "projection_kind": "Pca",
        "inner_sphere": {
            # 10 sentences per category — lower gate so inner spheres build.
            "min_size": 8,
            "min_evr_improvement": 0.01,
            "kernel_pca_min_size": 20,
            "min_kernel_improvement": 0.05,
        },
        "routing": {
            # Lower threshold so hierarchical_nearest actually routes on
            # a 64-d hash corpus where EVR sits below 0.35.
            "low_evr_threshold": 0.10,
            "num_domain_groups": 3,
        },
    }

    pipeline = sphereql.Pipeline(categories, embeddings, config=config)
    print(f"  items:              {pipeline.num_items}")
    print(f"  projection_kind:    {pipeline.projection_kind}")
    print(f"  explained_variance: {pipeline.explained_variance_ratio:.4f}")

    # Round-tripped config dict reflects every default that got filled in.
    back = pipeline.config()
    print(f"  low_evr_threshold:  {back['routing']['low_evr_threshold']}")
    print(f"  num_domain_groups:  {back['routing']['num_domain_groups']}")

    # ── 2. Projection warnings ────────────────────────────────────────
    header("2. Projection quality warnings")
    print("When EVR is below the configured warn threshold, the pipeline")
    print("emits structured warnings so downstream code can decide whether")
    print("to trust fine-grained angular distances.\n")

    warnings = pipeline.projection_warnings()
    if not warnings:
        print("  (no warnings — projection is healthy)")
    for w in warnings:
        print(f"  [{w.severity}] evr={w.evr:.4f}")
        print(f"    {w.message}")

    # ── 3. Category summaries with bridge_quality ─────────────────────
    header("3. Category summaries")
    print("Every category gets a summary. bridge_quality is the mean")
    print("territorial-adjusted bridge strength on outgoing edges —")
    print("higher = this category is connected via clean, genuine bridges.\n")

    summaries, inner_reports = pipeline.category_stats()
    summaries.sort(key=lambda s: -s.bridge_quality)
    print(f"  {'Category':<12} {'members':>7} {'cohesion':>9} "
          f"{'bridge_q':>9}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 9} {'-' * 9}")
    for s in summaries:
        print(f"  {s.name:<12} {s.member_count:>7} "
              f"{s.cohesion:>9.4f} {s.bridge_quality:>9.4f}")

    if inner_reports:
        print(f"\n  Inner spheres built for {len(inner_reports)} categories:")
        for r in inner_reports:
            print(f"    {r.category_name:<12} "
                  f"{r.projection_type:<10} "
                  f"inner_evr={r.inner_evr:.4f} "
                  f"improvement={r.evr_improvement:+.4f}")
    else:
        print("\n  (no inner spheres — tighten inner_sphere.min_size to build them)")

    # ── 4. Category neighbors ─────────────────────────────────────────
    header("4. Category neighbors")
    print("category_neighbors ranks the k closest categories on S² by")
    print("angular distance between centroids.\n")

    neighbors = pipeline.category_neighbors("science", k=4)
    print(f"  Closest categories to 'science':\n")
    for n in neighbors:
        print(f"    {n.name:<12} members={n.member_count}, "
              f"cohesion={n.cohesion:.4f}, bridge_q={n.bridge_quality:.4f}")

    # ── 5. Category concept path ──────────────────────────────────────
    header("5. Category concept path (classification + confidence)")
    print("category_concept_path traces the shortest inter-category route.")
    print("Each step carries its bridge items with classification tags and")
    print("a hop_confidence. path_confidence is the product across hops.\n")

    path = pipeline.category_concept_path("science", "cooking")
    if path is None:
        print("  (no path — categories are disconnected)")
    else:
        print(f"  path_confidence: {path.path_confidence:.4f}")
        print(f"  total_distance:  {path.total_distance:.4f}\n")
        for step in path.steps:
            print(f"  {step.category_name:<12} "
                  f"cum_dist={step.cumulative_distance:.4f} "
                  f"hop_conf={step.hop_confidence:.4f}")
            # Show the strongest bridge and its classification tag.
            bridges = sorted(
                step.bridges_to_next,
                key=lambda b: -b.bridge_strength,
            )[:2]
            for b in bridges:
                marker = {
                    "Genuine": "OK",
                    "OverlapArtifact": "!!",
                    "Weak": "..",
                }.get(b.classification, "??")
                idx = b.item_index
                print(f"     [{marker}] {b.classification:<16} "
                      f"strength={b.bridge_strength:.4f} "
                      f"item={idx}")
                print(f"          \"{texts[idx][:55]}...\"")

    # ── 6. Drill down inside one category ─────────────────────────────
    header("6. Drill-down within a category")
    print("drill_down restricts k-NN to one category, using that category's")
    print("inner-sphere projection when it exists for sharper resolution.\n")

    query_idx = 0  # science
    query_emb = embeddings[query_idx]
    print(f"  Query: \"{texts[query_idx][:55]}...\"\n")

    hits = pipeline.drill_down("science", query_emb, k=3)
    for h in hits:
        tag = "(inner)" if h.used_inner_sphere else "(outer)"
        print(f"    item={h.item_index:<3} distance={h.distance:.4f} {tag}")
        print(f"      \"{texts[h.item_index][:55]}...\"")

    # ── 7. Domain groups (coarse routing) ─────────────────────────────
    header("7. Domain groups")
    print("Below config.routing.low_evr_threshold, angular distance on the")
    print("outer sphere is too noisy to trust. Domain groups collapse the N")
    print("categories into a handful of super-groups derived from Voronoi")
    print("adjacency + cap overlap — coarse but robust.\n")

    groups = pipeline.domain_groups()
    for i, g in enumerate(groups):
        names = ", ".join(g.category_names)
        print(f"  Group {i}: {g.total_items} items, cohesion={g.cohesion:.4f}")
        print(f"    [{names}]")

    # ── 8. hierarchical_nearest vs nearest ────────────────────────────
    header("8. hierarchical_nearest vs plain nearest")
    print("When EVR is above low_evr_threshold, hierarchical_nearest is")
    print("identical to nearest. Below it, the query is routed to a domain")
    print("group first, then drilled into each member category's inner")
    print("sphere. Both modes are exposed so callers can A/B them.\n")

    flat = pipeline.nearest(query_emb, k=5)
    hier = pipeline.hierarchical_nearest(query_emb, k=5)

    print(f"  {'rank':<6} {'flat nearest':<28} {'hierarchical_nearest':<28}")
    print(f"  {'-' * 6} {'-' * 28} {'-' * 28}")
    for i, (f, h) in enumerate(zip(flat, hier), 1):
        fl = f"[{f.category}] d={f.distance:.3f}"
        hl = f"[{h.category}] d={h.distance:.3f}"
        print(f"  {i:<6} {fl:<28} {hl:<28}")

    # ── Done ──────────────────────────────────────────────────────────
    header("Done!")
    print("What you saw:")
    print("  - PipelineConfig: one dict, every knob.")
    print("  - projection_warnings: structured health signals.")
    print("  - category_stats: per-category cohesion + bridge_quality.")
    print("  - category_concept_path: bridges with Genuine / OverlapArtifact")
    print("    / Weak classification, hop_confidence, path_confidence.")
    print("  - drill_down: per-category k-NN with inner-sphere fallback.")
    print("  - domain_groups + hierarchical_nearest: robust routing when")
    print("    EVR is too low for angular distance to be trustworthy.")
    print()
    print("The metalearning framework (auto_tune + MetaModel + feedback)")
    print("picks these knobs for you from corpus profile — see")
    print("metalearning.py for that walkthrough.\n")


if __name__ == "__main__":
    main()
