"""
Metalearning: corpus features → auto-tune → meta-model → feedback.

Walks through SphereQL's three-level self-optimization hierarchy from
Python:

  L1  auto_tune       — per-corpus search over a PipelineConfig space.
  L2  MetaModel       — predict best_config for a new corpus from past
                        (corpus_features, best_config) pairs.
  L3  FeedbackEvent + — blend automated scores with observed user
      FeedbackAggregator satisfaction.

The goal isn't to find the best real-world config on this toy corpus
(100 sentences, 64-d hash embeddings). It's to show the API end to end
so you can wire it up against your own data.

Usage:
    pip install sphereql
    python metalearning.py
"""

import math
import sphereql
from dataset import SENTENCES


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    categories = [s["category"] for s in SENTENCES]
    embeddings = [s["embedding"] for s in SENTENCES]

    # ── 1. Extract a corpus profile ───────────────────────────────────
    header("1. Corpus profile (CorpusFeatures)")
    print("Every meta-model consumes the same 11-field corpus profile.")
    print("This is the query you feed MetaModel.predict().\n")

    features = sphereql.corpus_features(categories, embeddings)
    for key in [
        "n_items", "n_categories", "dim",
        "mean_members_per_category", "category_size_entropy",
        "mean_sparsity", "axis_utilization_entropy",
        "noise_estimate",
        "mean_intra_category_similarity",
        "mean_inter_category_similarity",
        "category_separation_ratio",
    ]:
        val = features[key]
        if isinstance(val, float):
            print(f"  {key:<34} {val:>9.4f}")
        else:
            print(f"  {key:<34} {val:>9}")

    # ── 2. Baseline score with default config ─────────────────────────
    header("2. Baseline: default PipelineConfig")

    baseline = sphereql.Pipeline(categories, embeddings)
    print(f"  projection_kind:    {baseline.projection_kind}")
    print(f"  explained_variance: {baseline.explained_variance_ratio:.4f}")

    # ── 3. auto_tune with a custom search_space ───────────────────────
    header("3. auto_tune: find a config that maximizes a quality metric")
    print("A small Grid sweep over a narrowed SearchSpace. On real data you")
    print("typically use 'random' or 'bayesian' with a larger budget. The")
    print("tuner prefits each projection family once, so per-trial cost is")
    print("just graph construction + quality evaluation.\n")

    search_space = {
        # Kernel PCA is O(n²·d) — skip it for speed on toy corpora.
        "projection_kinds": ["Pca", "LaplacianEigenmap"],
        # Override defaults for three knobs. Others fall back.
        "num_domain_groups": [3, 5],
        "low_evr_threshold": [0.15, 0.35],
        "laplacian_k_neighbors": [10, 20],
    }

    tuned_pipeline, report = sphereql.auto_tune(
        categories, embeddings,
        metric="default_composite",
        strategy="grid",
        search_space=search_space,
    )
    best_config = report["best_config"]

    print(f"  metric:         {report['metric_name']}")
    print(f"  trials:         {report['trials_count']} "
          f"(failures: {report['failures_count']})")
    print(f"  mean_score:     {report['mean_score']:.4f}")
    print(f"  best_score:     {report['best_score']:.4f}")
    print(f"  best kind:      {best_config['projection_kind']}")
    print(f"  best groups:    {best_config['routing']['num_domain_groups']}")
    print(f"  best low_evr:   {best_config['routing']['low_evr_threshold']}")

    print(f"\n  Tuned pipeline live:")
    print(f"    projection_kind    = {tuned_pipeline.projection_kind}")
    print(f"    explained_variance = "
          f"{tuned_pipeline.explained_variance_ratio:.4f}")

    # ── 4. Build a MetaModel training set ─────────────────────────────
    header("4. MetaModel: predict a config for a new corpus")
    print("A real training set comes from saving MetaTrainingRecords after")
    print("each auto_tune run. Here we synthesize three records by tuning")
    print("three perturbed subsets of the same dataset — enough to show")
    print("the NN and distance-weighted models diverging.\n")

    def stratified_subset(fraction):
        """Take the first `fraction` of each category (stable, deterministic)."""
        per_cat = {}
        for i, c in enumerate(categories):
            per_cat.setdefault(c, []).append(i)
        keep = []
        for idxs in per_cat.values():
            cutoff = max(2, math.ceil(len(idxs) * fraction))
            keep.extend(idxs[:cutoff])
        keep.sort()
        return (
            [categories[i] for i in keep],
            [embeddings[i] for i in keep],
        )

    records = []
    for corpus_id, frac in [("half_corpus", 0.5), ("quarter_corpus", 0.25),
                            ("full_corpus", 1.0)]:
        cats, embs = stratified_subset(frac)
        feats = sphereql.corpus_features(cats, embs)
        _, sub_report = sphereql.auto_tune(
            cats, embs,
            metric="default_composite",
            strategy="random",
            budget=6,
            seed=1,
        )
        records.append({
            "corpus_id": corpus_id,
            "features": feats,
            "best_config": sub_report["best_config"],
            "best_score": sub_report["best_score"],
            "metric_name": sub_report["metric_name"],
            "strategy": f"random(budget=6, seed=1)",
            "timestamp": "0",
        })
        print(f"  trained on {corpus_id:<16} "
              f"(n={feats['n_items']}, best_score={sub_report['best_score']:.4f})")

    # ── 5. Fit both meta-models and compare predictions ───────────────
    header("5. Predict a config for the full corpus")

    nn_model = sphereql.NearestNeighborMetaModel()
    nn_model.fit(records)

    dw_model = sphereql.DistanceWeightedMetaModel(epsilon=0.1)
    dw_model.fit(records)

    nn_config = nn_model.predict(features)
    dw_config = dw_model.predict(features)

    print(f"  NN model:  projection={nn_config['projection_kind']}, "
          f"groups={nn_config['routing']['num_domain_groups']}")
    print(f"  DW model:  projection={dw_config['projection_kind']}, "
          f"groups={dw_config['routing']['num_domain_groups']}")

    # Build a pipeline from the NN-predicted config without re-running
    # the tuner — the recall path.
    recalled = sphereql.Pipeline(categories, embeddings, config=nn_config)
    print(f"\n  Pipeline from NN prediction (no tuner re-run):")
    print(f"    projection_kind    = {recalled.projection_kind}")
    print(f"    explained_variance = "
          f"{recalled.explained_variance_ratio:.4f}")

    # ── 6. Feedback loop (L3) ─────────────────────────────────────────
    header("6. Feedback: blend user signals with automated scores")
    print("Automated quality metrics only see pipeline geometry. Feedback")
    print("captures what users actually experience. FeedbackAggregator")
    print("summarizes events per corpus_id.\n")

    agg = sphereql.FeedbackAggregator()
    for q, score in [
        ("q-0001", 0.9), ("q-0002", 0.7), ("q-0003", 1.0),
        ("q-0004", 0.3),  # one unhappy user
        ("q-0005", 0.85),
    ]:
        agg.record(sphereql.FeedbackEvent("full_corpus", q, score))

    print(f"  aggregator has {len(agg)} events across "
          f"{len(agg.corpus_ids())} corpora")

    summary = agg.summarize("full_corpus")
    print(f"\n  full_corpus summary:")
    print(f"    n_events:  {summary['n_events']}")
    print(f"    mean:      {summary['mean_score']:.4f}")
    print(f"    range:     [{summary['min_score']:.2f}, "
          f"{summary['max_score']:.2f}]")

    # Blend automated best_score with the feedback mean. Pass alpha=0
    # to ignore feedback entirely, alpha=1 to trust it fully.
    record = records[-1]  # full_corpus
    blended = (
        0.5 * record["best_score"] + 0.5 * summary["mean_score"]
    )
    print(f"\n  automated best_score: {record['best_score']:.4f}")
    print(f"  feedback mean:        {summary['mean_score']:.4f}")
    print(f"  blended (alpha=0.5):  {blended:.4f}")

    # ── 7. Persisting records for next session ────────────────────────
    header("7. Persisting records")
    print("MetaTrainingRecord.append_to_default_store() writes to")
    print("~/.sphereql/meta_records.json. FeedbackAggregator.save(path)")
    print("does the same for feedback events. Here we skip the write so")
    print("the example is idempotent; the equivalent Python is:\n")
    print("    sphereql.append_to_default_store(record)")
    print("    existing = sphereql.load_default_store()")
    print()
    print("    event = sphereql.FeedbackEvent(cid, qid, 0.9)")
    print("    event.append_to_default_store()")
    print("    agg = sphereql.FeedbackAggregator.load_default()")

    # ── Done ──────────────────────────────────────────────────────────
    header("Done!")
    print("You've walked the metalearning ladder end to end:")
    print("  L1  auto_tune(...)          — search best config per corpus")
    print("  L2  MetaModel(...).predict  — recall config for new corpora")
    print("  L3  FeedbackAggregator       — refine scores from real use")
    print()
    print("Wiring these in sequence gives you a pipeline that gets better")
    print("the more corpora you ship it against — not just better per-run.\n")


if __name__ == "__main__":
    main()
