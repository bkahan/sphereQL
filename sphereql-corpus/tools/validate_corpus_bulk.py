#!/usr/bin/env python3
"""Validate a clustered bulk corpus against Phase 7 invariants.

This is the bulk-ingest counterpart to `validate_corpus.py`. It
operates on a Parquet (not JSON), and the categories are emergent
cluster labels (not a fixed 31-cat whitelist), so it can't check
for "expected categories" — it instead checks shape invariants that
matter regardless of which clusters fell out of MiniBatchKMeans.

Checks (sorted by severity):

 1. ✗ Total row count >= `bulk.target_size * 0.9` (allow 10 % loss to
       upstream filters: low-citation OpenAlex works, missing-label
       Wikidata items, etc.)
 2. ✗ Every cluster has >= `cluster.min_size_per_cluster` members
       (auto-merge should make this hold; failure means the clustering
       script didn't run or was given an inconsistent `min` config).
 3. ✗ Mean features/row inside `[validation.min_mean_features,
       validation.max_mean_features]` — same band the legacy validator
       enforces.
 4. ✗ source_confidence has reasonable spread (std > 0.01) — if every
       row is at the floor, the source isn't differentiating quality.
 5. ⚠ Cluster size distribution: warn if any cluster > 10 × the
       median (one mega-cluster typically means we under-clustered).

Exit code: 0 on all-pass, 1 on any ✗.

Usage:
    python3 validate_corpus_bulk.py [path/to/clustered.parquet] \
        [--config PATH] [--set bulk.target_size=500000]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus_config import CorpusConfig, add_config_args, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "bulk_corpus.clustered.parquet"


class Report:
    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  {mark} {label}{suffix}")
        if not ok:
            self.errors += 1
        return ok

    def warn(self, label: str) -> None:
        print(f"  ⚠ {label}")
        self.warnings += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=DEFAULT_CORPUS,
        help="path to clustered parquet (default data/bulk_corpus.clustered.parquet)",
    )
    add_config_args(parser)
    ns = parser.parse_args()

    cfg: CorpusConfig = load_config(ns.config, ns.set)
    rep = Report()

    print(f"Validating {ns.corpus}\n")
    if not ns.corpus.exists():
        rep.check(False, "parquet exists", str(ns.corpus))
        return 1
    rep.check(True, "parquet exists")

    table = pq.read_table(ns.corpus)
    n_rows = table.num_rows

    target = cfg.bulk.target_size
    floor = int(target * 0.9)
    rep.check(
        n_rows >= floor,
        f"row count >= {floor:,} (target {target:,})",
        f"got {n_rows:,}",
    )

    # Category column → cluster id counts. We don't compare names to a
    # whitelist; we just look at the size distribution.
    cats = table.column("category").to_pylist()
    counts = Counter(cats)
    cluster_sizes = sorted(counts.values())
    n_clusters = len(counts)
    smallest = cluster_sizes[0] if cluster_sizes else 0
    largest = cluster_sizes[-1] if cluster_sizes else 0
    median = int(np.median(cluster_sizes)) if cluster_sizes else 0

    rep.check(
        smallest >= cfg.cluster.min_size_per_cluster,
        f"every cluster >= {cfg.cluster.min_size_per_cluster}",
        f"smallest={smallest:,} across {n_clusters} clusters",
    )

    # Feature stats from the list column.
    feats_col = table.column("features").combine_chunks()
    feature_counts = np.array(
        [len(x) for x in feats_col.to_pylist()], dtype=np.int32
    )
    mean_feats = float(feature_counts.mean()) if len(feature_counts) else 0.0
    rep.check(
        cfg.validation.min_mean_features <= mean_feats <= cfg.validation.max_mean_features,
        f"mean features/row in [{cfg.validation.min_mean_features}, "
        f"{cfg.validation.max_mean_features}]",
        f"got {mean_feats:.2f}",
    )

    # source_confidence spread — if the source can't differentiate,
    # downstream quality signals will collapse.
    sc = np.asarray(table.column("source_confidence").to_pylist(), dtype=np.float64)
    sc_std = float(sc.std()) if len(sc) else 0.0
    rep.check(
        sc_std > 0.01,
        "source_confidence shows variance",
        f"std={sc_std:.4f}",
    )

    # Soft warning on a top-heavy distribution.
    if median > 0 and largest > median * 10:
        rep.warn(
            f"cluster size distribution is top-heavy "
            f"(largest {largest:,} vs median {median:,}); consider raising cluster.k"
        )

    print(
        f"\nclusters: {n_clusters}  rows: {n_rows:,}  "
        f"min/median/max size: {smallest:,}/{median:,}/{largest:,}  "
        f"mean features/row: {mean_feats:.2f}  conf std: {sc_std:.4f}"
    )
    print(f"\nerrors: {rep.errors}  warnings: {rep.warnings}")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
