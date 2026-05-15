#!/usr/bin/env python3
"""Emergent clustering for a bulk-ingested corpus.

Reads a Parquet emitted by `bulk_ingest`, fits MiniBatchKMeans on the
sparse 128-axis features, and writes a *new* Parquet with the
``category`` column overwritten by the cluster label each row was
assigned. Cluster names are stable, human-readable strings of the
form ``cluster_NN__<top-anchor-1>__<top-anchor-2>`` derived from the
two rows closest to each centroid.

Why a separate file from `bulk_ingest`: clustering is a
post-processing step that requires the full set of features in memory
(at least at fit time). The Rust ingest path stays streaming + bounded;
this Python step is the only piece that touches the whole corpus at
once. For 50M+ rows, fit on a `--sample-size` and assign the rest by
nearest-centroid (sklearn does both in one call).

Usage:
    python3 cluster_bulk.py [--corpus PATH] [--out PATH] [--config PATH]
    python3 cluster_bulk.py --corpus /tmp/bulk_corpus.parquet --set cluster.k=128

The `[cluster]` table in `tools/corpus_config.toml` carries the
defaults; override at the CLI with `--set cluster.k=128` etc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse as sp
from sklearn.cluster import MiniBatchKMeans

from corpus_config import ClusterConfig, CorpusConfig, add_config_args, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "bulk_corpus.parquet"
NUM_AXES = 128
LABEL_SLUG_RE = re.compile(r"[^0-9A-Za-z]+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="path to ingested parquet (default data/bulk_corpus.parquet)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="path to clustered parquet (default <corpus>.clustered.parquet)",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="path to JSON cluster-metadata sidecar (default <out>.clusters.json)",
    )
    add_config_args(parser)
    ns = parser.parse_args()

    cfg: CorpusConfig = load_config(ns.config, ns.set)
    cluster_cfg = cfg.cluster
    corpus_path = ns.corpus or DEFAULT_CORPUS
    out_path = ns.out or _default_out(corpus_path)
    sidecar_path = ns.sidecar or out_path.with_suffix(out_path.suffix + ".clusters.json")

    if not corpus_path.exists():
        print(f"error: corpus parquet not found at {corpus_path}", file=sys.stderr)
        return 2

    print(f"→ loading {corpus_path}")
    t0 = time.perf_counter()
    table = pq.read_table(corpus_path)
    n_rows = table.num_rows
    print(f"  loaded {n_rows:,} concepts in {time.perf_counter() - t0:.2f}s")
    if n_rows == 0:
        print("error: empty corpus, nothing to cluster", file=sys.stderr)
        return 2

    print("→ materializing sparse feature matrix")
    t0 = time.perf_counter()
    features = _build_sparse_features(table, NUM_AXES)
    print(
        f"  built {features.shape} CSR, nnz={features.nnz:,} "
        f"({features.nnz / max(1, n_rows):.1f} features/row avg) "
        f"in {time.perf_counter() - t0:.2f}s"
    )

    print(f"→ fitting MiniBatchKMeans k={cluster_cfg.k} batch={cluster_cfg.batch_size}")
    t0 = time.perf_counter()
    centroids, assignments = _fit_kmeans(features, cluster_cfg)
    print(f"  fit + assign in {time.perf_counter() - t0:.2f}s")

    if cluster_cfg.auto_merge:
        assignments, centroids, merges = _auto_merge_tiny_clusters(
            features, assignments, centroids, cluster_cfg.min_size_per_cluster
        )
        if merges:
            print(f"  merged {len(merges)} underweight clusters → larger neighbors")

    sizes = np.bincount(assignments, minlength=centroids.shape[0])
    nonempty = np.count_nonzero(sizes)
    print(f"  {nonempty} non-empty clusters; min={sizes[sizes > 0].min()} max={sizes.max()}")

    print("→ generating emergent labels per cluster")
    labels = table.column("label").to_pylist()
    cluster_names, anchors_per_cluster = _name_clusters(
        features,
        assignments,
        centroids,
        labels,
        cluster_cfg,
    )

    print(f"→ writing {out_path}")
    new_table = _replace_category(table, [cluster_names[c] for c in assignments])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, out_path, compression="snappy")

    sidecar = _build_sidecar(
        cluster_cfg, features, assignments, centroids, cluster_names, anchors_per_cluster
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"  sidecar: {sidecar_path}")

    return 0


def _default_out(corpus_path: Path) -> Path:
    stem = corpus_path.stem
    return corpus_path.with_name(f"{stem}.clustered.parquet")


def _build_sparse_features(table: pa.Table, num_axes: int) -> sp.csr_matrix:
    """Project the parquet `features` column into a sparse (rows × axes) CSR.

    The features column is a `list<struct<axis: u32, weight: f64>>`;
    pyarrow returns it as a list-of-list-of-dicts via `to_pylist()`.
    Reading via `combine_chunks` first keeps the structure consistent
    across row groups.
    """
    col = table.column("features").combine_chunks()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_idx, feats in enumerate(col.to_pylist()):
        if not feats:
            continue
        for f in feats:
            axis = int(f["axis"])
            if 0 <= axis < num_axes:
                rows.append(row_idx)
                cols.append(axis)
                data.append(float(f["weight"]))
    n_rows = table.num_rows
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(n_rows, num_axes), dtype=np.float32
    )


def _fit_kmeans(features: sp.csr_matrix, cfg: ClusterConfig):
    """Fit MiniBatchKMeans, optionally on a random row sample.

    Returns (centroids, full-corpus assignments). At very large scale,
    `cfg.sample_size > 0` fits on a sample but always assigns the full
    corpus via `predict` — the assignment pass is cheap (one pass,
    bounded memory) so it's safe to apply to billions of rows when
    needed.
    """
    n = features.shape[0]
    rng = np.random.default_rng(cfg.random_seed)
    if cfg.sample_size > 0 and cfg.sample_size < n:
        idx = rng.choice(n, cfg.sample_size, replace=False)
        fit_set = features[idx]
    else:
        fit_set = features

    k = min(cfg.k, max(2, int(n ** 0.5))) if n < cfg.k else cfg.k
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=cfg.batch_size,
        max_iter=cfg.max_iter,
        n_init=cfg.n_init,
        random_state=cfg.random_seed,
        verbose=0,
    )
    km.fit(fit_set)
    assignments = km.predict(features).astype(np.int32)
    return km.cluster_centers_.astype(np.float32), assignments


def _auto_merge_tiny_clusters(
    features: sp.csr_matrix,
    assignments: np.ndarray,
    centroids: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Reassign members of clusters below `min_size` to their nearest
    *surviving* centroid. Iterates until every remaining cluster
    satisfies the size floor or we run out of merges. Returns the
    rewritten assignments, the pruned centroid matrix, and the merge
    history (small_cluster → absorbed_into)."""
    assignments = assignments.copy()
    survivors = list(range(centroids.shape[0]))
    merges: list[tuple[int, int]] = []
    sizes = np.bincount(assignments, minlength=centroids.shape[0])
    while True:
        # Smallest survivor that's still under the floor.
        small = [c for c in survivors if sizes[c] < min_size]
        if not small or len(survivors) <= 2:
            break
        c = min(small, key=lambda i: sizes[i])
        keep = [s for s in survivors if s != c]
        # Nearest surviving centroid by L2.
        kept_centroids = centroids[keep]
        dist = np.linalg.norm(kept_centroids - centroids[c], axis=1)
        target = keep[int(np.argmin(dist))]
        mask = assignments == c
        assignments[mask] = target
        sizes[target] += sizes[c]
        sizes[c] = 0
        merges.append((c, target))
        survivors = keep

    # Pack the surviving cluster ids down to a dense range so the
    # consumer's per-cluster arrays line up. Build a translation map.
    remap = {old: new for new, old in enumerate(sorted(survivors))}
    new_assignments = np.array([remap[c] for c in assignments], dtype=np.int32)
    new_centroids = centroids[sorted(survivors)]
    return new_assignments, new_centroids, merges


def _name_clusters(
    features: sp.csr_matrix,
    assignments: np.ndarray,
    centroids: np.ndarray,
    labels: list[str],
    cfg: ClusterConfig,
) -> tuple[list[str], list[list[str]]]:
    """Pick the rows closest to each centroid as readable anchors.

    Naming pattern: ``cluster_NN__<anchor1>__<anchor2>``. Anchors are
    slugged to ascii-safe identifiers — the column is consumed by Rust
    code that expects byte-clean strings. NN is zero-padded to 4 digits
    so lex order matches numeric order at any reasonable k.
    """
    n_clusters = centroids.shape[0]
    names: list[str] = []
    anchors_per_cluster: list[list[str]] = []
    for k in range(n_clusters):
        member_mask = assignments == k
        if not member_mask.any():
            names.append(f"cluster_{k:04d}")
            anchors_per_cluster.append([])
            continue
        member_idx = np.where(member_mask)[0]
        # L2 distance from each member to the centroid.
        sub = features[member_idx].toarray()
        diffs = sub - centroids[k]
        dist = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dist)[: max(2, cfg.top_concepts_per_cluster)]
        chosen = [labels[member_idx[i]] for i in order]
        slug_anchors = [_slug(a) for a in chosen[:2]]
        suffix = "__".join(s for s in slug_anchors if s)
        names.append(f"cluster_{k:04d}__{suffix}" if suffix else f"cluster_{k:04d}")
        anchors_per_cluster.append(chosen)
    return names, anchors_per_cluster


def _slug(label: str) -> str:
    s = LABEL_SLUG_RE.sub("_", label).strip("_")
    # Trim long slugs to keep the category column tidy.
    return s[:48].lower()


def _replace_category(table: pa.Table, new_categories: list[str]) -> pa.Table:
    """Return a new Arrow table identical to `table` except for the
    `category` column, which is replaced by `new_categories`. Keeps
    the rest of the schema (and per-row metadata columns like source,
    openalex_id) untouched."""
    col_index = table.schema.get_field_index("category")
    if col_index == -1:
        raise SystemExit("input parquet is missing the `category` column")
    arr = pa.array(new_categories, type=pa.string())
    return table.set_column(col_index, "category", arr)


def _build_sidecar(
    cfg: ClusterConfig,
    features: sp.csr_matrix,
    assignments: np.ndarray,
    centroids: np.ndarray,
    cluster_names: list[str],
    anchors_per_cluster: list[list[str]],
) -> dict:
    """Compact JSON metadata for inspection / downstream consumers."""
    n_clusters = centroids.shape[0]
    per_cluster_top_axes = []
    sizes = np.bincount(assignments, minlength=n_clusters).tolist()
    for k in range(n_clusters):
        # The centroid is already the mean feature vector for cluster k.
        idx = np.argsort(-centroids[k])[: cfg.top_axes_per_cluster]
        per_cluster_top_axes.append(
            [{"axis": int(i), "mean_weight": float(centroids[k][i])} for i in idx]
        )

    return {
        "cluster_config": {
            "k_requested": cfg.k,
            "k_effective": n_clusters,
            "batch_size": cfg.batch_size,
            "max_iter": cfg.max_iter,
            "n_init": cfg.n_init,
            "random_seed": cfg.random_seed,
            "min_size_per_cluster": cfg.min_size_per_cluster,
            "auto_merge": cfg.auto_merge,
            "sample_size": cfg.sample_size,
        },
        "corpus": {
            "n_rows": int(features.shape[0]),
            "num_axes": int(features.shape[1]),
            "nnz": int(features.nnz),
        },
        "clusters": [
            {
                "id": k,
                "name": cluster_names[k],
                "size": sizes[k],
                "top_axes": per_cluster_top_axes[k],
                "anchor_labels": anchors_per_cluster[k][: cfg.top_concepts_per_cluster],
            }
            for k in range(n_clusters)
        ],
    }


if __name__ == "__main__":
    sys.exit(main())
