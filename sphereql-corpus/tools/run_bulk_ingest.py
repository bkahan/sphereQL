#!/usr/bin/env python3
"""Drive the Rust `bulk_ingest` example from corpus_config.toml.

Reads the `[bulk]` table (plus the per-source subtable that matches
`bulk.source`), renders the corresponding CLI args, and `exec`s the
Rust binary. This is the supported way to scale the corpus to 500K
or beyond — bypassing the legacy Python `generate_extended.py` path
that loads everything into memory.

Usage:
    python3 run_bulk_ingest.py [--config PATH] [--set bulk.target_size=1000000]

Common overrides:
    --set bulk.source=openalex_shard
    --set bulk.target_size=5000000
    --set bulk.output=/data/wikidata_5m.parquet
    --set bulk.resume=true

The default `bulk.source` is "wikidata_sparql", which only needs the
default cargo features. "openalex_shard" requires `--features bulk-gzip`
(on by default), and "wikidata_dump" requires `--features bulk-dump`
(off by default).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from corpus_config import BulkConfig, CorpusConfig, add_config_args, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CRATE_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent


def _resolve_output(raw: str) -> Path:
    """[bulk].output is relative to tools/ by convention. The Rust
    binary runs from REPO_ROOT, so relative paths must be anchored
    here, not handed through verbatim."""
    p = Path(raw)
    return p if p.is_absolute() else (TOOLS_DIR / p).resolve()


def render_args(bulk: BulkConfig) -> list[str]:
    """Translate a BulkConfig into CLI flags for `bulk_ingest`."""
    args: list[str] = [
        "--source",
        bulk.source,
        "--out",
        str(_resolve_output(bulk.output)),
        "--target-size",
        str(bulk.target_size),
        "--num-axes",
        str(bulk.num_axes),
        "--axis-seed",
        hex(bulk.axis_seed),
        "--batch-size",
        str(bulk.batch_size),
    ]
    if bulk.resume:
        args.append("--resume")

    if bulk.source == "wikidata_sparql":
        s = bulk.wikidata_sparql
        args += [
            "--sparql-page-size",
            str(s.page_size),
            "--sparql-sleep-ms",
            str(s.inter_page_sleep_ms),
            "--sparql-retries",
            str(s.retries),
        ]
    elif bulk.source == "openalex_shard":
        s = bulk.openalex_shard
        args += [
            "--shard-dir",
            s.shard_dir,
            "--min-cited-by",
            str(s.min_cited_by),
            "--min-year",
            str(s.min_year),
        ]
    elif bulk.source == "wikidata_dump":
        s = bulk.wikidata_dump
        args += ["--dump", s.dump_path]
        if not s.only_items:
            args.append("--dump-all-types")
        if not s.require_english_label:
            args.append("--dump-allow-missing-label")
    elif bulk.source == "dbpedia":
        d = bulk.dbpedia
        args += [
            "--dbpedia-dir",
            d.dir,
            "--dbpedia-oversample",
            str(d.oversample),
        ]
    else:
        raise SystemExit(
            f"unknown bulk.source {bulk.source!r}; "
            "expected wikidata_sparql | openalex_shard | wikidata_dump | dbpedia"
        )
    return args


def required_features(bulk: BulkConfig) -> list[str]:
    """Cargo --features needed for the chosen source."""
    if bulk.source == "wikidata_sparql":
        return ["bulk-http"]
    if bulk.source == "openalex_shard":
        return ["bulk-gzip"]
    if bulk.source == "wikidata_dump":
        return ["bulk-dump"]
    if bulk.source == "dbpedia":
        return ["bulk-dbpedia"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--release",
        action="store_true",
        default=True,
        help="build in release mode (default: on; use --no-release to override)",
    )
    parser.add_argument(
        "--no-release",
        dest="release",
        action="store_false",
        help="build in dev mode (faster compile, much slower run)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print the cargo command and exit without running it",
    )
    ns = parser.parse_args()

    cfg: CorpusConfig = load_config(ns.config, ns.set)
    bulk = cfg.bulk

    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("`cargo` not on PATH; install Rust toolchain via rustup")

    features = required_features(bulk)
    cmd = [
        cargo,
        "run",
        "--manifest-path",
        str(CRATE_DIR / "Cargo.toml"),
        "--example",
        "bulk_ingest",
    ]
    if ns.release:
        cmd.append("--release")
    if features:
        cmd += ["--features", ",".join(features)]
    cmd.append("--")
    cmd += render_args(bulk)

    print("$ " + " ".join(_shell_quote(a) for a in cmd))
    if ns.print_only:
        return

    out_dir = _resolve_output(bulk.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=os.environ.copy())
    sys.exit(rc)


def _shell_quote(s: str) -> str:
    if any(c in s for c in " \t\n\"'\\$`!&|;<>()[]{}*?#~="):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    main()
