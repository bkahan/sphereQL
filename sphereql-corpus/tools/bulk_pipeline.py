#!/usr/bin/env python3
"""End-to-end bulk corpus pipeline.

Chains the four scripts that together turn an empty data directory
into a self-tuned, emergent-cluster corpus:

  1. `run_bulk_ingest.py`       Rust streaming ingest (Wikidata SPARQL
                                / OpenAlex shards / Wikidata dump) →
                                ``bulk.output`` parquet + checkpoint.
  2. `cluster_bulk.py`          MiniBatchKMeans on the 128-axis sparse
                                features → ``*.clustered.parquet`` with
                                emergent category labels + JSON sidecar.
  3. `validate_corpus_bulk.py`  Shape invariants (size, mean features,
                                source confidence spread, cluster
                                minimum size).
  4. `corpus_self_tune`         Rust binary; reweights + prunes using
                                the softened `[self_tune_bulk]` profile,
                                writes ``*.tuned.parquet``.

Each stage can be skipped via flags so the pipeline acts as both an
end-to-end runner and a "resume from stage N" tool.

Usage:
    # full pipeline with defaults from corpus_config.toml
    python3 bulk_pipeline.py

    # smoke test on an existing synthesized parquet (skip ingest)
    python3 bulk_pipeline.py --skip-ingest \
        --corpus /tmp/synthetic_500k.parquet --set bulk.target_size=500000

    # ingest only (e.g. resume into a fresh cluster pass later)
    python3 bulk_pipeline.py --only-ingest
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from corpus_config import (
    CorpusConfig,
    SelfTuneBulkConfig,
    add_config_args,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="override ingest output / cluster input (default = [bulk].output)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="skip stage 1 (Rust ingest)"
    )
    parser.add_argument(
        "--skip-cluster", action="store_true", help="skip stage 2 (k-means)"
    )
    parser.add_argument(
        "--skip-validate", action="store_true", help="skip stage 3 (validator)"
    )
    parser.add_argument(
        "--skip-self-tune", action="store_true", help="skip stage 4 (Rust self-tune)"
    )
    parser.add_argument(
        "--only-ingest", action="store_true", help="run stage 1 only"
    )
    parser.add_argument(
        "--only-cluster", action="store_true", help="run stage 2 only"
    )
    parser.add_argument(
        "--release/--no-release",
        dest="release",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--release",
        dest="release",
        action="store_true",
        default=True,
        help="cargo --release (default; --no-release for dev build)",
    )
    parser.add_argument(
        "--no-release", dest="release", action="store_false", help=argparse.SUPPRESS
    )
    ns = parser.parse_args()

    if ns.only_ingest:
        ns.skip_cluster = ns.skip_validate = ns.skip_self_tune = True
    if ns.only_cluster:
        ns.skip_ingest = ns.skip_validate = ns.skip_self_tune = True

    cfg: CorpusConfig = load_config(ns.config, ns.set)
    corpus_path = Path(ns.corpus) if ns.corpus else _resolve_bulk_output(cfg)

    # Per-stage artifacts.
    clustered_path = corpus_path.with_name(
        corpus_path.stem + ".clustered.parquet"
    )
    tuned_path = clustered_path.with_name(
        clustered_path.stem + ".tuned.parquet"
    )

    t_total = time.perf_counter()

    if not ns.skip_ingest:
        _section("stage 1/4 — Rust bulk_ingest")
        rc = _run(
            [
                sys.executable,
                str(TOOLS_DIR / "run_bulk_ingest.py"),
                *_passthrough_config(ns),
            ]
        )
        if rc != 0:
            return rc

    if not ns.skip_cluster:
        _section("stage 2/4 — cluster_bulk.py")
        rc = _run(
            [
                sys.executable,
                str(TOOLS_DIR / "cluster_bulk.py"),
                "--corpus",
                str(corpus_path),
                "--out",
                str(clustered_path),
                *_passthrough_config(ns),
            ]
        )
        if rc != 0:
            return rc

    if not ns.skip_validate:
        _section("stage 3/4 — validate_corpus_bulk.py")
        rc = _run(
            [
                sys.executable,
                str(TOOLS_DIR / "validate_corpus_bulk.py"),
                str(clustered_path),
                *_passthrough_config(ns),
            ]
        )
        if rc != 0:
            print(
                "validation failed; refusing to self-tune a non-passing corpus",
                file=sys.stderr,
            )
            return rc

    if not ns.skip_self_tune:
        _section("stage 4/4 — sphereql-embed corpus_self_tune")
        rc = _run_self_tune(clustered_path, tuned_path, cfg.self_tune_bulk, ns.release)
        if rc != 0:
            return rc

    print(
        f"\n✓ pipeline complete in {time.perf_counter() - t_total:.1f}s "
        f"({_summary(corpus_path, clustered_path, tuned_path, ns)})"
    )
    return 0


def _resolve_bulk_output(cfg: CorpusConfig) -> Path:
    """The TOML [bulk].output is relative to the tools/ directory by
    convention (matches how the Rust binary resolves cwd-relative
    paths). Anchor it back to an absolute path here."""
    p = Path(cfg.bulk.output)
    if p.is_absolute():
        return p
    return (TOOLS_DIR / p).resolve()


def _passthrough_config(ns: argparse.Namespace) -> list[str]:
    """Forward --config and --set flags to subcommands so every stage
    sees the same merged config."""
    out: list[str] = []
    if ns.config is not None:
        out += ["--config", str(ns.config)]
    for override in ns.set or []:
        out += ["--set", override]
    return out


def _run_self_tune(
    clustered: Path,
    tuned_out: Path,
    bulk_cfg: SelfTuneBulkConfig,
    release: bool,
) -> int:
    cargo = shutil.which("cargo")
    if cargo is None:
        print("error: cargo not on PATH", file=sys.stderr)
        return 2
    cmd = [
        cargo,
        "run",
        "--manifest-path",
        str(REPO_ROOT / "sphereql-embed" / "Cargo.toml"),
        "--example",
        "corpus_self_tune",
    ]
    if release:
        cmd.append("--release")
    cmd += [
        "--",
        "--corpus",
        str(clustered),
        "--out",
        str(tuned_out),
        "--max-iters",
        str(bulk_cfg.max_iterations),
        "--plateau-eps",
        str(bulk_cfg.plateau_epsilon),
        "--min-quality-to-keep",
        str(bulk_cfg.min_quality_to_keep),
        "--min-per-category",
        str(bulk_cfg.min_concepts_per_category),
    ]
    return _run(cmd, cwd=REPO_ROOT)


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print("$ " + " ".join(_shell_quote(a) for a in cmd))
    return subprocess.call(cmd, cwd=cwd, env=os.environ.copy())


def _summary(
    corpus: Path, clustered: Path, tuned: Path, ns: argparse.Namespace
) -> str:
    bits: list[str] = []
    if not ns.skip_ingest:
        bits.append(f"ingest={corpus.name}")
    if not ns.skip_cluster:
        bits.append(f"clustered={clustered.name}")
    if not ns.skip_self_tune:
        bits.append(f"tuned={tuned.name}")
    return ", ".join(bits) if bits else "no stages ran"


def _section(title: str) -> None:
    bar = "─" * max(8, 60 - len(title))
    print(f"\n┌── {title} {bar}")


def _shell_quote(s: str) -> str:
    if any(c in s for c in " \t\n\"'\\$`!&|;<>()[]{}*?#~="):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    sys.exit(main())
