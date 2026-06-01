#!/usr/bin/env python3
"""Generate a 500K synthetic Parquet corpus for scale testing.

Tiles the committed 5K corpus until row count exceeds the target, then
truncates. Relabels each row with a per-copy suffix so the `label`
column stays globally unique (Rust's `labels_are_unique` test would
otherwise fail on a load attempt).

Output: ``/tmp/synthetic_500k.parquet`` (DO NOT commit; checked-in
artifacts must come from `generate_extended.py`).

Usage:
    python3 synthesize_500k.py [--target N] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SRC = Path(__file__).parent.parent / "data" / "extended_corpus.parquet"
DEFAULT_DST = Path("/tmp/synthetic_500k.parquet")
DEFAULT_TARGET = 500_000


def synthesize(src: Path, dst: Path, target: int) -> None:
    table = pq.read_table(src)
    n = table.num_rows
    if n == 0:
        raise SystemExit(f"{src} is empty")
    copies = (target + n - 1) // n

    base_labels = table.column("label").to_pylist()
    new_labels: list[str] = []
    for k in range(copies):
        for lbl in base_labels:
            new_labels.append(f"{lbl}__{k:04d}")
            if len(new_labels) >= target:
                break
        if len(new_labels) >= target:
            break

    cols: dict[str, list] = {}
    for name in table.column_names:
        py = table.column(name).to_pylist()
        if len(py) < target:
            tiled = (py * copies)[:target]
        else:
            tiled = py[:target]
        cols[name] = tiled
    cols["label"] = new_labels[:target]

    new_table = pa.Table.from_pydict(cols, schema=table.schema)
    pq.write_table(
        new_table,
        dst,
        compression="snappy",
        row_group_size=4096,
        use_dictionary=["category", "source"],
    )
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"wrote {dst} ({size_mb:.2f} MB, {target} rows from {n}×{copies})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source Parquet (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination Parquet (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target row count (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()
    synthesize(args.src, args.out, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
