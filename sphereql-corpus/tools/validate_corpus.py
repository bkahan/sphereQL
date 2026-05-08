#!/usr/bin/env python3
"""Validate extended_corpus.json against SphereQL corpus invariants.

Usage:
    python3 validate_corpus.py [path/to/extended_corpus.json]

Exit code 0 = all checks pass, 1 = any check fails.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_CATEGORIES = {
    "physics", "mathematics", "biology", "chemistry", "medicine", "neuroscience",
    "computer_science", "data_science", "engineering", "nanotechnology", "astronomy",
    "earth_science", "environmental_science", "psychology", "philosophy", "religion",
    "linguistics", "literature", "history", "sociology", "anthropology",
    "political_science", "law", "economics", "education", "visual_arts", "music",
    "film_studies", "performing_arts", "culinary_arts", "architecture",
}

# Mirrors tools/mappings.py::DOMAIN_AXIS_RANGES. 107..128 are cross-cutting.
DOMAIN_AXIS_RANGES: list[range] = [
    range(0, 7), range(7, 12), range(12, 16), range(16, 19),
    range(19, 23), range(23, 26), range(26, 30), range(30, 34),
    range(34, 38), range(38, 41), range(41, 45), range(45, 49),
    range(49, 52), range(52, 55), range(55, 59), range(59, 63),
    range(63, 68), range(68, 71), range(71, 73), range(73, 76),
    range(76, 79), range(79, 82), range(82, 85), range(85, 89),
    range(89, 92), range(92, 96), range(96, 100), range(100, 102),
    range(102, 104), range(104, 107),
]

MIN_TOTAL_CONCEPTS = 5000
MIN_PER_CATEGORY = 50
WARN_THIN_CATEGORY = 80
MIN_FEATURES = 4
MAX_FEATURES = 8
MIN_MEAN_FEATURES = 4.5
MAX_MEAN_FEATURES = 7.0
WEIGHT_MIN = 0.2
WEIGHT_MAX = 1.0
NUM_AXES = 128
MIN_BRIDGE_RATIO = 0.75


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


def _resolve_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).resolve()
    return (Path(__file__).resolve().parent.parent / "data" / "extended_corpus.json").resolve()


def _bar(count: int, max_count: int, width: int = 30) -> str:
    if max_count <= 0:
        return ""
    fill = int(round(width * count / max_count))
    return "█" * fill + "·" * (width - fill)


def main(argv: list[str]) -> int:
    path = _resolve_path(argv)
    print(f"Validating {path}\n")

    rep = Report()

    # Check 1: parse
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        rep.check(False, "JSON parses", str(exc))
        return 1
    rep.check(True, "JSON parses")

    concepts = data.get("concepts")
    if not isinstance(concepts, list):
        rep.check(False, "top-level 'concepts' is a list")
        return 1
    rep.check(True, "top-level 'concepts' is a list")

    # Check 2: total size
    rep.check(
        len(concepts) >= MIN_TOTAL_CONCEPTS,
        f"total concepts ≥ {MIN_TOTAL_CONCEPTS}",
        f"got {len(concepts)}",
    )

    # Check 3: categories
    cats: Counter = Counter(c.get("category", "") for c in concepts)
    cat_set = set(cats.keys())
    missing = EXPECTED_CATEGORIES - cat_set
    extra = cat_set - EXPECTED_CATEGORIES
    rep.check(
        not missing and not extra,
        "all 31 expected categories present",
        f"missing={sorted(missing)} extra={sorted(extra)}" if (missing or extra) else "",
    )

    # Check 4: per-category minimums
    under = [(c, n) for c, n in cats.items() if n < MIN_PER_CATEGORY]
    rep.check(
        not under,
        f"every category has ≥{MIN_PER_CATEGORY} concepts",
        f"under-floor: {under}" if under else "",
    )
    for cat, n in cats.items():
        if MIN_PER_CATEGORY <= n < WARN_THIN_CATEGORY:
            rep.warn(f"thin category '{cat}' has {n} concepts (<{WARN_THIN_CATEGORY})")

    # Checks 5–9: feature shape, weights, axis bounds, dup axes
    feat_lens: list[int] = []
    bad_len: list[str] = []
    bad_weight: list[tuple[str, float]] = []
    bad_axis: list[tuple[str, int]] = []
    dup_axes: list[str] = []

    label_counter: Counter = Counter()
    used_axes: set[int] = set()

    for c in concepts:
        label = c.get("label", "?")
        label_counter[label] += 1
        feats = c.get("features") or []
        feat_lens.append(len(feats))
        if not (MIN_FEATURES <= len(feats) <= MAX_FEATURES):
            bad_len.append(label)

        seen_axes: set[int] = set()
        for pair in feats:
            if not (isinstance(pair, list) and len(pair) == 2):
                bad_axis.append((label, -1))
                continue
            axis_raw, weight = pair
            try:
                axis = int(axis_raw)
            except (TypeError, ValueError):
                bad_axis.append((label, -1))
                continue
            if not (0 <= axis < NUM_AXES):
                bad_axis.append((label, axis))
                continue
            if axis in seen_axes:
                dup_axes.append(label)
            seen_axes.add(axis)
            used_axes.add(axis)
            try:
                w = float(weight)
            except (TypeError, ValueError):
                bad_weight.append((label, float("nan")))
                continue
            if not (WEIGHT_MIN - 1e-9 <= w <= WEIGHT_MAX + 1e-9):
                bad_weight.append((label, w))

    rep.check(
        not bad_len,
        f"every concept has {MIN_FEATURES}–{MAX_FEATURES} features",
        f"{len(bad_len)} violations" if bad_len else "",
    )
    mean_feats = sum(feat_lens) / len(feat_lens) if feat_lens else 0.0
    rep.check(
        MIN_MEAN_FEATURES <= mean_feats <= MAX_MEAN_FEATURES,
        f"mean features/concept in [{MIN_MEAN_FEATURES}, {MAX_MEAN_FEATURES}]",
        f"got {mean_feats:.2f}",
    )
    rep.check(
        not bad_weight,
        f"all weights in [{WEIGHT_MIN}, {WEIGHT_MAX}]",
        f"{len(bad_weight)} violations" if bad_weight else "",
    )
    rep.check(
        not bad_axis,
        f"all axis indices in [0, {NUM_AXES - 1}]",
        f"{len(bad_axis)} violations" if bad_axis else "",
    )
    rep.check(
        not dup_axes,
        "no duplicate axis indices within any concept",
        f"{len(dup_axes)} violations" if dup_axes else "",
    )

    # Check 10: unique labels
    dups = [(lbl, n) for lbl, n in label_counter.items() if n > 1]
    rep.check(
        not dups,
        "no duplicate labels globally",
        f"{len(dups)} duplicates" if dups else "",
    )

    # Check 11: all 128 axes used
    missing_axes = [a for a in range(NUM_AXES) if a not in used_axes]
    rep.check(
        not missing_axes,
        f"all {NUM_AXES} axes used at least once",
        f"unused: {missing_axes}" if missing_axes else "",
    )

    # Check 12: bridge ratio
    bridge_count = 0
    for c in concepts:
        groups: set[int] = set()
        for pair in c.get("features") or []:
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            try:
                axis = int(pair[0])
            except (TypeError, ValueError):
                continue
            for i, rng in enumerate(DOMAIN_AXIS_RANGES):
                if axis in rng:
                    groups.add(i)
                    break
        if len(groups) >= 2:
            bridge_count += 1
    bridge_ratio = bridge_count / len(concepts) if concepts else 0.0
    rep.check(
        bridge_ratio >= MIN_BRIDGE_RATIO,
        f"bridge ratio ≥ {MIN_BRIDGE_RATIO:.0%}",
        f"got {bridge_ratio:.1%} ({bridge_count}/{len(concepts)})",
    )

    # Per-category bar chart
    print("\nPer-category counts:")
    max_count = max(cats.values()) if cats else 0
    for cat in sorted(cats):
        n = cats[cat]
        print(f"  {cat:24s} {n:>5d}  {_bar(n, max_count)}")

    # Summary
    print(f"\nSummary: {rep.errors} error(s), {rep.warnings} warning(s)")
    if rep.errors:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
