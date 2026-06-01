#!/usr/bin/env python3
"""Validate extended_corpus.json against SphereQL corpus invariants.

Usage:
    python3 validate_corpus.py [path/to/extended_corpus.json] [--config PATH] [--set k=v]

Exit code 0 = all checks pass, 1 = any check fails.

Thresholds are loaded from tools/corpus_config.toml (see [validation] +
[generation] sections). Override at the CLI with --config or --set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from corpus_config import CorpusConfig, add_config_args, load_config

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


def _bar(count: int, max_count: int, width: int = 30) -> str:
    if max_count <= 0:
        return ""
    fill = int(round(width * count / max_count))
    return "█" * fill + "·" * (width - fill)


def main(corpus_path: Path, config: CorpusConfig) -> int:
    print(f"Validating {corpus_path}\n")

    rep = Report()

    min_total_concepts = config.validation.min_total_concepts
    min_per_category = config.validation.min_per_category
    warn_thin_category = config.validation.warn_thin_category
    min_features = config.generation.min_features
    max_features = config.generation.max_features
    min_mean_features = config.validation.min_mean_features
    max_mean_features = config.validation.max_mean_features
    weight_min = config.generation.weight_floor
    weight_max = config.generation.weight_ceiling
    num_axes = config.generation.num_axes
    min_bridge_ratio = config.validation.min_bridge_ratio
    max_misrouted_ratio = config.validation.max_misrouted_ratio

    # Check 1: parse
    try:
        with corpus_path.open("r", encoding="utf-8") as fh:
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
        len(concepts) >= min_total_concepts,
        f"total concepts ≥ {min_total_concepts}",
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
    under = [(c, n) for c, n in cats.items() if n < min_per_category]
    rep.check(
        not under,
        f"every category has ≥{min_per_category} concepts",
        f"under-floor: {under}" if under else "",
    )
    for cat, n in cats.items():
        if min_per_category <= n < warn_thin_category:
            rep.warn(f"thin category '{cat}' has {n} concepts (<{warn_thin_category})")

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
        if not (min_features <= len(feats) <= max_features):
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
            if not (0 <= axis < num_axes):
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
            if not (weight_min - 1e-9 <= w <= weight_max + 1e-9):
                bad_weight.append((label, w))

    rep.check(
        not bad_len,
        f"every concept has {min_features}–{max_features} features",
        f"{len(bad_len)} violations" if bad_len else "",
    )
    mean_feats = sum(feat_lens) / len(feat_lens) if feat_lens else 0.0
    rep.check(
        min_mean_features <= mean_feats <= max_mean_features,
        f"mean features/concept in [{min_mean_features}, {max_mean_features}]",
        f"got {mean_feats:.2f}",
    )
    rep.check(
        not bad_weight,
        f"all weights in [{weight_min}, {weight_max}]",
        f"{len(bad_weight)} violations" if bad_weight else "",
    )
    rep.check(
        not bad_axis,
        f"all axis indices in [0, {num_axes - 1}]",
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

    # Check 11: all axes used
    missing_axes = [a for a in range(num_axes) if a not in used_axes]
    rep.check(
        not missing_axes,
        f"all {num_axes} axes used at least once",
        f"unused: {missing_axes}" if missing_axes else "",
    )

    # Check 11.5: category-axis alignment (misroute detection)
    # A concept is "misrouted" if NONE of its feature axes match the primary
    # axes for its assigned category. This catches OpenAlex topics dumped
    # into wrong SphereQL categories by FIELD_MULTI_MAP defaults.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mappings import CATEGORY_PRIMARY_AXES  # type: ignore
    except ImportError:
        CATEGORY_PRIMARY_AXES = {}

    misrouted: list[tuple[str, str, list[int]]] = []
    for c in concepts:
        cat = c.get("category", "")
        primaries = set(CATEGORY_PRIMARY_AXES.get(cat, []))
        if not primaries:
            continue
        feat_axes: set[int] = set()
        for pair in c.get("features") or []:
            try:
                feat_axes.add(int(pair[0]))
            except (TypeError, ValueError, IndexError):
                pass
        if not (feat_axes & primaries):
            misrouted.append((cat, str(c.get("label", "?")), sorted(feat_axes)))

    misrouted_ratio = len(misrouted) / len(concepts) if concepts else 0.0
    ok = rep.check(
        misrouted_ratio <= max_misrouted_ratio,
        f"misrouted concepts ≤ {max_misrouted_ratio:.0%}",
        f"{len(misrouted)}/{len(concepts)} ({misrouted_ratio:.1%}) have zero overlap "
        f"with their category's primary axes",
    )
    if not ok and misrouted:
        by_cat: Counter = Counter(m[0] for m in misrouted)
        print(f"    Misroutes by category (top 5):")
        for c, n in by_cat.most_common(5):
            print(f"      {c}: {n}")
        print(f"    Sample (3):")
        for cat, lbl, axes in misrouted[:3]:
            print(f"      [{cat}] {lbl[:60]} axes={axes}")

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
        bridge_ratio >= min_bridge_ratio,
        f"bridge ratio ≥ {min_bridge_ratio:.0%}",
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
    parser = argparse.ArgumentParser(
        description="Validate the extended corpus JSON against configured invariants."
    )
    add_config_args(parser)
    parser.add_argument(
        "corpus_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to corpus JSON (default: from config.output.path).",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    corpus_path = (
        args.corpus_path.resolve()
        if args.corpus_path is not None
        else (Path(__file__).resolve().parent / config.output.path).resolve()
    )
    raise SystemExit(main(corpus_path, config))
