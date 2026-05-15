#!/usr/bin/env python3
"""
Generate the extended SphereQL corpus (~8,000+ concepts) from OpenAlex Topics.

Usage:
    OPENALEX_API_KEY=your_key python3 generate_extended.py [--config PATH] [--set k=v]

The OPENALEX_API_KEY may be either an OpenAlex Premium API key or a
contact email address for the free "polite pool" (auto-detected by '@').

Generation knobs live in corpus_config.toml. Override at the CLI with
--config /path/to/other.toml or per-key --set generation.min_features=3.

Output:
    ../data/extended_corpus.json (path configurable via [output].path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from corpus_config import CorpusConfig, add_config_args, load_config
from gap_fill_data import GAP_FILL_CONCEPTS
from mappings import (
    CATEGORY_PRIMARY_AXES,
    CONTENT_OVERRIDES,
    DOMAIN_AXIS_RANGES,
    FIELD_MULTI_MAP,
    FIELD_TO_CATEGORY,
    KEYWORD_TO_AXIS,
)

OPENALEX_BASE = "https://api.openalex.org"
TOPIC_SELECT = "id,display_name,description,keywords,subfield,field,domain,works_count"
SUBFIELD_SELECT = "id,display_name,field,domain,works_count"


# ─── HTTP helpers ────────────────────────────────────────────────────────

def _auth_params(api_key: str) -> dict[str, str]:
    """Auto-detect whether the key is an email (polite pool) or a Premium key."""
    if "@" in api_key:
        return {"mailto": api_key}
    return {"api_key": api_key}


def _get_with_retry(url: str, params: dict[str, Any], config: CorpusConfig) -> dict:
    """GET with exponential backoff sourced from config.http."""
    delay = config.http.backoff_base
    retries = config.http.retries
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.http.timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise
            print(
                f"  retry {attempt + 1}/{retries} after {delay:.1f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= config.http.backoff_multiplier
    raise RuntimeError("unreachable")


def _paginate(endpoint: str, select: str, api_key: str, config: CorpusConfig) -> list[dict]:
    """Cursor-paginate an OpenAlex endpoint until exhaustion."""
    out: list[dict] = []
    cursor = "*"
    page = 0
    while cursor:
        params = {
            "per_page": config.http.per_page,
            "cursor": cursor,
            "select": select,
            **_auth_params(api_key),
        }
        data = _get_with_retry(f"{OPENALEX_BASE}/{endpoint}", params, config)
        results = data.get("results", [])
        out.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        page += 1
        print(
            f"  {endpoint} page {page}: +{len(results)} (total {len(out)})",
            file=sys.stderr,
        )
        time.sleep(config.http.inter_page_sleep_seconds)
        if not results:
            break
    return out


def fetch_all_topics(api_key: str, config: CorpusConfig) -> list[dict]:
    """Fetch every OpenAlex Topic record (~4,500 as of 2026)."""
    print("Fetching topics from OpenAlex…", file=sys.stderr)
    return _paginate("topics", TOPIC_SELECT, api_key, config)


def fetch_all_subfields(api_key: str, config: CorpusConfig) -> list[dict]:
    """Fetch every OpenAlex Subfield record (~254 as of 2026)."""
    print("Fetching subfields from OpenAlex…", file=sys.stderr)
    return _paginate("subfields", SUBFIELD_SELECT, api_key, config)


# ─── Category resolution ────────────────────────────────────────────────

def _extract_field_id(field_obj: Any) -> int | None:
    """Pull the integer field ID from either a dict {id, display_name} or url string."""
    if isinstance(field_obj, dict):
        raw = field_obj.get("id")
    else:
        raw = field_obj
    if not isinstance(raw, str):
        return None
    tail = raw.rstrip("/").rsplit("/", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _topic_text(topic: dict) -> str:
    """Lowercase concat of display_name + description + subfield_name."""
    parts: list[str] = []
    if dn := topic.get("display_name"):
        parts.append(str(dn))
    if desc := topic.get("description"):
        parts.append(str(desc))
    sub = topic.get("subfield")
    if isinstance(sub, dict) and (sdn := sub.get("display_name")):
        parts.append(str(sdn))
    return " ".join(parts).lower()


def _apply_content_override(category: str, text: str) -> str:
    """Override the field-resolved category when topic text strongly indicates
    a different domain. Targets OpenAlex misfilings (e.g., Geochemistry under
    Computer Science, Superconducting Materials under Medicine)."""
    for kw, target, exclude in CONTENT_OVERRIDES:
        if kw in text and category != target:
            if exclude and exclude in text:
                continue
            return target
    return category


def resolve_category(topic: dict) -> str | None:
    """Map an OpenAlex topic/subfield to a SphereQL category, or None to skip."""
    field_id = _extract_field_id(topic.get("field"))
    if field_id is None:
        return None

    text = _topic_text(topic)

    if field_id in FIELD_TO_CATEGORY:
        return _apply_content_override(FIELD_TO_CATEGORY[field_id], text)

    if field_id in FIELD_MULTI_MAP:
        rules = FIELD_MULTI_MAP[field_id]
        for kw, cat in rules.items():
            if kw == "default":
                continue
            if kw in text:
                return _apply_content_override(cat, text)
        default_cat = rules.get("default")
        if default_cat is None:
            return None
        return _apply_content_override(default_cat, text)

    return None


# ─── Feature generation ─────────────────────────────────────────────────

def _hash_int(label: str) -> int:
    """Deterministic 64-bit-ish hash for label-conditioned variation."""
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _scan_keywords(text: str) -> Counter:
    """Count keyword→axis hits across the combined text."""
    hits: Counter = Counter()
    for kw, axis in KEYWORD_TO_AXIS.items():
        if kw in text:
            hits[axis] += 1
    return hits


def _weight_for_rank(rank: int, hits: int, config: CorpusConfig) -> float:
    """Look up base + divisor from config; clamp + apply jitter happens at call site.

    The 0.2 multiplier is intentionally retained as a literal — it is the
    fixed "weight delta" in the original formula and is not in scope for
    this phase.
    """
    # TODO(phase-5): make 0.2 configurable if quality metric needs it
    curve = config.generation.weight_curve
    entry = curve[rank] if rank < len(curve) else curve[-1]
    return entry.base + 0.2 * min(hits / entry.divisor, 1.0)


def generate_features(topic: dict, category: str, config: CorpusConfig) -> list[tuple[int, float]]:
    """
    Produce a sparse feature vector matching the hand-crafted corpus
    distribution: 4–8 features, weights in [0.2, 1.0].
    """
    label = str(topic.get("display_name") or "").strip()

    parts: list[str] = [label]
    if desc := topic.get("description"):
        parts.append(str(desc))
    keywords = topic.get("keywords") or []
    if isinstance(keywords, list):
        parts.extend(str(k) for k in keywords)
    sub = topic.get("subfield")
    if isinstance(sub, dict) and (sdn := sub.get("display_name")):
        parts.append(str(sdn))
    text = " ".join(parts).lower()

    hits: Counter = _scan_keywords(text)

    # Force the top two category-primary axes into the feature set.
    # Without this, high-frequency cross-cutting keywords (e.g., "analysis",
    # "system", "data") can push the category anchor out of the top-N
    # ranking, leaving concepts with zero overlap with their category's
    # primaries — i.e., misrouted in the spatial sense.
    primaries = CATEGORY_PRIMARY_AXES.get(category, [])
    primary_seed_hits = config.generation.primary_seed_hits
    for axis in primaries[:2]:
        hits[axis] = max(hits.get(axis, 0), primary_seed_hits)  # type: ignore[assignment]

    label_hash = _hash_int(label)
    n_min = config.generation.n_features_min
    n_max = config.generation.n_features_max
    span = n_max - n_min + 1
    n_features = n_min + (label_hash % span)

    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = ranked[:n_features]

    weight_floor = config.generation.weight_floor
    weight_ceiling = config.generation.weight_ceiling
    jitter_range = config.generation.weight_jitter_range
    round_decimals = config.generation.weight_round_decimals

    weighted: list[tuple[int, float]] = []
    for rank, (axis, h) in enumerate(chosen):
        base = _weight_for_rank(rank, h, config)

        # Deterministic micro-variation centered on zero with width jitter_range.
        jitter_byte = (label_hash >> (rank * 8)) & 0xFF
        jitter = (jitter_byte / 255.0 - 0.5) * jitter_range
        w = max(weight_floor, min(weight_ceiling, base + jitter))
        w = round(w, round_decimals)
        weighted.append((int(axis), w))

    min_features = config.generation.min_features
    max_features = config.generation.max_features
    if len(weighted) < min_features:
        # Fill with a weight one tenth above the floor, rounded the same
        # way main-path weights are. Matches pre-Phase-1 literal `0.3`
        # exactly when floor=0.2, round_decimals=1.
        fill_weight = round(weight_floor + 0.1, round_decimals)
        used = {a for a, _ in weighted}
        for axis in primaries:
            if axis not in used:
                weighted.append((int(axis), fill_weight))
                used.add(axis)
                if len(weighted) >= min_features:
                    break

    weighted.sort(key=lambda kv: kv[0])
    if len(weighted) > max_features:
        weighted = weighted[:max_features]
    return weighted


# ─── Bridge metric ──────────────────────────────────────────────────────

def count_bridges(concepts: list[dict]) -> int:
    """How many concepts activate axes from 2+ domain groups (ranges 0..107)."""
    count = 0
    for c in concepts:
        groups = set()
        for ax, _ in c["features"]:
            for i, rng in enumerate(DOMAIN_AXIS_RANGES):
                if ax in rng:
                    groups.add(i)
                    break
        if len(groups) >= 2:
            count += 1
    return count


# ─── Pseudo-topic helpers for subfields and gap-fill ────────────────────

def _topic_from_subfield(sf: dict) -> dict:
    """Wrap a subfield record so it flows through the topic pipeline."""
    return {
        "id": sf.get("id"),
        "display_name": sf.get("display_name"),
        "description": "",
        "keywords": [],
        "field": sf.get("field"),
        "subfield": {"display_name": sf.get("display_name")},
    }


def _topic_from_gap_fill(label: str, keywords: list[str], category: str) -> dict:
    return {
        "id": f"gapfill:{category}:{label}",
        "display_name": label,
        "description": " ".join(keywords),
        "keywords": list(keywords),
        "field": None,
        "subfield": None,
        "_gap_fill_category": category,
    }


# ─── Main ───────────────────────────────────────────────────────────────

def _openalex_id_tail(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return raw.rstrip("/").rsplit("/", 1)[-1] or None


def _emit_concept(
    topic: dict,
    category: str,
    source: str,
    seen_labels: set[str],
    config: CorpusConfig,
) -> dict | None:
    label = str(topic.get("display_name") or "").strip()
    if not label:
        return None
    final_label = label
    if final_label in seen_labels and source == "openalex_subfield":
        final_label = f"{label} (subfield)"
    if final_label in seen_labels:
        return None
    seen_labels.add(final_label)

    features = generate_features(topic, category, config)
    record: dict[str, Any] = {
        "label": final_label,
        "category": category,
        "features": [[int(a), float(w)] for a, w in features],
        "source": source,
    }
    if oa_id := _openalex_id_tail(topic.get("id")):
        record["openalex_id"] = oa_id
    return record


def main(config: CorpusConfig, output_override: Path | None) -> int:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENALEX_API_KEY env var is required. "
            "Set it to a Premium API key or your contact email "
            "(see https://openalex.org/settings/api).",
            file=sys.stderr,
        )
        return 1

    topics = fetch_all_topics(api_key, config)
    subfields = fetch_all_subfields(api_key, config)

    seen_labels: set[str] = set()
    concepts: list[dict] = []
    skipped = 0

    for t in topics:
        cat = resolve_category(t)
        if cat is None:
            skipped += 1
            continue
        rec = _emit_concept(t, cat, "openalex", seen_labels, config)
        if rec is not None:
            concepts.append(rec)

    for sf in subfields:
        wrapped = _topic_from_subfield(sf)
        cat = resolve_category(wrapped)
        if cat is None:
            skipped += 1
            continue
        rec = _emit_concept(wrapped, cat, "openalex_subfield", seen_labels, config)
        if rec is not None:
            concepts.append(rec)

    for category, entries in GAP_FILL_CONCEPTS.items():
        for label, kws in entries:
            wrapped = _topic_from_gap_fill(label, list(kws), category)
            rec = _emit_concept(wrapped, category, "gap_fill", seen_labels, config)
            if rec is not None:
                concepts.append(rec)

    by_cat: Counter = Counter(c["category"] for c in concepts)
    feat_lens = [len(c["features"]) for c in concepts]
    bridge_count = count_bridges(concepts)

    print("\nPer-category counts:", file=sys.stderr)
    for cat in sorted(by_cat):
        print(f"  {cat:24s} {by_cat[cat]:>5d}", file=sys.stderr)
    print(
        f"\nTotal: {len(concepts)} concepts ({skipped} skipped) "
        f"across {len(by_cat)} categories",
        file=sys.stderr,
    )
    print(
        f"Features/concept: mean={sum(feat_lens) / len(feat_lens):.2f}, "
        f"min={min(feat_lens)}, max={max(feat_lens)}",
        file=sys.stderr,
    )
    print(
        f"Bridge concepts: {bridge_count} "
        f"({100 * bridge_count / len(concepts):.1f}%)",
        file=sys.stderr,
    )

    output = {
        "version": "1.0.0",
        "generator": "generate_extended.py",
        "source": "OpenAlex Topics API + gap fill",
        "generated_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "stats": {
            "total_concepts": len(concepts),
            "categories": len(by_cat),
            "min_per_category": min(by_cat.values()) if by_cat else 0,
            "max_per_category": max(by_cat.values()) if by_cat else 0,
            "mean_features_per_concept": (
                round(sum(feat_lens) / len(feat_lens), 2) if feat_lens else 0
            ),
            "bridge_concept_ratio": (
                round(bridge_count / len(concepts), 3) if concepts else 0
            ),
        },
        "concepts": concepts,
    }

    out_path = (
        output_override.resolve()
        if output_override is not None
        else (Path(__file__).resolve().parent / config.output.path).resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, separators=(",", ":"), ensure_ascii=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.2f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the extended corpus from OpenAlex + gap-fill data."
    )
    add_config_args(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: from config).",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    raise SystemExit(main(config, args.output))
