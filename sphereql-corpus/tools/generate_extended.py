#!/usr/bin/env python3
"""
Generate the extended SphereQL corpus (~8,000+ concepts) from OpenAlex Topics.

Usage:
    OPENALEX_API_KEY=your_key python3 generate_extended.py

The OPENALEX_API_KEY may be either an OpenAlex Premium API key or a
contact email address for the free "polite pool" (auto-detected by '@').

Output:
    ../data/extended_corpus.json
"""

from __future__ import annotations

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

NUM_AXES = 128
MAX_FEATURES = 8
MIN_FEATURES = 4


# ─── HTTP helpers ────────────────────────────────────────────────────────

def _auth_params(api_key: str) -> dict[str, str]:
    """Auto-detect whether the key is an email (polite pool) or a Premium key."""
    if "@" in api_key:
        return {"mailto": api_key}
    return {"api_key": api_key}


def _get_with_retry(url: str, params: dict[str, Any], retries: int = 3) -> dict:
    """GET with exponential backoff: 1s, 2s, 4s on transient errors."""
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
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
            delay *= 2
    raise RuntimeError("unreachable")


def _paginate(endpoint: str, select: str, api_key: str) -> list[dict]:
    """Cursor-paginate an OpenAlex endpoint until exhaustion."""
    out: list[dict] = []
    cursor = "*"
    page = 0
    while cursor:
        params = {
            "per_page": 100,
            "cursor": cursor,
            "select": select,
            **_auth_params(api_key),
        }
        data = _get_with_retry(f"{OPENALEX_BASE}/{endpoint}", params)
        results = data.get("results", [])
        out.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        page += 1
        print(
            f"  {endpoint} page {page}: +{len(results)} (total {len(out)})",
            file=sys.stderr,
        )
        time.sleep(0.1)  # polite rate limiting
        if not results:
            break
    return out


def fetch_all_topics(api_key: str) -> list[dict]:
    """Fetch every OpenAlex Topic record (~4,500 as of 2026)."""
    print("Fetching topics from OpenAlex…", file=sys.stderr)
    return _paginate("topics", TOPIC_SELECT, api_key)


def fetch_all_subfields(api_key: str) -> list[dict]:
    """Fetch every OpenAlex Subfield record (~254 as of 2026)."""
    print("Fetching subfields from OpenAlex…", file=sys.stderr)
    return _paginate("subfields", SUBFIELD_SELECT, api_key)


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


def generate_features(topic: dict, category: str) -> list[tuple[int, float]]:
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

    # Seed the top two category-primary axes with weak hits if absent —
    # ensures every concept anchors to its category.
    primaries = CATEGORY_PRIMARY_AXES.get(category, [])
    for axis in primaries[:2]:
        if axis not in hits:
            hits[axis] = 0.5  # type: ignore[assignment]

    label_hash = _hash_int(label)
    n_features = 5 + (label_hash % 3)  # 5, 6, or 7

    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = ranked[:n_features]

    weighted: list[tuple[int, float]] = []
    for rank, (axis, h) in enumerate(chosen):
        if rank == 0:
            base = 0.8 + 0.2 * min(h / 4.0, 1.0)
        elif rank == 1:
            base = 0.6 + 0.2 * min(h / 3.0, 1.0)
        elif rank in (2, 3):
            base = 0.4 + 0.2 * min(h / 2.0, 1.0)
        else:
            base = 0.3 + 0.2 * min(h / 2.0, 1.0)

        # Deterministic micro-variation in [-0.05, +0.05]
        jitter_byte = (label_hash >> (rank * 8)) & 0xFF
        jitter = (jitter_byte / 255.0 - 0.5) * 0.1
        w = max(0.2, min(1.0, base + jitter))
        w = round(w, 1)
        weighted.append((int(axis), w))

    if len(weighted) < MIN_FEATURES:
        used = {a for a, _ in weighted}
        for axis in primaries:
            if axis not in used:
                weighted.append((int(axis), 0.3))
                used.add(axis)
                if len(weighted) >= MIN_FEATURES:
                    break

    weighted.sort(key=lambda kv: kv[0])
    if len(weighted) > MAX_FEATURES:
        weighted = weighted[:MAX_FEATURES]
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

    # generate_features uses the original label hash for determinism
    features = generate_features(topic, category)
    record: dict[str, Any] = {
        "label": final_label,
        "category": category,
        "features": [[int(a), float(w)] for a, w in features],
        "source": source,
    }
    if oa_id := _openalex_id_tail(topic.get("id")):
        record["openalex_id"] = oa_id
    return record


def main() -> int:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENALEX_API_KEY env var is required. "
            "Set it to a Premium API key or your contact email "
            "(see https://openalex.org/settings/api).",
            file=sys.stderr,
        )
        return 1

    topics = fetch_all_topics(api_key)
    subfields = fetch_all_subfields(api_key)

    seen_labels: set[str] = set()
    concepts: list[dict] = []
    skipped = 0

    for t in topics:
        cat = resolve_category(t)
        if cat is None:
            skipped += 1
            continue
        rec = _emit_concept(t, cat, "openalex", seen_labels)
        if rec is not None:
            concepts.append(rec)

    for sf in subfields:
        wrapped = _topic_from_subfield(sf)
        cat = resolve_category(wrapped)
        if cat is None:
            skipped += 1
            continue
        rec = _emit_concept(wrapped, cat, "openalex_subfield", seen_labels)
        if rec is not None:
            concepts.append(rec)

    for category, entries in GAP_FILL_CONCEPTS.items():
        for label, kws in entries:
            wrapped = _topic_from_gap_fill(label, list(kws), category)
            rec = _emit_concept(wrapped, category, "gap_fill", seen_labels)
            if rec is not None:
                concepts.append(rec)

    # Stats
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

    out_path = Path(__file__).resolve().parent.parent / "data" / "extended_corpus.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, separators=(",", ":"), ensure_ascii=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.2f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
