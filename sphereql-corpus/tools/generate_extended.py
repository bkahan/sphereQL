#!/usr/bin/env python3
"""
Generate the extended SphereQL corpus from one or more pluggable sources.

Usage:
    OPENALEX_API_KEY=your_key python3 generate_extended.py \
        [--source openalex] [--source wikidata] [--config PATH] [--set k=v]

The OPENALEX_API_KEY may be either an OpenAlex Premium API key or a
contact email address for the free "polite pool" (auto-detected by '@').

Generation knobs live in corpus_config.toml. Override at the CLI with
--config /path/to/other.toml or per-key --set generation.min_features=3.

Phase 4: this module is a pure orchestrator. All external fetching
lives in `sources/*.py`. To add a source, implement the `Source`
Protocol in `sources/<name>.py` and register it in `sources/__init__.py`.

Output:
    ../data/extended_corpus.json (path configurable via [output].path)
    ../data/extended_corpus.parquet (configurable via [output].parquet_path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

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
from sources import SOURCE_REGISTRY, RawTopic, SourceConfig, make_source


# ─── Source orchestration ───────────────────────────────────────────────

def gather_raw_topics(
    source_names: list[str], config: CorpusConfig
) -> Iterator[RawTopic]:
    """Drain every configured source in declaration order.

    Topics from earlier sources arrive first; the de-dup step in
    `_emit_concept_from_raw` resolves label collisions by skipping the
    second (later) occurrence, with one exception: an openalex_subfield
    label that collides with an earlier openalex topic gets the
    " (subfield)" suffix instead.
    """
    for name in source_names:
        source = make_source(name)
        source_cfg = _per_source_config(name, config)
        print(f"== fetching from {name} ==", file=sys.stderr)
        n = 0
        for raw in source.fetch(source_cfg):
            yield raw
            n += 1
        print(f"== {name}: yielded {n} raw topics ==", file=sys.stderr)


def _per_source_config(name: str, cfg: CorpusConfig) -> SourceConfig:
    """Per-source overrides. API keys come from `<NAME>_API_KEY` env vars."""
    return SourceConfig(
        api_key=os.environ.get(f"{name.upper()}_API_KEY"),
        max_items=None,
        cache_dir=None,
        http=cfg.http,
    )


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


def _resolve_category(raw: RawTopic) -> str | None:
    """Map a `RawTopic` to a SphereQL category, or `None` to skip.

    Routing precedence:
      1. `gap_fill` items carry the category in `raw_category_hint`.
      2. `openalex*` items route via OpenAlex field ID (preserves
         pre-Phase-4 bit-identity).
      3. All other sources prefer the source's `category_hints()`,
         then fall back to keyword scan against `KEYWORD_TO_AXIS` +
         `CATEGORY_PRIMARY_AXES`.
    """
    if raw.source_name == "gap_fill":
        return raw.raw_category_hint

    if raw.source_name.startswith("openalex"):
        return _resolve_openalex_category(raw)

    provider = raw.source_name.split("_")[0]
    if provider in SOURCE_REGISTRY:
        hints = make_source(provider).category_hints(raw)
        if hints:
            return max(hints, key=lambda kv: kv[1])[0]

    return _keyword_to_category(raw)


def _resolve_openalex_category(raw: RawTopic) -> str | None:
    """Bit-identical port of pre-Phase-4 `resolve_category` for OpenAlex rows."""
    field_id = _extract_field_id(raw.metadata.get("field"))
    if field_id is None:
        return None

    parts: list[str] = []
    if raw.label:
        parts.append(raw.label)
    if raw.description:
        parts.append(raw.description)
    sub_display = raw.metadata.get("subfield_display")
    if sub_display:
        parts.append(str(sub_display))
    text = " ".join(parts).lower()

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


def _keyword_to_category(raw: RawTopic) -> str | None:
    """Fallback for sources without a field-ID taxonomy.

    Scans `raw` text for `KEYWORD_TO_AXIS` hits, then picks the category
    whose `CATEGORY_PRIMARY_AXES` capture the most hits. Returns `None`
    when no keyword matches.
    """
    parts = [raw.label, raw.description, *raw.keywords]
    sub = raw.metadata.get("subfield_display")
    if sub:
        parts.append(str(sub))
    text = " ".join(p for p in parts if p).lower()

    hits: Counter[int] = Counter()
    for kw, axis in KEYWORD_TO_AXIS.items():
        if kw in text:
            hits[axis] += 1
    if not hits:
        return None

    best_cat: str | None = None
    best_score = 0
    for cat, primaries in CATEGORY_PRIMARY_AXES.items():
        score = sum(hits.get(a, 0) for a in primaries)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


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
    """Look up base + divisor from config; clamp + apply jitter happens at call site."""
    curve = config.generation.weight_curve
    entry = curve[rank] if rank < len(curve) else curve[-1]
    return entry.base + 0.2 * min(hits / entry.divisor, 1.0)


def generate_features_from_raw(
    raw: RawTopic, category: str, config: CorpusConfig
) -> list[tuple[int, float]]:
    """Produce the sparse feature vector for a single `RawTopic`.

    Bit-identical to the pre-Phase-4 `generate_features(topic, category, config)`
    when the inputs are constructed from the same OpenAlex JSON: text
    is `label + description + keywords + subfield_display` lowercased,
    and `_hash_int` keys off the stripped label.
    """
    # Bit-identical text construction: always seed parts with the
    # stripped label (matching pre-Phase-4 `parts: list[str] = [label]`),
    # then conditionally append description / keywords / subfield_display.
    label = (raw.label or "").strip()
    parts: list[str] = [label]
    if raw.description:
        parts.append(raw.description)
    parts.extend(raw.keywords)
    sub_display = raw.metadata.get("subfield_display")
    if sub_display:
        parts.append(str(sub_display))
    text = " ".join(parts).lower()

    hits: Counter = _scan_keywords(text)

    # Force the top two category-primary axes into the feature set.
    # Without this, high-frequency cross-cutting keywords can push the
    # category anchor out of the top-N ranking, leaving concepts with
    # zero overlap with their category's primaries.
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
        jitter_byte = (label_hash >> (rank * 8)) & 0xFF
        jitter = (jitter_byte / 255.0 - 0.5) * jitter_range
        w = max(weight_floor, min(weight_ceiling, base + jitter))
        w = round(w, round_decimals)
        weighted.append((int(axis), w))

    min_features = config.generation.min_features
    max_features = config.generation.max_features
    if len(weighted) < min_features:
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


# ─── Bridge metric (aggregate stats) ────────────────────────────────────

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


# ─── Derived signals (mirrors sphereql-corpus/src/derived.rs) ────────────
#
# These four helpers compute the Phase 2 quality signal fields. They are
# deliberately a one-to-one port of `src/derived.rs` so the generator and
# the Rust loader/validator agree byte-for-byte. Any change here must be
# mirrored in derived.rs and vice versa; the round-trip test in
# extended.rs::phase2_tests::bridge_degree_matches_authored_features will
# fail otherwise.


def _bridge_degree(features: list[tuple[int, float]]) -> int:
    """Count distinct domain ranges activated by feature axes (0..=30)."""
    hit = [False] * len(DOMAIN_AXIS_RANGES)
    for axis, _ in features:
        for i, r in enumerate(DOMAIN_AXIS_RANGES):
            if axis in r:
                hit[i] = True
                break
    return sum(hit)


def _axis_coherence(
    features: list[tuple[int, float]], primaries: list[int]
) -> float:
    """Fraction of total feature mass placed on `primaries`, in [0, 1]."""
    total = sum(abs(w) for _, w in features)
    if total == 0:
        return 0.0
    primary_set = set(primaries)
    on = sum(abs(w) for a, w in features if a in primary_set)
    return max(0.0, min(1.0, on / total))


def _home_affinity(
    features: list[tuple[int, float]], primaries: list[int]
) -> float:
    """Cosine of feature mass vs. uniform mass over `primaries`, in [0, 1]."""
    if not features or not primaries:
        return 0.0
    a = [0.0] * 128
    b = [0.0] * 128
    for axis, w in features:
        if 0 <= axis < 128:
            a[axis] = w
    for axis in primaries:
        if 0 <= axis < 128:
            b[axis] = 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _composite_quality(ha: float, ac: float, sc: float, bd: int) -> float:
    """0.4*home + 0.3*coherence + 0.2*source + 0.1*min(1, bd/3.0)."""
    bridge_score = min(1.0, bd / 3.0)
    q = 0.4 * ha + 0.3 * ac + 0.2 * sc + 0.1 * bridge_score
    return max(0.0, min(1.0, q))


# ─── Concept emission ───────────────────────────────────────────────────

def _gap_fill_to_raw(label: str, keywords: list[str], category: str) -> RawTopic:
    """Wrap a hand-curated gap-fill row as a `RawTopic` so it flows through
    the same orchestration pipeline as fetched sources."""
    return RawTopic(
        external_id=f"gapfill:{category}:{label}",
        label=label,
        description=" ".join(keywords),
        keywords=list(keywords),
        raw_category_hint=category,
        source_name="gap_fill",
        metadata={},
    )


def _emit_concept_from_raw(
    raw: RawTopic,
    category: str,
    seen_labels: set[str],
    config: CorpusConfig,
) -> dict | None:
    """Build the emitted concept dict from a `RawTopic`. Returns `None` if
    the label is empty or has already been seen (with `openalex_subfield`
    receiving a " (subfield)" disambiguation suffix on first collision).
    """
    label = (raw.label or "").strip()
    if not label:
        return None
    final_label = label
    if final_label in seen_labels and raw.source_name == "openalex_subfield":
        final_label = f"{label} (subfield)"
    if final_label in seen_labels:
        return None
    seen_labels.add(final_label)

    features = generate_features_from_raw(raw, category, config)
    int_features = [(int(a), float(w)) for a, w in features]
    primaries = list(CATEGORY_PRIMARY_AXES.get(category, []))

    bridge_degree = _bridge_degree(int_features)
    axis_coherence = _axis_coherence(int_features, primaries)
    home_affinity = _home_affinity(int_features, primaries)

    if raw.source_name == "gap_fill":
        source_confidence = 0.5
    else:
        provider = raw.source_name.split("_")[0]
        if provider in SOURCE_REGISTRY:
            source_confidence = make_source(provider).confidence(raw)
        else:
            source_confidence = 0.0

    quality = _composite_quality(
        home_affinity, axis_coherence, source_confidence, bridge_degree
    )

    record: dict[str, Any] = {
        "label": final_label,
        "category": category,
        "features": [[a, w] for a, w in int_features],
        "quality": quality,
        "axis_coherence": axis_coherence,
        "bridge_degree": bridge_degree,
        "source_confidence": source_confidence,
        "home_affinity": home_affinity,
        "source": raw.source_name,
    }
    if raw.external_id:
        if raw.source_name.startswith("openalex"):
            record["openalex_id"] = raw.external_id
        elif raw.source_name == "wikidata":
            record["wikidata_id"] = raw.external_id
    return record


# ─── Main ───────────────────────────────────────────────────────────────

def main(
    config: CorpusConfig,
    source_names: list[str],
    output_override: Path | None,
) -> int:
    seen_labels: set[str] = set()
    concepts: list[dict] = []
    skipped = 0

    for raw in gather_raw_topics(source_names, config):
        cat = _resolve_category(raw)
        if cat is None:
            skipped += 1
            continue
        rec = _emit_concept_from_raw(raw, cat, seen_labels, config)
        if rec is not None:
            concepts.append(rec)

    for category, entries in GAP_FILL_CONCEPTS.items():
        for label, kws in entries:
            wrapped = _gap_fill_to_raw(label, list(kws), category)
            rec = _emit_concept_from_raw(wrapped, category, seen_labels, config)
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
    if feat_lens:
        print(
            f"Features/concept: mean={sum(feat_lens) / len(feat_lens):.2f}, "
            f"min={min(feat_lens)}, max={max(feat_lens)}",
            file=sys.stderr,
        )
    if concepts:
        print(
            f"Bridge concepts: {bridge_count} "
            f"({100 * bridge_count / len(concepts):.1f}%)",
            file=sys.stderr,
        )

    output = {
        "version": "1.0.0",
        "generator": "generate_extended.py",
        "source": " + ".join(source_names) + " + gap fill",
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

    # Phase 3: emit Parquet alongside JSON. The Rust loader prefers the
    # Parquet path; the JSON file remains for diffability and as a
    # `json-fallback`-gated emergency path.
    parquet_cfg = Path(config.output.parquet_path)
    if parquet_cfg.is_absolute():
        parquet_path = parquet_cfg
    else:
        parquet_path = (Path(__file__).resolve().parent / parquet_cfg).resolve()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(concepts, parquet_path)
    pq_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {parquet_path} ({pq_size_mb:.2f} MB)", file=sys.stderr)
    return 0


# ─── Parquet emit ────────────────────────────────────────────────────────


def write_parquet(concepts: list[dict], path: Path) -> None:
    """Emit the corpus as a Parquet file matching `parquet_loader.rs`'s schema.

    Columns:
      label (string, non-null), category (string, non-null),
      features (list<struct<axis:u32, weight:f64>>, non-null),
      quality / axis_coherence / source_confidence / home_affinity (f64),
      bridge_degree (u8),
      source (string, nullable), openalex_id (string, nullable).

    Compression SNAPPY, row group size 4096, dictionary encoding on for
    repeated string columns. Any drift between this writer and the Rust
    reader will be caught by the parquet_matches_json round-trip test.
    """
    feature_struct = pa.struct([
        pa.field("axis", pa.uint32(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
    ])
    schema = pa.schema([
        pa.field("label", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("features", pa.list_(feature_struct), nullable=False),
        pa.field("quality", pa.float64(), nullable=False),
        pa.field("axis_coherence", pa.float64(), nullable=False),
        pa.field("bridge_degree", pa.uint8(), nullable=False),
        pa.field("source_confidence", pa.float64(), nullable=False),
        pa.field("home_affinity", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=True),
        pa.field("openalex_id", pa.string(), nullable=True),
    ])

    arrays = {
        "label": pa.array([c["label"] for c in concepts], type=pa.string()),
        "category": pa.array([c["category"] for c in concepts], type=pa.string()),
        "features": pa.array(
            [
                [{"axis": int(a), "weight": float(w)} for a, w in c["features"]]
                for c in concepts
            ],
            type=pa.list_(feature_struct),
        ),
        "quality": pa.array([c["quality"] for c in concepts], type=pa.float64()),
        "axis_coherence": pa.array(
            [c["axis_coherence"] for c in concepts], type=pa.float64()
        ),
        "bridge_degree": pa.array(
            [c["bridge_degree"] for c in concepts], type=pa.uint8()
        ),
        "source_confidence": pa.array(
            [c["source_confidence"] for c in concepts], type=pa.float64()
        ),
        "home_affinity": pa.array(
            [c["home_affinity"] for c in concepts], type=pa.float64()
        ),
        "source": pa.array(
            [c.get("source") for c in concepts], type=pa.string()
        ),
        "openalex_id": pa.array(
            [c.get("openalex_id") for c in concepts], type=pa.string()
        ),
    }
    table = pa.Table.from_pydict(arrays, schema=schema)
    pq.write_table(
        table,
        path,
        compression="snappy",
        row_group_size=4096,
        use_dictionary=["label", "category", "source"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the extended corpus from one or more sources."
    )
    add_config_args(parser)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        choices=sorted(SOURCE_REGISTRY),
        help=(
            "Source to fetch from. Repeatable. "
            f"Registered: {sorted(SOURCE_REGISTRY)}. Default: openalex."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: from config).",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    source_names = args.source or ["openalex"]
    raise SystemExit(main(config, source_names, args.output))
