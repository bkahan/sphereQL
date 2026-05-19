"""Unit tests for the Phase 4 source-resolver pipeline.

Targets `_resolve_category` (the orchestrator's routing entry point)
and verifies:
  - OpenAlex rows route via field-id taxonomy (FIELD_TO_CATEGORY).
  - Non-OpenAlex rows without a category hint fall back to keyword scan.
  - Topics with neither a field nor any keyword match return None.
  - gap_fill rows pass their pre-assigned category through unchanged.
  - The source registry rejects unknown names.

Run from anywhere:
    python3 sphereql-corpus/tools/tests/test_resolver.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOLS))

from generate_extended import _resolve_category  # noqa: E402
from sources import SOURCE_REGISTRY, RawTopic, make_source  # noqa: E402


def _openalex_field(field_id: int) -> dict:
    """Build the OpenAlex-shaped `field` payload that the orchestrator unwraps."""
    return {
        "id": f"https://openalex.org/fields/{field_id}",
        "display_name": f"Field {field_id}",
    }


def test_openalex_field_routes_via_field_id() -> None:
    """field_id=16 → 'chemistry' in FIELD_TO_CATEGORY (bit-identical to pre-Phase-4)."""
    raw = RawTopic(
        external_id="T11999",
        label="Mass spectrometry techniques",
        description="",
        keywords=[],
        raw_category_hint=None,
        source_name="openalex",
        metadata={"field": _openalex_field(16), "subfield_display": None},
    )
    assert _resolve_category(raw) == "chemistry", (
        f"expected 'chemistry', got {_resolve_category(raw)!r}"
    )


def test_openalex_subfield_routes_same_as_topic() -> None:
    """openalex_subfield rows share the field-id routing path."""
    raw = RawTopic(
        external_id="2706",
        label="Organic Chemistry",
        description="",
        keywords=[],
        raw_category_hint=None,
        source_name="openalex_subfield",
        metadata={"field": _openalex_field(16), "subfield_display": "Organic Chemistry"},
    )
    assert _resolve_category(raw) == "chemistry"


def test_wikidata_falls_back_to_keyword_scan() -> None:
    """Wikidata items without a parent-class hint fall through to keyword scan."""
    raw = RawTopic(
        external_id="Q12345",
        label="Quantum chromodynamics",
        description="theory of the strong interaction between quarks",
        keywords=["quantum", "particle", "boson"],
        raw_category_hint=None,
        source_name="wikidata",
        metadata={"parent_qid": "Q11862829"},  # "academic discipline" → no hint
    )
    cat = _resolve_category(raw)
    assert cat == "physics", f"expected 'physics', got {cat!r}"


def test_wikidata_medicine_parent_takes_hint() -> None:
    """Wikidata items under Q1047113 (specialty) → 'medicine' via category_hints()."""
    raw = RawTopic(
        external_id="Q123",
        label="Hepatology",
        description="branch of medicine focused on the liver",
        keywords=[],
        raw_category_hint=None,
        source_name="wikidata",
        metadata={"parent_qid": "Q1047113", "sitelink_count": 30},
    )
    assert _resolve_category(raw) == "medicine"


def test_unknown_openalex_topic_returns_none() -> None:
    """OpenAlex rows with no field_id are skipped (no keyword fallback)."""
    raw = RawTopic(
        external_id="???",
        label="asdfqwer",
        description="",
        keywords=[],
        raw_category_hint=None,
        source_name="openalex",
        metadata={},
    )
    assert _resolve_category(raw) is None


def test_gap_fill_passes_category_through() -> None:
    """gap_fill rows carry the category in raw_category_hint."""
    raw = RawTopic(
        external_id="gapfill:biology:foo",
        label="foo",
        description="",
        keywords=[],
        raw_category_hint="biology",
        source_name="gap_fill",
        metadata={},
    )
    assert _resolve_category(raw) == "biology"


def test_source_registry_is_closed_set() -> None:
    """Only `openalex` and `wikidata` are registered in Phase 4."""
    assert sorted(SOURCE_REGISTRY) == ["openalex", "wikidata"]


def test_unknown_source_rejected_with_keyerror() -> None:
    try:
        make_source("bogus")
    except KeyError as e:
        msg = str(e)
        assert "bogus" in msg
        return
    raise AssertionError("expected KeyError for unknown source")


def test_openalex_confidence_from_works_count() -> None:
    """OpenAlexSource.confidence is log10(1+works)/6 clamped to [0,1]."""
    import math
    src = make_source("openalex")
    raw = RawTopic(
        external_id="T1",
        label="x",
        description="",
        keywords=[],
        raw_category_hint=None,
        source_name="openalex",
        metadata={"works_count": 10000},
    )
    expected = max(0.0, min(1.0, math.log10(1.0 + 10000) / 6.0))
    assert abs(src.confidence(raw) - expected) < 1e-9


def test_wikidata_confidence_from_sitelinks() -> None:
    """WikidataSource.confidence is log10(1+sitelinks)/3 clamped to [0,1]."""
    import math
    src = make_source("wikidata")
    raw = RawTopic(
        external_id="Q1",
        label="x",
        description="",
        keywords=[],
        raw_category_hint=None,
        source_name="wikidata",
        metadata={"sitelink_count": 50},
    )
    expected = max(0.0, min(1.0, math.log10(1.0 + 50) / 3.0))
    assert abs(src.confidence(raw) - expected) < 1e-9


def _run_all() -> int:
    tests = [
        test_openalex_field_routes_via_field_id,
        test_openalex_subfield_routes_same_as_topic,
        test_wikidata_falls_back_to_keyword_scan,
        test_wikidata_medicine_parent_takes_hint,
        test_unknown_openalex_topic_returns_none,
        test_gap_fill_passes_category_through,
        test_source_registry_is_closed_set,
        test_unknown_source_rejected_with_keyerror,
        test_openalex_confidence_from_works_count,
        test_wikidata_confidence_from_sitelinks,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ok  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
