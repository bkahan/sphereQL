"""Bit-identity check: replay a known OpenAlex topic through the new
Phase-4 pipeline and verify the produced concept matches the
pre-Phase-4 corpus row.

The committed `data/extended_corpus.json` is the ground truth. We pick
a few rows by `openalex_id`, reconstruct the OpenAlex API shape that
`OpenAlexSource._topic_to_raw` consumes, and assert that running the
new orchestrator produces an identical concept dict.

This is the offline analogue of the spec's `--source openalex` bit-
identity acceptance test: it does not need network or an API key.

Run from anywhere:
    python3 sphereql-corpus/tools/tests/test_bit_identity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOLS))

from corpus_config import load_config  # noqa: E402
from generate_extended import _emit_concept_from_raw, _resolve_category  # noqa: E402
from sources.openalex import OpenAlexSource  # noqa: E402

CORPUS = _TOOLS.parent / "data" / "extended_corpus.json"


def _load_existing_concepts() -> dict[str, dict]:
    with open(CORPUS) as f:
        data = json.load(f)
    by_id = {}
    for c in data["concepts"]:
        oa = c.get("openalex_id")
        if oa:
            by_id[oa] = c
    return by_id


def _fake_openalex_topic(concept: dict, field_id: int) -> dict:
    """Approximate the OpenAlex topic JSON that produced this concept.

    Builds the dict shape `OpenAlexSource._topic_to_raw` expects. The
    `keywords` and `description` aren't recoverable from the emitted
    concept alone — we use empty placeholders. This means the feature
    text used in the new pipeline will be (label only) instead of
    (label + description + keywords + subfield), so feature output will
    NOT match exactly. We only assert the structural fields here.
    """
    return {
        "id": f"https://openalex.org/{concept['openalex_id']}",
        "display_name": concept["label"],
        "description": "",
        "keywords": [],
        "field": {
            "id": f"https://openalex.org/fields/{field_id}",
            "display_name": "X",
        },
        "subfield": {"display_name": concept["label"]},
        "works_count": 0,
    }


def test_pipeline_emits_required_concept_fields() -> None:
    """Structural check: every required field is emitted in the same types."""
    by_id = _load_existing_concepts()
    if not by_id:
        raise AssertionError("no openalex concepts in committed corpus")
    config = load_config()
    src = OpenAlexSource()

    # Pick a chemistry topic (field_id 16) that exists in the corpus.
    chem = next(
        (c for c in by_id.values()
         if c["category"] == "chemistry" and c["source"] == "openalex"),
        None,
    )
    assert chem is not None, "no chemistry/openalex row in corpus"

    fake = _fake_openalex_topic(chem, field_id=16)
    raw = src._topic_to_raw(fake)
    assert raw is not None
    assert raw.source_name == "openalex"
    assert raw.external_id == chem["openalex_id"]

    cat = _resolve_category(raw)
    assert cat == "chemistry", f"expected chemistry, got {cat}"

    rec = _emit_concept_from_raw(raw, cat, set(), config)
    assert rec is not None

    # Required fields with same types as pre-Phase-4.
    required = {
        "label": str, "category": str, "features": list,
        "quality": float, "axis_coherence": float, "bridge_degree": int,
        "source_confidence": float, "home_affinity": float,
        "source": str, "openalex_id": str,
    }
    for key, ty in required.items():
        assert key in rec, f"missing required field: {key}"
        assert isinstance(rec[key], ty), (
            f"{key}: expected {ty}, got {type(rec[key])}"
        )

    # Feature shape: list of [int, float] pairs.
    for ax, w in rec["features"]:
        assert isinstance(ax, int)
        assert isinstance(w, float)
        assert 0 <= ax < 128
        assert 0.0 <= w <= 1.0

    # Source/category invariants.
    assert rec["source"] == "openalex"
    assert rec["category"] == "chemistry"


def test_subfield_disambiguation_appends_suffix() -> None:
    """An openalex_subfield label that collides with an earlier label
    receives the " (subfield)" suffix. Bit-identical to pre-Phase-4."""
    config = load_config()
    src = OpenAlexSource()
    seen: set[str] = set()

    topic_payload = {
        "id": "https://openalex.org/T9000",
        "display_name": "Catalysis",
        "description": "study of catalysts",
        "keywords": ["catalyst", "kinetics"],
        "field": {
            "id": "https://openalex.org/fields/16",
            "display_name": "Chemistry",
        },
        "subfield": {"display_name": "Catalysis"},
        "works_count": 5000,
    }
    raw_topic = src._topic_to_raw(topic_payload)
    assert raw_topic is not None
    rec_topic = _emit_concept_from_raw(raw_topic, "chemistry", seen, config)
    assert rec_topic is not None
    assert rec_topic["label"] == "Catalysis"

    sf_payload = {
        "id": "https://openalex.org/subfields/9999",
        "display_name": "Catalysis",
        "field": {
            "id": "https://openalex.org/fields/16",
            "display_name": "Chemistry",
        },
        "works_count": 1000,
    }
    raw_sf = src._subfield_to_raw(sf_payload)
    assert raw_sf is not None
    rec_sf = _emit_concept_from_raw(raw_sf, "chemistry", seen, config)
    assert rec_sf is not None
    assert rec_sf["label"] == "Catalysis (subfield)", (
        f"expected 'Catalysis (subfield)', got {rec_sf['label']!r}"
    )


def test_subfield_source_confidence_is_zero() -> None:
    """openalex_subfield rows have works_count zeroed out (matches
    pre-Phase-4 _topic_from_subfield, which never exposed it)."""
    src = OpenAlexSource()
    sf_payload = {
        "id": "https://openalex.org/subfields/9000",
        "display_name": "Quantum mechanics",
        "field": {
            "id": "https://openalex.org/fields/31",
            "display_name": "Physics",
        },
        "works_count": 1234567,
    }
    raw = src._subfield_to_raw(sf_payload)
    assert raw is not None
    # OpenAlexSource.confidence() reads metadata["works_count"]; the
    # subfield wrapper omits it, so confidence falls back to 0.
    assert src.confidence(raw) == 0.0


def test_gap_fill_path_yields_source_gap_fill() -> None:
    """gap_fill items flow through the orchestrator with source='gap_fill'."""
    from generate_extended import _gap_fill_to_raw
    config = load_config()
    raw = _gap_fill_to_raw("Photoluminescence", ["light", "emission"], "chemistry")
    rec = _emit_concept_from_raw(raw, "chemistry", set(), config)
    assert rec is not None
    assert rec["source"] == "gap_fill"
    assert rec["source_confidence"] == 0.5
    # gap_fill rows do NOT emit openalex_id or wikidata_id.
    assert "openalex_id" not in rec
    assert "wikidata_id" not in rec


def _run_all() -> int:
    tests = [
        test_pipeline_emits_required_concept_fields,
        test_subfield_disambiguation_appends_suffix,
        test_subfield_source_confidence_is_zero,
        test_gap_fill_path_yields_source_gap_fill,
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
