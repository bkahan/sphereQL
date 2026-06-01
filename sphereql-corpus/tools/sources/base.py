"""Source interface — every external taxonomy implements this.

A `Source` is a stateless fetcher + confidence assigner. Stateful
caching, retries, and rate-limiting live in the implementation, not in
the orchestrator. The generator drains every configured source in
order, de-duplicates labels, and feeds each `RawTopic` through the
existing axis-scoring + category-resolution pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from corpus_config import HttpConfig


@dataclass(frozen=True)
class RawTopic:
    """Provenance-agnostic representation of one fetched item.

    The orchestrator converts `RawTopic` → corpus `Concept` via the
    existing axis-scoring + category-resolution pipeline. The
    `source_name` field is preserved verbatim into the emitted concept
    so downstream tooling (the JSON `source` column, Parquet readers,
    quality metrics) keeps its provenance handle.
    """

    external_id: str
    """Stable identifier from the source taxonomy (OpenAlex 'T11999',
    Wikidata 'Q12345', etc.)."""

    label: str
    """Human-readable concept name."""

    description: str
    """One-paragraph description, used for keyword scanning."""

    keywords: list[str]
    """Pre-extracted keywords from the source. May be empty."""

    raw_category_hint: str | None
    """Optional source-side category label (OpenAlex field name,
    Wikidata parent class). The orchestrator falls back to keyword
    scan when absent."""

    source_name: str
    """The label that ends up in the emitted concept's `source` field
    ("openalex", "openalex_subfield", "wikidata", "gap_fill"). The
    provider is recovered via `source_name.split('_')[0]`."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form provenance — works_count, sitelink_count, field_id,
    subfield_display, parent_qid, etc."""


@dataclass(frozen=True)
class SourceConfig:
    """Per-source configuration. Carries the global HTTP settings so
    HTTP-backed sources share retry/timeout semantics with the rest of
    the generator."""

    api_key: str | None = None
    max_items: int | None = None
    """Optional cap on items fetched. None = no cap."""
    cache_dir: str | None = None
    """Optional local cache directory for slow SPARQL responses."""
    http: HttpConfig | None = None
    """Global HTTP knobs (timeout, retries, per-page, backoff). HTTP
    sources should read this; non-HTTP sources (gap_fill) ignore it."""


class Source(Protocol):
    """The pluggable interface every source implements."""

    name: str

    def fetch(self, config: SourceConfig) -> Iterator[RawTopic]:
        """Yield raw topics. Implementations may stream large taxonomies."""

    def confidence(self, topic: RawTopic) -> float:
        """Return a per-topic trust score in [0, 1].

        OpenAlex: `log10(1 + works_count) / 6`.
        Wikidata: `log10(1 + sitelink_count) / 3`.
        gap_fill: constant 0.5.
        """

    def category_hints(self, topic: RawTopic) -> list[tuple[str, float]]:
        """Return ranked (sphereql_category, score) hints, or `[]` for
        "no opinion".

        The orchestrator picks the highest-scored hint when present;
        otherwise it falls back to OpenAlex field IDs (when applicable)
        and finally to keyword scan via `mappings.KEYWORD_TO_AXIS`.
        """
