"""OpenAlex Topics + Subfields adapter.

Refactored from the inline OpenAlex code in `generate_extended.py`
(pre-Phase 4). Behavior is preserved bit-identically: same endpoint
order (topics then subfields), same SELECT field lists, same cursor
pagination, same exponential-backoff retry, same `mailto`/`api_key`
auto-detection.

The orchestrator's bit-identity acceptance test reruns
`generate_extended.py --source openalex` against the pre-Phase-4
output (modulo `generated_at`); any drift here breaks that check.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Iterator

import requests

from .base import RawTopic, SourceConfig

OPENALEX_BASE = "https://api.openalex.org"
TOPIC_SELECT = "id,display_name,description,keywords,subfield,field,domain,works_count"
SUBFIELD_SELECT = "id,display_name,field,domain,works_count"


class OpenAlexSource:
    """Fetches OpenAlex Topics (~4,500) and Subfields (~254).

    Yields topics first, then subfields, to match the pre-Phase-4 emit
    order. Topics carry `source_name="openalex"`; subfields carry
    `source_name="openalex_subfield"`. The subfield " (subfield)"
    suffix on label collisions is applied by the orchestrator's
    de-duper, not here — this source emits the canonical display_name.
    """

    name = "openalex"

    def fetch(self, config: SourceConfig) -> Iterator[RawTopic]:
        api_key = config.api_key or os.environ.get("OPENALEX_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAlexSource requires api_key in SourceConfig or "
                "OPENALEX_API_KEY env var"
            )
        if config.http is None:
            raise RuntimeError(
                "OpenAlexSource.fetch requires SourceConfig.http to be set"
            )

        n_emitted = 0

        print("Fetching topics from OpenAlex…", file=sys.stderr)
        for topic in self._paginate("topics", TOPIC_SELECT, api_key, config.http):
            if config.max_items is not None and n_emitted >= config.max_items:
                return
            raw = self._topic_to_raw(topic)
            if raw is not None:
                yield raw
                n_emitted += 1

        print("Fetching subfields from OpenAlex…", file=sys.stderr)
        for sf in self._paginate("subfields", SUBFIELD_SELECT, api_key, config.http):
            if config.max_items is not None and n_emitted >= config.max_items:
                return
            raw = self._subfield_to_raw(sf)
            if raw is not None:
                yield raw
                n_emitted += 1

    def confidence(self, topic: RawTopic) -> float:
        works = topic.metadata.get("works_count") or 0
        try:
            works = int(works)
        except (TypeError, ValueError):
            works = 0
        return max(0.0, min(1.0, math.log10(1.0 + works) / 6.0))

    def category_hints(self, topic: RawTopic) -> list[tuple[str, float]]:
        # OpenAlex provides field IDs; the orchestrator resolves them via
        # `mappings.FIELD_TO_CATEGORY` and `FIELD_MULTI_MAP`, plus the
        # `CONTENT_OVERRIDES` keyword table. Returning `[]` keeps the
        # routing logic centralized in the orchestrator (where it has
        # always been) rather than duplicating mappings.py here.
        return []

    # ─── internals ──────────────────────────────────────────────────────

    def _paginate(
        self,
        endpoint: str,
        select: str,
        api_key: str,
        http,
    ) -> Iterator[dict[str, Any]]:
        cursor = "*"
        page = 0
        total = 0
        while cursor:
            params = {
                "per_page": http.per_page,
                "cursor": cursor,
                "select": select,
                **self._auth_params(api_key),
            }
            data = self._get_with_retry(f"{OPENALEX_BASE}/{endpoint}", params, http)
            results = data.get("results", [])
            total += len(results)
            page += 1
            print(
                f"  {endpoint} page {page}: +{len(results)} (total {total})",
                file=sys.stderr,
            )
            for r in results:
                yield r
            cursor = data.get("meta", {}).get("next_cursor")
            time.sleep(http.inter_page_sleep_seconds)
            if not results:
                break

    @staticmethod
    def _get_with_retry(url: str, params: dict[str, Any], http) -> dict[str, Any]:
        delay = http.backoff_base
        retries = http.retries
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=http.timeout_seconds)
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
                delay *= http.backoff_multiplier
        raise RuntimeError("unreachable")

    @staticmethod
    def _auth_params(api_key: str) -> dict[str, str]:
        # Polite-pool email vs Premium API key, auto-detected.
        if "@" in api_key:
            return {"mailto": api_key}
        return {"api_key": api_key}

    @classmethod
    def _topic_to_raw(cls, topic: dict[str, Any]) -> RawTopic | None:
        label = topic.get("display_name")
        if not label:
            return None
        sub = topic.get("subfield")
        subfield_display = (
            sub.get("display_name") if isinstance(sub, dict) else None
        )
        return RawTopic(
            external_id=cls._id_tail(topic.get("id")) or "",
            label=str(label),
            description=str(topic.get("description") or ""),
            keywords=[str(k) for k in (topic.get("keywords") or [])],
            raw_category_hint=None,
            source_name="openalex",
            metadata={
                "works_count": topic.get("works_count", 0),
                "field": topic.get("field"),
                "subfield_display": subfield_display,
            },
        )

    @classmethod
    def _subfield_to_raw(cls, sf: dict[str, Any]) -> RawTopic | None:
        label = sf.get("display_name")
        if not label:
            return None
        return RawTopic(
            external_id=cls._id_tail(sf.get("id")) or "",
            label=str(label),
            description="",
            keywords=[],
            raw_category_hint=None,
            source_name="openalex_subfield",
            metadata={
                # works_count intentionally omitted: pre-Phase-4
                # `_topic_from_subfield` discarded it, so
                # source_confidence collapses to 0 for subfield rows.
                # Reintroducing it would break the bit-identity check.
                "field": sf.get("field"),
                "subfield_display": str(label),
            },
        )

    @staticmethod
    def _id_tail(raw: Any) -> str | None:
        if not isinstance(raw, str):
            return None
        tail = raw.rstrip("/").rsplit("/", 1)[-1]
        return tail or None
