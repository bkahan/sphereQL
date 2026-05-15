"""Wikidata adapter — fetches academic concepts via SPARQL.

Query targets Q-items that are instances of (P31) or subclasses of
(P279) academic-discipline parents. The Wikidata Query Service has a
60-second timeout and ~10K-row result cap per request, so we split
the fetch into per-parent queries and paginate via LIMIT/OFFSET.

Responses are cached at `cache_dir` (default
`~/.cache/sphereql-corpus/wikidata`) so re-runs are network-free.
Confidence is sitelink-count → log10 in [0, 1].
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from .base import RawTopic, SourceConfig

WDQS_URL = "https://query.wikidata.org/sparql"

# Parent classes whose P279/P31 descendants we want as concepts.
#   Q11862829 — academic discipline
#   Q4671286  — academic major
#   Q1047113  — specialty (medicine-leaning)
#   Q11862588 — academic subdiscipline
#   Q336      — science
PARENT_QIDS: list[str] = [
    "Q11862829",
    "Q4671286",
    "Q1047113",
    "Q11862588",
    "Q336",
]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sphereql-corpus" / "wikidata"

USER_AGENT = (
    "sphereql-corpus/0.2 (https://github.com/bkahan/sphereQL; "
    "benkahan1@gmail.com) python-requests"
)

# Wikidata parent-QID → sphereql category hint. Most parents are too
# broad to opine; "specialty" leans medicine. Unknown/blank parents
# return `None` so the orchestrator falls through to keyword scan.
_PARENT_TO_CATEGORY_RAW: dict[str, str] = {
    "Q11862829": "",
    "Q4671286": "",
    "Q1047113": "medicine",
    "Q11862588": "",
    "Q336": "",
}
_PARENT_TO_CATEGORY: dict[str, str | None] = {
    k: (v or None) for k, v in _PARENT_TO_CATEGORY_RAW.items()
}


class WikidataSource:
    """Stateless SPARQL-backed source. Caches per-parent responses on disk."""

    name = "wikidata"

    def fetch(self, config: SourceConfig) -> Iterator[RawTopic]:
        cache = (
            Path(config.cache_dir) if config.cache_dir else DEFAULT_CACHE_DIR
        )
        cache.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        n_emitted = 0
        for parent in PARENT_QIDS:
            for raw in self._fetch_subclasses(parent, cache):
                if config.max_items is not None and n_emitted >= config.max_items:
                    return
                if raw.external_id in seen:
                    continue
                seen.add(raw.external_id)
                yield raw
                n_emitted += 1
            print(
                f"  wikidata {parent}: cumulative emitted={n_emitted}",
                file=sys.stderr,
            )

    def confidence(self, topic: RawTopic) -> float:
        sitelinks = topic.metadata.get("sitelink_count") or 0
        try:
            sitelinks = int(sitelinks)
        except (TypeError, ValueError):
            sitelinks = 0
        if sitelinks <= 0:
            return 0.0
        return max(0.0, min(1.0, math.log10(1.0 + sitelinks) / 3.0))

    def category_hints(self, topic: RawTopic) -> list[tuple[str, float]]:
        parent = topic.metadata.get("parent_qid")
        cat = _PARENT_TO_CATEGORY.get(parent) if isinstance(parent, str) else None
        if cat:
            return [(cat, 1.0)]
        return []

    # ─── internals ──────────────────────────────────────────────────────

    def _fetch_subclasses(
        self, parent_qid: str, cache: Path
    ) -> Iterator[RawTopic]:
        cache_file = cache / f"{parent_qid}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                rows = json.load(f)
            print(
                f"  wikidata {parent_qid}: cache hit ({len(rows)} rows)",
                file=sys.stderr,
            )
        else:
            print(
                f"  wikidata {parent_qid}: SPARQL fetch (no cache)…",
                file=sys.stderr,
            )
            rows = self._sparql_query(parent_qid)
            cache_file.write_text(json.dumps(rows, ensure_ascii=False, indent=0))
            print(
                f"  wikidata {parent_qid}: cached {len(rows)} rows → {cache_file}",
                file=sys.stderr,
            )

        for row in rows:
            raw = self._row_to_raw(row, parent_qid)
            if raw is not None:
                yield raw

    @staticmethod
    def _sparql_query(parent_qid: str, batch_size: int = 5000) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        offset = 0
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        }
        while True:
            query = (
                "SELECT ?item ?itemLabel ?itemDescription "
                "(COUNT(DISTINCT ?sitelink) AS ?sitelinks) "
                '(GROUP_CONCAT(DISTINCT ?aliasLabel; separator="|") AS ?aliases) '
                "WHERE { "
                f"?item (wdt:P279|wdt:P31)* wd:{parent_qid} . "
                "OPTIONAL { ?sitelink schema:about ?item . } "
                "OPTIONAL { ?item skos:altLabel ?aliasLabel . "
                'FILTER(LANG(?aliasLabel) = "en") } '
                "SERVICE wikibase:label { "
                'bd:serviceParam wikibase:language "en". } '
                "} "
                "GROUP BY ?item ?itemLabel ?itemDescription "
                "ORDER BY ?item "
                f"LIMIT {batch_size} OFFSET {offset}"
            )
            resp = requests.get(
                WDQS_URL,
                params={"query": query},
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", {}).get("bindings", [])
            if not results:
                break
            all_rows.extend(results)
            print(
                f"    {parent_qid} offset={offset}: +{len(results)} "
                f"(total {len(all_rows)})",
                file=sys.stderr,
            )
            offset += batch_size
            time.sleep(1.0)
            if len(results) < batch_size:
                break
        return all_rows

    @staticmethod
    def _row_to_raw(row: dict[str, Any], parent_qid: str) -> RawTopic | None:
        item_uri = row.get("item", {}).get("value", "")
        qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
        if not qid.startswith("Q"):
            return None
        label = row.get("itemLabel", {}).get("value", "")
        if not label or label == qid:
            return None
        description = row.get("itemDescription", {}).get("value", "")
        aliases_str = row.get("aliases", {}).get("value", "")
        aliases = [a for a in aliases_str.split("|") if a]
        sitelinks_raw = row.get("sitelinks", {}).get("value", "0")
        try:
            sitelinks = int(sitelinks_raw)
        except (TypeError, ValueError):
            sitelinks = 0

        return RawTopic(
            external_id=qid,
            label=label,
            description=description,
            keywords=aliases,
            raw_category_hint=None,
            source_name="wikidata",
            metadata={
                "parent_qid": parent_qid,
                "sitelink_count": sitelinks,
            },
        )
