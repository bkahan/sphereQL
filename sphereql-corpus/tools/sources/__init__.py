"""Pluggable corpus-source registry.

A `Source` fetches `RawTopic`s from an external taxonomy and provides
per-topic confidence + category hints. The generator orchestrates one
or more sources, de-duplicates labels, and emits a unified corpus.

To add a source: implement the `Source` Protocol in `sources/<name>.py`,
import its class here, and add it to `SOURCE_REGISTRY`. Nothing else in
the generator needs to change.
"""

from __future__ import annotations

from .base import RawTopic, Source, SourceConfig
from .openalex import OpenAlexSource
from .wikidata import WikidataSource

SOURCE_REGISTRY: dict[str, type] = {
    "openalex": OpenAlexSource,
    "wikidata": WikidataSource,
}


def make_source(name: str) -> Source:
    """Instantiate a registered source by name. Raises `KeyError` if missing.

    The provider portion of an emitted concept's `source` field is
    recovered via `source_name.split('_')[0]`, so this accepts the bare
    provider name ("openalex" for both topic and subfield rows).
    """
    if name not in SOURCE_REGISTRY:
        raise KeyError(
            f"unknown source: {name!r}. Registered: {sorted(SOURCE_REGISTRY)}"
        )
    return SOURCE_REGISTRY[name]()


__all__ = [
    "RawTopic",
    "Source",
    "SourceConfig",
    "OpenAlexSource",
    "WikidataSource",
    "SOURCE_REGISTRY",
    "make_source",
]
