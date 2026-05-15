"""Config loader for the corpus generator + validator.

Loads tools/corpus_config.toml into a frozen dataclass. Supports per-key
overrides via --set key.subkey=value. Designed to be imported by both
generate_extended.py and validate_corpus.py with no side effects.

Python 3.11+: uses stdlib tomllib. Older versions: install tomli.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class WeightCurveEntry:
    base: float
    divisor: float


@dataclass(frozen=True)
class GenerationConfig:
    num_axes: int
    min_features: int
    max_features: int
    n_features_min: int
    n_features_max: int
    weight_floor: float
    weight_ceiling: float
    weight_jitter_range: float
    weight_round_decimals: int
    primary_seed_hits: int
    weight_curve: tuple[WeightCurveEntry, ...]


@dataclass(frozen=True)
class ValidationConfig:
    min_total_concepts: int
    min_per_category: int
    warn_thin_category: int
    min_mean_features: float
    max_mean_features: float
    min_bridge_ratio: float
    max_misrouted_ratio: float


@dataclass(frozen=True)
class HttpConfig:
    timeout_seconds: int
    per_page: int
    inter_page_sleep_seconds: float
    retries: int
    backoff_base: float
    backoff_multiplier: float


@dataclass(frozen=True)
class OutputConfig:
    path: str
    parquet_path: str


@dataclass(frozen=True)
class QualityMetricConfig:
    """Weights for sphereql_embed::CorpusQuality (Phase 5).

    Mirror of the `[quality_metric]` TOML table. Phase 6 consumes these
    when constructing the Rust metric from Python; Phase 5 only requires
    the schema to round-trip cleanly.
    """

    w_evr: float
    w_bridge: float
    w_curvature: float
    w_balance: float


@dataclass(frozen=True)
class CorpusConfig:
    generation: GenerationConfig
    validation: ValidationConfig
    http: HttpConfig
    output: OutputConfig
    quality_metric: QualityMetricConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "corpus_config.toml"


def load_config(
    path: Path | None = None, overrides: list[str] | None = None
) -> CorpusConfig:
    """Load config from TOML, apply --set key.subkey=value overrides."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    with open(cfg_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    for override in overrides or []:
        _apply_override(raw, override)

    gen = raw["generation"]
    weight_curve = tuple(
        WeightCurveEntry(base=float(e["base"]), divisor=float(e["divisor"]))
        for e in gen["weight_curve"]
    )
    if len(weight_curve) < 1:
        raise ValueError("generation.weight_curve must have at least one entry")

    return CorpusConfig(
        generation=GenerationConfig(
            num_axes=int(gen["num_axes"]),
            min_features=int(gen["min_features"]),
            max_features=int(gen["max_features"]),
            n_features_min=int(gen["n_features_min"]),
            n_features_max=int(gen["n_features_max"]),
            weight_floor=float(gen["weight_floor"]),
            weight_ceiling=float(gen["weight_ceiling"]),
            weight_jitter_range=float(gen["weight_jitter_range"]),
            weight_round_decimals=int(gen["weight_round_decimals"]),
            primary_seed_hits=int(gen["primary_seed_hits"]),
            weight_curve=weight_curve,
        ),
        validation=ValidationConfig(
            min_total_concepts=int(raw["validation"]["min_total_concepts"]),
            min_per_category=int(raw["validation"]["min_per_category"]),
            warn_thin_category=int(raw["validation"]["warn_thin_category"]),
            min_mean_features=float(raw["validation"]["min_mean_features"]),
            max_mean_features=float(raw["validation"]["max_mean_features"]),
            min_bridge_ratio=float(raw["validation"]["min_bridge_ratio"]),
            max_misrouted_ratio=float(raw["validation"]["max_misrouted_ratio"]),
        ),
        http=HttpConfig(
            timeout_seconds=int(raw["http"]["timeout_seconds"]),
            per_page=int(raw["http"]["per_page"]),
            inter_page_sleep_seconds=float(raw["http"]["inter_page_sleep_seconds"]),
            retries=int(raw["http"]["retries"]),
            backoff_base=float(raw["http"]["backoff_base"]),
            backoff_multiplier=float(raw["http"]["backoff_multiplier"]),
        ),
        output=OutputConfig(
            path=str(raw["output"]["path"]),
            parquet_path=str(raw["output"]["parquet_path"]),
        ),
        quality_metric=QualityMetricConfig(
            w_evr=float(raw["quality_metric"]["w_evr"]),
            w_bridge=float(raw["quality_metric"]["w_bridge"]),
            w_curvature=float(raw["quality_metric"]["w_curvature"]),
            w_balance=float(raw["quality_metric"]["w_balance"]),
        ),
    )


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a single --set key.subkey=value override. Raises on malformed input."""
    if "=" not in override:
        raise ValueError(f"override must be key=value, got: {override!r}")
    key, _, value = override.partition("=")
    parts = key.split(".")
    cursor: Any = raw
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise KeyError(f"override path {key!r}: {part!r} is not a table")
        cursor = cursor[part]
    leaf = parts[-1]
    if leaf not in cursor:
        raise KeyError(f"override path {key!r}: leaf {leaf!r} not present")
    cursor[leaf] = _coerce(cursor[leaf], value)


def _coerce(existing: Any, value: str) -> Any:
    """Coerce CLI string to the type of the existing config value."""
    if isinstance(existing, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add --config and --set arguments to an argparse parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to corpus_config.toml (default: tools/corpus_config.toml)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key. Repeatable. Example: --set generation.min_features=3",
    )
