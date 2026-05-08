"""
Live Qdrant integration test.

Skipped unless QDRANT_API_KEY and QDRANT_CLUSTER_ENDPOINT are set
(typically via .env.local at the repo root). Exercises the full path:
real text -> embedding -> Qdrant upsert -> sphereQL pipeline -> query.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import sphereql

# Reuse the demo's encoder + REST upsert helper.
import sys
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
from dataset import SENTENCES, encode  # noqa: E402
from qdrant_e2e import (  # noqa: E402
    DIM,
    QDRANT_GRPC_PORT,
    grpc_url,
    upsert_via_rest,
)


COLLECTION = "sphereql_pytest_e2e"


def _load_env() -> tuple[str | None, str | None]:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env.local"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass
    return os.environ.get("QDRANT_API_KEY"), os.environ.get("QDRANT_CLUSTER_ENDPOINT")


@pytest.fixture(scope="module")
def qdrant_creds():
    api_key, endpoint = _load_env()
    if not api_key or not endpoint:
        pytest.skip("QDRANT_API_KEY / QDRANT_CLUSTER_ENDPOINT not set")
    return api_key, endpoint


@pytest.fixture(scope="module")
def seeded_bridge(qdrant_creds):
    api_key, endpoint = qdrant_creds
    subset = [
        s for s in SENTENCES
        if s["category"] in ("science", "technology", "cooking")
    ][:30]
    records = [
        {
            "id": f"pytest-{i:03d}",
            "vector": encode(s["text"], dim=DIM),
            "metadata": {"category": s["category"], "text": s["text"]},
        }
        for i, s in enumerate(subset)
    ]

    _drop_collection(endpoint, api_key, COLLECTION)
    upsert_via_rest(endpoint, api_key, COLLECTION, DIM, records)
    time.sleep(1.0)

    bridge = sphereql.QdrantBridge(
        url=grpc_url(endpoint),
        collection=COLLECTION,
        dimension=DIM,
        api_key=api_key,
    )
    bridge.build_pipeline(category_key="category")

    yield bridge, records

    _drop_collection(endpoint, api_key, COLLECTION)


def _drop_collection(endpoint: str, api_key: str, collection: str) -> None:
    import urllib.error
    import urllib.request

    base = endpoint.rstrip("/")
    if "://" not in base:
        base = f"https://{base}"
    req = urllib.request.Request(
        f"{base}/collections/{collection}",
        headers={"api-key": api_key},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        if e.code not in (404,):
            raise


def test_bridge_loaded_all_records(seeded_bridge):
    bridge, records = seeded_bridge
    assert len(bridge) == len(records)
    assert bridge.projection_kind == "pca"


def test_query_nearest_returns_ranked_results(seeded_bridge):
    bridge, _ = seeded_bridge
    query = encode(
        "Photosynthesis converts sunlight into chemical energy.",
        dim=DIM,
    )
    hits = bridge.query_nearest(query, k=5)
    assert len(hits) == 5
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)
    # query_nearest works in the projected sphere and exposes sphereQL's
    # synthetic point ids (s-NNNN); the original Qdrant id round-trips
    # through hybrid_search instead. Just confirm the shape is right.
    assert all(h.id.startswith("s-") for h in hits)
    assert all(h.category in {"science", "technology", "cooking"} for h in hits)


def test_hybrid_search_rerank_against_qdrant(seeded_bridge):
    bridge, _ = seeded_bridge
    query = encode(
        "Caramelizing onions develops deep, sweet flavors over low heat.",
        dim=DIM,
    )
    results = bridge.hybrid_search(query, final_k=5, recall_k=15)
    assert len(results) == 5
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # Every result should carry the original payload back from Qdrant.
    for r in results:
        assert "category" in r["metadata"]
        assert "text" in r["metadata"]
    # The closest match for a cooking query should land in the cooking
    # category — proves the Qdrant round-trip preserved the embedding.
    assert results[0]["metadata"]["category"] == "cooking"


def test_category_stats_match_seeded_categories(seeded_bridge):
    bridge, records = seeded_bridge
    summaries, _ = bridge.category_stats()
    seen = {s.name for s in summaries}
    expected = {r["metadata"]["category"] for r in records}
    assert expected.issubset(seen)


def test_sync_projections_writes_back_to_qdrant(seeded_bridge):
    bridge, records = seeded_bridge
    written = bridge.sync_projections()
    assert written == len(records)
