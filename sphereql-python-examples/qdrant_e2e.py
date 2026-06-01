"""
End-to-end demo against a live Qdrant cluster.

Real input -> embedding -> Qdrant upsert -> SphereQL pipeline -> query response.

Setup
-----
The `qdrant` Cargo feature must be enabled at build time. The default
PyPI wheel and `maturin develop` (no flags) do NOT include it.

    pip install python-dotenv
    cd sphereql-python
    # Build into the *active* Python. Without VIRTUAL_ENV set, maturin
    # falls through to the system Python's user site-packages, which
    # may not be the interpreter you run this script with.
    python -m maturin develop --features qdrant --release

Verify the build landed in the right place:

    python -c "import sphereql; print(sphereql.__file__, hasattr(sphereql, 'QdrantBridge'))"

If `QdrantBridge` is False, the wheel was built against a different
Python. Set VIRTUAL_ENV (or activate the venv) and re-run maturin.

Env (read from .env.local at the repo root):
    QDRANT_API_KEY            Qdrant Cloud API key
    QDRANT_CLUSTER_ENDPOINT   https://<cluster>.<region>.aws.cloud.qdrant.io

Run:
    python qdrant_e2e.py

Remaining work
--------------
Things this demo papers over that we should address before calling the
Qdrant integration "done":

1. PyQdrantBridge does not surface `upsert` / `delete` / `count`.
   We seed the collection over the REST API as a workaround. The
   bridge should expose write methods so callers don't need a second
   client. (See sphereql-python/src/vectordb.rs — InMemoryStore has
   `upsert`; the Qdrant/Pinecone bridges only have constructors.)

2. The id duality is a footgun. `query_nearest` returns sphereQL's
   synthetic `s-NNNN` ids (projected-sphere queries), while
   `hybrid_search` returns the original Qdrant ids. We should either
   unify on the original id everywhere or document this clearly in
   the bridge's docstring.

3. No collection cleanup. This example leaves `sphereql_e2e_demo`
   behind on the cluster on success. Either wrap in try/finally with
   a REST DELETE (the test fixture already does this) or expose a
   `bridge.drop()` from Python.

4. 64-d FNV-1a hash embeddings. The encoder in `dataset.py` is
   reproducible-but-toy; cooking-vs-science separation works because
   the lexicons barely overlap. A real story needs sentence-transformers
   (or another semantic embedder) plumbed in — probably an optional
   `examples/qdrant_real_embeddings.py` that imports `sentence-transformers`.

5. gRPC port handling. We hard-code `:6334` for Qdrant Cloud. Self-hosted
   Qdrant on a custom port would break this. Should accept the full URL
   verbatim or parse a port out of QDRANT_CLUSTER_ENDPOINT.

6. CI integration. `tests/test_qdrant_live.py` skips when env vars are
   missing, which is the right default — but we have no CI job that
   sets the secret and runs the live tests. Add a manual-trigger GitHub
   Actions workflow gated on a repo secret.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import sphereql
from dataset import SENTENCES, encode

# Qdrant Cloud REST is on :6333; gRPC (what the Rust client uses) is on :6334.
QDRANT_GRPC_PORT = 6334
DIM = 64
COLLECTION = "sphereql_e2e_demo"


def header(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def load_env() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env.local")
    api_key = os.environ.get("QDRANT_API_KEY")
    endpoint = os.environ.get("QDRANT_CLUSTER_ENDPOINT")
    if not api_key or not endpoint:
        sys.exit(
            "missing QDRANT_API_KEY / QDRANT_CLUSTER_ENDPOINT — "
            f"checked {repo_root / '.env.local'}"
        )
    return api_key, endpoint


def grpc_url(endpoint: str) -> str:
    """Qdrant Cloud HTTPS endpoint -> gRPC URL on :6334.

    The qdrant-client Rust crate parses scheme+host+port; it keeps TLS when
    the scheme is https, which is what Cloud requires.
    """
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    # Strip a trailing port if the user pasted one, then re-add gRPC.
    base, _, _ = endpoint.partition("://")[2].partition(":")
    scheme = endpoint.split("://", 1)[0]
    return f"{scheme}://{base}:{QDRANT_GRPC_PORT}"


def main() -> int:
    api_key, endpoint = load_env()
    url = grpc_url(endpoint)

    header(f"Connect to Qdrant ({url})")
    bridge = sphereql.QdrantBridge(
        url=url,
        collection=COLLECTION,
        dimension=DIM,
        api_key=api_key,
    )
    print(f"  collection: {COLLECTION!r} (created if missing)")

    # ── 1. Real input: raw text -> deterministic 64-d embedding ─────────
    header("1. Encode 30 real sentences (3 categories) to 64-d vectors")
    subset = [
        s for s in SENTENCES
        if s["category"] in ("science", "technology", "cooking")
    ][:30]
    records = [
        {
            "id": f"e2e-{i:03d}",
            "vector": encode(s["text"], dim=DIM),
            "metadata": {"category": s["category"], "text": s["text"]},
        }
        for i, s in enumerate(subset)
    ]
    print(f"  encoded {len(records)} records "
          f"({sum(1 for r in records if r['metadata']['category'] == 'science')} science, "
          f"{sum(1 for r in records if r['metadata']['category'] == 'technology')} tech, "
          f"{sum(1 for r in records if r['metadata']['category'] == 'cooking')} cooking)")

    # ── 2. Live API call: upsert to the cluster ─────────────────────────
    # PyQdrantBridge surfaces the read path; seed the collection over the
    # Qdrant REST API (stdlib only) so build_pipeline has data to fit.
    header("2. Upsert into Qdrant (live API call)")
    upsert_count = upsert_via_rest(endpoint, api_key, COLLECTION, DIM, records)
    print(f"  upserted {upsert_count} points")
    # Qdrant Cloud needs a moment for the points to be searchable.
    time.sleep(1.0)

    # ── 3. Build the SphereQL pipeline over live Qdrant data ────────────
    header("3. Pull vectors from Qdrant + fit SphereQL projection")
    bridge.build_pipeline(category_key="category")
    print(f"  records in bridge:  {len(bridge)}")
    print(f"  projection_kind:    {bridge.projection_kind}")

    # ── 4. Real input -> SphereQL query -> response ─────────────────────
    header("4. Query: real input -> SphereQL response")
    query_text = (
        "Photosynthesis converts sunlight into chemical energy stored in glucose."
    )
    query_vec = encode(query_text, dim=DIM)
    print(f"  Query: {query_text!r}\n")

    print("  bridge.query_nearest(k=5):")
    for i, hit in enumerate(bridge.query_nearest(query_vec, k=5), 1):
        print(f"    {i}. [{hit.category:<10}] d={hit.distance:.4f}  id={hit.id}")

    print("\n  bridge.hybrid_search(final_k=5, recall_k=15)  "
          "[Qdrant ANN + cosine re-rank]:")
    for i, r in enumerate(
        bridge.hybrid_search(query_vec, final_k=5, recall_k=15), 1
    ):
        text = r["metadata"].get("text", "")
        snippet = (text[:55] + "...") if len(text) > 55 else text
        print(f"    {i}. [{r['metadata']['category']:<10}] "
              f"score={r['score']:.4f}  {snippet}")

    # ── 5. Category enrichment over live data ───────────────────────────
    header("5. Category structure inferred from Qdrant points")
    summaries, _ = bridge.category_stats()
    summaries.sort(key=lambda s: -s.member_count)
    for s in summaries:
        print(f"  {s.name:<12} members={s.member_count:>2}  "
              f"cohesion={s.cohesion:.4f}  bridge_q={s.bridge_quality:.4f}")

    # ── 6. Write spherical coords back to Qdrant payload ────────────────
    header("6. sync_projections -> Qdrant payload")
    written = bridge.sync_projections()
    print(f"  wrote r/theta/phi to {written} Qdrant points")
    print("  (downstream services can now read sphereQL coords from payload)")

    header("Done")
    return 0


def upsert_via_rest(
    endpoint: str,
    api_key: str,
    collection: str,
    dim: int,
    records: list[dict],
) -> int:
    """Upsert points into Qdrant Cloud via the REST API on :6333.

    The Python bridge wraps gRPC and doesn't surface upsert directly, so
    we seed the collection over HTTPS using only the stdlib. Once the
    collection exists, the bridge handles the read path natively.
    """
    import json
    import urllib.request
    import uuid

    base = endpoint.rstrip("/")
    if "://" not in base:
        base = f"https://{base}"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    # Ensure collection exists (matches QdrantStore default: cosine).
    create_url = f"{base}/collections/{collection}"
    create_body = json.dumps(
        {"vectors": {"size": dim, "distance": "Cosine"}}
    ).encode()
    req = urllib.request.Request(
        create_url, data=create_body, headers=headers, method="PUT"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        # 409 / 400 if it already exists with the same config — tolerate.
        if e.code not in (400, 409):
            raise

    # Same SHA-256 -> UUID derivation as QdrantStore::string_to_point_id,
    # so the ids the bridge sees on read line up with what we write.
    points = [
        {
            "id": _id_to_uuid(r["id"]),
            "vector": r["vector"],
            "payload": {
                "_sphereql_id": r["id"],
                **r["metadata"],
            },
        }
        for r in records
    ]
    upsert_url = f"{base}/collections/{collection}/points?wait=true"
    body = json.dumps({"points": points}).encode()
    req = urllib.request.Request(
        upsert_url, data=body, headers=headers, method="PUT"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if result.get("status") != "ok":
        raise RuntimeError(f"qdrant upsert failed: {result}")
    return len(points)


def _id_to_uuid(s: str) -> str:
    """Mirror of sphereql-vectordb::qdrant::string_to_point_id."""
    import hashlib
    h = hashlib.sha256(s.encode()).digest()
    return (
        f"{h[0]:02x}{h[1]:02x}{h[2]:02x}{h[3]:02x}-"
        f"{h[4]:02x}{h[5]:02x}-"
        f"{h[6]:02x}{h[7]:02x}-"
        f"{h[8]:02x}{h[9]:02x}-"
        f"{h[10]:02x}{h[11]:02x}{h[12]:02x}{h[13]:02x}{h[14]:02x}{h[15]:02x}"
    )


if __name__ == "__main__":
    sys.exit(main())
