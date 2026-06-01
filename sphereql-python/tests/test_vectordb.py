import pytest
import sphereql


DIM = 10
N = 20


def make_records(n=N, dim=DIM):
    """Synthetic embeddings: half strong in dim-0, half strong in dim-1."""
    records = []
    for i in range(n):
        v = [0.0] * dim
        if i < n // 2:
            v[0] = 1.0 + i * 0.01
            v[1] = 0.1
            cat = "group_a"
        else:
            v[0] = 0.1
            v[1] = 1.0 + i * 0.01
            cat = "group_b"
        v[2] = 0.05 * i
        records.append({"id": f"rec-{i}", "vector": v, "metadata": {"category": cat}})
    return records


def make_store_and_bridge(n=N, dim=DIM):
    store = sphereql.InMemoryStore("test", dim)
    store.upsert(make_records(n, dim))
    bridge = sphereql.VectorStoreBridge(store, batch_size=100)
    bridge.build_pipeline(category_key="category")
    return store, bridge


class TestInMemoryStore:
    def test_upsert_and_count(self):
        store = sphereql.InMemoryStore("test", DIM)
        store.upsert(make_records())
        assert store.count() == N
        assert len(store) == N

    def test_repr(self):
        store = sphereql.InMemoryStore("my-collection", 768)
        assert "my-collection" in repr(store)


class TestVectorStoreBridge:
    def test_build_and_nearest(self):
        _, bridge = make_store_and_bridge()
        query = [0.9] * DIM
        results = bridge.query_nearest(query, k=5)
        assert len(results) == 5
        for r in results:
            assert hasattr(r, "id")
            assert hasattr(r, "category")
            assert hasattr(r, "distance")
            assert hasattr(r, "certainty")
            assert hasattr(r, "intensity")
        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_hybrid_search(self):
        _, bridge = make_store_and_bridge(n=30)
        query = [0.9] * DIM
        results = bridge.hybrid_search(query, final_k=5, recall_k=15)
        assert len(results) == 5
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_sync_projections(self):
        _, bridge = make_store_and_bridge()
        count = bridge.sync_projections()
        assert count == N

    def test_concept_path(self):
        _, bridge = make_store_and_bridge()
        query = [0.9] * DIM
        path = bridge.query_concept_path(
            "rec-0", "rec-15", graph_k=10, embedding=query
        )
        assert path is not None
        assert len(path.steps) >= 2
        assert path.steps[0].id == "rec-0"
        assert path.steps[-1].id == "rec-15"

    def test_detect_globs(self):
        _, bridge = make_store_and_bridge(n=30)
        query = [0.9] * DIM
        globs = bridge.query_detect_globs(query, k=2, max_k=5)
        assert len(globs) == 2
        total = sum(g.member_count for g in globs)
        assert total == 30

    def test_before_build_raises(self):
        store = sphereql.InMemoryStore("test", DIM)
        store.upsert(make_records())
        bridge = sphereql.VectorStoreBridge(store)
        query = [0.9] * DIM
        with pytest.raises(RuntimeError, match="pipeline not built"):
            bridge.query_nearest(query, k=5)

    def test_len(self):
        _, bridge = make_store_and_bridge()
        assert len(bridge) == N

    def test_category_stats(self):
        _, bridge = make_store_and_bridge()
        summaries, inner_reports = bridge.category_stats()
        assert len(summaries) == 2
        names = {s.name for s in summaries}
        assert names == {"group_a", "group_b"}
        assert isinstance(inner_reports, list)

    def test_category_neighbors(self):
        _, bridge = make_store_and_bridge()
        neighbors = bridge.category_neighbors("group_a", k=1)
        assert len(neighbors) >= 1

    def test_drill_down(self):
        _, bridge = make_store_and_bridge()
        query = [0.9] * DIM
        hits = bridge.drill_down("group_a", query, k=3)
        assert len(hits) == 3
        for h in hits:
            assert isinstance(h.item_index, int)
            assert isinstance(h.distance, float)

    def test_domain_groups(self):
        _, bridge = make_store_and_bridge()
        groups = bridge.domain_groups()
        assert isinstance(groups, list)
        total = sum(g.total_items for g in groups)
        assert total == N

    def test_hierarchical_nearest(self):
        _, bridge = make_store_and_bridge()
        query = [0.9] * DIM
        results = bridge.hierarchical_nearest(query, k=3)
        assert len(results) == 3

    def test_projection_warnings(self):
        _, bridge = make_store_and_bridge()
        warnings = bridge.projection_warnings()
        assert isinstance(warnings, list)
        for w in warnings:
            assert w.severity in ("Info", "Warning", "Critical")

    def test_projection_kind(self):
        _, bridge = make_store_and_bridge()
        kind = bridge.projection_kind
        assert kind in ("pca", "kernel_pca", "laplacian_eigenmap")

    def test_config(self):
        _, bridge = make_store_and_bridge()
        cfg = bridge.config()
        assert cfg is not None
        assert isinstance(cfg, dict)
        assert "projection_kind" in cfg

    def test_repr(self):
        _, bridge = make_store_and_bridge()
        assert "VectorStoreBridge(" in repr(bridge)
        assert str(N) in repr(bridge)
