import json
import pytest
import sphereql


# ── Fixtures ────────────────────────────────────────────────────────────

CATEGORIES = ["science", "cooking", "science", "cooking", "science",
              "cooking", "science", "cooking", "science", "cooking"]

EMBEDDINGS = [
    [1.0, 0.1, 0.0, 0.2],
    [0.1, 1.0, 0.0, 0.2],
    [0.9, 0.2, 0.1, 0.3],
    [0.2, 0.9, 0.1, 0.3],
    [0.8, 0.3, 0.2, 0.1],
    [0.3, 0.8, 0.2, 0.1],
    [0.85, 0.15, 0.05, 0.25],
    [0.15, 0.85, 0.05, 0.25],
    [0.95, 0.05, 0.1, 0.15],
    [0.05, 0.95, 0.1, 0.15],
]

QUERY = [0.9, 0.1, 0.0, 0.2]


@pytest.fixture
def pipeline():
    return sphereql.Pipeline(CATEGORIES, EMBEDDINGS)


# ── Pipeline construction ───────────────────────────────────────────────

class TestPipelineConstruction:
    def test_native_constructor(self, pipeline):
        assert len(pipeline) == 10
        assert bool(pipeline)
        assert "Pipeline(items=10)" == repr(pipeline)

    def test_from_json(self):
        data = json.dumps({"categories": CATEGORIES, "embeddings": EMBEDDINGS})
        p = sphereql.Pipeline.from_json(data)
        assert len(p) == 10

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="categories length"):
            sphereql.Pipeline(["a", "b"], [[1, 0, 0, 0]])

    def test_from_json_missing_categories(self):
        with pytest.raises(ValueError, match="categories"):
            sphereql.Pipeline.from_json('{"embeddings": [[1,0,0]]}')

    def test_from_json_missing_embeddings(self):
        with pytest.raises(ValueError, match="embeddings"):
            sphereql.Pipeline.from_json('{"categories": ["a"]}')

    def test_properties(self, pipeline):
        assert pipeline.num_items == 10
        assert pipeline.categories == CATEGORIES

    def test_with_projection(self):
        pca = sphereql.PcaProjection.fit(EMBEDDINGS)
        p = sphereql.Pipeline(CATEGORIES, EMBEDDINGS, projection=pca)
        assert p.num_items == 10


# ── Nearest ─────────────────────────────────────────────────────────────

class TestNearest:
    def test_returns_k_results(self, pipeline):
        results = pipeline.nearest(QUERY, 3)
        assert len(results) == 3

    def test_sorted_by_distance(self, pipeline):
        results = pipeline.nearest(QUERY, 5)
        distances = [r.distance for r in results]
        assert distances == sorted(distances)

    def test_result_attributes(self, pipeline):
        results = pipeline.nearest(QUERY, 1)
        r = results[0]
        assert isinstance(r.id, str)
        assert isinstance(r.category, str)
        assert isinstance(r.distance, float)
        assert isinstance(r.certainty, float)
        assert isinstance(r.intensity, float)

    def test_repr(self, pipeline):
        r = pipeline.nearest(QUERY, 1)[0]
        assert r.id in repr(r)
        assert "NearestHit(" in repr(r)

    def test_json_roundtrip(self, pipeline):
        r = pipeline.nearest(QUERY, 1)[0]
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["id"] == r.id
        assert parsed["category"] == r.category

        restored = sphereql.NearestHit.from_json(j)
        assert restored.id == r.id
        assert restored.category == r.category

    def test_equality(self, pipeline):
        a = pipeline.nearest(QUERY, 1)
        b = pipeline.nearest(QUERY, 1)
        assert a[0] == b[0]

    def test_nearest_json(self, pipeline):
        j = pipeline.nearest_json(QUERY, 3)
        parsed = json.loads(j)
        assert len(parsed) == 3
        assert "id" in parsed[0]

    def test_default_k(self, pipeline):
        results = pipeline.nearest(QUERY)
        assert len(results) == 5


# ── Similar above ───────────────────────────────────────────────────────

class TestSimilarAbove:
    def test_returns_results(self, pipeline):
        results = pipeline.similar_above(QUERY, 0.5)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, sphereql.NearestHit)

    def test_json_variant(self, pipeline):
        j = pipeline.similar_above_json(QUERY, 0.5)
        parsed = json.loads(j)
        assert isinstance(parsed, list)


# ── Concept path ────────────────────────────────────────────────────────

class TestConceptPath:
    def test_finds_path(self, pipeline):
        path = pipeline.concept_path("s-0000", "s-0009", graph_k=5, query=QUERY)
        assert path is not None
        assert path.total_distance > 0
        assert len(path.steps) >= 2
        assert path.steps[0].id == "s-0000"
        assert path.steps[-1].id == "s-0009"

    def test_path_repr(self, pipeline):
        path = pipeline.concept_path("s-0000", "s-0009", graph_k=5, query=QUERY)
        assert "PathResult(" in repr(path)

    def test_path_json_roundtrip(self, pipeline):
        path = pipeline.concept_path("s-0000", "s-0009", graph_k=5, query=QUERY)
        j = path.to_json()
        parsed = json.loads(j)
        assert "total_distance" in parsed
        assert "steps" in parsed

        restored = sphereql.PathResult.from_json(j)
        assert len(restored.steps) == len(path.steps)

    def test_concept_path_json(self, pipeline):
        j = pipeline.concept_path_json("s-0000", "s-0009", graph_k=5, query=QUERY)
        parsed = json.loads(j)
        assert parsed is not None
        assert "steps" in parsed

    def test_concept_path_no_query(self, pipeline):
        path = pipeline.concept_path("s-0000", "s-0009", graph_k=5)
        assert path is not None


# ── Detect globs ────────────────────────────────────────────────────────

class TestDetectGlobs:
    def test_fixed_k(self, pipeline):
        globs = pipeline.detect_globs(k=2, max_k=5, query=QUERY)
        assert len(globs) == 2
        total = sum(g.member_count for g in globs)
        assert total == 10

    def test_auto_k(self, pipeline):
        globs = pipeline.detect_globs(max_k=5, query=QUERY)
        assert len(globs) >= 1

    def test_glob_attributes(self, pipeline):
        globs = pipeline.detect_globs(k=2, max_k=5, query=QUERY)
        g = globs[0]
        assert isinstance(g.id, int)
        assert len(g.centroid) == 3
        assert isinstance(g.member_count, int)
        assert isinstance(g.radius, float)
        assert isinstance(g.top_categories, list)

    def test_glob_json_roundtrip(self, pipeline):
        g = pipeline.detect_globs(k=2, max_k=5, query=QUERY)[0]
        j = g.to_json()
        parsed = json.loads(j)
        assert parsed["member_count"] == g.member_count

        restored = sphereql.GlobInfo.from_json(j)
        assert restored.member_count == g.member_count
        assert restored.id == g.id

    def test_detect_globs_json(self, pipeline):
        j = pipeline.detect_globs_json(k=2, max_k=5, query=QUERY)
        parsed = json.loads(j)
        assert len(parsed) == 2

    def test_detect_globs_no_query(self, pipeline):
        globs = pipeline.detect_globs(k=2, max_k=5)
        assert len(globs) == 2


# ── Local manifold ──────────────────────────────────────────────────────

class TestLocalManifold:
    def test_returns_manifold(self, pipeline):
        m = pipeline.local_manifold(QUERY, neighborhood_k=5)
        assert isinstance(m, sphereql.ManifoldInfo)
        assert len(m.centroid) == 3
        assert len(m.normal) == 3
        assert 0.0 < m.variance_ratio <= 1.0

    def test_manifold_repr(self, pipeline):
        m = pipeline.local_manifold(QUERY, neighborhood_k=5)
        assert "ManifoldInfo(" in repr(m)

    def test_manifold_json_roundtrip(self, pipeline):
        m = pipeline.local_manifold(QUERY, neighborhood_k=5)
        j = m.to_json()
        parsed = json.loads(j)
        assert "centroid" in parsed
        assert "normal" in parsed
        assert "variance_ratio" in parsed

    def test_local_manifold_json(self, pipeline):
        j = pipeline.local_manifold_json(QUERY, neighborhood_k=5)
        parsed = json.loads(j)
        assert "variance_ratio" in parsed
