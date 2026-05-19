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


# ── Laplacian projection (standalone + pipeline-config) ─────────────────

class TestLaplacianEigenmap:
    def test_standalone_fit_and_project(self):
        proj = sphereql.LaplacianEigenmap.fit(EMBEDDINGS)
        # `dimensionality` reports input dim (4 here); output is always
        # 3 (points on S²).
        assert proj.dimensionality == 4
        assert 0.0 <= proj.connectivity_ratio <= 1.0
        assert proj.connectivity_ratio == proj.explained_variance_ratio
        ev = proj.eigenvalues
        assert len(ev) == 3
        for v in ev:
            assert -1.0 <= v <= 1.0
        # Default radial strategy is "magnitude" → r equals the input's
        # L2 norm. Use a fixed radial below if you want unit-sphere r.
        point = proj.project(QUERY)
        expected_r = sum(v * v for v in QUERY) ** 0.5
        assert abs(point.r - expected_r) < 1e-6

    def test_standalone_fixed_radial(self):
        proj = sphereql.LaplacianEigenmap.fit(EMBEDDINGS, radial=1.0)
        point = proj.project(QUERY)
        assert abs(point.r - 1.0) < 1e-6

    def test_standalone_fit_with_params(self):
        proj = sphereql.LaplacianEigenmap.fit(
            EMBEDDINGS, k_neighbors=4, active_threshold=0.01
        )
        assert proj.dimensionality == 4

    def test_standalone_project_batch(self):
        proj = sphereql.LaplacianEigenmap.fit(EMBEDDINGS)
        points = proj.project_batch(EMBEDDINGS)
        assert len(points) == len(EMBEDDINGS)

    def test_standalone_repr(self):
        proj = sphereql.LaplacianEigenmap.fit(EMBEDDINGS)
        assert "LaplacianEigenmap(" in repr(proj)

    def test_pipeline_with_laplacian_config(self):
        cfg = {"projection_kind": "LaplacianEigenmap"}
        p = sphereql.Pipeline(CATEGORIES, EMBEDDINGS, config=cfg)
        assert p.num_items == 10
        assert p.projection_kind == "laplacian_eigenmap"
        # Pipeline still queryable end-to-end.
        results = p.nearest(QUERY, 3)
        assert len(results) == 3

    def test_pipeline_with_laplacian_full_config(self):
        cfg = {
            "projection_kind": "LaplacianEigenmap",
            "laplacian": {
                "k_neighbors": 4,
                "active_threshold": 0.01,
            },
        }
        p = sphereql.Pipeline(CATEGORIES, EMBEDDINGS, config=cfg)
        assert p.projection_kind == "laplacian_eigenmap"


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


# ── Pipeline introspection ──────────────────────────────────────────────

class TestPipelineIntrospection:
    def test_exported_points(self, pipeline):
        points = pipeline.exported_points()
        assert len(points) == 10
        p = points[0]
        for key in ("id", "category", "r", "theta", "phi", "x", "y", "z",
                    "certainty", "intensity"):
            assert key in p

    def test_unique_categories(self, pipeline):
        cats = pipeline.unique_categories()
        assert set(cats) == {"science", "cooking"}

    def test_config_round_trips(self, pipeline):
        cfg = pipeline.config()
        assert isinstance(cfg, dict)
        assert "projection_kind" in cfg

    def test_projection_kind(self, pipeline):
        assert pipeline.projection_kind in ("pca", "kernel_pca", "laplacian_eigenmap")

    def test_explained_variance_ratio(self, pipeline):
        evr = pipeline.explained_variance_ratio
        assert 0.0 < evr <= 1.0

    def test_to_json(self, pipeline):
        j = pipeline.to_json()
        assert isinstance(j, str)
        assert len(j) > 0

    def test_to_csv(self, pipeline):
        csv = pipeline.to_csv()
        assert isinstance(csv, str)
        lines = csv.strip().split("\n")
        assert len(lines) >= 2  # header + at least one row

    def test_projection_warnings(self, pipeline):
        warnings = pipeline.projection_warnings()
        assert isinstance(warnings, list)
        for w in warnings:
            assert w.severity in ("Info", "Warning", "Critical")
            assert isinstance(w.evr, float)
            assert isinstance(w.message, str)

    def test_domain_groups(self, pipeline):
        groups = pipeline.domain_groups()
        assert isinstance(groups, list)
        total_items = sum(g.total_items for g in groups)
        assert total_items == pipeline.num_items

    def test_hierarchical_nearest(self, pipeline):
        results = pipeline.hierarchical_nearest(QUERY, k=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, sphereql.NearestHit)


# ── Category enrichment ─────────────────────────────────────────────────

class TestCategoryEnrichment:
    def test_category_stats(self, pipeline):
        summaries, inner_reports = pipeline.category_stats()
        assert len(summaries) == 2
        for s in summaries:
            assert s.name in ("science", "cooking")
            assert s.member_count == 5
            assert 0.0 <= s.cohesion <= 1.0
            assert s.bridge_quality >= 0.0
        assert isinstance(inner_reports, list)

    def test_category_neighbors(self, pipeline):
        neighbors = pipeline.category_neighbors("science", k=1)
        assert len(neighbors) >= 1
        assert all(hasattr(n, "name") for n in neighbors)

    def test_category_concept_path(self, pipeline):
        path = pipeline.category_concept_path("science", "cooking")
        # may be None if categories are disconnected on this small corpus
        if path is not None:
            assert path.total_distance > 0
            assert len(path.steps) >= 1
            assert 0.0 <= path.path_confidence <= 1.0

    def test_drill_down(self, pipeline):
        hits = pipeline.drill_down("science", QUERY, k=3)
        assert len(hits) == 3
        for h in hits:
            assert isinstance(h.item_index, int)
            assert isinstance(h.distance, float)
            assert isinstance(h.used_inner_sphere, bool)

    def test_dimension_mismatch_raises(self, pipeline):
        with pytest.raises(ValueError, match="dimension mismatch"):
            pipeline.nearest([0.1, 0.2], k=3)


# ── Navigator ───────────────────────────────────────────────────────────

class TestNavigator:
    def test_run_navigator_default_config(self, pipeline):
        report = sphereql.run_navigator(pipeline)
        assert report.num_items == pipeline.num_items
        assert report.num_categories == 2
        assert 0.0 <= report.coverage.coverage_fraction <= 1.0
        assert isinstance(report.antipodal, list)
        assert isinstance(report.voronoi, list)
        assert isinstance(report.lunes, list)

    def test_navigator_config(self, pipeline):
        cfg = sphereql.NavigatorConfig(
            coverage_samples=1000,
            voronoi_samples=1000,
        )
        report = sphereql.run_navigator(pipeline, config=cfg)
        assert report.num_items == pipeline.num_items

    def test_navigator_report_evr(self, pipeline):
        report = sphereql.run_navigator(pipeline)
        assert report.explained_variance_ratio == pytest.approx(
            pipeline.explained_variance_ratio, rel=1e-6
        )


# ── Corpus features + Auto-tune ─────────────────────────────────────────

class TestMetalearning:
    def test_corpus_features(self):
        features = sphereql.corpus_features(CATEGORIES, EMBEDDINGS)
        assert isinstance(features, dict)
        expected_keys = (
            "n_items", "n_categories", "dim",
            "mean_members_per_category", "category_size_entropy",
        )
        for key in expected_keys:
            assert key in features
        assert features["n_items"] == 10
        assert features["n_categories"] == 2

    def test_auto_tune_returns_pipeline_and_report(self):
        pipeline, report = sphereql.auto_tune(
            CATEGORIES, EMBEDDINGS,
            metric="territorial_health",
            strategy="random",
            budget=2,
            seed=0,
        )
        assert isinstance(pipeline, sphereql.Pipeline)
        assert pipeline.num_items == 10
        assert "best_score" in report
        assert "best_config" in report
        assert "trials_count" in report
        assert report["trials_count"] >= 1

    def test_auto_tune_invalid_metric(self):
        with pytest.raises(ValueError, match="unknown metric"):
            sphereql.auto_tune(CATEGORIES, EMBEDDINGS, metric="invalid_metric")

    def test_nearest_neighbor_meta_model(self):
        features = sphereql.corpus_features(CATEGORIES, EMBEDDINGS)
        _, report = sphereql.auto_tune(
            CATEGORIES, EMBEDDINGS,
            strategy="random",
            budget=2,
            seed=0,
        )
        record = {
            "corpus_id": "test",
            "features": features,
            "best_config": report["best_config"],
            "best_score": report["best_score"],
            "metric_name": report["metric_name"],
            "strategy": "random",
            "timestamp": "0",
        }
        model = sphereql.NearestNeighborMetaModel()
        model.fit([record])
        predicted = model.predict(features)
        assert isinstance(predicted, dict)
        assert "projection_kind" in predicted

    def test_distance_weighted_meta_model(self):
        features = sphereql.corpus_features(CATEGORIES, EMBEDDINGS)
        _, report = sphereql.auto_tune(
            CATEGORIES, EMBEDDINGS,
            strategy="random",
            budget=2,
            seed=0,
        )
        record = {
            "corpus_id": "test",
            "features": features,
            "best_config": report["best_config"],
            "best_score": report["best_score"],
            "metric_name": report["metric_name"],
            "strategy": "random",
            "timestamp": "0",
        }
        model = sphereql.DistanceWeightedMetaModel(epsilon=0.1)
        model.fit([record])
        predicted = model.predict(features)
        assert isinstance(predicted, dict)
        assert "projection_kind" in predicted


# ── Feedback ────────────────────────────────────────────────────────────

class TestFeedback:
    def test_feedback_event_creation(self):
        event = sphereql.FeedbackEvent("corpus-1", "query-1", 0.85)
        assert event.corpus_id == "corpus-1"
        assert event.query_id == "query-1"
        assert event.score == pytest.approx(0.85)
        assert isinstance(event.timestamp, str)

    def test_feedback_event_with_timestamp(self):
        event = sphereql.FeedbackEvent("c", "q", 0.5, timestamp="2024-01-01T00:00:00Z")
        assert event.timestamp == "2024-01-01T00:00:00Z"

    def test_feedback_event_repr(self):
        event = sphereql.FeedbackEvent("corpus-1", "query-1", 0.85)
        assert "FeedbackEvent(" in repr(event)
        assert "corpus-1" in repr(event)

    def test_feedback_event_to_dict(self):
        event = sphereql.FeedbackEvent("c", "q", 0.75)
        d = event.to_dict()
        assert isinstance(d, dict)
        assert d["corpus_id"] == "c"
        assert d["score"] == pytest.approx(0.75)

    def test_feedback_aggregator(self):
        agg = sphereql.FeedbackAggregator()
        assert len(agg) == 0
        assert not bool(agg)

        agg.record(sphereql.FeedbackEvent("corp", "q1", 0.9))
        agg.record(sphereql.FeedbackEvent("corp", "q2", 0.7))
        assert len(agg) == 2
        assert bool(agg)

        ids = agg.corpus_ids()
        assert "corp" in ids

        summary = agg.summarize("corp")
        assert summary is not None
        assert summary["n_events"] == 2
        assert summary["mean_score"] == pytest.approx(0.8)

    def test_feedback_aggregator_missing_corpus(self):
        agg = sphereql.FeedbackAggregator()
        assert agg.summarize("nonexistent") is None

    def test_feedback_aggregator_summarize_all(self):
        agg = sphereql.FeedbackAggregator()
        agg.record(sphereql.FeedbackEvent("a", "q1", 0.5))
        agg.record(sphereql.FeedbackEvent("b", "q1", 0.9))
        all_summaries = agg.summarize_all()
        assert isinstance(all_summaries, dict)
        assert set(all_summaries.keys()) == {"a", "b"}
