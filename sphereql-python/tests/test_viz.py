import os
import tempfile

import pytest
import sphereql


DIM = 10
N = 20


def make_data(n=N, dim=DIM):
    categories = []
    embeddings = []
    for i in range(n):
        v = [0.0] * dim
        if i < n // 2:
            v[0] = 1.0 + i * 0.01
            v[1] = 0.1
            categories.append("science")
        else:
            v[0] = 0.1
            v[1] = 1.0 + i * 0.01
            categories.append("cooking")
        v[2] = 0.05 * i
        embeddings.append(v)
    return categories, embeddings


class TestVisualize:
    def test_creates_file(self, tmp_path):
        cats, embs = make_data()
        out = str(tmp_path / "test_viz.html")
        result = sphereql.visualize(cats, embs, output=out, open_browser=False)
        assert os.path.exists(out)
        assert result.endswith("test_viz.html")
        content = open(out).read()
        assert "<script" in content
        assert "THREE" in content

    def test_categories_in_output(self, tmp_path):
        cats, embs = make_data()
        out = str(tmp_path / "test_cats.html")
        sphereql.visualize(cats, embs, output=out, open_browser=False)
        content = open(out).read()
        assert "science" in content
        assert "cooking" in content

    def test_stats_panel(self, tmp_path):
        cats, embs = make_data()
        out = str(tmp_path / "test_stats.html")
        sphereql.visualize(cats, embs, output=out, open_browser=False)
        content = open(out).read()
        assert "evr" in content
        assert "PCA variance" in content

    def test_with_labels(self, tmp_path):
        cats, embs = make_data()
        labels = [f"item-{i}" for i in range(N)]
        out = str(tmp_path / "test_labels.html")
        sphereql.visualize(
            cats, embs, output=out, labels=labels, open_browser=False
        )
        content = open(out).read()
        assert "item-0" in content

    def test_with_title(self, tmp_path):
        cats, embs = make_data()
        out = str(tmp_path / "test_title.html")
        sphereql.visualize(
            cats, embs, output=out, title="My Test Sphere", open_browser=False
        )
        content = open(out).read()
        assert "My Test Sphere" in content


class TestVisualizePipeline:
    def test_creates_file(self, tmp_path):
        cats, embs = make_data()
        pipeline = sphereql.Pipeline(cats, embs)
        out = str(tmp_path / "test_pipeline_viz.html")
        result = sphereql.visualize_pipeline(
            pipeline, output=out, open_browser=False
        )
        assert os.path.exists(out)
        assert result.endswith("test_pipeline_viz.html")
        content = open(out).read()
        assert "<script" in content
        assert "THREE" in content
