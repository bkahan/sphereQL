"""Tests for the sphereql Python module.

Run with:
    cd sphereql-python
    maturin develop
    pytest tests/
"""
import math
import numpy as np
import pytest

import sphereql


# ── Core types ──────────────────────────────────────────────────────────

class TestSphericalPoint:
    def test_creation(self):
        p = sphereql.SphericalPoint(1.0, 0.5, 1.0)
        assert p.r == 1.0
        assert p.theta == 0.5
        assert p.phi == 1.0

    def test_validation_negative_r(self):
        with pytest.raises(ValueError):
            sphereql.SphericalPoint(-1.0, 0.5, 1.0)

    def test_validation_theta_out_of_range(self):
        with pytest.raises(ValueError):
            sphereql.SphericalPoint(1.0, 7.0, 1.0)  # theta must be in [0, 2*pi)

    def test_validation_phi_out_of_range(self):
        with pytest.raises(ValueError):
            sphereql.SphericalPoint(1.0, 0.5, 4.0)  # phi must be in [0, pi]

    def test_repr(self):
        p = sphereql.SphericalPoint(1.0, 0.5, 1.0)
        r = repr(p)
        assert "SphericalPoint(" in r
        assert "r=1" in r

    def test_equality(self):
        a = sphereql.SphericalPoint(1.0, 0.5, 1.0)
        b = sphereql.SphericalPoint(1.0, 0.5, 1.0)
        assert a == b


class TestCartesianPoint:
    def test_creation(self):
        p = sphereql.CartesianPoint(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_magnitude(self):
        p = sphereql.CartesianPoint(3.0, 4.0, 0.0)
        assert abs(p.magnitude() - 5.0) < 1e-10

    def test_normalize(self):
        p = sphereql.CartesianPoint(3.0, 4.0, 0.0)
        n = p.normalize()
        assert abs(n.magnitude() - 1.0) < 1e-10

    def test_repr(self):
        p = sphereql.CartesianPoint(1.0, 2.0, 3.0)
        assert "CartesianPoint(" in repr(p)

    def test_equality(self):
        a = sphereql.CartesianPoint(1.0, 2.0, 3.0)
        b = sphereql.CartesianPoint(1.0, 2.0, 3.0)
        assert a == b


class TestGeoPoint:
    def test_creation(self):
        p = sphereql.GeoPoint(45.0, 90.0, 100.0)
        assert p.lat == 45.0
        assert p.lon == 90.0
        assert p.alt == 100.0

    def test_validation_invalid_lat(self):
        with pytest.raises(ValueError):
            sphereql.GeoPoint(91.0, 0.0, 0.0)

    def test_validation_invalid_lon(self):
        with pytest.raises(ValueError):
            sphereql.GeoPoint(0.0, 181.0, 0.0)

    def test_repr(self):
        p = sphereql.GeoPoint(45.0, 90.0, 100.0)
        assert "GeoPoint(" in repr(p)


# ── Roundtrip conversions ───────────────────────────────────────────────

class TestConversions:
    def test_roundtrip_spherical_cartesian(self):
        p = sphereql.SphericalPoint(2.0, 1.0, 0.5)
        cart = p.to_cartesian()
        back = cart.to_spherical()
        assert abs(back.r - p.r) < 1e-10
        assert abs(back.theta - p.theta) < 1e-10
        assert abs(back.phi - p.phi) < 1e-10

    def test_roundtrip_spherical_geo(self):
        p = sphereql.SphericalPoint(1.0, 1.0, 0.5)
        geo = p.to_geo()
        back = geo.to_spherical()
        assert abs(back.r - p.r) < 1e-10
        assert abs(back.theta - p.theta) < 1e-10
        assert abs(back.phi - p.phi) < 1e-10

    def test_function_spherical_to_cartesian(self):
        p = sphereql.SphericalPoint(1.0, math.pi / 2, 0.0)
        c = sphereql.spherical_to_cartesian(p)
        assert isinstance(c, sphereql.CartesianPoint)

    def test_function_cartesian_to_spherical(self):
        c = sphereql.CartesianPoint(1.0, 0.0, 0.0)
        p = sphereql.cartesian_to_spherical(c)
        assert isinstance(p, sphereql.SphericalPoint)

    def test_function_spherical_to_geo(self):
        p = sphereql.SphericalPoint(1.0, 1.0, 0.5)
        g = sphereql.spherical_to_geo(p)
        assert isinstance(g, sphereql.GeoPoint)

    def test_function_geo_to_spherical(self):
        g = sphereql.GeoPoint(45.0, 90.0, 0.0)
        p = sphereql.geo_to_spherical(g)
        assert isinstance(p, sphereql.SphericalPoint)


# ── Distance functions ──────────────────────────────────────────────────

class TestDistances:
    def test_angular_distance_same_point(self):
        p = sphereql.SphericalPoint(1.0, 1.0, 1.0)
        assert abs(sphereql.angular_distance(p, p)) < 1e-10

    def test_angular_distance_opposite(self):
        a = sphereql.SphericalPoint(1.0, 0.01, 0.0)
        b = sphereql.SphericalPoint(1.0, math.pi - 0.01, math.pi)
        d = sphereql.angular_distance(a, b)
        assert abs(d - math.pi) < 0.1

    def test_great_circle_distance(self):
        a = sphereql.SphericalPoint(1.0, math.pi / 2, 0.0)
        b = sphereql.SphericalPoint(1.0, math.pi / 2, math.pi / 2)
        d = sphereql.great_circle_distance(a, b, 1.0)
        assert abs(d - math.pi / 2) < 1e-10

    def test_chord_distance_antipodal(self):
        a = sphereql.SphericalPoint(1.0, 0.01, 0.0)
        b = sphereql.SphericalPoint(1.0, math.pi - 0.01, math.pi)
        d = sphereql.chord_distance(a, b)
        assert abs(d - 2.0) < 0.1


# ── Projections ─────────────────────────────────────────────────────────

class TestPcaProjection:
    @pytest.fixture
    def embeddings(self):
        np.random.seed(42)
        return np.random.randn(20, 64)

    @pytest.fixture
    def pca(self, embeddings):
        return sphereql.PcaProjection.fit(embeddings)

    def test_fit_and_project(self, pca, embeddings):
        pt = pca.project(embeddings[0])
        assert isinstance(pt, sphereql.SphericalPoint)
        assert pt.r > 0
        assert 0 <= pt.theta < 2 * math.pi  # theta in [0, 2*pi)
        assert 0 <= pt.phi <= math.pi        # phi in [0, pi]

    def test_project_rich(self, pca, embeddings):
        rpt = pca.project_rich(embeddings[0])
        assert isinstance(rpt, sphereql.ProjectedPoint)
        assert 0 <= rpt.certainty <= 1

    def test_batch(self, pca, embeddings):
        pts = pca.project_batch(embeddings[:5])
        assert len(pts) == 5
        for pt in pts:
            assert isinstance(pt, sphereql.SphericalPoint)

    def test_rich_batch(self, pca, embeddings):
        rpts = pca.project_rich_batch(embeddings[:5])
        assert len(rpts) == 5
        for rpt in rpts:
            assert isinstance(rpt, sphereql.ProjectedPoint)

    def test_explained_variance(self, pca):
        assert 0 < pca.explained_variance_ratio <= 1.0

    def test_dimensionality(self, pca):
        assert pca.dimensionality == 64

    def test_volumetric(self, embeddings):
        pca_normal = sphereql.PcaProjection.fit(embeddings)
        pca_vol = sphereql.PcaProjection.fit(embeddings, volumetric=True)
        pt_normal = pca_normal.project(embeddings[0])
        pt_vol = pca_vol.project(embeddings[0])
        # volumetric may change r (or not, depending on data), but
        # both should be valid SphericalPoints
        assert isinstance(pt_vol, sphereql.SphericalPoint)

    def test_radial_fixed(self, embeddings):
        pca = sphereql.PcaProjection.fit(embeddings, radial=2.5)
        pt = pca.project(embeddings[0])
        assert abs(pt.r - 2.5) < 1e-10

    def test_radial_magnitude(self, embeddings):
        pca = sphereql.PcaProjection.fit(embeddings, radial="magnitude")
        pt = pca.project(embeddings[0])
        assert pt.r > 0

    def test_invalid_radial(self, embeddings):
        with pytest.raises(ValueError, match="unknown radial strategy"):
            sphereql.PcaProjection.fit(embeddings, radial="invalid")

    def test_repr(self, pca):
        r = repr(pca)
        assert "PcaProjection(" in r
        assert "dim=" in r


class TestKernelPcaProjection:
    @pytest.fixture
    def embeddings(self):
        np.random.seed(42)
        return np.random.randn(20, 64)

    @pytest.fixture
    def kpca(self, embeddings):
        return sphereql.KernelPcaProjection.fit(embeddings)

    def test_fit_and_project(self, kpca, embeddings):
        pt = kpca.project(embeddings[0])
        assert isinstance(pt, sphereql.SphericalPoint)
        assert pt.r > 0
        assert 0 <= pt.theta < 2 * math.pi
        assert 0 <= pt.phi <= math.pi

    def test_project_rich(self, kpca, embeddings):
        rpt = kpca.project_rich(embeddings[0])
        assert isinstance(rpt, sphereql.ProjectedPoint)
        assert 0 <= rpt.certainty <= 1

    def test_batch(self, kpca, embeddings):
        pts = kpca.project_batch(embeddings[:5])
        assert len(pts) == 5
        for pt in pts:
            assert isinstance(pt, sphereql.SphericalPoint)

    def test_rich_batch(self, kpca, embeddings):
        rpts = kpca.project_rich_batch(embeddings[:5])
        assert len(rpts) == 5
        for rpt in rpts:
            assert isinstance(rpt, sphereql.ProjectedPoint)

    def test_explained_variance(self, kpca):
        assert 0 < kpca.explained_variance_ratio <= 1.0

    def test_dimensionality(self, kpca):
        assert kpca.dimensionality == 64

    def test_sigma(self, kpca):
        assert kpca.sigma > 0

    def test_num_training_points(self, kpca):
        assert kpca.num_training_points == 20

    def test_explicit_sigma(self, embeddings):
        kpca = sphereql.KernelPcaProjection.fit(embeddings, sigma=0.5)
        assert abs(kpca.sigma - 0.5) < 1e-12

    def test_volumetric(self, embeddings):
        kpca_vol = sphereql.KernelPcaProjection.fit(embeddings, volumetric=True)
        pt = kpca_vol.project(embeddings[0])
        assert isinstance(pt, sphereql.SphericalPoint)

    def test_radial_fixed(self, embeddings):
        kpca = sphereql.KernelPcaProjection.fit(embeddings, radial=2.5)
        pt = kpca.project(embeddings[0])
        assert abs(pt.r - 2.5) < 1e-10

    def test_radial_magnitude(self, embeddings):
        kpca = sphereql.KernelPcaProjection.fit(embeddings, radial="magnitude")
        pt = kpca.project(embeddings[0])
        assert pt.r > 0

    def test_invalid_radial(self, embeddings):
        with pytest.raises(ValueError, match="unknown radial strategy"):
            sphereql.KernelPcaProjection.fit(embeddings, radial="invalid")

    def test_repr(self, kpca):
        r = repr(kpca)
        assert "KernelPcaProjection(" in r
        assert "sigma=" in r

    def test_out_of_sample(self, kpca):
        np.random.seed(99)
        new_vec = np.random.randn(64)
        pt = kpca.project(new_vec)
        assert isinstance(pt, sphereql.SphericalPoint)

    def test_f32_input(self):
        embs = np.random.randn(20, 64).astype(np.float32)
        kpca = sphereql.KernelPcaProjection.fit(embs)
        pt = kpca.project(embs[0])
        assert isinstance(pt, sphereql.SphericalPoint)

    def test_list_input(self, kpca, embeddings):
        vec_list = [float(x) for x in embeddings[0]]
        pt = kpca.project(vec_list)
        assert isinstance(pt, sphereql.SphericalPoint)


class TestRandomProjection:
    def test_deterministic(self):
        rp1 = sphereql.RandomProjection(64, seed=123)
        rp2 = sphereql.RandomProjection(64, seed=123)
        vec = [float(i) for i in range(64)]
        pt1 = rp1.project(vec)
        pt2 = rp2.project(vec)
        assert pt1 == pt2

    def test_different_seeds(self):
        rp1 = sphereql.RandomProjection(64, seed=1)
        rp2 = sphereql.RandomProjection(64, seed=2)
        vec = [float(i) for i in range(64)]
        pt1 = rp1.project(vec)
        pt2 = rp2.project(vec)
        assert pt1 != pt2

    def test_dimensionality(self):
        rp = sphereql.RandomProjection(128)
        assert rp.dimensionality == 128

    def test_project_rich(self):
        rp = sphereql.RandomProjection(64)
        vec = [float(i) for i in range(64)]
        rpt = rp.project_rich(vec)
        assert isinstance(rpt, sphereql.ProjectedPoint)

    def test_batch(self):
        rp = sphereql.RandomProjection(64)
        embs = np.random.randn(5, 64)
        pts = rp.project_batch(embs)
        assert len(pts) == 5

    def test_repr(self):
        rp = sphereql.RandomProjection(64)
        assert "RandomProjection(dim=64)" in repr(rp)


# ── Numpy integration ──────────────────────────────────────────────────

class TestNumpyIntegration:
    def test_f32_upcast(self):
        embs = np.random.randn(20, 64).astype(np.float32)
        pca = sphereql.PcaProjection.fit(embs)
        pt = pca.project(embs[0])
        assert isinstance(pt, sphereql.SphericalPoint)

    def test_list_input(self):
        embs = np.random.randn(20, 64)
        pca = sphereql.PcaProjection.fit(embs)
        vec_list = [float(x) for x in embs[0]]
        pt = pca.project(vec_list)
        assert isinstance(pt, sphereql.SphericalPoint)

    def test_pipeline_numpy_input(self):
        cats = ["a"] * 10 + ["b"] * 10
        embs = np.random.randn(20, 32)
        p = sphereql.Pipeline(cats, embs)
        assert p.num_items == 20

    def test_pipeline_f32_input(self):
        cats = ["a"] * 10 + ["b"] * 10
        embs = np.random.randn(20, 32).astype(np.float32)
        p = sphereql.Pipeline(cats, embs)
        assert p.num_items == 20
