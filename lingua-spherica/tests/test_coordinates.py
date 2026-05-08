"""Tests for `lingua_spherica.coordinates` — spherical math primitives."""

import math
import pytest

from lingua_spherica.coordinates import (
    angular_distance,
    circular_variance,
    circular_weighted_mean,
    geodesic_path,
    phi_distance,
    semantic_distance,
    slerp,
    spherical_centroid,
    theta_distance,
)
from lingua_spherica.types import SphericalPoint


def _sp(r: float = 1.0, theta: float = 0.0, phi: float = math.pi / 2) -> SphericalPoint:
    return SphericalPoint(r=r, theta=theta, phi=phi)


class TestAngularDistance:
    def test_self_distance_zero(self):
        p = _sp(theta=1.0, phi=1.0)
        assert angular_distance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_antipodal_distance_pi(self):
        a = SphericalPoint(r=1.0, theta=0.0, phi=0.0)
        b = SphericalPoint(r=1.0, theta=0.0, phi=math.pi)
        assert angular_distance(a, b) == pytest.approx(math.pi, abs=1e-12)

    def test_stable_for_near_identical_points(self):
        a = _sp(theta=2.0, phi=1.0)
        b = _sp(theta=2.0 + 1e-12, phi=1.0)
        d = angular_distance(a, b)
        assert d >= 0.0
        assert math.isfinite(d)


class TestThetaDistance:
    def test_shortest_path_around_circle(self):
        assert theta_distance(0.1, 2 * math.pi - 0.1) == pytest.approx(0.2, abs=1e-12)

    def test_symmetric(self):
        assert theta_distance(1.0, 4.0) == theta_distance(4.0, 1.0)

    def test_max_distance_is_pi(self):
        d = theta_distance(0.0, math.pi)
        assert d == pytest.approx(math.pi, abs=1e-12)


class TestPhiDistance:
    def test_absolute_difference(self):
        assert phi_distance(0.3, 0.7) == pytest.approx(0.4, abs=1e-12)


class TestCircularWeightedMean:
    def test_empty_returns_zero(self):
        assert circular_weighted_mean([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            circular_weighted_mean([0.1, 0.2], [1.0])

    def test_handles_seam_wraparound(self):
        # Two angles symmetric around 0: the linear mean would be ~π
        # (the wrong side), the circular mean is 0.
        m = circular_weighted_mean([0.1, 2 * math.pi - 0.1], [1.0, 1.0])
        # Result is normalized to [0, 2π); near 0 means near 0 OR near 2π.
        assert min(m, 2 * math.pi - m) == pytest.approx(0.0, abs=1e-12)

    def test_weighted_pull(self):
        # Heavier weight on the second angle should pull the mean toward it.
        unweighted = circular_weighted_mean([0.0, 1.0], [1.0, 1.0])
        weighted = circular_weighted_mean([0.0, 1.0], [1.0, 9.0])
        assert weighted > unweighted

    def test_zero_total_weight_returns_first_angle(self):
        assert circular_weighted_mean([0.5], [0.0]) == 0.5


class TestCircularVariance:
    def test_empty_zero(self):
        assert circular_variance([]) == 0.0

    def test_concentrated_low_variance(self):
        v = circular_variance([0.0, 0.001, -0.001])
        assert 0.0 <= v < 1e-5

    def test_uniform_high_variance(self):
        # Four angles at 0, π/2, π, 3π/2 — perfectly cancelling.
        v = circular_variance([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        assert v == pytest.approx(1.0, abs=1e-12)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            circular_variance([0.0, 1.0], [1.0])


class TestSlerp:
    def test_endpoints(self):
        a = _sp(theta=0.0, phi=math.pi / 2)
        b = _sp(theta=1.0, phi=math.pi / 2)
        s0 = slerp(a, b, 0.0)
        s1 = slerp(a, b, 1.0)
        assert s0.theta == pytest.approx(a.theta, abs=1e-9)
        assert s1.theta == pytest.approx(b.theta, abs=1e-9)

    def test_clamps_t_to_unit_interval(self):
        a = _sp(theta=0.0, phi=math.pi / 2)
        b = _sp(theta=1.0, phi=math.pi / 2)
        assert slerp(a, b, -0.5).theta == pytest.approx(a.theta, abs=1e-9)
        assert slerp(a, b, 1.5).theta == pytest.approx(b.theta, abs=1e-9)

    def test_midpoint_lies_between(self):
        a = _sp(theta=0.0, phi=math.pi / 2)
        b = _sp(theta=math.pi / 2, phi=math.pi / 2)
        m = slerp(a, b, 0.5)
        d_am = angular_distance(a, m)
        d_mb = angular_distance(m, b)
        assert d_am == pytest.approx(d_mb, abs=1e-9)

    def test_radius_linearly_interpolated(self):
        a = _sp(r=1.0, theta=0.0, phi=math.pi / 2)
        b = _sp(r=3.0, theta=1.0, phi=math.pi / 2)
        m = slerp(a, b, 0.5)
        assert m.r == pytest.approx(2.0, abs=1e-9)

    def test_antipodal_branch_returns_finite_point_on_sphere(self):
        north = SphericalPoint(r=1.0, theta=0.0, phi=0.0)
        south = SphericalPoint(r=1.0, theta=0.0, phi=math.pi)
        m = slerp(north, south, 0.5)
        x, y, z = m.to_cartesian()
        assert math.isfinite(x)
        assert math.isfinite(y)
        assert math.isfinite(z)
        assert math.sqrt(x * x + y * y + z * z) == pytest.approx(1.0, abs=1e-9)


class TestGeodesicPath:
    def test_minimum_two_points(self):
        with pytest.raises(ValueError, match=r"n_points.*>=\s*2"):
            geodesic_path(_sp(), _sp(theta=1.0), n_points=1)

    def test_endpoints_match(self):
        a = _sp(theta=0.0, phi=math.pi / 2)
        b = _sp(theta=1.0, phi=math.pi / 2)
        path = geodesic_path(a, b, n_points=10)
        assert len(path) == 10
        assert path[0].theta == pytest.approx(a.theta, abs=1e-9)
        assert path[-1].theta == pytest.approx(b.theta, abs=1e-9)


class TestSphericalCentroid:
    def test_empty_returns_default(self):
        c = spherical_centroid([])
        assert c.r == 0.5
        assert c.phi == pytest.approx(math.pi / 2, abs=1e-12)

    def test_single_point_recovered(self):
        pt = _sp(r=1.5, theta=1.0, phi=0.7)
        c = spherical_centroid([pt])
        assert c.theta == pytest.approx(pt.theta, abs=1e-9)
        assert c.phi == pytest.approx(pt.phi, abs=1e-9)
        assert c.r == pytest.approx(pt.r, rel=1e-9)

    def test_unit_normalization_strips_radial_bias(self):
        # Centroid direction should be the SAME whether one point is at
        # r=1 or r=100 — radii contribute to the average r, not direction
        # weighting (when weights=None, all weights are 1).
        a = _sp(r=1.0, theta=0.0, phi=math.pi / 2)
        b_small = _sp(r=1.0, theta=math.pi / 2, phi=math.pi / 2)
        b_big = _sp(r=100.0, theta=math.pi / 2, phi=math.pi / 2)
        c1 = spherical_centroid([a, b_small])
        c2 = spherical_centroid([a, b_big])
        assert c1.theta == pytest.approx(c2.theta, abs=1e-9)
        assert c1.phi == pytest.approx(c2.phi, abs=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            spherical_centroid([_sp()], weights=[1.0, 1.0])


class TestSemanticDistance:
    def test_zero_for_identical_points(self):
        p = _sp(r=1.0, theta=1.0, phi=1.0)
        assert semantic_distance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_increases_with_separation(self):
        p = _sp(r=1.0, theta=0.0, phi=math.pi / 2)
        near = _sp(r=1.0, theta=0.1, phi=math.pi / 2)
        far = _sp(r=1.0, theta=1.0, phi=math.pi / 2)
        assert semantic_distance(p, near) < semantic_distance(p, far)
