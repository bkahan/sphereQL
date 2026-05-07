"""Tests for `lingua_spherica.types` — point/concept/graph primitives."""

import math
import pytest

from lingua_spherica.types import (
    Concept,
    ConceptGraph,
    DomainAnchor,
    Relation,
    RelationType,
    SphericalPoint,
)


def _sp(r=1.0, theta=0.0, phi=math.pi / 2):
    return SphericalPoint(r=r, theta=theta, phi=phi)


class TestSphericalPoint:
    def test_field_order_matches_rust(self):
        # Rust SphericalPoint::new(r, theta, phi) — positional args must agree.
        p = SphericalPoint(0.5, 1.0, 1.5)
        assert p.r == 0.5
        assert p.theta == 1.0
        assert p.phi == 1.5

    def test_theta_normalized_into_unit_circle(self):
        p = SphericalPoint(r=1.0, theta=3 * math.pi, phi=math.pi / 2)
        assert 0.0 <= p.theta < 2 * math.pi
        assert math.isclose(p.theta, math.pi, abs_tol=1e-12)

    def test_phi_clamped_to_polar_range(self):
        assert SphericalPoint(r=1.0, theta=0.0, phi=-1.0).phi == 0.0
        assert SphericalPoint(r=1.0, theta=0.0, phi=4.0).phi == math.pi

    def test_nonpositive_radius_rejected(self):
        with pytest.raises(ValueError):
            SphericalPoint(r=0.0, theta=0.0, phi=0.0)
        with pytest.raises(ValueError):
            SphericalPoint(r=-1.0, theta=0.0, phi=0.0)

    def test_nonfinite_inputs_rejected(self):
        with pytest.raises(ValueError):
            SphericalPoint(r=float("nan"), theta=0.0, phi=0.0)
        with pytest.raises(ValueError):
            SphericalPoint(r=1.0, theta=float("inf"), phi=0.0)

    def test_cartesian_roundtrip_preserves_direction(self):
        original = SphericalPoint(r=1.7, theta=1.3, phi=0.7)
        x, y, z = original.to_cartesian()
        recovered = SphericalPoint.from_cartesian(x, y, z)
        assert math.isclose(recovered.r, original.r, rel_tol=1e-12)
        assert math.isclose(recovered.theta, original.theta, abs_tol=1e-12)
        assert math.isclose(recovered.phi, original.phi, abs_tol=1e-12)

    def test_from_cartesian_at_origin_uses_radius_floor(self):
        p = SphericalPoint.from_cartesian(0.0, 0.0, 0.0)
        # Floored at 1e-6 (matches __post_init__ floor) rather than 0.0.
        assert p.r == pytest.approx(1e-6)

    def test_angular_distance_self_is_zero(self):
        p = _sp(theta=1.2, phi=0.8)
        assert p.angular_distance_to(p) == pytest.approx(0.0, abs=1e-12)

    def test_angular_distance_antipodal_is_pi(self):
        north = SphericalPoint(r=1.0, theta=0.0, phi=0.0)
        south = SphericalPoint(r=1.0, theta=0.0, phi=math.pi)
        assert north.angular_distance_to(south) == pytest.approx(math.pi, abs=1e-12)

    def test_angular_distance_orthogonal_pair_is_half_pi(self):
        a = SphericalPoint(r=1.0, theta=0.0, phi=math.pi / 2)
        b = SphericalPoint(r=1.0, theta=math.pi / 2, phi=math.pi / 2)
        assert a.angular_distance_to(b) == pytest.approx(math.pi / 2, abs=1e-12)

    def test_angular_distance_stable_near_identical(self):
        # acos-form would lose precision here; Vincenty stays well-conditioned.
        a = SphericalPoint(r=1.0, theta=1.0, phi=1.0)
        b = SphericalPoint(r=1.0, theta=1.0 + 1e-9, phi=1.0)
        d = a.angular_distance_to(b)
        assert d > 0.0
        assert d < 1e-8


class TestConcept:
    def test_hash_and_equality_use_normalized_form(self):
        a = Concept(text="Cat", normalized="cat")
        b = Concept(text="cats", normalized="cat", frequency=99)
        assert a == b
        assert hash(a) == hash(b)
        assert a != Concept(text="dog", normalized="dog")
        assert a != "cat"


class TestRelation:
    def test_geodesic_length_none_when_unresolved(self):
        a = Concept(text="a", normalized="a")
        b = Concept(text="b", normalized="b")
        r = Relation(source=a, target=b, relation_type=RelationType.IS_A)
        assert r.geodesic_length() is None

    def test_geodesic_length_matches_angular_distance(self):
        a = Concept(text="a", normalized="a", point=_sp(theta=0.0, phi=math.pi / 2))
        b = Concept(text="b", normalized="b", point=_sp(theta=math.pi / 2, phi=math.pi / 2))
        r = Relation(source=a, target=b, relation_type=RelationType.RELATED_TO)
        assert r.geodesic_length() == pytest.approx(math.pi / 2, abs=1e-12)


class TestConceptGraph:
    def test_centroid_none_for_unresolved_graph(self):
        g = ConceptGraph(concepts=[Concept(text="a", normalized="a")])
        assert g.centroid() is None

    def test_centroid_of_single_point_returns_that_direction(self):
        pt = _sp(r=2.0, theta=1.0, phi=1.0)
        g = ConceptGraph(concepts=[Concept(text="a", normalized="a", point=pt)])
        c = g.centroid()
        assert c is not None
        assert math.isclose(c.theta, pt.theta, abs_tol=1e-9)
        assert math.isclose(c.phi, pt.phi, abs_tol=1e-9)
        assert math.isclose(c.r, pt.r, rel_tol=1e-9)

    def test_centroid_of_antipodal_pair_is_undefined(self):
        north = _sp(r=1.0, theta=0.0, phi=0.0)
        south = _sp(r=1.0, theta=0.0, phi=math.pi)
        g = ConceptGraph(
            concepts=[
                Concept(text="n", normalized="n", point=north),
                Concept(text="s", normalized="s", point=south),
            ]
        )
        # Resultant vector cancels: centroid direction is undefined.
        assert g.centroid() is None

    def test_centroid_weights_by_radius(self):
        # Two equator points at θ=0 and θ=π/2. With equal weights the
        # centroid lies at θ=π/4. Weighting the second 9× heavier should
        # pull the centroid toward π/2.
        a = _sp(r=1.0, theta=0.0, phi=math.pi / 2)
        b = _sp(r=9.0, theta=math.pi / 2, phi=math.pi / 2)
        g = ConceptGraph(
            concepts=[
                Concept(text="a", normalized="a", point=a),
                Concept(text="b", normalized="b", point=b),
            ]
        )
        c = g.centroid()
        assert c is not None
        # θ should be in (π/4, π/2) — closer to b than the unweighted mean.
        assert c.theta > math.pi / 4 + 1e-3
        assert c.theta < math.pi / 2

    def test_neighbors_returns_both_directions(self):
        a = Concept(text="a", normalized="a")
        b = Concept(text="b", normalized="b")
        c = Concept(text="c", normalized="c")
        g = ConceptGraph(
            concepts=[a, b, c],
            relations=[
                Relation(source=a, target=b, relation_type=RelationType.IS_A),
                Relation(source=c, target=a, relation_type=RelationType.PART_OF),
            ],
        )
        names = sorted(n.normalized for n, _ in g.neighbors(a))
        assert names == ["b", "c"]


class TestDomainAnchor:
    def test_zero_or_negative_width_rejected(self):
        with pytest.raises(ValueError):
            DomainAnchor(name="x", theta=0.0, angular_width=0.0)
        with pytest.raises(ValueError):
            DomainAnchor(name="x", theta=0.0, angular_width=-0.1)

    def test_full_circle_width_rejected(self):
        with pytest.raises(ValueError):
            DomainAnchor(name="x", theta=0.0, angular_width=2 * math.pi)

    def test_theta_range_normalized(self):
        a = DomainAnchor(name="x", theta=1.0, angular_width=0.4)
        lo, hi = a.theta_range
        assert 0.0 <= lo < 2 * math.pi
        assert 0.0 <= hi < 2 * math.pi

    def test_theta_range_wraps_at_seam(self):
        # Anchor centered at θ=0 with width 0.4 spans (-0.2, 0.2) raw,
        # which after normalization wraps: lo near 2π, hi near 0.2.
        a = DomainAnchor(name="seam", theta=0.0, angular_width=0.4)
        lo, hi = a.theta_range
        assert lo > hi  # documented wrap signal
        assert lo == pytest.approx(2 * math.pi - 0.2, abs=1e-12)
        assert hi == pytest.approx(0.2, abs=1e-12)
