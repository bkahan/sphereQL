"""Cross-language parity: ``lingua_spherica`` (pure Python) vs ``sphereql`` (Rust).

``lingua_spherica`` is a hand-written Python reimplementation of the spherical
coordinate math in ``sphereql-core``. The two surfaces are meant to compute
*bit-for-bit equivalent* results so a corpus projected by one can be reasoned
about by the other. This module is the automated tripwire that fails the moment
they diverge.

The Rust ``sphereql`` wheel is OPTIONAL here. When it is absent (or was built
without the ``core`` feature, so the coordinate twins are missing) every test in
this file skips. The pure-Python math suite in ``test_coordinates.py`` /
``test_types.py`` is what actually gates ``lingua_spherica`` correctness; this
file only adds the cross-language check on top when the wheel is available.

Tier 1 covers the functions that have a Python-reachable Rust twin in the
compiled wheel: ``angular_distance``, ``SphericalPoint`` construction, and the
spherical<->cartesian conversions. Tier 2 documents the functions whose Rust
twin is *not* exposed to Python yet (slerp, centroid, the circular stats, ...);
those are explicit skips so the gap is visible rather than silently omitted.

Domain note: the Rust ``SphericalPoint`` constructor VALIDATES and REJECTS
out-of-range inputs (theta must be in ``[0, 2*pi)``, phi in ``[0, pi]``, r >= 0),
whereas ``lingua_spherica.SphericalPoint`` NORMALIZES (theta % 2*pi, phi clamped
to ``[0, pi]``, r floored at 1e-6). Empirically verified against the installed
wheel. The shared ``CASES`` therefore live in the intersection of both accepted
domains (already normalized, r comfortably positive) so a single tuple feeds
both constructors without one raising. The seam value sits just below 2*pi for
the same reason.
"""

from __future__ import annotations

import math
import random

import pytest

# Optional dependency: the Rust wheel. If it is not installed at all, skip the
# whole module — this file must never break the pure-Python suite.
sphereql = pytest.importorskip("sphereql")

# The wheel can be built without the `core` feature, in which case the
# coordinate twins this file compares against are simply absent. Skip rather
# than error so a feature-trimmed wheel still lets the rest of the suite run.
if not hasattr(sphereql, "angular_distance"):
    pytest.skip(
        "sphereql wheel built without the `core` feature; "
        "no Python-reachable coordinate twins to compare against",
        allow_module_level=True,
    )

from lingua_spherica.coordinates import angular_distance as lingua_angular_distance
from lingua_spherica.types import SphericalPoint as LinguaPoint

TWO_PI = 2.0 * math.pi
PI = math.pi

# Deterministic shared cases as raw (r, theta, phi) tuples. Each must satisfy:
#   r  > 0                  (lingua rejects r <= 0; Rust rejects r < 0)
#   theta in [0, 2*pi)      (Rust rejects theta == 2*pi exactly)
#   phi   in [0, pi]        (both endpoints accepted)
# so the SAME tuple constructs cleanly under both implementations with no
# normalization divergence to mask a real bug.
_FIXED_CASES: list[tuple[float, float, float]] = [
    # Equator (phi = pi/2), spread around the seam.
    (1.0, 0.0, PI / 2),
    (1.0, PI / 2, PI / 2),
    (1.0, PI, PI / 2),
    (1.0, 3 * PI / 2, PI / 2),
    # Poles — theta is ill-defined here; the test guards cartesian theta
    # comparisons accordingly, but distance/construction must still agree.
    (1.0, 0.0, 0.0),  # north pole
    (1.0, 1.3, 0.0),  # north pole, different (irrelevant) theta
    (1.0, 0.0, PI),  # south pole
    (1.0, 2.7, PI),  # south pole, different theta
    # Just inside the seam: theta as close to 2*pi as we can get while still
    # being accepted by the Rust validator.
    (1.0, math.nextafter(TWO_PI, 0.0), 0.6),
    # Near-identical pair (stresses the small-angle arm of Vincenty).
    (1.0, 1.000000, 0.800000),
    (1.0, 1.000000 + 1e-11, 0.800000 + 1e-11),
    # Near-antipodal pair (stresses the large-angle arm of Vincenty).
    (1.0, 0.0, 1e-4),
    (1.0, PI, PI - 1e-4),
    # Assorted radii and generic positions.
    (0.5, 2.0, 1.0),
    (2.0, 4.5, 2.4),
    (3.14159, 5.5, 0.3),
    (0.001, 0.25, 2.9),
    (10.0, 3.3, 1.7),
]


def _random_cases(n: int, seed: int = 0) -> list[tuple[float, float, float]]:
    rng = random.Random(seed)
    out: list[tuple[float, float, float]] = []
    for _ in range(n):
        r = rng.uniform(0.05, 5.0)
        # Keep theta strictly below 2*pi (Rust rejects the exact endpoint).
        theta = rng.uniform(0.0, math.nextafter(TWO_PI, 0.0))
        phi = rng.uniform(0.0, PI)
        out.append((r, theta, phi))
    return out


CASES: list[tuple[float, float, float]] = _FIXED_CASES + _random_cases(50)


def _is_near_antipodal(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True when the angular separation is within ~1e-3 of pi.

    Vincenty stays well-conditioned at antipodes, but the two independent f64
    evaluation orders (Python `math` vs Rust `f64`) can disagree by a few ulps
    that get amplified near pi. We loosen the tolerance there rather than pretend
    the last bit is reproducible across languages.
    """
    sep = lingua_angular_distance(LinguaPoint(*a), LinguaPoint(*b))
    return abs(sep - PI) < 1e-3


# ── Tier 1: functions with a Python-reachable Rust twin ──────────────────────


def _ordered_pairs() -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    # Full grid over the fixed cases (covers every interesting geometry pairing)
    # plus a deterministic sample of random×random pairs for breadth.
    pairs = [(a, b) for a in _FIXED_CASES for b in _FIXED_CASES]
    rnd = _random_cases(50)
    rng = random.Random(1)
    for _ in range(200):
        pairs.append((rng.choice(rnd), rng.choice(rnd)))
    return pairs


@pytest.mark.parametrize("a,b", _ordered_pairs())
def test_angular_distance_parity(a, b):
    """angular_distance must agree across languages for every ordered pair."""
    lingua = lingua_angular_distance(LinguaPoint(*a), LinguaPoint(*b))
    rust = sphereql.angular_distance(
        sphereql.SphericalPoint(*a), sphereql.SphericalPoint(*b)
    )
    if _is_near_antipodal(a, b):
        assert lingua == pytest.approx(rust, abs=1e-9, rel=1e-9)
    else:
        assert lingua == pytest.approx(rust, abs=1e-12, rel=1e-9)


@pytest.mark.parametrize("case", CASES)
def test_spherical_point_normalization_parity(case):
    """Constructed .r/.theta/.phi must match field-for-field.

    Every CASE already lives in both constructors' accepted domain, so neither
    side renormalizes — but asserting it here means a future change to either
    normalization rule (or a constructor that silently mutates an in-range
    input) is caught immediately.
    """
    lp = LinguaPoint(*case)
    rp = sphereql.SphericalPoint(*case)
    assert lp.r == pytest.approx(rp.r, abs=1e-12)
    assert lp.theta == pytest.approx(rp.theta, abs=1e-12)
    assert lp.phi == pytest.approx(rp.phi, abs=1e-12)


def _rust_cartesian_scales_by_r() -> bool:
    """Empirically determine whether sphereql.spherical_to_cartesian scales by r.

    Do NOT assume. Probe with r=3 on the equator at theta=0, where the x
    component is r if scaled and 1 if the function returns a unit direction.
    """
    probe = sphereql.SphericalPoint(3.0, 0.0, PI / 2)
    c = sphereql.spherical_to_cartesian(probe)
    return abs(c.x - 3.0) < 1e-9


RUST_CART_SCALES_BY_R = _rust_cartesian_scales_by_r()


@pytest.mark.parametrize("case", CASES)
def test_spherical_to_cartesian_parity(case):
    """spherical -> cartesian must agree component-wise.

    lingua's ``to_cartesian`` always scales by r. We branch on the empirically
    measured Rust behaviour so this test stays correct whether the binding
    scales by r or returns a unit vector.
    """
    r, theta, phi = case
    lp = LinguaPoint(*case)
    rp = sphereql.SphericalPoint(*case)
    lx, ly, lz = lp.to_cartesian()
    rc = sphereql.spherical_to_cartesian(rp)

    if RUST_CART_SCALES_BY_R:
        ex, ey, ez = lx, ly, lz
    else:
        # Rust returns a unit direction; compare against lingua divided by r.
        ex, ey, ez = lx / r, ly / r, lz / r

    assert rc.x == pytest.approx(ex, abs=1e-9)
    assert rc.y == pytest.approx(ey, abs=1e-9)
    assert rc.z == pytest.approx(ez, abs=1e-9)


@pytest.mark.parametrize("case", CASES)
def test_cartesian_to_spherical_roundtrip_parity(case):
    """cartesian -> spherical must agree, modulo the pole singularity.

    Near the poles (phi -> 0 or phi -> pi) theta is geometrically undefined, so
    the two implementations may report different (equally valid) theta values.
    Skip the theta comparison there; r and phi must always agree.
    """
    lp = LinguaPoint(*case)
    rp = sphereql.SphericalPoint(*case)

    lx, ly, lz = lp.to_cartesian()
    lback = LinguaPoint.from_cartesian(lx, ly, lz)
    rback = sphereql.cartesian_to_spherical(sphereql.spherical_to_cartesian(rp))

    assert lback.r == pytest.approx(rback.r, abs=1e-9)
    assert lback.phi == pytest.approx(rback.phi, abs=1e-9)

    near_pole = lback.phi < 1e-6 or lback.phi > PI - 1e-6
    if not near_pole:
        # theta lives on a circle; compare modulo 2*pi via the shortest arc.
        dtheta = abs(lback.theta - rback.theta) % TWO_PI
        dtheta = min(dtheta, TWO_PI - dtheta)
        assert dtheta == pytest.approx(0.0, abs=1e-9)


# ── Tier 2: functions with NO Python-reachable Rust twin ─────────────────────
#
# These exist in `lingua_spherica` and (for some) in `sphereql-core`'s Rust
# source, but are NOT exported through the compiled `sphereql` wheel — they have
# no `#[pyfunction]` binding in sphereql-python/src/. There is therefore nothing
# to call from Python to compare against. They are skipped (not omitted) so this
# suite documents exactly where the cross-language guarantee currently stops.
#
# Notes for whoever closes these gaps:
#   * slerp: sphereql-core HAS `slerp` and `full_slerp` (interpolation.rs). The
#     Rust `slerp` returns a UNIT-radius direction; `full_slerp` additionally
#     lerps r. lingua's `slerp` lerps r, so it corresponds to Rust `full_slerp`,
#     and the two algorithms (Vincenty omega + cartesian blend + the
#     deterministic |z|<0.9 antipodal-tangent fallback) are line-for-line the
#     same. Once a `#[pyfunction]` exposes `full_slerp`, promote this to a real
#     parity test mirroring `test_angular_distance_parity`.
#   * spherical_centroid / circular_weighted_mean / circular_variance /
#     theta_distance / phi_distance / geodesic_path / semantic_distance: no Rust
#     twin reachable from Python (some have no Rust implementation at all).
#
# See TODO.md for the binding-surface gap.


@pytest.mark.skip(reason="no python-reachable Rust twin yet (full_slerp unexposed); see TODO")
def test_slerp_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_spherical_centroid_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_circular_weighted_mean_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_circular_variance_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_theta_distance_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_phi_distance_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_geodesic_path_parity():
    ...


@pytest.mark.skip(reason="no python-reachable Rust twin yet; see TODO")
def test_semantic_distance_parity():
    ...
