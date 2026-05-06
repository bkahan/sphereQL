"""
Lingua Spherica — Core Types
=============================

Mathematical Convention (Physics convention):
    θ (theta) ∈ [0, 2π)  — azimuthal angle (domain/longitude)
    φ (phi)   ∈ [0, π]   — polar angle (abstraction/colatitude)
    r         ∈ (0, ∞)   — radius (epistemic weight)

    Cartesian: x = r·sin(φ)·cos(θ), y = r·sin(φ)·sin(θ), z = r·cos(φ)
    North pole (φ=0) = maximally abstract
    South pole (φ=π) = maximally concrete
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math


@dataclass
class SphericalPoint:
    """A point in the SphereQL coordinate system.

    Field order matches the canonical Rust `SphericalPoint::new(r, theta, phi)`
    in `sphereql-core` so positional construction is portable across the
    Python and Rust surfaces.
    """
    r: float
    theta: float
    phi: float

    def __post_init__(self):
        self.theta = self.theta % (2 * math.pi)
        self.phi = max(0.0, min(math.pi, self.phi))
        self.r = max(1e-6, self.r)

    def to_cartesian(self) -> tuple[float, float, float]:
        x = self.r * math.sin(self.phi) * math.cos(self.theta)
        y = self.r * math.sin(self.phi) * math.sin(self.theta)
        z = self.r * math.cos(self.phi)
        return (x, y, z)

    @staticmethod
    def from_cartesian(x: float, y: float, z: float) -> SphericalPoint:
        r = math.sqrt(x*x + y*y + z*z)
        if r < 1e-10:
            # Degenerate origin input — `__post_init__` floors `r` at 1e-6,
            # so use that sentinel explicitly rather than passing 0.0 and
            # silently being clamped.
            return SphericalPoint(r=1e-6, theta=0.0, phi=0.0)
        phi = math.acos(max(-1, min(1, z / r)))
        theta = math.atan2(y, x) % (2 * math.pi)
        return SphericalPoint(r=r, theta=theta, phi=phi)

    def angular_distance_to(self, other: SphericalPoint) -> float:
        """Great-circle distance via the Vincenty atan2 form.

        Numerically stable for both near-identical and near-antipodal pairs,
        unlike the textbook `acos(...)` formula which loses precision near
        the endpoints of its domain.
        """
        sin_phi1, cos_phi1 = math.sin(self.phi), math.cos(self.phi)
        sin_phi2, cos_phi2 = math.sin(other.phi), math.cos(other.phi)
        delta_theta = self.theta - other.theta
        cos_dt, sin_dt = math.cos(delta_theta), math.sin(delta_theta)
        num = math.sqrt(
            (sin_phi2 * sin_dt) ** 2
            + (sin_phi1 * cos_phi2 - cos_phi1 * sin_phi2 * cos_dt) ** 2
        )
        den = cos_phi1 * cos_phi2 + sin_phi1 * sin_phi2 * cos_dt
        return math.atan2(num, den)


@dataclass
class Concept:
    """A semantic concept extracted from natural language."""
    text: str
    normalized: str
    point: SphericalPoint | None = None
    frequency: int = 1
    positions: list[int] = field(default_factory=list)
    domain_scores: dict[str, float] = field(default_factory=dict)
    abstraction_score: float = 0.5
    salience_score: float = 0.5
    primary_domain: str | None = None
    hierarchy_depth: int = 0

    def __hash__(self):
        return hash(self.normalized)

    def __eq__(self, other):
        if isinstance(other, Concept):
            return self.normalized == other.normalized
        return False


class RelationType(Enum):
    """Semantic relation types with spherical interpretations.

    IS_A:           Same θ, source has smaller φ (more abstract)
    INSTANCE_OF:    Same θ, source has larger φ (more concrete)
    PART_OF:        Containment — source sphere inside target sphere
    RELATED_TO:     Nearby θ, similar φ
    CAUSES:         Directed geodesic arc
    CONTRASTS:      Large Δθ or near-antipodal
    PARAMETERIZES:  Source defines coordinate of target
    TRANSFORMS_TO:  Directed geodesic with domain shift (Δθ ≠ 0)
    DEMONSTRATES:   Evidential link
    CONTAINS:       Hierarchical containment (sphere nesting)
    NEAR:           Small angular distance
    FAR_FROM:       Large angular distance
    """
    IS_A = auto()
    INSTANCE_OF = auto()
    PART_OF = auto()
    RELATED_TO = auto()
    CAUSES = auto()
    CONTRASTS = auto()
    PARAMETERIZES = auto()
    TRANSFORMS_TO = auto()
    DEMONSTRATES = auto()
    CONTAINS = auto()
    NEAR = auto()
    FAR_FROM = auto()


@dataclass
class Relation:
    """A directed semantic relation between two concepts."""
    source: Concept
    target: Concept
    relation_type: RelationType
    weight: float = 1.0
    evidence: str | None = None

    def geodesic_length(self) -> float | None:
        if self.source.point and self.target.point:
            return self.source.point.angular_distance_to(self.target.point)
        return None


@dataclass
class ConceptGraph:
    """A graph of concepts and relations, fully resolved in spherical space."""
    concepts: list[Concept] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    source_text: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def concept_map(self) -> dict[str, Concept]:
        return {c.normalized: c for c in self.concepts}

    def get_concept(self, normalized: str) -> Concept | None:
        for c in self.concepts:
            if c.normalized == normalized:
                return c
        return None

    def neighbors(self, concept: Concept) -> list[tuple[Concept, Relation]]:
        result = []
        for r in self.relations:
            if r.source == concept:
                result.append((r.target, r))
            elif r.target == concept:
                result.append((r.source, r))
        return result

    def centroid(self) -> SphericalPoint | None:
        """Weighted spherical centroid (epistemic-weighted Fréchet mean).

        Each concept's Cartesian position is weighted by its radius `r`
        (epistemic weight per the module convention), the weighted sum
        is normalized by the total weight, and the resulting direction
        is re-projected onto the sphere with radius = mean(r).
        """
        resolved = [c for c in self.concepts if c.point is not None]
        if not resolved:
            return None
        cx = cy = cz = 0.0
        total_r = 0.0
        for c in resolved:
            x, y, z = c.point.to_cartesian()
            norm = math.sqrt(x * x + y * y + z * z)
            w = c.point.r
            if norm > 1e-10:
                cx += w * x / norm
                cy += w * y / norm
                cz += w * z / norm
            total_r += w
        if total_r <= 0.0:
            return None
        cx /= total_r
        cy /= total_r
        cz /= total_r
        # Resultant length: if the unit vectors cancel, the centroid
        # direction is undefined (e.g. perfectly antipodal pairs). Signal
        # that with `None` rather than collapsing to an arbitrary pole.
        if math.sqrt(cx * cx + cy * cy + cz * cz) < 1e-10:
            return None
        avg_r = total_r / len(resolved)
        pt = SphericalPoint.from_cartesian(cx, cy, cz)
        pt.r = avg_r
        return pt


@dataclass
class DomainAnchor:
    """A fixed reference domain with a known θ position on the atlas sphere."""
    name: str
    theta: float
    angular_width: float = 0.3
    keywords: list[str] = field(default_factory=list)
    parent: str | None = None

    def __post_init__(self):
        two_pi = 2 * math.pi
        if self.angular_width <= 0:
            raise ValueError(
                f"DomainAnchor.angular_width must be > 0, got {self.angular_width}"
            )
        if self.angular_width >= two_pi:
            raise ValueError(
                "DomainAnchor.angular_width must be < 2π "
                f"(would cover the full circle), got {self.angular_width}"
            )

    @property
    def theta_range(self) -> tuple[float, float]:
        """Theta interval covered by this anchor, normalized into [0, 2π).

        The raw `(theta - half, theta + half)` form can produce negative
        endpoints or values past 2π near the seam — callers comparing a
        normalized theta against the range would miss matches. Both ends
        are taken modulo 2π. When the band straddles the seam (`lo > hi`)
        callers should treat the range as the wrap-around union
        `[lo, 2π) ∪ [0, hi]`.
        """
        two_pi = 2 * math.pi
        half = self.angular_width / 2
        lo = (self.theta - half) % two_pi
        hi = (self.theta + half) % two_pi
        return (lo, hi)
