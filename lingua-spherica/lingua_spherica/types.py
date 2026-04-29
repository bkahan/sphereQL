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
from typing import Optional, List, Dict, Tuple
import math


@dataclass
class SphericalPoint:
    """A point in the SphereQL coordinate system."""
    theta: float
    phi: float
    r: float

    def __post_init__(self):
        self.theta = self.theta % (2 * math.pi)
        self.phi = max(0.0, min(math.pi, self.phi))
        self.r = max(1e-6, self.r)

    def to_cartesian(self) -> Tuple[float, float, float]:
        x = self.r * math.sin(self.phi) * math.cos(self.theta)
        y = self.r * math.sin(self.phi) * math.sin(self.theta)
        z = self.r * math.cos(self.phi)
        return (x, y, z)

    @staticmethod
    def from_cartesian(x: float, y: float, z: float) -> 'SphericalPoint':
        r = math.sqrt(x*x + y*y + z*z)
        if r < 1e-10:
            return SphericalPoint(0.0, 0.0, 0.0)
        phi = math.acos(max(-1, min(1, z / r)))
        theta = math.atan2(y, x) % (2 * math.pi)
        return SphericalPoint(theta, phi, r)

    def angular_distance_to(self, other: 'SphericalPoint') -> float:
        cos_d = (math.sin(self.phi) * math.sin(other.phi) *
                 math.cos(self.theta - other.theta) +
                 math.cos(self.phi) * math.cos(other.phi))
        return math.acos(max(-1.0, min(1.0, cos_d)))


@dataclass
class Concept:
    """A semantic concept extracted from natural language."""
    text: str
    normalized: str
    point: Optional[SphericalPoint] = None
    frequency: int = 1
    positions: List[int] = field(default_factory=list)
    domain_scores: Dict[str, float] = field(default_factory=dict)
    abstraction_score: float = 0.5
    salience_score: float = 0.5
    primary_domain: Optional[str] = None
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
    evidence: Optional[str] = None

    def geodesic_length(self) -> Optional[float]:
        if self.source.point and self.target.point:
            return self.source.point.angular_distance_to(self.target.point)
        return None


@dataclass
class ConceptGraph:
    """A graph of concepts and relations, fully resolved in spherical space."""
    concepts: List[Concept] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    source_text: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def concept_map(self) -> Dict[str, Concept]:
        return {c.normalized: c for c in self.concepts}

    def get_concept(self, normalized: str) -> Optional[Concept]:
        for c in self.concepts:
            if c.normalized == normalized:
                return c
        return None

    def neighbors(self, concept: Concept) -> List[Tuple[Concept, Relation]]:
        result = []
        for r in self.relations:
            if r.source == concept:
                result.append((r.target, r))
            elif r.target == concept:
                result.append((r.source, r))
        return result

    def centroid(self) -> Optional[SphericalPoint]:
        """Spherical centroid via Cartesian mean and re-projection."""
        resolved = [c for c in self.concepts if c.point is not None]
        if not resolved:
            return None
        cx, cy, cz = 0.0, 0.0, 0.0
        total_r = 0.0
        for c in resolved:
            x, y, z = c.point.to_cartesian()
            w = c.point.r
            cx += x; cy += y; cz += z
            total_r += w
        n = len(resolved)
        cx /= n; cy /= n; cz /= n
        avg_r = total_r / n
        pt = SphericalPoint.from_cartesian(cx, cy, cz)
        pt.r = avg_r
        return pt


@dataclass
class DomainAnchor:
    """A fixed reference domain with a known θ position on the atlas sphere."""
    name: str
    theta: float
    angular_width: float = 0.3
    keywords: List[str] = field(default_factory=list)
    parent: Optional[str] = None

    @property
    def theta_range(self) -> Tuple[float, float]:
        half = self.angular_width / 2
        return (self.theta - half, self.theta + half)
