"""
Lingua Spherica — Language to SphereQL Mapping System

This Python package contains the core spherical-geometry primitives
(types and coordinate math). The full six-stage pipeline lives in the
`sphereql-lingua` Rust crate and is exposed to Python via the
`sphereql-python` bindings — Rust is the source of truth.
"""

from .types import (
    SphericalPoint, Concept, Relation, RelationType,
    ConceptGraph, DomainAnchor
)
from .coordinates import (
    angular_distance, theta_distance, phi_distance,
    circular_weighted_mean, circular_variance,
    slerp, geodesic_path, spherical_centroid,
    semantic_distance
)

__all__ = [
    'SphericalPoint', 'Concept', 'Relation', 'RelationType',
    'ConceptGraph', 'DomainAnchor',
    'angular_distance', 'theta_distance', 'phi_distance',
    'circular_weighted_mean', 'circular_variance',
    'slerp', 'geodesic_path', 'spherical_centroid',
    'semantic_distance',
]
