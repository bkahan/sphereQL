"""
Lingua Spherica — Language to SphereQL Mapping System
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
from .engine import LinguaSphericaEngine

__all__ = [
    'SphericalPoint', 'Concept', 'Relation', 'RelationType',
    'ConceptGraph', 'DomainAnchor',
    'angular_distance', 'theta_distance', 'phi_distance',
    'circular_weighted_mean', 'circular_variance',
    'slerp', 'geodesic_path', 'spherical_centroid',
    'semantic_distance',
    'LinguaSphericaEngine',
]
