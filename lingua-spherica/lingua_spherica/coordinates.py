"""
Lingua Spherica - Coordinate Mathematics

Spherical geometry operations for the SphereQL coordinate system.
Physics convention: theta=azimuthal, phi=polar.
Great-circle distance: d(p1,p2) = arccos(sin phi1 sin phi2 cos(theta1-theta2) + cos phi1 cos phi2)
"""

import math
from typing import List, Tuple, Optional
from .types import SphericalPoint

TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0
EPSILON = 1e-10


def angular_distance(p1: SphericalPoint, p2: SphericalPoint) -> float:
    """Great-circle distance via Vincenty formula (stable near antipodal points)."""
    sin_phi1, cos_phi1 = math.sin(p1.phi), math.cos(p1.phi)
    sin_phi2, cos_phi2 = math.sin(p2.phi), math.cos(p2.phi)
    delta_theta = p1.theta - p2.theta
    cos_dt, sin_dt = math.cos(delta_theta), math.sin(delta_theta)
    num = math.sqrt((sin_phi2 * sin_dt) ** 2 +
                    (sin_phi1 * cos_phi2 - cos_phi1 * sin_phi2 * cos_dt) ** 2)
    den = cos_phi1 * cos_phi2 + sin_phi1 * sin_phi2 * cos_dt
    return math.atan2(num, den)


def theta_distance(theta1: float, theta2: float) -> float:
    """Shortest angular distance on S1."""
    d = abs(theta1 - theta2) % TWO_PI
    return min(d, TWO_PI - d)


def phi_distance(phi1: float, phi2: float) -> float:
    """Absolute polar angle difference."""
    return abs(phi1 - phi2)


def circular_weighted_mean(angles: List[float], weights: List[float]) -> float:
    """Von Mises MLE for mean direction. Handles wraparound correctly."""
    if not angles:
        return 0.0
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return angles[0]
    sin_sum = sum(w * math.sin(a) for a, w in zip(angles, weights))
    cos_sum = sum(w * math.cos(a) for a, w in zip(angles, weights))
    mean = math.atan2(sin_sum / total_weight, cos_sum / total_weight)
    return mean % TWO_PI


def circular_variance(angles: List[float], weights: Optional[List[float]] = None) -> float:
    """V = 1 - R_bar. V=0 means concentrated, V~1 means uniform."""
    if not angles:
        return 0.0
    if weights is None:
        weights = [1.0] * len(angles)
    total = sum(weights)
    if total < EPSILON:
        return 0.0
    sin_sum = sum(w * math.sin(a) for a, w in zip(angles, weights))
    cos_sum = sum(w * math.cos(a) for a, w in zip(angles, weights))
    r_bar = math.sqrt(sin_sum**2 + cos_sum**2) / total
    return 1.0 - r_bar


def slerp(p1: SphericalPoint, p2: SphericalPoint, t: float) -> SphericalPoint:
    """Spherical linear interpolation. Traces the geodesic from p1 to p2.
    slerp(q1, q2, t) = sin((1-t)O)/sin(O) * q1 + sin(tO)/sin(O) * q2
    Radius linearly interpolated."""
    t = max(0.0, min(1.0, t))
    c1 = (math.sin(p1.phi) * math.cos(p1.theta),
           math.sin(p1.phi) * math.sin(p1.theta), math.cos(p1.phi))
    c2 = (math.sin(p2.phi) * math.cos(p2.theta),
           math.sin(p2.phi) * math.sin(p2.theta), math.cos(p2.phi))
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(c1, c2))))
    omega = math.acos(dot)
    if omega < EPSILON:
        r = (1 - t) * p1.r + t * p2.r
        return SphericalPoint(p1.theta, p1.phi, r)
    sin_omega = math.sin(omega)
    a = math.sin((1 - t) * omega) / sin_omega
    b = math.sin(t * omega) / sin_omega
    x = a * c1[0] + b * c2[0]
    y = a * c1[1] + b * c2[1]
    z = a * c1[2] + b * c2[2]
    result = SphericalPoint.from_cartesian(x, y, z)
    result.r = (1 - t) * p1.r + t * p2.r
    return result


def geodesic_path(p1: SphericalPoint, p2: SphericalPoint,
                  n_points: int = 50) -> List[SphericalPoint]:
    """Discrete geodesic path: n_points uniformly spaced on the great circle."""
    return [slerp(p1, p2, i / (n_points - 1)) for i in range(n_points)]


def spherical_centroid(points: List[SphericalPoint],
                       weights: Optional[List[float]] = None) -> SphericalPoint:
    """Frechet mean on S2: Cartesian mean then re-project."""
    if not points:
        return SphericalPoint(0.0, HALF_PI, 0.5)
    if weights is None:
        weights = [1.0] * len(points)
    total_w = sum(weights)
    if total_w < EPSILON:
        total_w = 1.0
    cx, cy, cz, cr = 0.0, 0.0, 0.0, 0.0
    for p, w in zip(points, weights):
        x, y, z = p.to_cartesian()
        norm = math.sqrt(x*x + y*y + z*z)
        if norm > EPSILON:
            cx += w * x / norm
            cy += w * y / norm
            cz += w * z / norm
        cr += w * p.r
    cx /= total_w; cy /= total_w; cz /= total_w; cr /= total_w
    result = SphericalPoint.from_cartesian(cx, cy, cz)
    result.r = cr
    return result


def semantic_distance(p1: SphericalPoint, p2: SphericalPoint,
                      w_theta: float = 0.5, w_phi: float = 0.3,
                      w_r: float = 0.2) -> float:
    """Weighted distance: w_t*d_theta + w_p*|dphi| + w_r*|dr|."""
    d_theta = theta_distance(p1.theta, p2.theta) / math.pi
    d_phi = abs(p1.phi - p2.phi) / math.pi
    d_r = abs(p1.r - p2.r)
    return w_theta * d_theta + w_phi * d_phi + w_r * d_r
