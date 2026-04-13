"""Type stubs for the sphereql Python module."""

from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

class SphericalPoint:
    r: float
    theta: float
    phi: float
    def __init__(self, r: float, theta: float, phi: float) -> None: ...
    def to_cartesian(self) -> CartesianPoint: ...
    def to_geo(self) -> GeoPoint: ...

class CartesianPoint:
    x: float
    y: float
    z: float
    def __init__(self, x: float, y: float, z: float) -> None: ...
    def magnitude(self) -> float: ...
    def normalize(self) -> CartesianPoint: ...
    def to_spherical(self) -> SphericalPoint: ...

class GeoPoint:
    lat: float
    lon: float
    alt: float
    def __init__(self, lat: float, lon: float, alt: float) -> None: ...
    def to_spherical(self) -> SphericalPoint: ...
    def to_cartesian(self) -> CartesianPoint: ...

class ProjectedPoint:
    position: SphericalPoint
    certainty: float
    intensity: float
    projection_magnitude: float

class PcaProjection:
    dimensionality: int
    explained_variance_ratio: float
    @classmethod
    def fit(
        cls,
        embeddings: NDArray[np.float64],
        *,
        radial: str | float = "magnitude",
        volumetric: bool = False,
    ) -> PcaProjection: ...
    def project(self, embedding: NDArray[np.float64] | Sequence[float]) -> SphericalPoint: ...
    def project_rich(self, embedding: NDArray[np.float64] | Sequence[float]) -> ProjectedPoint: ...
    def project_batch(self, embeddings: NDArray[np.float64]) -> list[SphericalPoint]: ...
    def project_rich_batch(self, embeddings: NDArray[np.float64]) -> list[ProjectedPoint]: ...

class RandomProjection:
    dimensionality: int
    def __init__(
        self,
        dim: int,
        *,
        radial: str | float = "magnitude",
        seed: int = 42,
    ) -> None: ...
    def project(self, embedding: NDArray[np.float64] | Sequence[float]) -> SphericalPoint: ...
    def project_rich(self, embedding: NDArray[np.float64] | Sequence[float]) -> ProjectedPoint: ...
    def project_batch(self, embeddings: NDArray[np.float64]) -> list[SphericalPoint]: ...
    def project_rich_batch(self, embeddings: NDArray[np.float64]) -> list[ProjectedPoint]: ...

class Pipeline:
    num_items: int
    categories: list[str]
    def __init__(
        self,
        categories: list[str],
        embeddings: NDArray[np.float64],
        *,
        projection: Optional[PcaProjection] = None,
    ) -> None: ...
    @staticmethod
    def from_json(json: str) -> Pipeline: ...
    def nearest(
        self,
        query: NDArray[np.float64] | Sequence[float],
        k: int = 5,
    ) -> list[NearestHit]: ...
    def nearest_json(
        self,
        query: NDArray[np.float64] | Sequence[float],
        k: int = 5,
    ) -> str: ...
    def similar_above(
        self,
        query: NDArray[np.float64] | Sequence[float],
        min_cosine: float = 0.8,
    ) -> list[NearestHit]: ...
    def similar_above_json(
        self,
        query: NDArray[np.float64] | Sequence[float],
        min_cosine: float = 0.8,
    ) -> str: ...
    def concept_path(
        self,
        source_id: str,
        target_id: str,
        *,
        graph_k: int = 10,
        query: Optional[NDArray[np.float64] | Sequence[float]] = None,
    ) -> Optional[PathResult]: ...
    def concept_path_json(
        self,
        source_id: str,
        target_id: str,
        *,
        graph_k: int = 10,
        query: Optional[NDArray[np.float64] | Sequence[float]] = None,
    ) -> str: ...
    def detect_globs(
        self,
        *,
        k: Optional[int] = None,
        max_k: int = 10,
        query: Optional[NDArray[np.float64] | Sequence[float]] = None,
    ) -> list[GlobInfo]: ...
    def detect_globs_json(
        self,
        *,
        k: Optional[int] = None,
        max_k: int = 10,
        query: Optional[NDArray[np.float64] | Sequence[float]] = None,
    ) -> str: ...
    def local_manifold(
        self,
        query: NDArray[np.float64] | Sequence[float],
        *,
        neighborhood_k: int = 10,
    ) -> ManifoldInfo: ...
    def local_manifold_json(
        self,
        query: NDArray[np.float64] | Sequence[float],
        *,
        neighborhood_k: int = 10,
    ) -> str: ...

class NearestHit:
    id: str
    category: str
    distance: float
    certainty: float
    intensity: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> NearestHit: ...

class PathResult:
    total_distance: float
    steps: list[PathStep]
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> PathResult: ...

class PathStep:
    id: str
    category: str
    cumulative_distance: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> PathStep: ...

class GlobInfo:
    id: int
    centroid: list[float]
    member_count: int
    radius: float
    top_categories: list[tuple[str, int]]
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> GlobInfo: ...

class ManifoldInfo:
    centroid: list[float]
    normal: list[float]
    variance_ratio: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> ManifoldInfo: ...

def angular_distance(a: SphericalPoint, b: SphericalPoint) -> float: ...
def great_circle_distance(a: SphericalPoint, b: SphericalPoint, radius: float) -> float: ...
def chord_distance(a: SphericalPoint, b: SphericalPoint) -> float: ...
def spherical_to_cartesian(p: SphericalPoint) -> CartesianPoint: ...
def cartesian_to_spherical(p: CartesianPoint) -> SphericalPoint: ...
def spherical_to_geo(p: SphericalPoint) -> GeoPoint: ...
def geo_to_spherical(p: GeoPoint) -> SphericalPoint: ...
