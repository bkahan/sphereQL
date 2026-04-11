"""Type stubs for the sphereql native module."""

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

# ── Core types ─────────────────────────────────────────────────────────

class SphericalPoint:
    """A point in spherical coordinates (r, theta, phi)."""

    r: float
    theta: float
    phi: float
    def __init__(self, r: float, theta: float, phi: float) -> None: ...
    def to_cartesian(self) -> CartesianPoint: ...
    def to_geo(self) -> GeoPoint: ...

class CartesianPoint:
    """A point in 3D Cartesian coordinates (x, y, z)."""

    x: float
    y: float
    z: float
    def __init__(self, x: float, y: float, z: float) -> None: ...
    def magnitude(self) -> float: ...
    def normalize(self) -> CartesianPoint: ...
    def to_spherical(self) -> SphericalPoint: ...

class GeoPoint:
    """A geographic point (latitude, longitude, altitude)."""

    lat: float
    lon: float
    alt: float
    def __init__(self, lat: float, lon: float, alt: float) -> None: ...
    def to_spherical(self) -> SphericalPoint: ...
    def to_cartesian(self) -> CartesianPoint: ...

class ProjectedPoint:
    """Result of projecting an embedding with rich metadata."""

    position: SphericalPoint
    certainty: float
    intensity: float
    projection_magnitude: float

# ── Distance functions ─────────────────────────────────────────────────

def angular_distance(a: SphericalPoint, b: SphericalPoint) -> float:
    """Angular distance (radians) between two spherical points."""
    ...

def great_circle_distance(
    a: SphericalPoint, b: SphericalPoint, radius: float
) -> float:
    """Great-circle distance on a sphere of the given radius."""
    ...

def chord_distance(a: SphericalPoint, b: SphericalPoint) -> float:
    """Euclidean chord distance between two spherical points."""
    ...

# ── Conversion functions ───────────────────────────────────────────────

def spherical_to_cartesian(p: SphericalPoint) -> CartesianPoint: ...
def cartesian_to_spherical(p: CartesianPoint) -> SphericalPoint: ...
def spherical_to_geo(p: SphericalPoint) -> GeoPoint: ...
def geo_to_spherical(p: GeoPoint) -> SphericalPoint: ...

# ── Projection ─────────────────────────────────────────────────────────

class PcaProjection:
    """PCA-based projection from high-dimensional embeddings to spherical coordinates."""

    dimensionality: int
    explained_variance_ratio: float
    @classmethod
    def fit(
        cls,
        embeddings: NDArray[np.float64] | NDArray[np.float32] | list[list[float]],
        *,
        radial: str | float = "magnitude",
        volumetric: bool = False,
    ) -> PcaProjection: ...
    def project(
        self, embedding: NDArray[np.float64] | NDArray[np.float32] | Sequence[float]
    ) -> SphericalPoint: ...
    def project_rich(
        self, embedding: NDArray[np.float64] | NDArray[np.float32] | Sequence[float]
    ) -> ProjectedPoint: ...
    def project_batch(
        self,
        embeddings: NDArray[np.float64] | NDArray[np.float32] | list[list[float]],
    ) -> list[SphericalPoint]: ...
    def project_rich_batch(
        self,
        embeddings: NDArray[np.float64] | NDArray[np.float32] | list[list[float]],
    ) -> list[ProjectedPoint]: ...

class RandomProjection:
    """Random projection from high-dimensional embeddings to spherical coordinates."""

    dimensionality: int
    def __init__(
        self,
        dim: int,
        *,
        radial: str | float = "magnitude",
        seed: int = 42,
    ) -> None: ...
    def project(
        self, embedding: NDArray[np.float64] | NDArray[np.float32] | Sequence[float]
    ) -> SphericalPoint: ...
    def project_rich(
        self, embedding: NDArray[np.float64] | NDArray[np.float32] | Sequence[float]
    ) -> ProjectedPoint: ...
    def project_batch(
        self,
        embeddings: NDArray[np.float64] | NDArray[np.float32] | list[list[float]],
    ) -> list[SphericalPoint]: ...
    def project_rich_batch(
        self,
        embeddings: NDArray[np.float64] | NDArray[np.float32] | list[list[float]],
    ) -> list[ProjectedPoint]: ...

# ── Pipeline ───────────────────────────────────────────────────────────

class Pipeline:
    """SphereQL pipeline: PCA projection + spatial queries over embeddings."""

    num_items: int
    categories: list[str]
    def __init__(
        self,
        categories: list[str],
        embeddings: NDArray[np.float64] | NDArray[np.float32] | list[list[float]],
        *,
        projection: Optional[PcaProjection] = None,
    ) -> None: ...
    @staticmethod
    def from_json(json: str) -> Pipeline: ...
    def nearest(
        self,
        query: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
        k: int = 5,
    ) -> list[NearestHit]: ...
    def nearest_json(
        self,
        query: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
        k: int = 5,
    ) -> str: ...
    def similar_above(
        self,
        query: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
        min_cosine: float = 0.8,
    ) -> list[NearestHit]: ...
    def similar_above_json(
        self,
        query: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
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
        query: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
        *,
        neighborhood_k: int = 10,
    ) -> ManifoldInfo: ...
    def local_manifold_json(
        self,
        query: NDArray[np.float64] | NDArray[np.float32] | Sequence[float],
        *,
        neighborhood_k: int = 10,
    ) -> str: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...

# ── Result types ───────────────────────────────────────────────────────

class NearestHit:
    """A nearest-neighbor search result."""

    id: str
    category: str
    distance: float
    certainty: float
    intensity: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> NearestHit: ...

class PathResult:
    """A shortest-path result between two items in projected space."""

    total_distance: float
    steps: list[PathStep]
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> PathResult: ...

class PathStep:
    """A single step along a concept path."""

    id: str
    category: str
    cumulative_distance: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> PathStep: ...

class GlobInfo:
    """A detected cluster (glob) in projected space."""

    id: int
    centroid: list[float]
    member_count: int
    radius: float
    top_categories: list[tuple[str, int]]
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> GlobInfo: ...

class ManifoldInfo:
    """Local manifold information around a query point."""

    centroid: list[float]
    normal: list[float]
    variance_ratio: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> ManifoldInfo: ...

# ── Vector store ───────────────────────────────────────────────────────

class InMemoryStore:
    """In-memory vector store using brute-force cosine similarity."""

    def __init__(self, collection: str, dimension: int) -> None: ...
    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or update records. Each dict needs 'id' (str), 'vector' (list[float]), optional 'metadata' (dict)."""
        ...
    def count(self) -> int: ...
    def __len__(self) -> int: ...

class VectorStoreBridge:
    """Bridge between an InMemoryStore and the SphereQL pipeline."""

    def __init__(
        self,
        store: InMemoryStore,
        *,
        batch_size: Optional[int] = None,
        max_records: Optional[int] = None,
    ) -> None: ...
    def build_pipeline(self, *, category_key: str = "category") -> None:
        """Pull vectors from the store and build the SphereQL pipeline."""
        ...
    def query_nearest(
        self, embedding: list[float], *, k: int = 5
    ) -> list[NearestHit]: ...
    def query_similar(
        self, embedding: list[float], *, min_cosine: float = 0.8
    ) -> list[NearestHit]: ...
    def query_concept_path(
        self,
        source_id: str,
        target_id: str,
        *,
        graph_k: int = 10,
        embedding: list[float],
    ) -> Optional[PathResult]: ...
    def query_detect_globs(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        max_k: int = 10,
    ) -> list[GlobInfo]: ...
    def hybrid_search(
        self,
        embedding: list[float],
        *,
        final_k: int = 5,
        recall_k: int = 20,
    ) -> list[dict[str, Any]]: ...
    def sync_projections(self) -> int:
        """Push spherical coordinates back to the store as metadata."""
        ...
    def __len__(self) -> int: ...

class QdrantBridge:
    """Bridge between a Qdrant collection and the SphereQL pipeline."""

    def __init__(
        self,
        url: str,
        collection: str,
        dimension: int,
        *,
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_records: Optional[int] = None,
    ) -> None: ...
    def build_pipeline(self, *, category_key: str = "category") -> None: ...
    def query_nearest(
        self, embedding: list[float], *, k: int = 5
    ) -> list[NearestHit]: ...
    def query_similar(
        self, embedding: list[float], *, min_cosine: float = 0.8
    ) -> list[NearestHit]: ...
    def query_concept_path(
        self,
        source_id: str,
        target_id: str,
        *,
        graph_k: int = 10,
        embedding: list[float],
    ) -> Optional[PathResult]: ...
    def query_detect_globs(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        max_k: int = 10,
    ) -> list[GlobInfo]: ...
    def hybrid_search(
        self,
        embedding: list[float],
        *,
        final_k: int = 5,
        recall_k: int = 20,
    ) -> list[dict[str, Any]]: ...
    def sync_projections(self) -> int: ...
    def __len__(self) -> int: ...

class PineconeBridge:
    """Bridge between a Pinecone index and the SphereQL pipeline."""

    def __init__(
        self,
        api_key: str,
        host: str,
        dimension: int,
        *,
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_records: Optional[int] = None,
    ) -> None: ...
    def build_pipeline(self, *, category_key: str = "category") -> None: ...
    def query_nearest(
        self, embedding: list[float], *, k: int = 5
    ) -> list[NearestHit]: ...
    def query_similar(
        self, embedding: list[float], *, min_cosine: float = 0.8
    ) -> list[NearestHit]: ...
    def query_concept_path(
        self,
        source_id: str,
        target_id: str,
        *,
        graph_k: int = 10,
        embedding: list[float],
    ) -> Optional[PathResult]: ...
    def query_detect_globs(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        max_k: int = 10,
    ) -> list[GlobInfo]: ...
    def hybrid_search(
        self,
        embedding: list[float],
        *,
        final_k: int = 5,
        recall_k: int = 20,
    ) -> list[dict[str, Any]]: ...
    def sync_projections(self) -> int: ...
    def __len__(self) -> int: ...

# ── Visualization ──────────────────────────────────────────────────────

def visualize(
    categories: list[str],
    embeddings: list[list[float]],
    output: str = "sphere_viz.html",
    labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    open_browser: bool = True,
) -> str:
    """Generate an interactive 3D sphere visualization from embeddings.

    Returns the absolute path of the generated HTML file.
    """
    ...

def visualize_pipeline(
    pipeline: Pipeline,
    output: str = "sphere_viz.html",
    title: Optional[str] = None,
    open_browser: bool = True,
) -> str:
    """Generate a visualization from an already-built Pipeline.

    Returns the absolute path of the generated HTML file.
    """
    ...
