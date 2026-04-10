# Prompt 05: Add Python tests and type stubs

## Context

The `sphereql-python` crate is fully implemented (prompts 01–04). This prompt
adds Python-level tests and a `.pyi` stub file for IDE autocompletion.

## Task

### 1. Python test suite

Create `sphereql-python/tests/test_sphereql.py`:

```python
"""Tests for the sphereql Python module.

Run with:
    cd sphereql-python
    maturin develop
    pytest tests/
"""
import math
import numpy as np
import pytest
```

Write tests for:

#### Core types

- `test_spherical_point_creation` — valid construction, property access
- `test_spherical_point_validation` — negative r, out-of-range theta/phi
  raise `ValueError`
- `test_cartesian_point` — construction, `magnitude()`, `normalize()`
- `test_geo_point_validation` — invalid lat/lon/alt raise `ValueError`
- `test_roundtrip_spherical_cartesian` — `p.to_cartesian().to_spherical()`
  round-trips within epsilon
- `test_roundtrip_spherical_geo` — same for geo
- `test_repr` — verify `repr()` output is readable

#### Distance functions

- `test_angular_distance_same_point` — should be ~0
- `test_angular_distance_opposite` — should be ~pi
- `test_great_circle_distance` — basic sanity check with known radius
- `test_chord_distance` — antipodal points on unit sphere = 2.0

#### Projections

- `test_pca_fit_and_project` — fit on random data, project returns
  `SphericalPoint` with valid ranges
- `test_pca_project_rich` — returns `ProjectedPoint` with certainty in [0, 1]
- `test_pca_batch` — `project_batch` returns correct length
- `test_pca_explained_variance` — `explained_variance_ratio` in (0, 1]
- `test_pca_volumetric` — volumetric=True changes results
- `test_pca_radial_fixed` — `radial=2.5` gives r=2.5 for all projections
- `test_pca_radial_magnitude` — `radial="magnitude"` gives r proportional
  to input magnitude
- `test_random_projection_deterministic` — same seed gives same results
- `test_random_projection_different_seeds` — different seeds give different
  results
- `test_pca_invalid_radial` — `radial="invalid"` raises `ValueError`
- `test_pca_dimension_mismatch` — projecting wrong-dim vector raises error

#### Pipeline

- `test_pipeline_nearest` — returns correct number of results, sorted by
  distance
- `test_pipeline_similar_above` — all results have distance below threshold
- `test_pipeline_concept_path` — returns a path with start and end matching
- `test_pipeline_concept_path_none` — disconnected nodes return None
- `test_pipeline_detect_globs` — fixed k returns that many globs, total
  members = num_items
- `test_pipeline_local_manifold` — returns ManifoldInfo with variance_ratio
  in (0, 1]
- `test_pipeline_with_projection` — constructing with pre-fitted PCA works
- `test_pipeline_properties` — `num_items`, `categories`

#### Acceptance: numpy integration

- `test_numpy_f32_upcast` — passing float32 array works (converted to f64)
- `test_list_input` — passing `list[float]` instead of numpy works for
  single embeddings

### 2. Type stub file

Create `sphereql-python/sphereql.pyi`:

```python
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
    def nearest(
        self,
        query: NDArray[np.float64] | Sequence[float],
        k: int = 5,
    ) -> list[NearestHit]: ...
    def similar_above(
        self,
        query: NDArray[np.float64] | Sequence[float],
        min_cosine: float = 0.8,
    ) -> list[NearestHit]: ...
    def concept_path(
        self,
        source_id: str,
        target_id: str,
        *,
        graph_k: int = 10,
        query: Optional[NDArray[np.float64] | Sequence[float]] = None,
    ) -> Optional[PathResult]: ...
    def detect_globs(
        self,
        *,
        k: Optional[int] = None,
        max_k: int = 10,
        query: Optional[NDArray[np.float64] | Sequence[float]] = None,
    ) -> list[GlobInfo]: ...
    def local_manifold(
        self,
        query: NDArray[np.float64] | Sequence[float],
        *,
        neighborhood_k: int = 10,
    ) -> ManifoldInfo: ...

class NearestHit:
    id: str
    category: str
    distance: float
    certainty: float
    intensity: float

class PathResult:
    total_distance: float
    steps: list[PathStep]

class PathStep:
    id: str
    category: str
    cumulative_distance: float

class GlobInfo:
    id: int
    centroid: list[float]
    member_count: int
    radius: float
    top_categories: list[tuple[str, int]]

class ManifoldInfo:
    centroid: list[float]
    normal: list[float]
    variance_ratio: float

def angular_distance(a: SphericalPoint, b: SphericalPoint) -> float: ...
def great_circle_distance(a: SphericalPoint, b: SphericalPoint, radius: float) -> float: ...
def chord_distance(a: SphericalPoint, b: SphericalPoint) -> float: ...
def spherical_to_cartesian(p: SphericalPoint) -> CartesianPoint: ...
def cartesian_to_spherical(p: CartesianPoint) -> SphericalPoint: ...
def spherical_to_geo(p: SphericalPoint) -> GeoPoint: ...
def geo_to_spherical(p: GeoPoint) -> SphericalPoint: ...
```

### 3. pytest configuration

Create `sphereql-python/pytest.ini`:

```ini
[pytest]
testpaths = tests
```

## Verification

1. `cd sphereql-python && maturin develop --release`
2. `pytest tests/ -v`
3. All tests should pass.

## Do NOT

- Install maturin or run the tests as part of this prompt execution. Just
  write the files. The user will run them.
