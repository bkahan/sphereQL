"""
SphereQL — Spherical coordinate knowledge representation.

Project high-dimensional embeddings onto a 3D sphere for fast semantic
search, interactive visualization, and knowledge structure analysis.

Quick start:
    >>> import sphereql
    >>> pipeline = sphereql.Pipeline(categories, embeddings)
    >>> results = pipeline.nearest(query_embedding, k=5)
    >>> sphereql.visualize(categories, embeddings)
"""

from sphereql.sphereql import *  # noqa: F401,F403

__version__ = "0.1.0"
