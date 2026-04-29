"""
Lingua Spherica - Pipeline Engine

Six-stage orchestrator: Text -> Concepts -> theta -> phi -> r -> Relations -> SphereQL
Each stage is a separate, replaceable module.
"""

from typing import List, Optional
from .types import Concept, ConceptGraph, SphericalPoint
from .concept_extractor import ConceptExtractor
from .domain_router import DomainRouter
from .abstraction_resolver import AbstractionResolver
from .salience_scorer import SalienceScorer
from .relation_encoder import RelationEncoder
from .sphereql_compositor import SphereQLCompositor


class LinguaSphericaEngine:
    """Main pipeline engine: Text -> SphereQL.

    Usage:
        engine = LinguaSphericaEngine()
        sphereql_output = engine.process(text)
    """

    def __init__(self,
                 extractor: Optional[ConceptExtractor] = None,
                 domain_router: Optional[DomainRouter] = None,
                 abstraction_resolver: Optional[AbstractionResolver] = None,
                 salience_scorer: Optional[SalienceScorer] = None,
                 relation_encoder: Optional[RelationEncoder] = None,
                 compositor: Optional[SphereQLCompositor] = None,
                 verbose: bool = False):
        self.extractor = extractor or ConceptExtractor()
        self.domain_router = domain_router or DomainRouter()
        self.abstraction_resolver = abstraction_resolver or AbstractionResolver()
        self.salience_scorer = salience_scorer or SalienceScorer()
        self.relation_encoder = relation_encoder or RelationEncoder()
        self.compositor = compositor or SphereQLCompositor()
        self.verbose = verbose

    def process(self, text: str) -> str:
        """Full pipeline: natural language text -> SphereQL query string."""
        graph = self.build_graph(text)
        return self.compositor.compose(graph)

    def build_graph(self, text: str) -> ConceptGraph:
        """Build the concept graph without composing to SphereQL."""
        if self.verbose:
            print(f"[Engine] Processing {len(text)} characters...")

        # Stage 1: Concept Extraction
        concepts = self.extractor.extract(text)
        if self.verbose:
            print(f"[Engine] Extracted {len(concepts)} concepts:")
            for c in concepts[:10]:
                print(f"  - {c.normalized} (freq={c.frequency})")

        # Stage 2: Domain Routing (theta)
        for concept in concepts:
            theta = self.domain_router.assign_theta(concept, text)
            concept.point = SphericalPoint(theta=theta, phi=0.0, r=0.5)
        if self.verbose:
            print(f"[Engine] theta assigned. Domain distribution:")
            for c in concepts[:10]:
                domain = self.domain_router.get_domain_name(c.point.theta)
                print(f"  - {c.normalized}: theta={c.point.theta:.3f} ({domain})")

        # Stage 3: Abstraction Resolution (phi)
        for concept in concepts:
            phi = self.abstraction_resolver.assign_phi(concept, text)
            concept.point.phi = phi
        if self.verbose:
            print(f"[Engine] phi assigned. Abstraction levels:")
            sorted_by_phi = sorted(concepts, key=lambda c: c.point.phi)
            for c in sorted_by_phi[:10]:
                label = "abstract" if c.point.phi < 1.0 else "concrete"
                print(f"  - {c.normalized}: phi={c.point.phi:.3f} ({label})")

        # Stage 4: Salience Scoring (r)
        self.salience_scorer.score_all(concepts, text)
        for concept in concepts:
            concept.point.r = self.salience_scorer.salience_to_r(concept.salience_score)
        if self.verbose:
            print(f"[Engine] r assigned. Top concepts by salience:")
            sorted_by_r = sorted(concepts, key=lambda c: c.point.r, reverse=True)
            for c in sorted_by_r[:10]:
                print(f"  - {c.normalized}: r={c.point.r:.3f} (salience={c.salience_score:.3f})")

        # Stage 5: Relation Encoding
        relations = self.relation_encoder.extract_relations(concepts, text)
        if self.verbose:
            print(f"[Engine] Extracted {len(relations)} relations:")
            for r in relations[:10]:
                print(f"  - {r.source.normalized} -[{r.relation_type.name}]-> "
                      f"{r.target.normalized} (w={r.weight:.2f})")

        # Build Graph
        graph = ConceptGraph(
            concepts=concepts, relations=relations,
            source_text=text,
            metadata={"n_concepts": len(concepts), "n_relations": len(relations)}
        )
        if self.verbose:
            centroid = graph.centroid()
            if centroid:
                print(f"[Engine] Discourse centroid: "
                      f"theta={centroid.theta:.3f}, phi={centroid.phi:.3f}, r={centroid.r:.3f}")
        return graph
