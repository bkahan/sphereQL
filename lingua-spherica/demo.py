#!/usr/bin/env python3
"""
Lingua Spherica — Demonstration
=================================

Maps the actual conversation that spawned this system into SphereQL.

This is the self-referential proof: the text asking "can language become
spherical coordinates?" is itself transformed into spherical coordinates.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingua_spherica.engine import LinguaSphericaEngine


CONVERSATION = """
hey, so right now, we are using projection to map onto the sphereql space.
i think that we need to revisit the sphere of spheres, where each sphere is
a representation of that specific domain. i want to come up with a method to
turn language (we can focus on english right now) into native sphereql units.
this is a significant, non-trivial question and research topic. the end goal:
be able to turn this conversation into a sphereql query that would capture
the meaning mathematically. the universality of language and mathematics is
closer than appears, as LLMs demonstrate.

i suspect that this is more than a native algorithm, but a system of components
working together. think in systems and think about how you, an LLM, interacts
and 'understands' language.

where: theta is the domain angle - longitude on a globe. Concepts in the same
semantic neighborhood cluster at nearby theta values. 'Oncology' and 'cardiology'
sit close together. 'Contract law' sits far away. Theta is your horizontal
clustering dimension.

phi is the abstraction level - latitude, but mapped to a hierarchy. Abstract
principles float toward the north pole. Specific, concrete instances sink toward
the south. 'Disease' lives near the top. 'Stage-3 pancreatic adenocarcinoma'
lives near the bottom. Phi encodes where a concept sits in its own ontological
hierarchy.

r (Radius) is epistemic weight - how central or significant is this concept?
High-frequency, heavily-referenced concepts get a distinct radial value encoding
their salience. It's a measure of how much work a concept does in the graph.
""".strip()


def main():
    print("=" * 70)
    print("  LINGUA SPHERICA - Language to SphereQL Demonstration")
    print("=" * 70)
    print()
    print(f"Input: {len(CONVERSATION)} characters of natural language")
    print(f"Source: The conversation that proposed this system")
    print()
    print("-" * 70)
    print()

    engine = LinguaSphericaEngine(verbose=True)

    print("Building concept graph...")
    print()
    graph = engine.build_graph(CONVERSATION)

    print()
    print("-" * 70)
    print()

    print("Composing SphereQL output...")
    print()

    sphereql_output = engine.compositor.compose(graph)
    print(sphereql_output)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "this_conversation.sphereql")
    with open(output_path, 'w') as f:
        f.write(sphereql_output)
    print(f"\n[Written to {output_path}]")

    print("\n" + "=" * 70)
    print("  GEOMETRIC ANALYSIS")
    print("=" * 70)

    from lingua_spherica.coordinates import angular_distance, slerp

    language = graph.get_concept("language")
    sphereql = graph.get_concept("sphereql")
    llm = graph.get_concept("llm")

    if language and sphereql and language.point and sphereql.point:
        d = angular_distance(language.point, sphereql.point)
        print(f"\n  Central Geodesic: language -> sphereql")
        print(f"    Angular distance: {d:.4f} radians ({d*180/3.14159:.1f} deg)")
        print(f"    language:  (t={language.point.theta:.4f}, "
              f"p={language.point.phi:.4f}, r={language.point.r:.4f})")
        print(f"    sphereql:  (t={sphereql.point.theta:.4f}, "
              f"p={sphereql.point.phi:.4f}, r={sphereql.point.r:.4f})")

        if llm and llm.point:
            d1 = angular_distance(language.point, llm.point)
            d2 = angular_distance(llm.point, sphereql.point)
            print(f"    via LLM:   (t={llm.point.theta:.4f}, "
                  f"p={llm.point.phi:.4f}, r={llm.point.r:.4f})")
            print(f"    language->LLM: {d1:.4f} rad, LLM->sphereql: {d2:.4f} rad")
            print(f"    path length via LLM: {d1+d2:.4f} rad "
                  f"(vs direct: {d:.4f} rad)")

        print(f"\n  Geodesic Waypoints (language -> sphereql):")
        for i in range(6):
            t = i / 5.0
            pt = slerp(language.point, sphereql.point, t)
            print(f"    t={t:.1f}: (t={pt.theta:.4f}, p={pt.phi:.4f}, r={pt.r:.4f})")

    print(f"\n  Abstraction Hierarchy (phi from abstract->concrete):")
    resolved = [c for c in graph.concepts if c.point is not None]
    sorted_by_phi = sorted(resolved, key=lambda c: c.point.phi)
    for c in sorted_by_phi:
        phi_pct = c.point.phi / 3.14159 * 100
        bar_len = int(phi_pct / 2)
        bar = "#" * bar_len + "." * (50 - bar_len)
        print(f"    {c.normalized:40s} phi={c.point.phi:.3f} [{bar}]")

    print()


if __name__ == "__main__":
    main()
