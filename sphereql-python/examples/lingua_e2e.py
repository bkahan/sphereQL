#!/usr/bin/env python3
"""
Lingua Spherica E2E - using the compiled sphereql package.

All types come from Rust via PyO3: no type duplication.
Requires: cd sphereql-python && maturin develop --features lingua
"""
import math
import sphereql

def main():
    print("=" * 70)
    print("  SPHEREQL LINGUA - E2E Demo (Rust-backed Python)")
    print("=" * 70)

    pipeline = sphereql.LinguaPipeline()

    texts = {
        "Meta/Formal": (
            "hey, so right now, we are using projection to map onto the sphereql space. "
            "i think that we need to revisit the sphere of spheres, where each sphere is "
            "a representation of that specific domain. i want to come up with a method to "
            "turn language (we can focus on english right now) into native sphereql units. "
            "this is a significant, non-trivial question and research topic. the end goal: "
            "be able to turn this conversation into a sphereql query that would capture "
            "the meaning mathematically. the universality of language and mathematics is "
            "closer than appears, as LLMs demonstrate.\n\n"
            "i suspect that this is more than a native algorithm, but a system of components "
            "working together. think in systems and think about how you, an LLM, interacts "
            "and 'understands' language.\n\n"
            "where: theta is the domain angle - longitude on a globe. Concepts in the same "
            "semantic neighborhood cluster at nearby theta values. 'Oncology' and 'cardiology' "
            "sit close together. 'Contract law' sits far away.\n\n"
            "phi is the abstraction level - latitude, but mapped to a hierarchy. Abstract "
            "principles float toward the north pole. 'Disease' lives near the top. "
            "'Stage-3 pancreatic adenocarcinoma' lives near the bottom.\n\n"
            "r (Radius) is epistemic weight - how central or significant is this concept? "
            "High-frequency, heavily-referenced concepts get a distinct radial value encoding "
            "their salience. It's a measure of how much work a concept does in the graph."
        ),
        "Medical": (
            "The patient presents with stage-3 pancreatic adenocarcinoma with hepatic "
            "metastasis. Oncology recommends aggressive chemotherapy using FOLFIRINOX "
            "protocol, while cardiology has flagged elevated troponin levels suggesting "
            "concurrent cardiac stress. The disease has progressed despite initial "
            "treatment with gemcitabine."
        ),
        "Philosophical": (
            "The relationship between language and mathematics reveals something profound "
            "about the nature of knowledge itself. When we say that a formal system can "
            "capture meaning, we are making an epistemological claim: that the structure "
            "of thought is, at some level, isomorphic to mathematical structure."
        ),
    }

    graphs = {}
    for label, text in texts.items():
        graph = pipeline.process(text)
        graphs[label] = graph
        c = graph.centroid()
        print(f"\n{'=' * 70}")
        print(f"  {label} ({len(text)} chars)")
        print(f"  Concepts: {graph.num_concepts}  Relations: {graph.num_relations}")
        if c:
            print(f"  Centroid: theta={c.theta:.3f} phi={c.phi:.3f} r={c.r:.3f}")
        top = sorted(graph.concepts(), key=lambda x: x.r or 0, reverse=True)[:5]
        print("  Top 5:")
        for con in top:
            print(f"    r={con.r:.3f} theta={con.theta:.3f} phi={con.phi:.3f}  {con.normalized}")

    # Validation
    print(f"\n{'=' * 70}")
    print("  VALIDATION")
    g = graphs["Meta/Formal"]
    passed = total = 0
    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        if cond: passed += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if detail: print(f"         {detail}")

    cp = lambda n: g.get_concept(n)
    onc, card, claw = cp("oncology"), cp("cardiology"), cp("contract law")
    dt = lambda a, b: min(abs(a-b) % (2*math.pi), 2*math.pi - abs(a-b) % (2*math.pi))
    d_oc = dt(onc.theta, card.theta)
    d_ol = dt(onc.theta, claw.theta)
    check("Oncology ~ Cardiology (domain)", d_oc < 0.5, f"dt={d_oc:.4f}")
    check("Oncology closer to Cardiology than Law", d_oc < d_ol)
    check("Disease phi < Oncology phi", cp("disease").phi < cp("oncology").phi)
    check("Disease phi < Stage-3 pancreatic adenocarcinoma phi",
          cp("disease").phi < cp("stage-3 pancreatic adenocarcinoma").phi)
    check("Language phi < English phi", cp("language").phi < cp("english").phi)
    check("SphereQL r > Graph r", cp("sphereql").r > cp("graph").r,
          f"sphereql={cp('sphereql').r:.3f} graph={cp('graph').r:.3f}")

    # Triangle inequality via Rust angular_distance
    mk = lambda n: sphereql.SphericalPoint(cp(n).r, cp(n).theta, cp(n).phi)
    dac = sphereql.angular_distance(mk("language"), mk("sphereql"))
    dab = sphereql.angular_distance(mk("language"), mk("llm"))
    dbc = sphereql.angular_distance(mk("llm"), mk("sphereql"))
    check("Triangle inequality", dac <= dab + dbc + 1e-9,
          f"d(l,s)={dac:.4f} <= d(l,llm)+d(llm,s)={dab+dbc:.4f}")

    print(f"\n  RESULT: {passed}/{total} checks passed")

    # SphereQL output
    print(f"\n{'=' * 70}")
    print("  SPHEREQL OUTPUT")
    print(g.to_sphereql())

    # Geodesic
    print(f"{'=' * 70}")
    print("  CENTRAL GEODESIC: language -> sphereql")
    print(f"  Direct: {dac:.4f} rad ({dac*180/math.pi:.1f} deg)")
    print(f"  Via LLM: {dab+dbc:.4f} rad ({(dab+dbc)*180/math.pi:.1f} deg)")
    print(f"  Overhead: {((dab+dbc)/dac-1)*100:.1f}%")

if __name__ == "__main__":
    main()
