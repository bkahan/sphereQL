//! E2E demo: 3 texts, validation checks, SphereQL output, geodesic trace.
//! Run: cargo run --example e2e -p sphereql-lingua

use sphereql_core::angular_distance;
use sphereql_lingua::{LinguaPipeline, ConceptGraph};
use sphereql_lingua::taxonomy::theta_distance;
use std::f64::consts::PI;

const TEXT_META: &str = "\
hey, so right now, we are using projection to map onto the sphereql space. \
i think that we need to revisit the sphere of spheres, where each sphere is \
a representation of that specific domain. i want to come up with a method to \
turn language (we can focus on english right now) into native sphereql units. \
this is a significant, non-trivial question and research topic. the end goal: \
be able to turn this conversation into a sphereql query that would capture \
the meaning mathematically. the universality of language and mathematics is \
closer than appears, as LLMs demonstrate.\n\n\
i suspect that this is more than a native algorithm, but a system of components \
working together. think in systems and think about how you, an LLM, interacts \
and 'understands' language.\n\n\
where: theta is the domain angle - longitude on a globe. Concepts in the same \
semantic neighborhood cluster at nearby theta values. 'Oncology' and 'cardiology' \
sit close together. 'Contract law' sits far away. Theta is your horizontal \
clustering dimension.\n\n\
phi is the abstraction level - latitude, but mapped to a hierarchy. Abstract \
principles float toward the north pole. Specific, concrete instances sink toward \
the south. 'Disease' lives near the top. 'Stage-3 pancreatic adenocarcinoma' \
lives near the bottom. Phi encodes where a concept sits in its own ontological \
hierarchy.\n\n\
r (Radius) is epistemic weight - how central or significant is this concept? \
High-frequency, heavily-referenced concepts get a distinct radial value encoding \
their salience. It's a measure of how much work a concept does in the graph.";

const TEXT_MEDICAL: &str = "\
The patient presents with stage-3 pancreatic adenocarcinoma with hepatic \
metastasis. Oncology recommends aggressive chemotherapy using FOLFIRINOX \
protocol, while cardiology has flagged elevated troponin levels suggesting \
concurrent cardiac stress. The disease has progressed despite initial \
treatment with gemcitabine.";

const TEXT_PHILOSOPHY: &str = "\
The relationship between language and mathematics reveals something profound \
about the nature of knowledge itself. When we say that a formal system can \
capture meaning, we are making an epistemological claim: that the structure \
of thought is, at some level, isomorphic to mathematical structure. This \
is not reductionism - it is recognition that abstraction and representation \
are universal operations of mind.";

fn pt(g: &ConceptGraph, name: &str) -> sphereql_core::SphericalPoint {
    g.get_concept(name).unwrap_or_else(|| panic!("'{name}' not found")).point.expect("unresolved")
}

fn main() {
    println!("{'='repeat 70}");
    println!("  SPHEREQL-LINGUA  End-to-End Rust Demonstration");
    println!("{'='repeat 70}\n");

    let pipeline = LinguaPipeline::new();
    let texts = [("Meta/Formal", TEXT_META), ("Medical", TEXT_MEDICAL), ("Philosophical", TEXT_PHILOSOPHY)];
    let mut graphs = Vec::new();

    for (label, text) in &texts {
        let g = pipeline.process(text);
        let n = g.concepts.iter().filter(|c| c.point.is_some()).count();
        println!("-- {label} ({} chars) --", text.len());
        if let Some(c) = g.centroid() {
            let dom = pipeline.taxonomy().domain_name(c.theta);
            let abs = if c.phi < PI/2.0 { "Abstract" } else { "Concrete" };
            println!("  Centroid: t={:.3} p={:.3} r={:.3} [{dom}, {abs}]", c.theta, c.phi, c.r);
        }
        println!("  Concepts: {n}  Relations: {}", g.relations.len());
        let mut top: Vec<_> = g.concepts.iter().filter_map(|c| c.point.map(|p| (&c.normalized, p))).collect();
        top.sort_by(|a, b| b.1.r.partial_cmp(&a.1.r).unwrap());
        for (name, p) in top.iter().take(5) {
            println!("    r={:.3} t={:.3} p={:.3}  {:<28} [{}]", p.r, p.theta, p.phi, name, pipeline.taxonomy().domain_name(p.theta));
        }
        println!();
        graphs.push(g);
    }

    // Validation
    println!("-- VALIDATION --\n");
    let (mut pass, mut total) = (0u32, 0u32);
    macro_rules! check {
        ($n:expr, $c:expr, $d:expr) => {{ total += 1; if $c { pass += 1; } println!("  [{}] {}", if $c {"PASS"} else {"FAIL"}, $n); println!("         {}", $d); }};
    }
    let g = &graphs[0];
    let dt_oc = theta_distance(pt(g,"oncology").theta, pt(g,"cardiology").theta);
    let dt_ol = theta_distance(pt(g,"oncology").theta, pt(g,"contract law").theta);
    check!("Oncology ~ Cardiology", dt_oc < 0.5, format!("dt={dt_oc:.4}"));
    check!("Oncology closer to Cardiology than Law", dt_oc < dt_ol, format!("{dt_oc:.4} < {dt_ol:.4}"));
    check!("Disease phi < Oncology phi", pt(g,"disease").phi < pt(g,"oncology").phi,
           format!("{:.3} < {:.3}", pt(g,"disease").phi, pt(g,"oncology").phi));
    check!("Disease phi < Stage-3", pt(g,"disease").phi < pt(g,"stage-3 pancreatic adenocarcinoma").phi,
           format!("{:.3} < {:.3}", pt(g,"disease").phi, pt(g,"stage-3 pancreatic adenocarcinoma").phi));
    check!("Language phi < English phi", pt(g,"language").phi < pt(g,"english").phi,
           format!("{:.3} < {:.3}", pt(g,"language").phi, pt(g,"english").phi));
    check!("Universality phi < Mathematics phi", pt(g,"universality").phi < pt(g,"mathematics").phi,
           format!("{:.3} < {:.3}", pt(g,"universality").phi, pt(g,"mathematics").phi));
    check!("SphereQL r > Graph r", pt(g,"sphereql").r > pt(g,"graph").r,
           format!("{:.3} > {:.3}", pt(g,"sphereql").r, pt(g,"graph").r));
    let (lang, llm, sph) = (pt(g,"language"), pt(g,"llm"), pt(g,"sphereql"));
    let (dac, dab, dbc) = (angular_distance(&lang, &sph), angular_distance(&lang, &llm), angular_distance(&llm, &sph));
    check!("Triangle inequality", dac <= dab + dbc + 1e-9,
           format!("{dac:.4} <= {:.4}", dab + dbc));

    println!("\n  RESULT: {pass}/{total} passed\n");

    // SphereQL output
    println!("-- SPHEREQL OUTPUT --\n");
    println!("{}", graphs[0].to_sphereql(pipeline.taxonomy()));

    // Geodesic
    println!("-- GEODESIC: language -> sphereql --\n");
    println!("  Direct: {dac:.4} rad ({:.1} deg)", dac * 180.0 / PI);
    println!("  Via LLM: {:.4} rad ({:.1} deg, {:.1}% overhead)", dab+dbc, (dab+dbc)*180.0/PI, ((dab+dbc)/dac-1.0)*100.0);
    for i in 0..=10 {
        let t = i as f64 / 10.0;
        let p = sphereql_core::full_slerp(&lang, &sph, t);
        println!("    t={t:.1} t={:.4} p={:.4} r={:.4} [{}]", p.theta, p.phi, p.r, pipeline.taxonomy().domain_name(p.theta));
    }
}
