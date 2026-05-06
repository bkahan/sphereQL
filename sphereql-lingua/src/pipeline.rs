use sphereql_core::SphericalPoint;

use crate::abstraction::AbstractionResolver;
use crate::concept::{ConceptExtractor, RegexExtractor};
use crate::graph::ConceptGraph;
use crate::relation::RelationEncoder;
use crate::salience::SalienceScorer;
use crate::taxonomy::DomainTaxonomy;

/// Six-stage pipeline: Text → ConceptGraph with resolved (θ, φ, r) coordinates.
///
/// ```no_run
/// use sphereql_lingua::LinguaPipeline;
///
/// let pipeline = LinguaPipeline::new();
/// let graph = pipeline.process("your text here");
/// println!("{}", graph.to_sphereql(pipeline.taxonomy()));
/// ```
pub struct LinguaPipeline {
    extractor: Box<dyn ConceptExtractor>,
    taxonomy: DomainTaxonomy,
    abstraction: AbstractionResolver,
    salience: SalienceScorer,
    relations: RelationEncoder,
}

impl Default for LinguaPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl LinguaPipeline {
    pub fn new() -> Self {
        Self {
            extractor: Box::new(RegexExtractor::new()),
            taxonomy: DomainTaxonomy::new(),
            abstraction: AbstractionResolver::new(),
            salience: SalienceScorer::new(),
            relations: RelationEncoder::default(),
        }
    }

    /// Swap in a custom concept extractor (e.g. LLM-backed).
    pub fn with_extractor(mut self, extractor: Box<dyn ConceptExtractor>) -> Self {
        self.extractor = extractor;
        self
    }

    /// Access the taxonomy (needed for `graph.to_sphereql()`).
    pub fn taxonomy(&self) -> &DomainTaxonomy {
        &self.taxonomy
    }

    /// Full pipeline: text → ConceptGraph.
    pub fn process(&self, text: &str) -> ConceptGraph {
        // Stage 1: extract concepts
        let mut concepts = self.extractor.extract(text);

        // Stage 2: assign θ (domain angle)
        for c in &mut concepts {
            let theta = self.taxonomy.assign_theta(c);
            c.point = Some(SphericalPoint::new_unchecked(0.5, theta, 0.0));
        }

        // Stage 3: assign φ (abstraction level)
        for c in &mut concepts {
            let phi = self.abstraction.assign_phi(c, text);
            if let Some(ref mut p) = c.point {
                *p = SphericalPoint::new_unchecked(p.r, p.theta, phi);
            }
        }

        // Stage 4: assign r (epistemic weight)
        self.salience.score_all(&mut concepts, text);
        for c in &mut concepts {
            let r = self.salience.salience_to_r(c.salience_score);
            if let Some(ref mut p) = c.point {
                *p = SphericalPoint::new_unchecked(r, p.theta, p.phi);
            }
        }

        // Stage 5: extract relations
        let relations = self.relations.extract(&concepts, text);

        // Stage 6: assemble graph
        ConceptGraph::new(concepts, relations, text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taxonomy::theta_distance;
    use sphereql_core::angular_distance;

    const TEST_TEXT: &str = "\
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

    fn get_pt(graph: &ConceptGraph, name: &str) -> SphericalPoint {
        graph.get_concept(name).unwrap().point.unwrap()
    }

    #[test]
    fn domain_proximity_oncology_cardiology() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        let dt = theta_distance(get_pt(&g, "oncology").theta, get_pt(&g, "cardiology").theta);
        assert!(
            dt < 0.5,
            "oncology and cardiology should be nearby: dt={dt}"
        );
    }

    #[test]
    fn domain_separation_oncology_contract_law() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        let onc = get_pt(&g, "oncology").theta;
        let claw = get_pt(&g, "contract law").theta;
        let card = get_pt(&g, "cardiology").theta;
        let d_oc = theta_distance(onc, claw);
        let d_ok = theta_distance(onc, card);
        assert!(
            d_ok < d_oc,
            "oncology should be closer to cardiology than to contract law"
        );
    }

    #[test]
    fn abstraction_disease_over_oncology() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        assert!(get_pt(&g, "disease").phi < get_pt(&g, "oncology").phi);
    }

    #[test]
    fn abstraction_disease_over_stage3() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        assert!(get_pt(&g, "disease").phi < get_pt(&g, "stage-3 pancreatic adenocarcinoma").phi);
    }

    #[test]
    fn abstraction_language_over_english() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        assert!(get_pt(&g, "language").phi < get_pt(&g, "english").phi);
    }

    #[test]
    fn salience_sphereql_above_graph() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        assert!(get_pt(&g, "sphereql").r > get_pt(&g, "graph").r);
    }

    #[test]
    fn triangle_inequality_on_sphere() {
        let g = LinguaPipeline::new().process(TEST_TEXT);
        let lang = get_pt(&g, "language");
        let llm = get_pt(&g, "llm");
        let sph = get_pt(&g, "sphereql");
        let dac = angular_distance(&lang, &sph);
        let dab = angular_distance(&lang, &llm);
        let dbc = angular_distance(&llm, &sph);
        assert!(dac <= dab + dbc + 1e-9, "triangle inequality violated");
    }

    #[test]
    fn sphereql_output_nonempty() {
        let p = LinguaPipeline::new();
        let g = p.process(TEST_TEXT);
        let out = g.to_sphereql(p.taxonomy());
        assert!(out.contains("NODE"));
        assert!(out.contains("EDGE"));
        assert!(out.contains("sphereql"));
    }
}
