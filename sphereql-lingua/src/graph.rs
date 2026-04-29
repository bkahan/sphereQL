use serde::{Deserialize, Serialize};
use sphereql_core::{SphericalPoint, angular_distance, cartesian_to_spherical, spherical_to_cartesian};

use crate::concept::Concept;
use crate::relation::Relation;
use crate::taxonomy::DomainTaxonomy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptGraph {
    pub concepts: Vec<Concept>,
    pub relations: Vec<Relation>,
    pub source_text: String,
}

impl ConceptGraph {
    pub fn new(concepts: Vec<Concept>, relations: Vec<Relation>, source_text: String) -> Self {
        Self { concepts, relations, source_text }
    }

    pub fn get_concept(&self, normalized: &str) -> Option<&Concept> {
        self.concepts.iter().find(|c| c.normalized == normalized)
    }

    pub fn get_concept_idx(&self, normalized: &str) -> Option<usize> {
        self.concepts.iter().position(|c| c.normalized == normalized)
    }

    pub fn centroid(&self) -> Option<SphericalPoint> {
        let resolved: Vec<&SphericalPoint> = self.concepts.iter().filter_map(|c| c.point.as_ref()).collect();
        if resolved.is_empty() { return None; }
        let n = resolved.len() as f64;
        let (mut cx, mut cy, mut cz, mut avg_r) = (0.0, 0.0, 0.0, 0.0);
        for p in &resolved {
            let c = spherical_to_cartesian(p);
            cx += c.x; cy += c.y; cz += c.z; avg_r += p.r;
        }
        cx /= n; cy /= n; cz /= n; avg_r /= n;
        let pt = cartesian_to_spherical(&sphereql_core::CartesianPoint::new(cx, cy, cz));
        Some(SphericalPoint::new_unchecked(avg_r, pt.theta, pt.phi))
    }

    pub fn to_sphereql(&self, taxonomy: &DomainTaxonomy) -> String {
        let mut out = String::new();
        out.push_str("-- Lingua Spherica -> SphereQL Output\n");
        out.push_str(&format!("-- Concepts: {}  Relations: {}\n\n", self.concepts.len(), self.relations.len()));
        if let Some(c) = self.centroid() {
            let dom = taxonomy.domain_name(c.theta);
            out.push_str(&format!("DECLARE SPHERE discourse (\n  center: ({:.4}, {:.4}, {:.4}),\n  domain: \"{dom}\"\n);\n\n", c.theta, c.phi, c.r));
        }
        let mut sorted: Vec<usize> = (0..self.concepts.len()).filter(|&i| self.concepts[i].point.is_some()).collect();
        sorted.sort_by(|&a, &b| { let ra = self.concepts[a].point.as_ref().unwrap().r; let rb = self.concepts[b].point.as_ref().unwrap().r; rb.partial_cmp(&ra).unwrap() });
        for &i in &sorted {
            let c = &self.concepts[i]; let p = c.point.as_ref().unwrap();
            let name = safe_ident(&c.normalized); let dom = taxonomy.domain_name(p.theta);
            out.push_str(&format!("NODE {name} AT (t: {:.4}, p: {:.4}, r: {:.4})  -- {dom}\n", p.theta, p.phi, p.r));
        }
        out.push('\n');
        for rel in &self.relations {
            let src = safe_ident(&self.concepts[rel.source_idx].normalized);
            let tgt = safe_ident(&self.concepts[rel.target_idx].normalized);
            out.push_str(&format!("EDGE {src} -[{:?}]-> {tgt} WITH weight: {:.2}", rel.relation_type, rel.weight));
            if let (Some(sp), Some(tp)) = (self.concepts[rel.source_idx].point.as_ref(), self.concepts[rel.target_idx].point.as_ref()) {
                out.push_str(&format!("  -- geodesic: {:.4} rad", angular_distance(sp, tp)));
            }
            out.push('\n');
        }
        out.push('\n');
        if let (Some(li), Some(si)) = (self.get_concept_idx("language"), self.get_concept_idx("sphereql")) {
            let llm = self.get_concept_idx("llm").map(|_| "       PASSING THROUGH llm\n".to_string()).unwrap_or_default();
            out.push_str(&format!("QUERY central_transformation {{\n  FIND GEODESIC FROM {src} TO {tgt}\n{llm}  WEIGHTED BY r\n}};\n",
                src=safe_ident(&self.concepts[li].normalized), tgt=safe_ident(&self.concepts[si].normalized), llm=llm));
        }
        out
    }
}

fn safe_ident(s: &str) -> String {
    s.replace(' ',"_").replace('-',"_").replace('\'',"").chars().filter(|c| c.is_alphanumeric() || *c == '_').collect::<String>().to_lowercase()
}
