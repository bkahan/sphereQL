//! DBpedia n-triples streaming source.
//!
//! DBpedia ships a per-language snapshot of Wikipedia restructured
//! as RDF triples. Three small bz2 files give us everything we need
//! for a broad, encyclopedic corpus:
//!
//! - `instance-types_lang=en.ttl.bz2` (~80 MB) — `<entity> a <type>`
//!   triples, one per (entity, ontology class) pair.
//! - `mappingbased-objects_lang=en.ttl.bz2` (~250 MB) — typed object
//!   links like `<Barack_Obama> <birthPlace> <Honolulu>`. These map
//!   one-for-one to our `Claim` shape.
//! - `labels_lang=en.ttl.bz2` (~100 MB) — `<entity> rdfs:label "X"@en`.
//!
//! Best for **10 K – 5 M** items. The full English DBpedia has
//! ~6 M entities and these three files sum to ~1 GB compressed, so a
//! laptop can ingest it in a single pass without a Wikidata-dump-sized
//! download. Above 5 M items the Wikidata JSON dump is the right tool.
//!
//! How to populate (one-time, ~1 GB):
//!
//! ```bash
//! mkdir -p /tmp/dbpedia
//! BASE=https://databus.dbpedia.org/dbpedia/snapshot/2022.12.01
//! curl -L -o /tmp/dbpedia/instance-types_lang=en.ttl.bz2 \
//!     $BASE/instance-types_lang=en.ttl.bz2
//! curl -L -o /tmp/dbpedia/mappingbased-objects_lang=en.ttl.bz2 \
//!     $BASE/mappingbased-objects_lang=en.ttl.bz2
//! curl -L -o /tmp/dbpedia/labels_lang=en.ttl.bz2 \
//!     $BASE/labels_lang=en.ttl.bz2
//! ```
//!
//! Then point the source at the directory — the actual filenames are
//! auto-discovered by suffix.
//!
//! The streaming model: each call to [`DBpediaTtlSource::next`] before
//! the first hit triggers a lazy "load" that reads all three files
//! into a single in-memory join keyed on entity IRI. We sample
//! entities deterministically by FNV hash so a `max_items` budget
//! gives a stable, diversity-friendly cross-section of DBpedia rather
//! than the alphabetic prefix you'd get from a naive head-N read. The
//! actual yield phase is then an `O(n)` walk over the sampled vector.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::bulk::{BulkItem, BulkSource, BulkSourceError, Claim};

/// Default oversample factor — keep this many entities for every
/// requested `max_items`, then filter to those with both a label and
/// at least one claim before truncating. ~30 % of DBpedia entities
/// have empty mappingbased-objects rows, so 4× is comfortable.
const DEFAULT_OVERSAMPLE: usize = 4;

#[derive(Debug, Clone)]
pub struct DBpediaConfig {
    /// Directory containing the three .ttl.bz2 files.
    pub dir: PathBuf,
    pub start_offset: usize,
    pub max_items: usize,
    /// Multiplier on `max_items` for the in-memory working set, to
    /// absorb entities that get filtered for missing labels/claims.
    pub oversample: usize,
}

impl DBpediaConfig {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            start_offset: 0,
            max_items: usize::MAX,
            oversample: DEFAULT_OVERSAMPLE,
        }
    }
}

pub struct DBpediaTtlSource {
    cfg: DBpediaConfig,
    items: Option<Vec<BulkItem>>,
    cursor: usize,
}

impl DBpediaTtlSource {
    pub fn new(cfg: DBpediaConfig) -> Self {
        Self {
            cfg,
            items: None,
            cursor: 0,
        }
    }

    fn load(&self) -> Result<Vec<BulkItem>, BulkSourceError> {
        let types_path = find_file(&self.cfg.dir, "instance-types")?;
        let objects_path = find_file(&self.cfg.dir, "mappingbased-objects")?;
        let labels_path = find_file(&self.cfg.dir, "labels")?;

        // Sample budget. usize::MAX × 4 overflows, so saturate.
        let working_cap = self
            .cfg
            .max_items
            .saturating_mul(self.cfg.oversample.max(1));

        let mut builders: HashMap<String, EntityBuilder> = HashMap::new();
        let mut entities_seen: usize = 0;

        // Pass 1: instance-types. Decide which entities to keep using
        // a stable FNV-1a-mod-N threshold so the working set is
        // deterministic across runs and spread across the alphabet.
        let mut keep_threshold: Option<u64> = None;
        for_each_triple(&types_path, |s, p, o| {
            entities_seen += 1;
            if !is_type_predicate(p) {
                return;
            }
            // Once we've seen enough entities to estimate the file
            // size, lock in a hash threshold that yields ~working_cap
            // entities. We sample by entity IRI hash so all triples
            // for one entity agree.
            if working_cap < usize::MAX
                && keep_threshold.is_none()
                && entities_seen > working_cap.saturating_mul(50)
            {
                // ~6M entities / working_cap target → keep ratio.
                let est_total = 6_000_000u64;
                let ratio = (working_cap as u64).max(1).min(est_total);
                keep_threshold = Some(((u64::MAX as u128) * (ratio as u128) / est_total as u128) as u64);
            }
            if let Some(thr) = keep_threshold
                && fnv1a64(s.as_bytes()) > thr
            {
                return;
            }
            if let Some(local) = ontology_local_name(o) {
                builders
                    .entry(s.to_string())
                    .or_insert_with(EntityBuilder::default)
                    .types
                    .push(local.to_string());
            }
            if working_cap < usize::MAX && builders.len() >= working_cap.saturating_mul(2) {
                // Stop slurping new entities once we have ≥ 2× the
                // sample budget. Existing entries can still grow.
                keep_threshold.get_or_insert(0);
            }
        })?;

        // Pass 2: labels. Only enrich entities we kept.
        for_each_triple(&labels_path, |s, _p, o| {
            if let Some(b) = builders.get_mut(s)
                && let Some(lit) = parse_literal_en(o)
            {
                b.label = Some(lit.to_string());
            }
        })?;

        // Pass 3: mappingbased-objects. Map dbo:<prop> + dbr:<obj> to
        // (predicate, object) claims; collect on retained entities.
        for_each_triple(&objects_path, |s, p, o| {
            let Some(b) = builders.get_mut(s) else {
                return;
            };
            let Some(pred) = ontology_local_name(p) else {
                return;
            };
            let Some(obj) = resource_local_name(o) else {
                return;
            };
            b.claims.push(Claim::new(pred, obj, 1.0));
        })?;

        // Materialize. Drop entities with no label or no claims.
        // Synthesize a quality_hint from claim count so denser items
        // float to the top of self-tune.
        let mut items: Vec<BulkItem> = Vec::with_capacity(builders.len());
        for (iri, b) in builders {
            let Some(label) = b.label else {
                continue;
            };
            if b.claims.is_empty() {
                continue;
            }
            // Synthetic claims from instance-types so the type
            // hierarchy participates in axis hashing. P31 mirrors the
            // Wikidata predicate name so cross-source corpora line
            // their axes up.
            let mut claims = b.claims;
            for t in &b.types {
                claims.push(Claim::new("P31", t, 1.0));
            }
            let category_hint = b.types.first().cloned();
            // log-scaled density → quality_hint in [0, 1].
            let q = ((1.0 + claims.len() as f64).ln() / 4.0).clamp(0.0, 1.0);
            items.push(BulkItem {
                external_id: resource_local_name(&iri).unwrap_or(&iri).to_string(),
                label,
                description: String::new(),
                claims,
                source_name: "dbpedia".into(),
                source_confidence: q,
                category_hint,
                quality_hint: q,
            });
        }
        // Stable order so two runs against the same inputs produce
        // the same Parquet rows. Sort by external_id.
        items.sort_by(|a, b| a.external_id.cmp(&b.external_id));
        if items.len() > self.cfg.max_items {
            items.truncate(self.cfg.max_items);
        }
        Ok(items)
    }
}

impl Iterator for DBpediaTtlSource {
    type Item = Result<BulkItem, BulkSourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.items.is_none() {
            match self.load() {
                Ok(items) => self.items = Some(items),
                Err(e) => return Some(Err(e)),
            }
        }
        let items = self.items.as_ref().expect("items loaded");
        loop {
            if self.cursor >= items.len() {
                return None;
            }
            let idx = self.cursor;
            self.cursor += 1;
            if idx < self.cfg.start_offset {
                continue;
            }
            return Some(Ok(items[idx].clone()));
        }
    }
}

impl BulkSource for DBpediaTtlSource {
    fn source_name(&self) -> &str {
        "dbpedia"
    }
}

#[derive(Default)]
struct EntityBuilder {
    label: Option<String>,
    types: Vec<String>,
    claims: Vec<Claim>,
}

fn find_file(dir: &Path, prefix: &str) -> Result<PathBuf, BulkSourceError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(n) = name.to_str() else { continue };
        if n.starts_with(prefix) && (n.ends_with(".ttl.bz2") || n.ends_with(".nt.bz2")) {
            return Ok(entry.path());
        }
    }
    Err(BulkSourceError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("no {prefix}*.ttl.bz2 file in {}", dir.display()),
    )))
}

fn for_each_triple<F>(path: &Path, mut visit: F) -> Result<(), BulkSourceError>
where
    F: FnMut(&str, &str, &str),
{
    let f = File::open(path)?;
    let reader = open_decompressed(f, path)?;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((s, p, o)) = parse_triple(trimmed) {
            visit(s, p, o);
        }
    }
    Ok(())
}

/// Parse one N-Triples-style line:
///   `<subject> <predicate> <object-or-literal> .`
/// Returns owned `&str` slices into the input. Tolerates extra
/// whitespace but does not attempt to handle multi-line triples
/// (DBpedia exports never produce them).
fn parse_triple(line: &str) -> Option<(&str, &str, &str)> {
    let line = line.strip_suffix('.')?.trim_end();
    let s_start = line.find('<')?;
    let s_end = line[s_start + 1..].find('>')?;
    let s = &line[s_start + 1..s_start + 1 + s_end];
    let rest = &line[s_start + 1 + s_end + 1..].trim_start();
    let p_start = rest.find('<')?;
    let p_end = rest[p_start + 1..].find('>')?;
    let p = &rest[p_start + 1..p_start + 1 + p_end];
    let o_rest = rest[p_start + 1 + p_end + 1..].trim_start();
    // Object: either <IRI> or "literal"@lang or "literal"^^<type>.
    if let Some(stripped) = o_rest.strip_prefix('<') {
        let end = stripped.find('>')?;
        return Some((s, p, &stripped[..end]));
    }
    if o_rest.starts_with('"') {
        // Cheap literal parse: trust DBpedia's escaping and find the
        // last unescaped quote on the line.
        let after_open = &o_rest[1..];
        let close = find_closing_quote(after_open)?;
        return Some((s, p, &after_open[..close]));
    }
    None
}

fn find_closing_quote(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

fn parse_literal_en<'a>(o: &'a str) -> Option<&'a str> {
    // After parse_triple stripped the surrounding quotes, the lang
    // tag (if any) was already discarded. So `o` is the literal
    // body. We accept everything — DBpedia's `labels_lang=en` file
    // is already filtered to English.
    Some(o)
}

const TYPE_PRED: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

fn is_type_predicate(p: &str) -> bool {
    p == TYPE_PRED
}

const ONTOLOGY_PREFIX: &str = "http://dbpedia.org/ontology/";
const RESOURCE_PREFIX: &str = "http://dbpedia.org/resource/";

fn ontology_local_name(iri: &str) -> Option<&str> {
    iri.strip_prefix(ONTOLOGY_PREFIX)
}

fn resource_local_name(iri: &str) -> Option<&str> {
    iri.strip_prefix(RESOURCE_PREFIX)
}

/// 64-bit FNV-1a. Same hash family the axis extractor uses; reused
/// here so the entity-sampling decision is stable across runs.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(feature = "bulk-dbpedia")]
fn open_decompressed(f: File, _path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    let r = bzip2_rs::DecoderReader::new(f);
    Ok(Box::new(BufReader::with_capacity(1 << 16, r)))
}

#[cfg(not(feature = "bulk-dbpedia"))]
fn open_decompressed(_f: File, path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    Err(BulkSourceError::Io(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "bulk-dbpedia feature is disabled; cannot read DBpedia .ttl.bz2 at {}",
            path.display()
        ),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_resource_triple() {
        let line = "<http://dbpedia.org/resource/Barack_Obama> <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/Honolulu> .";
        let (s, p, o) = parse_triple(line).expect("parse");
        assert_eq!(s, "http://dbpedia.org/resource/Barack_Obama");
        assert_eq!(p, "http://dbpedia.org/ontology/birthPlace");
        assert_eq!(o, "http://dbpedia.org/resource/Honolulu");
    }

    #[test]
    fn parses_literal_triple() {
        let line = r#"<http://dbpedia.org/resource/Barack_Obama> <http://www.w3.org/2000/01/rdf-schema#label> "Barack Obama"@en ."#;
        let (s, p, o) = parse_triple(line).expect("parse");
        assert_eq!(s, "http://dbpedia.org/resource/Barack_Obama");
        assert_eq!(p, "http://www.w3.org/2000/01/rdf-schema#label");
        assert_eq!(o, "Barack Obama");
    }

    #[test]
    fn skips_comments_and_blanks() {
        assert!(parse_triple("# comment line").is_none());
        assert!(parse_triple("").is_none());
    }

    #[test]
    fn detects_local_names() {
        assert_eq!(
            ontology_local_name("http://dbpedia.org/ontology/Person"),
            Some("Person")
        );
        assert_eq!(
            resource_local_name("http://dbpedia.org/resource/Q42"),
            Some("Q42")
        );
        assert!(ontology_local_name("http://other.example/Person").is_none());
    }

    #[test]
    fn fnv_is_deterministic() {
        assert_eq!(fnv1a64(b"abc"), fnv1a64(b"abc"));
        assert_ne!(fnv1a64(b"abc"), fnv1a64(b"abd"));
    }

    #[test]
    fn source_name_is_stable() {
        let s = DBpediaTtlSource::new(DBpediaConfig::new(PathBuf::from("/nowhere")));
        assert_eq!(s.source_name(), "dbpedia");
    }
}
