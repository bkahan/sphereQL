//! Wikidata full-dump streaming source.
//!
//! Reads `latest-all.json.bz2` from
//! `https://dumps.wikimedia.org/wikidatawiki/entities/` — the
//! canonical JSON snapshot. The dump is a single bzip2-compressed
//! file (~150 GB compressed, ~1.5 TB expanded) shaped as one JSON
//! array, but newline-delimited: the first line is `[`, the last
//! line is `]`, and every other line is one entity object followed
//! by `,` (except the final entity, which has no trailing comma).
//!
//! Best for **50 M – 500 M** items. Streams via [`bzip2-rs`] in
//! constant memory, so the same machine that can handle a 10 GB
//! local download can finish a full extract in 24–48 h on a single
//! core. To shard across cores, run multiple binaries with disjoint
//! `(start_offset, max_items)` windows against the same dump.
//!
//! Why no per-item HTTP fallback: at 100 M Wikidata items, even a
//! generous 100 ms/request is 115 days of HTTP. The dump is the
//! only feasible source above the 1 M SPARQL ceiling.
//!
//! How to populate:
//!
//! ```bash
//! curl -OL https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
//! ```
//!
//! Then point the source at the file. Resume is supported via
//! `start_offset` — the source counts entities from the file head
//! and skips until it reaches the offset.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::bulk::{BulkItem, BulkSource, BulkSourceError, Claim};

/// Predicates we extract claims for. P31 = `instance of`, P279 =
/// `subclass of`, P361 = `part of`. Same three the SPARQL source
/// pulls — keeping them aligned means the hashed axes line up
/// across sources.
const CLAIM_PREDICATES: &[&str] = &["P31", "P279", "P361"];

#[derive(Debug, Clone)]
pub struct WikidataDumpConfig {
    pub dump_path: PathBuf,
    pub start_offset: usize,
    pub max_items: usize,
    pub only_items: bool,
    pub require_english_label: bool,
}

impl WikidataDumpConfig {
    pub fn new(dump_path: PathBuf) -> Self {
        Self {
            dump_path,
            start_offset: 0,
            max_items: usize::MAX,
            only_items: true,
            require_english_label: true,
        }
    }
}

pub struct WikidataDumpSource {
    cfg: WikidataDumpConfig,
    reader: Option<Box<dyn BufRead>>,
    n_consumed: usize,
    n_yielded: usize,
}

impl WikidataDumpSource {
    pub fn new(cfg: WikidataDumpConfig) -> Self {
        Self {
            cfg,
            reader: None,
            n_consumed: 0,
            n_yielded: 0,
        }
    }

    fn open(&mut self) -> Result<(), BulkSourceError> {
        if self.reader.is_some() {
            return Ok(());
        }
        let f = File::open(&self.cfg.dump_path)?;
        self.reader = Some(open_decompressed(f, &self.cfg.dump_path)?);
        Ok(())
    }

    fn next_entity_line(&mut self) -> Result<Option<String>, BulkSourceError> {
        self.open()?;
        let r = self.reader.as_mut().expect("reader present");
        loop {
            let mut buf = String::new();
            let n = r.read_line(&mut buf)?;
            if n == 0 {
                return Ok(None);
            }
            let trimmed = buf.trim();
            // Skip the array-bracket framing lines and any blank
            // lines so the JSON parser only ever sees object rows.
            if trimmed.is_empty() || trimmed == "[" || trimmed == "]" {
                continue;
            }
            // Trailing comma is part of the JSON-array framing, not
            // the entity object. Strip it so `serde_json` is happy.
            let cleaned = trimmed.strip_suffix(',').unwrap_or(trimmed);
            return Ok(Some(cleaned.to_string()));
        }
    }
}

impl Iterator for WikidataDumpSource {
    type Item = Result<BulkItem, BulkSourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n_yielded >= self.cfg.max_items {
            return None;
        }
        loop {
            let line = match self.next_entity_line() {
                Ok(Some(l)) => l,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            };
            self.n_consumed += 1;
            if self.n_consumed <= self.cfg.start_offset {
                continue;
            }
            match parse_entity_line(&line, &self.cfg) {
                Ok(Some(item)) => {
                    self.n_yielded += 1;
                    return Some(Ok(item));
                }
                Ok(None) => continue,
                Err(e) => {
                    // Skip malformed entities — one bad row mustn't
                    // kill a multi-day stream.
                    eprintln!("warning: wikidata dump parse: {e}");
                    continue;
                }
            }
        }
    }
}

impl BulkSource for WikidataDumpSource {
    fn source_name(&self) -> &str {
        "wikidata_dump"
    }
}

#[cfg(feature = "bulk-dump")]
fn open_decompressed(f: File, _path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    let r = bzip2_rs::DecoderReader::new(f);
    Ok(Box::new(BufReader::with_capacity(1 << 16, r)))
}

#[cfg(not(feature = "bulk-dump"))]
fn open_decompressed(_f: File, path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    Err(BulkSourceError::Io(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "bulk-dump feature is disabled; cannot read .bz2 dump at {}",
            path.display()
        ),
    )))
}

/// Parse one entity-object line from the Wikidata dump.
///
/// Returns `Ok(None)` when the row passes JSON parsing but fails
/// configured filters (non-item type, missing English label).
/// Returns `Err` only for unparseable JSON.
pub fn parse_entity_line(
    line: &str,
    cfg: &WikidataDumpConfig,
) -> Result<Option<BulkItem>, BulkSourceError> {
    if line.is_empty() {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(line)
        .map_err(|e| BulkSourceError::Parse(format!("wikidata dump line: {e}")))?;

    if cfg.only_items {
        let kind = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if kind != "item" {
            return Ok(None);
        }
    }

    let Some(id) = v.get("id").and_then(|x| x.as_str()) else {
        return Ok(None);
    };
    let id = id.to_string();

    let label = v
        .pointer("/labels/en/value")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    if cfg.require_english_label && label.is_none() {
        return Ok(None);
    }
    let description = v
        .pointer("/descriptions/en/value")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    let claims = extract_claims(&v);
    let sitelinks = count_sitelinks(&v);
    // Same scaling the SPARQL source uses, so the two paths produce
    // comparable confidence numbers when the same QID is ingested
    // from either.
    let conf = (((1.0 + sitelinks as f64).ln()) / 5.0).clamp(0.0, 1.0);

    Ok(Some(BulkItem {
        external_id: id.clone(),
        label: label.unwrap_or(id),
        description,
        claims,
        source_name: "wikidata_dump".into(),
        source_confidence: conf,
        category_hint: None,
        quality_hint: conf,
    }))
}

fn extract_claims(v: &Value) -> Vec<Claim> {
    let Some(obj) = v.get("claims").and_then(|x| x.as_object()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for pred in CLAIM_PREDICATES {
        let Some(arr) = obj.get(*pred).and_then(|x| x.as_array()) else {
            continue;
        };
        for c in arr {
            let Some(target) = c
                .pointer("/mainsnak/datavalue/value/id")
                .and_then(|x| x.as_str())
            else {
                continue;
            };
            out.push(Claim::new(*pred, target, 1.0));
        }
    }
    out
}

fn count_sitelinks(v: &Value) -> u64 {
    v.get("sitelinks")
        .and_then(|x| x.as_object())
        .map(|m| m.len() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_item() -> &'static str {
        r#"{
          "type": "item",
          "id": "Q42",
          "labels": {"en": {"language": "en", "value": "Douglas Adams"}},
          "descriptions": {"en": {"language": "en", "value": "English writer"}},
          "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
            "P106": [{"mainsnak": {"datavalue": {"value": {"id": "Q36180"}}}}],
            "P279": []
          },
          "sitelinks": {
            "enwiki": {"site": "enwiki", "title": "Douglas Adams"},
            "dewiki": {"site": "dewiki", "title": "Douglas Adams"}
          }
        }"#
    }

    fn cfg() -> WikidataDumpConfig {
        WikidataDumpConfig::new(PathBuf::from("/dev/null"))
    }

    #[test]
    fn parses_typical_item() {
        let item = parse_entity_line(sample_item(), &cfg())
            .expect("parse")
            .expect("filter-pass");
        assert_eq!(item.external_id, "Q42");
        assert_eq!(item.label, "Douglas Adams");
        assert_eq!(item.description, "English writer");
        assert_eq!(item.source_name, "wikidata_dump");
        // Only P31 in our whitelist; P106 is skipped, P279 is empty.
        assert_eq!(item.claims.len(), 1);
        assert_eq!(item.claims[0].predicate, "P31");
        assert_eq!(item.claims[0].object, "Q5");
    }

    #[test]
    fn confidence_grows_with_sitelinks() {
        let item = parse_entity_line(sample_item(), &cfg())
            .expect("parse")
            .expect("filter-pass");
        assert!(item.source_confidence > 0.0);
        let no_sitelinks = sample_item().replace(
            "\"sitelinks\": {\n            \"enwiki\": {\"site\": \"enwiki\", \"title\": \"Douglas Adams\"},\n            \"dewiki\": {\"site\": \"dewiki\", \"title\": \"Douglas Adams\"}\n          }",
            "\"sitelinks\": {}",
        );
        let bare = parse_entity_line(&no_sitelinks, &cfg())
            .expect("parse")
            .expect("filter-pass");
        assert!(bare.source_confidence < item.source_confidence);
    }

    #[test]
    fn drops_properties_when_only_items() {
        let prop = sample_item().replace("\"type\": \"item\"", "\"type\": \"property\"");
        let r = parse_entity_line(&prop, &cfg()).expect("parse");
        assert!(r.is_none());
    }

    #[test]
    fn includes_properties_when_only_items_disabled() {
        let prop = sample_item().replace("\"type\": \"item\"", "\"type\": \"property\"");
        let mut c = cfg();
        c.only_items = false;
        let r = parse_entity_line(&prop, &c).expect("parse");
        assert!(r.is_some());
    }

    #[test]
    fn drops_when_missing_label_and_required() {
        let no_label = sample_item().replace(
            "\"labels\": {\"en\": {\"language\": \"en\", \"value\": \"Douglas Adams\"}}",
            "\"labels\": {}",
        );
        let r = parse_entity_line(&no_label, &cfg()).expect("parse");
        assert!(r.is_none());
    }

    #[test]
    fn falls_back_to_id_when_label_not_required() {
        let no_label = sample_item().replace(
            "\"labels\": {\"en\": {\"language\": \"en\", \"value\": \"Douglas Adams\"}}",
            "\"labels\": {}",
        );
        let mut c = cfg();
        c.require_english_label = false;
        let item = parse_entity_line(&no_label, &c)
            .expect("parse")
            .expect("filter-pass");
        assert_eq!(item.label, "Q42");
    }

    #[test]
    fn malformed_json_returns_err() {
        let r = parse_entity_line("not-json", &cfg());
        assert!(matches!(r, Err(BulkSourceError::Parse(_))));
    }

    #[test]
    fn source_name_is_stable() {
        let s = WikidataDumpSource::new(cfg());
        assert_eq!(s.source_name(), "wikidata_dump");
    }
}
