//! OpenAlex Works snapshot shard reader.
//!
//! OpenAlex publishes a public S3 snapshot
//! (`s3://openalex/data/works/`) split into ~25 GB of compressed
//! shards, each `~300 MB` of `.jsonl.gz`. Each line is one Work
//! object with structured fields including `topics`, `concepts`,
//! `fields`, and `cited_by_count`. That's exactly the shape we need
//! for hashed-claim axis extraction — no per-item HTTP, just stream
//! the shard line-by-line.
//!
//! Best for **500 K – 50 M** items. Each shard is `~300 MB`
//! download, ~2–3 M Works per shard, so the user picks how many
//! shards to point at and the source stops at `max_items`.
//!
//! How to populate the local shard directory:
//!
//! ```bash
//! mkdir -p /tmp/openalex_shards
//! aws s3 sync s3://openalex/data/works/ /tmp/openalex_shards \
//!     --no-sign-request --exclude "*" --include "*/part_000.gz"
//! ```
//!
//! Then point the source at the directory (or pass a list of shard
//! paths directly).
//!
//! The newline-delimited JSON format means the only thing the source
//! needs from us is per-line JSON parsing — there's no embedded
//! framing or escaping to worry about.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::bulk::{BulkItem, BulkSource, BulkSourceError, Claim};

#[derive(Debug, Clone)]
pub struct OpenAlexShardConfig {
    pub shard_paths: Vec<PathBuf>,
    pub start_offset: usize,
    pub max_items: usize,
    pub min_cited_by: u64,
    pub min_year: u32,
}

impl OpenAlexShardConfig {
    pub fn new(shard_paths: Vec<PathBuf>) -> Self {
        Self {
            shard_paths,
            start_offset: 0,
            max_items: usize::MAX,
            min_cited_by: 5,
            min_year: 2010,
        }
    }

    /// Collect every `*.gz` file in `dir` (non-recursive). Useful
    /// when the user has run `aws s3 sync` into a flat directory.
    pub fn from_directory(dir: &Path) -> Result<Self, BulkSourceError> {
        let mut shards = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("gz") {
                shards.push(p);
            }
        }
        shards.sort();
        if shards.is_empty() {
            return Err(BulkSourceError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no .gz shards in {}", dir.display()),
            )));
        }
        Ok(Self::new(shards))
    }
}

pub struct OpenAlexShardSource {
    cfg: OpenAlexShardConfig,
    shard_index: usize,
    reader: Option<Box<dyn BufRead>>,
    n_consumed: usize,
    n_yielded: usize,
}

impl OpenAlexShardSource {
    pub fn new(cfg: OpenAlexShardConfig) -> Self {
        Self {
            cfg,
            shard_index: 0,
            reader: None,
            n_consumed: 0,
            n_yielded: 0,
        }
    }

    fn open_next_shard(&mut self) -> Result<bool, BulkSourceError> {
        while self.shard_index < self.cfg.shard_paths.len() {
            let path = self.cfg.shard_paths[self.shard_index].clone();
            self.shard_index += 1;
            match open_shard(&path) {
                Ok(r) => {
                    self.reader = Some(r);
                    return Ok(true);
                }
                Err(e) => {
                    eprintln!("warning: failed to open shard {}: {e}", path.display());
                    continue;
                }
            }
        }
        Ok(false)
    }

    fn next_line(&mut self) -> Result<Option<String>, BulkSourceError> {
        loop {
            if self.reader.is_none() && !self.open_next_shard()? {
                return Ok(None);
            }
            let r = self.reader.as_mut().expect("reader present");
            let mut buf = String::new();
            let n = r.read_line(&mut buf)?;
            if n == 0 {
                self.reader = None;
                continue;
            }
            return Ok(Some(buf));
        }
    }
}

impl Iterator for OpenAlexShardSource {
    type Item = Result<BulkItem, BulkSourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n_yielded >= self.cfg.max_items {
            return None;
        }
        loop {
            let line = match self.next_line() {
                Ok(Some(l)) => l,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            };
            self.n_consumed += 1;
            if self.n_consumed <= self.cfg.start_offset {
                continue;
            }
            match parse_work_line(line.trim(), self.cfg.min_cited_by, self.cfg.min_year) {
                Ok(Some(item)) => {
                    self.n_yielded += 1;
                    return Some(Ok(item));
                }
                Ok(None) => continue,
                Err(e) => {
                    // Skip malformed rows — one bad shard line
                    // shouldn't kill an 8-hour ingest. Surface a
                    // count via stderr; integrators can sum.
                    eprintln!("warning: openalex shard parse: {e}");
                    continue;
                }
            }
        }
    }
}

impl BulkSource for OpenAlexShardSource {
    fn source_name(&self) -> &str {
        "openalex_shard"
    }
}

fn open_shard(path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    let f = File::open(path)?;
    open_decompressed(f, path)
}

#[cfg(feature = "bulk-gzip")]
fn open_decompressed(f: File, _path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    let r = flate2::read::MultiGzDecoder::new(f);
    Ok(Box::new(BufReader::with_capacity(1 << 16, r)))
}

#[cfg(not(feature = "bulk-gzip"))]
fn open_decompressed(_f: File, path: &Path) -> Result<Box<dyn BufRead>, BulkSourceError> {
    Err(BulkSourceError::Io(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "bulk-gzip feature is disabled; cannot read .gz shard at {}",
            path.display()
        ),
    )))
}

/// Parse one Works line. Returns `Ok(None)` when the line passes
/// JSON parsing but fails the configured filters (low citation
/// count, old paper, retracted, missing title). Returns `Err` only
/// for unparseable JSON.
pub fn parse_work_line(
    line: &str,
    min_cited_by: u64,
    min_year: u32,
) -> Result<Option<BulkItem>, BulkSourceError> {
    if line.is_empty() {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(line)
        .map_err(|e| BulkSourceError::Parse(format!("openalex line: {e}")))?;

    // Filters. All of these are recoverable — return Ok(None).
    let cited = v
        .get("cited_by_count")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    if cited < min_cited_by {
        return Ok(None);
    }
    let year = v
        .get("publication_year")
        .and_then(|x| x.as_u64())
        .map(|y| y as u32)
        .unwrap_or(0);
    if year < min_year {
        return Ok(None);
    }
    if v.get("is_retracted").and_then(|x| x.as_bool()) == Some(true) {
        return Ok(None);
    }
    let Some(id_uri) = v.get("id").and_then(|x| x.as_str()) else {
        return Ok(None);
    };
    let id = id_uri.rsplit('/').next().unwrap_or(id_uri).to_string();
    let title = v
        .get("title")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(&id)
        .to_string();

    let mut claims = Vec::new();
    if let Some(topics) = v.get("topics").and_then(|x| x.as_array()) {
        for t in topics {
            if let Some(tid_uri) = t.get("id").and_then(|x| x.as_str()) {
                let tid = tid_uri.rsplit('/').next().unwrap_or(tid_uri).to_string();
                let score = t.get("score").and_then(|x| x.as_f64()).unwrap_or(1.0);
                claims.push(Claim::new("topic", tid, score.clamp(0.0, 1.0)));
            }
        }
    }
    if let Some(concepts) = v.get("concepts").and_then(|x| x.as_array()) {
        for c in concepts {
            if let Some(cid_uri) = c.get("id").and_then(|x| x.as_str()) {
                let cid = cid_uri.rsplit('/').next().unwrap_or(cid_uri).to_string();
                let score = c.get("score").and_then(|x| x.as_f64()).unwrap_or(1.0);
                claims.push(Claim::new("concept", cid, score.clamp(0.0, 1.0)));
            }
        }
    }
    // Primary topic gives us the source-side category guess.
    let category_hint = v
        .pointer("/primary_topic/display_name")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            v.pointer("/primary_topic/subfield/display_name")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        });

    // Quality from cited_by_count (Wikipedia-equivalent of
    // OpenAlex's `log10(1 + works_count) / 6`). Clamps at 1.0
    // around ~10 000 citations.
    let quality = ((1.0 + cited as f64).log10() / 4.0).clamp(0.0, 1.0);
    Ok(Some(BulkItem {
        external_id: id,
        label: title,
        description: String::new(),
        claims,
        source_name: "openalex_shard".into(),
        source_confidence: 0.9,
        category_hint,
        quality_hint: quality,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_line() -> &'static str {
        r#"{
          "id": "https://openalex.org/W123456789",
          "title": "Sample paper",
          "publication_year": 2022,
          "cited_by_count": 42,
          "is_retracted": false,
          "primary_topic": {
            "display_name": "Machine Learning",
            "subfield": {"display_name": "Artificial Intelligence"}
          },
          "topics": [
            {"id": "https://openalex.org/T10001", "score": 0.91},
            {"id": "https://openalex.org/T10002", "score": 0.42}
          ],
          "concepts": [
            {"id": "https://openalex.org/C100", "score": 0.7}
          ]
        }"#
    }

    #[test]
    fn parses_typical_work() {
        let item = parse_work_line(sample_line(), 5, 2010)
            .expect("parse")
            .expect("filter-pass");
        assert_eq!(item.external_id, "W123456789");
        assert_eq!(item.label, "Sample paper");
        assert_eq!(item.source_name, "openalex_shard");
        assert_eq!(item.category_hint.as_deref(), Some("Machine Learning"));
        // 2 topics + 1 concept.
        assert_eq!(item.claims.len(), 3);
        assert!(item.quality_hint > 0.0);
        assert!((item.source_confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn drops_low_citations() {
        let r = parse_work_line(sample_line(), 100, 2010).expect("parse");
        assert!(r.is_none());
    }

    #[test]
    fn drops_old_papers() {
        let r = parse_work_line(sample_line(), 5, 2030).expect("parse");
        assert!(r.is_none());
    }

    #[test]
    fn drops_retracted_papers() {
        let s = sample_line().replace("\"is_retracted\": false", "\"is_retracted\": true");
        let r = parse_work_line(&s, 5, 2010).expect("parse");
        assert!(r.is_none());
    }

    #[test]
    fn unknown_optional_fields_are_handled() {
        let minimal = r#"{
          "id": "https://openalex.org/W9",
          "title": "minimal",
          "publication_year": 2020,
          "cited_by_count": 10
        }"#;
        let item = parse_work_line(minimal, 5, 2010)
            .expect("parse")
            .expect("filter-pass");
        assert_eq!(item.external_id, "W9");
        assert_eq!(item.claims.len(), 0);
        assert_eq!(item.category_hint, None);
    }

    #[test]
    fn malformed_json_returns_err() {
        let r = parse_work_line("not-json", 5, 2010);
        assert!(matches!(r, Err(BulkSourceError::Parse(_))));
    }

    #[test]
    fn source_name_is_stable() {
        let s = OpenAlexShardSource::new(OpenAlexShardConfig::new(vec![]));
        assert_eq!(s.source_name(), "openalex_shard");
    }
}
