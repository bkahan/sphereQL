//! Paginated Wikidata SPARQL source.
//!
//! Pulls items from the Wikidata Query Service (WDQS) in pages,
//! ordered by sitelink count so high-relevance items come first.
//! Each row already carries label, description, sitelink count, and
//! `GROUP_CONCAT`-encoded P31 / P279 / P361 claim arrays — one HTTP
//! call per page, no per-item REST follow-ups.
//!
//! Best for **5 K – 1 M** items. WDQS has a 60 s per-query timeout
//! and a polite rate limit; at 500 / page that's ~2 000 queries for
//! 1 M items, or ~30 min – 2 h with sleeps. For larger scales swap
//! to [`OpenAlexShardSource`](crate::bulk::OpenAlexShardSource) (no
//! network ceiling, sharded) or
//! [`WikidataDumpSource`](crate::bulk::WikidataDumpSource) (full
//! dump, no HTTP).

use std::time::Duration;

use serde_json::Value;

use crate::bulk::{BulkItem, BulkSource, BulkSourceError, Claim};

pub const WDQS_ENDPOINT: &str = "https://query.wikidata.org/sparql";
pub const DEFAULT_PAGE_SIZE: usize = 500;
pub const DEFAULT_SLEEP_MS: u64 = 250;

/// Configuration for a SPARQL run. All fields have defaults that
/// match the WDQS rate-limit etiquette (<= 5 req/s burst, 60 s
/// timeout per query).
#[derive(Debug, Clone)]
pub struct SparqlConfig {
    pub endpoint: String,
    pub user_agent: String,
    pub page_size: usize,
    pub start_offset: usize,
    pub max_items: usize,
    pub sleep_ms: u64,
    pub timeout: Duration,
    pub retries: usize,
}

impl Default for SparqlConfig {
    fn default() -> Self {
        Self {
            endpoint: WDQS_ENDPOINT.to_string(),
            user_agent: format!(
                "sphereQL-corpus/{} (https://github.com/sphereql; ingest)",
                env!("CARGO_PKG_VERSION")
            ),
            page_size: DEFAULT_PAGE_SIZE,
            start_offset: 0,
            max_items: usize::MAX,
            sleep_ms: DEFAULT_SLEEP_MS,
            timeout: Duration::from_secs(60),
            retries: 3,
        }
    }
}

pub struct WikidataSparqlSource {
    cfg: SparqlConfig,
    /// Items already yielded across all pages (incl. start_offset).
    n_yielded: usize,
    /// Items consumed from the current page buffer.
    page_cursor: usize,
    page_buffer: Vec<BulkItem>,
    finished: bool,
}

impl WikidataSparqlSource {
    pub fn new(cfg: SparqlConfig) -> Self {
        Self {
            n_yielded: cfg.start_offset,
            page_cursor: 0,
            page_buffer: Vec::new(),
            finished: false,
            cfg,
        }
    }

    fn fetch_next_page(&mut self) -> Result<(), BulkSourceError> {
        let query = render_query(self.n_yielded, self.cfg.page_size);
        let body = post_with_retry(&self.cfg, &query)?;
        let parsed = parse_response(&body)?;
        if parsed.is_empty() {
            self.finished = true;
        } else {
            self.page_buffer = parsed;
            self.page_cursor = 0;
        }
        Ok(())
    }
}

impl Iterator for WikidataSparqlSource {
    type Item = Result<BulkItem, BulkSourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n_yielded >= self.cfg.start_offset.saturating_add(self.cfg.max_items) {
            return None;
        }
        if self.finished && self.page_cursor >= self.page_buffer.len() {
            return None;
        }
        if self.page_cursor >= self.page_buffer.len() {
            if let Err(e) = self.fetch_next_page() {
                self.finished = true;
                return Some(Err(e));
            }
            if self.page_buffer.is_empty() {
                return None;
            }
            if self.cfg.sleep_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.cfg.sleep_ms));
            }
        }
        let item = self.page_buffer[self.page_cursor].clone();
        self.page_cursor += 1;
        self.n_yielded += 1;
        Some(Ok(item))
    }
}

impl BulkSource for WikidataSparqlSource {
    fn source_name(&self) -> &str {
        "wikidata_sparql"
    }
}

/// SPARQL query pulling N items ordered by sitelink count, with
/// `GROUP_CONCAT`-encoded claim arrays per item.
///
/// We use `?item` as the GROUP BY key but ORDER BY sitelinks for
/// relevance. Pagination is `OFFSET ... LIMIT ...` — WDQS supports
/// it but warns about late offsets; for our scale it's fine, and
/// the dump source picks up where SPARQL becomes painful.
fn render_query(offset: usize, limit: usize) -> String {
    format!(
        "SELECT ?item ?itemLabel ?itemDescription ?sitelinks \
         (GROUP_CONCAT(DISTINCT STR(?p31); SEPARATOR=\"|\") AS ?p31s) \
         (GROUP_CONCAT(DISTINCT STR(?p279); SEPARATOR=\"|\") AS ?p279s) \
         (GROUP_CONCAT(DISTINCT STR(?p361); SEPARATOR=\"|\") AS ?p361s) \
         WHERE {{ \
           ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . \
           ?item wikibase:sitelinks ?sitelinks . \
           OPTIONAL {{ ?item wdt:P31 ?p31 }} \
           OPTIONAL {{ ?item wdt:P279 ?p279 }} \
           OPTIONAL {{ ?item wdt:P361 ?p361 }} \
           SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\" }} \
         }} \
         GROUP BY ?item ?itemLabel ?itemDescription ?sitelinks \
         ORDER BY DESC(?sitelinks) ?item \
         OFFSET {offset} LIMIT {limit}"
    )
}

#[cfg(feature = "bulk-http")]
fn post_with_retry(cfg: &SparqlConfig, query: &str) -> Result<String, BulkSourceError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(cfg.timeout)
        .user_agent(&cfg.user_agent)
        .build();
    let mut last_err: Option<BulkSourceError> = None;
    for attempt in 0..=cfg.retries {
        let res = agent
            .post(&cfg.endpoint)
            .set("Accept", "application/sparql-results+json")
            .send_form(&[("query", query)]);
        match res {
            Ok(resp) => {
                return resp
                    .into_string()
                    .map_err(|e| BulkSourceError::Network(e.to_string()));
            }
            Err(e) => {
                last_err = Some(BulkSourceError::Network(format!(
                    "attempt {} of {}: {}",
                    attempt + 1,
                    cfg.retries + 1,
                    e
                )));
                if attempt < cfg.retries {
                    std::thread::sleep(Duration::from_millis(
                        cfg.sleep_ms.saturating_mul(1u64 << attempt),
                    ));
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| BulkSourceError::Network("no attempts made".into())))
}

#[cfg(not(feature = "bulk-http"))]
fn post_with_retry(_cfg: &SparqlConfig, _query: &str) -> Result<String, BulkSourceError> {
    Err(BulkSourceError::Network(
        "bulk-http feature is disabled; rebuild sphereql-corpus with --features bulk-http".into(),
    ))
}

/// Parse the WDQS JSON response. Returns one [`BulkItem`] per
/// binding row. Items with no parseable QID are silently dropped —
/// they would fail validation downstream anyway.
pub fn parse_response(body: &str) -> Result<Vec<BulkItem>, BulkSourceError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| BulkSourceError::Parse(format!("sparql response: {e}")))?;
    let bindings = v
        .pointer("/results/bindings")
        .and_then(|b| b.as_array())
        .ok_or_else(|| BulkSourceError::Parse("missing results.bindings".into()))?;
    let mut out = Vec::with_capacity(bindings.len());
    for b in bindings {
        let Some(item_uri) = b
            .pointer("/item/value")
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        let qid = qid_from_uri(item_uri);
        let label = b
            .pointer("/itemLabel/value")
            .and_then(|v| v.as_str())
            .unwrap_or(&qid)
            .to_string();
        let description = b
            .pointer("/itemDescription/value")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let sitelinks: u64 = b
            .pointer("/sitelinks/value")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut claims = Vec::new();
        for (pred, key) in [("P31", "p31s"), ("P279", "p279s"), ("P361", "p361s")] {
            let Some(s) = b
                .pointer(&format!("/{key}/value"))
                .and_then(|v| v.as_str())
            else {
                continue;
            };
            for piece in s.split('|') {
                let piece = piece.trim();
                if piece.is_empty() {
                    continue;
                }
                let obj = qid_from_uri(piece);
                if !obj.is_empty() {
                    claims.push(Claim::new(pred, obj, 1.0));
                }
            }
        }
        let conf = (((1.0 + sitelinks as f64).ln()) / 5.0).clamp(0.0, 1.0);
        out.push(BulkItem {
            external_id: qid.clone(),
            label,
            description,
            claims,
            source_name: "wikidata_sparql".into(),
            source_confidence: conf,
            category_hint: None,
            quality_hint: conf,
        });
    }
    Ok(out)
}

fn qid_from_uri(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> &'static str {
        r#"{
          "head": {"vars": ["item", "itemLabel", "itemDescription", "sitelinks", "p31s", "p279s", "p361s"]},
          "results": {"bindings": [
            {
              "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q42"},
              "itemLabel": {"type": "literal", "value": "Douglas Adams"},
              "itemDescription": {"type": "literal", "value": "English writer and humorist"},
              "sitelinks": {"type": "literal", "value": "12"},
              "p31s": {"type": "literal", "value": "http://www.wikidata.org/entity/Q5"},
              "p279s": {"type": "literal", "value": ""},
              "p361s": {"type": "literal", "value": ""}
            },
            {
              "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"},
              "itemLabel": {"type": "literal", "value": "universe"},
              "sitelinks": {"type": "literal", "value": "80"},
              "p31s": {"type": "literal", "value": "http://www.wikidata.org/entity/Q1454986|http://www.wikidata.org/entity/Q36906466"},
              "p279s": {"type": "literal", "value": ""},
              "p361s": {"type": "literal", "value": ""}
            }
          ]}
        }"#
    }

    #[test]
    fn parses_two_rows_with_claims() {
        let items = parse_response(fixture()).expect("parse");
        assert_eq!(items.len(), 2);
        let q42 = &items[0];
        assert_eq!(q42.external_id, "Q42");
        assert_eq!(q42.label, "Douglas Adams");
        assert_eq!(q42.claims.len(), 1);
        assert_eq!(q42.claims[0].predicate, "P31");
        assert_eq!(q42.claims[0].object, "Q5");
        let q1 = &items[1];
        assert_eq!(q1.claims.len(), 2);
        assert_eq!(q1.claims[0].object, "Q1454986");
        assert_eq!(q1.claims[1].object, "Q36906466");
    }

    #[test]
    fn confidence_grows_with_sitelinks() {
        let items = parse_response(fixture()).expect("parse");
        assert!(items[1].source_confidence > items[0].source_confidence);
        assert!(items[0].source_confidence > 0.0);
    }

    #[test]
    fn render_query_has_offset_and_limit() {
        let q = render_query(1500, 500);
        assert!(q.contains("OFFSET 1500"));
        assert!(q.contains("LIMIT 500"));
        assert!(q.contains("ORDER BY DESC(?sitelinks)"));
    }

    #[test]
    fn source_name_is_stable() {
        let s = WikidataSparqlSource::new(SparqlConfig::default());
        assert_eq!(s.source_name(), "wikidata_sparql");
    }

    #[test]
    fn empty_bindings_returns_empty() {
        let body = r#"{"head":{"vars":[]},"results":{"bindings":[]}}"#;
        assert!(parse_response(body).expect("parse").is_empty());
    }
}
