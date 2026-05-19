//! Unified corpus registry.
//!
//! [`CorpusId`] names every corpus this crate ships. Pass a `CorpusId`
//! (or a slice of them) to load or stream any corpus without touching
//! per-corpus modules. Adding a new corpus is one new variant here plus
//! a data file — no new module required.
//!
//! ```no_run
//! use sphereql_corpus::CorpusId;
//!
//! // Load a single corpus.
//! let concepts = CorpusId::DBpedia50k.load().expect("load");
//!
//! // Iterate over all named corpora.
//! for id in CorpusId::all() {
//!     println!("{}: {} concepts", id.name(), id.load().unwrap().len());
//! }
//!
//! // Arbitrary parquet file.
//! let concepts = CorpusId::Parquet("/tmp/my_corpus.parquet".into()).load().expect("load");
//! ```

use std::path::PathBuf;

use crate::concept::Concept;
use crate::parquet_loader::{self, ParquetLoadError};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Every corpus this crate knows about.
///
/// Named variants resolve to fixed paths under `data/`. The
/// [`Parquet`] escape hatch accepts any path for corpora not yet
/// named here. In-memory corpora ([`HandCrafted`], [`Full`],
/// [`Stress`]) never return `Err` from [`load`] or [`stream`];
/// Parquet-backed corpora surface [`ParquetLoadError`] on missing
/// or corrupt files.
///
/// [`Parquet`]: CorpusId::Parquet
/// [`HandCrafted`]: CorpusId::HandCrafted
/// [`Full`]: CorpusId::Full
/// [`Stress`]: CorpusId::Stress
/// [`load`]: CorpusId::load
/// [`stream`]: CorpusId::stream
#[derive(Debug, Clone)]
pub enum CorpusId {
    /// 775 hand-crafted concepts across 31 academic domains (in-memory).
    HandCrafted,
    /// ~5,000+ OpenAlex-derived concepts — `extended_corpus.parquet`.
    Extended,
    /// [`HandCrafted`] followed by [`Extended`].
    ///
    /// [`HandCrafted`]: CorpusId::HandCrafted
    /// [`Extended`]: CorpusId::Extended
    Full,
    /// 300 synthetic, high-noise concepts for SNR stress tests (in-memory).
    Stress,
    /// DBpedia 50K raw ingest — `dbpedia_50k.parquet`.
    DBpedia50k,
    /// DBpedia 50K after k-means clustering — `dbpedia_50k.clustered.parquet`.
    DBpedia50kClustered,
    /// DBpedia 50K after clustering + self-tuning — `dbpedia_50k.clustered.tuned.parquet`.
    DBpedia50kTuned,
    /// DBpedia 500K raw ingest — `dbpedia_500k.parquet`.
    DBpedia500k,
    /// DBpedia 500K after k-means clustering — `dbpedia_500k.clustered.parquet`.
    DBpedia500kClustered,
    /// DBpedia 500K after clustering + self-tuning — `dbpedia_500k.clustered.tuned.parquet`.
    DBpedia500kTuned,
    /// Wikidata 50K raw ingest — `wikidata_50k.parquet`.
    Wikidata50k,
    /// Any Parquet file at the given path — for corpora not yet named above.
    Parquet(PathBuf),
}

static ALL_NAMED: [CorpusId; 11] = [
    CorpusId::HandCrafted,
    CorpusId::Extended,
    CorpusId::Full,
    CorpusId::Stress,
    CorpusId::DBpedia50k,
    CorpusId::DBpedia50kClustered,
    CorpusId::DBpedia50kTuned,
    CorpusId::DBpedia500k,
    CorpusId::DBpedia500kClustered,
    CorpusId::DBpedia500kTuned,
    CorpusId::Wikidata50k,
];

impl CorpusId {
    /// All named variants (excludes [`Parquet`]), in a stable order.
    ///
    /// [`Parquet`]: CorpusId::Parquet
    pub fn all() -> &'static [CorpusId] {
        &ALL_NAMED
    }

    /// Short identifier suitable for display, file names, and metric tags.
    pub fn name(&self) -> &str {
        match self {
            Self::HandCrafted => "hand_crafted",
            Self::Extended => "extended",
            Self::Full => "full",
            Self::Stress => "stress",
            Self::DBpedia50k => "dbpedia_50k",
            Self::DBpedia50kClustered => "dbpedia_50k_clustered",
            Self::DBpedia50kTuned => "dbpedia_50k_tuned",
            Self::DBpedia500k => "dbpedia_500k",
            Self::DBpedia500kClustered => "dbpedia_500k_clustered",
            Self::DBpedia500kTuned => "dbpedia_500k_tuned",
            Self::Wikidata50k => "wikidata_50k",
            Self::Parquet(p) => p.file_stem().and_then(|s| s.to_str()).unwrap_or("custom"),
        }
    }

    /// Path to the Parquet file backing this corpus, if any.
    ///
    /// Returns `None` for in-memory corpora ([`HandCrafted`], [`Full`],
    /// [`Stress`]).
    ///
    /// [`HandCrafted`]: CorpusId::HandCrafted
    /// [`Full`]: CorpusId::Full
    /// [`Stress`]: CorpusId::Stress
    pub fn parquet_path(&self) -> Option<PathBuf> {
        let d = data_dir();
        match self {
            Self::HandCrafted | Self::Full | Self::Stress => None,
            Self::Extended => Some(d.join("extended_corpus.parquet")),
            Self::DBpedia50k => Some(d.join("dbpedia_50k.parquet")),
            Self::DBpedia50kClustered => Some(d.join("dbpedia_50k.clustered.parquet")),
            Self::DBpedia50kTuned => Some(d.join("dbpedia_50k.clustered.tuned.parquet")),
            Self::DBpedia500k => Some(d.join("dbpedia_500k.parquet")),
            Self::DBpedia500kClustered => Some(d.join("dbpedia_500k.clustered.parquet")),
            Self::DBpedia500kTuned => Some(d.join("dbpedia_500k.clustered.tuned.parquet")),
            Self::Wikidata50k => Some(d.join("wikidata_50k.parquet")),
            Self::Parquet(p) => Some(p.clone()),
        }
    }

    /// Eagerly load the entire corpus into a `Vec<Concept>`.
    ///
    /// In-memory corpora succeed unconditionally. Parquet-backed corpora
    /// return [`ParquetLoadError`] on a missing or corrupt file. For
    /// [`Full`], the Extended Parquet file must be present.
    ///
    /// For corpora over ~100K concepts, prefer [`stream`] to avoid
    /// materializing the full vector upfront.
    ///
    /// [`Full`]: CorpusId::Full
    /// [`stream`]: CorpusId::stream
    pub fn load(&self) -> Result<Vec<Concept>, ParquetLoadError> {
        match self {
            Self::HandCrafted => Ok(crate::corpus::build_corpus()),
            Self::Stress => Ok(crate::stress_corpus::build_stress_corpus()),
            Self::Full => {
                let mut v = crate::corpus::build_corpus();
                // Extended always has a parquet path; the unwrap is an
                // invariant — the match above exhausted all in-memory
                // variants and Extended is never in-memory.
                v.extend(parquet_loader::load_concepts(
                    CorpusId::Extended
                        .parquet_path()
                        .expect("Extended always has a parquet path"),
                )?);
                Ok(v)
            }
            other => {
                let path = other.parquet_path().ok_or_else(|| {
                    ParquetLoadError::Schema(format!(
                        "corpus {:?} has no parquet path",
                        other.name()
                    ))
                })?;
                parquet_loader::load_concepts(path)
            }
        }
    }

    /// Stream the corpus row-by-row without materializing the full vector.
    ///
    /// Suited for 500K+ corpora where holding the full `Vec<Concept>`
    /// in memory is expensive. In-memory corpora ([`HandCrafted`],
    /// [`Full`], [`Stress`]) are collected eagerly and then boxed —
    /// for those, [`load`] is more direct.
    ///
    /// [`HandCrafted`]: CorpusId::HandCrafted
    /// [`Full`]: CorpusId::Full
    /// [`Stress`]: CorpusId::Stress
    /// [`load`]: CorpusId::load
    pub fn stream(
        &self,
    ) -> Result<Box<dyn Iterator<Item = Result<Concept, ParquetLoadError>> + Send>, ParquetLoadError>
    {
        match self {
            Self::HandCrafted => {
                let v = crate::corpus::build_corpus();
                Ok(Box::new(v.into_iter().map(Ok)))
            }
            Self::Stress => {
                let v = crate::stress_corpus::build_stress_corpus();
                Ok(Box::new(v.into_iter().map(Ok)))
            }
            Self::Full => {
                let mut v = crate::corpus::build_corpus();
                v.extend(parquet_loader::load_concepts(
                    CorpusId::Extended
                        .parquet_path()
                        .expect("Extended always has a parquet path"),
                )?);
                Ok(Box::new(v.into_iter().map(Ok)))
            }
            other => {
                let path = other.parquet_path().ok_or_else(|| {
                    ParquetLoadError::Schema(format!(
                        "corpus {:?} has no parquet path",
                        other.name()
                    ))
                })?;
                parquet_loader::stream_concepts(path)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_eleven_named_variants() {
        assert_eq!(CorpusId::all().len(), 11);
    }

    #[test]
    fn names_are_unique() {
        let names: Vec<&str> = CorpusId::all().iter().map(|c| c.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            names.len(),
            sorted.len(),
            "duplicate names in CorpusId::all()"
        );
    }

    #[test]
    fn in_memory_corpora_have_no_parquet_path() {
        assert!(CorpusId::HandCrafted.parquet_path().is_none());
        assert!(CorpusId::Full.parquet_path().is_none());
        assert!(CorpusId::Stress.parquet_path().is_none());
    }

    #[test]
    fn parquet_backed_corpora_have_paths() {
        for id in CorpusId::all() {
            match id {
                CorpusId::HandCrafted | CorpusId::Full | CorpusId::Stress => {
                    assert!(
                        id.parquet_path().is_none(),
                        "{} should have no path",
                        id.name()
                    );
                }
                _ => {
                    assert!(
                        id.parquet_path().is_some(),
                        "{} should have a parquet path",
                        id.name()
                    );
                }
            }
        }
    }

    #[test]
    fn custom_parquet_path_round_trips() {
        let p = PathBuf::from("/tmp/my_corpus.parquet");
        let id = CorpusId::Parquet(p.clone());
        assert_eq!(id.parquet_path().unwrap(), p);
        assert_eq!(id.name(), "my_corpus");
    }

    #[test]
    fn hand_crafted_loads_775_concepts() {
        let v = CorpusId::HandCrafted.load().expect("load");
        assert_eq!(v.len(), 775);
    }

    #[test]
    fn stress_loads_300_concepts() {
        let v = CorpusId::Stress.load().expect("load");
        assert_eq!(v.len(), 300);
    }

    #[test]
    fn extended_load_and_stream_agree() {
        let eager = CorpusId::Extended.load().expect("eager load");
        let streamed: usize = CorpusId::Extended
            .stream()
            .expect("stream open")
            .filter_map(Result::ok)
            .count();
        assert_eq!(eager.len(), streamed);
    }
}
