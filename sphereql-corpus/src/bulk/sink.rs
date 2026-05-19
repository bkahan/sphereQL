//! Streaming Parquet sink with on-disk checkpointing.
//!
//! Writes the same schema [`crate::parquet_writer`] uses, but in
//! batches: `push(item)` buffers a row, `flush()` writes the buffer
//! as a Parquet row group, `close()` finalizes the file. After each
//! flush, a sidecar `<path>.checkpoint.json` records how many items
//! the source has produced — so an interrupted run can resume by
//! advancing the source iterator past that offset before pushing.
//!
//! Memory: bounded to `batch_size` rows (default 10 000). The
//! writer's internal Parquet page buffer is similarly bounded by
//! `ArrowWriter`'s default settings.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::bulk::{BulkItem, HashedClaimAxisExtractor};
use crate::parquet_loader::ParquetLoadError;
use crate::parquet_writer::{ConceptRow, build_batch, build_schema};

/// Default rows per batch flush. Roughly one Parquet row group at
/// our schema's payload size; matches what the existing
/// `parquet_writer` uses for non-streaming writes.
pub const DEFAULT_BATCH_SIZE: usize = 10_000;

/// Default categorical hint when a source can't suggest one — the
/// post-ingest clustering step is responsible for filling these.
pub const UNCATEGORIZED: &str = "_uncategorized";

/// What the sidecar JSON records after each flush. `source_offset`
/// is the count of *source items consumed* (including skipped ones),
/// which is what a resuming source needs to seek past — not the
/// count of *rows written*, which can differ when items are dropped
/// for having no extractable features.
#[derive(Debug, Clone, Copy)]
pub struct SinkCheckpoint {
    pub n_written: usize,
    pub source_offset: usize,
}

impl SinkCheckpoint {
    fn to_json(self) -> String {
        format!(
            "{{\"n_written\":{},\"source_offset\":{}}}",
            self.n_written, self.source_offset
        )
    }

    /// Best-effort parser; tolerates extra whitespace but expects
    /// the two integer fields. Returns `None` for any deviation
    /// rather than fighting an error type for a 2-field JSON.
    pub fn parse(s: &str) -> Option<Self> {
        let n_written = extract_int(s, "\"n_written\"")?;
        let source_offset = extract_int(s, "\"source_offset\"")?;
        Some(Self {
            n_written,
            source_offset,
        })
    }

    /// Load the checkpoint sidecar for `parquet_path`, if it exists.
    pub fn load_for(parquet_path: &Path) -> Option<Self> {
        let cp = Self::sidecar_for(parquet_path);
        std::fs::read_to_string(cp).ok().and_then(|s| Self::parse(&s))
    }

    pub fn sidecar_for(parquet_path: &Path) -> PathBuf {
        let mut p = parquet_path.to_path_buf();
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "corpus.parquet".to_string());
        p.set_file_name(format!("{name}.checkpoint.json"));
        p
    }
}

fn extract_int(s: &str, key: &str) -> Option<usize> {
    let i = s.find(key)?;
    let after = &s[i + key.len()..];
    let colon = after.find(':')?;
    let tail = &after[colon + 1..];
    let mut start: Option<usize> = None;
    let mut end = 0;
    for (n, ch) in tail.char_indices() {
        if ch.is_ascii_digit() {
            if start.is_none() {
                start = Some(n);
            }
            end = n + 1;
        } else if start.is_some() {
            break;
        }
    }
    let s = start?;
    tail[s..end].parse().ok()
}

/// One staged row's owned data — the sink holds these until a flush,
/// at which point we borrow `&str` slices off them into the existing
/// [`ConceptRow`] and call [`build_batch`].
struct StagedRow {
    label: String,
    category: String,
    features: Vec<(usize, f64)>,
    quality: f64,
    axis_coherence: f64,
    bridge_degree: u8,
    source_confidence: f64,
    home_affinity: f64,
    source: Option<String>,
    openalex_id: Option<String>,
}

pub struct ParquetSink {
    writer: Option<ArrowWriter<File>>,
    schema: Arc<Schema>,
    batch_size: usize,
    buffered: Vec<StagedRow>,
    n_written: usize,
    source_offset: usize,
    path: PathBuf,
    extractor: HashedClaimAxisExtractor,
}

impl ParquetSink {
    /// Open `path` for writing. Overwrites any existing file at
    /// `path` — pair this with [`SinkCheckpoint::load_for`] +
    /// `append=true` semantics in a higher layer if you need true
    /// resumption (the current binary builds the resumed parquet at
    /// a sibling path and merges at the end).
    pub fn create(
        path: impl AsRef<Path>,
        extractor: HashedClaimAxisExtractor,
        batch_size: usize,
    ) -> Result<Self, ParquetLoadError> {
        assert!(batch_size > 0, "batch_size must be > 0");
        let schema = Arc::new(build_schema());
        let file = File::create(path.as_ref())?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(batch_size))
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(ParquetLoadError::from)?;
        Ok(Self {
            writer: Some(writer),
            schema,
            batch_size,
            buffered: Vec::with_capacity(batch_size),
            n_written: 0,
            source_offset: 0,
            path: path.as_ref().to_path_buf(),
            extractor,
        })
    }

    /// Stage one source item. Items that yield zero features (no
    /// claims, or all claims hashed to a single bucket and pruned
    /// for being below `MIN_WEIGHT`) are silently skipped — the
    /// `source_offset` still advances so checkpoints stay correct,
    /// but `n_written` does not.
    pub fn push(&mut self, item: BulkItem) -> Result<(), ParquetLoadError> {
        self.source_offset += 1;
        let features = self.extractor.extract(&item);
        if features.is_empty() {
            return Ok(());
        }
        self.buffered.push(StagedRow {
            label: item.label,
            category: item.category_hint.unwrap_or_else(|| UNCATEGORIZED.to_string()),
            features,
            quality: item.quality_hint.clamp(0.0, 1.0),
            axis_coherence: 1.0,
            bridge_degree: 1,
            source_confidence: item.source_confidence.clamp(0.0, 1.0),
            home_affinity: 1.0,
            source: Some(item.source_name),
            openalex_id: if item.external_id.starts_with('W') || item.external_id.starts_with('T') {
                Some(item.external_id)
            } else {
                None
            },
        });
        if self.buffered.len() >= self.batch_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Write any staged rows as one Parquet row group, then update
    /// the sidecar checkpoint. Safe to call on an empty buffer
    /// (no-op apart from refreshing the checkpoint).
    pub fn flush(&mut self) -> Result<(), ParquetLoadError> {
        if !self.buffered.is_empty() {
            let rows: Vec<ConceptRow<'_>> = self
                .buffered
                .iter()
                .map(|r| ConceptRow {
                    label: r.label.as_str(),
                    category: r.category.as_str(),
                    features: r.features.as_slice(),
                    quality: r.quality,
                    axis_coherence: r.axis_coherence,
                    bridge_degree: r.bridge_degree,
                    source_confidence: r.source_confidence,
                    home_affinity: r.home_affinity,
                    source: r.source.as_deref(),
                    openalex_id: r.openalex_id.as_deref(),
                })
                .collect();
            let batch = build_batch(self.schema.clone(), &rows)?;
            self.writer
                .as_mut()
                .expect("writer present until close()")
                .write(&batch)
                .map_err(ParquetLoadError::from)?;
            self.n_written += self.buffered.len();
            self.buffered.clear();
        }
        self.write_checkpoint()?;
        Ok(())
    }

    /// Finalize the Parquet footer. Calls `flush()` first so any
    /// trailing partial batch lands in the file. Drops the writer.
    pub fn close(mut self) -> Result<SinkCheckpoint, ParquetLoadError> {
        self.flush()?;
        if let Some(w) = self.writer.take() {
            w.close().map_err(ParquetLoadError::from)?;
        }
        Ok(SinkCheckpoint {
            n_written: self.n_written,
            source_offset: self.source_offset,
        })
    }

    pub fn n_written(&self) -> usize {
        self.n_written
    }

    pub fn source_offset(&self) -> usize {
        self.source_offset
    }

    fn write_checkpoint(&self) -> Result<(), ParquetLoadError> {
        let cp = SinkCheckpoint {
            n_written: self.n_written,
            source_offset: self.source_offset,
        };
        let tmp = SinkCheckpoint::sidecar_for(&self.path);
        let mut f = File::create(&tmp)?;
        f.write_all(cp.to_json().as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bulk::{BulkItem, Claim};
    use crate::parquet_loader::load_concepts_with_metadata;
    use tempfile::tempdir;

    fn synth_item(id: u32, weight: f64) -> BulkItem {
        BulkItem {
            external_id: format!("Q{id}"),
            label: format!("item_{id}"),
            description: String::new(),
            claims: vec![
                Claim::new("P31", format!("Q{}", id % 10), weight),
                Claim::new("P279", format!("Q{}", (id / 10) % 10), weight),
            ],
            source_name: "synth".into(),
            source_confidence: 0.7,
            category_hint: Some(format!("cat_{}", id % 5)),
            quality_hint: 0.6,
        }
    }

    #[test]
    fn sink_writes_and_round_trips() {
        let dir = tempdir().expect("tmpdir");
        let path = dir.path().join("out.parquet");
        let ex = HashedClaimAxisExtractor::new(64, 0);
        let mut sink = ParquetSink::create(&path, ex, 25).expect("create");
        for i in 0..73 {
            sink.push(synth_item(i, 1.0)).expect("push");
        }
        let cp = sink.close().expect("close");
        assert_eq!(cp.source_offset, 73);
        assert_eq!(cp.n_written, 73);

        let loaded = load_concepts_with_metadata(&path).expect("load");
        assert_eq!(loaded.len(), 73);
        // Spot-check that source survived round-trip.
        assert_eq!(loaded[0].1.source.as_deref(), Some("synth"));
    }

    #[test]
    fn empty_claim_items_are_skipped() {
        let dir = tempdir().expect("tmpdir");
        let path = dir.path().join("out.parquet");
        let ex = HashedClaimAxisExtractor::new(64, 0);
        let mut sink = ParquetSink::create(&path, ex, 10).expect("create");
        for i in 0..10 {
            sink.push(synth_item(i, 1.0)).expect("push");
        }
        // 5 items with no claims — should not write rows but should
        // bump source_offset.
        for i in 100..105 {
            let mut it = synth_item(i, 1.0);
            it.claims.clear();
            sink.push(it).expect("push");
        }
        let cp = sink.close().expect("close");
        assert_eq!(cp.source_offset, 15);
        assert_eq!(cp.n_written, 10);

        let loaded = load_concepts_with_metadata(&path).expect("load");
        assert_eq!(loaded.len(), 10);
    }

    #[test]
    fn checkpoint_sidecar_is_written_and_parsed() {
        let dir = tempdir().expect("tmpdir");
        let path = dir.path().join("out.parquet");
        let ex = HashedClaimAxisExtractor::new(64, 0);
        let mut sink = ParquetSink::create(&path, ex, 5).expect("create");
        for i in 0..12 {
            sink.push(synth_item(i, 1.0)).expect("push");
        }
        sink.flush().expect("flush");
        let cp = SinkCheckpoint::load_for(&path).expect("sidecar exists");
        // 12 pushed, 2 full batches (10) auto-flushed plus 2 staged
        // and emitted by the explicit flush() call.
        assert_eq!(cp.source_offset, 12);
        assert_eq!(cp.n_written, 12);
        let _ = sink.close();
    }

    #[test]
    fn parse_handles_whitespace_variants() {
        let s = "{ \"n_written\" : 1234 , \"source_offset\" : 5678 }";
        let cp = SinkCheckpoint::parse(s).expect("parse");
        assert_eq!(cp.n_written, 1234);
        assert_eq!(cp.source_offset, 5678);
    }
}
