//! Parquet read path for the extended corpus.
//!
//! Replaces the JSON+`Box::leak` strategy at scale. Strings are still
//! leaked per-concept because [`Concept`] is `&'static str` today; that
//! ownership migration is a separate refactor. The win in this phase is
//! getting off `include_str!` (which blows up the binary at >50K
//! concepts) and off `serde_json::from_str` (which forces a full DOM in
//! memory before allocation).
//!
//! Rows are read in Arrow record batches; the eager loader collects to
//! `Vec<Concept>`, and the streaming loader yields an iterator that the
//! embed pipeline can consume row-by-row without holding the whole
//! corpus in memory.
//!
//! ## Schema
//!
//! | column              | arrow type                                | nullable |
//! |---------------------|-------------------------------------------|----------|
//! | label               | Utf8                                      | no       |
//! | category            | Utf8                                      | no       |
//! | features            | List\<Struct{axis:UInt32, weight:Float64}\> | no     |
//! | quality             | Float64                                   | no       |
//! | axis_coherence      | Float64                                   | no       |
//! | bridge_degree       | UInt8                                     | no       |
//! | source_confidence   | Float64                                   | no       |
//! | home_affinity       | Float64                                   | no       |
//! | source              | Utf8                                      | yes      |
//! | openalex_id         | Utf8                                      | yes      |
//!
//! Compression: SNAPPY. Row group size: 4096 (set by the writer).
//! Dictionary encoding is on for `label`, `category`, `source`.

use std::fs::File;
use std::path::Path;

use arrow::array::{
    Array, Float64Array, ListArray, StringArray, StructArray, UInt8Array, UInt32Array,
};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::ParquetError;

use crate::concept::Concept;

/// Errors raised by the Parquet loader.
#[derive(Debug)]
pub enum ParquetLoadError {
    Io(std::io::Error),
    Parquet(ParquetError),
    Arrow(ArrowError),
    Schema(String),
}

impl std::fmt::Display for ParquetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Parquet(e) => write!(f, "parquet error: {e}"),
            Self::Arrow(e) => write!(f, "arrow error: {e}"),
            Self::Schema(s) => write!(f, "schema error: {s}"),
        }
    }
}

impl std::error::Error for ParquetLoadError {}

impl From<std::io::Error> for ParquetLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParquetError> for ParquetLoadError {
    fn from(e: ParquetError) -> Self {
        // ParquetError sometimes wraps an io::Error inside an external
        // variant — surface it as the Io variant so the NotFound
        // fallback in loader.rs works.
        if let ParquetError::External(boxed) = &e
            && let Some(io_err) = boxed.downcast_ref::<std::io::Error>()
        {
            return Self::Io(std::io::Error::new(io_err.kind(), io_err.to_string()));
        }
        Self::Parquet(e)
    }
}

impl From<ArrowError> for ParquetLoadError {
    fn from(e: ArrowError) -> Self {
        // The Arrow reader returns ArrowError::IoError on file-level
        // problems; surface NotFound to the Io variant so the
        // json-fallback path in loader.rs still triggers correctly.
        if let ArrowError::IoError(_, ref io_err) = e {
            return Self::Io(std::io::Error::new(io_err.kind(), io_err.to_string()));
        }
        Self::Arrow(e)
    }
}

/// Eager loader: reads the entire Parquet file into a `Vec<Concept>`.
///
/// Strings are **owned** via `Box::leak` (necessary because `Concept`
/// uses `&'static str`). Each call leaks `2 * N` strings; call once
/// per process — same constraint as the JSON loader.
///
/// For corpora >100K, prefer [`stream_concepts`] to avoid materializing
/// the whole vector upfront.
pub fn load_concepts<P: AsRef<Path>>(path: P) -> Result<Vec<Concept>, ParquetLoadError> {
    let file = File::open(path.as_ref())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut concepts = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        concepts.extend(batch_to_concepts(&batch)?);
    }
    Ok(concepts)
}

/// Streaming iterator: yields concepts in row-group order without
/// materializing the full corpus.
///
/// Returns a boxed iterator so callers don't depend on the exact
/// `parquet` crate types. Strings are leaked per-concept as in
/// [`load_concepts`]. The returned iterator is `Send` so it can be
/// shipped to a worker thread.
pub fn stream_concepts<P: AsRef<Path>>(
    path: P,
) -> Result<Box<dyn Iterator<Item = Result<Concept, ParquetLoadError>> + Send>, ParquetLoadError>
{
    let file = File::open(path.as_ref())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let iter = reader.flat_map(|batch_result| {
        let v: Vec<Result<Concept, ParquetLoadError>> = match batch_result {
            Ok(batch) => match batch_to_concepts(&batch) {
                Ok(cs) => cs.into_iter().map(Ok).collect(),
                Err(e) => vec![Err(e)],
            },
            Err(e) => vec![Err(e.into())],
        };
        v.into_iter()
    });
    Ok(Box::new(iter))
}

fn batch_to_concepts(batch: &RecordBatch) -> Result<Vec<Concept>, ParquetLoadError> {
    let labels = col_str(batch, "label")?;
    let categories = col_str(batch, "category")?;
    let features = col_list(batch, "features")?;
    let quality = col_f64(batch, "quality")?;
    let axis_coherence = col_f64(batch, "axis_coherence")?;
    let bridge_degree = col_u8(batch, "bridge_degree")?;
    let source_confidence = col_f64(batch, "source_confidence")?;
    let home_affinity = col_f64(batch, "home_affinity")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let feats = parse_feature_list(features, i)?;
        out.push(Concept {
            label: leak_str(labels.value(i)),
            category: leak_str(categories.value(i)),
            features: feats,
            quality: quality.value(i),
            axis_coherence: axis_coherence.value(i),
            bridge_degree: bridge_degree.value(i),
            source_confidence: source_confidence.value(i),
            home_affinity: home_affinity.value(i),
        });
    }
    Ok(out)
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, ParquetLoadError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ParquetLoadError::Schema(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ParquetLoadError::Schema(format!("{name}: not Utf8")))
}

fn col_f64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, ParquetLoadError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ParquetLoadError::Schema(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ParquetLoadError::Schema(format!("{name}: not Float64")))
}

fn col_u8<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt8Array, ParquetLoadError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ParquetLoadError::Schema(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| ParquetLoadError::Schema(format!("{name}: not UInt8")))
}

fn col_list<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray, ParquetLoadError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| ParquetLoadError::Schema(format!("missing column: {name}")))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| ParquetLoadError::Schema(format!("{name}: not List")))
}

fn parse_feature_list(
    list: &ListArray,
    row: usize,
) -> Result<Vec<(usize, f64)>, ParquetLoadError> {
    let inner = list.value(row);
    let s = inner
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| ParquetLoadError::Schema("features inner: not Struct".into()))?;
    let axis = s
        .column_by_name("axis")
        .ok_or_else(|| ParquetLoadError::Schema("features.axis missing".into()))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| ParquetLoadError::Schema("features.axis: not UInt32".into()))?;
    let weight = s
        .column_by_name("weight")
        .ok_or_else(|| ParquetLoadError::Schema("features.weight missing".into()))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ParquetLoadError::Schema("features.weight: not Float64".into()))?;
    let n = axis.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((axis.value(i) as usize, weight.value(i)));
    }
    Ok(out)
}

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/extended_corpus.parquet")
    }

    #[test]
    fn load_concepts_returns_nonempty() {
        let concepts = load_concepts(fixture()).expect("load");
        assert!(
            concepts.len() >= 5000,
            "expected ≥5000 concepts, got {}",
            concepts.len()
        );
    }

    #[test]
    fn stream_concepts_matches_load_count() {
        let eager = load_concepts(fixture()).expect("load").len();
        let streamed: usize = stream_concepts(fixture())
            .expect("stream")
            .filter_map(Result::ok)
            .count();
        assert_eq!(eager, streamed);
    }

    /// Bit-equivalence with the JSON path, modulo floating-point round-trip.
    /// Phase 6 is allowed to drop the JSON file; until then they must agree.
    #[cfg(feature = "json-fallback")]
    #[test]
    fn parquet_matches_json() {
        use crate::loader::load_from_json;
        let parquet = load_concepts(fixture()).expect("parquet");
        let json = load_from_json();
        assert_eq!(parquet.len(), json.len(), "row count differs");
        for (a, b) in parquet.iter().zip(&json) {
            assert_eq!(a.label, b.label, "label mismatch");
            assert_eq!(a.category, b.category, "category mismatch on {}", a.label);
            assert_eq!(a.features, b.features, "features mismatch on {}", a.label);
            assert!(
                (a.quality - b.quality).abs() < 1e-9,
                "quality mismatch on {}: {} vs {}",
                a.label,
                a.quality,
                b.quality
            );
            assert!(
                (a.axis_coherence - b.axis_coherence).abs() < 1e-9,
                "axis_coherence mismatch on {}",
                a.label
            );
            assert_eq!(
                a.bridge_degree, b.bridge_degree,
                "bridge_degree mismatch on {}",
                a.label
            );
            assert!(
                (a.source_confidence - b.source_confidence).abs() < 1e-9,
                "source_confidence mismatch on {}",
                a.label
            );
            assert!(
                (a.home_affinity - b.home_affinity).abs() < 1e-9,
                "home_affinity mismatch on {}",
                a.label
            );
        }
    }
}
