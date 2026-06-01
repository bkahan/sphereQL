//! Parquet writer for the extended corpus.
//!
//! Mirrors `tools/generate_extended.py::write_parquet` exactly so the
//! self-tune loop in `sphereql-embed` (Phase 6) can round-trip the
//! corpus through a transformation pass without drifting from the
//! schema that `parquet_loader::load_concepts` expects.
//!
//! Schema (matches [`parquet_loader`](crate::parquet_loader)):
//!
//! | column            | arrow type                                  | nullable |
//! |-------------------|---------------------------------------------|----------|
//! | label             | Utf8                                        | no       |
//! | category          | Utf8                                        | no       |
//! | features          | List\<Struct{axis:UInt32, weight:Float64}\> | no       |
//! | quality           | Float64                                     | no       |
//! | axis_coherence    | Float64                                     | no       |
//! | bridge_degree     | UInt8                                       | no       |
//! | source_confidence | Float64                                     | no       |
//! | home_affinity     | Float64                                     | no       |
//! | source            | Utf8                                        | yes      |
//! | openalex_id       | Utf8                                        | yes      |
//!
//! SNAPPY compression, row group size 4096. Dictionary encoding is on
//! by default for short repeated columns.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, ListBuilder, StringArray, StructBuilder, UInt8Array,
    UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::parquet_loader::{ConceptMetadata, ParquetLoadError};

/// One row as the writer expects it. The label / category / features /
/// signal fields are required; provenance fields are optional. The
/// writer accepts borrowed `&str` for the required strings so callers
/// can pass either owned `String`s or `Concept`'s `&'static str`s with
/// no allocation.
#[derive(Debug, Clone)]
pub struct ConceptRow<'a> {
    pub label: &'a str,
    pub category: &'a str,
    pub features: &'a [(usize, f64)],
    pub quality: f64,
    pub axis_coherence: f64,
    pub bridge_degree: u8,
    pub source_confidence: f64,
    pub home_affinity: f64,
    pub source: Option<&'a str>,
    pub openalex_id: Option<&'a str>,
}

impl<'a> ConceptRow<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        label: &'a str,
        category: &'a str,
        features: &'a [(usize, f64)],
        quality: f64,
        axis_coherence: f64,
        bridge_degree: u8,
        source_confidence: f64,
        home_affinity: f64,
        metadata: &'a ConceptMetadata,
    ) -> Self {
        Self {
            label,
            category,
            features,
            quality,
            axis_coherence,
            bridge_degree,
            source_confidence,
            home_affinity,
            source: metadata.source.as_deref(),
            openalex_id: metadata.openalex_id.as_deref(),
        }
    }
}

const ROW_GROUP_SIZE: usize = 4096;

/// Write `rows` to `path` as a SNAPPY-compressed Parquet file matching
/// the canonical extended-corpus schema. Overwrites if the file exists.
pub fn write_concepts<'a, P, I>(rows: I, path: P) -> Result<(), ParquetLoadError>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = ConceptRow<'a>>,
{
    let rows: Vec<ConceptRow<'a>> = rows.into_iter().collect();
    let schema = Arc::new(build_schema());
    let batch = build_batch(schema.clone(), &rows)?;

    let file = File::create(path.as_ref())?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .build();
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(props)).map_err(ParquetLoadError::from)?;
    writer.write(&batch).map_err(ParquetLoadError::from)?;
    writer.close().map_err(ParquetLoadError::from)?;
    Ok(())
}

pub(crate) fn build_schema() -> Schema {
    let feat_struct = DataType::Struct(Fields::from(vec![
        Field::new("axis", DataType::UInt32, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    Schema::new(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new(
            "features",
            // ListBuilder<StructBuilder> produces a nullable inner
            // Struct field, matching what PyArrow's `pa.list_(struct)`
            // emits by default (the canonical Parquet on disk).
            DataType::List(Arc::new(Field::new("item", feat_struct, true))),
            false,
        ),
        Field::new("quality", DataType::Float64, false),
        Field::new("axis_coherence", DataType::Float64, false),
        Field::new("bridge_degree", DataType::UInt8, false),
        Field::new("source_confidence", DataType::Float64, false),
        Field::new("home_affinity", DataType::Float64, false),
        Field::new("source", DataType::Utf8, true),
        Field::new("openalex_id", DataType::Utf8, true),
    ])
}

pub(crate) fn build_batch(
    schema: Arc<Schema>,
    rows: &[ConceptRow<'_>],
) -> Result<RecordBatch, ParquetLoadError> {
    let labels = StringArray::from(rows.iter().map(|r| r.label).collect::<Vec<_>>());
    let cats = StringArray::from(rows.iter().map(|r| r.category).collect::<Vec<_>>());

    let inner_fields: Fields = Fields::from(vec![
        Field::new("axis", DataType::UInt32, false),
        Field::new("weight", DataType::Float64, false),
    ]);
    let inner_builder = StructBuilder::new(
        inner_fields,
        vec![
            Box::new(UInt32Builder::new()),
            Box::new(Float64Builder::new()),
        ],
    );
    let mut features_builder = ListBuilder::new(inner_builder);
    for row in rows {
        {
            let struct_builder = features_builder.values();
            for &(axis, weight) in row.features {
                struct_builder
                    .field_builder::<UInt32Builder>(0)
                    .expect("axis builder")
                    .append_value(axis as u32);
                struct_builder
                    .field_builder::<Float64Builder>(1)
                    .expect("weight builder")
                    .append_value(weight);
                struct_builder.append(true);
            }
        }
        features_builder.append(true);
    }
    let features_array: ArrayRef = Arc::new(features_builder.finish());

    let quality = Float64Array::from(rows.iter().map(|r| r.quality).collect::<Vec<_>>());
    let coherence = Float64Array::from(rows.iter().map(|r| r.axis_coherence).collect::<Vec<_>>());
    let bridge = UInt8Array::from(rows.iter().map(|r| r.bridge_degree).collect::<Vec<_>>());
    let conf = Float64Array::from(rows.iter().map(|r| r.source_confidence).collect::<Vec<_>>());
    let home = Float64Array::from(rows.iter().map(|r| r.home_affinity).collect::<Vec<_>>());

    let source = StringArray::from(rows.iter().map(|r| r.source).collect::<Vec<_>>());
    let openalex = StringArray::from(rows.iter().map(|r| r.openalex_id).collect::<Vec<_>>());

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(labels),
            Arc::new(cats),
            features_array,
            Arc::new(quality),
            Arc::new(coherence),
            Arc::new(bridge),
            Arc::new(conf),
            Arc::new(home),
            Arc::new(source),
            Arc::new(openalex),
        ],
    )
    .map_err(ParquetLoadError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_loader::load_concepts_with_metadata;
    use tempfile::NamedTempFile;

    #[allow(clippy::type_complexity)]
    fn sample_rows() -> Vec<(String, String, Vec<(usize, f64)>, ConceptMetadata)> {
        vec![
            (
                "alpha".into(),
                "physics".into(),
                vec![(0, 1.0), (3, 0.5)],
                ConceptMetadata {
                    source: Some("openalex".into()),
                    openalex_id: Some("T1234".into()),
                },
            ),
            (
                "beta".into(),
                "biology".into(),
                vec![(7, 0.9)],
                ConceptMetadata {
                    source: None,
                    openalex_id: None,
                },
            ),
        ]
    }

    #[test]
    fn round_trips_required_and_optional_columns() {
        let rows = sample_rows();
        let concept_rows: Vec<ConceptRow<'_>> = rows
            .iter()
            .map(|(label, category, features, meta)| ConceptRow {
                label,
                category,
                features,
                quality: 0.8,
                axis_coherence: 0.7,
                bridge_degree: 2,
                source_confidence: 0.5,
                home_affinity: 0.6,
                source: meta.source.as_deref(),
                openalex_id: meta.openalex_id.as_deref(),
            })
            .collect();
        let tmp = NamedTempFile::new().expect("tmpfile");
        write_concepts(concept_rows, tmp.path()).expect("write");

        let loaded = load_concepts_with_metadata(tmp.path()).expect("load");
        assert_eq!(loaded.len(), 2);
        let (c0, m0) = &loaded[0];
        assert_eq!(c0.label, "alpha");
        assert_eq!(c0.category, "physics");
        assert_eq!(c0.features, vec![(0, 1.0), (3, 0.5)]);
        assert!((c0.quality - 0.8).abs() < 1e-12);
        assert_eq!(m0.source.as_deref(), Some("openalex"));
        assert_eq!(m0.openalex_id.as_deref(), Some("T1234"));

        let (c1, m1) = &loaded[1];
        assert_eq!(c1.label, "beta");
        assert_eq!(m1.source, None);
        assert_eq!(m1.openalex_id, None);
    }
}
