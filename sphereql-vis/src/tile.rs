//! Compact binary point tiles — the on-the-wire unit of the out-of-core,
//! server-backed viewer.
//!
//! The single-blob `Scene` JSON does not scale past a few hundred thousand
//! points (one string, parsed once). For millions of points the viewer streams
//! the visible working set as **tiles**: small binary blobs of `(position,
//! category, row)` records that decode straight into typed arrays on the
//! client. Aggregates (overlays, stats, palette) travel separately in the
//! [`crate::manifest::Manifest`]; per-point metadata (label, raw vector,
//! quality) is fetched lazily by `row` only when a point is inspected.
//!
//! Format (little-endian), `magic = b"SQT1"`:
//! ```text
//! header  (16 bytes): magic[4] · version:u16 · flags:u16 · count:u32 · reserved:u32
//! record  (20 bytes): x:f32 y:f32 z:f32 · cat:u16 · _pad:u16 · row:u32
//! ```
//! Positions are `f32` (exact, no quantization) in v1 — quantization can be
//! added behind `flags`/`version` without breaking older readers. The JS side
//! decodes the same layout with a `DataView`.

/// One point in a tile: display position, category id, and global row index.
///
/// `row` indexes back into the full corpus so the viewer can lazily fetch this
/// point's metadata (label / raw vector / quality) without it riding in the
/// tile. `cat` indexes the manifest's palette.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub cat: u16,
    pub row: u32,
}

const MAGIC: &[u8; 4] = b"SQT1";
/// Current tile format version.
pub const TILE_VERSION: u16 = 1;
const HEADER_LEN: usize = 16;
const RECORD_LEN: usize = 20;

/// Why a tile failed to decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileError {
    /// Buffer shorter than the 16-byte header.
    TooShort,
    /// Header magic was not `SQT1`.
    BadMagic,
    /// Header version is newer than this reader understands.
    UnsupportedVersion(u16),
    /// The declared record count doesn't match the buffer length.
    LengthMismatch {
        declared: usize,
        actual_records: usize,
    },
}

impl std::fmt::Display for TileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileError::TooShort => write!(f, "tile buffer shorter than the 16-byte header"),
            TileError::BadMagic => write!(f, "tile magic is not SQT1"),
            TileError::UnsupportedVersion(v) => write!(f, "unsupported tile version {v}"),
            TileError::LengthMismatch {
                declared,
                actual_records,
            } => write!(
                f,
                "tile declares {declared} records but the buffer holds {actual_records}"
            ),
        }
    }
}

impl std::error::Error for TileError {}

/// Encode points into a tile blob (header + packed little-endian records).
pub fn encode_tile(points: &[TilePoint]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + points.len() * RECORD_LEN);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&TILE_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // flags
    out.extend_from_slice(&(points.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    for p in points {
        out.extend_from_slice(&p.x.to_le_bytes());
        out.extend_from_slice(&p.y.to_le_bytes());
        out.extend_from_slice(&p.z.to_le_bytes());
        out.extend_from_slice(&p.cat.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // _pad
        out.extend_from_slice(&p.row.to_le_bytes());
    }
    out
}

/// Decode a tile blob produced by [`encode_tile`]. Validates magic, version,
/// and that the buffer length matches the declared record count.
pub fn decode_tile(bytes: &[u8]) -> Result<Vec<TilePoint>, TileError> {
    if bytes.len() < HEADER_LEN {
        return Err(TileError::TooShort);
    }
    if &bytes[0..4] != MAGIC {
        return Err(TileError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version > TILE_VERSION {
        return Err(TileError::UnsupportedVersion(version));
    }
    let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let body = &bytes[HEADER_LEN..];
    let actual = body.len() / RECORD_LEN;
    if actual != count {
        return Err(TileError::LengthMismatch {
            declared: count,
            actual_records: actual,
        });
    }
    let mut pts = Vec::with_capacity(count);
    for r in body.chunks_exact(RECORD_LEN) {
        pts.push(TilePoint {
            x: f32::from_le_bytes([r[0], r[1], r[2], r[3]]),
            y: f32::from_le_bytes([r[4], r[5], r[6], r[7]]),
            z: f32::from_le_bytes([r[8], r[9], r[10], r[11]]),
            cat: u16::from_le_bytes([r[12], r[13]]),
            row: u32::from_le_bytes([r[16], r[17], r[18], r[19]]),
        });
    }
    Ok(pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<TilePoint> {
        vec![
            TilePoint {
                x: 1.0,
                y: -2.5,
                z: 0.0,
                cat: 0,
                row: 0,
            },
            TilePoint {
                x: 0.25,
                y: 0.5,
                z: -0.75,
                cat: 7,
                row: 12345,
            },
            TilePoint {
                x: -1.0,
                y: 0.0,
                z: 1.0,
                cat: 65535,
                row: u32::MAX,
            },
        ]
    }

    #[test]
    fn round_trips_exactly() {
        let pts = sample();
        let bytes = encode_tile(&pts);
        assert_eq!(bytes.len(), HEADER_LEN + pts.len() * RECORD_LEN);
        assert_eq!(decode_tile(&bytes).unwrap(), pts);
    }

    #[test]
    fn empty_tile_round_trips() {
        let bytes = encode_tile(&[]);
        assert_eq!(bytes.len(), HEADER_LEN);
        assert!(decode_tile(&bytes).unwrap().is_empty());
    }

    #[test]
    fn rejects_short_buffer() {
        assert_eq!(decode_tile(&[1, 2, 3]), Err(TileError::TooShort));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = encode_tile(&sample());
        b[0] = b'X';
        assert_eq!(decode_tile(&b), Err(TileError::BadMagic));
    }

    #[test]
    fn rejects_future_version() {
        let mut b = encode_tile(&sample());
        b[4] = 99;
        assert_eq!(decode_tile(&b), Err(TileError::UnsupportedVersion(99)));
    }

    /// Cross-language wire-format lock. The JS viewer's `decodeTile`
    /// (sphereql-vis/src/viewer.js) is tested against this exact byte string
    /// (js-tests/08-tile-decode.test.cjs), so any change here that isn't
    /// mirrored there — or vice versa — breaks one side's suite. `cat=260` and
    /// `row=70000` exceed a byte, pinning u16/u32 little-endian layout.
    #[test]
    fn golden_bytes_match() {
        let pts = vec![
            TilePoint {
                x: 1.5,
                y: -2.0,
                z: 0.5,
                cat: 3,
                row: 7,
            },
            TilePoint {
                x: 0.0,
                y: 0.25,
                z: -0.75,
                cat: 260,
                row: 70000,
            },
        ];
        let hex: String = encode_tile(&pts)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            hex,
            "535154310100000002000000000000000000c03f000000c00000003f030000000700000000000000\
             0000803e000040bf0401000070110100"
        );
    }

    #[test]
    fn rejects_truncated_body() {
        let mut b = encode_tile(&sample());
        b.truncate(b.len() - 4); // lose part of the last record
        assert!(matches!(
            decode_tile(&b),
            Err(TileError::LengthMismatch { .. })
        ));
    }
}
