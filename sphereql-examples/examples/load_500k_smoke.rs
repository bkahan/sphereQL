//! Phase 3 scale smoke test: stream a 500K Parquet corpus and report
//! wall time + final concept count.
//!
//! Build the synthetic corpus first:
//!
//! ```text
//! python3 sphereql-corpus/tools/synthesize_500k.py
//! cargo run -p sphereql-corpus --example load_500k_smoke --release
//! ```
//!
//! Expected: < 30s wall, < 2 GB RSS. Phase 3 acceptance criterion.
//!
//! The synthetic file lives at `/tmp/synthetic_500k.parquet`. Override
//! with the `SYNTHETIC_PARQUET` env var.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use sphereql_corpus::parquet_loader::stream_concepts;

fn main() {
    let path = env::var_os("SYNTHETIC_PARQUET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/synthetic_500k.parquet"));

    if !path.exists() {
        eprintln!(
            "synthetic corpus not found at {}. Run:\n  \
             python3 sphereql-corpus/tools/synthesize_500k.py",
            path.display()
        );
        std::process::exit(2);
    }

    println!("streaming {}…", path.display());
    let t0 = Instant::now();
    let stream = stream_concepts(&path).expect("open stream");
    let mut n = 0usize;
    let mut errors = 0usize;
    for c in stream {
        match c {
            Ok(_) => n += 1,
            Err(e) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("row error: {e}");
                }
            }
        }
    }
    let elapsed = t0.elapsed();
    println!(
        "loaded {n} concepts in {:.2}s ({:.0} rows/sec), {errors} errors",
        elapsed.as_secs_f64(),
        n as f64 / elapsed.as_secs_f64()
    );
}
