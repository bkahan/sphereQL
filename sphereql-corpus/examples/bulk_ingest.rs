//! Streaming bulk-corpus ingest.
//!
//! Wires one of three [`BulkSource`]s through
//! [`HashedClaimAxisExtractor`] and into [`ParquetSink`], one row at a
//! time. Memory stays bounded to one batch (10K rows) regardless of
//! how many concepts the source produces — so the same binary handles
//! 500K Wikidata items, 5M OpenAlex Works, or 500M dump entries.
//!
//! Example runs:
//!
//! ```bash
//! # Default: 500K from Wikidata SPARQL (~30 min – 2 h depending on
//! # rate limit). Requires --features bulk-http (the default).
//! cargo run -p sphereql-corpus --example bulk_ingest --release -- \
//!     --source wikidata_sparql \
//!     --out /tmp/wikidata_500k.parquet \
//!     --target-size 500000
//!
//! # 5M from OpenAlex shards. Requires --features bulk-gzip (default)
//! # and a local shard directory populated via `aws s3 sync`.
//! cargo run -p sphereql-corpus --example bulk_ingest --release -- \
//!     --source openalex_shard \
//!     --shard-dir /tmp/openalex_shards \
//!     --out /tmp/openalex_5m.parquet \
//!     --target-size 5000000
//!
//! # 50M from the full Wikidata dump. Requires --features bulk-dump.
//! cargo run -p sphereql-corpus --example bulk_ingest --release \
//!     --features bulk-dump -- \
//!     --source wikidata_dump \
//!     --dump /data/latest-all.json.bz2 \
//!     --out /data/wikidata_50m.parquet \
//!     --target-size 50000000
//! ```
//!
//! Resume: after every Parquet flush the sink writes
//! `<out>.checkpoint.json`. Pass `--resume` to start the source at the
//! checkpoint's `source_offset` instead of zero. The Parquet file
//! itself is rewritten — checkpoint resume is meant for chaining
//! across short runs while iterating, not for splice-recovering a
//! single long run. For partition-style parallelism, give each worker
//! its own `--start-offset` and `--out`.

use std::path::PathBuf;
use std::time::Instant;

use sphereql_corpus::bulk::{
    BulkItem, BulkSource, BulkSourceError, HashedClaimAxisExtractor, OpenAlexShardConfig,
    OpenAlexShardSource, ParquetSink, SinkCheckpoint,
};
#[cfg(feature = "bulk-dbpedia")]
use sphereql_corpus::bulk::{DBpediaConfig, DBpediaTtlSource};
use sphereql_corpus::bulk::{SparqlConfig, WikidataSparqlSource};
#[cfg(feature = "bulk-dump")]
use sphereql_corpus::bulk::{WikidataDumpConfig, WikidataDumpSource};

const DEFAULT_NUM_AXES: usize = 128;
const DEFAULT_AXIS_SEED: u64 = 0xDEAD_BEEF;
const DEFAULT_BATCH_SIZE: usize = 10_000;
const DEFAULT_TARGET_SIZE: usize = 500_000;
const PROGRESS_EVERY: usize = 10_000;

#[derive(Debug)]
struct Args {
    source: SourceKind,
    out: PathBuf,
    target_size: usize,
    start_offset: usize,
    resume: bool,
    num_axes: usize,
    axis_seed: u64,
    batch_size: usize,
    // Per-source bits.
    sparql_page_size: usize,
    sparql_sleep_ms: u64,
    sparql_retries: usize,
    shard_dir: Option<PathBuf>,
    shard_min_cited_by: u64,
    shard_min_year: u32,
    dump_path: Option<PathBuf>,
    dump_only_items: bool,
    dump_require_english_label: bool,
    dbpedia_dir: Option<PathBuf>,
    dbpedia_oversample: usize,
}

#[derive(Debug, Clone, Copy)]
enum SourceKind {
    WikidataSparql,
    OpenAlexShard,
    WikidataDump,
    DBpedia,
}

impl SourceKind {
    fn parse(s: &str) -> Self {
        match s {
            "wikidata_sparql" | "sparql" => SourceKind::WikidataSparql,
            "openalex_shard" | "openalex" => SourceKind::OpenAlexShard,
            "wikidata_dump" | "dump" => SourceKind::WikidataDump,
            "dbpedia" => SourceKind::DBpedia,
            other => panic!(
                "unknown --source {other:?}; expected wikidata_sparql | openalex_shard | wikidata_dump | dbpedia"
            ),
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut a = Self::defaults();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--source" => a.source = SourceKind::parse(&need(&mut args, "--source")),
                "--out" => a.out = PathBuf::from(need(&mut args, "--out")),
                "--target-size" => a.target_size = parse_usize(&need(&mut args, "--target-size")),
                "--start-offset" => {
                    a.start_offset = parse_usize(&need(&mut args, "--start-offset"));
                }
                "--resume" => a.resume = true,
                "--num-axes" => a.num_axes = parse_usize(&need(&mut args, "--num-axes")),
                "--axis-seed" => a.axis_seed = parse_u64(&need(&mut args, "--axis-seed")),
                "--batch-size" => a.batch_size = parse_usize(&need(&mut args, "--batch-size")),
                // SPARQL
                "--sparql-page-size" => {
                    a.sparql_page_size = parse_usize(&need(&mut args, "--sparql-page-size"));
                }
                "--sparql-sleep-ms" => {
                    a.sparql_sleep_ms = parse_u64(&need(&mut args, "--sparql-sleep-ms"));
                }
                "--sparql-retries" => {
                    a.sparql_retries = parse_usize(&need(&mut args, "--sparql-retries"));
                }
                // OpenAlex
                "--shard-dir" => a.shard_dir = Some(PathBuf::from(need(&mut args, "--shard-dir"))),
                "--min-cited-by" => {
                    a.shard_min_cited_by = parse_u64(&need(&mut args, "--min-cited-by"));
                }
                "--min-year" => {
                    a.shard_min_year = parse_u64(&need(&mut args, "--min-year")) as u32;
                }
                // Wikidata dump
                "--dump" => a.dump_path = Some(PathBuf::from(need(&mut args, "--dump"))),
                "--dump-all-types" => a.dump_only_items = false,
                "--dump-allow-missing-label" => a.dump_require_english_label = false,
                // DBpedia
                "--dbpedia-dir" => {
                    a.dbpedia_dir = Some(PathBuf::from(need(&mut args, "--dbpedia-dir")));
                }
                "--dbpedia-oversample" => {
                    a.dbpedia_oversample = parse_usize(&need(&mut args, "--dbpedia-oversample"));
                }
                "--help" | "-h" => {
                    eprintln!("{HELP}");
                    std::process::exit(0);
                }
                other => panic!("unknown arg: {other}"),
            }
        }
        a
    }

    fn defaults() -> Self {
        Self {
            source: SourceKind::WikidataSparql,
            out: PathBuf::from("/tmp/bulk_corpus.parquet"),
            target_size: DEFAULT_TARGET_SIZE,
            start_offset: 0,
            resume: false,
            num_axes: DEFAULT_NUM_AXES,
            axis_seed: DEFAULT_AXIS_SEED,
            batch_size: DEFAULT_BATCH_SIZE,
            sparql_page_size: 500,
            sparql_sleep_ms: 250,
            sparql_retries: 3,
            shard_dir: None,
            shard_min_cited_by: 5,
            shard_min_year: 2010,
            dump_path: None,
            dump_only_items: true,
            dump_require_english_label: true,
            dbpedia_dir: None,
            dbpedia_oversample: 4,
        }
    }
}

fn need<I: Iterator<Item = String>>(args: &mut I, flag: &str) -> String {
    args.next()
        .unwrap_or_else(|| panic!("{flag} needs a value"))
}

fn parse_usize(s: &str) -> usize {
    s.replace('_', "")
        .parse()
        .unwrap_or_else(|_| panic!("expected non-negative integer, got {s:?}"))
}

fn parse_u64(s: &str) -> u64 {
    let cleaned = s.replace('_', "");
    if let Some(hex) = cleaned
        .strip_prefix("0x")
        .or_else(|| cleaned.strip_prefix("0X"))
    {
        return u64::from_str_radix(hex, 16)
            .unwrap_or_else(|_| panic!("expected hex u64, got {s:?}"));
    }
    cleaned
        .parse()
        .unwrap_or_else(|_| panic!("expected u64, got {s:?}"))
}

const HELP: &str = "\
usage: bulk_ingest [OPTIONS]

Common:
  --source NAME           wikidata_sparql | openalex_shard | wikidata_dump
  --out PATH              output parquet (default /tmp/bulk_corpus.parquet)
  --target-size N         stop after N items (default 500000)
  --start-offset N        skip N source items before writing
  --resume                start at checkpoint.source_offset if present
  --num-axes N            hashed-claim axis count (default 128)
  --axis-seed HEX|N       FNV-1a seed (default 0xdeadbeef)
  --batch-size N          rows per parquet flush (default 10000)

wikidata_sparql:
  --sparql-page-size N    items per HTTP query (default 500)
  --sparql-sleep-ms N     between pages (default 250)
  --sparql-retries N      retries per failed query (default 3)

openalex_shard:
  --shard-dir PATH        directory with *.gz shards (required)
  --min-cited-by N        drop works with fewer citations (default 5)
  --min-year N            drop works published before N (default 2010)

wikidata_dump (requires --features bulk-dump):
  --dump PATH             latest-all.json.bz2 (required)
  --dump-all-types        include properties + lexemes (default items only)
  --dump-allow-missing-label  keep items without an English label

dbpedia (requires --features bulk-dbpedia):
  --dbpedia-dir PATH      directory with instance-types/mappingbased-objects/labels .ttl.bz2
  --dbpedia-oversample N  working-set multiplier on --target-size (default 4)
";

fn main() {
    let args = Args::parse();
    run(args);
}

fn run(mut args: Args) {
    if args.resume
        && let Some(cp) = SinkCheckpoint::load_for(&args.out)
        && cp.source_offset > args.start_offset
    {
        println!(
            "→ resume: advancing source_offset {} → {} (checkpoint.json)",
            args.start_offset, cp.source_offset
        );
        args.start_offset = cp.source_offset;
    }

    let extractor = HashedClaimAxisExtractor::new(args.num_axes, args.axis_seed);
    println!(
        "→ writing {} (target={}, batch={}, axes={}, seed=0x{:x})",
        args.out.display(),
        args.target_size,
        args.batch_size,
        args.num_axes,
        args.axis_seed,
    );
    let mut sink =
        ParquetSink::create(&args.out, extractor, args.batch_size).expect("open parquet sink");

    let started = Instant::now();
    let mut consumed = 0usize;
    let mut soft_errors = 0usize;
    let mut last_log = Instant::now();

    let source = build_source(&args);
    println!("→ source: {}", source.source_name());

    for next in source {
        match next {
            Ok(item) => {
                consumed += 1;
                if let Err(e) = sink.push(item) {
                    eprintln!("error: sink push failed: {e}");
                    break;
                }
                if consumed >= args.target_size {
                    break;
                }
                if consumed.is_multiple_of(PROGRESS_EVERY) {
                    let now = Instant::now();
                    let rate = PROGRESS_EVERY as f64 / now.duration_since(last_log).as_secs_f64();
                    last_log = now;
                    println!(
                        "  consumed={consumed:>9}  written={:>9}  rate≈{rate:>6.0} rows/s",
                        sink.n_written()
                    );
                }
            }
            Err(e) => {
                soft_errors += 1;
                if soft_errors <= 10 {
                    eprintln!("warning: source item error: {e}");
                } else if soft_errors == 11 {
                    eprintln!("warning: further per-item source errors suppressed");
                }
            }
        }
    }

    let cp = sink.close().expect("close parquet sink");
    let elapsed = started.elapsed();
    let prefix = if cp.n_written == 0 { "✗" } else { "✓" };
    println!(
        "\n{prefix} ingest done in {:.1}s — consumed {} source items, wrote {} rows ({} soft errors)",
        elapsed.as_secs_f64(),
        cp.source_offset,
        cp.n_written,
        soft_errors
    );
    println!("  output:     {}", args.out.display());
    println!(
        "  checkpoint: {}",
        SinkCheckpoint::sidecar_for(&args.out).display()
    );
    if cp.n_written == 0 {
        eprintln!(
            "error: 0 rows written. Source produced no items (likely network / rate-limit / \
             query timeout); refusing to leave an empty Parquet for downstream stages."
        );
        std::process::exit(2);
    }
}

fn build_source(args: &Args) -> Box<dyn BulkSource<Item = Result<BulkItem, BulkSourceError>>> {
    match args.source {
        SourceKind::WikidataSparql => {
            let cfg = SparqlConfig {
                start_offset: args.start_offset,
                max_items: args.target_size,
                page_size: args.sparql_page_size,
                sleep_ms: args.sparql_sleep_ms,
                retries: args.sparql_retries,
                ..Default::default()
            };
            Box::new(WikidataSparqlSource::new(cfg))
        }
        SourceKind::OpenAlexShard => {
            let dir = args
                .shard_dir
                .as_ref()
                .expect("openalex_shard source requires --shard-dir PATH");
            let mut cfg = OpenAlexShardConfig::from_directory(dir).expect("locate openalex shards");
            cfg.start_offset = args.start_offset;
            cfg.max_items = args.target_size;
            cfg.min_cited_by = args.shard_min_cited_by;
            cfg.min_year = args.shard_min_year;
            Box::new(OpenAlexShardSource::new(cfg))
        }
        SourceKind::WikidataDump => build_dump_source(args),
        SourceKind::DBpedia => build_dbpedia_source(args),
    }
}

#[cfg(feature = "bulk-dump")]
fn build_dump_source(args: &Args) -> Box<dyn BulkSource<Item = Result<BulkItem, BulkSourceError>>> {
    let path = args
        .dump_path
        .clone()
        .expect("wikidata_dump source requires --dump PATH");
    let mut cfg = WikidataDumpConfig::new(path);
    cfg.start_offset = args.start_offset;
    cfg.max_items = args.target_size;
    cfg.only_items = args.dump_only_items;
    cfg.require_english_label = args.dump_require_english_label;
    Box::new(WikidataDumpSource::new(cfg))
}

#[cfg(not(feature = "bulk-dump"))]
fn build_dump_source(
    _args: &Args,
) -> Box<dyn BulkSource<Item = Result<BulkItem, BulkSourceError>>> {
    panic!(
        "wikidata_dump source requires the `bulk-dump` feature. Rebuild with \
         --features bulk-dump"
    );
}

#[cfg(feature = "bulk-dbpedia")]
fn build_dbpedia_source(
    args: &Args,
) -> Box<dyn BulkSource<Item = Result<BulkItem, BulkSourceError>>> {
    let dir = args
        .dbpedia_dir
        .clone()
        .expect("dbpedia source requires --dbpedia-dir PATH");
    let mut cfg = DBpediaConfig::new(dir);
    cfg.start_offset = args.start_offset;
    cfg.max_items = args.target_size;
    cfg.oversample = args.dbpedia_oversample.max(1);
    Box::new(DBpediaTtlSource::new(cfg))
}

#[cfg(not(feature = "bulk-dbpedia"))]
fn build_dbpedia_source(
    _args: &Args,
) -> Box<dyn BulkSource<Item = Result<BulkItem, BulkSourceError>>> {
    panic!(
        "dbpedia source requires the `bulk-dbpedia` feature. Rebuild with \
         --features bulk-dbpedia"
    );
}
