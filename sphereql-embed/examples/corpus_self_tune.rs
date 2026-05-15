//! Corpus self-tuning binary.
//!
//! Loads `sphereql-corpus/data/extended_corpus.parquet`, runs the
//! Phase-6 self-tune loop, prints a per-iteration summary table, and
//! optionally writes the mutated corpus back to disk.
//!
//! Run modes:
//! ```bash
//! # Default — dry run; writes to extended_corpus.tuned.parquet:
//! cargo run -p sphereql-embed --example corpus_self_tune --release
//!
//! # Commit in place — requires --commit-confirm to avoid accidents:
//! cargo run -p sphereql-embed --example corpus_self_tune --release -- \
//!     --commit --commit-confirm
//!
//! # Custom corpus path / iteration cap:
//! cargo run -p sphereql-embed --example corpus_self_tune --release -- \
//!     --corpus /tmp/my.parquet --max-iters 5
//! ```
//!
//! The loop is deterministic: same `embed` seed + same input parquet
//! produces the same per-iteration trajectory. Acceptance criterion #4
//! from `docs/.prompts/500k-corpus`: the final composite must be
//! ≥ the first composite (modulo `plateau_epsilon`).

use std::path::PathBuf;

use sphereql_corpus::{ConceptMetadata, ConceptRow, load_concepts_with_metadata, write_concepts};
use sphereql_embed::{
    CorpusQuality, PipelineConfig, SelfTuneConfig, TunableConcept, run_self_tune,
};

const DEFAULT_CORPUS: &str = "sphereql-corpus/data/extended_corpus.parquet";
const TUNED_SUFFIX: &str = ".tuned.parquet";
const EMBED_SEED: u64 = 0xDEAD_BEEF;

struct Args {
    corpus: PathBuf,
    out: Option<PathBuf>,
    commit: bool,
    commit_confirm: bool,
    max_iters: Option<usize>,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut corpus = PathBuf::from(DEFAULT_CORPUS);
        let mut out: Option<PathBuf> = None;
        let mut commit = false;
        let mut commit_confirm = false;
        let mut max_iters: Option<usize> = None;
        while let Some(a) = args.next() {
            match a.as_str() {
                "--corpus" => {
                    corpus = args.next().map(PathBuf::from).expect("--corpus needs a value");
                }
                "--out" => {
                    out = Some(args.next().map(PathBuf::from).expect("--out needs a value"));
                }
                "--commit" => commit = true,
                "--commit-confirm" => commit_confirm = true,
                "--max-iters" => {
                    max_iters = Some(
                        args.next()
                            .expect("--max-iters needs a value")
                            .parse()
                            .expect("--max-iters must be a positive integer"),
                    );
                }
                "--help" | "-h" => {
                    eprintln!(
                        "usage: corpus_self_tune [--corpus PATH] [--out PATH] [--commit --commit-confirm] [--max-iters N]"
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown arg: {other}"),
            }
        }
        Self {
            corpus,
            out,
            commit,
            commit_confirm,
            max_iters,
        }
    }

    fn resolved_out(&self) -> PathBuf {
        if let Some(o) = &self.out {
            return o.clone();
        }
        if self.commit && self.commit_confirm {
            return self.corpus.clone();
        }
        let mut p = self.corpus.clone();
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("corpus");
        p.set_file_name(format!("{stem}{TUNED_SUFFIX}"));
        p
    }
}

fn main() {
    let args = Args::parse();

    println!("→ loading {}", args.corpus.display());
    let loaded = load_concepts_with_metadata(&args.corpus).expect("load extended corpus");
    println!("  loaded {} concepts", loaded.len());

    let tunable: Vec<TunableConcept> = loaded
        .iter()
        .map(|(c, meta)| TunableConcept {
            label: c.label.to_string(),
            category: c.category.to_string(),
            features: c.features.clone(),
            quality: c.quality,
            axis_coherence: c.axis_coherence,
            bridge_degree: c.bridge_degree,
            source_confidence: c.source_confidence,
            home_affinity: c.home_affinity,
            source: meta.source.clone(),
            openalex_id: meta.openalex_id.clone(),
        })
        .collect();

    let mut cfg = SelfTuneConfig::default();
    if let Some(n) = args.max_iters {
        cfg.max_iterations = n;
    }
    println!(
        "→ running self-tune: max_iter={} plateau_eps={:.4} min_q={:.2} min_per_cat={}",
        cfg.max_iterations,
        cfg.plateau_epsilon,
        cfg.min_quality_to_keep,
        cfg.min_concepts_per_category
    );

    let metric = CorpusQuality::default();
    let pipeline_config = PipelineConfig::default();
    let embed_fn = |f: &[(usize, f64)]| sphereql_corpus::embed(f, EMBED_SEED);

    let started = std::time::Instant::now();
    let (tuned, report) = run_self_tune(tunable, embed_fn, pipeline_config, &metric, &cfg);
    let elapsed = started.elapsed();

    println!("\niter | n_concepts | composite | evr   | bridge | curv  | balance | n_pruned | mean_q | Δmean_q");
    println!("-----|------------|-----------|-------|--------|-------|---------|----------|--------|--------");
    for it in &report.iterations {
        let bd = &it.breakdown;
        println!(
            "{:>4} | {:>10} | {:>9.4} | {:>5.3} | {:>6.3} | {:>5.3} | {:>7.3} | {:>8} | {:>6.3} | {:>+7.3}",
            it.iteration,
            it.n_concepts,
            it.composite_score,
            bd.evr,
            bd.bridge_coherence,
            bd.curvature_health,
            bd.category_balance,
            it.n_pruned,
            it.mean_quality,
            it.mean_quality_delta
        );
    }
    println!("\nstopped: {:?}  ({:.2}s)", report.stopped_reason, elapsed.as_secs_f64());

    if let (Some(first), Some(last)) = (report.iterations.first(), report.iterations.last()) {
        let delta = last.composite_score - first.composite_score;
        println!(
            "composite: first={:.4}  last={:.4}  Δ={:+.4}",
            first.composite_score, last.composite_score, delta
        );
        // Acceptance: final ≥ first modulo plateau_epsilon.
        if delta + cfg.plateau_epsilon < 0.0 {
            eprintln!(
                "warning: composite regressed by {:.4} (> plateau_epsilon {:.4})",
                -delta, cfg.plateau_epsilon
            );
        }
    }

    let out_path = args.resolved_out();
    if args.commit && !args.commit_confirm {
        eprintln!(
            "→ --commit was given without --commit-confirm; refusing to overwrite. \
             Re-run with both flags or remove --commit to write to {}",
            out_path.display()
        );
        std::process::exit(2);
    }

    println!("→ writing {} ({} concepts)", out_path.display(), tuned.len());
    let rows: Vec<ConceptRow<'_>> = tuned
        .iter()
        .map(|c| ConceptRow {
            label: c.label.as_str(),
            category: c.category.as_str(),
            features: c.features.as_slice(),
            quality: c.quality,
            axis_coherence: c.axis_coherence,
            bridge_degree: c.bridge_degree,
            source_confidence: c.source_confidence,
            home_affinity: c.home_affinity,
            source: c.source.as_deref(),
            openalex_id: c.openalex_id.as_deref(),
        })
        .collect();
    write_concepts(rows, &out_path).expect("write tuned parquet");

    // Pull metadata back through a sanity load to verify the round-trip
    // (cheap on a corpus this size). Fails loudly if the writer drifts
    // from the loader's schema.
    let reloaded = load_concepts_with_metadata(&out_path).expect("re-load tuned parquet");
    assert_eq!(reloaded.len(), tuned.len(), "round-trip count mismatch");
    let meta_kept: usize = reloaded
        .iter()
        .filter(|(_, m): &&(_, ConceptMetadata)| m.source.is_some() || m.openalex_id.is_some())
        .count();
    println!("  round-trip ok; {meta_kept} rows carry source/openalex_id");
}
